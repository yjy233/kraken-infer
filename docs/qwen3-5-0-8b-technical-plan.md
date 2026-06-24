# Qwen3.5 系列 GGUF 推理技术方案

本文档描述如何参考 `~/code/llama.cpp` 在本仓库实现 Qwen3.5 系列推理。Qwen3.5 0.8B dense 文本模型是最小基线，但完成范围必须同时覆盖：

- Qwen3.5 dense 文本模型，GGUF arch `qwen35`。
- Qwen3.5 MoE 文本模型，GGUF arch `qwen35moe`。
- Qwen3.5 VL/Omni 多模态输入路径，包括 text GGUF 和 mmproj GGUF。
- MTP/NextN speculative decoding。
- 多 batch、多 sequence 的 hybrid KV/recurrent cache 和 recurrent rollback。
- GGUF 量化权重。

性能目标是和 llama.cpp Metal 路径同量级。Qwen3.5 路径只接受 GGUF，不复用现有 safetensors 权重加载路径；现有 Qwen3 0.6B safetensors 路径可以继续作为旧模型兼容路径存在。

## llama.cpp 参考点

核心参考文件：

- `~/code/llama.cpp/src/models/qwen35.cpp`
  - `llama_model_qwen35::load_arch_hparams`
  - `llama_model_qwen35::load_arch_tensors`
  - `llama_model_qwen35::graph::build_layer_attn`
  - `llama_model_qwen35::graph::build_layer_attn_linear`
  - `llama_model_qwen35::graph_mtp`
- `~/code/llama.cpp/src/models/qwen35moe.cpp`
  - `llama_model_qwen35moe::load_arch_hparams`
  - `llama_model_qwen35moe::load_arch_tensors`
  - routed expert + shared expert FFN
  - MoE MTP graph
- `~/code/llama.cpp/src/models/delta-net-base.cpp`
  - `build_conv_state`
  - `build_recurrent_attn`
  - `build_delta_net_autoregressive`
  - `build_delta_net_chunking`
  - `build_delta_net_fused`
- `~/code/llama.cpp/src/llama-memory-recurrent.cpp`
  - `n_rs_seq`
  - `rs_idx`
  - `seq_rm` partial rollback
  - recurrent state tensors widened to `mem_size * (1 + n_rs_seq)`
- `~/code/llama.cpp/conversion/qwen.py`
  - `Qwen3_5TextModel`
  - `Qwen3_5MoeTextModel`
  - `_LinearAttentionVReorderBase`
  - `_Qwen35MRopeMixin`
  - `_Qwen35MtpMixin`
- `~/code/llama.cpp/conversion/qwen3vl.py`
  - `Qwen3VLVisionModel`
  - `Qwen3OmniMmprojModel`
  - `Qwen3VLTextModel`
  - `Qwen3VLMoeTextModel`
  - `Qwen3OmniMoeTextModel`
- `~/code/llama.cpp/tools/mtmd/clip.cpp`
- `~/code/llama.cpp/tools/mtmd/models/qwen3vl.cpp`
- `~/code/llama.cpp/tools/mtmd/models/qwen3a.cpp`
- `~/code/llama.cpp/tools/mtmd/mtmd-audio.cpp`
- `~/code/llama.cpp/src/llama-vocab.cpp`
- `~/code/llama.cpp/src/unicode.cpp`
  - `unicode_regex_split_custom_qwen35`
- `~/code/llama.cpp/src/llama-arch.cpp`
- `~/code/llama.cpp/gguf-py/gguf/constants.py`
- `~/code/llama.cpp/src/llama-quant.cpp`
- `~/code/llama.cpp/gguf-py/tests/test_quants.py`

本仓库预计新增或改造入口：

- `include/toyllm/model/model_config.hpp`
- `src/model/model_config.cpp`
- `include/toyllm/runtime/gguf_reader.hpp`
- `src/runtime/gguf_reader.cpp`
- `include/toyllm/runtime/llama_tokenizer.hpp`
- `src/runtime/llama_tokenizer.cpp`
- `include/toyllm/runtime/qwen35_hybrid_cache.hpp`
- `src/runtime/qwen35_hybrid_cache.cpp`
- `src/runtime/cpu/qwen35_cpu_model.cpp`
- `src/runtime/cpu/qwen35moe_cpu_model.cpp`
- `src/runtime/qwen35_speculative_decoder.cpp`
- `src/runtime/qwen35_multimodal.cpp`
- `src/backends/mpsgraph/qwen35_mpsgraph_model.cpp`
- `src/backends/mpsgraph/qwen35moe_mpsgraph_model.cpp`
- `src/backends/metal/qwen35_kernels.mm`
- `src/backends/metal/gguf_quant_kernels.mm`

## 关键结论

Qwen3.5 不能当作 Qwen3 0.6B 的 config 变体处理。本仓库现有 Qwen3 路径是 dense full attention block：

```text
RMSNorm -> Q/K/V -> QK norm -> RoPE -> KV cache attention -> O proj -> residual
        -> RMSNorm -> SwiGLU MLP -> residual
```

Qwen3.5 dense/MoE 文本模型是 hybrid decoder：

- 24 个主干层且 `hidden_size == 1024` 是 Qwen3.5 0.8B dense 基线。
- 默认 `full_attention_interval = 4`。
- 如果 GGUF 没有显式 `attention.recurrent_layers`，则第 `4, 8, 12, ...` 层是 full attention，其余层是 recurrent linear attention。
- full attention 层仍然使用 KV cache，但 Q projection 输出包含 query 和 gate 两部分。
- recurrent 层使用 gated delta net，需要独立的 conv state 和 SSM state cache。
- Qwen3.5 使用 MRoPE，默认 section 是 `[11, 11, 10, 0]`。
- Qwen3.5 MoE 在 attention 后使用 routed experts，并叠加 gated shared expert。
- MTP/NextN 层作为额外 decoder block 附加在主干层之后，主模型 forward 不执行它们，speculative draft graph 单独执行。
- tokenizer 以 llama.cpp 的 GGUF tokenizer 实现为准；Qwen3.5 pre-tokenizer regex 和 Qwen2/Qwen3 有差异，字母 run 需要包含 Unicode combining mark。

性能一致的实现必须把 full attention、gated delta net、MoE routing、quantized matmul 和 MTP draft graph 都放到设备侧。prefill 必须批处理，decode 必须设备常驻 cache，不能退化成逐 token、逐算子的 host 调度。

## 支持范围

完整支持范围：

- `general.architecture = "qwen35"` 的 Qwen3.5 dense text GGUF。
- `general.architecture = "qwen35moe"` 的 Qwen3.5 MoE text GGUF。
- Qwen3.5 VL/Omni 的 text GGUF：
  - `qwen35` / `qwen35moe`，用于 Qwen3.5 conditional text decoder。
  - `qwen3vl` / `qwen3vlmoe`，用于 Qwen3 VL/Omni text decoder。
- Qwen3.5 VL/Omni 的 mmproj GGUF：
  - `general.architecture = "clip"`。
  - `clip.projector_type = "qwen3vl_merger"`。
  - `clip.vision.projector_type = "qwen3vl_merger"`。
  - `clip.audio.projector_type = "qwen3a"`。
- full attention + recurrent linear attention hybrid stack。
- dense SwiGLU FFN 和 Qwen3.5 MoE FFN。
- MTP/NextN speculative decoding。
- 多 batch、多 sequence prefill/decode。
- recurrent rollback，用于 speculative rejection、sequence truncation 和 batch 内多序列回退。
- GGUF F32/F16/BF16 和 llama.cpp 常用 GGUF quant types。
- greedy、sampling 和 speculative sampling。
- 从 GGUF tokenizer metadata 和 `tokenizer.chat_template` 构造 prompt。

非目标：

- 不从 safetensors 加载 Qwen3.5 权重。
- 不在运行时重复执行 HF-to-GGUF conversion 已完成的权重变换。
- 不用本仓库现有 HF JSON tokenizer 作为 Qwen3.5 tokenizer 的权威实现。

## GGUF 读取和元数据解析

新增 GGUF reader，不从 `config.json` / `generation_config.json` / `model.safetensors` 加载 Qwen3.5。reader 负责：

- 校验 GGUF magic/version/endian。
- 读取 metadata key/value，保留原始 key 以便调试。
- 读取 tensor directory、GGML dtype、shape、offset、alignment。
- 支持 mmap 和一次性读取；MPSGraph/Metal 上传时按 tensor span 取数据。
- 支持单文件 GGUF 和 split GGUF。
- 支持 text GGUF 和 mmproj GGUF。
- 对 tensor name、shape、dtype 做严格校验，错误信息包含 GGUF key/tensor name。

Qwen3.5 text arch 支持：

```text
general.architecture = "qwen35"
general.architecture = "qwen35moe"
general.architecture = "qwen3vl"
general.architecture = "qwen3vlmoe"
```

mmproj arch 支持：

```text
general.architecture = "clip"
```

在 model config 中新增：

```cpp
enum class ModelArch {
  qwen3,
  qwen35,
  qwen35moe,
  qwen3vl,
  qwen3vlmoe,
};

enum class Qwen35LayerKind {
  full_attention,
  linear_attention,
};

struct Qwen35Config {
  ModelArch arch;
  std::vector<std::int64_t> rope_dimension_sections;
  std::vector<bool> attention_recurrent_layers;
  std::int64_t main_layer_count{0};
  std::int64_t total_layer_count{0};
  std::int64_t full_attention_interval{4};
  std::int64_t linear_conv_kernel_dim{0};
  std::int64_t linear_key_head_dim{0};
  std::int64_t linear_value_head_dim{0};
  std::int64_t linear_num_key_heads{0};
  std::int64_t linear_num_value_heads{0};
  std::int64_t linear_inner_size{0};
  std::int64_t mtp_num_hidden_layers{0};
  std::int64_t expert_count{0};
  std::int64_t expert_used_count{0};
  std::int64_t expert_feed_forward_length{0};
  std::int64_t expert_shared_feed_forward_length{0};
  float expert_weights_scale{0.0f};
};
```

字段映射按 llama.cpp `src/llama-arch.cpp` / `conversion/qwen.py`。表中的 `{arch}` 是 `qwen35` 或 `qwen35moe`：

| GGUF metadata key | llama.cpp hparam | 本仓库字段 |
| --- | --- | --- |
| `{arch}.block_count` | `n_layer_all` | `total_layer_count` |
| `{arch}.embedding_length` | `n_embd` | `hidden_size` |
| `{arch}.feed_forward_length` | `n_ff` | `intermediate_size` |
| `{arch}.attention.head_count` | `n_head` | `num_attention_heads` |
| `{arch}.attention.head_count_kv` | `n_head_kv` | `num_key_value_heads` |
| `{arch}.attention.layer_norm_rms_epsilon` | `f_norm_rms_eps` | `rms_norm_eps` |
| `{arch}.rope.freq_base` | `freq_base` | `rope_theta` |
| `{arch}.rope.dimension_sections` | `rope_sections` | `rope_dimension_sections` |
| `{arch}.attention.recurrent_layers` | `is_recr_impl` | `attention_recurrent_layers` |
| `{arch}.full_attention_interval` | `full_attention_interval` | `full_attention_interval` |
| `{arch}.ssm.conv_kernel` | `ssm_d_conv` | `linear_conv_kernel_dim` |
| `{arch}.ssm.state_size` | `ssm_d_state` | `linear_key_head_dim`, `linear_value_head_dim` |
| `{arch}.ssm.group_count` | `ssm_n_group` | `linear_num_key_heads` |
| `{arch}.ssm.time_step_rank` | `ssm_dt_rank` | `linear_num_value_heads` |
| `{arch}.ssm.inner_size` | `ssm_d_inner` | `linear_inner_size` |
| `{arch}.nextn_predict_layers` | `n_layer_nextn` | `mtp_num_hidden_layers` |
| `{arch}.expert_count` | `n_expert` | `expert_count` |
| `{arch}.expert_used_count` | `n_expert_used` | `expert_used_count` |
| `{arch}.expert_feed_forward_length` | `n_ff_exp` | `expert_feed_forward_length` |
| `{arch}.expert_shared_feed_forward_length` | `n_ff_shexp` | `expert_shared_feed_forward_length` |
| `{arch}.expert_weights_scale` | `expert_weights_scale` | `expert_weights_scale` |
| `tokenizer.ggml.*` | tokenizer vocab/config | tokenizer config |

如果 `rope.dimension_sections` 缺失，使用 llama.cpp 默认 `[11, 11, 10, 0]`。如果 `attention.recurrent_layers` 缺失，则按 llama.cpp 规则生成：

```cpp
is_recurrent[layer] = layer < main_layer_count &&
    ((layer + 1) % full_attention_interval) != 0;
```

MTP 层必须从主干层数中分离：

```text
main_layer_count = total_layer_count - mtp_num_hidden_layers
mtp_layer_count  = mtp_num_hidden_layers
```

主模型 forward 只遍历 `main_layer_count`。MTP/NextN draft graph 遍历 `[main_layer_count, total_layer_count)`。

## Tokenizer

Qwen3.5 tokenizer 从 GGUF metadata 读取，不再从 `tokenizer.json` / `tokenizer_config.json` 读取。实现以 llama.cpp 为准迁移：

- `tokenizer.ggml.model`
- `tokenizer.ggml.pre`
- `tokenizer.ggml.tokens`
- `tokenizer.ggml.token_type`
- `tokenizer.ggml.scores`
- `tokenizer.ggml.merges`
- `tokenizer.ggml.bos_token_id`
- `tokenizer.ggml.eos_token_id`
- `tokenizer.ggml.eot_token_id`
- `tokenizer.chat_template`

需要迁移 llama.cpp 的 vocab/BPE/unicode 逻辑，而不是扩展现有 `QwenTokenizer` 的 HF JSON parser。现有代码里算法完全相同的部分可以复用，但必须先用 llama.cpp tokenizer output 做等价测试。

Qwen3.5 pre-tokenizer regex 和 Qwen2/Qwen3 的主要差异是 letter run 包含 combining mark：

```text
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|
[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|
\p{N}|
 ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|
\s*[\r\n]+|
\s+(?!\S)|
\s+
```

实现要求：

- 新增 `src/runtime/llama_tokenizer.cpp` 或等价模块，按 llama.cpp `llama-vocab.cpp` / `unicode.cpp` 行为实现。
- 对 `tokenizer.ggml.pre == "qwen35"` 做精确分支。
- 保留其他 GGUF pre-tokenizer 分支，供 `qwen3vl` / `qwen3vlmoe` 和旧 Qwen 模型使用。
- 支持 byte fallback、special token、added token、BPE merge rank、decode escaping。
- chat template 优先使用 GGUF 内嵌 `tokenizer.chat_template`；缺失时才允许使用本仓库 fallback。
- 多模态 prompt 中的 image/audio placeholder token 必须和 tokenizer special token 表一致，不能用字符串硬编码绕过 tokenizer。

测试要求：

- 覆盖 combining mark，例如拉丁字符加重音组合符。
- 覆盖中文、emoji、换行、前导空格、ChatML special token。
- 用同一段文本和同一 GGUF tokenizer metadata 对齐 llama.cpp token id 序列。

## 权重布局

### 通用权重

主干权重使用 llama.cpp GGUF 张量名：

```text
token_embd.weight
output_norm.weight
output.weight
blk.{i}.attn_norm.weight
blk.{i}.post_attention_norm.weight
```

如果 `output.weight` 缺失，按 llama.cpp 逻辑复用 `token_embd.weight`。

### Dense MLP

Qwen3.5 dense FFN 使用：

```text
blk.{i}.ffn_gate.weight
blk.{i}.ffn_up.weight
blk.{i}.ffn_down.weight
```

计算：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

### Qwen3.5 MoE FFN

Qwen3.5 MoE 使用 routed experts 加 gated shared expert。权重：

```text
blk.{i}.ffn_gate_inp.weight
blk.{i}.ffn_up_exps.weight
blk.{i}.ffn_gate_exps.weight
blk.{i}.ffn_down_exps.weight
blk.{i}.ffn_gate_up_exps.weight
blk.{i}.ffn_gate_inp_shexp.weight
blk.{i}.ffn_gate_shexp.weight
blk.{i}.ffn_up_shexp.weight
blk.{i}.ffn_down_shexp.weight
```

`ffn_gate_up_exps` 是可选 fused tensor。loader 需要支持两种 GGUF：

- 有 `ffn_gate_up_exps` 时按 fused gate/up expert 权重绑定。
- 没有 `ffn_gate_up_exps` 时使用独立 `ffn_gate_exps` 和 `ffn_up_exps`。

forward 逻辑：

```text
router_logits = ffn_gate_inp(x)
topk_experts  = top_k(softmax(router_logits), expert_used_count)
routed_out    = sum(topk_weight * expert_swiglu(x, expert_id))

shared_gate   = sigmoid(ffn_gate_inp_shexp(x))
shared_out    = shared_swiglu(x) * shared_gate
ffn_out       = routed_out + shared_out
```

必须保留 `expert_weights_scale` 和 llama.cpp 的 expert gating 语义。MoE routing 的 top-k、权重归一、expert dispatch 和 combine 都需要设备侧实现；CPU 只作为 correctness reference。

### Full Attention 层

full attention 层使用：

```text
blk.{i}.attn_q.weight
blk.{i}.attn_k.weight
blk.{i}.attn_v.weight
blk.{i}.attn_output.weight
blk.{i}.attn_q_norm.weight
blk.{i}.attn_k_norm.weight
```

Qwen3.5 full attention 的 `q_proj` 输出是 query 和 gate 拼接：

```text
q_proj output shape = [2 * num_attention_heads * head_dim]
query = first half per head
gate  = second half per head
```

forward 顺序按 llama.cpp：

```text
x_norm = RMSNorm(x, input_layernorm)
q_full = q_proj(x_norm)
q      = q_full.query
gate   = q_full.gate
k      = k_proj(x_norm)
v      = v_proj(x_norm)
q      = RMSNorm(q, q_norm) + MRoPE
k      = RMSNorm(k, k_norm) + MRoPE
attn   = causal_gqa_attention(q, k_cache, v_cache)
attn   = attn * sigmoid(gate)
x      = x + o_proj(attn)
```

gate 乘法发生在 attention 输出之后、`o_proj` 之前。

### Linear Attention 层

recurrent linear attention 层使用：

```text
blk.{i}.attn_qkv.weight
blk.{i}.attn_gate.weight
blk.{i}.ssm_conv1d.weight
blk.{i}.ssm_dt.bias
blk.{i}.ssm_a
blk.{i}.ssm_beta.weight
blk.{i}.ssm_alpha.weight
blk.{i}.ssm_norm.weight
blk.{i}.ssm_out.weight
```

GGUF conversion 已经完成这些变换：

- HF `conv1d.weight` squeeze 成 `blk.{i}.ssm_conv1d.weight`。
- HF `dt_bias` 映射到 `blk.{i}.ssm_dt.bias`。
- HF `A_log` 转换为 `blk.{i}.ssm_a = -exp(A_log)`。
- HF V 相关权重按 `_LinearAttentionVReorderBase` 重排。

运行时禁止再次执行这些转换，只校验 GGUF 张量形状和 dtype。

维度定义：

```text
head_k_dim = linear_key_head_dim
head_v_dim = linear_value_head_dim
num_k_heads = linear_num_key_heads
num_v_heads = linear_num_value_heads
d_inner = head_v_dim * num_v_heads
key_dim = head_k_dim * num_k_heads
value_dim = head_v_dim * num_v_heads
conv_dim = key_dim * 2 + value_dim
```

## GGUF 量化权重

GGUF reader 必须保留 tensor 的 GGML dtype 和 block layout。权重 store 不能把所有 quantized weights 在 load 时全量反量化成 F16/F32，否则内存和带宽都无法对齐 llama.cpp。

支持 dtype 范围：

- scalar/float: `F32`, `F16`, `BF16`。
- legacy quants: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`。
- K-quants: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`。
- IQ quants: 按 llama.cpp 当前 Metal 支持面实现常用 `IQ*` 类型，并用 gguf-py quant tests 对齐 block decode。

实现分层：

- CPU reference：实现 block dequant，覆盖所有目标 quant type，用于 tests 和数值 oracle。
- MPSGraph path：只用于 F16/BF16/F32 matmul 和非量化算子。
- Metal quant path：为 quantized 2D/3D matmul 提供 `qmatmul` kernels，输入 activation 为 F16/F32，权重按 GGUF block 直接读取。
- fallback：如果某个小 tensor 被量化但不走 matmul，允许按 tensor 粒度一次性反量化到设备临时 buffer，保证 correctness。

量化策略按 llama.cpp `llama-quant.cpp`：

- norm、router gate、小型 SSM conv、position embedding、部分 multimodal patch/merger tensor 通常保持 float。
- 2D matmul weight 和 3D expert weight 是主要量化对象。
- `output.weight` 是否量化取决于 GGUF；runtime 不做重新量化。

性能验收必须分别跑 F16/BF16 baseline 和至少一种 common quant GGUF，例如 Q4_K_M/Q5_K_M/Q8_0，指标和 llama.cpp 同格式比较。

## Cache 设计

Qwen3.5 需要 hybrid cache：full attention KV cache + recurrent R/S cache。

### Full Attention KV Cache

只给 full attention 层和 MTP draft attention 层分配 KV cache。布局：

```text
key_cache[full_layer_index][cell][num_key_value_heads][head_dim]
value_cache[full_layer_index][cell][num_key_value_heads][head_dim]
```

cache cell 记录：

```cpp
struct CacheCell {
  std::int64_t pos{-1};
  std::vector<std::int64_t> seq_ids;
  std::int32_t src{-1};
  std::int32_t tail{-1};
};
```

多 sequence 时同一个 cell 可被多个 `seq_id` 引用，用于 prompt sharing 或 batch 内共享前缀。attention mask 从 `(seq_id, pos)` 关系生成。

### Recurrent Cache

linear attention 层需要两个 state：

```text
R conv state: (linear_conv_kernel_dim - 1) * (d_inner + 2 * num_k_heads * head_k_dim)
S delta state: head_k_dim * d_inner
```

对应 llama.cpp：

```text
n_embd_r() = (ssm_d_conv - 1) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state)
n_embd_s() = ssm_d_state * ssm_d_inner
```

为支持多 sequence 和 rollback，布局按 llama.cpp recurrent memory：

```text
recurrent_r[layer][mem_size * (1 + n_rs_seq)][n_embd_r]
recurrent_s[layer][mem_size * (1 + n_rs_seq)][n_embd_s]
rs_idx[seq_id] in [0, n_rs_seq]
```

第 0 plane 是当前 state，其余 `1..n_rs_seq` 是每个 sequence 的回滚快照。`seq_rm(seq_id, p0, p1)` 删除尾部 token 时，如果回滚距离在 `n_rs_seq` 内，则只移动 `rs_idx[seq_id]` 并把 tail position 设为 `p0 - 1`；否则返回失败，让上层重新 prefill 该 sequence。

prefill 批处理时：

- R cache 写入最后 `kernel - 1` 个 convolved input。
- S cache 写入 gated delta net 的最终 state。
- 对每个 sequence 独立写 state 和 rollback snapshots。

decode 时：

- 每个 sequence 读取自己的 current state plane。
- 接受 token 后更新 current plane 并按 `n_rs_seq` 滚动保存快照。
- speculative rejection 或 truncation 通过 `rs_idx` 恢复，不重新计算完整 prefix。

## Linear Attention 算子

linear attention 层 forward：

```text
x_norm = RMSNorm(x, input_layernorm)
qkv    = in_proj_qkv(x_norm)
z      = in_proj_z(x_norm)
beta   = sigmoid(in_proj_b(x_norm))
alpha  = softplus(in_proj_a(x_norm) + dt_bias)
gate   = alpha * ssm_a

conv_input = concat(previous_conv_state, qkv)
conv_out   = silu(depthwise_conv1d(conv_input, conv1d.weight))
q, k, v    = split(conv_out)
q          = l2_norm(q)
k          = l2_norm(k)

core       = gated_delta_net(q, k, v, gate, beta, recurrent_s)
core       = RMSNorm(core, linear_attn.norm) * silu(z)
out        = linear_attn.out_proj(core)
x          = x + out
```

必须按 llama.cpp `build_delta_net_*` 行为实现 `gated_delta_net`：

- decode `n_tokens == 1` 使用 autoregressive/fused 路径。
- prefill `n_tokens > 1` 使用 chunking/fused 路径。
- 输入形状约定为 `[head_dim, heads, tokens, seqs]`。
- 输出形状为 `[head_v_dim, num_v_heads, tokens, seqs]`。
- 新 state 形状为 `[head_k_dim, head_v_dim, heads, seqs]`，实际内存展平成 `n_embd_s`。

CPU reference 实现 autoregressive 和 chunking 两条路径，用于小张量测试。性能路径必须实现 batched prefill 和 decode fused kernel，不能在 CPU 上执行 linear attention。

## MRoPE

当前 `apply_rope` / `rope_f32` 是普通 half-split RoPE。Qwen3.5 需要 MRoPE：

- 支持 `rope_dimension_sections`。
- 默认 `[11, 11, 10, 0]`，总旋转维度为 32。
- 使用 `rope_theta` 和 position ids 生成 cos/sin。
- 行为以 llama.cpp `ggml_rope_multi` 为准。

实现方式：

- CPU 增加 `apply_mrope(...)`，作为 correctness oracle。
- MPSGraph/Metal 增加 `mrope_f32` 或直接集成进 full attention fused layer。
- 多模态输入需要支持 text/image/audio 对应 position ids 和 MRoPE section。
- 对 Qwen3 保留现有 RoPE 路径，避免回归。

## MTP/NextN Speculative Decoding

GGUF conversion 会把 HF `mtp.*` 映射到 layer-indexed nextn tensor。MTP block 位于主干层之后：

```text
blk.{main_layer_count + k}.nextn.eh_proj.weight
blk.{main_layer_count + k}.nextn.enorm.weight
blk.{main_layer_count + k}.nextn.hnorm.weight
blk.{main_layer_count + k}.nextn.embed_tokens.weight
blk.{main_layer_count + k}.nextn.shared_head_head.weight
blk.{main_layer_count + k}.nextn.shared_head_norm.weight
```

MTP block 本身还带一个 full-attention decoder block：

```text
blk.{main_layer_count + k}.attn_norm.weight
blk.{main_layer_count + k}.attn_q.weight
blk.{main_layer_count + k}.attn_k.weight
blk.{main_layer_count + k}.attn_v.weight
blk.{main_layer_count + k}.attn_output.weight
blk.{main_layer_count + k}.attn_q_norm.weight
blk.{main_layer_count + k}.attn_k_norm.weight
blk.{main_layer_count + k}.post_attention_norm.weight
```

Dense MTP 使用 dense FFN；MoE MTP 使用同 Qwen3.5 MoE 的 routed expert + shared expert FFN。

主模型 graph 需要暴露 `h_nextn`：

```text
h_nextn = output_norm(hidden_before_lm_head)
```

draft graph 输入：

```text
token embedding for candidate token
h_nextn from main or previous draft step
position ids
draft KV cache
```

draft graph 逻辑：

```text
h_norm   = RMSNorm(h_input, nextn.hnorm)
e_norm   = RMSNorm(token_embedding, nextn.enorm)
cur      = nextn.eh_proj(concat(e_norm, h_norm))
cur      = full_attention_block(cur)
cur      = dense_or_moe_ffn(cur)
cur      = RMSNorm(cur, nextn.shared_head_norm or output_norm)
logits   = (nextn.shared_head_head or output)(cur)
h_nextn  = cur
```

speculative driver：

1. 主模型 decode 一个 token，同时保留 `h_nextn` 和 pre-accept cache snapshot。
2. MTP/NextN draft graph 连续提出 `K = mtp_num_hidden_layers` 或配置指定的 draft tokens。
3. 主模型用批量 verify pass 验证 draft tokens。
4. 接受连续匹配 token；遇到 rejection 时采样主模型该位置 token。
5. 对 accepted tokens commit KV/recurrent cache。
6. 对 rejected draft tokens 用 recurrent rollback 和 KV tail truncation 恢复。

多 batch 时，每个 sequence 独立维护 draft chain、accept count、rollback depth 和 sampling state。

## 多 Batch、多 Sequence Recurrent Rollback

需要新增 `Qwen35HybridCache`，统一管理：

- full attention KV cells。
- recurrent R/S state planes。
- `seq_id -> tail cell`。
- `seq_id -> rs_idx`。
- batch/ubatch split。
- speculative snapshots。

接口建议：

```cpp
struct Qwen35CacheView {
  std::span<std::int32_t> token_positions;
  std::span<std::int32_t> seq_ids;
  DeviceTensor kv_indices;
  DeviceTensor recurrent_indices;
  DeviceTensor attention_mask;
};

class Qwen35HybridCache {
 public:
  Qwen35CacheView prepare_prefill(const BatchInput& batch);
  Qwen35CacheView prepare_decode(const BatchInput& batch);
  void commit(const BatchCommit& commit);
  bool rollback(std::int64_t seq_id, std::int64_t new_length);
  bool remove_sequence(std::int64_t seq_id);
  void clear();
};
```

batch split 规则：

- 没有 rollback 压力时允许 equal split，提高 prefill 吞吐。
- `n_rs_seq > 0` 时按 sequence split，和 llama.cpp `[TAG_RECURRENT_ROLLBACK_SPLITS]` 行为一致，避免一个 ubatch 内多个 sequence 的 state snapshot 顺序混乱。
- decode 支持 batch 中每个 sequence 长度不同、position 不同、是否启用 speculative 不同。

rollback 成功条件：

- 删除的是 sequence tail。
- 删除 token 数量 `<= n_rs_seq`。
- KV cache tail cell 可以截断或移除对应 `seq_id`。
- recurrent `rs_idx` 可移动到对应 snapshot。

rollback 失败时：

- 清理该 sequence 的 cache。
- 从保留 token 重新 prefill。
- 记录 metric，不能静默退化成错误 state。

## 多模态 Qwen3.5 VL/Omni

多模态运行时由两部分组成：

```text
text GGUF:   qwen35 / qwen35moe / qwen3vl / qwen3vlmoe
mmproj GGUF: clip arch, vision/audio encoder + projector
```

CLI/API 形态：

```text
kraken-infer chat --model text.gguf --mmproj mmproj.gguf --image img.png
kraken-infer chat --model text.gguf --mmproj mmproj.gguf --audio input.wav
kraken-infer chat --model text.gguf --mmproj mmproj.gguf --image img.png --audio input.wav
```

mmproj metadata：

```text
clip.has_vision_encoder
clip.has_audio_encoder
clip.projector_type
clip.vision.projector_type
clip.vision.image_size
clip.vision.patch_size
clip.vision.spatial_merge_size
clip.vision.block_count
clip.vision.embedding_length
clip.vision.feed_forward_length
clip.vision.attention.head_count
clip.vision.attention.layer_norm_epsilon
clip.vision.is_deepstack_layers
clip.audio.projector_type
clip.audio.num_mel_bins
clip.audio.embedding_length
clip.audio.feed_forward_length
clip.audio.block_count
clip.audio.chunk_size
clip.audio.conv_kernel_size
clip.audio.max_pos_emb
clip.audio.projector.stack_factor
clip.audio.projector.window_size
clip.audio.projector.downsample_rate
clip.audio.projector.head_count
```

Qwen3VL vision path follows llama.cpp `tools/mtmd/models/qwen3vl.cpp`:

- image preprocessing: resize/tile/normalize according to mmproj metadata。
- patch embedding: Qwen3VL Conv3D temporal kernel is split into `v.patch_embd.weight` and `v.patch_embd.weight.1` by converter。
- vision transformer: `v.blk.{i}.*` tensors。
- merger/projector: `mm.0.*` and `mm.2.*` for Qwen3VL merger。
- deepstack: `v.deepstack.{i}.norm/fc1/fc2` for `clip.vision.is_deepstack_layers`。

Qwen3 Omni audio path follows llama.cpp `tools/mtmd/models/qwen3a.cpp` and `mtmd-audio.cpp`:

- audio preprocessing: resample、mel feature、chunk/window layout。
- audio tower: `a.*` tensors from mmproj GGUF。
- audio projector: `clip.audio.projector_type = "qwen3a"` 对应 projector。

embedding injection：

- tokenizer 生成包含 media placeholder 的 token 序列。
- vision/audio encoder 输出 projected embeddings。
- runtime 用 projected embeddings 替换 placeholder span 的 token embeddings。
- text token、image patch token、audio frame token 的 position ids 和 MRoPE section 需要和 llama.cpp mtmd 行为对齐。
- 多图、多音频输入时，每个 media item 保留独立 grid/chunk metadata，batch 内不能混用 shape。

Omni 以 llama.cpp GGUF scope 为准：text decoder 接收 vision/audio embeddings 并生成文本 token。若 GGUF 包含额外 talker/vocoder 模块，需要作为新的 mm output backend 接入同一 media pipeline。

## Runtime 结构

不要把 Qwen3.5 硬塞进现有 `QwenMpsGraphModel` 的 Qwen3 假设。建议拆分：

```text
src/runtime/qwen35_loader.cpp
src/runtime/qwen35_hybrid_cache.cpp
src/runtime/qwen35_speculative_decoder.cpp
src/runtime/qwen35_multimodal.cpp
src/runtime/cpu/qwen35_cpu_model.cpp
src/runtime/cpu/qwen35moe_cpu_model.cpp
src/backends/mpsgraph/qwen35_mpsgraph_model.cpp
src/backends/mpsgraph/qwen35moe_mpsgraph_model.cpp
src/backends/metal/qwen35_kernels.mm
src/backends/metal/gguf_quant_kernels.mm
```

公共结构：

```text
GgufTensorView
GgufQuantTensor
Qwen35DenseMlpWeights
Qwen35MoeWeights
Qwen35FullAttentionWeights
Qwen35LinearAttentionWeights
Qwen35MtpWeights
Qwen35LayerKind
Qwen35HybridCache
Qwen35MultimodalProjector
```

运行时选择：

```text
existing Qwen3 model directory with config.json/model.safetensors -> existing Qwen3 path
GGUF general.architecture == "qwen35"    -> Qwen3.5 dense path
GGUF general.architecture == "qwen35moe" -> Qwen3.5 MoE path
GGUF general.architecture == "qwen3vl"   -> Qwen3 VL text path + optional mmproj
GGUF general.architecture == "qwen3vlmoe"-> Qwen3 VL/Omni MoE text path + optional mmproj
GGUF general.architecture == "clip"      -> mmproj loader, must be paired with text GGUF
```

`kraken-infer chat` 需要支持：

- `--model path/to/text.gguf`
- `--mmproj path/to/mmproj.gguf`
- `--image`
- `--audio`
- `--batch-size`
- `--parallel-seqs`
- `--n-rs-seq`
- `--speculative`
- `--draft-tokens`

## 性能方案

当前本仓库 MPSGraph Qwen3 path 是逐 token 调用 `transformer_layer_f32`。这对 Qwen3.5 不够：

- Qwen3.5 prefill 必须能一次处理多个 prompt token 和多个 sequence。
- recurrent linear attention 的 chunking/fused 路径是性能关键。
- MoE routing 和 expert matmul 不能在 CPU 上执行。
- quantized matmul 需要自定义 Metal kernels。
- full attention、MTP draft graph 和 verification pass 需要减少 host 和 graph execute 次数。

性能路径：

1. 设备常驻权重和 cache，decode 期间不做 host/device 往返。
2. `qwen35_prefill(batch)` 一次处理多 sequence prompt。
3. `qwen35_decode(batch)` 一次处理多 sequence next-token。
4. `qwen35_verify(batch, draft_tokens)` 对 speculative draft 做 batched verify。
5. full attention fused graph/kernel：
   - input RMSNorm
   - q/k/v projection
   - q/k norm
   - MRoPE
   - KV store
   - causal GQA attention
   - sigmoid gate
   - o projection
   - residual
6. linear attention fused graph/kernel：
   - qkv/z/alpha/beta projection
   - conv state update
   - depthwise conv
   - l2 norm q/k
   - gated delta net
   - gated RMSNorm
   - out projection
   - residual
7. MoE fused path：
   - router projection
   - top-k softmax
   - token/expert dispatch
   - expert gate/up/down matmul
   - shared expert matmul
   - combine
8. quantized matmul path：
   - direct block decode in Metal
   - activation tiling shared by dense, attention, MoE, MTP
9. multimodal path：
   - vision/audio preprocessing can start on CPU, but tower/projector forward should run on device where available。
   - projected embeddings are uploaded once and consumed by text prefill without per-token host copies。

性能验收以同一台机器、同一模型、同一上下文长度、同一精度/量化格式为基准：

- llama.cpp 和本仓库使用同一个 GGUF。
- 两边都使用 Metal/MPSGraph/Metal 设备路径。
- batch、context、prompt、decode 长度、sampling 参数完全一致。
- 记录 prefill tokens/s、decode tokens/s、speculative accepted tokens/s。
- dense 0.8B：decode 吞吐不低于 llama.cpp 的 90%，prefill 吞吐不低于 llama.cpp 的 85%。
- MoE/quant/multimodal：以同配置 llama.cpp 为 baseline，吞吐不低于 85%，显存不超过 115%。

不能接受的性能实现：

- prompt 逐 token prefill。
- 每层多个小 graph execute。
- linear attention、MoE routing 或 quantized matmul 在 CPU 上执行。
- 每步把 hidden/cache 拉回 host。
- decode 时重新计算完整 prefix。
- speculative rejection 后无条件重算整个 batch。

## 验证计划

### 单元和 Smoke Test

新增 focused tests：

- `test_qwen35_gguf_metadata()`
  - 解析 `{arch}.ssm.*`、`full_attention_interval`、`rope.dimension_sections`。
  - 缺省 `mrope_section` 时得到 `[11, 11, 10, 0]`。
  - 缺省 recurrent layers 时每第 4 层为 full attention。
- `test_qwen35moe_gguf_metadata()`
  - 解析 expert count、top-k、expert FFN、shared expert FFN。
- `test_llama_tokenizer_qwen35()`
  - 从 tiny GGUF tokenizer metadata 读取 vocab/merges/special tokens。
  - 覆盖 combining mark，并和 llama.cpp token id 序列对齐。
- `test_qwen35_gguf_weight_shape_validation()`
  - full attention q_proj 是 `2 * attn_dim`。
  - linear attention 检查 `blk.N.attn_qkv`、`blk.N.attn_gate`、`blk.N.ssm_*`。
- `test_qwen35moe_weight_shape_validation()`
  - routed experts、fused gate/up experts、shared experts shape。
- `test_qwen35_quant_dequant_blocks()`
  - 对齐 llama.cpp/gguf-py dequant output。
- `test_qwen35_quant_matmul_tiny()`
  - tiny F32 activation x quantized weight 对齐 dequant reference。
- `test_qwen35_recurrent_cache_layout()`
  - 校验 `n_embd_r`、`n_embd_s`、`mem_size * (1 + n_rs_seq)`。
- `test_qwen35_recurrent_rollback()`
  - 多 sequence tail rollback 和失败重算路径。
- `test_qwen35_linear_attention_single_token()`
  - 小张量 autoregressive gated delta net。
- `test_qwen35_linear_attention_chunking()`
  - prefill chunking 对齐 token-by-token reference。
- `test_qwen35_full_attention_gate()`
  - 验证 sigmoid gate 在 attention 后、o_proj 前。
- `test_qwen35_mtp_graph()`
  - `nextn.eh_proj/enorm/hnorm`、shared head fallback、dense/MoE MTP。
- `test_qwen35_speculative_accept_reject()`
  - accepted commit 和 rejected rollback。
- `test_qwen35_multimodal_mmproj_metadata()`
  - 解析 Qwen3VL/Qwen3A mmproj metadata。
- `test_qwen35_multimodal_embedding_injection()`
  - media placeholder span 替换 projected embeddings。

### 数值对齐

用三层递进对齐：

1. CPU 小张量 fixture，对齐手算结果。
2. 本仓库 CPU/MPSGraph/Metal 对齐同一 tiny GGUF fixture。
3. 本仓库真实 GGUF 在固定 prompt 下对齐 llama.cpp dump。

建议增加 debug dump 对齐点：

```text
layer.{i}.input_norm
layer.{i}.full.q_norm_mrope
layer.{i}.full.attn_gated
layer.{i}.linear.conv_out
layer.{i}.linear.gdn_out
layer.{i}.linear.state_s
layer.{i}.moe.router_logits
layer.{i}.moe.topk
layer.{i}.moe.routed_out
layer.{i}.moe.shared_out
layer.{i}.mlp_out
mtp.{k}.eh_proj
mtp.{k}.logits
final_norm
logits
```

多模态 debug dump：

```text
vision.patch_embd
vision.block.{i}.out
vision.merger.out
audio.features
audio.block.{i}.out
audio.projector.out
text.injected_embeddings
```

### 性能基准

新增脚本：

```text
scripts/bench_qwen35_dense.sh
scripts/bench_qwen35_moe.sh
scripts/bench_qwen35_quant.sh
scripts/bench_qwen35_speculative.sh
scripts/bench_qwen35_multimodal.sh
scripts/compare_qwen35_llamacpp.py
```

基准参数：

```text
prompt tokens: 128, 512, 2048
decode tokens: 128
batch: 1, 4, 8
parallel sequences: 1, 4, 8
temperature: 0 or fixed-seed sampling
context: same as llama.cpp run
quant: F16/BF16, Q4_K_M, Q5_K_M, Q8_0
speculative draft tokens: 1, 2, 4
```

输出至少包含：

```text
prefill tokens/s
decode tokens/s
speculative accepted tokens/s
acceptance rate
peak device memory
graph build/compile/execute count
host-to-device/device-to-host bytes
rollback count
rollback re-prefill count
```

## 实施步骤

1. GGUF/tokenizer/shape validation
   - 新增 GGUF reader，支持 text/mmproj/split GGUF。
   - 迁移 llama.cpp tokenizer/vocab/BPE/unicode 逻辑。
   - 增加 qwen35/qwen35moe/qwen3vl/qwen3vlmoe metadata 解析。
   - 增加 GGUF tensor shape 和 quant dtype 校验。

2. CPU correctness path
   - 新增 Qwen3.5 layer 类型和 GGUF 权重绑定。
   - 实现 dense full attention gate。
   - 实现 linear attention autoregressive 和 chunking reference。
   - 实现 MoE routed/shared expert reference。
   - 实现 MRoPE reference。

3. Hybrid cache and rollback
   - 实现 full attention KV cache。
   - 实现 recurrent R/S cache。
   - 支持多 batch、多 sequence。
   - 支持 `n_rs_seq` snapshots 和 `seq_rm` tail rollback。

4. MPSGraph/Metal performance path
   - 设备侧 recurrent cache。
   - full attention fused path。
   - linear attention fused GDN path。
   - dense MLP fused path。
   - MoE router/dispatch/expert fused path。
   - device-side logits/argmax 或最小 readback。

5. Quantized GGUF
   - CPU block dequant tests。
   - Metal qmatmul kernels。
   - quantized dense/attention/MoE/MTP matmul 接入。
   - F16/BF16 和 quant GGUF 数值/性能对齐。

6. MTP/NextN speculative decoding
   - 加载 nextn tensors。
   - 主模型暴露 `h_nextn`。
   - 实现 dense/MoE draft graph。
   - 实现 batched verify、accept/reject、rollback commit。

7. Multimodal VL/Omni
   - 加载 mmproj GGUF。
   - 实现 Qwen3VL vision preprocessing/tower/merger/deepstack。
   - 实现 Qwen3A audio preprocessing/tower/projector。
   - 实现 media embedding injection 和 multimodal MRoPE position。
   - 接入 chat CLI/API。

8. Benchmark and regression gate
   - 固定 llama.cpp baseline。
   - 在本仓库输出 profile JSON。
   - Dense/MoE/quant/speculative/multimodal 全部达到性能阈值后宣布完成。

## 风险和处理

- **MPSGraph 缺少高效 gated delta net**：多个小 graph 会达不到目标。处理方式是自定义 Metal kernel，MPSGraph 只负责适合的 matmul/norm fallback。
- **MoE dispatch 开销过高**：batch 小时 top-k dispatch 容易被 launch overhead 吃掉。处理方式是合并 router/top-k/dispatch/combine，decode 使用 token-major 小 batch kernel。
- **量化 matmul 覆盖不足**：GGUF quant type 多。处理方式是先以 llama.cpp Metal 常用类型建表，CPU dequant 覆盖全类型，Metal qmatmul 按 benchmark 优先级补齐。
- **MRoPE/多模态 position 偏差**：必须用 llama.cpp dump 对齐，不能只靠生成文本观察。
- **V head 重排错误**：转换已完成 V reorder，运行时只校验，不再重排。通过小张量 fixture 覆盖 K/V head 不同的情况。
- **MTP cache 污染主模型 cache**：draft KV/recurrent state 必须和 main cache 隔离，只在 accept 后 commit。
- **rollback 快照不足**：`n_rs_seq` 过小时 rejection 可能需要 re-prefill。CLI 需要暴露配置并记录 metric。
- **多模态 shape 动态性**：图片 tile 数、音频 chunk 数会改变 embedding span。prefill batch 需要按 shape 分组或使用动态 shape graph。
- **性能比较不公平**：必须记录精度、量化格式、上下文、prompt/decode 长度、batch、后端和 llama.cpp commit。

## 完成标准

实现完成需要同时满足：

- `ctest --preset debug` 通过。
- Qwen3 0.6B 现有 smoke test 不回归。
- Qwen3.5 dense 0.8B GGUF 固定 prompt logits top token 和 llama.cpp 一致。
- Qwen3.5 MoE GGUF 固定 prompt logits top token 和 llama.cpp 一致。
- Qwen3.5 quant GGUF 的 logits 和 llama.cpp 在同量化格式下满足误差阈值。
- MTP/NextN speculative decoding 的 accept/reject 输出和非 speculative 解码一致。
- 多 batch、多 sequence recurrent rollback tests 通过，并覆盖 rejection rollback。
- Qwen3.5 VL image prompt 和 llama.cpp mtmd 输出对齐。
- Qwen3.5 Omni audio/image prompt 和 llama.cpp mtmd 输出对齐。
- 128/512/2048 prompt、batch 1/4/8、128 decode benchmark 达到本文性能阈值。
- `./build/debug/kraken-infer chat --model models/qwen3.5-0.8b/qwen35-0.8b-f16.gguf --device mpsgraph` 能生成稳定文本。
- `./build/debug/kraken-infer chat --model text.gguf --mmproj mmproj.gguf --image image.png --device mpsgraph` 能完成 VL 推理。
- `./build/debug/kraken-infer chat --model text.gguf --mmproj mmproj.gguf --audio audio.wav --device mpsgraph` 能完成 Omni/audio 推理。
