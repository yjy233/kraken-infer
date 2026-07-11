# Qwen3.5 0.8B MTP llama.cpp 调研与 kraken-infer 实现方案

本文调研 `~/code/llama.cpp` 中 Qwen3.5 0.8B 的 MTP/NextN speculative
decoding 实现，并给出 `kraken-infer` 的原生 Metal 实现路线。

用户口头说的 `qwen3 0.8b` 在当前仓库语境下对应 Qwen3.5 0.8B：GGUF
`general.architecture = "qwen35"`，本仓库已有主干文本推理路径。

## 结论

llama.cpp 的 Qwen3.5 MTP 不是把主 decoder 一次输出多个 token，而是把 MTP block
作为 speculative draft context：

1. target context 正常跑 Qwen3.5 主干层。
2. target 每次 prefill / verify 后导出 post-output-norm hidden，即 `h_nextn`。
3. MTP context 读取 token embedding 和前一个位置的 `h_nextn`，经过
   `nextn.eh_proj` 和一个附加的 full-attention decoder block 生成 draft logits。
4. speculative scheduler 用 MTP context 先草拟若干 token，再让 target context 一次验证
   `last_sampled + draft_tokens`。
5. target sampler 从每一行 logits 逐个采样；只要采样 token 和 draft token 相同就接受，
   第一个不相同处停止。

截至 2026-07-10，`kraken-infer` 已完成第一版 correctness-first 原生文本 MTP：

- `ModelConfig::mtp_num_hidden_layers` 已从 GGUF `nextn_predict_layers` 读取。
- `Qwen35WeightMap` 已能映射 `Qwen35LayerKind::mtp` 和 `nextn.*` tensors。
- 已下载并验证 `models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf`：
  `MTP/NextN layers: 1`。
- 主推理现在导出 target `h_nextn`，prefill 会同步推进 MTP cache。
- 已实现 MTP KV cache、`nextn.eh_proj/enorm/hnorm` forward、shared head fallback
  和 draft logits。
- 已实现 greedy speculative loop：target 产出 token，MTP 连续 draft，target batch
  verify 后逐行接受匹配 token。
- Gateway/CLI 已支持 `--mtp|--no-mtp`、`--mtp-draft-tokens`、request `mtp` 和
  `mtp_draft_tokens`。
- Gateway/CLI 已支持 `mtp_p_min` / `--mtp-p-min` 置信度门控。
- 非 streaming gateway 响应会返回 `X-Kraken-MTP-*` headers。
- 新增 `scripts/test_qwen35_mtp_gateway.py` 做端到端测试。
- MTP GGUF 中的 Q8_0 projector 和 F32 2D 小矩阵通过 dense F32 transpose fallback
  接入现有 Metal matmul。
- MPS backend 已增加 Q4_K/Q5_K/Q6_K fused top-1 LM head，MTP draft 可以直接产出
  top-1 token 和 exact top-1 probability，不再 materialize 完整 vocab logits。
- MPS backend 已增加 Q4_K/Q5_K/Q6_K token-only argmax LM head，`mtp_p_min = 0`
  时不再计算 softmax denominator。
- decode loop 已增加 adaptive draft budget：低收益窗口降预算，高收益窗口逐步恢复，
  并通过 CLI、headers、profiler 暴露 `adaptive_budget` / `adaptive_changes`。

当前限制：

- 只在 greedy 路径启用；request 设置 `temperature/top_p/top_k/seed` 会视为 sampling，
  第一版自动禁用 MTP。
- target verify 已经 batch decode；但 sampler/accept 仍是逐行 greedy argmax 对比，
  尚未接入 llama.cpp common sampler 风格的 top-k/top-p accept。
- `cache_prompt` 与 MTP 暂不同时启用；attention KV cache 仍正常使用。
- 图片请求仍通过 `llama-mtmd-cli` 桥接完成 VL，MTP headers 会标记 MTP 未启用。
  llama.cpp 自身的 draft-mtp 代码也仍有 vision token TODO，因此这里不伪装成原生
  VL+MTP。

## 参考文件

llama.cpp：

- `~/code/llama.cpp/src/models/qwen35.cpp`
- `~/code/llama.cpp/src/models/qwen35moe.cpp`
- `~/code/llama.cpp/common/speculative.cpp`
- `~/code/llama.cpp/common/speculative.h`
- `~/code/llama.cpp/tools/server/server-context.cpp`
- `~/code/llama.cpp/common/download.cpp`
- `~/code/llama.cpp/common/arg.cpp`
- `~/code/llama.cpp/gguf-py/gguf/tensor_mapping.py`

kraken-infer：

- `include/toyllm/model/model_config.hpp`
- `src/model/model_config.cpp`
- `include/toyllm/runtime/qwen35_weight_map.hpp`
- `src/runtime/qwen35_weight_map.cpp`
- `include/toyllm/runtime/qwen35_runtime.hpp`
- `src/runtime/qwen35_runtime.cpp`
- `include/toyllm/runtime/openai_gateway.hpp`
- `src/runtime/openai_gateway.cpp`
- `apps/kraken_infer_main.cpp`

外部模型：

- `unsloth/Qwen3.5-0.8B-MTP-GGUF`
- `Qwen/Qwen3.5-0.8B`

## 模型资产

当前本地模型：

```bash
./build/debug/kraken-infer inspect \
  models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
```

本机结果：

```text
Architecture: qwen35
Layers total: 24
Layers main: 24
MTP/NextN layers: 0
```

因此它不能触发 MTP。要测试 MTP，需要换成包含 `nextn_predict_layers > 0` 的 GGUF。
当前可用的 MTP GGUF 发行是：

```text
https://hf-mirror.com/unsloth/Qwen3.5-0.8B-MTP-GGUF
https://huggingface.co/unsloth/Qwen3.5-0.8B-MTP-GGUF
```

可选文件包括：

```text
Qwen3.5-0.8B-Q4_K_M.gguf
Qwen3.5-0.8B-Q5_K_M.gguf
Qwen3.5-0.8B-Q6_K.gguf
Qwen3.5-0.8B-UD-Q4_K_XL.gguf
Qwen3.5-0.8B-UD-Q5_K_XL.gguf
mmproj-BF16.gguf
```

建议本仓库本地路径：

```text
models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf
```

当前已下载并验证：

```bash
./build/debug/kraken-infer inspect \
  models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf
```

本机结果：

```text
Architecture: qwen35
Tensors: 335
Layers total: 25
Layers main: 24
MTP/NextN layers: 1
Validation: ok
```

下载示例：

```bash
mkdir -p models/qwen3.5-0.8b-mtp
curl -L \
  -o models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf \
  https://hf-mirror.com/unsloth/Qwen3.5-0.8B-MTP-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf
```

llama.cpp 的 download 层还支持另一种形态：主模型 GGUF 旁边有 `mtp-*.gguf`
sibling。`common/download.cpp` 会在 `--spec-type draft-mtp` 时尝试寻找 `mtp-`
前缀文件，并把它填到 draft model path。Unsloth 这套 Qwen3.5 0.8B MTP 发布更像
一体化 GGUF，文件名没有 `mtp-` 前缀。

kraken 第一阶段建议支持一体化 MTP GGUF；第二阶段再支持 target/draft 分文件。

## llama.cpp Qwen3.5 MTP 实现

### Metadata

`llama_model_qwen35::load_arch_hparams()`：

- 读取 `LLM_KV_NEXTN_PREDICT_LAYERS` 到 `hparams.n_layer_nextn`。
- 要求 `n_layer_nextn < n_layer_all`。
- `hparams.n_layer()` 表示主干层数，即 `n_layer_all - n_layer_nextn`。
- recurrent layer flags 只对主干层有效；MTP layers 被视作 dense full-attention block。

对应 kraken 字段：

```cpp
model.total_layer_count = gguf["block_count"];
model.mtp_num_hidden_layers = gguf["nextn_predict_layers"];
model.main_layer_count = total_layer_count - mtp_num_hidden_layers;
```

### Tensor Layout

Qwen3.5 dense MTP layer 附加在主干层之后。以 0.8B 为例，如果主干 24 层、MTP 1
层，则 MTP layer index 是 `24`。

MTP layer 复用一个 full-attention Qwen3.5 block 的普通 tensors：

```text
blk.N.attn_norm.weight
blk.N.attn_q.weight
blk.N.attn_k.weight
blk.N.attn_v.weight
blk.N.attn_output.weight
blk.N.attn_q_norm.weight
blk.N.attn_k_norm.weight
blk.N.post_attention_norm.weight
blk.N.ffn_gate.weight
blk.N.ffn_up.weight
blk.N.ffn_down.weight
```

MTP 专属 tensors：

```text
blk.N.nextn.eh_proj.weight
blk.N.nextn.enorm.weight
blk.N.nextn.hnorm.weight
blk.N.nextn.embed_tokens.weight        optional
blk.N.nextn.shared_head_head.weight    optional
blk.N.nextn.shared_head_norm.weight    optional
```

`gguf-py/gguf/tensor_mapping.py` 把 HF tensor 映射成这些 GGUF 名字：

```text
model.layers.{bid}.eh_proj           -> blk.N.nextn.eh_proj.weight
model.layers.{bid}.embed_tokens      -> blk.N.nextn.embed_tokens.weight
model.layers.{bid}.enorm             -> blk.N.nextn.enorm.weight
model.layers.{bid}.hnorm             -> blk.N.nextn.hnorm.weight
model.layers.{bid}.shared_head.head  -> blk.N.nextn.shared_head_head.weight
model.layers.{bid}.shared_head.norm  -> blk.N.nextn.shared_head_norm.weight
```

kraken 已经在 `Qwen35MtpBindings` 中表达了这些 binding。

### 主图输出 h_nextn

Qwen3.5 主图不会执行 MTP layer。它只跑 `0..n_layer-1` 主干层，然后：

```text
main hidden
-> output_norm
-> h_nextn
-> lm_head
-> logits
```

llama.cpp 把 output norm 后的 hidden 记录到 `res->t_h_nextn`。这点很关键：
MTP graph 输入的不是 pre-output-norm hidden，而是 post-output-norm hidden。

kraken 当前 `run_qwen35_output_logits()` 在内部做 output norm 并直接接 lm head，但没有
把 norm 后 hidden 暴露出来。因此实现 MTP 前，需要把 output norm 拆成独立 helper：

```cpp
Result<mps::MpsBuffer> run_qwen35_output_norm(...);
Result<Qwen35LogitsOutput> run_qwen35_lm_head(...);
```

或者让 `Qwen35LogitsOutput` 同时返回：

```cpp
mps::MpsBuffer h_nextn;
mps::MpsBuffer logits;
mps::MpsBuffer argmax_output;
```

### MTP Graph

`llama_model_qwen35::graph_mtp` 目前限制：

```text
n_layer_nextn == 1
```

Qwen3.5 0.8B 的第一阶段实现也应先限制 1 个 MTP layer。

MTP graph 输入：

- token ids，表示当前位置的 token embedding。
- `h`，表示前一个位置的 target/MTP `h_nextn`。
- position ids，用于 MRoPE。
- MTP KV cache。

MTP graph 计算：

```text
tok_embd = nextn.embed_tokens ? nextn.embed_tokens[token] : token_embd[token]
h_norm   = RMSNorm(h, nextn.hnorm)
e_norm   = RMSNorm(tok_embd, nextn.enorm)
cur      = eh_proj(concat(e_norm, h_norm))

inpSA    = cur
cur      = RMSNorm(cur, attn_norm)
QG       = attn_q(cur)
Q        = RMSNorm(split_q(QG), attn_q_norm)
gate     = split_gate(QG)
K        = RMSNorm(attn_k(cur), attn_k_norm)
V        = attn_v(cur)
Q,K      = MRoPE(Q,K, pos)
attn     = causal_attention(Q,K,V, mtp_kv_cache)
attn     = attn * sigmoid(gate)
cur      = attn_output(attn)
cur      = cur + inpSA

ffn_res  = cur
cur      = RMSNorm(cur, post_attention_norm)
cur      = ffn_down(silu(ffn_gate(cur)) * ffn_up(cur))
cur      = cur + ffn_res

h_nextn  = RMSNorm(cur, nextn.shared_head_norm ? nextn.shared_head_norm : output_norm)
logits   = (nextn.shared_head_head ? nextn.shared_head_head : output)(h_nextn)
```

这和普通 full attention layer 很像，但有三处不同：

- 输入不是直接 token embedding，而是 `eh_proj(concat(enorm(token), hnorm(h)))`。
- 使用独立的 MTP KV cache。
- head norm/head weight 可由 `nextn.shared_head_*` 覆盖，否则 fallback 到主模型
  `output_norm` / `output.weight`。

### Speculative Driver

llama.cpp `common_speculative_impl_draft_mtp` 的状态：

```text
pending_h[seq]       上一个已确认位置的 h_nextn
verify_h[seq]        最近一次 target verify batch 的 h_nextn rows
verify_h_rows[seq]   verify_h row count
batch.token          MTP 输入 token ids
batch.embd           MTP 输入 h rows
```

target process hook：

1. target 跑完一批 token 后，导出每一行 `h_nextn`。
2. MTP catch-up batch 使用同样 token 和 position。
3. MTP batch 的 hidden rows 向右错一位：
   - row 0 使用上一批遗留的 `pending_h`。
   - row i 使用 target `h_nextn[i - 1]`。
4. MTP context 跑 catch-up decode，把 prompt / accepted prefix 的 MTP KV 补齐。
5. `pending_h` 更新为 target 当前批最后一行 `h_nextn`。

draft hook：

1. 用 `id_last` 和 `pending_h` 跑 MTP graph。
2. 从 draft logits 采样 top-1 或 top-k。
3. 保存 draft token。
4. 如果还没到 `n_max`，把本次 MTP graph 输出的 `h_nextn` 作为下一步 hidden，
   token 换成刚 draft 的 token，position 前进一位，继续跑。
5. 如果 draft token 概率小于 `p_min`，停止 draft。

verify hook：

1. target batch 一次 decode `id_last + draft_tokens`。
2. sampler 从 target logits 逐行采样。
3. 如果第 i 行采样 token 等于 `draft_tokens[i]`，接受第 i 个 draft。
4. 第一次不相等处停止。
5. 如果所有 draft 都被接受，再从最后一行 logits 采样一个普通 target token。

llama.cpp 的 `common_sampler_sample_and_accept_n()` 做的是“匹配 draft 则继续，否则停止”。
greedy 模式下可等价实现为逐行 argmax 对比。

## kraken-infer 实现方案

### 阶段 0：资产与开关

目标：

- 引入 MTP-capable GGUF 到 `models/qwen3.5-0.8b-mtp/`。
- `inspect` 应显示 `MTP/NextN layers: 1`。
- 非 MTP GGUF 自动走现有普通路径。

建议 CLI：

```bash
./build/debug/kraken-infer infer \
  --model models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf \
  --device mps \
  --prompt "hello" \
  --max-new-tokens 64 \
  --mtp \
  --mtp-draft-tokens 3
```

`serve`：

```bash
./build/debug/kraken-infer serve \
  --host 127.0.0.1 \
  --port 18080 \
  --model models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf \
  --model-id qwen3.5-0.8b-mtp \
  --device mps \
  --max-new-tokens 128 \
  --mtp \
  --mtp-draft-tokens 3
```

当前 `serve` 已有 `--no-mtp`，可以保留默认自动启用策略：

```text
enable_mtp = true && model.mtp_num_hidden_layers > 0 && request 没有图片
```

同时新增显式参数更便于测试：

```text
--mtp
--no-mtp
--mtp-draft-tokens N
--mtp-draft-min N
--mtp-draft-p-min P
```

OpenAI gateway 可增加 request fields：

```json
{
  "mtp": true,
  "mtp_draft_tokens": 3,
  "mtp_draft_min": 0,
  "mtp_draft_p_min": 0.0
}
```

### 阶段 1：输出 h_nextn

改造目标：

- 主 prefill 每个 chunk 能得到 all-row `h_nextn`。
- decode/verify batch 能得到 all-row `h_nextn`。
- logits path 从 `h_nextn` 计算，不重复 output norm。

建议拆分 helper：

```cpp
Result<mps::MpsBuffer> run_qwen35_output_norm(
  const mps::MpsContext& context,
  const mps::MpsBuffer& hidden,
  std::size_t tokens,
  std::size_t hidden_size,
  float rms_eps,
  const Qwen35MetalWeight& output_norm);

Result<Qwen35LogitsOutput> run_qwen35_lm_head(
  const mps::MpsContext& context,
  const mps::MpsBuffer& h_nextn_last_or_rows,
  std::size_t rows,
  std::size_t hidden_size,
  const Qwen35MetalWeight& output_weight);
```

现有 `run_qwen35_output_logits()` 可保留为 wrapper，避免一次性改动过大。

注意：

- prefix cache 现在保存的是 pre-output-norm `last_hidden`。MTP 需要的是 post-output-norm
  `h_nextn`。如果 MTP 和 prefix cache 同时启用，cache payload 需要新增 `last_h_nextn`，
  或恢复后重新跑 output norm。
- 第一阶段可规定 MTP 与 prompt prefix cache 不同时启用，降低复杂度。

### 阶段 2：target batch decode

现有 `run_qwen35_decode_token()` 只支持一个 token。MTP verify 需要一次 decode 多个
连续 token：

```cpp
Result<Qwen35MainForwardOutput> run_qwen35_decode_tokens(
  const mps::MpsContext& context,
  const Qwen35MetalWeightStore& weights,
  Qwen35MetalCache& cache,
  const Qwen35WeightMap& map,
  const ModelConfig& model,
  const Qwen35ExecutionPlan& plan,
  const std::array<std::size_t, 4>& mrope_sections,
  std::span<const std::int64_t> token_ids,
  std::size_t start_position);
```

输出：

```cpp
struct Qwen35MainForwardOutput {
  mps::MpsBuffer hidden;   // pre-output-norm rows
  mps::MpsBuffer h_nextn;  // post-output-norm rows
};
```

实现可复用 prefill chunk 的 main layer loop，因为 full attention 和 linear attention
helper 本来支持 `tokens > 1`。

### 阶段 3：MTP cache

MTP block 是 full attention block，没有 Qwen3.5 recurrent R/S state。

新增：

```cpp
struct Qwen35MtpCachePlan {
  std::size_t mtp_layers{0};
  std::size_t capacity_tokens{0};
  std::size_t kv_dim{0};
  std::size_t kv_element_bytes{sizeof(std::uint16_t)};
  std::uint64_t key_bytes{0};
  std::uint64_t value_bytes{0};
};

struct Qwen35MtpCache {
  mps::MpsBuffer key_cache;
  mps::MpsBuffer value_cache;
  bool kv_cache_f16{true};
};
```

第一阶段只支持：

```text
map.mtp_layers == 1
config.mtp_num_hidden_layers == 1
sequence_slots == 1
```

### 阶段 4：MTP block forward

新增 helper：

```cpp
Result<Qwen35MtpForwardOutput> run_qwen35_mtp_tokens(
  const mps::MpsContext& context,
  const Qwen35MetalWeightStore& weights,
  Qwen35MtpCache& mtp_cache,
  const Qwen35LayerBindings& mtp_layer,
  const Qwen35MetalWeight& token_embedding,
  const Qwen35MetalWeight& output_norm,
  const Qwen35MetalWeight& output_weight,
  const std::array<std::size_t, 4>& mrope_sections,
  std::span<const std::int64_t> token_ids,
  const mps::MpsBuffer& h_rows,
  std::size_t start_position,
  const Qwen35ExecutionPlan& plan,
  float rope_theta,
  float rms_eps);
```

输出：

```cpp
struct Qwen35MtpForwardOutput {
  mps::MpsBuffer h_nextn;
  mps::MpsBuffer logits;
  mps::MpsBuffer argmax_output;
  std::size_t logits_values{0};
};
```

实现步骤按 llama.cpp graph_mtp 对齐：

1. token embedding：优先 `nextn.embed_tokens`，缺失时用主 `token_embd.weight`。
2. `h_norm = RMSNorm(h_rows, nextn.hnorm)`。
3. `e_norm = RMSNorm(tok_embd, nextn.enorm)`。
4. concat 成 `[2 * hidden, tokens]`。
5. `eh_proj` quant matmul 得到 `[hidden, tokens]`。
6. full attention block：QG split、Q/K norm、MRoPE、causal attention、gate sigmoid、
   attn output projection。
7. residual + post-attn norm + dense FFN。
8. head norm：优先 `nextn.shared_head_norm`，fallback `output_norm`。
9. head weight：优先 `nextn.shared_head_head`，fallback `output.weight`。

可先把 full attention layer helper 抽成一个更通用的内部函数，接受：

```cpp
Qwen35FullAttentionBindings attention;
Qwen35TensorBinding attn_norm;
Qwen35TensorBinding attn_post_norm;
Qwen35TensorBinding ffn_up/gate/down;
Qwen35MetalCache-like cache;
```

这样主 full-attention layer 和 MTP block 共用实现。

### 阶段 5：speculative scheduler

新增内部状态：

```cpp
struct Qwen35MtpState {
  bool enabled{false};
  std::size_t draft_max{3};
  std::size_t draft_min{0};
  float draft_p_min{0.0F};

  Qwen35MtpCache cache;
  std::vector<float> pending_h;  // hidden_size, host fallback for first version

  std::size_t draft_tokens_total{0};
  std::size_t draft_tokens_accepted{0};
  std::size_t verify_steps{0};
};
```

第一阶段可以让 `pending_h` 保存在 host，MTP draft 每步再 copy 到 device。这样简单但会有
host/device copy；后续再常驻 device。

prefill catch-up：

```text
for each prompt chunk:
  target main forward -> chunk_h_nextn
  mtp_process_tokens(
    token_ids = chunk tokens,
    h_rows = shift_right(previous_pending_h, chunk_h_nextn[0..n-2]),
    positions = chunk_start..chunk_start+n-1,
    logits = false
  )
  pending_h = chunk_h_nextn[last]
```

draft：

```text
draft_tokens = []
token = last_sampled_target_token
h = pending_h
pos = next_position

while draft_tokens.size < draft_max:
  mtp_out = mtp_forward(token, h, pos, logits=true)
  draft = argmax_or_sample_topk10(mtp_out.logits)
  if prob(draft) < p_min: break
  draft_tokens.push_back(draft)
  token = draft
  h = mtp_out.h_nextn
  pos += 1
```

verify：

```text
target_verify = run target on [last_sampled_target_token] + draft_tokens
for i in 0..draft_tokens.size-1:
  target_next = sample/argmax(target_verify.logits_row[i])
  if target_next != draft_tokens[i]:
    stop
  accepted += 1
pending_h = target_verify.h_nextn_row[accepted]
```

生成输出：

- 每一轮先从当前 target logits 采样 `last_sampled_target_token` 并输出。
- 然后 MTP draft 若干 token。
- target verify 后输出被接受的 draft tokens。
- 如果 draft 全接受，下一轮可以直接使用 verify 最后一行 logits；否则重新以 target
  实际 logits 继续。

### 回滚策略

Qwen3.5 有 recurrent linear attention。验证 draft 时如果部分拒绝，target cache 已经包含
未接受 token 的 KV 和 recurrent R/S state。不能只把 position 往回拨。

正确性优先的第一阶段：

1. 在 live cache 外分配 `verify_cache`。
2. 每轮 verify 前把 live cache copy 到 verify cache。
3. 在 verify cache 上 decode `last_sampled + draft_tokens`。
4. 得到 accepted count 后，在 live cache 上 replay `last_sampled + accepted_drafts`。
5. live cache 只包含已确认 token。

代价：

- verify 多一次 cache copy。
- accepted tokens 需要 replay。
- 性能不是最终形态，但 correctness 简单。

优化阶段：

- 增加 `Qwen35CacheSnapshot`，只保存当前位置之后会被污染的部分。
- recurrent R/S 保存完整 current slot state。
- full attention KV 对 `[position, end)` 可直接覆盖，不必清零。
- 如果 all draft accepted，可以把 verify cache swap 成 live cache，避免 replay。

### 与 VL/mmproj 的关系

当前 kraken 的 VL 路径通过 llama.cpp `llama-mtmd-cli` 桥接，不走 native text runtime。
MTP 第一阶段应在以下条件下禁用：

```text
chat_messages_have_image_content(request.messages) == true
request.mmproj_path not empty
```

原因：

- llama.cpp MTP README 说明 `--mmproj` 暂不支持 MTP。
- 多模态 prompt 中 image embedding 不是普通 token id；MTP process hook 当前只处理
  token batch。
- kraken 当前 VL bridge 不是 native cache path，无法共享 `h_nextn`。

后续 native VL graph 完成后，再重新设计 image chunk 的 `h_nextn` / MTP process key。

### 与 prefix cache 的关系

prefix cache 第一阶段建议默认与 MTP 互斥，或者只在 cache miss 时启用 MTP。

如果要同时启用，需要扩展 cache payload：

```text
last_hidden       pre-output-norm hidden，现有 logits path 用
last_h_nextn      post-output-norm hidden，MTP pending_h 用
mtp_key_cache     MTP block KV cache
mtp_value_cache   MTP block KV cache
```

否则恢复 prompt cache 后 target cache 有主干 KV/R/S，但 MTP cache 没有 catch-up，
draft context 不可用。

## 测试计划

### 静态/metadata 测试

新增 smoke test：

- 构造 tiny GGUF，`block_count = 25`，`nextn_predict_layers = 1`。
- 验证：
  - `main_layer_count == 24`
  - `mtp_num_hidden_layers == 1`
  - `weight_map.mtp_layers == 1`
  - `Qwen35LayerKind::mtp` 在最后一层
  - optional `nextn.embed_tokens` / `shared_head_*` 缺失时仍可通过

### MTP forward shape 测试

在没有真实模型权重的 smoke test 中，用 tiny/synthetic tensors 验证：

- `run_qwen35_mtp_tokens(tokens=1)` 输出：
  - `h_nextn`: `[hidden]`
  - `logits`: `[vocab]`
- `run_qwen35_mtp_tokens(tokens=N)` 能填 MTP KV cache。

真实模型测试脚本：

```bash
python3 scripts/test_qwen35_mtp_gateway.py \
  --model models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf \
  --max-tokens 64 \
  --draft-tokens 3
```

脚本应断言：

- response 非空。
- headers 或 JSON debug 字段里有 MTP stats：
  - `draft_tokens_total > 0`
  - `verify_steps > 0`
  - `accepted_tokens <= draft_tokens_total`
- `--no-mtp` 同 prompt 能正常返回。

### llama.cpp 对齐测试

llama.cpp reference：

```bash
/Users/bill/code/llama.cpp/build/bin/llama-server \
  -m models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf \
  -ngl 99 -c 8192 -fa on -np 1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3
```

本机 reference smoke 已通过：

```text
common_speculative_impl_draft_mtp: adding speculative implementation 'draft-mtp'
timings.draft_n: 11
timings.draft_n_accepted: 11
draft acceptance: 1.00000
```

kraken：

```bash
./build/debug/kraken-infer serve \
  --model models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf \
  --model-id qwen3.5-0.8b-mtp \
  --device mps \
  --max-new-tokens 64 \
  --mtp \
  --mtp-draft-tokens 3
```

本机 kraken smoke 已通过：

```text
./build/debug/kraken-infer infer --model models/qwen3.5-0.8b-mtp ...
mtp: enabled, layers=1, draft_tokens=3, drafted=8, accepted=5, verify_steps=6

python3 scripts/test_qwen35_mtp_gateway.py --max-tokens 8
{"accepted_tokens": 4, "drafted_tokens": 9, "mtp_enabled": 1, "mtp_layers": 1, "verify_steps": 6}
```

### 2026-07-10 性能测量

同一 prompt，Apple Metal path，`max-new-tokens=64`。

旧 full-logits draft head 结果：

| 模式 | wall time | 说明 |
| --- | ---: | --- |
| `--no-mtp` | 10.98s | baseline greedy decode |
| `--mtp --mtp-draft-tokens 3 --mtp-p-min 0` | 15.80s | full vocab logits draft head |
| `--mtp --mtp-draft-tokens 3 --mtp-p-min 0.20` | 12.80s | device argmax/probability, still materialized logits |

fused top-1 / token-only argmax / adaptive budget 结果：

| 模式 | wall time | 说明 |
| --- | ---: | --- |
| `--no-mtp` | 15.17s | baseline greedy decode；Debug timing 有负载波动 |
| `--mtp --mtp-draft-tokens 3 --mtp-p-min 0` | 14.38s | drafted=62 accepted=28 verify_steps=50 adaptive_budget=1 adaptive_changes=4 |
| `--mtp --mtp-draft-tokens 3 --mtp-p-min 0.20` | 11.43s | drafted=45 accepted=28 verify_steps=36 confidence_stops=26 adaptive_budget=3 adaptive_changes=2 |

### 2026-07-11 MTP 参数 sweep

同一 prompt、Debug build、`max-new-tokens=64`，使用
`scripts/sweep_qwen35_mtp.py --draft-tokens 1,2,3 --p-min 0,0.1,0.2,0.3`
重新扫描：

| 模式 | wall time | 说明 |
| --- | ---: | --- |
| `--no-mtp` | 11.05s | baseline greedy decode |
| `--mtp --mtp-draft-tokens 1 --mtp-p-min 0.30` | 10.44s | drafted=23 accepted=21 verify_steps=23 acceptance=91.3% |
| `--mtp --mtp-draft-tokens 2 --mtp-p-min 0.30` | 10.67s | drafted=30 accepted=27 verify_steps=30 acceptance=90.0% |
| `--mtp --mtp-draft-tokens 3 --mtp-p-min 0.30` | 10.92s | drafted=28 accepted=26 verify_steps=28 acceptance=92.9% |
| `--mtp --mtp-draft-tokens 3 --mtp-p-min 0.20` | 11.56s | drafted=45 accepted=28 verify_steps=36 acceptance=62.2% |

结论：`p_min=0` / `0.10` 会让低置信 draft 过多，MTP block 额外计算抵消
verify 步数收益；`p_min=0.30` 在这轮 sweep 中最稳定。CLI、OpenAI gateway
和脚本默认 `mtp_p_min` 因此改为 `0.30`。默认仍保留 `mtp_draft_tokens=3`，
用于保持多 token draft 能力；追求单 prompt 最低延迟时可以显式设为
`--mtp-draft-tokens 1`。

已实现 batch target verify 和 device-side draft chain：

- target verify 会一次 decode `[sampled_token + draft_tokens]`。
- full-accepted draft span 直接保留 verify 写入的 KV/recurrent state。
- partial reject 会恢复 recurrent snapshot，再 batch commit accepted prefix。
- MTP draft chain 会在单个 Metal command buffer 内把上一步 argmax 作为下一步
  embedding row id，避免每个 draft token 都做 CPU readback 再启动下一图。
- Q4_K/Q5_K/Q6_K fused top-1 head 让 draft 不再 materialize 完整 vocab logits。
- Q4_K/Q5_K/Q6_K token-only argmax head 让 `p_min=0` 不再计算 softmax denominator。
- adaptive budget 会在低 acceptance 窗口把 draft budget 从请求值逐步降下来，
  减少低收益深层 draft 的 MTP block 计算。

现在推荐默认 `p_min=0.30`；`p_min=0` 通过 adaptive budget 减少了无效
draft，但主要瓶颈仍是低置信 draft 带来的 MTP block 额外计算。后续要继续提速，
优先方向是：

1. 优化 MTP block 内 `nextn.eh_proj` 和 full-attention block。
2. 提高 target verify 的小 batch kernel 效率。
3. 做运行时自适应：acceptance/cost 不划算时自动关闭或降低 draft budget。

对齐策略：

- 第一阶段只对齐 greedy deterministic output。
- 使用相同 prompt；kraken 第一版不要传 `temperature/top_k/top_p/seed`，保持
  request sampling 字段缺省以启用 greedy MTP。
- 验证最终文本一致或 token 序列一致。
- 记录 draft acceptance rate，不要求完全等同 llama.cpp 的 backend sampler。
- 新增 `scripts/compare_qwen35_llamacpp.py` 用于同 prompt / 同 GGUF 记录
  kraken no-MTP、kraken MTP、llama.cpp no-MTP 的 wall time、文本尾部和 stats；
  可用 `--include-llama-mtp` 额外尝试 llama.cpp `--spec-type draft-mtp`。
- 新增 `scripts/sweep_qwen35_mtp.py` 用于批量扫描 `mtp_draft_tokens` 和
  `mtp_p_min`，输出 aggregate table、JSON 和可选 CSV，帮助选择默认参数。

示例：

```bash
python3 scripts/compare_qwen35_llamacpp.py \
  --max-tokens 64 \
  --p-min 0.30 \
  --json-out build/qwen35-compare.json

python3 scripts/sweep_qwen35_mtp.py \
  --max-tokens 64 \
  --draft-tokens 1,2,3 \
  --p-min 0,0.1,0.2,0.3 \
  --json-out build/qwen35-mtp-sweep.json \
  --csv-out build/qwen35-mtp-sweep.csv
```

## 验收标准

第一阶段：

- `inspect` 能识别 MTP-capable Qwen3.5 0.8B GGUF。
- 非 MTP 版模型继续走现有文本路径，不报错。
- MTP 版模型在 `--mtp-draft-tokens 1..3` 下能完成 greedy generation。
- 部分 draft rejection 时输出仍正确，不污染后续 generation。
- `--no-mtp` 可强制回退普通 decode。
- gateway 响应暴露 MTP stats。
- `ctest --preset debug` 通过。
- 新增 `scripts/test_qwen35_mtp_gateway.py` 能在真实 MTP GGUF 存在时跑通。
- 新增 `scripts/compare_qwen35_llamacpp.py` 能生成 kraken/llama.cpp 对照 JSON。
- 新增 `scripts/sweep_qwen35_mtp.py` 能生成 MTP 参数 sweep JSON/CSV。

第二阶段：

- 支持 sampling 下的 `top_k/top_p/temperature` accept 逻辑。
- 支持 all-accepted fast path，把 verify cache swap 成 live cache。
- 支持 prefix cache + MTP cache payload。
- 支持分离的 `mtp-*.gguf` draft file。

暂不做：

- MTP + VL/mmproj。
- 多 parallel slots。
- Qwen3.5 MoE MTP。
- 多 MTP head chain，例如 Step3.5 的 `nextn_layer_offset` 路径。
