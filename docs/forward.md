# Qwen3 0.6B Forward Inference

本文根据当前仓库代码说明 `Qwen/Qwen3-0.6B` 的前向推理过程。重点是 CPU
reference path：它是最容易读清楚的纯计算实现，也是 MPS backend 对齐正确性的基准。

核心代码：

- [`src/runtime/cpu_inference.cpp`](../src/runtime/cpu_inference.cpp)：公共 CPU 推理入口。
- [`src/runtime/cpu/qwen_cpu_model.cpp`](../src/runtime/cpu/qwen_cpu_model.cpp)：Qwen3 权重绑定、forward、sampling。
- [`src/runtime/cpu/kv_cache.hpp`](../src/runtime/cpu/kv_cache.hpp) 和
  [`src/runtime/cpu/kv_cache.cpp`](../src/runtime/cpu/kv_cache.cpp)：CPU KV cache。
- [`src/runtime/cpu/tokenizer.cpp`](../src/runtime/cpu/tokenizer.cpp)：Qwen chat prompt 和 BPE tokenizer。
- [`src/model/model_config.cpp`](../src/model/model_config.cpp)：Qwen3 config 解析和结构校验。

当前 CPU path 的定位：

- batch size 固定为 `1`。
- 按 token 顺序执行 forward，不做 sequence-level prefill GEMM。
- 权重来自 `model.safetensors`，dtype 是 BF16；计算时转成 F32。
- activation、attention score、KV cache、logits 都是 F32。
- attention 是朴素循环实现，用于正确性 reference，不追求吞吐。
- causal 约束通过只遍历 `0..position` 的 KV cache 实现，没有显式 mask 矩阵。

## Qwen3 0.6B 结构参数

模型配置来自 [`models/qwen3-0.6b/config.json`](../models/qwen3-0.6b/config.json)：

```text
architecture            = Qwen3ForCausalLM
model_type              = qwen3
num_hidden_layers       = 28
hidden_size             = 1024
num_attention_heads     = 16
num_key_value_heads     = 8
head_dim                = 128
intermediate_size       = 3072
vocab_size              = 151936
max_position_embeddings = 40960
max_window_layers       = 28
use_sliding_window      = false
attention_bias          = false
attention_dropout       = 0.0
hidden_act              = silu
rms_norm_eps            = 1e-6
rope_theta              = 1000000
torch_dtype             = bfloat16
tie_word_embeddings     = true
use_cache               = true
```

本文使用这些符号：

```text
L      = num_hidden_layers       = 28
H      = hidden_size             = 1024
Nq     = num_attention_heads     = 16
Nkv    = num_key_value_heads     = 8
D      = head_dim                = 128
A      = Nq * D                  = 2048
KVD    = Nkv * D                 = 1024
I      = intermediate_size       = 3072
V      = vocab_size              = 151936
group  = Nq / Nkv                = 2
```

`A` 是 Q/O attention 投影维度，`KVD` 是 K/V 投影维度。因为 Qwen3 0.6B 使用
GQA（grouped-query attention），`Nq=16` 但 `Nkv=8`，所以每 2 个 query head
共享 1 个 KV head。

## Transformer 结构简介

Qwen3 0.6B 在本项目里按 dense decoder-only Transformer 执行：

```text
messages / prompt
  -> Qwen chat template + tokenizer
  -> token ids
  -> token embedding
  -> 28 x decoder layer
       -> input RMSNorm
       -> self attention
          -> Q/K/V projection
          -> Q/K per-head RMSNorm
          -> RoPE
          -> KV cache store
          -> GQA causal attention over KV cache
          -> O projection
          -> residual add
       -> post-attention RMSNorm
       -> gated MLP
          -> gate projection
          -> up projection
          -> SiLU(gate) * up
          -> down projection
          -> residual add
  -> final RMSNorm
  -> lm_head logits
  -> greedy or sampling
  -> next token
```

这是 decoder-only 语言模型，所以每一步只预测下一个 token。训练时可以并行看完整
prefix，推理时 decode 是自回归的：第 `t` 个输出 token 依赖 prompt 和前面已经生成的
`0..t-1` token。

## 权重布局

`CpuQwenModel::load()` 会读取 config、tokenizer 和 safetensors，然后
`bind_weights()` 将 safetensors 里的名字绑定到内部 `TensorView`。

全局权重：

```text
model.embed_tokens.weight    [V, H]
model.norm.weight            [H]
lm_head.weight               [V, H]
```

每层权重：

```text
model.layers.{i}.input_layernorm.weight          [H]
model.layers.{i}.self_attn.q_proj.weight         [A, H]
model.layers.{i}.self_attn.q_norm.weight         [D]
model.layers.{i}.self_attn.k_proj.weight         [KVD, H]
model.layers.{i}.self_attn.k_norm.weight         [D]
model.layers.{i}.self_attn.v_proj.weight         [KVD, H]
model.layers.{i}.self_attn.o_proj.weight         [H, A]
model.layers.{i}.post_attention_layernorm.weight [H]
model.layers.{i}.mlp.gate_proj.weight            [I, H]
model.layers.{i}.mlp.up_proj.weight              [I, H]
model.layers.{i}.mlp.down_proj.weight            [H, I]
```

所有 projection 权重按行主序当作矩阵使用。比如 `q_proj.weight [A, H]` 表示每一行
产生 Q 向量中的一个输出维度。

## 推理入口

公共入口是 `generate_cpu()`，它把外部 `CpuGenerationRequest` 转给
`cpu::generate_text()`。真正的模型对象由 `cached_model(model_dir)` 缓存，避免同一个
模型目录重复加载权重。

生成主循环在 `CpuQwenModel::generate()`：

```text
prompt_tokens = tokenizer.encode_chat_messages(messages, enable_thinking)
reset_kv_cache(prompt_tokens.size + max_new_tokens + 1)

for token in prompt_tokens:
  hidden = forward_token(token, position)
  position += 1

for step in 0..max_new_tokens-1:
  logits = compute_logits(hidden)
  next_token = select_next_token(logits)
  if eos(next_token): break
  generated.push(next_token)
  hidden = forward_token(next_token, position)
  position += 1
```

注意两个细节：

- prompt 阶段也会逐 token forward，并把 prompt 的每个 K/V 写进 cache。
- 第一轮 decode 的 logits 来自最后一个 prompt token 的 hidden state；采样出
  `next_token` 后，才会对这个新 token 再跑 `forward_token()`，为下一轮 logits 做准备。

## Prompt 和 Thinking

输入可以是简单 `prompt`，也可以是 chat messages。公共入口会把纯 prompt 包成一条
`user` message。

tokenizer 使用 Qwen chat 模板，把消息转换成类似：

```text
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
```

`enable_thinking` 会影响 Qwen chat prompt formatting，让模型决定是否输出
thinking 相关内容。这个开关发生在 tokenizer/prompt 层，不改变后续 Transformer
forward 算法。

## 单 Token Forward

`forward_token(token, position)` 是 CPU 纯计算主线。输入是一个 token id 和它的绝对
位置，输出是最终 RMSNorm 后的 hidden，shape 是 `[H]`。

### 1. Embedding Lookup

embedding 是查表：

```text
hidden[i] = BF16_TO_F32(model.embed_tokens.weight[token, i])
```

输出：

```text
hidden: [H] = [1024]
```

这里没有矩阵乘，只有按 token id 取 embedding 矩阵的一行。

### 2. 28 个 Decoder Layer

每层调用 `apply_layer(layer, layer_index, position, hidden)`，并原地更新 `hidden`。

一层的计算图：

```text
x0 = hidden

a = RMSNorm(x0, input_layernorm.weight)
q = q_proj * a
k = k_proj * a
v = v_proj * a
q = QKNorm(q, q_norm.weight)
k = QKNorm(k, k_norm.weight)
q = RoPE(q, position)
k = RoPE(k, position)
KVCache[layer, position] = (k, v)
attn = GQA_Attention(q, KVCache[layer, 0..position])
attn_projected = o_proj * attn
x1 = x0 + attn_projected

m = RMSNorm(x1, post_attention_layernorm.weight)
gate = gate_proj * m
up = up_proj * m
mlp_hidden = SiLU(gate) * up
mlp_out = down_proj * mlp_hidden
hidden = x1 + mlp_out
```

28 层结构相同，差异只在每层自己的权重。

### 3. Final RMSNorm

全部 layer 执行完后：

```text
normed = RMSNorm(hidden, model.norm.weight)
```

`forward_token()` 返回 `normed`，而不是未归一化的 `hidden`。后续 `lm_head` 使用这个
`normed` 计算 logits。

## Decoder Layer 细节

### Input RMSNorm

RMSNorm 用于 attention 前的 pre-norm：

```text
mean_square = mean(x[i]^2)
scale = 1 / sqrt(mean_square + rms_norm_eps)
out[i] = x[i] * scale * weight[i]
```

实现中 `mean_square` 用 `double` 累加，输出是 `float`。`input_layernorm.weight` 是
BF16，使用时逐元素转 F32。

### Q/K/V Projection

Q/K/V 都是 BF16 weight + F32 activation 的矩阵向量乘：

```text
q = q_proj * normed    q_proj: [2048, 1024] -> q: [2048] = [16, 128]
k = k_proj * normed    k_proj: [1024, 1024] -> k: [1024] = [8, 128]
v = v_proj * normed    v_proj: [1024, 1024] -> v: [1024] = [8, 128]
```

这里体现了 GQA：Q 有 16 个 head，K/V 只有 8 个 head。

### Q/K Norm

Qwen3 在 Q/K projection 后还有 per-head RMSNorm：

```text
for each head:
  segment = values[head * D : (head + 1) * D]
  segment = RMSNorm(segment, q_norm.weight or k_norm.weight)
```

`q_norm.weight` 和 `k_norm.weight` shape 都是 `[128]`，在每个 head 上重复使用。
V 不做这个 norm。

### RoPE

RoPE 把位置信息注入 Q/K。当前实现使用 Qwen/LLaMA 常见的 half-rotation：

```text
half = D / 2
freq_i = 1 / rope_theta ^ ((2 * i) / D)
angle = position * freq_i

x0 = values[i]
x1 = values[half + i]

values[i]        = x0 * cos(angle) - x1 * sin(angle)
values[half + i] = x1 * cos(angle) + x0 * sin(angle)
```

RoPE 只作用在 Q/K，不作用在 V。`rope_theta=1000000`，`max_position_embeddings=40960`。

### KV Cache Store

RoPE 后的 K 和原始 V 会写入 cache：

```text
keys[layer, position, :] = k
values[layer, position, :] = v
```

注意 cache 存的是已经做过 Q/K norm 和 RoPE 的 K；V 是 `v_proj` 输出，不做 RoPE。

### GQA Causal Attention

attention 对每个 query head 单独计算。Q head 到 KV head 的映射：

```text
group = Nq / Nkv = 2
kv_head = q_head / group
```

所以：

```text
q_head 0,1   -> kv_head 0
q_head 2,3   -> kv_head 1
...
q_head 14,15 -> kv_head 7
```

score 计算：

```text
score[t] = dot(q[q_head], key[layer, t, kv_head]) / sqrt(D)
t = 0..position
```

只遍历 `0..position`，因此未来 token 不会进入 attention，不需要额外构造 causal mask。

softmax 使用 max trick：

```text
max_score = max(score[t])
weight[t] = exp(score[t] - max_score)
prob[t] = weight[t] / sum(weight)
```

输出：

```text
attn[q_head, d] = sum_t prob[t] * value[layer, t, kv_head, d]
```

attention 输出 shape 是：

```text
attn: [A] = [16, 128] = [2048]
```

### Attention Output Projection 和残差

attention 输出通过 `o_proj` 回到 hidden size：

```text
projected = o_proj * attn    o_proj: [1024, 2048]
hidden = hidden + projected
```

这是第一个 residual add。因为 `q_proj` 的输出维度是 2048，大于 `hidden_size=1024`，
所以必须通过 `o_proj` 映射回 `[H]` 后才能加回残差。

### Post-Attention RMSNorm

MLP 前再做一次 RMSNorm：

```text
mlp_input = RMSNorm(hidden, post_attention_layernorm.weight)
```

Qwen3 这里仍然是 pre-norm 结构：attention 和 MLP 两个子层都先 norm，再做主计算，
最后各自 residual。

### Gated MLP

MLP 是 gated FFN：

```text
gate = gate_proj * mlp_input    [3072]
up   = up_proj   * mlp_input    [3072]
act  = SiLU(gate) * up          [3072]
down = down_proj * act          [1024]
hidden = hidden + down
```

SiLU：

```text
SiLU(x) = x / (1 + exp(-x))
```

这类结构也常被称为 SwiGLU/GLU-style MLP。`gate_proj` 提供门控，`up_proj` 提供被门控的
值，二者逐元素相乘后再通过 `down_proj` 回到 hidden size。

## 算子实现说明

### BF16 到 F32

safetensors 权重是 BF16。CPU 算子不直接用 BF16 计算，而是在读取每个权重元素时调用
`bf16_to_float()` 转成 F32，然后与 F32 activation 相乘。

这使实现更直接，也避免手写 BF16 累加误差处理；代价是 CPU path 速度较慢。

### MatVec

`matvec(weight, input, output)` 是朴素行主序矩阵向量乘：

```text
for row in rows:
  sum = 0
  for col in cols:
    sum += BF16_TO_F32(weight[row, col]) * input[col]
  output[row] = sum
```

当前没有 BLAS、SIMD、线程池，也没有把多个 projection fusion 到一个矩阵乘里。

使用这个算子的地方：

- Q/K/V projection
- attention output projection
- MLP gate/up/down projection

`lm_head` 也做同类计算，但实现单独写在 `compute_logits()` 里，因为它按 vocab 行遍历。

### RMSNorm

`rms_norm()` 输入、输出都是 `std::vector<float>`，权重是 BF16。它用于：

- 每层 attention 前的 `input_layernorm`
- 每层 MLP 前的 `post_attention_layernorm`
- 全部 layer 后的 `model.norm`

`qk_norm()` 是 per-head 版本，用于 Q/K。

### RoPE

`apply_rope()` 对每个 head 循环，再对 head 内前后半维做成对旋转。当前实现每次根据
`position` 直接计算 `pow/cos/sin`，没有预计算 cos/sin table。

这更容易验证正确性，但不是最高性能实现。

### Attention

`attention()` 是 CPU path 中最关键的动态算子：

- 外层遍历 `Nq=16` 个 query head。
- 每个 head 根据 `group=2` 找到对应 KV head。
- 对 `0..position` 所有历史 token 计算 score。
- softmax 后，对同一范围的 value 做加权求和。

单 token、单层 attention 复杂度大约是：

```text
O(Nq * (position + 1) * D)
```

decode 越往后，`position` 越大，attention 读取的历史 KV 越多。

### Residual Add

residual add 是简单逐元素加：

```text
hidden[i] += projected[i]
hidden[i] += down[i]
```

当前没有单独封装成 CPU helper。

### SiLU Gate

MLP 中间激活原地写回 `gate`：

```text
gate[i] = silu(gate[i]) * up[i]
```

之后 `gate` 这个 buffer 就代表 MLP 中间结果 `act`。

### Logits

`compute_logits(hidden, false)` 遍历 vocab：

```text
for token_id in 0..V-1:
  logits[token_id] = dot(lm_head[token_id, :], hidden)
```

shape：

```text
hidden:  [1024]
lm_head: [151936, 1024]
logits:  [151936]
```

这是每个 decode step 的大头之一，复杂度是 `O(V * H)`。

## KV Cache 详细说明

### 为什么需要 KV Cache

自回归生成时，每次只新增一个 token。没有 KV cache 的情况下，为了计算下一个 token，
模型需要反复对整个 prefix 重新计算所有历史 token 的 K/V。

KV cache 的思想是：

- 历史 token 的 K/V 一旦算出，就不会再变。
- 新 step 只 forward 新 token。
- attention 时读取历史 K/V cache，加上当前 token 新写入的 K/V。

因此 decode 阶段避免重复计算历史 token 的 Q/K/V projection、RoPE 和 MLP，只保留
attention 对历史 K/V 的读取。

### Cache 逻辑形状

CPU KV cache 分成两块连续 F32 vector：

```text
keys_:   [L, capacity_tokens, KVD]
values_: [L, capacity_tokens, KVD]
```

对于 Qwen3 0.6B：

```text
L   = 28
KVD = Nkv * D = 8 * 128 = 1024
```

也可以按 head 看成：

```text
keys_:   [28, capacity_tokens, 8, 128]
values_: [28, capacity_tokens, 8, 128]
```

### Offset 计算

`KvCache::offset(layer, position, kv_head)`：

```text
offset = (layer * capacity_tokens + position) * KVD + kv_head * D
```

`key_ptr(layer, position, kv_head)` 和 `value_ptr(...)` 返回的是指向某个 KV head 的
连续 `[D]` float 指针。attention 内部会用这个指针做 dot 或 weighted sum。

### Reset 生命周期

每次 generation request 开始时：

```text
capacity = prompt_tokens.size + max_new_tokens + 1
kv_cache.reset(L, capacity, Nkv, D)
```

`+1` 是保守容量，避免 prompt 后第一轮 decode/边界情况下越界。`reset()` 会：

- 校验 `layers/capacity/kv_heads/head_dim` 都大于 0。
- 检查乘法是否溢出。
- 分配并清零 `keys_` 和 `values_`。
- 把 `used_tokens_` 置 0。

### Store 生命周期

每个 token 的每层执行完 K/V projection、Q/K norm 和 RoPE 后：

```text
kv_cache.store(layer, position, k, v)
```

`store()` 会：

- 校验 layer、position、kv_head 范围。
- 校验 `k.size()` 和 `v.size()` 都等于 `KVD`。
- 把整段 K/V copy 到对应的连续位置。
- 更新 `used_tokens_ = max(used_tokens_, position + 1)`。

### Read 生命周期

attention 计算当前 `position` 时只读：

```text
t = 0..position
kv_head = q_head / group
key_ptr(layer, t, kv_head)
value_ptr(layer, t, kv_head)
```

这保证：

- 当前 token 可以看见自己和过去 token。
- 当前 token 看不见未来 token。
- 不需要额外 causal mask。

### GQA 对 Cache 的影响

如果是普通 multi-head attention，K/V head 数通常等于 Q head 数，即 `Nkv=Nq=16`。
Qwen3 0.6B 使用 GQA，`Nkv=8`，所以 KV cache 的 KVD 是：

```text
KVD = 8 * 128 = 1024
```

如果没有 GQA，KVD 会是：

```text
16 * 128 = 2048
```

因此当前 KV cache 内存和 K/V projection 计算量大约是 full MHA 的一半。代价是每两个
Q head 共享一个 KV head。

### Cache 内存估算

每个 token、每层、每个 K 或 V 的 float 数：

```text
KVD = 1024 floats
```

每个 token 的所有层 K cache：

```text
L * KVD * sizeof(float)
= 28 * 1024 * 4
= 114688 bytes
≈ 112 KiB
```

K + V 合计：

```text
2 * 114688 = 229376 bytes ≈ 224 KiB / token
```

常见容量估算：

```text
128 tokens   -> about 28 MiB
512 tokens   -> about 112 MiB
2048 tokens  -> about 448 MiB
40960 tokens -> about 8.75 GiB
```

实际分配不是按 `max_position_embeddings=40960` 固定分配，而是按本次请求的
`prompt_tokens + max_new_tokens + 1` 分配。

### KV Cache Stats

`KvCache::stats()` 返回：

```text
available
layers
kv_heads
head_dim
kv_dim
capacity_tokens
used_tokens
key_bytes
value_bytes
total_bytes
```

公共 API 会转换成 `CpuKvCacheReport`。CLI 的 `--kv-cache-stats` 可以输出这些信息。

### KV Cache Verification

`--verify-kv-cache` 会额外跑 full-prefix recompute：

```text
for each decode step:
  reset_kv_cache(tokens.size + 1)
  for position in full current tokens:
    hidden = forward_token(tokens[position], position)
  logits = compute_logits(hidden)
  next_token = select_next_token(logits)
```

然后把 full-prefix recompute 的 generated token 序列和 cached decode 的 generated token
序列逐项比较。

如果不一致：

```text
KV cache verification failed: cached decode differs from full-prefix recompute
```

这个验证只比较 token 选择结果，不逐元素比较所有 hidden/attention 中间张量。需要更细
的排查时，可以结合 debug dump。

### 当前 KV Cache 限制

CPU KV cache 当前是 correctness-first 实现：

- F32 cache，内存占用较高。
- 连续 `std::vector<float>`，没有 paged KV cache。
- 每次请求重新 reset，不支持跨请求 prompt cache。
- 没有 KV quantization。
- attention 仍然按历史长度线性扫描。

这些限制是后续性能优化方向，不影响当前 reference path 的清晰性。

## Sampling 和停止条件

`select_next_token()` 支持两条路径：

- `do_sample=false`：greedy，取最大 logit。
- `do_sample=true`：temperature、top-k、top-p sampling。

sampling 流程：

```text
1. 把所有 vocab logit 转成候选列表。
2. 按 logit 降序排序。
3. 如果 top_k > 0，保留前 top_k。
4. 用 temperature 计算 exp((logit - max_logit) / temperature)。
5. 如果 top_p < 1.0，按累计概率截断。
6. 用随机数在剩余权重中采样。
```

EOS 判定覆盖：

- `config.eos_token_id`
- internal `kEndOfText`
- `generation_config.eos_token_ids`

命中 EOS 后停止生成，并且不把 EOS 放入 `generated`。

## Debug Dump

如果设置 `debug_dump_dir`，CPU forward 会导出主要中间张量。命名包括：

```text
prompt_tokens
position.{p}.embedding
position.{p}.layer.{l}.input_norm
position.{p}.layer.{l}.q_proj
position.{p}.layer.{l}.k_proj
position.{p}.layer.{l}.v_proj
position.{p}.layer.{l}.q_norm_rope
position.{p}.layer.{l}.k_norm_rope
position.{p}.layer.{l}.attention_out
position.{p}.layer.{l}.attention_projected
position.{p}.layer.{l}.attention_residual
position.{p}.layer.{l}.post_attention_norm
position.{p}.layer.{l}.mlp_gate_proj
position.{p}.layer.{l}.mlp_up_proj
position.{p}.layer.{l}.mlp_gate_silu_mul
position.{p}.layer.{l}.mlp_down_proj
position.{p}.layer.{l}.layer_output
position.{p}.final_norm
position.{p}.logits
generated_tokens
```

这些 dump 点覆盖了 embedding、norm、projection、RoPE、attention、MLP、logits 等关键
边界，便于和 MPS 或外部实现逐算子对齐。

## 当前实现边界

CPU path 目前刻意保持简单：

- 单 batch、单序列。
- prompt prefill 也是逐 token forward，没有 batched GEMM。
- 权重 BF16，计算 F32。
- RoPE 的 sin/cos 没有预计算。
- attention 是朴素 `O(Nq * position * D)`。
- logits 是朴素 `O(V * H)`。
- KV cache 是 F32 contiguous vector，没有 paging/quantization/prompt reuse。
- 没有 CPU 多线程、SIMD intrinsic 或 BLAS。

因此它的主要价值是可读、可调试、可对齐；性能优化主要放在 MPS path 和后续 backend
优化上。
