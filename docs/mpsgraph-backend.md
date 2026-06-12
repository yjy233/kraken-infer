# MPSGraph Backend Design

本文档定义一条新的 `MPSGraph` backend 路线。它不是现有 `mps` backend 的重构，也不是
在 `MpsContext` 上继续堆算子，而是一条完全独立的图执行路径。

目标 backend 名称：

```text
mpsgraph
```

目标使用方式：

```bash
./build/manual/kraken-infer infer --device mpsgraph --prompt hello --max-new-tokens 16
./build/manual/kraken-infer serve --device mpsgraph --model models/qwen3-0.6b
```

## Hard Requirements

### 1. 不和现有 MPS backend 耦合

`mpsgraph` backend 必须独立于当前 `mps` backend。

禁止：

- include `include/toyllm/backends/mps/mps_backend.hpp`
- 调用 `src/backends/mps/mps_backend.mm` 中的任何类型或函数
- 复用 `MpsContext`
- 复用 `MpsBuffer`
- 复用现有 Metal kernel
- 通过 `mps` backend 做 fallback
- 把 `mpsgraph` 写成 `mps` backend 的分支

允许复用：

- `ModelConfig` / `GenerationConfig`
- tokenizer 和 chat template
- safetensors 解析和权重 metadata
- sampling 配置结构
- CPU reference backend 作为测试 oracle

也就是说，模型资产读取可以复用，但实际推理 backend 必须是新实现。

### 2. 全部图计算使用 MPSGraph

MPSGraph backend 的计算图必须使用 Apple `MetalPerformanceShadersGraph` API 表达。

允许的 API 边界：

- `MPSGraph`
- `MPSGraphTensor`
- `MPSGraphTensorData`
- `MPSGraphExecutable`
- `MPSGraphDevice`
- `MPSGraphCompilationDescriptor`
- `MPSGraphExecutionDescriptor`
- MPSGraph tensor ops

禁止：

- 手写 Metal shader
- `MTLComputePipelineState` 自定义 kernel
- `MPSMatrixMultiplication` / `MPSNDArray` 等直接 MPS primitive 路径
- CPU 实现 attention、MLP、lm_head、sampling 后再拼到图路径里

如果某个 Qwen3 算子不能用 MPSGraph 表达，第一版应返回明确 `unavailable`，而不是静默
fallback 到 CPU 或旧 MPS。

### 3. 推理热路径不做 CPU/GPU 往返

这里的“无 CPU/GPU 内存交换”定义为：

- prefill 过程中不读回 hidden、logits、attention score、KV cache
- decode 每一步不读回 logits
- decode 每一步不把 sampled token 从 CPU 再写回 GPU
- sampling 不在 CPU 上执行
- KV cache 始终 device-resident
- weights 始终 device-resident
- intermediate tensors 始终 device-resident

允许的 I/O 边界只有：

- 模型加载阶段：权重从 safetensors 初始化到 MPSGraph 可执行使用的 device-resident tensor
- 请求开始：prompt token ids 作为输入进入 device-resident graph
- 请求结束：最终 generated token ids 或最终文本所需 token ids 返回 host

如果要支持 token streaming，它会天然要求每步把 token id 读回 host 再 decode 成文本。严格
`mpsgraph` 模式第一版不支持 streaming；后续可以单独定义 `mpsgraph-streaming` 降级策略，
但不能混进“无往返”验收标准。

## Backend Layout

建议新增目录：

```text
include/toyllm/backends/mpsgraph/
  mpsgraph_backend.hpp

src/backends/mpsgraph/
  mpsgraph_backend.mm
  mpsgraph_backend_stub.cpp
  qwen_mpsgraph_model.mm
  qwen_mpsgraph_model.hpp
  mpsgraph_weight_store.mm
  mpsgraph_weight_store.hpp
  mpsgraph_kv_cache.mm
  mpsgraph_kv_cache.hpp
  mpsgraph_sampling.mm
  mpsgraph_sampling.hpp
```

公共 header 只暴露 C++ 类型。Objective-C++ 和 Apple framework 类型只允许出现在 `.mm`
实现或 private header 中。

命名空间：

```cpp
namespace toyllm::mpsgraph {
  // ...
}
```

不要放进 `toyllm::mps`。

## Runtime Integration

当前 public API 里 device 只有 `cpu` / `mps`。MPSGraph 需要新增一个独立 device kind：

```text
cpu
mps
mpsgraph
```

设计上，runtime dispatch 应变成：

```text
DeviceKind::cpu      -> CPU reference path
DeviceKind::mps      -> existing MPS backend
DeviceKind::mpsgraph -> new MPSGraph backend
```

`generate_cpu()` 这个名字已经不准确。MPSGraph 接入时应新增更中性的 entrypoint，例如：

```cpp
Result<GenerationResult> generate_text(const GenerationRequest& request);
```

短期为了少改 gateway，也可以保留旧入口名，但内部不能把 `mpsgraph` 路径塞进
`src/runtime/cpu/qwen_cpu_model.cpp`。新的图 backend 应有自己的 runtime 文件。

## Graph Granularity

MPSGraph backend 不应按 CPU/MPS 现有算子逐个执行。目标是更粗粒度的图：

```text
graph_init_weights
graph_prefill
graph_decode_loop
graph_finalize
```

### graph_prefill

输入：

```text
prompt_ids: [prompt_len]
prompt_len: scalar
max_seq_len: compile-time or graph parameter
```

输出或更新：

```text
kv_cache_k: [layers, max_seq_len, kv_heads, head_dim]
kv_cache_v: [layers, max_seq_len, kv_heads, head_dim]
last_hidden: [hidden_size]
position: scalar
```

prefill 需要对 prompt 中每个 token 顺序执行 transformer block。第一版可以有两种实现策略：

1. 使用 MPSGraph control flow 表达 token loop
2. 编译固定 `prompt_len` bucket 的 graph，例如 `1, 8, 16, 32, 64, 128, 256`

严格无往返模式下，不能在 host 上逐 token 读取结果，也不能把每步 logits 读回 CPU。

### graph_decode_loop

输入：

```text
kv_cache_k
kv_cache_v
last_hidden
position
max_new_tokens
sampling config
eos token ids
```

输出：

```text
generated_ids: [max_new_tokens]
generated_count: scalar
finish_reason: scalar enum
```

decode loop 必须在 device 侧完成：

```text
for step in 0..max_new_tokens:
  logits = lm_head(last_hidden)
  next_token = sample(logits)
  if eos(next_token): break
  generated_ids[step] = next_token
  last_hidden = forward_token(next_token, position)
  position += 1
```

第一版 sampling 可以只实现 greedy `argmax`。只要 sampling 发生在 MPSGraph 内部，就满足无
per-token CPU/GPU 往返的要求。

temperature / top-k / top-p 可以作为后续阶段。

## Qwen3 Graph Decomposition

单 token forward 的图结构：

```text
token id
  -> embedding gather
  -> for each layer:
       input RMSNorm
       q/k/v matmul
       q/k RMSNorm
       RoPE
       KV cache write
       GQA attention over cache[0..position]
       o_proj matmul
       residual add
       post-attention RMSNorm
       gate/up matmul
       SiLU(gate) * up
       down_proj matmul
       residual add
  -> final RMSNorm
  -> hidden
```

### Embedding

MPSGraph 表达：

```text
hidden = gather(embedding_weight, token_id, axis=0)
```

输出：

```text
hidden: [1024]
```

### RMSNorm

公式：

```text
mean_square = mean(x * x)
scale = rsqrt(mean_square + eps)
y = x * scale * weight
```

需要用 MPSGraph 的 elementwise、reduction、broadcast 表达。

### Linear Projection

所有 projection 都是 matmul 或 matrix-vector：

```text
y = weight @ x
```

Qwen3 0.6B 权重是 BF16。第一版要优先验证 MPSGraph BF16 tensor 支持；如果当前系统对
BF16 graph op 支持不足，应明确失败，而不是自动转 CPU。

可接受的第一阶段权衡：

- load 阶段把 BF16 权重转换成 FP16 或 FP32 device tensor
- 但转换只能发生在模型加载阶段
- forward 热路径不能 CPU 参与

### Q/K Norm

Q/K norm 是按 head 做 RMSNorm：

```text
q: [16, 128]
k: [8, 128]
weight: [128]
```

Graph 中需要 reshape 后沿 `head_dim` reduce。

### RoPE

RoPE 只作用于 Q/K。推荐预先构造 device-resident cos/sin table：

```text
cos: [max_seq_len, head_dim / 2]
sin: [max_seq_len, head_dim / 2]
```

forward 时用 `position` gather 当前行：

```text
cos_p = gather(cos, position)
sin_p = gather(sin, position)
```

然后对每个 head 做 half-rotation。

### KV Cache

逻辑 shape：

```text
k_cache: [28, max_seq_len, 8, 128]
v_cache: [28, max_seq_len, 8, 128]
```

硬约束：

- cache 在 device 侧分配
- prefill 和 decode 都写同一份 cache
- attention 只读取 `0..position`
- 不允许为了 debug 或 sampling 读回 cache

MPSGraph 实现需要验证最合适的更新方式：

- graph variable update
- scatter update
- slice + concat
- 或固定 bucket graph 输出新 cache tensor

如果只能通过输出整个 cache 再由 host 重新 feed，同样必须保持 tensor data 绑定 device
buffer，不能把 cache 内容拷到 CPU。

### Causal GQA Attention

Qwen3 0.6B：

```text
query heads = 16
kv heads    = 8
group       = 2
head_dim    = 128
```

对每个 query head：

```text
kv_head = q_head / 2
score[t] = dot(q[q_head], k_cache[layer, t, kv_head]) / sqrt(128)
t = 0..position
prob = softmax(score)
out[q_head] = sum_t prob[t] * v_cache[layer, t, kv_head]
```

在图里需要显式保证 future token 不参与：

- decode graph 只 slice `cache[0..position]`
- 或构造 causal mask，把 `t > position` 的 score 置为 `-inf`

推荐第一版使用 fixed max sequence 的 mask：

```text
mask[t] = 0      if t <= position
mask[t] = -inf   if t > position
```

这样 graph shape 固定，更容易 compile/cache。

### MLP

```text
gate = gate_proj @ x
up   = up_proj @ x
mid  = silu(gate) * up
down = down_proj @ mid
```

SiLU：

```text
silu(x) = x / (1 + exp(-x))
```

全部用 MPSGraph elementwise 表达。

### LM Head And Sampling

`lm_head` 输出：

```text
logits: [151936]
```

严格模式下不能读回 CPU 做 argmax。第一版 sampling：

```text
next_token = argmax(logits)
```

后续再扩展：

- temperature scaling
- top-k
- top-p
- seedable random sampling

## Executable Cache

MPSGraph 编译成本可能很高，需要缓存 executable。

推荐 key：

```text
model_id
max_seq_len
prompt_bucket
max_new_tokens_bucket
dtype_policy
sampling_mode
```

缓存对象：

```text
MpsGraphExecutableBundle {
  prefill executable
  decode executable
  target tensors
  feed placeholders
  static graph constants
}
```

Graph 不能每个请求重新构建。请求级状态只应该是：

- prompt ids tensor
- KV cache tensor data
- generated ids tensor
- scalar config tensor

## Memory Policy

第一版要明确区分四类内存：

```text
weights        device-resident, model lifetime
kv cache       device-resident, request/session lifetime
activations    device-resident, graph execution lifetime
outputs        generated token ids, request end readback only
```

禁止：

- decode step 中读回 logits
- decode step 中读回 next token
- debug dump 默认读回中间 tensor
- CPU/GPU 混合 attention
- CPU/GPU 混合 MLP

Debug 策略：

- `--dump-dir` 在 `mpsgraph` strict mode 下默认不可用
- 如需调试，必须显式启用 `--mpsgraph-debug-readback`
- debug readback 模式不算性能验收路径

## Error And Fallback Policy

`mpsgraph` backend 失败时必须明确报错：

```text
MPSGraph backend unavailable: <reason>
```

禁止静默 fallback：

```text
mpsgraph -> mps
mpsgraph -> cpu
```

允许用户显式选择其它 backend：

```bash
--device cpu
--device mps
```

## Validation Strategy

MPSGraph backend 必须用 CPU reference path 对齐，但对齐过程不能污染 strict runtime。

### Unit / Smoke

- MPSGraph availability probe
- tiny graph add/mul/reduce smoke
- RMSNorm graph vs CPU
- matmul graph vs CPU
- RoPE graph vs CPU
- SiLU graph vs CPU
- greedy argmax graph vs CPU

### Layer-Level

- layer 0 position 0 output vs CPU reference
- attention output vs CPU reference
- MLP output vs CPU reference

### Full Model

- `hello`, greedy 1 token：CPU/MPSGraph token id 一致
- `hello`, greedy 2 tokens：CPU/MPSGraph token ids 一致
- short chat prompt：首 token 一致

### Residency

需要增加测试或 instrumentation 证明：

- decode loop 没有 logits readback
- decode loop 没有 per-token CPU feed
- KV cache 没有 CPU mirror
- sampling 在 graph 内完成

## Known Risks

- MPSGraph control flow 和 dynamic shape 能力需要实测；如果 while loop 不适合，需要走
  fixed bucket graph。
- BF16 op 支持和性能需要按 macOS / Xcode / Apple Silicon 型号验证。
- `vocab_size = 151936` 的 lm_head + argmax graph 对编译时间和内存压力都很大。
- KV cache scatter/update 的 graph 表达可能成为第一阶段最大风险。
- 严格无 readback 模式和 OpenAI SSE streaming 天然冲突。
- MPSGraph 的 graph compile cache 需要设计好，否则首请求延迟会很高。

## References

- Apple Developer Documentation: [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph)
- Apple Developer Documentation: [MPSGraphExecutable](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutable)
- Apple Developer Documentation: [MPSGraphTensorData](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphtensordata)
- Apple Developer Documentation: [Metal Performance Shaders Graph](https://developer.apple.com/documentation/metalperformanceshadersgraph)
