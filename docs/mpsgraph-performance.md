# MPSGraph Performance Optimization Plan

本文档定义 `mpsgraph` backend 下一阶段性能优化方案。M10 的目标是严格无 readback
路径下跑通 Qwen3 greedy 推理；M11 的目标是在不破坏独立 backend 约束的前提下，把
它从“正确但慢”推进到“可持续优化、可观测、可复用”的路径。

相关入口：

- [`include/toyllm/backends/mpsgraph/mpsgraph_backend.hpp`](../include/toyllm/backends/mpsgraph/mpsgraph_backend.hpp)
- [`include/toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp`](../include/toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp)
- [`src/backends/mpsgraph/mpsgraph_backend.mm`](../src/backends/mpsgraph/mpsgraph_backend.mm)
- [`src/backends/mpsgraph/qwen_mpsgraph_model.cpp`](../src/backends/mpsgraph/qwen_mpsgraph_model.cpp)
- [`src/runtime/mpsgraph_inference.cpp`](../src/runtime/mpsgraph_inference.cpp)
- [`include/toyllm/runtime/profiling.hpp`](../include/toyllm/runtime/profiling.hpp)
- [`docs/M10/task.md`](M10/task.md)

## Current State

当前 `mpsgraph` backend 已经具备第一版正确性能力：

1. `DeviceKind::mpsgraph` 已接入 CLI 和 gateway。
2. Qwen3 权重可以上传到 MPSGraph/Metal device buffer。
3. prefill 和 decode 热路径不读回 hidden、logits、KV cache 或 next token。
4. greedy argmax、generated ids 写入、eos 状态更新在 device 侧完成。
5. 请求结束只读回 generation status 和必要的 generated ids。
6. strict no-readback 模式下明确拒绝 streaming。
7. profile summary 已能标记 `mpsgraph.load_weights`、`request.prefill`、`request.decode`
   和最终 readback。

但当前实现仍是“算子级图调用”：

```text
request
  -> create MpsGraphContext
  -> load and upload all weights
  -> create run state
  -> prefill:
       for token:
         embedding graph
         for layer:
           rms_norm graph
           matvec graph
           qk_norm graph
           rope graph
           kv_write graph
           attention graph
           ...
  -> decode:
       for step:
         argmax graph
         record token graph
         update status graph
         forward_next_token graph
```

这个结构容易验证，但性能上有几个明确瓶颈：

1. 每次请求重新创建 context、读取权重、上传权重。
2. 每个小算子临时构建 `MPSGraph` 并执行，缺少 executable/cache 复用。
3. decode 每 token 仍由 C++ runtime 循环调度多个图，host submit 开销很高。
4. EOS 后已经能记录 stop 状态，但还不能在 graph 内提前跳过后续 forward。
5. 第一版权重上传为 f32，显存和带宽压力都偏大。
6. profile 还没有明确区分 graph build、compile、execute、cache hit/miss。

## Goals

M11 的目标：

1. 让 gateway 的第二次及后续请求复用已上传权重。
2. 建立 MPSGraph graph/executable cache 基础设施。
3. 降低 decode 阶段 per-token/per-layer host 调度次数。
4. 明确 prefill、decode、compile、execute、cache 的 profile 指标。
5. 为后续 control-flow decode loop 和 mixed precision 做好接口边界。
6. 保持 `mpsgraph` backend 独立，不依赖旧 `mps` backend 或 CPU fallback。

第一阶段优先追求“明显消除冷启动重复成本”和“建立可量化优化闭环”。不要在没有数据的
情况下直接大改所有算子。

## Non-Goals

M11 不解决：

1. strict no-readback streaming。
2. 多 batch。
3. top-k / top-p / temperature sampling 的完整实现。
4. prompt cache 跨请求复用。
5. 旧 `mps` backend 性能重构。
6. CPU/MPS/MPSGraph 混合执行。
7. 生产级并发调度和显存淘汰策略。

## Hard Constraints

这些约束继承自 M10，并在性能优化阶段继续有效：

1. `mpsgraph` 不 include `toyllm/backends/mps/*`。
2. `mpsgraph` 不调用 `toyllm::mps::*`。
3. 不复用 `MpsContext`、`MpsBuffer` 或现有 Metal kernels。
4. 不做 `mpsgraph -> mps` fallback。
5. 不做 `mpsgraph -> cpu` fallback。
6. prefill/decode 热路径不读回 logits、hidden、KV cache、next token。
7. sampling 仍在 graph/device 侧完成。
8. profile 和 debug readback 只能作为显式 debug 模式，不能进入性能验收路径。

## Baseline Metrics

优化前必须固定一个 profile 采集口径。建议至少记录两类请求：

```bash
./build/debug/kraken-infer infer \
  --model models/qwen3-0.6b \
  --device mpsgraph \
  --prompt hello \
  --max-new-tokens 1 \
  --profile all

curl --location 'http://127.0.0.1:8080/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'x-kraken-profile: all' \
  --data '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16,
    "stream": false,
    "device": "mpsgraph"
  }'
```

summary 中至少要包含：

```text
request.total_ms
request.tokenize_ms
mpsgraph.create_context_ms
mpsgraph.load_weights_ms
mpsgraph.create_run_state_ms
request.prefill_ms
request.decode_ms
generated_tokens
tokens_per_second
mpsgraph_h2d_calls
mpsgraph_d2h_calls
mpsgraph_graph_build_ms
mpsgraph_graph_compile_ms
mpsgraph_graph_execute_ms
mpsgraph_cache_hit_count
mpsgraph_cache_miss_count
```

当前已知的主要性能问题是：

1. 冷请求大头在 `mpsgraph.load_weights` 和 prefill。
2. 多 token 请求的 decode 时间会被 `mpsgraph.decode.forward_next_token` 拉高。
3. 第二次请求如果仍重新上传权重，就无法体现常驻服务的收益。

## Architecture Overview

M11 建议引入三个新边界：

```text
MpsGraphRuntimeCache
  -> owns shared MpsGraphContext
  -> owns loaded QwenMpsGraphModel
  -> owns executable caches
  -> records cache/profile counters

MpsGraphExecutableCache
  -> key: op/graph type + dtype policy + static shape + bucket
  -> value: compiled graph/executable bundle

MpsGraphRunPool
  -> optional request-state buffer pool
  -> owns reusable KV/activation/generated buffers by capacity bucket
```

请求路径应逐步变成：

```text
request
  -> tokenize
  -> get_or_load MpsGraphRuntimeCache(model_dir, dtype_policy)
  -> allocate or reuse run state bucket
  -> execute cached prefill graph or coarse prefill stage
  -> execute cached decode stage
  -> final status/id readback
```

第一版可以先只做 `MpsGraphRuntimeCache`，把 context 和 uploaded weights 跨请求复用；
后续再接 executable cache 和 run-state pool。

## Phase 1: Weight And Context Reuse

这是收益最高、风险最低的优化。

当前 `generate_mpsgraph()` 每次请求都执行：

```text
MpsGraphContext::create()
QwenMpsGraphModel::load_all_weights()
QwenMpsGraphModel::create_run_state()
```

应改为：

```text
runtime cache miss:
  create context
  load metadata
  upload all weights
  store model in cache

runtime cache hit:
  reuse context
  reuse uploaded weights
  only create request run state
```

建议缓存 key：

```text
realpath(model_dir)
device_name or registry id
dtype_policy
max_seq_capacity_policy
backend_version
```

第一版可以只支持单模型常驻：

```text
static mutex
static optional<MpsGraphRuntimeCache>
```

gateway 当前主要是单模型服务，先做单模型缓存足够。后续如果要支持多模型，再扩展成
LRU map。

### Thread Safety

当前 server 请求处理偏串行，但 cache 不能假定永远单线程。建议：

1. cache 初始化用 `std::mutex` 保护。
2. `QwenMpsGraphModel` 的 device weights 只读，可以跨请求共享。
3. `QwenMpsGraphRunState` 必须请求私有，不能跨并发请求共享同一实例。
4. profile metadata 记录 `mpsgraph_model_cache=hit|miss`。

### Acceptance

1. 同一进程第二次 `mpsgraph` 请求不再出现完整权重上传耗时。
2. 第二次请求 profile 中 `mpsgraph.load_weights` 为 0 或标记为 cache hit。
3. `mpsgraph_h2d_calls` 相比冷请求显著下降。
4. tiny correctness smoke 和真实 Qwen3 smoke 都保持通过。

## Phase 2: Graph/Executable Cache

当前 `MpsGraphContext` 中每个 op 都临时创建 graph 并 `runWithMTLCommandQueue`。这导致
大量重复 graph construction/compile 成本。M11 需要先建立 cache 结构，再逐步迁移 op。

推荐 key：

```text
struct MpsGraphExecutableKey {
  std::string graph_name;
  std::vector<std::int64_t> input_shapes;
  std::vector<std::int64_t> output_shapes;
  std::string dtype_policy;
  std::size_t capacity_bucket;
  std::size_t prompt_bucket;
  std::size_t max_new_tokens_bucket;
  std::uint64_t flags;
};
```

第一批适合缓存的图：

1. `rms_norm_f32`
2. `qk_norm_f32`
3. `silu_mul_f32`
4. `add_f32`
5. `argmax_i32`
6. `write_i32_token`
7. `update_generation_status_i32`
8. `matvec_f32` 的固定 shape 版本

注意：如果某个 MPSGraph API 版本的 executable 绑定 MTLBuffer 方式不稳定，可以先实现
“graph builder cache”或“shape-specialized helper object”，但 profile 上仍要把 build /
compile / execute 拆开。

### Profile Fields

每次执行 graph 记录：

```text
mpsgraph.graph_build
mpsgraph.graph_compile
mpsgraph.graph_execute
mpsgraph.executable_cache_hit
mpsgraph.executable_cache_miss
```

summary 聚合时要能回答：

1. 哪些图 build 最频繁。
2. 哪些图 execute 最耗时。
3. cache hit rate 是否随第二次请求提高。

## Phase 3: Coarse Forward Graphs

算子级 cache 做完后，下一步是减少 host submit 数量。目标不是把整个模型一次性塞进一个
不可维护的大图，而是按 Qwen3 的天然边界做粗粒度图。

推荐顺序：

1. 单层 attention 子图：
   ```text
   input hidden
     -> input rms norm
     -> q/k/v projection
     -> q/k norm
     -> rope
     -> kv write
     -> attention
     -> o_proj
     -> residual add
   ```

2. 单层 MLP 子图：
   ```text
   hidden
     -> post attention rms norm
     -> gate/up
     -> silu_mul
     -> down_proj
     -> residual add
   ```

3. 单层完整 transformer block 子图。
4. 单 token full forward 子图。
5. fixed bucket prefill graph。

这样做的理由：

1. 每一步都有 CPU reference 和现有 MPSGraph op 可对齐。
2. 出错时能定位到 attention、MLP 或 block。
3. 不需要一开始处理完整 decode control flow。

## Phase 4: Decode Loop And Early Stop

当前 decode 虽然 device-side 记录 eos 状态，但 C++ loop 仍会按照 `max_new_tokens` 调度，
只是在最终 readback 后知道 `finish_reason`。后续优化应把循环控制逐步移到 graph 侧：

```text
decode_loop(max_new_tokens):
  if finished:
    skip forward
  logits = lm_head(hidden)
  token = argmax(logits)
  write generated token
  update finish status
  if finished:
    skip next-token forward
  hidden = forward_token(token, position)
  position += 1
```

如果 MPSGraph control-flow API 不适合完整 loop，可以先做固定 bucket：

```text
max_new_tokens bucket: 1, 2, 4, 8, 16, 32, 64, 128
```

每个 bucket 内用 device-side `finished` mask 控制后续结果不再改变。即使物理上仍执行若干
op，也要保证 generated count 和 output 语义正确。真正跳过计算可以作为下一阶段。

## Phase 5: Run-State And Buffer Reuse

`QwenMpsGraphRunState` 目前按请求分配：

```text
hidden / normed / q / k / v / logits / generated_tokens / generation_status / kv_cache
```

这些 buffer 的 shape 大多只由模型结构和 capacity 决定。M11 后半段可以加 bucket pool：

```text
capacity bucket: 128, 256, 512, 1024, 2048
```

请求结束时 run state 回收到 pool。下一次请求如果 capacity bucket 匹配，则只 reset
generation status 和 KV used 状态，不重新分配全部 buffer。

第一版先不做复杂并发池，只做单线程可用的 freelist 即可。

## Phase 6: DType Policy

当前 MPSGraph weight upload 走 f32，简单但慢且占内存。后续应引入明确 dtype policy：

```text
f32_debug
fp16_weights_f32_accum
bf16_weights_f32_accum
```

策略：

1. `f32_debug` 继续作为 correctness baseline。
2. 优先 spike `fp16_weights_f32_accum`，因为 Apple GPU/Graph 路径通常更稳定。
3. 再 spike 原始 BF16 权重直接进入 graph。
4. 每个 policy 都必须有 tiny model correctness smoke。
5. profile metadata 记录 `mpsgraph_dtype_policy`。

验收不应只看速度，还要记录：

```text
device_weight_bytes
max_abs_error
mean_abs_error
first_token_match
```

## Profiling Strategy

性能优化不能只靠一次体感测试。M11 应固定 profile 输出结构：

```text
metadata:
  model_dir
  device
  dtype_policy
  prompt_tokens
  generated_tokens
  cache_hit/miss
  h2d/d2h calls

spans:
  request.total
  request.tokenize
  mpsgraph.cache.lookup
  mpsgraph.create_context
  mpsgraph.load_weights
  mpsgraph.create_run_state
  request.prefill
  request.decode
  mpsgraph.graph_build
  mpsgraph.graph_compile
  mpsgraph.graph_execute
  mpsgraph.final_readback.generation_status
  mpsgraph.final_readback.generated_ids
```

profile 页面需要能看：

1. cold request vs warm request。
2. top operators by self time。
3. cache hit/miss。
4. graph build/compile/execute 占比。
5. H2D/D2H 调用次数。

## Suggested Implementation Order

1. 先补 profile 字段，把 cold/warm 的瓶颈量化。
2. 实现单模型 `MpsGraphRuntimeCache`，复用 context 和 weights。
3. 在 gateway 启动时可选 warmup 或 lazy load。
4. 给 `MpsGraphContext` 增加 graph/executable cache 的 key 和统计。
5. 迁移小 op 到 cached executable。
6. 迁移 `matvec` 和 attention 相关固定 shape graph。
7. 做单层 attention/MLP/block 粗粒度 graph。
8. 做 decode loop 或 fixed bucket decode graph。
9. 引入 run-state buffer pool。
10. 引入 dtype policy spike。

## Risks

1. `MPSGraphExecutable` 的 API 行为和 macOS/Xcode 版本强相关，需要保留 stub 和明确
   unavailable 错误。
2. Graph cache 可能增加显存和内存占用，需要 profile cache size。
3. 过早构建全模型大图会降低可调试性。
4. fp16/bf16 policy 可能造成首 token 不一致，必须先以 tiny model 和 f32 baseline 对齐。
5. 并发请求复用同一个 context/queue 时可能暴露线程安全问题，第一版要用 mutex 保守处理。

## Acceptance

M11 完成时应满足：

1. `make test` 通过。
2. `cmake --build --preset debug && ctest --preset debug` 通过。
3. `infer --device mpsgraph --profile summary` 可生成包含 cache 字段的 summary。
4. gateway 连续两次非 streaming `mpsgraph` 请求都能成功。
5. 第二次请求复用 context 和 weights，不再完整重新上传权重。
6. strict no-readback 约束仍通过：decode 内无 logits/next token readback。
7. profile 页面能区分 cold/warm 请求，并展示 graph build/compile/execute 或等价统计。
8. 如果 executable cache 未完全落地，文档和 profile 必须明确当前仍处于 graph-run fallback。
