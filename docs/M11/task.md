# M11 Task: MPSGraph Performance Optimization

目标：在 M10 独立 `mpsgraph` backend 已跑通 Qwen3 greedy 推理的基础上，优化常驻服务
性能。重点是权重跨请求复用、graph/executable cache、粗粒度 forward graph、decode loop
优化、run-state 复用和 dtype policy。

技术方案见 [`../mpsgraph-performance.md`](../mpsgraph-performance.md)。

## Current Status

- [x] M10 已实现独立 `mpsgraph` backend
- [x] M10 已实现 strict no-readback greedy decode
- [x] M10 已实现 device-side generated ids / finish status
- [x] M10 已接入 gateway 非 streaming 请求
- [x] M10 已接入 profile summary 基础埋点
- [x] MPSGraph context 跨请求复用
- [x] MPSGraph weights 跨请求复用
- [x] graph build / compile / execute 分离统计
- [ ] executable cache
- [ ] 粗粒度 transformer block graph
- [ ] graph-side decode loop 或 fixed bucket decode graph
- [ ] run-state buffer pool
- [ ] mixed precision dtype policy

## Scope

本阶段范围：

1. 建立 MPSGraph 性能 baseline。
2. 复用 `MpsGraphContext` 和已上传权重。
3. 建立 graph/executable cache key、统计和逐步迁移。
4. 降低 decode 每 token 的 host graph submit 数量。
5. 增强 profile 输出，支持 cold/warm request 对比。
6. 引入 request run-state buffer 复用方案。
7. 设计并 spike fp16/bf16 dtype policy。

本阶段不包括：

- strict no-readback streaming
- 多 batch
- prompt cache 跨请求复用
- top-k / top-p / temperature sampling 完整实现
- 旧 `mps` backend 重构
- CPU/MPS/MPSGraph 混合执行

## Hard Constraints

- [ ] 不 include `toyllm/backends/mps/*`
- [ ] 不调用 `toyllm::mps::*`
- [ ] 不复用 `MpsContext`
- [ ] 不复用 `MpsBuffer`
- [ ] 不复用现有 Metal kernels
- [ ] 不做 `mpsgraph -> mps` fallback
- [ ] 不做 `mpsgraph -> cpu` fallback
- [ ] prefill 不读回 hidden / logits / KV cache
- [ ] decode 每步不读回 logits
- [ ] decode 每步不读回 next token
- [ ] decode 每步不从 CPU feed next token
- [ ] sampling 保持在 graph/device 侧
- [ ] debug readback 必须显式开启且不进入性能验收

## Tasks

### Baseline And Profiling

- [ ] 固定 MPSGraph cold request benchmark 命令
- [ ] 固定 MPSGraph warm request benchmark 命令
- [ ] 在 summary 中输出 `request.total_ms`
- [ ] 在 summary 中输出 `mpsgraph.create_context_ms`
- [ ] 在 summary 中输出 `mpsgraph.load_weights_ms`
- [ ] 在 summary 中输出 `mpsgraph.create_run_state_ms`
- [ ] 在 summary 中输出 `request.prefill_ms`
- [ ] 在 summary 中输出 `request.decode_ms`
- [ ] 在 summary 中输出 `tokens_per_second`
- [x] 在 summary 中输出 `mpsgraph_h2d_calls`
- [x] 在 summary 中输出 `mpsgraph_d2h_calls`
- [x] 在 summary 中输出 `mpsgraph_model_cache=hit|miss`
- [x] 在 summary 中输出 `mpsgraph_executable_cache_hit_count`
- [x] 在 summary 中输出 `mpsgraph_executable_cache_miss_count`
- [ ] profile 页面支持 cold/warm 请求对比
- [ ] profile 页面展示 graph build / compile / execute 占比

### Runtime Cache

- [x] 新增 `MpsGraphRuntimeCache` 或等价类型
- [x] cache 持有 `MpsGraphContext`
- [x] cache 持有已上传完整权重的 `QwenMpsGraphModel`
- [x] cache key 包含 canonical `model_dir`
- [ ] cache key 包含 backend device 标识
- [ ] cache key 包含 dtype policy
- [x] cache 初始化使用 mutex 保护
- [x] `generate_mpsgraph()` 从 cache 获取 context/model
- [x] cache miss 时创建 context 并上传权重
- [x] cache hit 时跳过 context 创建和权重上传
- [x] profile metadata 标记 runtime cache hit/miss
- [x] gateway 连续请求复用同一份 uploaded weights
- [x] CLI 单次请求行为保持可用

### Gateway Warmup And Lifecycle

- [x] `serve --device mpsgraph` 支持 lazy load cache
- [ ] 可选新增 `--mpsgraph-warmup` 或等价 warmup 开关
- [ ] warmup 失败时返回明确错误
- [ ] 服务退出时释放 MPSGraph runtime cache
- [ ] 未来多模型场景保留 cache map 扩展点
- [ ] 文档说明当前第一版只支持单模型常驻 cache

### Executable Cache Foundation

- [ ] 新增 `MpsGraphExecutableKey`
- [ ] key 支持 graph name
- [ ] key 支持 input/output shape
- [ ] key 支持 dtype policy
- [ ] key 支持 capacity bucket
- [ ] key 支持 prompt bucket
- [ ] key 支持 max-new-tokens bucket
- [x] 新增 executable cache hit/miss 统计
- [x] 新增 graph build / compile / execute 统计
- [ ] 编译失败时返回明确错误
- [ ] cache size 可观测
- [ ] 不可用平台 stub 保持可编译

### Cached Small Ops

- [ ] `rms_norm_f32` 支持 cached graph/executable
- [ ] `qk_norm_f32` 支持 cached graph/executable
- [ ] `silu_mul_f32` 支持 cached graph/executable
- [ ] `add_f32` 支持 cached graph/executable
- [ ] `argmax_i32` 支持 cached graph/executable
- [ ] `write_i32_token` 支持 cached graph/executable
- [ ] `update_generation_status_i32` 支持 cached graph/executable
- [ ] `write_kv_cache_f32` 支持 cached graph/executable
- [x] K/V cache pair 写入合并为单次 MPSGraph run
- [ ] 每个 cached op 保留 tiny smoke test

### Cached Matvec And Attention Ops

- [ ] `matvec_f32` 支持 fixed shape cache
- [x] input RMSNorm + q/k/v projection + q/k norm + RoPE 合并为一次 MPSGraph run
- [x] q/k/v projection 合并为一次 MPSGraph run
- [x] q/k norm + RoPE 合并为一次 MPSGraph run
- [x] attention output projection + residual + post RMSNorm 合并为一次 MPSGraph run
- [x] gate/up projection 合并为一次 MPSGraph run
- [x] SwiGLU + down projection + MLP residual 合并为一次 MPSGraph run
- [ ] q/k/v/o projection 复用 matvec executable
- [ ] gate/up/down projection 复用 matvec executable
- [ ] lm_head matvec 使用专用 cache key
- [x] `attention_f32` 按 KV head 分组计算 GQA，减少 per-query-head 子图重复
- [x] `attention_f32` 在 macOS 15+ 优先使用 MPSGraph SDPA fast path，失败后自动回退
- [ ] `attention_f32` 支持 fixed capacity bucket cache
- [ ] attention causal mask 支持 fixed shape
- [ ] attention cache key 包含 layer/capacity/head layout
- [ ] attention 输出继续与 CPU/tiny reference 对齐

### Coarse Graph Stages

- [x] profile 中细分 MPSGraph layer / operator 耗时
- [ ] 设计 single-layer attention graph
- [ ] 实现 single-layer attention graph smoke
- [ ] 设计 single-layer MLP graph
- [ ] 实现 single-layer MLP graph smoke
- [ ] 设计 full transformer block graph
- [ ] 实现 tiny full block graph smoke
- [ ] 设计 single-token full forward graph
- [ ] 对比 op-by-op path 和 coarse graph path 首 token 输出
- [ ] profile 中区分 coarse graph path 和 op graph path
- [ ] 支持通过内部 flag 回退到 op graph path 做 correctness 对照

### Prefill Optimization

- [ ] 支持 prompt length bucket 设计
- [ ] 支持 fixed max sequence capacity
- [ ] prefill graph 输入 prompt ids
- [ ] prefill graph 输入或内建 position 序列
- [ ] prefill graph 写入所有 layer K/V cache
- [ ] prefill graph 输出 last hidden 或更新 run state
- [ ] prefill 不输出 logits 到 CPU
- [ ] prefill 不读回中间 tensor
- [ ] profile 中记录 prompt bucket

### Decode Loop Optimization

- [ ] 设计 graph-side decode loop
- [ ] spike MPSGraph control-flow API 可行性
- [ ] 如果 control-flow 不合适，设计 fixed max-new-tokens bucket
- [ ] decode graph 输入 last hidden / position / KV cache / generated buffer
- [ ] decode graph 内执行 lm_head
- [ ] decode graph 内执行 greedy argmax
- [ ] decode graph 内写 generated ids
- [ ] decode graph 内更新 generated count / finish reason
- [ ] decode graph 内执行 next-token forward
- [x] EOS 后不再改变 generated output
- [x] host status poll 模式可跳过 EOS 后续 forward
- [x] 能跳过 EOS 后续 forward 时记录 `mpsgraph_early_break=true`
- [x] 不能物理跳过时记录 `mpsgraph_early_break=false`
- [x] profile metadata 记录 `mpsgraph_decode_forward_steps`
- [x] profile summary 固定输出 `tokenize_ms`

### Run-State Buffer Pool

- [ ] 设计 capacity bucket：128 / 256 / 512 / 1024 / 2048
- [ ] 新增 `MpsGraphRunStatePool` 或等价类型
- [ ] pool 只复用 request-private run state
- [ ] pool 不复用并发中的 run state
- [ ] 请求结束后回收 run state
- [ ] 获取 run state 时 reset generation status
- [ ] 获取 run state 时 reset KV used 状态
- [ ] profile metadata 标记 run-state pool hit/miss
- [ ] pool size 可配置或至少可观测

### DType Policy

- [ ] 新增 `MpsGraphDTypePolicy`
- [ ] 支持 `f32_debug`
- [ ] spike `fp16_weights_f32_accum`
- [ ] spike `bf16_weights_f32_accum`
- [ ] weight upload 根据 dtype policy 转换
- [ ] profile metadata 记录 dtype policy
- [ ] summary 输出 `device_weight_bytes`
- [ ] tiny model 对比 f32 baseline
- [ ] 记录 max abs error
- [ ] 记录 mean abs error
- [ ] 记录 first token 是否一致

### Tests

- [ ] runtime cache miss/hit 单元或 smoke test
- [ ] cache hit 不重复调用完整 weight upload 的测试
- [ ] mpsgraph strict no-readback regression test
- [ ] executable cache key equality/hash test
- [ ] cached small op smoke
- [ ] cached matvec smoke
- [ ] cached attention smoke
- [ ] coarse layer graph vs op path smoke
- [ ] run-state pool capacity bucket test
- [ ] dtype policy tiny correctness test
- [ ] gateway 非 streaming mpsgraph 连续请求测试
- [ ] profile summary 包含 cache 字段测试

## Acceptance

- [ ] `make test` 通过
- [ ] `cmake --build --preset debug` 通过
- [ ] `ctest --preset debug` 通过
- [ ] `git diff --check` 通过
- [ ] `infer --device mpsgraph --profile summary` 可运行
- [ ] `serve --device mpsgraph` 连续两个非 streaming 请求可运行
- [ ] warm request 复用 context 和 weights
- [ ] warm request 的 H2D 调用次数明显低于 cold request
- [ ] profile summary 能看到 runtime cache hit/miss
- [ ] profile summary 能看到 graph build / compile / execute 或明确 fallback 标记
- [ ] decode 内部仍不读回 logits
- [ ] decode 内部仍不读回 next token
- [ ] strict streaming 请求仍返回明确 unsupported
- [ ] 不引入 `mpsgraph` 到旧 `mps` backend 的依赖

## Benchmark Commands

Cold CLI request:

```bash
./build/debug/kraken-infer infer \
  --model models/qwen3-0.6b \
  --device mpsgraph \
  --prompt hello \
  --max-new-tokens 1 \
  --profile all
```

Warm gateway requests:

```bash
./build/debug/kraken-infer serve \
  --model models/qwen3-0.6b \
  --device mpsgraph \
  --profile-dir profiles
```

```bash
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

Profile page:

```text
http://127.0.0.1:8080/profile_page
```
