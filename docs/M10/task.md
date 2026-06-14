# M10 Task: Independent MPSGraph Backend

目标：新增一条完全独立的 `mpsgraph` backend。它不复用现有 `mps` backend，不使用手写
Metal kernel，不在 prefill/decode 热路径做 CPU/GPU 往返，主要 Qwen3 推理计算全部用
MPSGraph API 表达和执行。

技术方案见 [`../mpsgraph-backend.md`](../mpsgraph-backend.md)。

## Current Status

- [x] 已明确 `mpsgraph` 是新 backend，不是现有 `mps` backend 的分支
- [x] 已明确不复用 `MpsContext` / `MpsBuffer` / 现有 Metal kernels
- [x] 已明确 strict mode 下不支持 token streaming
- [x] 已明确第一版 sampling 先做 graph-side greedy argmax
- [x] 已创建 `include/toyllm/backends/mpsgraph/`
- [x] 已创建 `src/backends/mpsgraph/`
- [x] 已接入 `--device mpsgraph`
- [x] 已实现 MPSGraph availability probe
- [x] 已实现 tiny MPSGraph smoke executable
- [x] 已实现 Qwen3 greedy prefill/decode 的第一版 MPSGraph 路径

## Scope

本阶段范围：

1. 新增独立 MPSGraph backend 目录和公共 C++ facade
2. 新增 `DeviceKind::mpsgraph`
3. 建立 MPSGraph availability probe
4. 建立 Qwen3 0.6B MPSGraph graph builder
5. 实现 device-resident weights
6. 实现 device-resident KV cache
7. 实现 prefill graph
8. 实现 decode graph 或 decode loop graph
9. 实现 graph-side greedy sampling
10. 只在请求结束 read back generated token ids
11. 用 CPU reference 做 correctness 对齐

本阶段不包括：

- streaming token 输出
- top-k / top-p / temperature sampling
- prompt cache 跨请求复用
- 多 batch
- MoE / hybrid attention
- 旧 MPS backend 性能重构

## Hard Constraints

- [x] `mpsgraph` 代码不得 include `toyllm/backends/mps/mps_backend.hpp`
- [x] `mpsgraph` 代码不得调用 `toyllm::mps::*`
- [x] 不复用 `MpsContext`
- [x] 不复用 `MpsBuffer`
- [x] 不复用现有 `.metal` / Metal kernel
- [x] 不使用 `MPSMatrixMultiplication` 等直接 MPS primitive 路径
- [x] 不做 `mpsgraph -> mps` fallback
- [x] 不做 `mpsgraph -> cpu` fallback
- [x] prefill 不读回 hidden / logits / KV cache
- [x] decode 每步不读回 logits
- [x] decode 每步不读回 next token
- [x] decode 每步不从 CPU feed next token
- [x] sampling 必须在 graph 内完成
- [x] KV cache 必须 device-resident
- [x] weights 必须 device-resident

## Tasks

### Documentation And Architecture

- [x] 新增 MPSGraph backend 技术设计文档
- [x] 新增 M10 task checklist
- [x] 更新 `docs/architecture.md`，加入 MPSGraph backend 层
- [x] 更新 `docs/milestones.md`，加入 M10 状态和目标
- [x] 在 README 中说明 `mpsgraph` 是实验 backend

### Build And Platform Gating

- [x] CMake 新增 `KRAKEN_INFER_ENABLE_MPSGRAPH`
- [x] Makefile 新增 MPSGraph source 列表
- [x] Apple + MPSGraph 可用时编译 `.mm`
- [x] 非 Apple 或禁用时编译 stub
- [x] stub 返回明确 unavailable
- [x] 不影响现有 `cpu` / `mps` build

### Public API And Device Selection

- [x] `DeviceKind` 新增 `mpsgraph`
- [x] `Device::mpsgraph()` helper
- [x] `Device::to_string()` 支持 `mpsgraph`
- [x] CLI `--device` 解析支持 `mpsgraph`
- [x] gateway request `device` 支持 `mpsgraph`
- [x] OpenAPI schema 增加 `mpsgraph`
- [x] `doctor` 输出 MPSGraph availability

### Backend Skeleton

- [x] 新增 `include/toyllm/backends/mpsgraph/mpsgraph_backend.hpp`
- [x] 新增 `src/backends/mpsgraph/mpsgraph_backend.mm`
- [x] 新增 `src/backends/mpsgraph/mpsgraph_backend_stub.cpp`
- [x] 新增 `toyllm::mpsgraph::BackendInfo`
- [x] 新增 `query_backend()`
- [x] 新增 tiny graph smoke test
- [x] 新增 `MpsGraphContext` / `MpsGraphBuffer`
- [x] 新增基础 f32 graph operator smoke
- [x] 所有 Apple framework 类型限制在 `.mm` 内

### Runtime Separation

- [x] 新增 `src/runtime/mpsgraph/` 或等价独立 runtime 目录
- [x] 新增 `QwenMpsGraphModel`
- [x] 新增 `generate_mpsgraph()`
- [x] runtime dispatch 支持 `DeviceKind::mpsgraph`
- [x] 不把 MPSGraph path 塞进 `src/runtime/cpu/qwen_cpu_model.cpp`
- [x] 不通过现有 MPS runtime 间接调用

### Weight Store

- [x] 读取 safetensors metadata
- [x] 校验 Qwen3 0.6B 权重 shape
- [x] 建立 MPSGraph weight tensor store
- [x] embedding weight device-resident smoke
- [x] lm_head weight device-resident
- [x] 28 层 attention weights device-resident
- [x] 28 层 MLP weights device-resident
- [x] final norm weight device-resident smoke
- [ ] 权重跨请求复用
- [ ] 模型卸载时释放 weight store

### Graph Builder Foundation

- [x] 封装 MPSGraph graph construction helpers
- [ ] 封装 executable cache key
- [ ] 封装 graph compile / specialize
- [ ] 支持 fixed max sequence length
- [ ] 支持 prompt length bucket
- [ ] 支持 max new tokens bucket
- [ ] 编译失败时返回明确错误
- [ ] executable cache 跨请求复用

### Core Ops

- [x] embedding gather f32 smoke
- [x] RMSNorm f32 smoke
- [x] Q/K per-head RMSNorm
- [ ] BF16/FP16/FP32 matmul policy spike
- [x] matvec f32 smoke
- [x] q_proj
- [x] k_proj
- [x] v_proj
- [x] o_proj
- [x] gate_proj
- [x] up_proj
- [x] down_proj
- [x] lm_head
- [x] RoPE
- [x] SiLU f32 smoke
- [x] residual add f32 smoke
- [x] single-token attention f32 smoke
- [x] argmax smoke

### KV Cache

- [x] 设计 MPSGraph KV cache layout
- [x] 分配 device-resident K cache
- [x] 分配 device-resident V cache
- [x] prefill 写入 cache
- [x] decode 追加 cache
- [x] attention operator 读取 `0..position`
- [ ] fixed-shape causal mask
- [x] cache capacity 越界返回明确错误
- [x] 不维护 CPU mirror

### Prefill Graph

- [x] 输入 prompt token ids
- [ ] 输入 prompt length
- [x] 对 prompt positions 执行 forward
- [x] 写入所有 layer K/V cache
- [x] 保留 last hidden 于 device run state
- [ ] 输出 current position
- [x] 不输出 logits 到 CPU
- [x] 不读回中间 tensor
- [ ] 单个 graph/control-flow prefill

### Decode Graph

- [x] 从 device run state 读取 last hidden
- [x] runtime 传入 current position
- [x] 从 device-resident KV cache 读取历史 K/V
- [x] graph-side lm_head
- [x] graph-side greedy argmax
- [x] graph-side eos check
- [x] graph-side generated ids 写入
- [x] graph-side next token forward
- [x] graph-side position 更新
- [x] 输出 generated ids
- [x] 输出 generated count
- [x] 输出 finish reason
- [x] decode loop 内无 CPU/GPU tensor 往返
- [ ] graph-side early break 跳过 EOS 后续 forward
- [ ] 单个 graph/control-flow decode loop

### Sampling

- [x] 第一版 greedy argmax
- [x] eos token ids 支持多值
- [x] generated count 正确
- [x] finish reason 区分 stop / length
- [ ] top-k 设计文档补充
- [ ] top-p 设计文档补充
- [ ] temperature 设计文档补充

### CLI And Gateway

- [x] `infer --device mpsgraph` 可运行
- [x] `run --device mpsgraph` 可运行
- [x] `chat --device mpsgraph` 可运行
- [x] `serve --device mpsgraph` 非 streaming 请求可进入 MPSGraph runtime
- [x] strict mode 下 `stream=true` 返回明确 unsupported
- [x] 非 streaming chat completion 可返回文本
- [x] completion usage token 统计不依赖中间 readback

### Debug And Profiling

- [x] strict mode 下禁用 `--dump-dir`
- [ ] 新增显式 `--mpsgraph-debug-readback`
- [ ] debug readback 不进入性能验收
- [x] profiler 标记 backend=`mpsgraph`
- [ ] profiler 区分 graph compile / graph execute
- [x] profiler 不强制 readback 中间 tensor
- [x] profiler 标记 MPSGraph load / prefill / decode / final readback 阶段

### Tests

- [x] MPSGraph availability smoke
- [x] tiny add/mul/reduce graph smoke
- [x] RMSNorm f32 smoke vs CPU expected value
- [x] Q/K per-head RMSNorm smoke vs CPU expected value
- [x] matvec f32 smoke vs CPU expected value
- [x] RoPE vs CPU
- [x] SiLU f32 smoke vs CPU expected value
- [x] argmax smoke
- [x] device token embedding smoke
- [x] generated token write smoke
- [x] MPSGraph weight store metadata smoke
- [x] QwenMpsGraphModel core weight load smoke
- [x] MPSGraph KV cache layout smoke
- [x] MPSGraph runtime greedy decode smoke
- [x] layer 0 position 0 vs CPU
- [x] attention output vs CPU
- [x] MLP output vs CPU
- [x] tiny model `hello`, 1 token greedy vs CPU
- [x] tiny model `hello`, 2 tokens greedy vs CPU
- [x] tiny EOS 首 token 走 device-side status
- [x] no fallback test
- [x] no per-token readback instrumentation test

## Acceptance

- [x] `cmake --build --preset debug` 通过
- [x] `ctest --preset debug` 通过
- [x] `make test` 通过
- [x] `kraken-infer doctor` 能显示 MPSGraph backend 状态
- [x] `infer --device mpsgraph --prompt hello --max-new-tokens 1` 可运行
- [x] tiny greedy 首 token 与 CPU reference 一致
- [x] decode 内部不读回 logits
- [x] decode 内部不读回 next token
- [x] decode 内部不从 CPU feed next token
- [x] KV cache 不存在 CPU mirror
- [x] 不使用旧 MPS backend 类型或函数
- [x] MPSGraph 不可用时返回明确 unavailable

## Known Risks

- MPSGraph control flow 是否适合完整 decode loop 需要 spike 验证。
- 如果必须走 fixed bucket graph，compile cache 和内存占用会变复杂。
- BF16 支持和性能需要实测，不同 macOS/Xcode 组合可能有差异。
- KV cache update 在 MPSGraph 中的表达方式可能影响性能和可实现性。
- 严格无 readback 与 streaming API 目标冲突，第一版应明确拒绝 streaming。
- 现有 runtime 名称 `generate_cpu()` 会拖累 backend 抽象，需要后续清理。
