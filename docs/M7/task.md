# M7 Task: MPS Backend Bring-Up

目标：建立可测试的 macOS MPS/Metal 计算基础，并先把 Qwen3 0.6B 的 `lm_head` logits matvec 接入真实生成路径。

## Scope

本阶段不是完整 MPS forward。attention、MLP、RoPE、RMSNorm、KV cache 仍走 CPU reference path；MPS 先负责 `lm_head.weight @ hidden`。

## Tasks

### Backend Foundation

- [x] 保留 macOS Metal/MPS device probe
- [x] `BackendInfo` 区分 `compute_ready` 和 `forward_ready`
- [x] 定义 opaque `MpsContext`
- [x] 定义 opaque `MpsBuffer`
- [x] 支持 Metal command queue
- [x] 支持 shared Metal buffer allocation
- [x] 支持 host 到 MPS buffer copy
- [x] 非 MPS build 提供 stub，并返回明确 unavailable

### Metal Operator

- [x] 内置 Metal compute kernel source
- [x] 实现 BF16 weight 到 F32 的 kernel 内转换
- [x] 实现 BF16 matrix + F32 vector -> F32 vector matvec
- [x] 支持 command buffer dispatch 和错误检查
- [x] 增加 `mps-smoke` CLI 验证小矩阵 matvec
- [x] smoke test 中覆盖 MPS operator；无 Metal device 时允许 unavailable

### Runtime Integration

- [x] `CpuGenerationRequest` 增加 `compute_device`
- [x] CLI `infer` 支持 `--device cpu|mps`
- [x] CLI `run` 支持 `--device cpu|mps`
- [x] CLI `chat` 支持 `--device cpu|mps`
- [x] `doctor` 输出 MPS compute/full-forward readiness
- [x] MPS 不可用时返回明确错误，不静默 fallback

### Qwen Path

- [x] `--device mps` 时初始化 MPS context
- [x] 上传 `lm_head.weight` 到 MPS buffer
- [x] `lm_head.weight` 跨请求复用
- [x] logits matvec 使用 MPS BF16 matvec kernel
- [x] CPU attention/MLP/KV cache 继续作为 correctness reference

## Acceptance

- [x] `make test` 通过
- [x] `help` 显示 `mps-smoke` 和 `--device cpu|mps`
- [x] `infer --device cpu --prompt hello --max-new-tokens 1` 可运行
- [x] 当前无 Metal device 环境下，`mps-smoke` 返回明确 unavailable
- [x] 当前无 Metal device 环境下，`infer --device mps` 返回明确 unavailable
- [x] Apple M4 本机 `mps-smoke` 可运行
- [x] Apple M4 本机 `infer --device mps --prompt hello --max-new-tokens 1` 可运行

## Historical Limitations Resolved In M8

- [x] 完整 MPS forward 已在 M8 实现
- [x] attention、MLP、RMSNorm、RoPE、KV cache 已在 M8 迁到 MPS
- [x] `lm_head.weight` 之外的 Qwen 权重已在 M8 上传到 MPS buffer 并跨请求复用

## Resolved After M7

- [x] MPS matvec 已从一行一个 thread 改为每行一个 threadgroup，并在 threadgroup 内做并行归约
- [x] 固定尺寸 matvec 已增加 reusable workspace，减少每个 generated token 的 input/output buffer 分配
- [x] 当前 Apple M4 环境可以枚举 Metal default device，并已验证 `--device mps` 真实文本输出

## Remaining Performance Follow-up

- [ ] 当前 storage mode 策略仍以 Apple Silicon shared buffer 为主；后续性能专项可评估 private buffer 和 blit/upload
- [ ] layer projection 目前使用 BF16 matvec 打通 full forward；后续性能专项可继续做 tiled matmul 或 fused kernels

## Next Slice Completed In M8

1. [x] MPS embedding lookup
2. [x] MPS RMSNorm
3. [x] MPS linear projection with reusable uploaded layer weights
4. [x] 单层 attention + MLP CPU/MPS tensor 对齐
5. [x] MPS KV cache device-resident layout
6. [x] 全模型 `--device mps` greedy token 与 CPU reference 对齐
