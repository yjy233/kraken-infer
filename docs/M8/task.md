# M8 Task: MPS Full Forward And Performance Optimization

目标：从 M7 的 MPS-assisted `lm_head` 继续推进到可交互速度的 MPS full forward。M8 按可验证切片推进，每个切片都要保留 CPU reference path，并用小输入对齐后再进入真实 Qwen3 0.6B 生成路径。

## Current Status

- [x] M7 已完成 Metal device probe、MPS context/buffer、BF16 matvec、`--device mps`
- [x] MPS-assisted `lm_head.weight @ hidden` 已接入 `infer`/`run`/`chat`
- [x] `lm_head.weight` 已上传到 Metal buffer 并跨请求复用
- [x] `lm_head` matvec 已改为 threadgroup 内并行归约，避免一行一个线程串行累加
- [x] `lm_head` matvec 已增加可复用 workspace，避免每个 generated token 重复分配 input/output buffer
- [x] `forward_ready` 在 full forward 可用时为 `true`
- [x] attention、MLP、RMSNorm、RoPE、KV cache 已迁到 MPS full-forward path

## Scope

M8 的最终完成标准是 `--device mps` 能把主要 forward 算子放到 MPS，并且 greedy decode token 与 CPU reference 对齐。M8 不要求实现通用深度学习框架，只覆盖 Qwen3 0.6B 当前需要的 batch size 1 prefill/decode 路径。

## Optimization Slices

### Slice 1: MPS `lm_head` Performance Cleanup

- [x] 为固定 rows/cols 的 matvec 增加 reusable workspace
- [x] 复用 input/output Metal buffer，减少 decode step 动态分配
- [x] kernel 使用每行一个 threadgroup 的并行归约
- [x] smoke test 覆盖 workspace 连续复用
- [x] 在 Apple Silicon 本机记录 CPU logits 与 MPS logits 最大误差：`position.12.logits` max abs `7.34329e-05`
- [x] 在 Apple Silicon 本机记录 `--device cpu` 与 `--device mps` 对比：`hello`, 2 tokens，CPU `21.29s`，MPS `4.57s`

### Slice 2: Device-Resident Weight Cache

- [x] 为 Qwen 权重建立 MPS buffer cache
- [x] 上传 embedding、per-layer attention、MLP、norm 权重
- [x] 权重 cache 与 `CpuQwenModel` 生命周期绑定，跨请求复用
- [x] 对 embedding/lm_head 分别缓存；当前模型提供显式 `lm_head.weight`
- [x] 明确 storage mode 策略：Apple Silicon bring-up 使用 shared buffer，并保留 unavailable/error 信息

### Slice 3: Elementwise And Norm Operators

- [x] MPS embedding lookup：token id -> hidden vector
- [x] MPS RMSNorm：final norm、input norm、post-attention norm
- [x] MPS Q/K norm
- [x] MPS RoPE
- [x] MPS SiLU gate multiply
- [x] 小 tensor smoke test 覆盖每个算子

### Slice 4: Linear Projection Operators

- [x] MPS BF16 weight + F32 activation matvec 泛化到 layer projections
- [x] q/k/v/o projection 接入单层 path
- [x] gate/up/down projection 接入单层 path
- [x] projection 使用 device input/output buffer，避免 CPU round-trip
- [x] 单层输出与 CPU reference 对齐；`position.0.layer.0.mlp_down_proj` max abs `2.98023e-06`

### Slice 5: MPS KV Cache And Attention

- [x] 设计 device-resident KV cache layout
- [x] prefill 写入 K/V cache
- [x] decode 追加当前 token K/V
- [x] grouped-query attention 读取 MPS KV cache
- [x] softmax 和 value aggregation 留在 MPS
- [x] cache capacity 越界保持明确错误

### Slice 6: Full Forward Integration

- [x] `--device mps` 走完整 MPS layer forward
- [x] 默认路径只在采样前取回 logits；`--dump-dir` 开启时才读回中间 tensor
- [x] 全模型首 token greedy 与 CPU reference 对齐：`hello` -> `Hello`
- [x] 短 prompt 多 token greedy 与 CPU reference 对齐：`hello`, 2 tokens -> `Hello!`
- [x] `BackendInfo.forward_ready` 在完整路径可用时改为 `true`

## Acceptance

- [x] `ctest --preset debug` 通过
- [x] `toyllm mps-smoke` 在 Apple Silicon 上通过
- [x] `infer --device mps --prompt hello --max-new-tokens 1` 可运行
- [x] `infer --device mps --prompt hello --max-new-tokens 8 --seed 1` 可运行
- [x] CPU/MPS greedy 首 token 对齐
- [x] CPU/MPS debug dump 的单层关键 tensor 误差在设定阈值内：抽样最大值 `1.50681e-04`
- [x] 无 Metal device 或非 MPS build 仍返回明确 unavailable，不静默 fallback

## Known Risks

- BF16 accumulation 与 CPU reference 的 F32 累加顺序不同，误差阈值需要按算子设置。
- 当前 batch size 1 假设应保留在接口或注释里，避免误用成批量路径。
- 当前使用 `MTLResourceStorageModeShared`，后续若继续做性能专项，可以引入 private buffer 和 blit/upload 策略。
- 每层 projection 直接用 matvec 能先打通路径，但最终性能可能需要 tiled matmul 或 fused kernels。
