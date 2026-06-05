# M6 Task: MPS Forward Path And Performance Optimization

目标：把 Qwen3 0.6B 的主要推理计算从 CPU 迁移到 macOS MPS/Metal，使 CLI 能通过 `--device mps` 运行，并逐步达到可交互速度。

## Current Status

- [x] 已完成 macOS Metal/MPS device probe
- [x] 已完成 CPU correctness path
- [x] 已完成 CPU KV cache
- [x] 已完成 CLI `infer` / `run` / `chat`
- [ ] 尚未实现 MPS forward
- [ ] 尚未实现 MPS 权重上传
- [ ] 尚未实现 MPS KV cache
- [ ] 尚未实现 MPS logits/sampling

当前 MPS 只用于环境探测；真实推理仍全部走 CPU。

## Optimization Strategy

### M6.1 Baseline And Profiling

- [ ] 给 CPU prefill/decode 加计时
- [ ] 统计 prompt tokens、generated tokens、prefill latency、decode latency、tokens/s
- [ ] 分阶段记录耗时：embedding、attention、MLP、lm head
- [ ] 记录内存占用：weights、KV cache、中间 buffer
- [ ] 在 CLI 输出可选 perf summary

先明确瓶颈。当前 batch size 为 1，decode 阶段大量是 GEMV；lm head `[151936, 1024]` 也会很重。

### M6.2 Device Buffer And Weight Upload

- [ ] 定义 MPS device buffer abstraction
- [ ] 支持从 safetensors mmap tensor 上传到 MPS buffer
- [ ] 权重只上传一次，跨请求复用
- [ ] 建立 tensor name 到 device tensor view 的映射
- [ ] 处理 BF16 支持检测；不支持时 fallback 到 FP16/FP32 staging
- [ ] 校验 device tensor shape、stride、dtype

权重文件仍由 CPU mmap 读取 metadata；真正计算前把必要权重上传到 GPU，避免每 token 重复 host/device copy。

### M6.3 Operator Bring-Up Order

- [ ] embedding lookup
- [ ] RMSNorm
- [ ] Q/K per-head RMSNorm
- [ ] RoPE
- [ ] linear projection / matvec
- [ ] Q/K/V projection
- [ ] grouped-query attention score
- [ ] attention softmax
- [ ] attention value accumulation
- [ ] output projection
- [ ] MLP gate/up/down projection
- [ ] SiLU + gate multiply
- [ ] final norm
- [ ] lm head logits
- [ ] greedy argmax

建议先做单算子 CPU/MPS 对齐，再做单层对齐，最后做完整模型。最先迁移 projection 和 lm head，因为它们占主要算量。

### M6.4 KV Cache Layout

- [ ] MPS KV cache 预分配
- [ ] K/V cache 保持 device resident
- [ ] cache layout 使用 MPS-friendly contiguous layout
- [ ] 逻辑形状：`[num_layers, num_key_value_heads, max_seq_len, head_dim]`
- [ ] decode 每步只写入当前 position 的 K/V
- [ ] attention 只读取 `[0, current_position]`
- [ ] 超过 cache capacity 时返回明确错误

关键目标是避免每步把 K/V 搬回 CPU。CPU 只负责调度和最终文本 decode。

### M6.5 Execution Model

- [ ] 第一版使用明确的 per-op command dispatch，方便调试
- [ ] 单层输出和 CPU reference 对齐后，再减少中间同步
- [ ] 合并小 kernel：RMSNorm、RoPE、SiLU、residual
- [ ] 复用 command buffer / encoder 资源
- [ ] 减少每 token 动态分配
- [ ] 避免 full logits 每步回传 CPU；先支持 CPU argmax，后续做 GPU reduction/top-k

先正确，再优化。早期允许多 kernel；对齐通过后再做 fusion。

### M6.6 Validation

- [ ] MPS embedding 输出对齐 CPU
- [ ] MPS RMSNorm 输出对齐 CPU
- [ ] MPS RoPE 输出对齐 CPU
- [ ] MPS attention 单 head 输出对齐 CPU
- [ ] MPS MLP 输出对齐 CPU
- [ ] MPS 单层输出对齐 CPU
- [ ] MPS 全模型首 token 与 CPU greedy 一致
- [ ] MPS 多 token decode 与 CPU greedy 一致
- [ ] CLI smoke 覆盖 `--device mps`

允许 FP16/BF16 带来的小误差，但 greedy token 必须在固定短 prompt 上可复现。

### M6.7 CLI And Runtime Integration

- [ ] Runtime 增加 backend dispatch：`cpu` / `mps`
- [ ] CLI 增加 `--device cpu|mps`
- [ ] `doctor` 输出 MPS forward readiness
- [ ] `infer --device mps` 可运行
- [ ] `chat --device mps` 可运行
- [ ] MPS 不可用时返回明确错误，不静默 fallback

CPU 保留为 reference backend；MPS 作为加速 backend。

## Acceptance

- [ ] `make test` 通过
- [ ] `doctor` 能区分 MPS device probe 与 MPS forward readiness
- [ ] `infer --device mps --prompt hello --max-new-tokens 4` 返回真实文本
- [ ] 固定短 prompt 下，MPS greedy token 与 CPU greedy token 一致
- [ ] MPS decode 阶段 K/V cache 常驻 device
- [ ] MPS 路径没有每层 full tensor 回传 CPU
- [ ] MPS tokens/s 明显高于当前 CPU path

## Risks

- BF16 在不同 macOS/MPS 环境的支持需要运行时确认
- batch size 1 的小 GEMV 可能受 dispatch overhead 影响
- lm head logits 很大，CPU 回传全 logits 会拖慢 decode
- 过早 kernel fusion 会增加调试难度
- 当前执行环境没有可用 Metal device，最终 MPS forward 需要在 Apple Silicon 本机验证

## Recommended First Implementation Slice

第一刀不要直接做整模型。建议按下面顺序：

1. `--device mps` 参数和 runtime backend skeleton
2. device buffer abstraction
3. upload `model.embed_tokens.weight` 和一层 RMSNorm weight
4. MPS embedding lookup + RMSNorm，与 CPU 对齐
5. MPS linear matvec，用一层 `q_proj` 做 CPU/MPS 对齐
6. 单层 attention + MLP 对齐
7. 全 28 层 prefill 首 token 对齐
8. MPS KV cache decode
9. MPS lm head + argmax
10. `infer/chat --device mps`
