# M5 Task: Basic KV Cache

目标：把 CPU decode 阶段的 K/V cache 从临时 vector 正式抽象成可复用、可校验、可观测的组件，避免每生成一个 token 都重复计算完整 prompt。

## Scope

本阶段只覆盖 CPU backend。MPS device-resident KV cache 后续在 MPS forward path 中实现。

## Tasks

### KV Cache Abstraction

- [x] 新增 `src/runtime/cpu/kv_cache.hpp`
- [x] 新增 `src/runtime/cpu/kv_cache.cpp`
- [x] 定义 `KvCache`
- [x] 定义 `KvCacheStats`
- [x] 支持 reset cache
- [x] 支持 K cache
- [x] 支持 V cache
- [x] cache layout 为 `[num_layers, max_seq_len, num_key_value_heads * head_dim]`
- [x] 支持按 layer/position/kv_head 查询 K 指针
- [x] 支持按 layer/position/kv_head 查询 V 指针
- [x] 支持 position/layer/head 越界校验
- [x] 支持 K/V vector size 校验
- [x] 支持 allocation size overflow 校验

### Prefill And Decode

- [x] prefill 阶段逐 token 写入每层 K/V
- [x] decode 阶段每步只 forward 新 token
- [x] decode 阶段追加当前 token K/V
- [x] attention 读取 `[0, current_position]` 范围
- [x] 超过 cache capacity 时返回明确错误
- [x] 每次 generation request 重置 cache

### Observability

- [x] `CpuGenerationResult` 返回 KV cache report
- [x] CLI 增加 `--kv-cache-stats`
- [x] CLI 增加 `--verify-kv-cache`
- [x] 输出 layers、kv_heads、head_dim、kv_dim
- [x] 输出 capacity tokens
- [x] 输出 used tokens
- [x] 输出 key bytes、value bytes、total bytes

### Verification

- [x] `make test` 通过
- [x] `infer --kv-cache-stats` 能输出 cache stats
- [x] `infer --verify-kv-cache` 能校验 cached decode 与 full-prefix recompute 输出一致
- [x] `infer --dump-dir` 能导出 KV cache 参与的中间 tensor 和 logits

## Acceptance

- [x] CPU decode 使用 KV cache
- [x] KV cache 不在每个 decode step 重新分配
- [x] decode 每步只处理一个新 token
- [x] cached decode 输出与 full-prefix recompute 输出一致
- [x] 当前 seq length、capacity、used bytes 可观测
- [x] 超过 cache capacity 时明确报错

## Known Limitations

- [ ] 还没有 MPS device-resident KV cache
- [ ] 还没有上下文窗口裁剪/滑窗策略
