# Logging And Profiling Task

目标：为 `kraken-infer` 增加一套可落地的 observability 基础设施，覆盖结构化日志、
逐阶段与逐算子耗时统计、trace / flame graph 导出，以及 gateway 内置 profile 查看页。

设计说明见 [`../logging-and-profiling.md`](../logging-and-profiling.md)。

## Current Status

- [x] 已完成 logging / profiling 技术方案文档
- [x] 已明确 gateway 内部 `request_id` 规则：
  `req-<DDMMYYYY>-<HHMMSS>-<random>`
- [x] 已明确 profile 输出目录结构和 `manifest.json`
- [x] 已明确 gateway `GET /profile_page` 和 `/profiles/*` 路由规划
- [x] 已实现 `ObservabilityConfig`、`ProfileMode`、`RequestProfiler`
- [x] 已实现 CPU path request / token / layer / attention / lm_head 埋点
- [x] 已实现 `summary.txt`、`summary.json`、`trace.json`、`profile.folded`
- [x] 已实现 gateway `X-Request-Id` 回传、`/profile_page`、`/profiles/index.json`
- [x] 已将 profile 页面前端拆分到 `web/profile_page.{html,css,js}`
- [x] `serve` 已支持 `--profile`、`--profile-dir`、`--profile-min-us`
- [x] `infer/run/chat` 已接入 `--profile`、`--profile-dir`、`--profile-min-us`
- [ ] 代码侧尚未实现统一 `Logger`
- [x] 代码侧已实现统一 `Profiler`
- [x] 代码侧已实现 profile 文件输出
- [x] gateway 已实现 profile 页面

## Scope

本任务范围包括：

1. 统一日志配置与输出
2. 统一 request 级上下文和 `request_id`
3. span-based profiler
4. CPU path 逐层、逐算子埋点
5. MPS path backend op 级埋点
6. profile summary / trace / folded stack / flame graph 输出
7. gateway profile 浏览页面与只读接口

本任务不包括：

- OpenTelemetry / Prometheus / ELK 直连
- 并发生产级 profile 存储系统
- 完整 GPU 硬件计数器级 profiling

## Tasks

### Configuration And Public API

- [ ] 在 `include/toyllm/runtime/` 下新增 logging / profiling 公共头文件
- [ ] 新增 `ObservabilityConfig`
- [ ] 在 `CpuGenerationRequest` 中挂载 `observability`
- [ ] 保持默认配置为 logging/profiling 关闭或低开销模式
- [ ] 保持不配置 observability 时现有 CLI / gateway 行为不变

### Logger Foundation

- [ ] 实现 `LogLevel`
- [ ] 实现 `LogFormat`
- [ ] 实现结构化 `LogEvent`
- [ ] 实现 text sink
- [ ] 实现 JSONL sink
- [ ] 支持写到 stderr / stdout
- [ ] 支持写到文件
- [ ] 支持最小公共字段：`request_id`、`device`、`model_dir`、`event`、`status`

### Request Context And Request ID

- [ ] 新增 request 生命周期 `RunContext`
- [ ] `RunContext` 持有 `Logger`
- [ ] `RunContext` 持有 `Profiler`
- [ ] CLI 请求生成内部 `request_id`
- [ ] gateway 请求始终生成内部 `request_id`
- [ ] gateway `request_id` 格式实现为 `req-<DDMMYYYY>-<HHMMSS>-<random>`
- [ ] `random` 使用 6 到 8 位小写 base36 随机串
- [ ] gateway 若收到 `X-Request-Id`，记录为 `client_request_id`
- [ ] 内部 `request_id` 不被外部 header 覆盖

### CLI Integration

- [ ] `kraken-infer infer/run/chat/serve` 支持 `--log-level`
- [ ] 支持 `--log-format`
- [ ] 支持 `--log-file`
- [ ] 支持 `--profile off|summary|trace|flamegraph|all`
- [ ] 支持 `--profile-dir`
- [ ] 支持 `--profile-min-us`
- [ ] `help` 输出新增 observability 参数

### CPU Runtime Profiling

- [ ] 实现 `ScopedSpan`
- [ ] 实现 request 级 span
- [ ] 实现 prefill / decode 级 span
- [ ] 在 `forward_token()` 增加 token 级 span
- [ ] 在 `apply_layer()` 增加 layer 级 span
- [ ] 为 `input_rms_norm` 增加 span
- [ ] 为 `q_proj / k_proj / v_proj` 增加 span
- [ ] 为 `q_norm / k_norm` 增加 span
- [ ] 为 `q_rope / k_rope` 增加 span
- [ ] 为 `kv_store` 增加 span
- [ ] 为 `attention.score` 增加 span
- [ ] 为 `attention.softmax` 增加 span
- [ ] 为 `attention.reduce` 增加 span
- [ ] 为 `o_proj` 和 attention residual add 增加 span
- [ ] 为 MLP gate/up/down 和 `silu_mul` 增加 span
- [ ] 为 `final_rms_norm` 增加 span
- [ ] 为 `compute_logits` / `lm_head` 增加 span
- [ ] 为 sampling 增加 span
- [ ] 为 token decode / stream emit 增加 span

### MPS Runtime Profiling

- [ ] 在 `generate_mps()` 增加 request / prefill / decode 级 span
- [ ] 在 `apply_layer_mps()` 增加 layer 级 span
- [ ] 在 `MpsContext::embedding_bf16_f32` 外层增加 span
- [ ] 在 `MpsContext::matvec_bf16_f32_device` 外层增加 span
- [ ] 在 `MpsContext::rope_f32` 外层增加 span
- [ ] 在 `MpsContext::attention_f32` 外层增加 span
- [ ] 在 `MpsContext::silu_mul_f32_in_place` 外层增加 span
- [ ] 在 `copy_f32_region` 外层增加 span
- [ ] 在 `copy_from_buffer` 外层增加 span
- [ ] 第一版只统计 MPS host wall time
- [ ] 输出上明确区分“暂未实现 GPU time”

### Summary Aggregation

- [ ] 支持 span tree 收集
- [ ] 支持 `total_ns`
- [ ] 支持 `self_ns`
- [ ] 支持按 `op_name` 聚合 summary
- [ ] 支持按 layer 聚合 summary
- [ ] 支持 request 级 summary
- [ ] 输出 `summary.txt`
- [ ] 输出 `summary.json`
- [ ] summary 中包含 token/s、prefill_ms、decode_ms、generated_tokens
- [ ] summary 中列出 top operators by self time

### Trace And Flame Graph Artifacts

- [ ] 输出 `trace.json`
- [ ] `trace.json` 使用 Chrome Trace Event 兼容格式
- [ ] 输出 `profile.folded`
- [ ] 若系统存在 `flamegraph.pl`，生成 `profile.svg`
- [ ] 若不存在 `flamegraph.pl`，主流程不失败
- [ ] 支持 `profile_mode=summary` 时不生成 trace / flamegraph 文件
- [ ] 支持 `profile_mode=all` 时完整生成全部产物

### Profile Output Layout

- [ ] profile 输出目录按 `<profile_output_dir>/<request_id>/` 组织
- [ ] 输出 `manifest.json`
- [ ] `manifest.json` 包含 `request_id`
- [ ] `manifest.json` 包含 `client_request_id`
- [ ] `manifest.json` 包含 `created_at`
- [ ] `manifest.json` 包含 `device`
- [ ] `manifest.json` 包含 `model_dir`
- [ ] `manifest.json` 包含 `profile_mode`
- [ ] `manifest.json` 包含 `has_trace`
- [ ] `manifest.json` 包含 `has_flamegraph`
- [ ] `manifest.json` 包含 `status`

### Gateway Integration

- [ ] gateway 请求开始记录 `request.start`
- [ ] gateway 请求结束记录 `request.finish`
- [ ] gateway 错误路径记录 `request.error`
- [ ] gateway 将内部 `request_id` 回传到响应 header
- [ ] gateway 支持 `x-kraken-profile: summary|trace|all`
- [ ] gateway 将 profile 配置下传到 `CpuGenerationRequest`
- [ ] gateway 生成的 profile 产物按 `request_id` 落盘

### Profile Read API

- [ ] 实现 `GET /profile_page`
- [ ] 实现 `GET /profiles/index.json`
- [ ] 实现 `GET /profiles/<request_id>/summary.json`
- [ ] 实现 `GET /profiles/<request_id>/trace.json`
- [ ] 实现 `GET /profiles/<request_id>/profile.folded`
- [ ] 实现 `GET /profiles/<request_id>/profile.svg`
- [ ] `request_id` 不存在时返回 404
- [ ] profile 文件缺失时返回明确错误 JSON 或合理降级

### Profile Page UI

- [ ] 将 `/profile_page` 前端文件从 `openai_gateway.cpp` 中拆出
- [ ] 页面展示最近 profile 请求列表
- [ ] 支持按 `request_id` 过滤
- [ ] 支持按 `device` 过滤
- [ ] 支持按 `model` 过滤
- [ ] 默认打开最近一次 profile
- [ ] 支持 `?request_id=<id>` 直接打开指定请求
- [ ] 展示 summary 核心指标
- [ ] 展示 top operators
- [ ] 支持内嵌 flame graph SVG 预览
- [ ] 支持 trace / folded / summary 下载链接
- [ ] 在 summary-only 模式下正确提示 trace / flamegraph 不可用

### Tests

- [ ] 为 logging / profiling 新增 focused tests
- [ ] profiling 关闭时结果输出与现有路径一致
- [ ] CPU path 至少生成 `summary.txt`
- [ ] `trace.json` 是合法 JSON
- [ ] `profile.folded` 行格式合法
- [ ] `manifest.json` 字段完整
- [ ] gateway `/profiles/index.json` 返回最近请求列表
- [ ] gateway `/profile_page` 能渲染至少一个 profile
- [ ] MPS path 在可用机器上能产出 profile summary

## Acceptance

- [ ] `kraken-infer help` 显示 observability 相关参数
- [ ] `infer --profile summary` 可生成 `summary.txt` 和 `summary.json`
- [ ] `infer --profile all` 可生成 `trace.json`、`profile.folded`
- [ ] 本机存在 `flamegraph.pl` 时可额外生成 `profile.svg`
- [ ] CPU path summary 能看到 `lm_head`、attention、MLP 相关项
- [ ] gateway 响应日志和 profile 目录都能按内部 `request_id` 对齐
- [ ] gateway 能访问 `/profiles/index.json`
- [ ] gateway 能访问 `/profile_page`
- [ ] `/profile_page` 能打开最近一次 profile 并显示 summary
- [ ] 错误请求也会留下失败日志和 `manifest.json`

## Known Risks

- profiling 过细时会放大 logging / span 管理开销
- trace 模式文件量可能很大，长对话下要注意磁盘占用
- 当前 MPS backend 是同步 `waitUntilCompleted` 风格，第一版只能稳定得到 host wall time
- 若 `profile.svg` 在服务端直接渲染过大，请求页面时需要避免一次性加载过多历史文件
