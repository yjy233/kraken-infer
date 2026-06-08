# Logging And Profiling Design

本文档描述 `kraken-infer` 现有代码基础上，如何补齐：

1. 统一日志体系
2. 逐阶段、逐 token、逐 layer、逐算子耗时统计
3. trace 导出
4. flame graph 生成

目标不是先追求“功能很多”，而是先把一条可落地、可逐步实现、不会和当前 runtime
结构冲突的路径定清楚。

相关代码入口：

- [`apps/kraken_infer_main.cpp`](../apps/kraken_infer_main.cpp)
- [`include/toyllm/runtime/cpu_inference.hpp`](../include/toyllm/runtime/cpu_inference.hpp)
- [`src/runtime/cpu_inference.cpp`](../src/runtime/cpu_inference.cpp)
- [`src/runtime/openai_gateway.cpp`](../src/runtime/openai_gateway.cpp)
- [`src/runtime/cpu/qwen_cpu_model.cpp`](../src/runtime/cpu/qwen_cpu_model.cpp)
- [`src/backends/mps/mps_backend.mm`](../src/backends/mps/mps_backend.mm)
- [`src/runtime/cpu/debug_dump.hpp`](../src/runtime/cpu/debug_dump.hpp)

## 背景

当前项目已经有两类和“可观测性”接近但还不够用的能力：

1. `DebugDumper`
   - 可以把 embedding、Q/K/V、RoPE、attention、MLP、logits 等中间张量导出
   - 主要用于 correctness 对齐
2. `KvCacheStats`
   - 可以输出 KV cache 的层数、head 数、字节数、容量和使用量
   - 主要用于容量观察

但现在缺的东西很明确：

1. 没有统一的 request 级结构化日志
2. 没有 request / prefill / decode / layer / op 级耗时统计
3. 没有 trace 或 flame graph 输出

因此目前遇到的性能问题，很难快速回答下面这些问题：

- CPU path 慢在 `lm_head`，还是慢在 attention？
- MPS path 慢在 kernel 执行，还是慢在 host/device 同步？
- prefill 和 decode 各自占比多少？
- 第几层最慢？
- 每个 operator 的 self time 和 total time 是多少？

## 目标

这套方案的目标：

1. CLI 和 HTTP gateway 共用同一套日志与 profiling 基础设施。
2. 每次请求都有稳定的 `request_id`。
3. 能输出结构化日志，至少带：
   - `request_id`
   - `model_dir`
   - `device`
   - `max_new_tokens`
   - `enable_thinking`
   - 错误信息
4. 能统计：
   - request 级耗时
   - prefill / decode 级耗时
   - token 级耗时
   - layer 级耗时
   - operator 级耗时
5. 能导出：
   - 文本 / JSON summary
   - Chrome trace / Perfetto trace
   - folded stack
   - flame graph SVG
6. HTTP gateway 提供一个本地 profile 查看页面，能直接浏览最近请求的 summary、trace
   和 flame graph。
7. profiling 关闭时，开销尽量低。

## 非目标

第一版不追求：

1. 分布式 tracing
2. OpenTelemetry / Prometheus / ELK 直连
3. 完整替代系统采样 profiler
4. 每个 GPU kernel 内部硬件计数器级分析

这里的定位是：

- 项目内做 deterministic span profiler
- 真要做 CPU 栈级采样或 GPU 硬件级分析，再配合 Instruments

## 现状约束

### Runtime 结构天然适合 span profiling

当前主路径非常清晰：

```text
request
  -> tokenize
  -> reset_kv_cache
  -> prefill: forward_token(prompt tokens)
  -> decode loop:
       compute_logits
       sampling
       forward_token(next_token)
```

这类强串行、边界明确的调用链，非常适合 scoped span。

### CPU 和 MPS 的 operator 粒度不一样

CPU 路径里，一些 operator 天然是函数级边界：

- `rms_norm`
- `qk_norm`
- `apply_rope`
- `matvec`
- `attention`
- `compute_logits`

MPS 路径里，更接近 backend op 或 kernel 边界：

- `embedding_bf16_f32`
- `matvec_bf16_f32_device`
- `rope_f32`
- `attention_f32`
- `silu_mul_f32_in_place`

所以 profiling 必须尊重真实实现粒度，不能在 MPS 路径里伪造出 CPU 那样的细分算子时间。

### 当前 MPS backend 是同步等待风格

`mps_backend.mm` 里大部分 op 都是：

```text
create command buffer
encode
commit
waitUntilCompleted
```

也就是一个 logical op 基本对应一次 command buffer 提交和同步等待。

这意味着第一版 profiling 非常简单：

- 直接在 `MpsContext` 的公开函数外层打 wall-clock span
- 这个 span 就能代表该 MPS op 的端到端耗时

后面如果要继续细分，再补 GPU time。

## 总体设计

建议引入一个按请求生命周期存在的 `RunContext`：

```text
request
  -> RunContext
     -> Logger
     -> Profiler
     -> request metadata
```

逻辑关系：

```text
CLI / HTTP request
  -> build ObservabilityConfig
  -> create RunContext
  -> generate_cpu(request, run_context)
  -> nested logs + spans
  -> flush summary / trace / flamegraph artifacts
```

建议新增的文件：

- `include/toyllm/runtime/logging.hpp`
- `include/toyllm/runtime/profiling.hpp`
- `include/toyllm/runtime/run_context.hpp`
- `src/runtime/logging.cpp`
- `src/runtime/profiling.cpp`
- `tools/profile/trace_to_folded.py`
- `tools/profile/render_flamegraph.sh`

如果想减少文件数，可以先把 `RunContext` 合到 `profiling.hpp` 里。

## 配置设计

建议不要把参数继续散落在 `CpuGenerationRequest` 顶层，而是单独引入配置对象：

```cpp
enum class LogLevel {
  error,
  warn,
  info,
  debug,
  trace,
};

enum class LogFormat {
  text,
  jsonl,
};

enum class ProfileMode {
  off,
  summary,
  trace,
  flamegraph,
  all,
};

struct ObservabilityConfig {
  LogLevel log_level{LogLevel::info};
  LogFormat log_format{LogFormat::text};
  ProfileMode profile_mode{ProfileMode::off};
  std::filesystem::path log_output;
  std::filesystem::path profile_output_dir;
  std::uint64_t min_duration_us{0};
  bool per_layer{true};
  bool per_operator{true};
  bool emit_summary_json{true};
};
```

然后挂到：

```cpp
struct CpuGenerationRequest {
  ...
  ObservabilityConfig observability;
};
```

这样 CLI 和 HTTP gateway 都能走同一套路径。

## 日志体系设计

### 1. 结构化事件

日志建议统一为结构化事件，而不是分散的 `std::cout`。

最小字段：

```text
timestamp
level
component
event
message
request_id
model_dir
device
position
layer
op
duration_us
status
error
```

可以抽象成：

```cpp
struct LogField {
  std::string key;
  std::string value;
};

struct LogEvent {
  std::uint64_t time_ns{0};
  LogLevel level{LogLevel::info};
  std::string component;
  std::string event;
  std::string message;
  std::vector<LogField> fields;
};
```

### 2. Sink

建议先实现两个 sink：

1. text sink
2. jsonl sink

text sink 适合终端直接看：

```text
2026-06-08T10:12:18.520Z INFO runtime request.start request_id=req-08062026-101218-a7k29m device=cpu model=models/qwen3-0.6b prompt_tokens=128 max_new_tokens=32
2026-06-08T10:12:18.742Z INFO profile request.summary request_id=req-08062026-101218-a7k29m total_ms=221.4 tok_s=17.6
```

jsonl sink 适合后处理：

```json
{"ts":"2026-06-08T10:12:18.520Z","level":"info","component":"runtime","event":"request.start","request_id":"req-08062026-101218-a7k29m","device":"cpu","model":"models/qwen3-0.6b","prompt_tokens":"128","max_new_tokens":"32"}
```

### 3. 级别

建议分级：

- `error`
  - request 失败
  - backend 不可用
  - trace/flamegraph 输出失败
- `warn`
  - fallback
  - profile 文件部分缺失
- `info`
  - request start / finish
  - summary
- `debug`
  - prefill/decode 级事件
  - token 级事件
- `trace`
  - operator 级事件

默认不要在 `info` 下写 operator 级日志，否则日志量会过大。

### 4. request_id

每次推理必须生成稳定的 `request_id`。

CLI 可以生成：

```text
cli-<unix_ns>-<pid>
```

HTTP gateway 建议始终生成一个内部 `request_id`，格式固定为：

```text
req-<DDMMYYYY>-<HHMMSS>-<random>
```

例如：

```text
req-08062026-101218-a7k29m
```

其中：

- `DDMMYYYY` 是日月年
- `HHMMSS` 是时分秒
- `random` 是 6 到 8 位小写 base36 随机串

这样做的理由：

1. 目录名天然按时间可读
2. page 跳转和 profile 文件定位稳定
3. 本地排障时看到 `request_id` 就知道大致请求时间

如果客户端传了 `X-Request-Id`，不要覆盖内部 `request_id`，而是额外记录：

```text
client_request_id
```

也就是说：

- `request_id`：gateway 内部主键
- `client_request_id`：外部透传字段

这样日志、trace、summary、profile 目录和 profile 页面都统一围绕 gateway 自己生成的
`request_id` 工作。

## Profiler 设计

### 1. Scoped span

建议用 RAII span：

```cpp
auto span = profiler.scoped("forward_token", {{"position", "17"}});
```

原因：

1. 当前路径是强串行和显式调用
2. 实现简单
3. 可以自然嵌套
4. 很容易生成 trace 和 folded stack

### 2. 数据结构

建议 profiler 内部保存 span 树。

```cpp
struct SpanField {
  std::string key;
  std::string value;
};

struct SpanRecord {
  std::uint32_t id{0};
  std::uint32_t parent_id{0};
  std::string name;
  std::uint64_t start_ns{0};
  std::uint64_t end_ns{0};
  std::uint32_t depth{0};
  std::vector<SpanField> fields;
};
```

时钟统一用：

```cpp
std::chrono::steady_clock
```

不要用 `system_clock` 做 duration。

### 3. total time 和 self time

每个 span 要统计：

- `total_ns`
- `self_ns`

公式：

```text
self_ns = total_ns - sum(child.total_ns)
```

两个数都重要：

- flame graph 更偏 total time
- 找真正热点时 self time 更关键

### 4. Summary 聚合 key

建议按下面维度聚合：

```text
device
phase
op_name
layer(optional)
```

默认 summary 不要带 `position`，否则 token 一多输出会失控。

建议：

- 默认按 `op_name`
- `per_layer=true` 时带上 `layer`
- 需要 token 明细时去看 trace

## 埋点层级

建议分 5 层。

### 1. request 级

- `request.total`
- `request.tokenize`
- `request.reset_kv_cache`
- `request.prefill`
- `request.decode`
- `request.flush_outputs`

### 2. decode step 级

- `decode_step`
- `decode_step.logits`
- `decode_step.sample`
- `decode_step.forward_next_token`

### 3. token 级

对 `forward_token(position)`：

- `forward_token`
- `forward_token.embedding`
- `forward_token.layers`
- `forward_token.final_norm`

### 4. layer 级

- `layer`
- `layer.attention`
- `layer.mlp`

### 5. operator 级

CPU path 建议至少打这些点：

- `embed_lookup`
- `input_rms_norm`
- `q_proj`
- `k_proj`
- `v_proj`
- `q_norm`
- `k_norm`
- `q_rope`
- `k_rope`
- `kv_store`
- `attention.score`
- `attention.softmax`
- `attention.reduce`
- `o_proj`
- `attention_residual_add`
- `post_attention_rms_norm`
- `mlp.gate_proj`
- `mlp.up_proj`
- `mlp.silu_mul`
- `mlp.down_proj`
- `mlp_residual_add`
- `final_rms_norm`
- `lm_head`
- `sampling`
- `token_decode`

MPS path 只能按真实 backend op 打点：

- `mps.embedding_bf16_f32`
- `mps.rmsnorm_f32_bf16`
- `mps.qk_norm_f32_bf16`
- `mps.rope_f32`
- `mps.attention_f32`
- `mps.silu_mul_f32_in_place`
- `mps.copy_from_buffer.logits`
- `mps.matvec_bf16_f32_device.q_proj`
- `mps.matvec_bf16_f32_device.k_proj`
- `mps.matvec_bf16_f32_device.v_proj`
- `mps.matvec_bf16_f32_device.o_proj`
- `mps.matvec_bf16_f32_device.gate_proj`
- `mps.matvec_bf16_f32_device.up_proj`
- `mps.matvec_bf16_f32_device.down_proj`
- `mps.matvec_bf16_f32_device.lm_head`

## CPU 路径的接入位置

`src/runtime/cpu/qwen_cpu_model.cpp` 是主埋点区。

建议优先接这几个函数：

1. `CpuQwenModel::generate`
2. `CpuQwenModel::generate_mps`
3. `forward_token`
4. `apply_layer`
5. `attention`
6. `compute_logits`
7. sampling 相关函数

对应到 operator 级别，大致是：

```text
generate
  -> request.tokenize
  -> request.reset_kv_cache
  -> request.prefill
     -> forward_token
        -> forward_token.embedding
        -> layer
           -> layer.attention
              -> input_rms_norm
              -> q_proj / k_proj / v_proj
              -> q_norm / k_norm
              -> q_rope / k_rope
              -> kv_store
              -> attention.score
              -> attention.softmax
              -> attention.reduce
              -> o_proj
              -> attention_residual_add
           -> layer.mlp
              -> post_attention_rms_norm
              -> mlp.gate_proj / mlp.up_proj
              -> mlp.silu_mul
              -> mlp.down_proj
              -> mlp_residual_add
        -> forward_token.final_norm
  -> request.decode
     -> decode_step
        -> decode_step.logits
        -> decode_step.sample
        -> decode_step.forward_next_token
```

`DebugDumper` 现有的 dump 边界和这些 span 边界基本一致，后续实现时可以直接沿用相同位置。

## MPS 路径的接入位置

### 1. `generate_mps()` 和 `apply_layer_mps()`

先在 model 语义层打上高层 span：

- `forward_token`
- `layer`
- `layer.attention`
- `layer.mlp`

这样 summary 至少可以对齐 CPU 路径的大结构。

### 2. `MpsContext` 公开 op

然后在 `src/backends/mps/mps_backend.mm` 的公开方法外围打 backend span：

- `matvec_bf16_f32_device`
- `rope_f32`
- `attention_f32`
- `silu_mul_f32_in_place`
- `copy_f32_region`
- `copy_from_buffer`

由于当前每个 op 基本都是单独的：

```text
command buffer
  -> encode
  -> commit
  -> waitUntilCompleted
```

所以这层 wall-clock span 非常有意义，它反映的就是该 MPS op 的实际端到端 latency。

### 3. host_us 和 gpu_us

第一版先统计：

- `host_us`

第二版再补：

- `gpu_us`

也就是 summary 最终最好能长这样：

```text
mps.attention_f32:
  host_total_ms = 31.8
  gpu_total_ms = 24.6
```

但在第一版实现之前，不要在文档或输出里假装已经拿到了 GPU time。

## Flame Graph 方案

### 1. 需要同时有 trace 和 flamegraph

两者解决的问题不同：

- trace 看时间线和顺序
- flame graph 看累计热点

所以建议同时导出：

- `trace.json`
- `profile.folded`

### 2. `trace.json`

建议用 Chrome Trace Event 格式，后续可直接在：

- Perfetto
- Chrome trace viewer
- SpeedScope

里打开。

单个 span 可以序列化为：

```json
{
  "name": "attention.score",
  "cat": "operator",
  "ph": "X",
  "ts": 123456789,
  "dur": 245,
  "pid": 1,
  "tid": 1,
  "args": {
    "layer": 12,
    "position": 37,
    "device": "cpu"
  }
}
```

建议 trace 输出单位用微秒。

### 3. `profile.folded`

标准 flame graph 需要 folded stack：

```text
request.total;request.decode;decode_step;forward_token;layer_12;layer.attention;attention.score 245
request.total;request.decode;decode_step;forward_token;layer_12;layer.attention;attention.softmax 18
request.total;request.decode;decode_step;forward_token;layer_12;layer.attention;attention.reduce 53
```

然后生成 SVG：

```bash
flamegraph.pl profile.folded > profile.svg
```

运行时建议这样做：

1. 总是先导出 `profile.folded`
2. 如果本机存在 `flamegraph.pl`，再生成 `profile.svg`
3. 如果没有该工具，不影响主流程

这样不会把外部依赖硬绑到主运行路径。

### 4. 为什么不用采样火焰图替代

采样型火焰图适合系统级热点分析，但不适合直接替代项目内 span flame graph。

原因：

1. 很难自然带出 `layer`、`position`、`prefill/decode` 语义
2. 对 MPS kernel 和 host 同步边界不够直观
3. 难以稳定对齐项目自己的 operator 名称

因此建议：

- 项目内先做 span -> folded stack
- 如果要做系统级验证，再用 Instruments 补充

## CLI 和 Gateway 配置入口

### CLI

在 `apps/kraken_infer_main.cpp` 增加参数：

- `--log-level error|warn|info|debug|trace`
- `--log-format text|json`
- `--log-file path`
- `--profile off|summary|trace|flamegraph|all`
- `--profile-dir path`
- `--profile-min-us N`

### HTTP Gateway

在 `src/runtime/openai_gateway.cpp` 里：

1. 进入请求时生成内部 `request_id`
2. 写 `request.start`
3. 构造 `CpuGenerationRequest.observability`
4. 请求结束时写 `request.finish`
5. 如果 profile 开启，把产物写到配置目录

如果后面要开放 HTTP 控制项，建议只开放：

- `x-kraken-profile: summary|trace|all`

不要一开始就把所有 profiling 细节暴露到公开 OpenAI-compatible schema 里。

### Profile Page

gateway 还应新增一个本地 profile 查看页面：

- `GET /profile_page`

这个页面不是 OpenAI-compatible API 的一部分，而是本地调试工具页，定位和现有
`/chat_page` 一样。

建议同时提供这些只读接口：

- `GET /profiles/index.json`
- `GET /profiles/<request_id>/summary.json`
- `GET /profiles/<request_id>/trace.json`
- `GET /profiles/<request_id>/profile.folded`
- `GET /profiles/<request_id>/profile.svg`

其中：

- `/profile_page` 负责前端展示
- `/profiles/index.json` 返回最近请求列表
- 其余路由返回单个 `request_id` 的 profile 产物

`/profile_page` 建议支持：

1. 左侧最近请求列表
2. 按 `request_id`、时间、device、model 过滤
3. 右侧 summary 面板
4. flame graph 预览
5. trace 文件下载或跳转
6. 在 summary-only 模式下提示该请求没有 `trace.json` 或 `profile.svg`

页面默认可以：

- 打开最近一次 profile
- 或通过 `GET /profile_page?request_id=<id>` 直接定位某个请求

因为项目当前是本地顺序 gateway，这个页面可以直接基于 profile 输出目录扫描结果生成，
不需要单独数据库。

## 建议的输出目录结构

建议一次请求落到：

```text
<profile_output_dir>/<request_id>/
  manifest.json
  summary.txt
  summary.json
  trace.json
  profile.folded
  profile.svg
```

其中：

- `manifest.json` 记录最小元信息，给 `/profiles/index.json` 和 `/profile_page` 用
- `summary.txt` 直接给人看
- `summary.json` 给脚本或 UI 消费
- `trace.json` 给 Perfetto/SpeedScope
- `profile.folded` 是 flame graph 输入
- `profile.svg` 是可选派生产物

`manifest.json` 建议至少包含：

```json
{
  "request_id": "req-08062026-101218-a7k29m",
  "client_request_id": "external-123",
  "created_at": "2026-06-08T10:12:18Z",
  "device": "mps",
  "model_dir": "models/qwen3-0.6b",
  "profile_mode": "all",
  "has_trace": true,
  "has_flamegraph": true,
  "status": "ok"
}
```

## Summary 输出建议

建议 `summary.txt` 至少包含：

```text
request_id: req-08062026-101218-a7k29m
device: cpu
model_dir: models/qwen3-0.6b
prompt_tokens: 128
generated_tokens: 32
total_ms: 221.4
prefill_ms: 133.8
decode_ms: 85.1
tok_s: 17.6

top operators by self time:
1. lm_head                  48.2 ms
2. attention.score          41.7 ms
3. q_proj                   22.4 ms
4. attention.reduce         19.8 ms
5. mlp.down_proj            17.1 ms
```

如果 `per_layer=true`，再附加：

```text
top layers by total time:
1. layer_12  11.2 ms
2. layer_13  10.9 ms
3. layer_11  10.8 ms
```

## 推荐实现顺序

### Phase 1: 统一日志

目标：

- `Logger`
- `request_id`
- text/jsonl sink
- CLI / gateway 共用配置

先不做 operator profiling。

### Phase 2: Summary profiler

目标：

- `ScopedSpan`
- request / token / layer / op 级耗时
- `summary.txt`
- `summary.json`

这一阶段已经足够支撑大部分性能分析。

### Phase 3: Trace 和 folded stack

目标：

- `trace.json`
- `profile.folded`
- 可选的 `profile.svg`

到这一步，flame graph 就有了。

### Phase 4: MPS GPU time

目标：

- `host_us`
- `gpu_us`

真正把 host 等待和 device 执行拆开。

### Phase 5: 开销优化

目标：

- profiling off 时接近零开销
- trace 模式下减少 string 拼装和小对象分配
- 控制大请求下的 profile 文件体积

## 测试建议

至少补这些测试：

1. profiling 关闭时，生成结果不变
2. summary 模式下，layer 数和 token 数统计符合预期
3. `trace.json` 是合法 JSON
4. `profile.folded` 行格式合法
5. CPU 和 MPS 两条路径都能产出文件
6. 请求失败时也能产出失败日志
7. `/profiles/index.json` 和 `/profile_page` 在有 profile 产物时能正常展示

还可以补一个很实用的回归测试：

- 固定 prompt 和 `max_new_tokens=1`
- 检查 summary 里至少出现：
  - `request.total`
  - `request.prefill`
  - `decode_step.logits`
  - `lm_head`

## 结论

最合适的路线不是先引入系统级 profiler，而是先在项目内部做一套轻量但结构清晰的
span-based logging/profiling：

1. `Logger` 负责结构化事件
2. `Profiler` 负责嵌套 span
3. `summary` 负责快速找热点
4. `trace.json` 负责看时间线
5. `profile.folded` 和 `profile.svg` 负责看 flame graph

这条路线和当前 `qwen_cpu_model.cpp` 的 forward 结构、`openai_gateway.cpp` 的请求入口、
以及 `mps_backend.mm` 的同步 op 风格都能自然对齐，实现风险最小。
