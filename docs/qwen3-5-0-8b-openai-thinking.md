# Qwen3.5 0.8B OpenAI-compatible thinking 技术方案

本文档记录 Qwen3.5 0.8B 在本仓库 OpenAI-compatible HTTP gateway 下的 thinking 控制方案。这里的 OpenAI-compatible 指 `/v1/chat/completions` 兼容接口；`/openapi.json` 只是该接口的机器可读 schema。

## 结论

Qwen3.5 thinking 的请求开关应以 `chat_template_kwargs.enable_thinking` 为主：

```json
{
  "model": "qwen3.5-0.8b",
  "messages": [{"role": "user", "content": "hello"}],
  "max_completion_tokens": 128,
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

本仓库已经兼容顶层 `enable_thinking`：

```json
{
  "model": "qwen3.5-0.8b",
  "messages": [{"role": "user", "content": "hello"}],
  "enable_thinking": false
}
```

字段优先级建议固定为：

1. `chat_template_kwargs.enable_thinking` 优先级最高，用于对齐 llama.cpp server 的请求写法。
2. 顶层 `enable_thinking` 作为本仓库已有扩展继续保留，便于旧客户端和 `/chat_page` 使用。
3. 两者都不存在时保持本仓库当前默认值 `false`，避免改变既有 CLI/gateway 默认输出风格。

`reasoning_format` 不负责打开或关闭 thinking。它只负责响应解析和返回形态：是否把 `<think>...</think>` 从 `message.content` 中拆到 `message.reasoning_content`。

## llama.cpp 参考结论

参考树：`/Users/bill/code/llama.cpp`。

关键参考点：

- `tools/server/README.md`: `/v1/chat/completions` 文档把 `chat_template_kwargs` 作为模板参数入口，并给出 `{"enable_thinking": false}` 示例。
- `tools/server/server-common.cpp`: 合并 server 默认 `chat_template_kwargs` 和请求体 `chat_template_kwargs`，再读取其中的 `enable_thinking` 覆盖 `inputs.enable_thinking`。
- `common/arg.cpp`: CLI `--reasoning on/off/auto` 会设置默认模板参数 `enable_thinking=true/false`；直接通过 `--chat-template-kwargs` 设置 `enable_thinking` 已被标记为 CLI 侧旧写法。
- `models/templates/Qwen3.5-4B.jinja`: Qwen3.5 模板在 `add_generation_prompt` 时读取 `enable_thinking`：
  - `enable_thinking == false`: 追加空 thinking block。
  - 其他情况: 追加打开的 `<think>\n`，让模型继续生成 thinking。
- `common/chat.cpp` 和 `tools/server/server-chat.cpp`: `reasoning_format != none` 时，解析出的 thinking 会进入 `reasoning_content`；streaming delta 使用 `delta.reasoning_content`。

因此，本仓库要对齐的是两个独立层次：

- prompt 控制层：`chat_template_kwargs.enable_thinking`。
- 响应解析层：`reasoning_format`。

## 当前本仓库状态

当前已经具备 prompt 控制层和响应解析层：

- `src/runtime/openai_gateway.cpp`:
  - 解析顶层 `enable_thinking`。
  - 解析 `chat_template_kwargs.enable_thinking`，并让 nested 字段覆盖顶层字段。
  - 解析 chat 请求的 `reasoning_format`，支持 `none`、`auto`、`deepseek`。
  - 将结果写入 `CpuGenerationRequest::enable_thinking`。
  - 非流式响应在 `reasoning_format != none` 时返回 `message.reasoning_content`。
  - 流式响应在 `reasoning_format != none` 时返回 `delta.reasoning_content`。
- `src/runtime/reasoning_parser.cpp`:
  - 负责 Qwen/DeepSeek-style `<think>...</think>` 拆分。
  - 支持 Qwen3.5 generation prompt 已预填 `<think>\n` 的隐式 thinking 状态。
  - 支持 streaming chunk 内或跨 chunk 的 `<think>` / `</think>` 标签识别。
- `src/runtime/gguf_tokenizer.cpp`:
  - `format_qwen35_chat_prompt(..., enable_thinking)` 根据开关写 Qwen3.5 chat prompt。
- `src/runtime/qwen35_runtime.cpp`:
  - Qwen3.5 GGUF chat 请求调用 `format_qwen35_chat_prompt(..., request.enable_thinking)`。
- `tests/smoke_test.cpp`:
  - 已覆盖 `enable_thinking=false` 和 `true` 生成的 Qwen3.5 prompt token 序列。
  - 已覆盖非流式 reasoning 拆分、隐式 Qwen thinking 状态、跨 chunk streaming 标签拆分和非法 `reasoning_format`。

## 请求字段设计

| 字段 | 类型 | 状态 | 语义 |
| --- | --- | --- | --- |
| `chat_template_kwargs.enable_thinking` | boolean | 推荐 | Qwen3.5 thinking 的主控制字段，传入 chat template。 |
| `enable_thinking` | boolean | 保留 | 本仓库已有顶层扩展；被 `chat_template_kwargs.enable_thinking` 覆盖。 |
| `reasoning_format` | string | 已实现 | 响应解析格式，不控制 prompt。支持 `none`、`auto`、`deepseek`。 |
| `thinking_budget_tokens` | integer | 后续 | llama.cpp 支持的 thinking token budget。第一版不做。 |
| `reasoning_control` | boolean | 后续 | llama.cpp 支持的流式实时结束 thinking。第一版不做。 |
| `reasoning_effort` | string | 不作为主字段 | OpenAI 风格客户端可能会发，但 Qwen3.5 Jinja 模板不靠它控制 thinking。需要兼容时只能作为别名层，不能替代 `enable_thinking`。 |

`chat_template_kwargs.enable_thinking` 必须是 JSON boolean。不要接受字符串 `"true"` / `"false"`，否则和 llama.cpp 的类型约束不一致，也容易让 OpenAPI schema 失真。

## Prompt 行为

Qwen3.5 chat prompt 基本结构：

```text
<|im_start|>user
hello<|im_end|>
<|im_start|>assistant
```

`enable_thinking=false` 时追加：

```text
<think>

</think>

```

这表示 prompt 中已经给了一个空 thinking block，模型应从最终回答继续。

`enable_thinking=true` 时追加：

```text
<think>
```

这表示 prompt 停在 thinking block 内，模型输出通常会先生成 reasoning，再生成 `</think>` 和最终回答。

重要约束：runtime 返回给 gateway 的文本通常只包含模型新生成部分，不包含 prompt 中预填的 `<think>\n`。因此 `reasoning_format` 解析不能假设输出文本一定带有起始 `<think>`。当 `enable_thinking=true` 且 `reasoning_format != none` 时，解析器初始状态视为已经在 thinking block 内，直到遇到 `</think>`。

## 响应格式目标

### `reasoning_format=none`

不解析 thinking，保持当前行为。非流式：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "raw generated text, possibly with <think>...</think>"
    }
  }]
}
```

流式：

```json
{"choices":[{"delta":{"content":"raw chunk"}}]}
```

### `reasoning_format=auto` 或 `deepseek`

解析 `<think>...</think>`，输出 OpenAI-compatible 扩展字段 `reasoning_content`。非流式：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "model thinking text",
      "content": "final answer"
    }
  }]
}
```

流式：

```json
{"choices":[{"delta":{"reasoning_content":"thinking chunk"}}]}
{"choices":[{"delta":{"content":"final answer chunk"}}]}
```

这与 llama.cpp Chat Completions 响应保持一致：`common_chat_msg::to_json_oaicompat()` 在 message 上写 `reasoning_content`，`server_chat_msg_diff_to_json_oaicompat()` 在 delta 上写 `reasoning_content`。

## 实现方案

### 阶段 1: 固化请求兼容性

本阶段已完成：

- `/v1/chat/completions` schema 明确：
  - 顶层 `enable_thinking`。
  - `chat_template_kwargs.enable_thinking`。
  - nested 字段覆盖顶层字段。
- `/v1/completions` 可以保留相同字段，但 Qwen3.5 thinking 主要面向 chat prompt。
- smoke test 增加 gateway 请求解析覆盖。如果 parse 函数继续留在 anonymous namespace，可以通过最小 HTTP 测试或抽出小的请求解析 helper 测试。

### 阶段 2: 增加 `reasoning_format`

本阶段已完成。

新增内部枚举：

```cpp
enum class ReasoningFormat {
  none,
  auto_,
  deepseek,
};
```

解析规则：

- 默认 `none`，保持现有响应不变。
- `auto` 第一版等同 `deepseek`。
- 非法值返回 `400 invalid_request_error`。

`ChatRequest` 增加 `reasoning_format`，并传到响应构造层。`CpuGenerationRequest` 不一定需要知道该字段，因为它不影响 forward，只影响 gateway 输出解析。

### 阶段 3: 非流式 thinking parser

本阶段已完成。

实现一个小的 Qwen/DeepSeek-style parser：

输入：

- `generated_text`
- `enable_thinking`
- `reasoning_format`

输出：

- `reasoning_content`
- `content`

规则：

- `reasoning_format=none`: 原样返回 content。
- `enable_thinking=true`: 初始状态为 `inside_thinking`，读取到第一个 `</think>` 前都进入 reasoning。
- `enable_thinking=false`: 初始状态为 `content`，但如果输出显式出现 `<think>`，仍可捕获到 matching `</think>` 之间的内容。
- 找不到 `</think>` 时，已捕获内容都留在 `reasoning_content`，`content` 为空或为标签后的文本。
- 多个 thinking block 第一版只解析第一个，后续文本全部作为 content；这和 Qwen3.5 正常输出预期一致。

### 阶段 4: 流式 thinking parser

本阶段已完成。

流式 parser 需要保存状态：

- `mode`: `inside_thinking` 或 `content`。
- `pending_suffix`: 缓存可能跨 chunk 的 `<think>` / `</think>` 标签尾部。
- `emitted_reasoning` / `emitted_content`: 用于测试和调试，不一定进入生产结构。

输出 chunk：

- thinking 模式输出 `delta.reasoning_content`。
- content 模式输出 `delta.content`。
- 标签本身不输出。

注意 UTF-8：当前 token streaming 是按 token 解码。parser 不应截断 UTF-8 字节序列；可以复用或新增一个 `validate_utf8` 风格 helper，只发送完整 UTF-8 前缀，剩余字节放入 pending。

### 阶段 5: `thinking_budget_tokens` 和 `reasoning_control`

第一版不实现。原因：

- `thinking_budget_tokens` 需要 sampler 层识别 thinking start/end token 序列，并在 budget 用尽时强制注入 end tag 或 message。
- `reasoning_control` 需要流式请求可被另一路 HTTP 请求命中并修改正在运行的生成状态；本仓库当前 gateway 是顺序 POSIX HTTP server，不适合直接搬 llama.cpp 的实时控制接口。

这两个字段可以先不进 OpenAPI schema，避免客户端误以为可用。

## OpenAPI schema 要求

`/openapi.json` 中 `/v1/chat/completions` 应包含：

```json
{
  "enable_thinking": {
    "type": "boolean",
    "description": "Local extension. Controls Qwen thinking prompt formatting. Overridden by chat_template_kwargs.enable_thinking."
  },
  "chat_template_kwargs": {
    "type": "object",
    "properties": {
      "enable_thinking": {
        "type": "boolean",
        "description": "Preferred Qwen thinking switch passed to the chat template."
      }
    },
    "additionalProperties": true
  },
  "reasoning_format": {
    "type": "string",
    "enum": ["none", "auto", "deepseek"],
    "description": "Controls parsing of generated thinking into reasoning_content. Does not enable thinking."
  }
}
```

当前 schema 已列出 `enable_thinking`、`chat_template_kwargs.enable_thinking` 和 `reasoning_format`。

## 验收标准

- `chat_template_kwargs.enable_thinking=false` 生成的 Qwen3.5 prompt 以空 thinking block 结束。
- `chat_template_kwargs.enable_thinking=true` 生成的 Qwen3.5 prompt 以打开的 `<think>\n` 结束。
- 同时传顶层 `enable_thinking=true` 和 nested `chat_template_kwargs.enable_thinking=false` 时，最终按 `false` 处理。
- `reasoning_format=none` 时，非流式和流式响应均不出现 `reasoning_content`。
- `reasoning_format=deepseek` 或 `auto` 时：
  - 非流式响应 `message.reasoning_content` 包含 thinking，`message.content` 只包含最终回答。
  - 流式响应先发送 `delta.reasoning_content`，再发送 `delta.content`。
- `reasoning_format` 非法值返回 OpenAI-style 400 error。
- `ctest --preset debug` 通过。

## 推荐 curl

关闭 thinking：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [{"role": "user", "content": "hello"}],
    "max_completion_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

打开 thinking，并在后续实现 parser 后拆分返回：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [{"role": "user", "content": "hello"}],
    "max_completion_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": true},
    "reasoning_format": "deepseek"
  }'
```
