# M9 Task: Inference Gateway And OpenAI-Compatible Protocol

目标：提供本地 OpenAI-compatible HTTP inference gateway，让现有 Qwen3 CPU/MPS runtime 可以通过标准 `/v1/*` 接口被客户端调用，并提供可机器读取的 OpenAPI schema。

## Scope

M9 实现本项目需要的 OpenAI-compatible 子集：模型列表、legacy text completions、chat completions、SSE streaming、基础 tools/tool_choice 协议、错误 JSON、OpenAPI schema、CLI server 启动参数。Gateway 不执行外部工具；tool calling 的职责是按 OpenAI 响应格式返回 tool call，由上层客户端执行工具后再把 `role: "tool"` 消息发回。

## Tasks

### Server Foundation

- [x] 新增 `toyllm serve` CLI
- [x] 支持 `--host`
- [x] 支持 `--port`
- [x] 支持 `--model`
- [x] 支持 `--model-id`
- [x] 支持 `--device cpu|mps`
- [x] 支持 `--max-new-tokens`
- [x] 实现顺序 POSIX HTTP server
- [x] 实现 request line、headers、Content-Length body 解析
- [x] 返回 `Connection: close`
- [x] 实现 `/health` 和 `/v1/health`
- [x] 实现 `/openapi.json` 和 `/v1/openapi.json`

### OpenAI-Compatible Routes

- [x] `GET /v1/models`
- [x] `POST /v1/completions`
- [x] `POST /v1/chat/completions`
- [x] 非流式 `text_completion` JSON 响应
- [x] 流式 `text_completion` SSE 响应
- [x] 非流式 `chat.completion` JSON 响应
- [x] 流式 `chat.completion.chunk` SSE 响应
- [x] SSE 结束发送 `data: [DONE]`
- [x] OpenAI-style error body：`{"error":{...}}`
- [x] 404 route error
- [x] 405 method error

### Request Compatibility

- [x] 解析 `model`
- [x] 解析 completions `prompt`
- [x] 支持 string prompt
- [x] 支持 string array prompt
- [x] 解析 `messages`
- [x] 支持 `system`、`user`、`assistant` roles
- [x] 支持 `tool` role，并把工具结果转为模型上下文
- [x] 支持 string content
- [x] 支持 content array 中的 text fragments
- [x] 解析 `max_tokens`
- [x] 解析 `max_completion_tokens`
- [x] 解析 `stream`
- [x] 解析 `temperature`
- [x] 解析 `top_p`
- [x] 解析 `seed`
- [x] 支持非标准但实用的 per-request `device`

### Tool Calling Protocol

- [x] 解析 `tools`
- [x] 解析 function tool name
- [x] 支持 `tool_choice: "none"` / `auto` 时走普通文本生成
- [x] 支持 `tool_choice: "required"` 时选择第一个 function tool
- [x] 支持 forced function tool choice object
- [x] 非流式响应返回标准 `message.tool_calls`
- [x] tool call `finish_reason` 为 `tool_calls`
- [x] 流式响应返回 `delta.tool_calls`
- [x] 流式 tool call 结束 chunk 的 `finish_reason` 为 `tool_calls`

### Runtime Integration

- [x] Gateway 调用现有 `generate_cpu` entrypoint
- [x] CPU backend 可用
- [x] MPS full-forward backend 可用
- [x] Completions 路径使用 prompt generation
- [x] 非流式路径返回 assistant content
- [x] 流式路径通过 `stream_token` 逐 token 发送 SSE
- [x] 默认不读取中间 tensor
- [x] 保留 tokenizer/prompt formatting 和 sampling 行为

## Acceptance

- [x] `cmake --build --preset debug` 通过
- [x] `ctest --preset debug` 通过
- [x] `toyllm help` 显示 `serve`
- [x] `GET /v1/models` 返回 OpenAI-compatible model list
- [x] `GET /openapi.json` 返回 OpenAPI 3.0 schema
- [x] `POST /v1/completions` 非流式可生成文本
- [x] `POST /v1/completions` `stream: true` 返回 SSE chunks 和 `[DONE]`
- [x] `POST /v1/chat/completions` 非流式可生成文本
- [x] `POST /v1/chat/completions` `stream: true` 返回 SSE chunks 和 `[DONE]`
- [x] forced tool call 非流式返回 `message.tool_calls`
- [x] forced tool call 流式返回 `delta.tool_calls`
- [x] MPS gateway smoke 通过：`hello`, `max_tokens: 1` -> `Hello`

## Known Boundaries

- 当前 server 是顺序处理请求，不是并发生产服务器。
- 当前 JSON parser 覆盖 gateway 需要的 OpenAI-compatible 请求子集，不是完整 JSON Schema validator。
- Tool calling 只实现协议兼容，不负责执行外部工具。
- Usage token 统计当前返回 `0`，后续可接 tokenizer 计数。
- Embeddings、responses API、audio、vision 等不在 M9 范围。
