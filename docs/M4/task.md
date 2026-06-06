# M4 Task: Interactive Chat CLI

目标：提供可交互的 `chat` 子命令，让用户能在 CLI 中和本地 Qwen3 0.6B 真实对话。

## Scope

本阶段基于 M3 CPU 推理路径，不引入 HTTP gateway，不做 MPS 加速，不实现流式输出。

## Tasks

### CLI Command

- [x] 增加 `chat` 子命令
- [x] `chat` 默认读取 `models/qwen3-0.6b`
- [x] `chat <model_dir>` 支持指定模型目录
- [x] `chat --model <model_dir>` 支持显式指定模型目录
- [x] `chat --max-new-tokens N` 支持控制每轮最大生成长度
- [x] `chat --enable-thinking` 支持打开 Qwen3 thinking 输出
- [x] `/exit` 退出会话
- [x] `/quit` 退出会话
- [x] 空输入不触发推理

### Conversation State

- [x] 每轮 user 输入写入会话历史
- [x] 每轮 assistant 输出写入会话历史
- [x] 下一轮推理使用完整 user/assistant 历史
- [x] 推理失败时回滚本轮 user 输入
- [x] 默认 no-thinking prompt，便于直接看到正文回复

### Verification

- [x] `make test` 通过
- [x] `./build/manual/kraken-infer help` 展示 chat 参数
- [x] `./build/manual/kraken-infer chat --max-new-tokens 8` 能进入 REPL
- [x] 输入 `hello` 后返回真实模型回复
- [x] 已验证输出：`Hello! How can I assist you today`
- [x] 输入 `/exit` 后进程正常退出

## Acceptance

- [x] 用户能通过 CLI 启动 chat
- [x] 用户能输入消息并得到真实模型回复
- [x] chat 命令支持多轮上下文
- [x] chat 命令可正常退出

## Known Limitations

- [ ] 当前 chat 不是流式输出
- [ ] 当前 chat 使用 CPU greedy decode，速度较慢
- [ ] 当前 chat 没有上下文窗口裁剪策略
- [ ] 当前 chat 没有 OpenAI/OpenAPI-compatible HTTP gateway
