# Milestones

本文档定义 Qwen3 0.6B 从模型结构读取到最终推理框架的阶段性功能目标。

当前目标模型：

- Model: `Qwen/Qwen3-0.6B`
- Local path: `models/qwen3-0.6b/`
- Architecture: `Qwen3ForCausalLM`
- Layers: `28`
- Hidden size: `1024`
- Attention heads: `16`
- KV heads: `8`
- Head dim: `128`
- Intermediate size: `3072`
- Vocab size: `151936`
- DType: `bfloat16`
- RoPE theta: `1000000`
- Tied embeddings: `true`

## Current Implementation Status

- [x] M1: model structure and project introspection
- [x] M2: safetensors mmap reader, Qwen3 weight mapping, shape validation, `weights` CLI
- [x] M3: CPU correctness inference path, tokenizer, greedy decode, KV cache, `infer`/`run` CLI
- [x] M4: interactive `chat` CLI with multi-turn history
- [x] M5: basic CPU KV cache abstraction, bounds checks, stats
- [ ] M6: OpenAI/OpenAPI-compatible HTTP gateway
- [ ] M7: MPS full forward path and performance optimization

当前 `chat` 已能通过本地 Qwen3 0.6B 真实生成回复；CPU 实现以正确性为先，速度较慢。下面保留原始阶段拆解，后续会按网关、采样/流式、MPS 加速继续细化。

## M1: Model Structure And Project Introspection

目标：让框架能识别本地模型目录，并准确读取 Qwen3 0.6B 的结构。

功能范围：

- 读取 `config.json`
- 读取 `generation_config.json`
- 解析模型基础结构：layer 数、hidden size、head 数、KV head 数、RoPE 参数、dtype
- 校验维度一致性：
  - `num_attention_heads * head_dim > 0`
  - `num_key_value_heads * head_dim > 0`
  - `num_attention_heads % num_key_value_heads == 0`
  - tokenizer explicit vocab 不超过 config vocab size
  - tokenizer max token id 小于 config vocab size
- CLI 支持打印模型摘要

验收标准：

- 能执行模型检查命令并输出 Qwen3 0.6B 的结构摘要
- 配置缺字段或维度不一致时给出明确错误
- 不加载权重也能完成结构检查

## M2: Safetensors Weight Reader

目标：框架能读取 safetensors 权重元数据，并建立权重名称到 tensor view 的映射。

功能范围：

- 解析 `model.safetensors` header
- 读取 tensor name、dtype、shape、offset
- 支持 mmap 或等价的只读文件映射
- 建立 Qwen3 权重映射：
  - embedding
  - per-layer attention q/k/v/o projection
  - per-layer MLP gate/up/down projection
  - per-layer RMSNorm
  - final RMSNorm
  - lm head 或 tied embedding
- 做权重 shape 校验

验收标准：

- 能列出全部权重 tensor
- 能按 layer index 查询权重
- 所有权重 shape 与 `config.json` 一致
- 不发生整文件复制到内存的行为

## M3: Tokenizer And Prompt Formatting

目标：能把用户输入转换成 Qwen3 所需 token ids，并能把输出 token ids 解码回文本。

功能范围：

- 读取 `tokenizer.json`
- 支持必要的 BPE encode/decode
- 支持 special tokens：
  - `<|endoftext|>`
  - `<|im_start|>`
  - `<|im_end|>`
- 实现最小 chat prompt 格式：
  - system 可选
  - user prompt
  - assistant generation prefix
- 读取默认采样参数：
  - temperature `0.6`
  - top_k `20`
  - top_p `0.95`
  - eos ids `151645`, `151643`

验收标准：

- 固定 prompt 的 token ids 稳定可复现
- encode 后 decode 能还原主要文本
- special token 不被错误拆分
- prompt token ids 能传入后续 forward pipeline

## M4: CPU Correctness Forward Path

目标：先在 CPU 上实现完整 Qwen3 forward，用于正确性基准。

功能范围：

- embedding lookup
- RMSNorm
- linear projection
- Q/K/V projection
- RoPE
- grouped-query attention
- attention softmax
- output projection
- MLP：gate projection、up projection、SiLU、down projection
- residual connection
- final norm
- logits projection

实现策略：

- 先支持 batch size `1`
- 先支持 prefill
- 再支持单 token decode
- CPU 版本不追求速度，只追求可对齐、可调试

验收标准：

- 能完成短 prompt 的完整 forward
- logits shape 为 `[vocab_size]`
- 单层中间输出可 dump
- 能与 Transformers/PyTorch 的小输入结果做误差对齐

## M5: Basic KV Cache

目标：实现 decode 阶段的 KV cache，避免每生成一个 token 都重复计算整个 prompt。

功能范围：

- 为每层维护 K cache 和 V cache
- cache 逻辑形状：
  - K: `[num_layers, num_key_value_heads, max_seq_len, head_dim]`
  - V: `[num_layers, num_key_value_heads, max_seq_len, head_dim]`
- prefill 阶段写入 prompt 全量 K/V
- decode 阶段每步追加当前 token 的 K/V
- attention 读取 `[0, current_seq_len)` 范围
- 支持 reset cache

验收标准：

- prefill + decode 输出与无 cache 路径一致
- decode 每步只处理一个新 token
- 当前 seq length、capacity、used bytes 可观测
- 超过 cache capacity 时明确报错

## M6: Sampling And Streaming CLI

目标：形成可用的命令行文本生成入口。

功能范围：

- CLI 支持：
  - `--model`
  - `--prompt`
  - `--max-new-tokens`
  - `--temperature`
  - `--top-k`
  - `--top-p`
  - `--seed`
  - `--device`
  - `--stream`
- 实现采样：
  - greedy
  - temperature
  - top-k
  - top-p
- 支持 eos 停止
- 支持流式输出 token

验收标准：

- 能在 CPU 路径生成完整文本
- 固定 seed 输出可复现
- 可打印 prefill latency、decode latency、tokens/s
- eos token 出现时正确停止

## M7: MPS Backend Bring-up

目标：把主要计算迁移到 macOS MPS/Metal 后端。

功能范围：

- 建立 MPS memory buffer 抽象
- 实现 host/device 拷贝
- 实现或封装 MPS matmul
- 优先迁移最重算子：
  - linear projection
  - Q/K/V projection
  - attention score matmul
  - attention value matmul
  - MLP projection
- CPU 保留为 reference backend
- Runtime 支持 backend dispatch

验收标准：

- MPS backend 能完成至少一层 forward
- MPS 单算子输出与 CPU reference 对齐
- 完整模型能在 MPS 路径生成文本
- CLI `--device mps` 可用

## M8: KV Cache And MPS Performance Optimization

目标：优化 decode 性能，使 Qwen3 0.6B 在 Apple Silicon 上具备可交互速度。

功能范围：

- KV cache 预分配，避免 decode 时动态分配
- cache layout 调整为 MPS-friendly contiguous layout
- 减少 CPU/GPU 同步点
- 减少中间 tensor 分配
- 融合小算子：
  - RMSNorm
  - RoPE
  - SiLU gate
  - residual add
- attention decode 优化：
  - 只处理当前 query token
  - K/V 读取历史 cache
  - 尽量避免重复转置
- 可选后续优化：
  - paged KV cache
  - quantized KV cache
  - weight quantization
  - speculative decoding

验收标准：

- decode 阶段不会重复跑完整 prompt
- prefill 和 decode 分开计时
- tokens/s 指标稳定输出
- cache 内存占用可估算、可打印
- 生成长文本时没有明显内存增长泄漏

## M9: Inference Gateway And OpenAPI Protocol

目标：在本地推理 runtime 之上提供 HTTP 网关，使外部应用可以通过标准 API 调用模型。

功能范围：

- 实现推理服务进程
- 支持 OpenAPI 3.0 schema 描述
- 提供基础 REST endpoints：
  - `GET /health`
  - `GET /v1/models`
  - `POST /v1/completions`
  - `POST /v1/chat/completions`
- 支持 OpenAI-compatible request/response 结构：
  - `model`
  - `messages`
  - `prompt`
  - `max_tokens`
  - `temperature`
  - `top_p`
  - `stream`
- 支持 streaming response
- 支持单实例请求队列
- 暴露基础性能指标：
  - prefill latency
  - decode latency
  - tokens/s
  - current queue size
- 支持 runtime 参数：
  - bind host
  - port
  - model path
  - device
  - max context length

实现策略：

- 第一阶段只支持单模型、单进程、batch size `1`
- 网关层不直接操作 tensor，只调用 runtime generate API
- streaming 先支持 server-sent events
- OpenAPI schema 作为独立文档或服务 endpoint 暴露
- API 错误返回统一结构，避免底层异常直接泄漏

验收标准：

- 能通过 HTTP 请求完成一次非流式 chat completion
- 能通过 HTTP 请求完成一次流式 chat completion
- `GET /v1/models` 返回当前加载模型
- OpenAPI schema 能描述已支持 endpoints
- 网关异常不会导致推理进程崩溃
- CLI 推理和 HTTP 推理共享同一套 runtime

## Final Acceptance

最终目标：形成一个从模型目录到文本生成的最小完整推理框架。

最终能力：

- 读取 Qwen3 0.6B 模型目录
- 解析模型配置
- 解析 tokenizer
- 读取 safetensors 权重
- CPU reference forward
- KV cache decode
- sampling
- streaming CLI
- MPS backend
- HTTP inference gateway
- OpenAPI schema
- 基础性能统计

最终命令形态：

```bash
./build/release/toyllm \
  --model models/qwen3-0.6b \
  --prompt "你好，介绍一下你自己" \
  --device mps \
  --max-new-tokens 128 \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --stream
```

非目标：

- 暂不支持训练
- 暂不支持多 batch
- 暂不支持多模型架构
- 暂不追求通用 LLM runtime 兼容性
- 暂不接入现成推理框架

## Recommended Implementation Order

推荐按以下顺序推进：

1. M1 model config
2. M2 safetensors metadata
3. M3 tokenizer
4. M4 CPU full forward
5. M5 KV cache
6. M6 sampling and CLI
7. M7 MPS backend
8. M8 performance optimization
9. M9 inference gateway and OpenAPI protocol

关键原则：

- CPU path 是 correctness reference，不能省。
- MPS path 每迁移一个算子，都要保留 CPU 对照。
- KV cache 先正确，再优化 layout。
- 每个 milestone 都要有可运行命令和验收输出。
