# kraken-infer

<p align="center">
  <img src="docs/assets/kraken-logo.png" alt="kraken-infer logo" width="180">
</p>

`kraken-infer` 是一个 C++20 本地 LLM inference runtime，当前主目标是
**Qwen3.5 0.8B GGUF** 在 Apple Silicon/macOS 上的原生 Metal 推理。项目方向是把
Qwen3.5 的 tokenizer、GGUF 权重读取、hybrid decoder、KV/recurrent cache、采样、
profiling 和 OpenAI-compatible gateway 都收敛到本仓库自己的 runtime/backend 中。

`Qwen/Qwen3-0.6B` safetensors 路径仍保留为 legacy/reference path，主要用于早期
CPU/MPS/MPSGraph correctness 对齐和 smoke test。

核心定位：

- 不依赖现成 LLM runtime 执行推理热路径。
- Qwen3.5 主线使用 GGUF，不走 safetensors 或旧 Qwen3 config 变体。
- Metal path 是 Qwen3.5 的主要执行路径；CPU 主要用于旧模型 reference、调试和测试。
- 公共 C++ 头文件保持平台无关，Apple API 隔离在 Objective-C++ `.mm` 文件。
- OpenAI-compatible gateway 作为本地调试和集成入口，不作为生产并发 server。

## Current Target

主要目标模型：

- Model family: Qwen3.5 dense text
- Baseline model: Qwen3.5 0.8B
- Runtime format: GGUF
- Recommended local example: `models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf`
- Main device: `--device mps`
- Architecture key: `general.architecture = "qwen35"`
- Decoder shape: hybrid full attention + recurrent linear attention
- Cache: full-attention KV cache plus Qwen3.5 recurrent R/S state

Legacy model:

- Model: `Qwen/Qwen3-0.6B`
- Local path: `models/qwen3-0.6b/`
- Format: HF-style config/tokenizer + `model.safetensors`
- Use case: CPU reference, old MPS/MPSGraph path, compatibility tests

真实模型权重不要提交到 git，统一放在 `models/` 下。

## Current Features

### Qwen3.5 GGUF Runtime

- 读取 Qwen3.5 GGUF metadata、tensor directory 和 mmap tensor payload。
- 支持单文件 GGUF 路径，也可从模型目录解析 GGUF。
- 原生 Qwen3.5 dense weight map，校验 embedding、output norm、full attention、
  recurrent linear attention、MTP 相关张量命名和 shape。
- 从 GGUF tokenizer metadata 加载 BPE tokenizer、special tokens、merges 和
  `tokenizer.chat_template`。
- 支持 Qwen chat prompt formatting 和 `enable_thinking` prompt 控制。
- 支持 `kraken-infer tokenize`，用于查看 native tokenizer token id。

### Qwen3.5 Metal Forward

默认 Qwen3.5 generation 入口已经走本仓库原生 `generate_qwen35_metal` 路径。

已覆盖：

- GGUF Q4_K/Q5_K/Q6_K quantized matvec/matmul。
- token embedding、RMSNorm、Q/K L2 norm、MRoPE。
- full-attention layer，包括 Q/G projection 拆分、F16/F32 KV cache 和 batched causal
  attention。
- recurrent linear attention layer，包括 conv state、gated delta net、R/S state cache、
  beta/gate prepare 和 norm-gated 后处理。
- chunked prefill，默认 `prefill_chunk_tokens = 1024`。
- greedy multi-token decode。
- Qwen3.5 MTP/NextN speculative decode，支持 `nextn_predict_layers=1` 的
  MTP GGUF，在 greedy 路径返回 drafted/accepted/verify 统计。
- 基础 `top_k` / `top_p` / `temperature` / `seed` host-side sampling。
- logits top-k readback、debug dump、KV cache stats 和 profile artifacts。

默认行为：

- F16 KV cache 默认开启；`KRAKEN_QWEN35_F16_KV=0` 可回退 F32 KV。
- 兼容形状下 full attention prefill 优先使用 Qwen3.5 `head_dim=256` 的 flash256 path；
  `KRAKEN_QWEN35_FLASH_ATTENTION=0` 可禁用。
- prefill 默认按 chunk 提交 command buffer；`KRAKEN_QWEN35_PREFILL_COMMIT=single`
  可回退单次提交作诊断。

### OpenAI-Compatible Gateway

Gateway 是一个顺序 POSIX HTTP server，提供 OpenAI-compatible 子集：

- `GET /health`
- `GET /v1/health`
- `GET /chat_page`
- `GET /v1/models`
- `GET /openapi.json`
- `GET /v1/openapi.json`
- `POST /v1/completions`
- `POST /v1/chat/completions`

支持能力：

- legacy text completions。
- chat completions。
- SSE streaming。
- `temperature`、`top_p`、`seed`、`max_tokens`、`max_completion_tokens`。
- `enable_thinking`、`chat_template_kwargs.enable_thinking`。
- `reasoning_format`，用于把 Qwen3.5 thinking 输出拆到 OpenAI-style
  `reasoning_content`。
- 非标准但实用的 per-request `device`。
- `prefill_chunk_tokens` request override。
- `mtp` / `mtp_draft_tokens` request override；非 streaming 响应会返回
  `X-Kraken-MTP-*` headers。
- `cache_prompt` / `n_cache_reuse` exact-prefix prompt cache。
- 基础 tools/tool_choice 协议兼容，返回 OpenAI-style `tool_calls`，但不执行外部工具。
- 浏览器对话页 `/chat_page`，支持 max new tokens、streaming 和 thinking 开关。

图片输入已接通 OpenAI content array 的 `image_url` data URL 路径。Gateway 会解析
`text` / `image_url`、校验 `qwen3vl_merger` mmproj，然后把请求桥接到 llama.cpp
`llama-mtmd-cli` 完成 Qwen3.5 0.8B VL 推理。这是可测试的端到端视觉理解路径；
本仓库原生 Metal vision graph 仍是后续工作。默认查找
`/Users/bill/code/llama.cpp/build/bin/llama-mtmd-cli`，也可以用
`KRAKEN_LLAMA_MTMD_CLI` 覆盖。使用带 MTP 的 text GGUF 启动时，文本请求可以走
原生 MTP；图片请求会走 `llama-mtmd-cli` 桥接，并通过 MTP headers 标记 MTP 未启用。

### Prefix Cache

Qwen3.5 path 支持单进程、单 active slot、host-backed exact-prefix block cache：

- 复用完整 prompt prefix block。
- 恢复 full-attention K/V cache。
- 恢复 Qwen3.5 recurrent R/S state。
- response headers 返回 cache hit/miss token 数。

当前还没有 paged KV、block-table attention、middle-chunk KV shifting 或多 slot 并发 cache。

### Diagnostics And Profiling

可用诊断入口：

- `mps` / `mps-smoke`: Metal/MPS availability 和 operator smoke test。
- `mpsgraph` / `mpsgraph-smoke`: MPSGraph availability 和 tiny graph smoke test。
- `inspect`: 检查模型配置和 tokenizer 结构。
- `weights`: 检查 safetensors 或 GGUF 权重结构。
- `doctor`: 一次性输出 backend、模型和权重诊断。
- `bench-qwen35-matmul`: 用真实 GGUF K-quant tensor 跑 Metal matmul benchmark。
- `bench-qwen35-gdn`: 用 synthetic Qwen3.5 标准形状跑 GDN benchmark。
- `bench-qwen35-attention`: 用 synthetic Qwen3.5 标准形状跑 attention benchmark。

Profile 支持 `off|summary|trace|flamegraph|all`，可以通过 CLI `--profile` 或 gateway
`x-kraken-profile` 相关路径写入 `build/profiles`。

## Quick Start

准备模型文件，例如：

```text
models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
models/qwen3.5-0.8b/mmproj-Qwen3.5-0.8B-BF16.gguf
models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf
```

构建和测试：

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

检查本机 Metal/MPS：

```bash
./build/debug/kraken-infer mps
./build/debug/kraken-infer mps-smoke
```

检查 Qwen3.5 GGUF：

```bash
./build/debug/kraken-infer inspect models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
./build/debug/kraken-infer weights models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
./build/debug/kraken-infer doctor models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
```

运行一次 CLI 推理：

```bash
./build/debug/kraken-infer infer \
  --model models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt "用一句话介绍你自己" \
  --device mps \
  --max-new-tokens 64 \
  --enable-thinking \
  --stream
```

启动本地 HTTP gateway（Qwen3.5 0.8B VL + MTP + F16 KV cache）：

```bash
KRAKEN_QWEN35_F16_KV=1 ./build/debug/kraken-infer serve \
  --host 127.0.0.1 \
  --port 18080 \
  --model models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf \
  --mmproj models/qwen3.5-0.8b/mmproj-Qwen3.5-0.8B-BF16.gguf \
  --model-id qwen3.5-0.8b-vl-mtp \
  --device mps \
  --max-new-tokens 128 \
  --prefill-chunk-tokens 32 \
  --mtp \
  --mtp-draft-tokens 3 \
  --no-cache-prompt \
  --profile summary
```

该命令同时启用 OpenAI `image_url` 图片输入、文本请求的原生 MTP speculative decode，
以及 Qwen3.5 full-attention F16 KV cache。当前图片请求通过 llama.cpp
`llama-mtmd-cli` 桥接完成视觉理解；跨请求 prompt prefix cache 见下方单独示例，
第一版 MTP 不和 `--cache-prompt` 同时启用。

浏览器对话页：

```text
http://127.0.0.1:18080/chat_page
```

OpenAI-compatible base URL：

```text
http://127.0.0.1:18080/v1
```

## HTTP Examples

Chat completion：

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-0.8b-vl-mtp",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
    "max_completion_tokens": 128,
    "mtp": true,
    "mtp_draft_tokens": 3,
    "chat_template_kwargs": {"enable_thinking": false},
    "device": "mps"
  }'
```

Streaming thinking：

```bash
curl -N http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-0.8b-vl-mtp",
    "messages": [{"role": "user", "content": "计算 23 * 19，并给出答案"}],
    "max_completion_tokens": 128,
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": true},
    "reasoning_format": "deepseek",
    "device": "mps"
  }'
```

Text completion：

```bash
curl http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-0.8b-vl-mtp",
    "prompt": "hello",
    "max_tokens": 32,
    "mtp": true,
    "mtp_draft_tokens": 3,
    "device": "mps"
  }'
```

Forced tool call protocol compatibility：

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-0.8b-vl-mtp",
    "messages": [{"role": "user", "content": "weather?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {}}
      }
    }],
    "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
  }'
```

## Prefix Cache Example

Start gateway with Qwen3.5 0.8B VL and prompt cache enabled. Keep
`--cache-block-tokens` aligned with `--prefill-chunk-tokens` for the current
implementation.

```bash
./build/debug/kraken-infer serve \
  --host 127.0.0.1 \
  --port 18080 \
  --model models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf \
  --mmproj models/qwen3.5-0.8b/mmproj-Qwen3.5-0.8B-BF16.gguf \
  --model-id qwen3.5-0.8b-vl \
  --device mps \
  --max-new-tokens 32 \
  --prefill-chunk-tokens 32 \
  --cache-prompt \
  --cache-reuse 32 \
  --cache-block-tokens 32 \
  --cache-capacity-blocks 16
```

Run the same request twice:

```bash
curl -i http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-0.8b-vl",
    "messages": [{"role": "user", "content": "请用三段话解释 KV cache 的作用、prefill 和 decode 的区别，以及为什么相同 system prompt 能复用 cache。每段都要包含一个具体例子。"}],
    "max_completion_tokens": 32,
    "prefill_chunk_tokens": 32,
    "cache_prompt": true,
    "cache_block_tokens": 32,
    "n_cache_reuse": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

第二次请求可观察 response headers：

```text
X-Kraken-Prompt-Cache-Hit-Tokens: 32
X-Kraken-Prompt-Cache-Miss-Tokens: ...
```

## Common Commands

Interactive terminal chat：

```bash
./build/debug/kraken-infer chat \
  --model models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf \
  --device mps \
  --max-new-tokens 128 \
  --enable-thinking \
  --stream
```

Sampling：

```bash
./build/debug/kraken-infer infer \
  --model models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt "写一个很短的科幻开头" \
  --device mps \
  --sample \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --seed 42
```

Tokenizer：

```bash
./build/debug/kraken-infer tokenize \
  --model models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt "hello"
```

Profile：

```bash
./build/debug/kraken-infer infer \
  --model models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt "hello" \
  --device mps \
  --max-new-tokens 8 \
  --profile all \
  --profile-dir build/profiles
```

Qwen3.5 Metal benchmark helpers：

```bash
./build/debug/kraken-infer bench-qwen35-matmul \
  --model models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf \
  --list

./build/debug/kraken-infer bench-qwen35-gdn \
  --tokens 1024 \
  --iterations 10 \
  --warmup 2

./build/debug/kraken-infer bench-qwen35-attention \
  --tokens 1024 \
  --start-position 0 \
  --capacity-tokens 1024 \
  --f16-kv \
  --iterations 10 \
  --warmup 2
```

Legacy Qwen3 0.6B smoke path：

```bash
./build/debug/kraken-infer infer \
  --model models/qwen3-0.6b \
  --prompt "hello" \
  --device mps \
  --max-new-tokens 32 \
  --stream
```

Makefile fallback：

```bash
make test
make mps-info
make inspect
make infer
make chat
make serve
```

Qwen3.5 VL gateway smoke test：

```bash
python3 scripts/test_qwen35_vl_gateway.py --max-tokens 24
make qwen35-vl-test
```

Qwen3.5 MTP gateway smoke test：

```bash
python3 scripts/test_qwen35_mtp_gateway.py --max-tokens 8
```

## Project Layout

```text
apps/                    CLI entrypoints
cmake/                   CMake helpers and warning policy
docs/                    Architecture notes, milestones, milestone tasks
docs/assets/             README and architecture images
include/toyllm/          Public C++ headers
models/                  Local model placeholders and downloaded model files
src/core/                Status, device, tensor primitives
src/model/               Model/generation/tokenizer config parsing
src/runtime/             Runtime orchestration, GGUF, gateway, Qwen3.5 path
src/runtime/cpu/         Legacy tokenizer, safetensors, Qwen CPU reference, KV cache
src/backends/mps/        Objective-C++ Metal/MPS backend
src/backends/mpsgraph/   Experimental MPSGraph backend for legacy Qwen3 path
tests/                   CTest smoke tests
web/                     Static browser chat page assets
```

更多实现说明：

- [Qwen3.5 GGUF 推理技术方案](docs/qwen3-5-0-8b-technical-plan.md)
- [Qwen3.5 cross-request KV cache](docs/qwen3-5-cross-request-kv-cache.md)
- [Qwen3.5 OpenAI thinking](docs/qwen3-5-0-8b-openai-thinking.md)
- [Qwen3.5 image input research](docs/qwen3-5-0-8b-image-input-llama-cpp.md)
- [Qwen3.5 MTP llama.cpp research](docs/qwen3-5-0-8b-mtp-llama-cpp.md)
- [CPU forward inference](docs/forward.md)
- [MPSGraph backend](docs/mpsgraph-backend.md)
- [Logging and profiling](docs/logging-and-profiling.md)

## Current Boundaries

- 当前主线覆盖 dense Qwen3.5 0.8B GGUF 文本路径。
- Qwen3.5 0.8B MTP/NextN 当前支持 greedy 文本 speculative decode；
  batched speculative、多 sequence、paged KV 和完整原生 VL/Omni 多模态 graph
  仍是后续工作。
- Qwen3.5 MoE 仍是后续工作。
- Qwen3.5 VL gateway 当前通过 llama.cpp `llama-mtmd-cli` 桥接完成可测试推理。
- Qwen3.5 runtime 不以旧 Qwen3 CPU reference path 作为生产 fallback。
- 采样目前仍会在需要时读回最后一行 logits 到 host。
- prefix cache 是 host-backed exact-prefix block cache，不是 paged attention cache。
- gateway 是顺序 POSIX HTTP server，不是高并发生产 server。
- gateway usage token 统计当前仍可能返回占位值。
- Tool calling 只做 OpenAI-compatible 协议，不执行工具。
- Audio/embeddings/Responses API 不在当前完整支持范围。
