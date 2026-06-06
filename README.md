# toy_llm_interface

一个从零实现的 C++20 toy LLM runtime，当前目标模型是本地
`Qwen/Qwen3-0.6B`。项目已经跑通 CPU reference inference、Apple
Silicon Metal/MPS full-forward、KV cache、sampling/streaming CLI，以及
OpenAI-compatible HTTP gateway。

![Current architecture](docs/assets/current-architecture.svg)

## Current Status

M1 到 M9 已完成：

- Model/config/tokenizer：读取 Qwen3 `config.json`、`generation_config.json`、
  `tokenizer.json`，支持 Qwen chat prompt formatting。
- Weights：只读 mmap 解析 `model.safetensors`，按 Qwen3 权重名绑定 tensor view，
  校验 embedding、attention、MLP、norm、lm head shape。
- CPU reference：完整 batch=1 forward，包含 embedding、RMSNorm、Q/K norm、RoPE、
  GQA attention、MLP、residual、final norm、lm head。
- KV cache：CPU 和 MPS decode 路径都支持 per-layer K/V cache，带 capacity 校验、
  stats 和 verify path。
- Sampling：greedy、temperature、top-k、top-p、seed，可 token 级 streaming。
- MPS backend：Metal device/context/buffer、BF16 matvec、embedding、RMSNorm、
  Q/K norm、RoPE、attention、MLP、residual add、device-resident KV cache、lm head。
- CLI：`inspect`、`weights`、`doctor`、`infer`、`run`、`chat`、`serve`、
  `mps`、`mps-smoke`。
- HTTP gateway：OpenAI-compatible `/v1/models`、`/v1/completions`、
  `/v1/chat/completions`、SSE streaming、basic tools/tool_choice protocol、
  OpenAPI schema。

## MPS Forward

`--device mps` 当前会把主要 forward 算子放到 MPS 上执行，CPU 仍保留为 reference
backend 和 sampling host。

![MPS forward path](docs/assets/mps-forward.svg)

MPS 已完成：

- BF16 weight + F32 activation matvec
- token embedding lookup
- RMSNorm 和 Q/K norm
- RoPE
- attention over device-resident KV cache
- MLP gate/up/down projection 和 SiLU gate
- residual add
- final norm + lm head logits

当前仍有优化空间：

- 许多 kernel dispatch 仍逐 op 等待 command buffer 完成。
- attention kernel 可进一步做 threadgroup-level score/softmax/value reduction。
- logits 仍会读回 CPU 做 sampling。
- KV cache 当前用 F32，后续可做 BF16/FP16 KV 或 paged KV。
- prefill 仍以 batch=1/token-wise 路径为主，尚未做 sequence GEMM 化。

## OpenAI-Compatible Gateway

![Gateway flow](docs/assets/openai-gateway.svg)

启动本地服务：

```bash
cd /Users/bill/code/toy_llm_interface
source ~/.zshrc
cmake --build --preset debug

./build/debug/toyllm serve \
  --host 127.0.0.1 \
  --port 8080 \
  --model models/qwen3-0.6b \
  --model-id qwen3-0.6b \
  --device mps \
  --max-new-tokens 32
```

OpenAI-compatible base URL：

```text
http://127.0.0.1:8080/v1
```

支持的 endpoints：

- `GET /health`
- `GET /v1/health`
- `GET /v1/models`
- `GET /openapi.json`
- `GET /v1/openapi.json`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Chat completion：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16
  }'
```

Streaming：

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16,
    "stream": true
  }'
```

Forced tool call：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
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

Tool calling 只实现协议兼容：gateway 返回 `tool_calls`，不执行外部工具。

## Build And Test

依赖：

- macOS
- Xcode Command Line Tools
- CMake 3.24+

Debug build：

```bash
source ~/.zshrc
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Release build：

```bash
source ~/.zshrc
cmake --preset release
cmake --build --preset release
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

## Common Commands

Inspect model and weights：

```bash
./build/debug/toyllm inspect models/qwen3-0.6b
./build/debug/toyllm weights models/qwen3-0.6b
./build/debug/toyllm doctor models/qwen3-0.6b
```

MPS status：

```bash
./build/debug/toyllm mps
./build/debug/toyllm mps-smoke
```

Inference：

```bash
./build/debug/toyllm infer --model models/qwen3-0.6b --prompt "hello"
./build/debug/toyllm infer --model models/qwen3-0.6b --prompt "hello" --device mps
./build/debug/toyllm infer --model models/qwen3-0.6b --prompt "hello" --stream
./build/debug/toyllm infer --model models/qwen3-0.6b --prompt "hello" \
  --sample --temperature 0.6 --top-k 20 --top-p 0.95 --seed 42
```

Interactive chat：

```bash
./build/debug/toyllm chat --model models/qwen3-0.6b --device mps --stream
```

Debug dump / KV verification：

```bash
./build/debug/toyllm infer --model models/qwen3-0.6b --prompt "hello" \
  --device mps --max-new-tokens 1 --dump-dir build/debug-dump

./build/debug/toyllm infer --model models/qwen3-0.6b --prompt "hello" \
  --device mps --max-new-tokens 2 --verify-kv-cache
```

## Model Files

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

模型权重不应该提交到 git。真实模型文件放在：

```text
models/qwen3-0.6b/
```

## Project Layout

```text
apps/                    CLI entrypoints
cmake/                   CMake helpers and warning policy
docs/                    Architecture notes, milestones, milestone tasks
include/toyllm/          Public C++ headers
models/                  Local model placeholders and downloaded model files
src/core/                Status, device, tensor primitives
src/model/               Model/generation/tokenizer config parsing
src/runtime/             Runtime orchestration and public inference wrapper
src/runtime/cpu/         Tokenizer, safetensors, Qwen CPU reference, KV cache
src/backends/mps/        Objective-C++ Metal/MPS backend
tests/                   CTest smoke tests
```

## Current Boundaries

- 主要支持 batch size `1`。
- 当前模型目标是 dense `Qwen3ForCausalLM`，不直接支持 Qwen3.5 hybrid
  architecture 或 MoE expert routing。
- MPS path 已 full-forward，但仍是 correctness-first/initial performance path。
- Gateway 是顺序 POSIX HTTP server，不是并发生产 server。
- Gateway usage token 统计当前返回 `0`。
- Tool calling 只做 OpenAI-compatible 协议，不执行工具。
- Vision、audio、embeddings、Responses API 不在当前范围。

## Next Work

推荐后续里程碑：

- M10: MPS performance pass
  - command buffer batching
  - fused kernels
  - optimized attention
  - GPU-side logits top-k/top-p
  - BF16/FP16 KV cache
- M11: Prompt cache / cache-control
  - prefix token hash
  - cross-request KV snapshot
  - LRU/TTL/memory budget
  - cached last logits or last hidden
- M12: Model family expansion
  - Qwen3.5 text/hybrid architecture
  - optional MoE routing path
  - model-specific weight mapping adapters

## Design Principles

- 不依赖现成 LLM runtime，核心推理链路自己实现。
- CPU path 优先作为 correctness reference。
- MPS path 逐算子对齐，再做性能优化。
- Apple API 只放在 Objective-C++ `.mm` 文件里，公共头文件保持 C++。
- 模型文件和本机构建产物不提交到 git。
