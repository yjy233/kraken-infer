# toy_llm_interface

一个从 0-1 实现 LLM 推理的 C++ 实验仓库。当前目标是在 macOS 上跑通 Qwen3 0.6B 的本地推理链路，并逐步迁移到 MPS。

## 当前状态

- C++20 + CMake 项目骨架，另有无 CMake 时可用的 Makefile 兜底构建
- macOS Metal/MPS 后端探测，使用 Objective-C++ 桥接
- 基础 `Device`、`TensorDesc`、`Tensor`、`Runtime` 类型
- Qwen3 `config.json`、`generation_config.json`、`tokenizer_config.json` 读取
- `model.safetensors` 只读 mmap 加载，BF16 权重按 tensor name 绑定
- Qwen/Qwen3 byte-level BPE tokenizer，支持 chat template 需要的 added tokens
- CPU correctness path：embedding、RMSNorm、Q/K norm、RoPE、GQA attention、MLP、lm head
- Decode 阶段 KV cache
- CLI：`inspect`、`weights`、`doctor`、`infer`、`run`、`chat`
- CTest smoke test

当前 CPU 推理是正确性优先的朴素实现，速度较慢；采样仍为 greedy，MPS 计算路径尚未接入完整 forward。

## 构建

依赖：

- macOS
- Xcode Command Line Tools
- CMake 3.24+

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

查看 MPS/Metal 设备：

```bash
./build/debug/toyllm --mps-info
```

检查本地模型结构：

```bash
./build/debug/toyllm --inspect-model models/qwen3-0.6b
```

Release 构建：

```bash
cmake --preset release
cmake --build --preset release
```

如果本机还没安装 CMake，也可以先用 macOS 自带 `clang++` 走兜底 Makefile：

```bash
make test
make mps-info
make inspect
make doctor
make infer
make chat
```

Makefile 兜底构建后也可以检查模型结构：

```bash
./build/manual/toyllm --inspect-model models/qwen3-0.6b
```

常用 CLI 子命令：

```bash
./build/manual/toyllm help
./build/manual/toyllm mps
./build/manual/toyllm inspect
./build/manual/toyllm weights
./build/manual/toyllm doctor
./build/manual/toyllm infer --prompt "hello"
./build/manual/toyllm infer --prompt "hello" --max-new-tokens 8
./build/manual/toyllm run --prompt "hello"
./build/manual/toyllm chat --max-new-tokens 16
```

## 模型

当前使用的官方模型地址：

- https://huggingface.co/Qwen/Qwen3-0.6B

模型已下载到：

```text
models/qwen3-0.6b/
```

本地目录包含 `config.json`、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json`、`merges.txt`、`vocab.json`、`model.safetensors` 和 `LICENSE`。

## 目录结构

```text
apps/                 CLI 入口
cmake/                CMake helper
docs/                 架构和路线文档
include/toyllm/       公共头文件
models/qwen3-0.6b/    本地模型文件占位目录
src/core/             基础类型和工具
src/runtime/          runtime 编排
src/backends/mps/     macOS Metal/MPS backend
tests/                smoke tests
```

## 下一步实现顺序

1. 模型配置读取：解析 Qwen3 `config.json`。
2. safetensors mmap 权重读取和 Qwen3 权重绑定。
3. tokenizer、chat prompt formatting、CPU correctness path、KV cache。
4. CLI 推理和交互式 chat。
5. OpenAI/OpenAPI-compatible gateway。
6. MPS path：逐步迁移 matmul、RMSNorm、RoPE、attention/KV cache、sampling。
7. sampling/streaming/perf：top-k、top-p、temperature、流式输出、tokens/s 指标。

## 设计原则

- 不依赖现成 LLM runtime，核心推理链路自己实现。
- 先正确性，后性能。
- MPS backend 通过 Objective-C++ 封装，C++ 侧只暴露稳定接口。
- 模型文件不提交到 git，统一放在 `models/qwen3-0.6b/`。
