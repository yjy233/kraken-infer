# toy_llm_interface

一个从 0-1 实现 LLM 推理的 C++ 实验仓库。当前目标是先在 macOS 上跑通工程骨架，并为 Qwen3 0.6B 推理实现预留 runtime、device、tensor、MPS backend、CLI 和测试入口。

## 当前状态

- C++20 + CMake 项目骨架
- macOS Metal/MPS 后端探测，使用 Objective-C++ 桥接
- 基础 `Device`、`TensorDesc`、`Tensor`、`Runtime` 类型
- CLI：`toyllm --mps-info`
- CTest smoke test

Qwen3 tokenizer、权重加载、Metal/MPS 算子、KV cache、采样器还没有实现。

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
./build/manual/toyllm doctor
./build/manual/toyllm run --prompt "hello"
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
2. tokenizer：接入 BPE/SentencePiece 解析或先实现最小 tokenizer loader。
3. 权重加载：优先支持 `safetensors`，建立 tensor name 到内存视图的映射。
4. CPU correctness path：先把 RMSNorm、RoPE、attention、MLP、sampling 用 CPU 跑通小输入。
5. MPS path：逐步迁移 matmul、RMSNorm、RoPE、attention/KV cache、sampling。
6. CLI 推理：`--model models/qwen3-0.6b --prompt "..."`

## 设计原则

- 不依赖现成 LLM runtime，核心推理链路自己实现。
- 先正确性，后性能。
- MPS backend 通过 Objective-C++ 封装，C++ 侧只暴露稳定接口。
- 模型文件不提交到 git，统一放在 `models/qwen3-0.6b/`。
