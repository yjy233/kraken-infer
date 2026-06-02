# Architecture

当前仓库只初始化最小运行骨架，后续按以下模块扩展：

```text
CLI
  |
Runtime
  |
  +-- Model config / tokenizer / sampler
  +-- Tensor metadata and memory planner
  +-- Backend dispatch
        |
        +-- CPU correctness backend
        +-- MPS backend
```

## Backend Boundary

C++ 侧通过 `toyllm::Device` 和 backend 查询接口表达设备能力。Apple API 只放在 `.mm` 文件中，避免 Foundation/Metal 类型泄漏到公共 C++ 头文件。

后续算子建议从小集合开始：

- `rms_norm`
- `rope`
- `matmul`
- `softmax`
- `attention`
- `silu`
- `sample_top_k_top_p`

## Qwen3 0.6B Bring-up

推荐顺序：

1. 解析 `config.json`，固化 hidden size、head 数、layer 数、rope 参数。
2. 读取 tokenizer 并验证 prompt token ids。
3. 支持 safetensors mmap 读取。
4. CPU 单步 decode，对齐 Python/PyTorch 中间值。
5. MPS 单算子替换，每次保留 CPU 对照测试。
6. 实现 KV cache 和流式输出。
