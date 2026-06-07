# Qwen3 0.6B 推理文档

这个目录按推理链路拆解 Qwen3 0.6B 在本项目里的实现。

## 文档目录

- [1. BPE Tokenizer](1-BPE-tokenizer.md)
- [2. Qwen3 Forward 推理整体架构](2-Qwen3-forward.md)
- [3. Qwen3 Transformer 模块](3-Transformer-block.md)

## 总流程

```text
messages / prompt
  -> Qwen chat template
  -> tokenizer.encode
  -> prefill
  -> decode
  -> tokenizer.decode
  -> text
```
