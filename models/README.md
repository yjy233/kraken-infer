# Models

模型文件不要提交到 git。

Qwen3 0.6B 的本地文件建议放在：

```text
models/qwen3-0.6b/
```

后续推理入口会按这个目录读取 `config.json`、tokenizer 文件和权重文件。
