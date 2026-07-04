# Models

模型文件不要提交到 git。

当前主线目标是 Qwen3.5 0.8B GGUF，推荐本地放置示例：

```text
models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
models/qwen3.5-0.8b/mmproj-Qwen3.5-0.8B-BF16.gguf
```

运行时可以直接把 GGUF 文件路径传给 `--model`，也可以传包含 GGUF 的模型目录。
图片输入需要额外用 `--mmproj` 指向对应的 mmproj GGUF。

Legacy Qwen3 0.6B safetensors 路径仍保留：

```text
models/qwen3-0.6b/
```

Qwen3 0.6B 官方地址：

```text
https://huggingface.co/Qwen/Qwen3-0.6B
```
