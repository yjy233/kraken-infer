# M3 Task: CPU Inference And CLI Generation

目标：在 CPU 上跑通 Qwen3 0.6B 的完整单 batch 推理链路，并让 CLI 能用真实模型做短文本生成。

## Scope

输入目录：

```text
models/qwen3-0.6b/
```

本阶段聚焦正确性和可调试性，不追求速度；MPS forward、采样和流式输出留到后续阶段。

## Tasks

### Tokenizer And Prompt

- [x] 读取 `vocab.json`
- [x] 读取 `merges.txt`
- [x] 读取 `tokenizer_config.json` 中的 added tokens
- [x] 支持 `<|im_start|>`、`<|im_end|>`、`<|endoftext|>`
- [x] 支持 `<think>`、`</think>` token
- [x] 实现 Qwen3 chat prompt 的 user/assistant 格式
- [x] 默认使用 no-thinking prompt，方便 CLI 直接输出正文
- [x] 支持 byte-level BPE encode
- [x] 支持 byte-level BPE decode

### Weight Loading

- [x] 只读打开 `model.safetensors`
- [x] mmap 映射权重文件
- [x] 解析 safetensors header length
- [x] 解析 safetensors JSON header
- [x] 建立 tensor name 到 view 的映射
- [x] 校验 BF16 tensor byte size 与 shape 一致
- [x] 绑定 embedding、lm head、final norm
- [x] 绑定 28 层 attention 权重
- [x] 绑定 28 层 MLP 权重
- [x] 绑定 Qwen3 q_norm/k_norm

### CPU Forward

- [x] BF16 权重转 float 读取
- [x] token embedding lookup
- [x] RMSNorm
- [x] Q/K per-head RMSNorm
- [x] Q/K/V projection
- [x] Qwen/Llama half-rotation RoPE
- [x] grouped-query attention
- [x] causal attention softmax
- [x] attention output projection
- [x] MLP gate/up/down projection
- [x] SiLU activation
- [x] residual connection
- [x] final norm
- [x] lm head logits
- [x] greedy next-token selection
- [x] 单层和中间 tensor dump
- [x] prompt token ids dump
- [x] final norm dump
- [x] logits dump

### KV Cache

- [x] 为每层维护 K cache
- [x] 为每层维护 V cache
- [x] prefill 阶段逐 token 写入 K/V
- [x] decode 阶段只 forward 新 token
- [x] attention 读取 `[0, current_position]` cache 范围
- [x] 每次请求重置 cache capacity

### CLI

- [x] `infer --prompt <text>` 使用真实 CPU 推理
- [x] `run --prompt <text>` 使用真实 CPU 推理
- [x] 支持 `--model <model_dir>`
- [x] 支持 `--max-new-tokens N`
- [x] 支持 `--enable-thinking`
- [x] 支持 `--dump-dir DIR`
- [x] 保留 `inspect`、`doctor`、`mps` 行为

### Correctness Alignment

- [x] 提供 `scripts/compare_cpu_transformers.py`
- [x] 对齐脚本读取 C++ dump 中的 prompt token ids
- [x] 对齐脚本用同一组 token ids 调 Hugging Face Transformers
- [x] 对齐脚本比较首个 next-token logits
- [x] 输出 top token、max abs diff、mean abs diff

### Tests

- [x] `make test` 通过
- [x] smoke test 覆盖 CPU generation entrypoint 参数校验
- [x] 手工验证 `infer --prompt hello --max-new-tokens 4`
- [x] 输出可见文本：`Hello! How can`

## Acceptance

- [x] CPU path 能加载 Qwen3 0.6B 权重
- [x] CPU path 能完成短 prompt 的完整 forward
- [x] KV cache decode 能继续生成 token
- [x] CLI `infer` 返回真实模型输出
- [x] 构建无严格编译 warning
- [x] 能通过 `--dump-dir` 导出单层和中间 tensor
- [x] 提供 Transformers/PyTorch logits 对齐脚本

## Known Limitations

- [ ] 当前实现为 greedy decode
- [ ] 当前实现没有流式 token 输出
- [ ] 当前 CPU matvec 未优化，速度较慢
- [ ] 当前 MPS backend 仅完成设备探测，尚未接入完整 forward
- [ ] Transformers/PyTorch 对齐脚本依赖本机安装 `torch`、`transformers`、`numpy`
