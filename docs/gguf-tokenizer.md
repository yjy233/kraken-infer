# Qwen3.5 GGUF 与 Tokenizer

本文只说明本项目使用的 Qwen3.5 GGUF，不讨论其他模型或 GGUF 架构。

当前示例文件：

```text
models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
```

## 1. GGUF 是什么

在本项目中，Qwen3.5 GGUF 是推理所需的单文件模型包。它主要包含：

```text
Qwen3.5-0.8B-Q4_K_M.gguf
├── GGUF 文件头
├── 模型 metadata
│   ├── architecture = qwen35
│   ├── hidden size、层数、Attention 配置
│   ├── RoPE、上下文长度和归一化参数
│   └── tokenizer 配置
├── tensor 描述
│   ├── tensor 名称
│   ├── shape
│   ├── GGML 数据类型
│   └── 文件偏移和字节数
└── tensor 数据
    ├── token embedding
    ├── 18 层 Linear Attention 权重
    ├── 6 层 Full Attention 权重
    ├── FFN 权重
    └── output norm 等权重
```

GGUF metadata 描述模型“怎么运行”，tensor 描述和数据保存模型“用什么参数
计算”。Tokenizer 也作为 metadata 保存在同一个文件中。

## 2. 当前 Qwen3.5 GGUF 的实际信息

通过项目自身的 GGUF reader 读取当前模型，可以得到：

```jsonc
{
  "format": "GGUF v3",
  "architecture": "qwen35",
  "file_size": 527502816,
  "metadata_count": 34,
  "tensor_count": 320,
  "hidden_size": 1024,
  "vocab_size": 248320,
  "total_layers": 24,
  "main_layers": 24,
  "linear_attention_layers": 18,
  "full_attention_layers": 6,
  "mtp_layers": 0,
  "context_length": 262144,
  "tokenizer_available": true
}
```

模型文件名中的 `Q4_K_M` 表示主要权重经过量化。当前文件中的 tensor 类型包括
`F32`、`Q4_K`、`Q5_K` 和 `Q6_K`；不同 tensor 可以使用不同的数据类型。

## 3. 项目如何读取 GGUF

Qwen3.5 推理入口在 `src/runtime/qwen35_runtime.cpp` 的
`generate_qwen35_metal()`。它首先加载模型配置，然后读取 GGUF：

```cpp
auto bundle = load_model_bundle(request.model_dir);
auto gguf = read_gguf_file(bundle.value().model_file);
```

两者职责不同：

- `bundle.value().model` 是从 metadata 解析出的 Qwen3.5 模型结构配置。
- `gguf.value()` 保存 GGUF metadata、tensor 清单、数据区位置等底层信息。

随后构建 weight map：

```cpp
auto map = build_qwen35_weight_map(
  bundle.value().model,
  gguf.value());
```

Weight map 将网络模块映射到 GGUF tensor 名称。例如第 0 层 Linear Attention
会绑定到：

```jsonc
{
  "index": 0,
  "kind": "linear_attention",
  "qkv": "blk.0.attn_qkv.weight",
  "gate": "blk.0.attn_gate.weight",
  "conv1d": "blk.0.ssm_conv1d.weight",
  "dt_bias": "blk.0.ssm_dt.bias",
  "a": "blk.0.ssm_a",
  "beta": "blk.0.ssm_beta.weight",
  "alpha": "blk.0.ssm_alpha.weight",
  "norm": "blk.0.ssm_norm.weight",
  "output": "blk.0.ssm_out.weight"
}
```

Weight map 只记录 tensor 的名称、shape、类型、文件偏移和大小，并不会复制
tensor 数据。后续 runtime 根据这些 binding 将权重加载为 Metal buffer。

## 4. Tokenizer 在 GGUF 里面吗

是。当前 Qwen3.5 推理不需要同目录下再提供 `tokenizer.json`。Tokenizer 的
词表、BPE merge 规则、特殊 token 和 Chat Template 都位于 GGUF metadata。

当前模型的 tokenizer 摘要为：

```jsonc
{
  "model": "gpt2",
  "pre": "qwen35",
  "token_count": 248320,
  "bos_token_id": 0,
  "eos_token_id": 248046
}
```

- `model=gpt2`：使用 GPT-2 风格的 byte-level BPE。
- `pre=qwen35`：BPE 前使用 Qwen3.5 对应的文本切分规则。
- `token_count=248320`：token 数量与模型 vocabulary size 一致。

## 5. GGUF 中的 Tokenizer metadata

`load_gguf_tokenizer()` 读取的 key 包括：

```jsonc
{
  "tokenizer.ggml.model": "gpt2",
  "tokenizer.ggml.pre": "qwen35",
  "tokenizer.ggml.tokens": ["...完整词表，共 248320 项..."],
  "tokenizer.ggml.token_type": [1, 1, 1],
  "tokenizer.ggml.merges": ["token_a token_b"],
  "tokenizer.ggml.bos_token_id": 0,
  "tokenizer.ggml.eos_token_id": 248046,
  "tokenizer.ggml.padding_token_id": "<optional token ID>",
  "tokenizer.ggml.add_bos_token": "<optional boolean>",
  "tokenizer.ggml.add_eos_token": "<optional boolean>",
  "tokenizer.chat_template": "...Qwen3.5 chat template..."
}
```

示例缩短了数组和 Chat Template。真实 GGUF 保存完整内容。除 tokens 外，部分
字段允许缺省；loader 会为缺失的可选字段设置默认值。

## 6. 加载后的 GgufTokenizer

代码中的 tokenizer 类型是 `GgufTokenizer`，定义在
`include/toyllm/runtime/gguf_tokenizer.hpp`。它逻辑上类似：

```jsonc
{
  "model": "gpt2",
  "pre": "qwen35",
  "chat_template": "...",
  "tokens": ["..."],
  "token_types": [1],
  "merges": ["token_a token_b"],

  // loader 根据 tokens 在内存中创建，GGUF 中没有这张独立映射表。
  "token_to_id": {"token_text": 123},

  // loader 根据 merges 的数组顺序在内存中创建。
  "merge_ranks": {"token_a + token_b": 0},

  "bos_token_id": 0,
  "eos_token_id": 248046,
  "pad_token_id": "<GGUF value, or -1 when absent>",
  "add_bos_token": "<GGUF value, or false when absent>",
  "add_eos_token": "<GGUF value, or false when absent>"
}
```

加载代码为：

```cpp
auto tokenizer = load_gguf_tokenizer(gguf.value());
```

返回类型是 `Result<GgufTokenizer>`，使用前必须检查 `is_ok()`。

## 7. Qwen3.5 编码与解码

完整的数据流是：

```text
ChatMessage / 普通 prompt
        │
        ├─ Chat 请求先套用 tokenizer.chat_template
        ▼
Qwen3.5 文本
        │ gguf_encode_text()
        ▼
token IDs
        │ embedding → Transformer → LM Head → sampling
        ▼
生成的 token IDs
        │ gguf_decode_token_text()
        ▼
UTF-8 输出文本
```

普通 prompt 编码：

```cpp
auto ids = gguf_encode_text(tokenizer, prompt, false, true);
```

- `add_special=false`：不自动添加 BOS/EOS。
- `parse_special=true`：识别输入中的 Qwen3.5 特殊 token。

Chat 请求先将 `role`、`content`、thinking 配置套入
`tokenizer.chat_template`，再执行相同的 BPE 编码。

生成结果解码：

```cpp
auto text = gguf_decode_token_text(tokenizer, generated_ids, true);
```

最后一个参数为 `true` 时，解码器跳过 control 和 unused 类型的 token。

## 8. 当前支持范围

当前 Qwen3.5 GGUF encoder 明确要求：

```text
tokenizer.ggml.model = gpt2
tokenizer.ggml.pre   = qwen35
```

如果 Qwen3.5 GGUF 使用其他 tokenizer model 或 pre-tokenizer，
`gguf_encode_text()` 会返回 `invalid_argument`。支持新规则时需要同步补充编码实现
和回归测试。

## 9. 本地检查命令

查看 Qwen3.5 模型 metadata 和 tokenizer 摘要：

```bash
./build/debug/kraken-infer inspect \
  models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
```

查看 tensor 清单和 Qwen3.5 weight map 校验结果：

```bash
./build/debug/kraken-infer weights \
  models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
```
