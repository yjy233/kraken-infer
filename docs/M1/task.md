# M1 Task: Model Structure And Project Introspection

目标：让框架能识别本地 Qwen3 0.6B 模型目录，读取结构配置，并在不加载权重的情况下完成模型摘要和基础一致性校验。

## Scope

本阶段只做模型结构读取和校验，不实现 tokenizer encode、权重读取、forward、KV cache 或 MPS 算子。

输入目录：

```text
models/qwen3-0.6b/
```

需要读取：

```text
config.json
generation_config.json
tokenizer_config.json
vocab.json
```

## Tasks

- 定义 `ModelConfig`
- 定义 `GenerationConfig`
- 定义 `TokenizerInfo`
- 实现模型目录加载函数
- 解析 `config.json`
- 解析 `generation_config.json`
- 读取 tokenizer base vocab 数量
- 读取 tokenizer added token 数量
- 校验 Qwen3 结构维度
- 校验 tokenizer vocab size 与 `config.json` 对齐
- CLI 增加模型检查入口
- smoke test 覆盖 Qwen3 0.6B 本地配置

## Validation Rules

- `architecture == Qwen3ForCausalLM`
- `model_type == qwen3`
- `num_attention_heads * head_dim > 0`
- `num_key_value_heads * head_dim > 0`
- `num_attention_heads % num_key_value_heads == 0`
- `num_hidden_layers > 0`
- `intermediate_size > hidden_size`
- `vocab_size > 0`
- `tokenizer explicit vocab <= vocab_size`
- tokenizer max token id 小于 `vocab_size`
- tokenizer explicit vocab 与 `vocab_size` 的差值作为 reserved vocab slots 输出
- `rms_norm_eps > 0`
- `temperature > 0`
- `0 < top_p <= 1`
- `top_k >= 0`
- `eos_token_id` 非空

## CLI

目标命令：

```bash
./build/manual/toyllm --inspect-model models/qwen3-0.6b
```

输出应包含：

- model path
- architecture
- model type
- layers
- hidden size
- attention heads
- KV heads
- GQA group size
- attention projection size
- KV projection size
- intermediate size
- max position embeddings
- dtype
- RoPE theta
- vocab size
- tokenizer vocab summary
- reserved vocab slots
- default generation parameters
- validation result

## Acceptance

- 本地模型目录存在时，`--inspect-model models/qwen3-0.6b` 返回成功
- 输出准确展示 Qwen3 0.6B 结构
- 配置文件缺失时返回明确错误
- 配置维度不一致时返回明确错误
- `make test` 通过
