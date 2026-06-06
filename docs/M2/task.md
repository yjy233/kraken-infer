# M2 Task: Safetensors Weight Reader

目标：让框架能读取 Qwen3 0.6B 的 `model.safetensors` 权重元数据，建立权重名称到只读 tensor view 的映射，并完成 shape 校验。本阶段不做 forward，不做权重计算，不把整个权重文件复制进内存。

## Scope

输入目录：

```text
models/qwen3-0.6b/
```

需要读取：

```text
model.safetensors
config.json
```

依赖 M1：

- `ModelConfig`
- `load_model_bundle`
- Qwen3 结构参数校验

非目标：

- 不实现 tokenizer
- 不实现 CPU forward
- 不实现 MPS weight upload
- 不实现量化
- 不实现多 safetensors shard

## Tasks

### File Format

- [x] 定义 `SafeTensorDType`
- [x] 定义 `SafeTensorInfo`
- [x] 定义 `SafeTensorFile`
- [x] 定义 `TensorView`
- [x] 读取 safetensors 前 8 字节 header length
- [x] 校验 header length 不超过文件大小
- [x] 读取 safetensors JSON header
- [x] 解析 header 中每个 tensor 的 `dtype`
- [x] 解析 header 中每个 tensor 的 `shape`
- [x] 解析 header 中每个 tensor 的 `data_offsets`
- [x] 跳过或记录 `__metadata__`
- [x] 校验 `data_offsets` 是两个整数
- [x] 校验 tensor offset 位于文件范围内
- [x] 校验 tensor byte size 与 dtype/shape 一致
- [x] 校验 tensor offset 不互相重叠
- [x] 校验 tensor offset 按文件范围覆盖合理

### Memory Mapping

- [x] 实现只读文件打开
- [x] 实现文件大小查询
- [x] 实现 mmap 只读映射
- [x] 实现 mmap 资源释放
- [x] 禁止把整个权重文件复制到普通 heap buffer
- [x] `TensorView` 只保存指针、字节长度、dtype、shape、name
- [x] 支持按 tensor name 查询 view

### Qwen3 Weight Mapping

- [x] 定义 `Qwen3LayerWeights`
- [x] 定义 `Qwen3Weights`
- [x] 映射 token embedding 权重
- [x] 映射每层 `input_layernorm.weight`
- [x] 映射每层 `self_attn.q_proj.weight`
- [x] 映射每层 `self_attn.k_proj.weight`
- [x] 映射每层 `self_attn.v_proj.weight`
- [x] 映射每层 `self_attn.o_proj.weight`
- [x] 映射每层 `post_attention_layernorm.weight`
- [x] 映射每层 `mlp.gate_proj.weight`
- [x] 映射每层 `mlp.up_proj.weight`
- [x] 映射每层 `mlp.down_proj.weight`
- [x] 映射 final norm 权重
- [x] 处理 `tie_word_embeddings=true` 的 lm head 逻辑
- [x] 如果存在独立 `lm_head.weight`，则单独映射
- [x] 如果没有独立 `lm_head.weight`，则 lm head 指向 token embedding

### Shape Validation

- [x] 校验 token embedding shape
- [x] 校验 q projection shape
- [x] 校验 k projection shape
- [x] 校验 v projection shape
- [x] 校验 o projection shape
- [x] 校验 MLP gate projection shape
- [x] 校验 MLP up projection shape
- [x] 校验 MLP down projection shape
- [x] 校验 RMSNorm weight shape
- [x] 校验 final norm shape
- [x] 校验 lm head shape 或 tied embedding
- [x] 校验 layer 数量等于 `num_hidden_layers`
- [x] 校验所有必需权重都存在
- [x] 校验 dtype 与模型 dtype 兼容

### CLI

- [x] 增加 `weights` 子命令
- [x] `kraken-infer weights` 默认读取 `models/qwen3-0.6b`
- [x] `kraken-infer weights <model_dir>` 支持指定模型目录
- [x] 输出 safetensors 文件大小
- [x] 输出 tensor 数量
- [x] 输出 dtype 统计
- [x] 输出前若干 tensor name/shape/dtype
- [x] 输出 Qwen3 weight mapping summary
- [x] 输出 validation result
- [x] `doctor` 子命令包含 weights 检查
- [x] 保留 M1 `inspect` 行为不变

### Tests

- [x] smoke test 覆盖 safetensors header 读取
- [x] smoke test 覆盖 tensor 数量大于 0
- [x] smoke test 覆盖 token embedding 存在
- [x] smoke test 覆盖所有 28 层权重映射存在
- [x] smoke test 覆盖 shape 校验通过
- [x] smoke test 覆盖 tied lm head 行为
- [x] regression test 覆盖 invalid header length
- [x] regression test 覆盖 missing required tensor
- [x] regression test 覆盖 shape mismatch

## Expected Qwen3 Weight Names

初步按 Hugging Face Qwen3 命名约定：

```text
model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

如果实际 safetensors 中命名不同，以实际 header 为准，并更新本文档。

## Expected Shapes

基于 Qwen3 0.6B 配置：

```text
hidden_size = 1024
num_attention_heads = 16
num_key_value_heads = 8
head_dim = 128
attention_projection_size = 2048
kv_projection_size = 1024
intermediate_size = 3072
vocab_size = 151936
num_hidden_layers = 28
```

预期 shape：

```text
model.embed_tokens.weight                  [151936, 1024]
model.layers.{i}.input_layernorm.weight    [1024]
model.layers.{i}.self_attn.q_proj.weight   [2048, 1024]
model.layers.{i}.self_attn.k_proj.weight   [1024, 1024]
model.layers.{i}.self_attn.v_proj.weight   [1024, 1024]
model.layers.{i}.self_attn.o_proj.weight   [1024, 2048]
model.layers.{i}.post_attention_layernorm.weight [1024]
model.layers.{i}.mlp.gate_proj.weight      [3072, 1024]
model.layers.{i}.mlp.up_proj.weight        [3072, 1024]
model.layers.{i}.mlp.down_proj.weight      [1024, 3072]
model.norm.weight                          [1024]
lm_head.weight                             [151936, 1024] or tied
```

## Acceptance

- [x] `make test` 通过
- [x] `./build/manual/kraken-infer weights` 返回成功
- [x] `./build/manual/kraken-infer doctor` 包含模型结构和权重检查
- [x] 输出 tensor 总数、dtype 统计、Qwen3 映射摘要
- [x] 不把整个 `model.safetensors` 复制进 heap
- [x] 所有 Qwen3 0.6B 必需权重 shape 校验通过
- [x] 缺失文件或损坏 header 时返回明确错误
