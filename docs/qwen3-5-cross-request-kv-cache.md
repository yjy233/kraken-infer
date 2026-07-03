# Qwen3.5 跨请求 KV / Prefix Cache 方案

本文档规划 Qwen3.5 0.8B GGUF Metal 路径的跨请求 cache。目标是参考
`~/code/llama.cpp` 的 slot/cell、prompt cache 和 recurrent checkpoint 思路，
在本仓库实现一个可逐步演进的 block/prefix cache。

## 目标

第一版目标：

- `/v1/chat/completions` 和 `/v1/completions` 能复用上一批请求留下的相同
  token prefix。
- 复用粒度是固定长度 block，第一版只复用完整 block prefix。
- 命中后跳过已缓存 prefix 的 prefill，只计算 miss suffix 和 decode。
- Qwen3.5 hybrid cache 必须同时恢复：
  - full attention 层的 K/V cache。
  - linear attention 层的 recurrent R/S state。
- 不改变无 cache 时的输出。固定 seed/greedy 下，cache 和 no-cache 输出应一致。

第一版非目标：

- 不做任意位置 chunk shifting。
- 不做多请求并发共享同一 cache。
- 不做 vLLM-style block table attention kernel。
- 不支持 fuzzy prefix、prompt canonicalization 或跨模型共享 cache。
- 不把 cache 持久化到磁盘。

## 当前代码约束

当前 Qwen3.5 GGUF 路径入口是
`src/runtime/qwen35_runtime.cpp` 的 `generate_qwen35_metal()`。它现在是每次请求：

1. 读取 GGUF / tokenizer / weight map。
2. 创建 `mps::MpsContext`。
3. 分配 `Qwen35MetalCache`。
4. 上传 Metal weights。
5. 执行完整 prompt prefill。
6. 执行 decode。

这意味着跨请求 cache 的第一步不是直接保存某个 buffer，而是把 runtime 改成
server/session 生命周期：

```text
OpenAI gateway process
  -> Qwen35RuntimeSession
     -> tokenizer / GGUF metadata / weight map
     -> MpsContext
     -> Metal weights
     -> active request cache
     -> prefix block cache
```

当前 full attention K/V layout 是按绝对位置写入：

```text
key_cache/value_cache:
  [full_attention_layer][capacity_tokens][kv_dim]
```

当前 recurrent cache 的逻辑 offset 只按 linear layer 计算：

```text
recurrent_r_offset = recurrent_layer_index * recurrent_r_elements_per_layer
recurrent_s_offset = recurrent_layer_index * recurrent_s_elements_per_layer
```

但 `Qwen35CachePlan` 已经有 `recurrent_rows`。跨请求 cache 需要把 recurrent
layout 明确成：

```text
recurrent_r_cache:
  [linear_attention_layer][recurrent_row][recurrent_r_elements_per_layer]

recurrent_s_cache:
  [linear_attention_layer][recurrent_row][recurrent_s_elements_per_layer]
```

active request 用 row 0。prefix block 的 snapshot 存在独立 arena，restore 时复制回
active row 0。

## llama.cpp 参考点

参考树：`~/code/llama.cpp`。

关键文件：

- `src/llama-kv-cache.h`
  - `llama_kv_cache::slot_info`
  - `prepare()`
  - `find_slot()`
  - `apply_ubatch()`
  - `seq_rm()`
  - `seq_cp()`
  - `seq_add()`
- `src/llama-kv-cache.cpp`
  - cell/slot 元数据和 cache position 管理。
  - `get_n_kv()` 会按 padding 取当前实际 cache 长度，便于复用 graph。
- `tools/server/server-context.cpp`
  - `cache_prompt` 通过 common prefix 跳过已有 prompt。
  - `n_cache_reuse` 通过 KV shifting 复用中间 chunk。
  - recurrent/SWA 不能 rollback 时恢复 checkpoint。
- `common/common.h` / `common/common.cpp`
  - `common_prompt_checkpoint` 保存 sequence state。
- `src/llama-memory-recurrent.cpp`
  - recurrent state 使用 `n_rs_seq` 和 `rs_idx` 支持有限 rollback。

本仓库第一版不照搬完整 `seq_add` shifting。原因是 Qwen3.5 full attention K/V 已经
经过 MRoPE，active cache 里保存的是带 absolute position 的 K，直接把中间 token 搬到
新位置会有正确性风险。第一版只复用“相同 token 在相同 absolute position”的 prefix。

### Qwen3.5 0.8B LV 对 cache 的影响

Qwen3.5 0.8B 的 text+image 路径在 llama.cpp 中不是把图片转成普通 token id：

- text decoder 仍是 `qwen35`。
- image 侧由单独 mmproj GGUF 负责 vision encoder 和 projector。
- mtmd 把 prompt 拆成 text chunk 和 image chunk。
- text chunk 进入 token embedding path。
- image chunk 先编码成 projected embedding，然后作为 raw embedding batch 喂给
  decoder。
- 对 MRoPE decoder，text token 的 position 会 broadcast 到所有 section；image
  embedding 使用二维 position。

因此跨请求 cache 的第一版不能继续假设 prompt prefix 是
`std::vector<std::int64_t> tokens`。纯文本请求可以继续走 token block key；一旦请求
包含图片，block key 必须描述 decoder 实际看到的多模态 prefix。

## Block Cache 设计

### Cache Key

每个 block 是 prefix chain 的一个节点。key 必须包含：

- model fingerprint：GGUF 路径、文件大小、mtime 或 GGUF metadata hash。
- tokenizer fingerprint：vocab / merges / chat template hash。
- runtime fingerprint：KV dtype、block size、Qwen3.5 layout version。
- mmproj fingerprint：mmproj GGUF 路径、文件大小、mtime 或 metadata hash；纯文本
  请求可为空。
- parent block hash。
- 当前 block 的 decoder items：
  - text item: token id。
  - image item: image bytes hash 或调用方提供的稳定 image id。
  - image item: preprocessor parameters、输出 grid `nx/ny`、temporal merge、logical
    position count。
  - embedding item: projected embedding hash，或能唯一推出该 embedding 的
    image/mmproj/preprocess fingerprint。
- 当前 block 的 decoder item count。

hash 只用于索引。命中后必须校验保存的 token ids / image ids / grid metadata，
避免 hash collision 或图像 preprocess 差异导致错误复用。

第一阶段可以先规定：包含 image chunk 的请求只允许在 image chunk 边界处切 block，
并且不跨 image chunk 做半个 block 的复用。后续再支持 text/image 混排 block。

### Metadata

建议新增 `include/toyllm/runtime/qwen35_prefix_cache.hpp`：

```cpp
struct Qwen35PrefixCacheConfig {
  bool enabled{false};
  std::size_t block_tokens{1024};
  std::size_t capacity_blocks{16};
  std::size_t min_reuse_tokens{0};
};

struct Qwen35PrefixCacheStats {
  std::size_t hit_tokens{0};
  std::size_t miss_tokens{0};
  std::size_t committed_tokens{0};
  std::size_t evicted_blocks{0};
};

struct Qwen35PrefixBlock {
  std::uint64_t hash{0};
  std::uint64_t parent_hash{0};
  std::size_t slot{0};
  std::size_t token_start{0};
  std::size_t token_count{0};
  std::vector<std::int64_t> tokens;
  std::uint64_t last_used_tick{0};
};
```

`slot` 指向 prefix arena 中的物理 block slot。

### Device Arena

prefix cache 不建议第一版放 host RAM，否则每次命中都要 CPU read/write，实际 prefill
收益会被 PCIe/UMA 同步和 command buffer 提交抵消。

建议新增一个 device-resident arena：

```text
prefix_key_cache:
  [cache_block_slot][full_attention_layer][block_tokens][kv_dim]

prefix_value_cache:
  [cache_block_slot][full_attention_layer][block_tokens][kv_dim]

prefix_recurrent_r:
  [cache_block_slot][linear_attention_layer][recurrent_r_elements_per_layer]

prefix_recurrent_s:
  [cache_block_slot][linear_attention_layer][recurrent_s_elements_per_layer]

prefix_last_hidden:
  [cache_block_slot][hidden_size]
```

需要在 `mps::MpsContext` 增加通用 byte-region copy：

```cpp
Status copy_buffer_region(const MpsBuffer& source,
                          MpsBuffer& destination,
                          std::size_t source_byte_offset,
                          std::size_t destination_byte_offset,
                          std::size_t byte_count) const;
```

现有 `copy_f32_region()` / `copy_f32_rows()` 只适合 F32 buffer，不适合 F16 KV 和混合
layout。通用 copy 可用 Metal blit encoder 实现，stub 后端做参数校验后返回
`unavailable` 或 no-op，取决于测试需要。

## 请求流程

### 1. Tokenize

仍使用当前逻辑：

- chat 请求先 `format_qwen35_chat_prompt()`。
- 再 `gguf_encode_text()`。

thinking 开关无需额外进入 cache key，因为它已经改变 chat prompt token 序列。但
chat template 本身必须进入 tokenizer fingerprint。

### 2. Find Prefix

把 prompt tokens 按 `block_tokens` 切分，只匹配完整 block：

```text
prefix_len = 0
parent_hash = root

for each complete block in prompt:
  block_hash = hash(parent_hash, block_tokens)
  entry = lookup(block_hash)
  if entry not found:
    break
  if entry.tokens != block_tokens:
    break
  prefix_len += block_tokens
  parent_hash = block_hash
```

如果 `prefix_len < min_reuse_tokens`，视为 miss，避免很小 prefix 反而增加 copy 成本。

### 3. Restore Active Cache

命中 `N` 个 blocks 后，把它们复制回 active cache 的相同 absolute positions：

```text
for block i in [0, N):
  active token range = [i * block_tokens, (i + 1) * block_tokens)
  copy prefix_key/value slot -> active key/value range

restore recurrent R/S from block N-1 snapshot -> active row 0
restore last_hidden from block N-1 if prompt exactly ends at prefix_len
```

第一版必须保证 active cache capacity 固定且大于 `prompt_tokens + max_new_tokens`。在
gateway 下建议以 `OpenAIGatewayConfig.context_size` 建 session capacity；未设置时按
第一次请求的需求建 session，但后续大请求可能需要重建 session cache 并清空 prefix
cache。

### 4. Prefill Miss Suffix

从 `prefix_len` 开始执行原有 chunked prefill。

第一版建议让 `block_tokens == prefill_chunk_tokens`，这样每个 prefill chunk 结束时天然
是 block 边界，可以直接 commit snapshot。后续如果要 block size 小于 chunk size，需要
在 chunk 内额外切分或支持 mid-chunk recurrent snapshot。

如果 prompt 完全命中完整 block：

- 不执行 prefill。
- 使用 cached `last_hidden` 计算 prompt logits。
- decode position 从 `prompt_tokens` 开始。

如果 prompt 有 miss suffix：

- 使用当前 prefill suffix 的最后 hidden 计算 logits。
- decode position 从 `prompt_tokens` 开始。

### 5. Commit New Blocks

prefill suffix 每完成一个完整 block，就 commit：

1. 计算 block hash。
2. 分配或复用一个 LRU slot。
3. 从 active K/V 的对应 absolute range 复制到 prefix arena。
4. 从 active recurrent row 0 复制 R/S snapshot 到 prefix arena。
5. 保存 block end 的 `last_hidden`。
6. 更新 metadata 和 LRU tick。

第一版只 commit prompt tokens。后续可以把 generated tokens 也串到同一 prefix chain，
但需要确认 chat template 中 assistant message 的 re-tokenization 与 decode token 序列
完全一致，否则不能安全复用。

## Runtime Session 改造

建议新增：

```cpp
class Qwen35RuntimeSession {
 public:
  static Result<std::unique_ptr<Qwen35RuntimeSession>> create(
    const Qwen35SessionConfig& config);

  Result<CpuGenerationResult> generate(const CpuGenerationRequest& request);
};
```

`Qwen35SessionConfig` 应来自 gateway config：

- `model_dir`
- `context_size`
- `prefill_chunk_tokens`
- `cache_prompt`
- `cache_block_tokens`
- `cache_capacity_blocks`
- `cache_reuse_min_tokens`

保留现有 `generate_qwen35_metal(request)` 作为 stateless CLI 路径。gateway 优先使用
session path。

第一版 gateway 串行执行：

```text
OpenAIGatewayRuntime
  -> mutex
  -> Qwen35RuntimeSession::generate()
```

`parallel_slots > 1` 可以先仍然串行，或直接返回 unsupported。真正多 slot 需要第二阶段
实现 active slot cache 和 block refcount。

## API 和配置

建议对齐 llama.cpp 命名：

CLI server flags：

- `--cache-prompt` / `--no-cache-prompt`
- `--cache-block-tokens N`
- `--cache-capacity-blocks N`
- `--cache-reuse N`

OpenAI-compatible request fields：

```json
{
  "cache_prompt": true,
  "n_cache_reuse": 1024
}
```

字段语义：

- `cache_prompt=false`：本请求不查 cache，也不 commit 新 blocks。
- `cache_prompt=true`：允许 exact prefix block cache。
- `n_cache_reuse`：第一版作为最小复用 token 数，不实现 llama.cpp 的中间 chunk shifting。

响应观测：

- HTTP header：
  - `X-Kraken-Prompt-Cache-Hit-Tokens`
  - `X-Kraken-Prompt-Cache-Miss-Tokens`
- profile metadata：
  - `prompt_cache_enabled`
  - `prompt_cache_hit_tokens`
  - `prompt_cache_miss_tokens`
  - `prompt_cache_committed_tokens`
  - `prompt_cache_blocks`
- OpenAI-compatible usage：
  - `usage.prompt_tokens_details.cached_tokens`

`cached_tokens` 统计已经从跨请求 cache 恢复的 decoder prompt 单元。纯文本时它等同于
token 数；多模态时第一版仍按 decoder 输入单元计数，并必须 clamp 到
`usage.prompt_tokens`。

## 实现拆分

### 阶段 A: Buffer copy 和 recurrent row layout

文件：

- `include/toyllm/backends/mps/mps_backend.hpp`
- `src/backends/mps/mps_backend.mm`
- `src/backends/mps/mps_backend_stub.cpp`
- `src/runtime/qwen35_runtime.cpp`

任务：

1. 增加 `copy_buffer_region()`。
2. 增加 recurrent offset helper：

   ```cpp
   recurrent_offset(layer, row, rows, elements_per_layer)
   ```

3. 把 linear attention decode/chunk path 的 R/S offset 都改成 row-aware。
4. 默认仍使用 row 0，先不改变行为。
5. smoke test 覆盖 helper 边界和 overflow。

### 阶段 B: Stateless Prefix Cache 组件

文件：

- `include/toyllm/runtime/qwen35_prefix_cache.hpp`
- `src/runtime/qwen35_prefix_cache.cpp`
- `tests/smoke_test.cpp`

任务：

1. 实现 token block hash。
2. 实现 longest complete prefix match。
3. 实现 LRU slot 分配。
4. 实现 metadata-only 单测，不依赖模型权重和 MPS。

### 阶段 C: Runtime Session

文件：

- `include/toyllm/runtime/qwen35_runtime.hpp`
- `src/runtime/qwen35_runtime.cpp`
- `src/runtime/openai_gateway.cpp`
- `include/toyllm/runtime/openai_gateway.hpp`

任务：

1. 抽出模型加载、context 创建、weights upload 到 `Qwen35RuntimeSession`。
2. 让 session 持有 fixed-capacity active cache。
3. gateway 启动时创建 session。
4. request path 通过 session generate。
5. stateless CLI path 保持可用。

### 阶段 D: Device Prefix Arena 接入

文件：

- `src/runtime/qwen35_runtime.cpp`
- `src/runtime/qwen35_prefix_cache.cpp`

任务：

1. 分配 prefix K/V/R/S/hidden arena。
2. 命中时 restore active cache。
3. prefill chunk 后 commit block。
4. 完整命中时使用 cached last hidden 算 logits。
5. profile/header 输出 hit/miss stats。

### 阶段 E: 测试和性能验证

测试：

- metadata 单测：
  - 完全相同 prompt 命中全部完整 blocks。
  - 不同 suffix 命中公共 prefix blocks。
  - block token mismatch 不命中。
  - LRU 淘汰后不命中。
- 集成测试：
  - 第一次请求 miss，第二次相同请求 hit。
  - 相同 system prompt、不同 user prompt 命中 system prefix。
  - `cache_prompt=false` 不 hit、不 commit。
- 正确性：
  - greedy no-cache 与 cache 输出一致。
  - sampled fixed seed 输出一致，或至少 logits top-k 一致。
- 性能：
  - 第二次长 prompt prefill span 明显下降。
  - restore copy time 单独 profile。

## 风险和处理

- **MRoPE absolute position 风险**：第一版只复用相同 absolute position 的 prefix，不做
  KV shifting。
- **多模态 key 风险**：image 请求不能只按文本 token 命中。key 必须包含 image id/hash、
  mmproj fingerprint、preprocess/grid metadata 和 MRoPE position 语义。
- **recurrent state 风险**：只在 block 边界保存 R/S snapshot，prefix hit 只能停在完整
  block 边界。
- **cache capacity 风险**：session active cache 必须固定 capacity；大请求需要拒绝或重建
  session cache 并清空 prefix cache。
- **并发风险**：第一版串行。并发需要 active slots、block refcount 和 per-slot recurrent
  rows。
- **commit 生成 token 风险**：第一版只 commit prompt block，避免 assistant text
  re-tokenization 不一致。
- **小 prompt 负收益**：`min_reuse_tokens` 默认可设为一个 block，低于阈值不 restore。

## 第二阶段演进

第二阶段再接近 llama.cpp/vLLM：

- 多 active slots，使用 `parallel_slots`。
- prefix block refcount 和 pinned blocks。
- generated token chain commit。
- 支持中间 chunk reuse，但需要正确处理 MRoPE 和 recurrent snapshot。
- attention kernel 支持 block table，避免 restore 到 contiguous active cache。
- recurrent rollback 支持 decode/speculative rejection。
- prompt checkpoint 策略对齐 llama.cpp 的 `common_prompt_checkpoint`，用于长 prompt
  分叉和 hybrid memory rollback。
