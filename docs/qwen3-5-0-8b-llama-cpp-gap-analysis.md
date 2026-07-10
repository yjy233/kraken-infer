# Qwen3.5 0.8B llama.cpp 对照差距调研

调研对象：

- llama.cpp：`~/code/llama.cpp`，HEAD `be4a6a63e`
- kraken-infer：`feat/mtp`，HEAD `2f72487`

## llama.cpp 的 Qwen3.5 dense 文本路径

llama.cpp 的 dense text decoder 使用 arch `qwen35`：

- `src/llama-arch.cpp` 注册 `LLM_ARCH_QWEN35 -> "qwen35"`，并注册
  `blk.%d.nextn.*` MTP tensor 名。
- `src/models/qwen35.cpp` 是主实现。
- tokenizer 侧用 `tokenizer.ggml.pre = "qwen35"`，对应
  `unicode_regex_split_custom_qwen35()`；它和 Qwen2 的关键差别是 letter run
  包含 Unicode combining mark。

`llama_model_qwen35::load_arch_hparams()` 做的事：

- 读取 `attention.layer_norm_rms_epsilon`。
- 读取 `rope.dimension_sections`，Qwen3.5 默认由 converter 写入
  `[11, 11, 10, 0]`。
- 读取 Gated Delta Net 相关 `ssm.*` 参数。
- 读取 `nextn_predict_layers`，把 MTP 层视为追加在主干层后的 decoder block。
- 从 `attention.recurrent_layers` 或 `full_attention_interval` 推导主干层的
  recurrent/full-attention 类型；MTP 层必须是 non-recurrent full-attention。

`llama_model_qwen35::load_arch_tensors()` 的权重布局：

- `token_embd.weight`
- `output_norm.weight`
- `output.weight` 可选；缺失时 tie 到 token embedding。
- 主干层：
  - recurrent 层加载 `attn_qkv`、`attn_gate`、`ssm_*`、FFN。
  - full-attention 层加载 joint Q/G projection、K/V、Q/K norm、attention out、FFN。
- MTP 层：
  - 作为追加层 `il >= n_layer`。
  - 加载一个完整 full-attention block。
  - 额外加载 `nextn.eh_proj`、`nextn.enorm`、`nextn.hnorm`。
  - `nextn.embed_tokens`、`nextn.shared_head_head`、`nextn.shared_head_norm` 是可选；
    缺失时回退到主模型 token embedding / output head / output norm。

kraken 的 `qwen35_weight_map.cpp` 在这些点上基本对齐：同样按
`nextn_predict_layers` 切 `main_layer_count`，主干 recurrent/full-attention 与追加
MTP 层的 tensor shape 也按 llama.cpp 绑定。

## llama.cpp 的主图与 MTP 图

llama.cpp 主 decoder graph：

1. `build_inp_embd(model.tok_embd)` 得到 token embedding。
2. 对 `il < n_layer` 执行主干层，跳过追加的 MTP 层。
3. 每层先 RMSNorm，再按 `is_recr(il)` 选择 Gated Delta Net 或 full attention。
4. full attention 使用 joint Q/G projection：
   `[q_head0, gate_head0, q_head1, gate_head1, ...]`。
5. Q/K 走 `ggml_rope_multi()`，使用 Qwen3.5 MRoPE sections。
6. 最后 `output_norm` 后同时产出：
   - `t_h_nextn`：MTP 需要的 hidden row。
   - `t_logits`：LM head logits。

llama.cpp MTP graph：

1. 输入 token id 或 raw embedding。
2. 输入主模型上一位置的 `h_nextn`。
3. token embedding 使用 `nextn.embed_tokens`，缺失则用主 `tok_embd`。
4. `h_nextn` 经 `nextn.hnorm`，token embedding 经 `nextn.enorm`。
5. concat 后过 `nextn.eh_proj`。
6. 跑追加 MTP full-attention block。
7. 用 `nextn.shared_head_norm` 或主 `output_norm` 做 head norm。
8. 用 `nextn.shared_head_head` 或主 `output.weight` 做 logits。
9. 当前 dense Qwen35 限制 `n_layer_nextn == 1`。

kraken 当前 `run_qwen35_mtp_tokens*()` 也按同一结构实现，语义上基本一致。

## llama.cpp 的 MTP speculative 调度

llama.cpp 的 MTP 不是在 qwen35 model 文件里直接完成完整投机采样，而是在
`common/speculative.cpp` 的 `common_speculative_impl_draft_mtp` 中调度。

关键点：

- target context 和 draft/MTP context 是两个上下文。
- `llama_set_embeddings_nextn(ctx_tgt, true, masked=false)` 让 target decode 输出
  每个 token 的 `h_nextn`。
- `llama_set_embeddings_nextn(ctx_dft, true, masked=true)` 让 draft context 输出
  draft token 对应的 `h_nextn`。
- `process()` 在 target batch decode 后抓取 `ctx_tgt` 的 `h_nextn`，并维护
  `pending_h`。
- 独立 draft context 会用 target `h_nextn` 右移后的行去 catch up 自己的 MTP KV。
- `draft()` 逐步调用 `llama_decode(ctx_dft, batch)` 生成 draft token。
- draft 采样默认使用 top-k sampler，并支持 `p_min`：低置信 draft 会提前停止。
- target verify 一次 decode `[id_last, draft0, ...]`，再调用
  `common_sampler_sample_and_accept_n()` 逐行比较。
- partial reject 时，llama.cpp server/simple 示例会通过 seq rm 或 checkpoint 机制
  回滚 target/draft context。

kraken 当前实现的相同点：

- 已经 batch verify `[sampled_token + draft_tokens]`。
- 已经维护 `pending_h`。
- 已经对 recurrent R/S state 做 snapshot/restore，partial reject 后重新 batch commit
  accepted prefix。
- full-accepted span 会保留 target verify 写入的 cache。

主要差距：

1. kraken 是单 runtime 手写 loop，没有通用 target/draft context 抽象。
2. kraken 已补 `p_min` top-1 confidence gate，但还没有 llama.cpp common sampler 风格的
   top-k draft sampling。
3. kraken 的 draft sampling 现在是 device argmax，只适合 greedy 对齐；llama.cpp
   走 common sampler，可接 sampling/grammar/backend sampler。
4. kraken 只支持单 active request/slot；llama.cpp speculative 框架支持多 seq/slot。
5. kraken 的 MTP KV 只在本地 loop 中管理；llama.cpp 把 MTP cache 作为独立 context
   memory 管理，状态保存/恢复更系统。

## 视觉路径对照

llama.cpp 的 Qwen3.5 视觉不是在 `qwen35.cpp` 里实现，也不是把图片转成普通 token id。
它的结构是：

- text GGUF：`general.architecture = "qwen35"`。
- mmproj GGUF：`general.architecture = "clip"`，
  projector type 是 `qwen3vl_merger`。
- converter 把 `Qwen3_5ForConditionalGeneration` 的 mmproj 路径映射到
  `qwen3vl` converter。
- `qwen3vl.py` 会跳过 text 与 `mtp.*` tensors，只保留 `visual.*`。

llama.cpp mtmd 运行时：

1. mtmd 载入 text model 和 mmproj。
2. 校验 `clip_n_mmproj_embd(mmproj)` 等于 text model input embedding width。
3. Qwen3VL projector 使用 dynamic-size image preprocessor。
4. vision graph `tools/mtmd/models/qwen3vl.cpp` 执行 patch embed、temporal/spatial
   merge、learned position resize、vision transformer、deepstack、`mm.0 -> GELU -> mm.2`
   projector。
5. image chunk 产出 raw F32 embedding。
6. raw embedding 通过 `llama_batch.embd` 喂给同一个 qwen35 decoder。
7. image chunk 使用 4 路 MRoPE position；logical position 推进遵循 mtmd 的 image grid
   规则，而不是简单按 token 数。

kraken 当前视觉实现：

- 已能解析 OpenAI `image_url` content array、data URL、图片 fingerprint。
- 已能读取并校验 `qwen3vl_merger` mmproj metadata。
- 当前真正推理是外部调用 `llama-mtmd-cli` 的桥接路径。
- 进入 VL bridge 后，MTP 被显式关闭，原因是 `vl_bridge_uses_llama_mtmd_without_mtp`。

因此，当前 kraken 和 llama.cpp 的最大功能差距是原生视觉：kraken 还没有
mmproj/vision graph、image embedding injection、image MRoPE position 与多模态 cache。

## 性能差距判断

当前 kraken MTP 实测仍慢于 no-MTP：

- no-MTP：约 12.06s / 64 tokens。
- MTP k=1：约 13.87s。
- MTP k=3：约 15.18s。

从 llama.cpp 对照看，可能的性能差距来源不是 MTP 公式本身，而是：

1. `p_min > 0` 可以提前停止低置信 draft，当前 top-1 probability 已在 GPU 端并行
   reduction，不再回读整条 logits；但 LM head 仍会 materialize 完整 vocab logits，
   尚未变成低成本 fused top-1 head。
2. MTP draft 每步仍完整跑 MTP block + full vocab head。
3. kraken 的 custom Metal kernels 没有 ggml/llama.cpp 那套成熟 scheduler 与 memory
   复用能力。
4. kraken 当前没有多 token/多 seq 的统一 speculative 调度，很多状态管理在请求 loop 内。
5. VL bridge 走外部进程，不可能和本地 MTP 共享 context/cache。

## 优先级建议

## 本轮落地状态

已完成：

- `CpuGenerationRequest` / CLI / OpenAI gateway 增加 `mtp_p_min`。
- Qwen35 MTP draft 支持 top-1 softmax probability gate：
  - 默认 `mtp_p_min = 0`，保持原 device argmax 快路径。
  - `mtp_p_min > 0` 时回读 draft logits 计算 greedy probability，低于阈值则停止本轮
    draft，不送入 target verify。
- `CpuMtpReport`、CLI 文本输出、OpenAI response headers 和 profiler metadata 增加：
  - `p_min`
  - `confidence_stops`
  - `verified_by_position`
  - `accepted_by_position`
- `scripts/test_qwen35_mtp_gateway.py` 支持 `--p-min`，可用于测 no-gate 与 gated MTP。
- MPS backend 增加并行 `argmax_f32_i32` 和 `argmax_prob_f32_i32`：
  - greedy argmax 不再由单 GPU 线程串行扫描 vocab logits。
  - `mtp_p_min > 0` 只回读 top-1 token/probability 两个标量，不再把整条 draft logits
    拷回 CPU。

仍未完成：

- MTP draft top-k/common sampler parity。
- fused top-1 draft head。当前概率 gate 已无 CPU logits readback，但仍需要完整 vocab
  logits buffer。
- native VL + native MTP 共用同一个 kraken context。当前图片请求仍走
  `llama-mtmd-cli` bridge，进入 bridge 后会禁用 kraken MTP。

## 优先级建议

短期优先：

1. 增加 `scripts/compare_qwen35_llamacpp.py`：
   - 同 prompt。
   - 同 GGUF。
   - no-MTP logits/top token 对齐。
   - MTP draft/accept 数对齐。
   - 记录 wall time、prefill、decode、draft。
2. 用 `scripts/test_qwen35_mtp_gateway.py --p-min` 做阈值 sweep，记录
   no-MTP、MTP `p_min=0`、MTP `p_min=0.10/0.20/0.30` 的 wall time 和 acceptance。
3. 对 MTP vocab head 做 fused top-1/argmax path，避免完整 vocab logits
   materialization 成为 draft 成本瓶颈。

中期：

1. 把 MTP loop 拆成 target context / draft context 风格的内部抽象。
2. 支持 image embedding batch 输入到 native qwen35 decoder。
3. 实现 qwen3vl_merger vision graph，替代 `llama-mtmd-cli` 外部进程。

长期：

1. 原生多 slot speculative。
2. 多模态 prefix cache：cache key 纳入 mmproj fingerprint、image bytes/preprocess grid、
   image MRoPE position 和 projected embedding shape。
3. 统一 Qwen35 dense/MoE/VL/MTP 的 graph 与 scheduler。
