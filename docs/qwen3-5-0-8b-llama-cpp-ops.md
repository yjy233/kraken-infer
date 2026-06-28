# Qwen3.5 0.8B llama.cpp operator survey

This note records what llama.cpp actually executes for the local
Qwen3.5 0.8B GGUF path. The reference tree is `/Users/bill/code/llama.cpp`.
The target for this repository is still an independent macOS Metal runtime:
no llama.cpp, no libllama, no libggml dependency in the final implementation.

## Summary

Qwen3.5 0.8B is a hybrid model in llama.cpp, not a plain dense attention
transformer. The main pass has 24 layers: 18 recurrent linear-attention
layers and 6 full-attention layers. The default schedule marks every fourth
layer as full attention, so layers 3, 7, 11, 15, 19, and 23 are full attention
when recurrent flags are not explicitly provided.

For performance parity with llama.cpp on Mac, the runtime has to migrate the
same graph structure and the same Metal-heavy operators:

- Quantized `ggml_mul_mat` for `Q4_K`, `Q5_K`, `Q6_K`, plus F32 paths.
- Full-attention KV cache and `ggml_flash_attn_ext`.
- MRoPE via `ggml_rope_multi`.
- RMSNorm and L2 norm, including Metal norm fusion where applicable.
- SSM conv via `ggml_ssm_conv`.
- Fused gated delta net via `ggml_gated_delta_net`.
- Graph/view/copy/cache plumbing: `view`, `reshape`, `permute`, `cont`,
  `get_rows`, `cpy`, `repeat`, `concat`, and recurrent-state writeback.
- Logits post-processing through llama.cpp's sampler chain for greedy,
  `top_k`, `top_p`, temperature, seeded distribution sampling, and later
  backend/offloaded samplers.

Do not base the migration on this repository's current toy operators. They do
not represent the llama.cpp Qwen3.5 execution path. Treat them only as local
experiments that should be replaced by llama.cpp-equivalent runtime/backend
code.

## Local model facts

From:

```text
./build/debug/kraken-infer inspect models/qwen3.5-0.8b
./build/debug/kraken-infer weights models/qwen3.5-0.8b
```

Observed local GGUF:

- File: `models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf`.
- Architecture: `qwen35`.
- Layers: 24 main layers, 0 MTP/NextN layers.
- Hidden: 1024.
- Attention heads: 8.
- KV heads: 2.
- Attention head dim: 256.
- FFN intermediate size: 3584.
- Context length: 262144.
- Full attention interval: 4.
- Linear conv kernel: 4.
- Linear state/head dim: 128.
- Linear key heads: 16.
- Linear value heads: 16.
- Linear inner size: 2048.
- MRoPE sections: `11, 11, 10, 0`.
- RMSNorm eps: `1e-06`.
- RoPE theta: `10000000`.
- Vocab: 248320.
- Output head: tied to `token_embd.weight`.

Tensor type distribution:

```text
F32: 133
Q4_K: 152
Q5_K: 18
Q6_K: 17
```

Important tensor examples:

- `blk.0.attn_gate.weight [1024, 2048] Q4_K`.
- `blk.0.attn_qkv.weight [1024, 6144] Q5_K`.
- `blk.0.ffn_down.weight [3584, 1024] Q6_K`.
- `blk.0.ssm_conv1d.weight [4, 6144] F32`.

## llama.cpp graph entry points

Qwen3.5 model-specific graph code is in:

- `src/models/qwen35.cpp`.
- `src/models/delta-net-base.cpp`.
- shared graph helpers in `src/llama-graph.cpp`.
- hybrid memory creation in `src/llama-model.cpp`.

Key source locations:

- Hyperparameters and recurrent layer mask:
  `src/models/qwen35.cpp:4-35`.
- Tensor loading:
  `src/models/qwen35.cpp:37-127`.
- Main decoder graph:
  `src/models/qwen35.cpp:136-228`.
- Full attention block:
  `src/models/qwen35.cpp:257-335`.
- Linear recurrent block:
  `src/models/qwen35.cpp:338-469`.
- Dense SwiGLU FFN:
  `src/models/qwen35.cpp:472-485`.
- Optional MTP graph:
  `src/models/qwen35.cpp:487-644`.

The main graph flow is:

```text
token embedding
for each layer:
  RMSNorm(attn_norm)
  if recurrent:
    linear attention / gated delta net block
  else:
    full attention block
  residual add
  RMSNorm(attn_post_norm)
  dense SwiGLU FFN
  residual add
final RMSNorm(output_norm)
LM head matmul, tied to token_embd when output.weight is absent
```

## Full attention layer

Code path: `src/models/qwen35.cpp:257-335`.

This is not the standard `Q, K, V` projection shape used by older Qwen models:

1. `wq` projects to query plus a gate:
   `Qcur_full = build_lora_mm(layer.wq, cur)`.
2. `Qcur` is a view into the first half of `Qcur_full`.
3. `Qcur` gets RMSNorm via `attn_q_norm`.
4. `wk` and `wv` are separate projections.
5. `Kcur` is reshaped and RMSNormed via `attn_k_norm`.
6. `gate` is a view into the second half of `Qcur_full`, then made contiguous.
7. Q and K use `ggml_rope_multi` with MRoPE sections.
8. Attention is built by `llm_graph_context::build_attn(...)`.
9. Output of attention is multiplied by `sigmoid(gate)`.
10. Final attention output projection is `wo`.

Shared attention helper:

- `build_attn_mha`: `src/llama-graph.cpp:2066-2199`.
- KV-cache `build_attn`: `src/llama-graph.cpp:2311-2384`.

When `cparams.flash_attn` is enabled and there is no KQ bias, llama.cpp uses
`ggml_flash_attn_ext`:

- Flash attention branch: `src/llama-graph.cpp:2089-2112`.
- Fallback branch: `ggml_mul_mat(k, q)`, `ggml_soft_max_ext`,
  `ggml_mul_mat(v, kq)` at `src/llama-graph.cpp:2131-2189`.

For our Mac-only target, performance parity means implementing the flash path,
not relying on the fallback attention graph.

## Linear recurrent layer

Code path: `src/models/qwen35.cpp:338-469`.

Layer flow:

1. `wqkv` projection:
   `build_qkvz()` calls `build_lora_mm(layer.wqkv, input)` and reshapes to
   `[qkv_dim, n_seq_tokens, n_seqs]`.
2. `wqkv_gate` projection produces `z`.
3. `ssm_beta` projection, reshape, then `sigmoid`.
4. `ssm_alpha` projection, add `ssm_dt.bias`, then `softplus`.
5. Recurrent gate:
   `gate = alpha_softplus * ssm_a`.
6. Read recurrent conv and SSM states from hybrid recurrent cache.
7. `build_conv_state()` concatenates previous conv state and current qkv,
   then writes the new conv state back to cache.
8. `ggml_ssm_conv(conv_input, ssm_conv1d)` computes depthwise temporal conv.
9. `silu` activation on conv output.
10. Q/K/V are views into the conv output.
11. Q and K use `ggml_l2_norm`.
12. If needed, Q/K are repeated to match value-head count when the fused GDN
    path is not available.
13. `build_recurrent_attn()` runs gated delta net and writes new state.
14. The output is gated RMSNorm:
    `RMSNorm(output, ssm_norm) * silu(z)`.
15. Final projection is `ssm_out`.

Delta-net helper source:

- Header contract: `src/models/models.h:24-91`.
- Fused GDN wrapper: `src/models/delta-net-base.cpp:373-423`.
- GDN dispatch choice: `src/models/delta-net-base.cpp:425-447`.
- Conv state read/write: `src/models/delta-net-base.cpp:449-525`.
- Recurrent state writeback: `src/models/delta-net-base.cpp:527-606`.
- Non-fused chunk fallback: `src/models/delta-net-base.cpp:16-287`.
- Non-fused autoregressive fallback: `src/models/delta-net-base.cpp:289-370`.

llama.cpp defaults to fused GDN:

- `cparams.fused_gdn_ar = true`,
  `cparams.fused_gdn_ch = true`,
  `cparams.auto_fgdn = true` in `src/llama-context.cpp:200-202`.
- Support is resolved during scheduler reserve by checking that
  `GGML_OP_GATED_DELTA_NET` stays on the intended device:
  `src/llama-context.cpp:519-599`.

For performance parity, implement fused `ggml_gated_delta_net`; the expanded
fallback graph has many small ops and triangular solve work and should not be
the main path.

Native fused GDN is measured with:

```text
./build/debug/kraken-infer bench-qwen35-gdn \
  --tokens 1024 --iterations 10 --warmup 2
```

This allocates synthetic Qwen3.5-shaped F32 Q/K/V/gate/beta/state buffers and
calls the same `gated_delta_net_f32_batched_in_place` path used by prefill. On
Apple M4, the current optimized dispatch `qwen35_simdgroup_4rows` measures
about 4.98ms for a 1024-token chunk and about 4.13ms for the 760-token final
chunk shape (`key_heads=value_heads=16`, `head_dim=128`). Normal generation
does not call this benchmark path.

Native full-attention prefill is measured with:

```text
./build/debug/kraken-infer bench-qwen35-attention \
  --tokens 1024 --iterations 10 --warmup 2
```

The benchmark allocates synthetic Qwen3.5 full-attention buffers with the real
shape (`heads=8`, `kv_heads=2`, `head_dim=256`) and calls the same
`attention_f32_batched_f16_kv` backend path used by default prefill. It keeps
the timed section around the attention call only; if the current KV length is
not 64-aligned, the backend's flash-tail padding copy is included because that
copy is inside the real attention path. On Apple M4, the current default
`flash256_f16_kv` dispatch measures about 5.59ms for a 1024-token chunk with
`key_count=1024`. The 11k final-chunk shape:

```text
./build/debug/kraken-infer bench-qwen35-attention \
  --tokens 760 --start-position 10240 --capacity-tokens 11000 \
  --iterations 10 --warmup 2
```

measures about 50.06ms for one full-attention layer and reports
`flash256_f16_kv+tail` after switching the tail pad buffers to reusable
`MpsContext` scratch storage. This is much slower than the 1024-token first chunk
because the kernel attends over `key_count=11000`, not only the 760 query
tokens. Normal generation does not call this benchmark path.

llama.cpp's Metal path for the corresponding graph node is
`GGML_OP_FLASH_ATTN_EXT`:

- Host dispatch: `ggml/src/ggml-metal/ggml-metal-ops.cpp:2517-3058`.
- Pipeline selection: `ggml/src/ggml-metal/ggml-metal-device.cpp:1295-1525`.
- Metal kernels: `ggml/src/ggml-metal/ggml-metal.metal:5717-7337`.
- Tile constants: `OP_FLASH_ATTN_EXT_NQPSG=8`,
  `OP_FLASH_ATTN_EXT_NCPSG=64` in
  `ggml/src/ggml-metal/ggml-metal-impl.h:106-107`.

For Qwen3.5's full-attention shape (`DK=DV=256`), llama.cpp's non-vector
half8x8 path chooses the same effective tile geometry as the native
`flash256` kernel (`Q=8`, `C=64`, `NSG=4`). The remaining parity work is
therefore not a tile-shape mismatch; it is the detailed pad/block/mask flow,
function-constant specialization, shared-memory layout, and tail buffer reuse.

### Native default parity status

The default native path should follow llama.cpp's Qwen3.5 graph decomposition.
Optimizations that are only semantically equivalent but not part of the default
llama.cpp decomposition must stay behind an explicit environment switch until
benchmarked and justified.

| llama.cpp node sequence | Native default | Status |
| --- | --- | --- |
| `wqkv`, `wqkv_gate`, `ssm_beta`, `ssm_alpha` via `build_lora_mm` | GGUF K-quant `mul_mm` for prompt chunks, `mul_mv_ext` for small token counts | Default path follows the same projection split; remaining work is Metal kernel parity/perf. |
| `beta -> ggml_sigmoid`; `alpha -> add(dt) -> softplus -> mul(ssm_a)` | `prepare_qwen35_gdn_gate_beta_f32` | Semantically the same chain fused into one Qwen3.5 kernel because the tensors are small and this improved 11k prefill. Debug dump path keeps the llama.cpp intermediate names. |
| `ggml_ssm_conv -> ggml_silu` | `ssm_conv_f32 -> silu_f32_in_place` | Same op sequence. |
| Q/K/V `ggml_view_4d` from conv output, then Q/K `ggml_l2_norm` | `split_qkv_l2_norm_f32_qwen35` | Equivalent to llama.cpp views plus L2 norm. Native runtime materializes contiguous Q/K/V because the current GDN kernel consumes contiguous buffers; this is a known implementation difference to revisit with strided/source-offset GDN. |
| `ggml_gated_delta_net` through fused GDN chunk path | `gated_delta_net_f32_batched_qwen35` for `head_dim=128` and `key_heads=value_heads` | Follows llama.cpp fused GDN intent; keep comparing kernel layout and state update order. |
| `build_norm_gated`: RMSNorm, `silu(z)`, `mul` | `qk_norm_f32_f32_batched -> silu_f32_in_place -> mul_f32_in_place` | Default follows llama.cpp op decomposition. Experimental `qwen35_norm_gated_f32_in_place` is available through `KRAKEN_QWEN35_FUSED_NORM_GATED=1` but is not default after a slower 11k run. |
| Full attention `build_attn` / flash attention backend | `attention_f32_batched_f16_kv` with Qwen3.5 `flash256` specialization and F16 KV cache | Semantics and cache layout are aligned for Qwen3.5 full-attention layers; remaining work is kernel-by-kernel parity with llama.cpp Metal flash-attn and tail handling. |
| `ssm_out`, residual, post-attn norm, SwiGLU FFN, down projection | Native quant matmul, residual add, RMSNorm, `silu_mul`, quant matmul | Same high-level sequence; remaining work is quant matmul backend parity and buffer reuse. |

Quant matmul parity is now measured with a native diagnostic command rather
than by inferring from full-request profile spans:

```text
./build/debug/kraken-infer bench-qwen35-matmul \
  --model models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf \
  --tensor blk.0.attn_qkv.weight --tokens 1024 --iterations 3 --warmup 1
```

`--list` prints the real 2D Q4_K/Q5_K/Q6_K tensors in the GGUF. The benchmark
loads the selected tensor into this repository's Metal backend and calls the
same `MpsContext::matmul_q*_k_f32_device` path used by prefill, with one graph
commit per timed iteration. It is only a diagnostic entry point; normal
generation does not call llama.cpp, libllama, or this benchmark path. Benchmark
output and normal trace spans include a `dispatch` label such as
`mul_mm_simd_64x32` or `mul_mv_ext_r1_4`, so a profile can show whether a
projection is using the llama.cpp-equivalent prompt path.

Current 1024-token microbench baselines on Apple M4, using 2 warmup and 10
timed iterations:

| Tensor | Type | Shape as rows x cols | Avg ms/iter |
| --- | --- | --- | --- |
| `blk.0.attn_qkv.weight` | Q5_K | 6144 x 1024 | 5.84 |
| `blk.0.ffn_gate.weight` | Q4_K | 3584 x 1024 | 3.58 |
| `blk.0.ffn_down.weight` | Q6_K | 1024 x 3584 | 3.84 |
| `blk.0.ssm_out.weight` | Q4_K | 1024 x 2048 | 2.53 |

The latest llama.cpp non-tensor `mul_mm` path uses 64x32 output tiles and can
request 6144 bytes of threadgroup memory for fully aligned output tiles, while
8192 bytes are needed for boundary stores. A local test of that 6144-byte
aligned-memory choice improved one short Q5/Q6 3-iteration sample but regressed
the 10-iteration Q4/Q5 baselines above, so the native default remains the
existing 8192-byte allocation until the full generic llama.cpp `kernel_mul_mm`
port is benchmarked end to end.

## FFN and output head

Qwen3.5 0.8B dense model does not use MoE FFN:

- `src/models/qwen35.cpp:472-485`.

The shared helper is:

- `llm_graph_context::build_ffn`: `src/llama-graph.cpp:1267-1452`.

For Qwen3.5 it is parallel SwiGLU:

```text
up   = mul_mat(ffn_up, cur)
gate = mul_mat(ffn_gate, cur)
act  = swiglu_split(gate, up)
out  = mul_mat(ffn_down, act)
```

Final output:

- `output_norm` is RMSNorm.
- LM head is `build_lora_mm(model.output, cur)`.
- In the local GGUF, `output.weight` is absent and llama.cpp ties output to
  `token_embd.weight`; see tensor loading in `src/models/qwen35.cpp:45-52`.

## MTP/NextN

The local 0.8B GGUF has `MTP/NextN layers: 0`, so it is not in the main
execution target.

llama.cpp still has Qwen3.5 MTP code:

- Tensor loading: `src/models/qwen35.cpp:96-118`.
- MTP graph: `src/models/qwen35.cpp:487-644`.

The MTP block is a dense full-attention decoder-like block:

```text
RMSNorm(h)
RMSNorm(token embedding)
concat
eh_proj matmul
full-attention Q+gate/K/V path with MRoPE
sigmoid gate
wo matmul
residual
RMSNorm
SwiGLU FFN
residual
shared head norm + LM head
```

It can be ignored for the first 0.8B runtime if we explicitly target the local
GGUF, but the graph design should not make it impossible to add later.

## Memory and cache model

Qwen3.5 uses llama.cpp hybrid memory:

- `llama_model::create_memory`: `src/llama-model.cpp:2003-2125`.
- For Qwen3.5, attention filter is `!hparams.is_recr(il)` and recurrent
  filter is `hparams.is_recr(il)` for main layers:
  `src/llama-model.cpp:2078-2084`.
- Non-SWA Qwen3.5 uses `llama_memory_hybrid` with F32 recurrent states:
  `src/llama-model.cpp:2107-2125`.
- Recurrent cache tensors are `cache_r_l{i}` and `cache_s_l{i}`:
  `src/llama-memory-recurrent.cpp:20-128`.
- Main graph creates combined hybrid input with attention KV input plus
  recurrent-state input:
  `src/llama-graph.cpp:2878-2886`.

Implication for kraken runtime:

- Need KV cache for the 6 full-attention layers.
- Need recurrent conv cache `R` and SSM state cache `S` for the 18 linear
  layers.
- Need cache row selection, copy/update, and optional rollback snapshot
  semantics if later supporting `n_rs_seq > 0`.
- These are runtime responsibilities, not just Metal shader responsibilities.

## GGML and Metal operator map

| Purpose | Graph source | GGML op/source | Metal implementation | Migration note |
| --- | --- | --- | --- | --- |
| Weight projections, FFN, LM head | `build_lora_mm`, `src/llama-graph.cpp:1085-1114`; FFN `1267-1452` | `ggml_mul_mat` | dispatch `ggml-metal-ops.cpp:2035-2263`; pipelines `ggml-metal-device.cpp:667-760`; shaders include Q4/Q5/Q6 K paths at `ggml-metal.metal:7946`, `8077`, `8185`, `10239-10265` | Highest priority. Current native path keeps Q4_K/Q5_K/Q6_K matvec for decode, uses migrated llama.cpp `mul_mv_ext_q4x4` for 4-8 token prompt chunks, and uses migrated llama.cpp legacy 64x32 simdgroup K-quant `mul_mm` for prompt chunks with more than 8 tokens. The old transposed F32 dense `MPSMatrixMultiplication` bridge is opt-in only via `KRAKEN_QWEN35_DENSE_PREFILL=1`. |
| RMSNorm | `build_norm`, `src/llama-graph.cpp:1154-1187` | `ggml_rms_norm`, `ggml.c:3114-3135` | dispatch `ggml-metal-ops.cpp:3349-3485`; pipeline `ggml-metal-device.cpp:1662-1703`; shaders `ggml-metal.metal:3056`, `3119-3125` | Include norm+mul fusion because llama.cpp fuses RMSNorm weight multiply. |
| L2 norm for recurrent Q/K | `src/models/qwen35.cpp:429-432` | `ggml_l2_norm`, `ggml.c:3198-3220` | dispatch `ggml-metal-ops.cpp:3230-3295`; pipeline `ggml-metal-device.cpp:1616-1639`; shaders `ggml-metal.metal:3128`, `3177-3178` | Required before GDN. |
| MRoPE | `src/models/qwen35.cpp:301-312`, `587-592` | `ggml_rope_multi`, `ggml.c:4197-4215` | dispatch `ggml-metal-ops.cpp:3487+`; pipeline `ggml-metal-device.cpp:1705-1735`; shaders `ggml-metal.metal:4511`, `4671-4672` | Must support 4-position MRoPE sections `11,11,10,0`. |
| Full attention | `build_attn_mha`, `src/llama-graph.cpp:2066-2199` | `ggml_flash_attn_ext`, `ggml.c:5363-5404` | dispatch `ggml-metal-ops.cpp:2641+`; pipelines `ggml-metal-device.cpp:1295-1507`; shaders `ggml-metal.metal:5722`, `5794`, `5895`, `6533`, `6592-6622` | Current native path has Qwen3.5-specific flash256 kernels for F32 query plus F32 or F16 K/V cache at `head_dim=256`; F16 KV is the default and `KRAKEN_QWEN35_F16_KV=0` falls back to F32 KV. It mirrors the llama.cpp non-vec flash shape at a smaller scope: 8 queries/threadgroup, 64 KV columns/block, simdgroup matrix QK and PV, and built-in causal masking for this runtime's cache layout. Non-64-aligned KV lengths use a GPU-side tail pad for the final partial K/V block, so arbitrary prompt lengths stay on flash for this model shape. `KRAKEN_QWEN35_FLASH_ATTENTION=0` disables it; `1` forces compatible shapes. Non-compatible shapes fall back to the stable 8-query online-softmax kernel or the shared K/V tiled bridge (`cache_tile=16`, optional `KRAKEN_QWEN35_TILED_ATTENTION_TILE=32`). Remaining parity gaps are llama.cpp's generic mask/sinks/bias/logit-softcap handling and broader dtype/head-size instantiations. |
| SSM conv | `src/models/qwen35.cpp:384-397` | `ggml_ssm_conv`, `ggml.c:5515-5539` | dispatch `ggml-metal-ops.cpp:1382-1452`; pipelines `ggml-metal-device.cpp:476-533`; shaders `ggml-metal.metal:2116-2227` | Input is F32 conv state + qkv, weight is F32 `ssm_conv1d`. |
| Fused gated delta net | `src/models/delta-net-base.cpp:373-423`, `527-606` | `ggml_gated_delta_net`, `ggml.c:6216-6268` | dispatch `ggml-metal-ops.cpp:1594-1665`; pipeline `ggml-metal-device.cpp:599-635`; shaders `ggml-metal.metal:2577`, `2720`, host names `2711-2713`, `2815-2817` | Critical for recurrent layers. Op is F32-only and returns output plus state snapshot rows. |
| Sigmoid, SiLU, softplus, exp, neg | Qwen35 gate/beta/alpha paths | `ggml_unary`, `ggml.c:2371-2380`, `2726-2738`, `2784+` | dispatch `ggml-metal-ops.cpp:747+`; shader `ggml-metal.metal:1017-1200` | Needed around attention gate, GDN beta/gate, conv activation, gated norm. |
| SwiGLU FFN | `build_ffn`, `src/llama-graph.cpp:1346-1375` | `ggml_swiglu_split`, `ggml.c:3021+` | GLU dispatch `ggml-metal-ops.cpp:840+`; shaders `ggml-metal.metal:1482-1507` | Used by every layer's dense FFN. |
| Binary add/mul/sub/div | residuals, gates, alpha/gate math | `ggml_add`, `ggml_mul`, etc. | dispatch `ggml-metal-ops.cpp:270-276`; pipeline `ggml-metal-device.cpp:1537-1581` | Include broadcasting cases used by norm weights and SSM vectors. |
| Copy/cont/view/get rows | KV cache, recurrent cache, output ids | `ggml_cpy`, `ggml_cont`, `ggml_get_rows`, views | dispatch `ggml-metal-ops.cpp:352-355`, `447-450`, `1143+`, `1854+`; view/reshape/permute are metadata ops in backend support | Required for cache writeback and for splitting Q/K/V/gate views without extra copies. |
| Greedy and sampling | `common/sampling.cpp:334-382`; public sampler API in `include/llama.h:1317-1340`; implementation in `src/llama-sampler.cpp:806+`, `1230+`, `1321+`, `1513+`, `1884+` | Not a Qwen3.5 GGML model op; consumes final logits | llama.cpp can run host sampler chains and has newer backend sampler hooks in `src/llama-context.cpp:1188-1238`, `3057-3137` | Current native runtime keeps greedy as device-side argmax with one token readback. When sampling is enabled, it reads the final logits row and applies basic `top_k`/`top_p`/temperature/seed sampling. Backend/offloaded sampler parity remains pending. |
| Fallback delta-net chunk ops | `src/models/delta-net-base.cpp:16-370` | `cumsum`, `tri`, `diag`, `solve_tri`, `pad`, repeated matmul | Metal support exists for many ops, e.g. dispatch cases `ggml-metal-ops.cpp:315-343`, `434-436` | Not first priority if fused GDN is implemented and enabled. Keep as validation/reference only. |

## Metal backend execution pattern

llama.cpp's Metal path is not one monolithic kernel. The general flow is:

```text
model graph builds GGML nodes
backend scheduler assigns nodes to Metal/CPU
ggml-metal-ops.cpp dispatches a GGML op
ggml-metal-device.cpp chooses/compiles a pipeline based on dtype/shape
ggml-metal.metal shader executes
```

Relevant Metal support list:

- `ggml-metal-common.cpp:261-286` lists common supported ops including
  `MUL_MAT`, `ROPE`, `RMS_NORM`, `L2_NORM`, `SSM_CONV`, `TRI`, `MUL`,
  `ADD`, `UNARY`, `GET_ROWS`, `CPY`, `CONT`, and `REPEAT`.
- `ggml-metal-ops.cpp:265-450` is the main op dispatcher.

For kraken, an equivalent backend should not call GGML, but it should preserve
the same separation:

1. Tensor metadata and graph nodes with shape/stride/view semantics.
2. Runtime scheduler and memory planner for Metal buffers.
3. Metal command encoding for each op.
4. Pipeline selection by dtype and shape.
5. Shader implementations copied/ported into this repository.

## 11k prompt prefill requirements

Long-prompt prefill is a first-class target, not a follow-up optimization. The
runtime must explicitly handle an 11k-token input prompt with throughput and
peak memory close to llama.cpp on the same Mac and GGUF.

Required design points:

1. Use a batched prefill graph, not a per-token decode loop. Projection and FFN
   work must hit quantized matrix-matrix style paths for prompt chunks, while
   decode can use matrix-vector style kernels. The current native runtime uses
   migrated llama.cpp `mul_mv_ext_q4x4` for Q4_K/Q5_K/Q6_K prompt chunks with
   4-8 tokens and migrated llama.cpp legacy 64x32 simdgroup K-quant `mul_mm`
   for chunks with more than 8 tokens, matching llama.cpp's `ne11_mm_min`
   dispatch threshold. The previous transposed F32 dense MPSMatrix bridge is
   retained only as an explicit comparison path with
   `KRAKEN_QWEN35_DENSE_PREFILL=1`; simple row-token tiling was tested and
   rejected because it reduced parallelism and regressed the 520-token prompt
   from the 14s range to the 24s range.
2. Split the 11k prompt into Metal-friendly chunks, similar in purpose to
   llama.cpp's `n_ubatch`, so memory stays bounded without starving GPU
   occupancy. The native default is 1024 tokens per prefill chunk, and the CLI
   exposes this as `--prefill-chunk-tokens N`. The runtime now also commits
   the Metal command buffer at each prefill chunk boundary by default, which
   mirrors llama.cpp's graph/ubatch boundary more closely than encoding the
   entire 11k prompt into one command buffer. `KRAKEN_QWEN35_PREFILL_COMMIT=single`
   restores the old one-shot command buffer for comparison; `layer` is a
   diagnostic-only mode and is not viable for normal execution.
   With auto flash256 enabled, repeated `hello` rough data is 2048 tokens at
   about 5.40s for chunk 2048. Disabling flash via
   `KRAKEN_QWEN35_FLASH_ATTENTION=0` falls back to the tiled path at about
   6.07s; older fully disabled tiled attention regressed the 2048-token point
   to about 6.75s. With default F16 KV cache and auto flash256, the 11k
   repeated-`hello` acceptance prompt completes and emits ` hello`. A chunk
   sweep with default chunk command-buffer submission measured 1024 at about
   25.76s, 512 at about 27.52s, 1536 at about 35.74s, and 2048 at about
   38.10s, so 1024 is the current default. After the Qwen3.5-specific GDN
   simdgroup specialization, fused conv-output Q/K/V split plus Q/K L2 norm
   kernel, and fused beta/gate prepare kernel, the same default 11k prompt
   measured about 16.97-17.38s wall-clock in earlier runs. The latest default
   profile summary measured 15.58s total, with prefill around 14.41s and prefill
   commit around 14.33s; the `/usr/bin/time` wall-clock was about 15.75s. An
   experimental fused norm-gated kernel matching
   llama.cpp's `build_norm_gated` graph exists behind
   `KRAKEN_QWEN35_FUSED_NORM_GATED=1`, but a single 11k run measured about
   19.69s, so it is not default. The old
   single-command-buffer path measured about 57-66s on the same prompt. A
   current llama.cpp reference run on the same machine reported
   `llama-bench pp11000 = 814.58 +/- 8.05 tok/s`, or about 13.50s pure prefill.
   On the pure prefill metric, native is about 1.07x slower, takes about 0.90s
   longer, and has about 6.3% lower throughput. The remaining gap is therefore
   inside chunk execution, especially K-quant matmul, full-attention flash/tail
   details, and remaining recurrent-layer overhead, not host tokenization or KV
   cache size. Use `bench-qwen35-matmul`, `bench-qwen35-gdn`, and
   `bench-qwen35-attention` to isolate the current native kernels before
   changing runtime scheduling or adding more fusions. The
   pre-flash 4096-token tiled datapoint was about 12.81s for chunk 2048 and
   should be remeasured after flash256 coverage expands.
3. Reuse compiled Metal pipelines, graph shape plans, command encoding helpers,
   and preallocated buffers across chunks. Chunk command-buffer boundaries must
   not trigger repeated allocation or shader/pipeline compilation.
4. Full-attention layers must use the flash-attention path for prefill. The
   fallback `QK -> softmax -> V` graph is not acceptable as the main path for
   11k prompts. The native runtime now uses a Qwen3.5-specific F32 flash256
   kernel when `head_dim=256`; F16 KV cache is the default to match the usual
   llama.cpp Metal KV-cache dtype and F32 KV remains available through
   `KRAKEN_QWEN35_F16_KV=0`. Non-64-aligned KV lengths are handled by GPU-side
   K/V tail padding for the final partial block, so the final chunk of an
   arbitrary-length prompt does not have to fall back just because its KV length
   is not divisible by 64. The runtime still keeps online-softmax and tiled
   attention as fallback paths. The tiled bridge has template
   specializations for 16-token and optional 32-token KV tiles; the latter is
   controlled by `KRAKEN_QWEN35_TILED_ATTENTION_TILE=32` and is not the final
   flash kernel. The llama.cpp reference path still to migrate
   completely for this model is the F32 query plus
   F32/F16 K/V `flash_attn_ext` family with
   `DK=DV=256`; llama.cpp's graph may cast F32 K/V to F16 before flash, so
   precision and logits parity must be checked before changing cache dtype.
5. Recurrent linear layers must use chunked fused gated delta net
   (`fused_gdn_ch` equivalent). Expanding GDN into many small fallback ops, or
   running it token-by-token, will not match llama.cpp. The native runtime now
   keeps recurrent state on device and uses a Qwen3.5 `head_dim=128`,
   `key_heads=value_heads` specialization that processes 4 state rows per
   threadgroup and uses `simd_sum`; other shapes fall back to the generic
   batched GDN kernel.
6. Cache writes should be contiguous/coalesced per chunk: KV cache for the 6
   full-attention layers and recurrent conv/SSM state writes for the 18 linear
   layers.
7. Avoid materializing logits for every prompt token. Match llama.cpp's
   `inp_out_ids` style behavior and compute/output only the final rows required
   by the caller unless full prompt logits are explicitly requested.
8. Host-side mask, position, MRoPE section, cache-index, and sequence metadata
   generation must be amortized across chunks and should not dominate prefill
   wall time.
9. Keep prompt and decode performance knobs separate. The runtime needs
   independent settings for prefill chunk size, decode batch size, flash
   attention enablement, and fused GDN chunk/autoregressive paths.

Acceptance benchmark:

- Prompt: at least one 11k-token input against
  `Qwen3.5-0.8B-Q4_K_M.gguf`.
- Compare against llama.cpp on the same Mac, same model, same context length,
  same effective prefill chunk size where possible, and same output-logits
  policy.
- Record prefill latency, prompt tokens/s, peak Metal memory, and first decode
  token latency after prefill.
- Treat regressions in either full-attention prefill or recurrent-layer prefill
  as backend issues, not model-level noise.

## Migration implications

Minimum independent Qwen3.5 0.8B runtime scope:

1. Implement GGUF tensor loading and Qwen3.5 tensor binding for the observed
   tensor names and dtypes.
2. Implement graph/tensor metadata compatible with the needed GGML semantics:
   `ne`, `nb`, views, reshape, permute, transpose, contiguity, and row copies.
3. Implement Metal quantized matmul for F32 activations against Q4_K/Q5_K/Q6_K
   weights, plus F32 weights for small projection/SSM tensors.
4. Implement full-attention KV cache plus MRoPE plus flash attention.
5. Implement recurrent cache, SSM conv, L2 norm, fused GDN, and gated norm.
6. Implement RMSNorm, binary ops, unary activations, SwiGLU, get_rows/cpy/cont.
7. Implement chunked prefill scheduling and output-row selection for long
   prompts, including the 11k-token acceptance case above.
8. Compare logits and per-token throughput with llama.cpp on the same GGUF and
   prompt-processing/decode batch sizes.

Do not make CPU correctness the primary path. Host-side code still needs to
prepare inputs, masks, and cache indices, but compute should target Metal.

## Validation targets

Suggested validation order:

1. Decode one token with all 24 layers and compare final logits against
   llama.cpp with the same prompt. Current native Metal path matches llama.cpp
   for `hello` and `hello world` after using the same per-head interleaved
   Q/G split in full-attention layers.
2. Isolate full-attention layers and compare Q/K after MRoPE, then attention
   output before gate and after gate.
3. Isolate linear layers and compare SSM conv output, L2-normalized Q/K,
   GDN output, and updated SSM state.
4. Compare prompt processing throughput and single-token decode throughput
   against llama.cpp using the same `Qwen3.5-0.8B-Q4_K_M.gguf`.

Performance acceptance should be measured separately for:

- Prompt processing, including the 11k-token prefill case where quantized
  matrix-matrix kernels, flash attention, chunked fused GDN, cache writeback,
  and output-row selection dominate.
- Decode, where quantized matvec, KV cache, SSM conv, and fused GDN dominate.
