# Qwen3.5 0.8B 图片输入 llama.cpp 调研

本文调研 `~/code/llama.cpp` 中 Qwen3.5 0.8B 图片输入的实现路径，作为
kraken-infer 后续实现 OpenAI 兼容 image input 的技术依据。

## 结论

目标模型不是 Qwen3VL text model，而是 HF 架构名
`Qwen3_5ForConditionalGeneration` 的 Qwen3.5 conditional model。llama.cpp 的
拆分方式是：

- text decoder 转成 GGUF arch `qwen35`，运行时走 `src/models/qwen35.cpp`。
- image encoder/projector 转成单独 mmproj GGUF，运行时走 mtmd/clip 管线。
- mmproj 侧在 llama.cpp 中复用了 `qwen3vl` / `qwen3vl_merger` /
  `PROJECTOR_TYPE_QWEN3VL` / `tools/mtmd/models/qwen3vl.cpp` 这些命名。

因此，文档和实现里应把 `qwen3vl` 视作 llama.cpp 对视觉 projector 路径的实现标签，
不能把本任务改成实现 Qwen3VL text decoder。

## 转换路径

llama.cpp 的 converter 明确把 `Qwen3_5ForConditionalGeneration` 分成两份：

- `conversion/__init__.py:202-203`：text model map 把
  `Qwen3_5ForConditionalGeneration` 映射到 `qwen` converter。
- `conversion/qwen.py:620-622`：`Qwen3_5TextModel` 注册
  `Qwen3_5ForConditionalGeneration`，并写出 `gguf.MODEL_ARCH.QWEN35`。
- `conversion/qwen.py:521-526`：Qwen3.5 默认 MRoPE section 是
  `[11, 11, 10, 0]`。
- `conversion/__init__.py:293`：mmproj model map 把
  `Qwen3_5ForConditionalGeneration` 映射到 `qwen3vl` converter。
- `conversion/qwen3vl.py:16-17`：`Qwen3VLVisionModel` 同时注册
  `Qwen3VLForConditionalGeneration` 和 `Qwen3_5ForConditionalGeneration`。

mmproj converter 只保留视觉侧张量：

- `conversion/qwen3vl.py:65-77` 跳过 `lm_head.*`、`mtp.*`，只接受
  `visual.*` / `model.visual.*`。
- `conversion/qwen3vl.py:42-59` 写入 projector type `QWEN3VL`、
  `clip.vision.spatial_merge_size`、vision attention layernorm eps 和 deepstack layer flags。
- `conversion/qwen3vl.py:107-125` 把 `visual.merger.linear_fc1/linear_fc2`
  映射成 GGUF 的 `mm.0.*` / `mm.2.*`。
- `conversion/qwen3vl.py:128-140` 把 `visual.patch_embed.proj.weight` 的 Conv3D
  temporal kernel 拆成两个 2D patch embedding 权重。

这说明 Qwen3.5 0.8B image support 的权重形态是：

```text
text GGUF:
  general.architecture = "qwen35"

mmproj GGUF:
  general.architecture = "clip"
  clip.projector_type / clip.vision.projector_type = "qwen3vl_merger"
```

## Text Decoder

`src/models/qwen35.cpp` 只实现 text decoder，不加载任何视觉张量：

- `src/models/qwen35.cpp:29-34` 通过 `n_layer == 24 && n_embd == 1024`
  判断 0.8B dense 类型。
- `src/models/qwen35.cpp:37-127` 加载 token embedding、output norm、full
  attention 层、linear recurrent attention 层和 MTP block；没有 `visual.*`
  或 mmproj 张量。
- `src/models/qwen35.cpp:148-155` decoder graph 入口使用通用
  `build_inp_embd(model.tok_embd)` 和 `build_inp_pos()`。
- `src/models/qwen35.cpp:301-312` full-attention 层通过 `ggml_rope_multi`
  使用 Qwen3.5 MRoPE sections。

图片不是被转换成普通 token id 后进入 tokenizer。图片先经过 mmproj 变成 projected
embedding，然后作为 raw embedding batch 喂给同一个 qwen35 decoder。

## Runtime Pipeline

llama.cpp 的运行时路径在 `tools/mtmd`：

1. 文本模型由 llama runtime 加载。
2. mmproj 由 mtmd/clip 加载。
3. mtmd 校验 mmproj 输出维度和 text model input embedding 维度一致。
4. prompt 被拆成 text chunk 与 image chunk。
5. text chunk 通过 token id decode。
6. image chunk 先由 `clip_image_batch_encode` 编码成 embedding，再通过
   `llama_batch.embd` decode。

关键证据：

- `tools/mtmd/mtmd.cpp:372-380` 校验 `clip_n_mmproj_embd(ctx_v)` 必须等于
  text model 的 `llama_model_n_embd_inp()`。
- `tools/mtmd/mtmd.cpp:456-465` 对 `PROJECTOR_TYPE_QWEN3VL` 使用
  `<|vision_start|>` / `<|vision_end|>`，并选择 dynamic-size image preprocessor。
- `tools/mtmd/mtmd.cpp:1056-1058` 在 image chunk 前插入 begin token。
- `tools/mtmd/mtmd.cpp:1194-1236` 创建 `MTMD_INPUT_CHUNK_TYPE_IMAGE`，记录
  `nx`、`ny`、`batch_f32` 和 image id。
- `tools/mtmd/mtmd.cpp:1239-1241` 在 image chunk 后插入 end token。
- `tools/mtmd/mtmd-cli.cpp:259-265` 将格式化后的 chat prompt 和 bitmap 一起
  tokenized 成 chunks。
- `tools/mtmd/mtmd-cli.cpp:280-295` text chunk 走 token decode。
- `tools/mtmd/mtmd-cli.cpp:309-355` media chunk 先 batch encode，再调用
  `mtmd_helper_decode_image_chunk()`。

实际 decoder 序列形态是：

```text
text tokens ... <|vision_start|> tokens
image projected embeddings
<|vision_end|> tokens ... text tokens
```

## Vision Encoder / Projector

虽然文件名是 `qwen3vl.cpp`，但这个 graph 是 Qwen3.5 conditional image side 在
llama.cpp 中实际复用的实现。

clip dispatcher：

- `tools/mtmd/clip-impl.h:322-385` 定义 `PROJECTOR_TYPE_QWEN3VL`，字符串名是
  `qwen3vl_merger`。
- `tools/mtmd/clip.cpp:906-909` 对 `PROJECTOR_TYPE_QWEN3VL` 创建
  `clip_graph_qwen3vl`。
- `tools/mtmd/clip.cpp:1448-1458` Qwen projector 默认 `n_merge = 2`，使用
  bilinear resize，并设置 image token limit `8..4096`。
- `tools/mtmd/clip.cpp:2060-2066` 加载 projector MLP 张量 `mm.0.*` 和 `mm.2.*`。
- `tools/mtmd/clip.cpp:3307-3319` dynamic image 输出 token 数按
  `(width / (patch_size * 2)) * (height / (patch_size * 2))` 计算。
- `tools/mtmd/clip.cpp:4491-4519` Qwen3 projector 输出宽度是
  `mm.2.bias_width * (1 + n_deepstack_layers)`，即 main path 与 deepstack
  features 拼接后的宽度。
- `tools/mtmd/clip.cpp:4595-4600` Qwen 系 projector 的 temporal merge 返回 2。

`tools/mtmd/models/qwen3vl.cpp` graph 结构：

- `qwen3vl.cpp:16-31` patch embedding 后做 temporal merge 与 2x2 spatial merge。
- `qwen3vl.cpp:39-52` resize learned absolute position embedding 并加到 patch 上。
- `qwen3vl.cpp:56-58` vision graph 输入 `positions`，长度是 `n_patches * 4`。
- `qwen3vl.cpp:103-109` vision self-attention 使用 `GGML_ROPE_TYPE_VISION` 的 MRoPE。
- `qwen3vl.cpp:143-157` deepstack layer 输出经过 norm + FFN，并沿 feature 维 concat。
- `qwen3vl.cpp:163-180` post norm 后经过 `mm.0 -> GELU -> mm.2` projector，
  再拼接 deepstack features。

kraken 侧如果要“一模一样”，需要实现的是这条 mmproj/vision graph，而不是把图片
塞给现有 Qwen3.5 tokenizer。

## Decoder Embedding 和 MRoPE Position

image chunk 进入 decoder 时使用 `llama_batch.embd`，不是 `llama_batch.token`：

- `tools/mtmd/mtmd.cpp:1442-1464` 调用 `clip_image_batch_encode()` 生成
  `n_tokens * n_embd_out` 的 float embedding。
- `tools/mtmd/mtmd-helper.cpp:150-158` `decode_embd_batch` 构造的 batch 中
  `tokens = nullptr`，`embd = encoded_embd`。
- `tools/mtmd/mtmd-helper.cpp:261-268` image embedding 宽度取
  `llama_model_n_embd_inp(model)`。
- `tools/mtmd/mtmd-helper.cpp:296-305` image embedding batch 最终进入
  `llama_decode()`。
- `src/llama-graph.cpp:1833-1893` `build_inp_embd()` 同时创建 token path 和
  vector embedding path，并按 `ubatch.token ? 0 : 1` 选择。
- `src/llama-graph.cpp:1901-1904` raw embedding 被视作 multimodal input，不走
  token embedding scale。

MRoPE position 规则：

- `tools/mtmd/mtmd.cpp:1199-1208` 对 MRoPE decoder 保存 image token grid 的
  `nx`、`ny` 和 position type。
- `tools/mtmd/mtmd.cpp:1898-1907` image embedding 的 decoder position 为
  `t = pos_0`、`x = pos_0 + col`、`y = pos_0 + row`、`z = 0`。
- `tools/mtmd/mtmd.cpp:1957-1961` image chunk 对 decoder logical position 的推进是
  `max(nx, ny)`，不是 image embedding token 总数。
- `tools/mtmd/mtmd-helper.cpp:171-180` image embedding 的 MRoPE position buffer 顺序是
  `[t, y, x, z]`。
- `src/llama-batch.cpp:713-719` text token 在 MRoPE 下把同一个 1D position broadcast
  到各 section；embedding input 则按传入的多维 position 原样复制。
- `src/llama-graph.cpp:142-159` graph input 层也保留了 text token 的 4D broadcast
  逻辑。

这对 kraken-infer 的影响是：mixed prefill 不能只维护一个 `std::vector<int64_t>
tokens`。它必须维护 decoder item stream，每个 item 至少包含 kind、logical
position、MRoPE section positions，以及 image embedding 的 grid 元数据。

## OpenAI 兼容输入映射

OpenAI chat request 的 content array 可映射成 mtmd 风格 chunks：

```json
{
  "model": "qwen3.5-0.8b",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "这张图里有什么？" },
        { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }
      ]
    }
  ]
}
```

server 侧应先按 Qwen chat template 生成 prompt，再在图片占位处形成：

```text
TEXT_CHUNK("<|im_start|>user ...")
TEXT_CHUNK("<|vision_start|>")
IMAGE_CHUNK(image_id, preprocessed_f32, nx, ny, n_temporal_merge)
TEXT_CHUNK("<|vision_end|>")
TEXT_CHUNK("... <|im_end|>\n<|im_start|>assistant\n")
```

第一版建议支持：

- `image_url.url = data:image/...;base64,...`
- 本地测试可选支持 `file://` 或 repo 内路径，但 OpenAI 兼容接口不要依赖它。
- HTTP(S) 图片下载可以放第二阶段，避免先引入网络 fetch、超时、MIME sniffing 和
  安全边界。

如果请求包含图片但未指定/未加载 mmproj，应返回 OpenAI 兼容错误，而不是悄悄忽略图片。

## 对 Prefix / Block KV Cache 的影响

Qwen3.5 图片输入会改变跨请求 cache 的 key：

- 纯文本仍可用 token block key。
- 多模态 prefix 必须把 image chunk 作为 decoder item 纳入 key。
- key 需要包含 mmproj fingerprint、image bytes hash、preprocessor 参数、输出
  `nx/ny`、`n_temporal_merge`、logical position count，以及 projected embedding
  能否由这些信息唯一确定。
- image chunk 的 logical position 推进是 `max(nx, ny)`；cache restore 时 full
  attention KV 和 recurrent state 必须按 decoder logical positions 对齐。
- begin/end vision token 是真实 text token，也必须进入 cache key。

第一版 cache 可以规定：

- 只在 image chunk 边界切 block。
- 不复用半个 image chunk。
- 如果一个 block 内混有 text token 与 image embedding，保存完整 decoder item
  metadata，用于命中后校验。

## kraken-infer 实现建议

## 当前实现状态

当前已落地第一阶段入口层，行为如下：

- OpenAI gateway 支持 `messages[].content` 为 string 或 content array。
- content array 支持 `type = "text"` 和 `type = "image_url"`。
- `image_url` 支持 OpenAI 兼容的对象形式
  `{ "url": "data:image/png;base64,...", "detail": "auto" }`，也兼容字符串形式。
- `data:image/...;base64,...` 会被解析出 MIME type 和 image bytes；HTTP(S) URL
  暂时只保留原始 URL，后续 fetch 阶段再处理。
- 内部 `ChatMessage` 保留 `content_parts`、`image_mime_type` 和 `image_bytes`，
  不会再把图片 part 静默丢掉。
- image part 会记录稳定 `image_fingerprint`。data URL 使用
  MIME/detail/image bytes 计算；HTTP(S) URL 在 fetch 尚未接入前只记录 URL/detail
  source fingerprint，不能作为最终 image bytes cache key。
- data URL 图片会尽量从 PNG IHDR 或 JPEG SOF header 推断 `image_width` /
  `image_height`。这只是 header metadata，不是像素 decode/preprocess。
- `serve --mmproj PATH` 会读取 mmproj GGUF metadata，校验
  `general.architecture = "clip"`，并记录 projector type。
- 对 `qwen3vl_merger` mmproj，会按 llama.cpp 张量名校验
  `v.patch_embd.weight`、`v.patch_embd.weight.1`、`v.patch_embd.bias`、
  `v.position_embd.weight`、`mm.0.weight`、`mm.0.bias`、`mm.2.weight`、
  `mm.2.bias`，并校验已出现的 deepstack 层包含
  `norm/fc1/fc2` 的 weight/bias。
- 对 `qwen3vl_merger` mmproj，会按 llama.cpp `clip_n_mmproj_embd()` 的规则计算
  `projector_output_width = mm.2.bias[0] * (1 + deepstack_layers)`，启动日志会打印
  `qwen3vl_required_tensors_present`、`deepstack_layer_count` 和
  `projector_output_width`。
- gateway 启动时会校验 `projector_output_width == text GGUF embedding_length`，
  对齐 llama.cpp `mtmd.cpp` 的 mmproj/text embedding 维度检查。
- 已新增 Qwen3VL image embedding plan helper，对齐 llama.cpp dynamic-size preprocess：
  使用 `patch_size * spatial_merge_size` 对齐 resize，默认 token limit 为
  `8..4096`，并输出 resized size、patch grid、merge grid 和 image token 数。
  这一步尚不做像素 decode/vision forward，但为 native mixed prefill 明确 image
  placeholder 长度。
- 已新增 Qwen3VL multimodal prompt plan：按 chat message/content part 顺序生成
  text chunk、image embedding chunk、`<|vision_start|>` / `<|vision_end|>` 边界，
  并统计 image token 总数和 MRoPE image position advance。
- 已新增 tokenizer-aware multimodal token plan：text chunk 使用 Qwen3.5 GGUF
  tokenizer + `parse_special` 编码，image chunk 保留 raw embedding token count；
  同时区分 `total_tokens` 和 image MRoPE `total_position_advance`。
- 已新增 native image decode/preprocess 入口：macOS 下通过 ImageIO/CoreGraphics
  将 data URL 图片解码为 RGB；随后按 llama.cpp Qwen3VL dynamic-size 规则做
  bilinear resize，并使用 mmproj GGUF 中的 `clip.vision.image_mean/std` 输出 HWC
  float32 normalized pixels。
- `qwen3vl_merger` mmproj 加载时会强制读取 `clip.vision.image_mean/std`，并派生
  native `image_min_pixels/image_max_pixels`，启动摘要会打印这些预处理参数。
- 已新增 Qwen3VL native vision graph plan：读取并校验 patch embedding、absolute
  position embedding、post norm、12 层 `v.blk.*`、projector `mm.0/mm.2` 以及
  deepstack tensor shape。`kraken-infer weights <mmproj.gguf>` 会打印 graph plan。
- 本地 Qwen3.5 0.8B mmproj
  `models/qwen3.5-0.8b/mmproj-Qwen3.5-0.8B-BF16.gguf` 已通过 graph plan 校验：
  `image_size=768`、`patch_size=16`、`embedding_length=768`、`block_count=12`、
  `projection_dim=1024`、`required_tensors=154`、`deepstack_layer_count=0`。
- gateway 图片请求预检会运行 multimodal prompt plan，提前暴露缺失图片尺寸等
  native VL 前置错误。
- gateway 启动日志会打印 native image plan 的 patch/merge/token limit 摘要。
- 图片请求没有 `--mmproj` 时返回 OpenAI 兼容 400。
- 图片请求带了非 `qwen3vl_merger` mmproj 时返回 OpenAI 兼容 400。
- 图片请求带了 `qwen3vl_merger` mmproj 时，当前返回 OpenAI 兼容 501，明确说明
  native vision graph execution 尚未实现。

当前还没有实现：

- `tools/mtmd/models/qwen3vl.cpp` 对应的 vision tower / merger / deepstack graph 执行。
  当前已经能校验 graph 所需 tensor/hparam shape，但还没有跑 MPS/MPSGraph/CPU forward。
- mixed token/raw-embedding prefill。
- 多模态 block cache key 的真实 image chunk commit/restore。当前已有
  `image_fingerprint` 前置字段，但图片请求不会返回真实 KV cache hit。

启动命令：

```bash
./build/debug/kraken-infer serve \
  --host 127.0.0.1 \
  --port 18081 \
  --model models/qwen3.5-0.8b \
  --model-id qwen35-test \
  --device mps \
  --mmproj path/to/mmproj-qwen3.5-0.8b.gguf \
  --max-new-tokens 64
```

请求 URL：

```text
POST http://127.0.0.1:18081/v1/chat/completions
```

示例请求：

```bash
curl -sS http://127.0.0.1:18081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "qwen35-test",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "这张图里有什么？" },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,AAAA",
              "detail": "auto"
            }
          }
        ]
      }
    ],
    "max_tokens": 64
  }'
```

OpenAPI schema：

```text
GET http://127.0.0.1:18081/openapi.json
GET http://127.0.0.1:18081/v1/openapi.json
```

阶段 1：OpenAI 请求解析和错误路径

- 支持 `messages[].content` 的 string 与 array 两种形式。
- 解析 `type=text` 和 `type=image_url`。
- 没有 mmproj 时，对含图片请求返回 OpenAI 兼容错误。
- 先把内部 prompt 表示改成 `Qwen35DecoderItem` stream，为后续 mixed prefill 做铺垫。

阶段 2：mmproj GGUF loader

- 支持 `general.architecture = "clip"`。
- 识别 `qwen3vl_merger` projector type，但在代码注释中明确它是 Qwen3.5
  conditional image side 的 llama.cpp 兼容标签。
- 已校验 `mm.0.*`、`mm.2.*`、patch embedding、position embedding，以及已出现的
  deepstack 层完整性。
- 已计算 `clip_n_mmproj_embd(mmproj)` 等价的 `projector_output_width`，并接入
  text GGUF `embedding_length` 的启动期一致性校验。
- vision block 的逐层 shape/数值校验放到阶段 3 的 native vision graph loader 中做。

阶段 3：image preprocessing + vision graph

- dynamic-size image embedding plan 已实现。
- macOS ImageIO image decode 已实现；CPU bilinear resize + mean/std normalize 已实现。
- native vision graph tensor/hparam plan 已实现，并已用真实 Qwen3.5 0.8B mmproj 校验。
- 下一步是把 normalized HWC float32 pixels 接入原生 vision graph 输入张量并执行
  patch embed、vision blocks、post norm 和 projector。
- 按 `tools/mtmd/models/qwen3vl.cpp` 实现 patch embed、position embedding resize、
  vision MRoPE attention、deepstack 和 projector。
- 输出 contiguous F32 embeddings，形状为 `[n_image_tokens, n_embd_inp]`。

阶段 4：mixed prefill

- multimodal prompt chunk plan 已实现，包含 text/image chunk 顺序和 image
  placeholder 长度。
- tokenizer-aware multimodal token plan 已实现，可直接作为 mixed prefill scheduler
  的输入结构。
- text chunk 走 token embedding path。
- image chunk 走 raw embedding path。
- MRoPE position buffer 支持 text broadcast 和 image `[t,y,x,z]`。
- `n_past` / logical position 推进遵循 mtmd：text 按 token 数推进，image 按
  `max(nx, ny)` 推进。

阶段 5：cache 与 usage

- block cache key 纳入 decoder item metadata 和 mmproj/image fingerprints。
- 已落地 image part 的 `image_fingerprint` 字段；最终 cache key 仍需要加入
  mmproj fingerprint、preprocess grid、MRoPE position 和 projected embedding shape。
- OpenAI usage 中已有 `prompt_tokens_details.cached_tokens`，多模态请求应继续返回该字段。
- 可以额外在内部 stats 中区分 `cached_text_tokens` 与 `cached_image_positions`，但
  对外仍保持 OpenAI 兼容格式。

阶段 6：对齐验证

- 用同一 text GGUF、同一 mmproj GGUF、同一图片，对比 llama.cpp `mtmd-cli` 的 chunk
  数、image `nx/ny`、image token 数、logical position 推进和首 token logits top-k。
- 再验证 OpenAI gateway 的非流式/流式 usage 与 cache hit。

## 非目标

第一版不做：

- Qwen3VL text decoder。
- audio / Omni。
- video temporal merge 的完整产品化；但 mmproj metadata 中的 temporal merge 必须保留，
  以便 image graph shape 与 cache key 不失真。
- 任意 HTTP 图片抓取和复杂安全策略。
- 半 image chunk 的 block cache 复用。
