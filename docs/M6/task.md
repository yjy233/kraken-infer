# M6 Task: Sampling And Streaming CLI

目标：让 CPU 推理 CLI 支持 greedy、temperature、top-k、top-p、seed 可复现采样，以及 token 级流式输出。

## Scope

本阶段只覆盖 CPU backend。MPS backend 尚未接入完整 forward。

## Tasks

### Sampling API

- [x] 定义 `CpuSamplingConfig`
- [x] 支持 greedy decode 作为默认行为
- [x] 支持 `do_sample`
- [x] 支持 `temperature`
- [x] 支持 `top_k`
- [x] 支持 `top_p`
- [x] 支持 `seed`
- [x] seed 固定时采样路径可复现
- [x] 支持 KV cache verification 使用同一 seed 做 full-prefix recompute

### Sampling Implementation

- [x] lm head 输出完整 logits
- [x] greedy 使用 argmax
- [x] sampling 使用 temperature 缩放
- [x] sampling 使用 top-k 裁剪
- [x] sampling 使用 top-p nucleus 裁剪
- [x] sampling 使用 `std::mt19937_64`
- [x] 采样参数非法时返回明确错误

### Streaming

- [x] `CpuGenerationRequest` 支持 token callback
- [x] CLI `infer` 支持 `--stream`
- [x] CLI `run` 支持 `--stream`
- [x] CLI `chat` 支持 `--stream`
- [x] stream 模式每生成一个 token 立即 decode 并 flush
- [x] stream 模式仍返回完整 assistant text，供 chat history 使用

### CLI

- [x] `--sample`
- [x] `--temperature T`
- [x] `--top-k K`
- [x] `--top-p P`
- [x] `--seed N`
- [x] `--stream`
- [x] 与 `--kv-cache-stats` 可组合
- [x] 与 `--verify-kv-cache` 可组合
- [x] 与 `--dump-dir` 可组合

### Verification

- [x] `make test` 通过
- [x] `infer --prompt hello --max-new-tokens 2` 可运行
- [x] `infer --prompt hello --max-new-tokens 2 --sample --temperature 0.6 --top-k 20 --top-p 0.95 --seed 42` 可运行
- [x] `infer --prompt hello --max-new-tokens 2 --stream` 可运行

## Acceptance

- [x] greedy decode 保持默认行为
- [x] temperature/top-k/top-p sampling 可用
- [x] 固定 seed 可复现采样路径
- [x] stream 模式不等待整段生成结束才打印
- [x] chat stream 仍保留多轮历史

## Known Limitations

- [ ] 采样实现当前在 CPU 上排序完整 vocab，速度不是最终形态
- [ ] 还没有 GPU/MPS top-k/top-p/reduction
- [ ] stream 当前按 token decode，byte-level tokenizer 在极端 Unicode 分片时可能出现终端暂态半字符
