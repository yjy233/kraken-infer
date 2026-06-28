#pragma once

#include "toyllm/core/status.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/gguf_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace toyllm {

enum class Qwen35LayerKind {
  full_attention,
  linear_attention,
  mtp,
};

struct Qwen35TensorBinding {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::uint32_t type{0};
  std::uint64_t offset{0};
  std::uint64_t absolute_offset{0};
  std::uint64_t byte_size{0};
};

struct Qwen35FullAttentionBindings {
  Qwen35TensorBinding q;
  Qwen35TensorBinding k;
  Qwen35TensorBinding v;
  Qwen35TensorBinding output;
  Qwen35TensorBinding q_norm;
  Qwen35TensorBinding k_norm;
};

struct Qwen35LinearAttentionBindings {
  Qwen35TensorBinding qkv;
  Qwen35TensorBinding gate;
  Qwen35TensorBinding conv1d;
  Qwen35TensorBinding dt_bias;
  Qwen35TensorBinding a;
  Qwen35TensorBinding beta;
  Qwen35TensorBinding alpha;
  Qwen35TensorBinding norm;
  Qwen35TensorBinding output;
};

struct Qwen35MtpBindings {
  Qwen35FullAttentionBindings attention;
  Qwen35TensorBinding eh_proj;
  Qwen35TensorBinding enorm;
  Qwen35TensorBinding hnorm;
  bool has_embed_tokens{false};
  bool has_shared_head_head{false};
  bool has_shared_head_norm{false};
  Qwen35TensorBinding embed_tokens;
  Qwen35TensorBinding shared_head_head;
  Qwen35TensorBinding shared_head_norm;
};

struct Qwen35LayerBindings {
  std::int64_t index{0};
  Qwen35LayerKind kind{Qwen35LayerKind::linear_attention};

  Qwen35TensorBinding attn_norm;
  Qwen35TensorBinding attn_post_norm;
  Qwen35TensorBinding ffn_gate;
  Qwen35TensorBinding ffn_down;
  Qwen35TensorBinding ffn_up;

  Qwen35FullAttentionBindings full_attention;
  Qwen35LinearAttentionBindings linear_attention;
  Qwen35MtpBindings mtp;
};

struct Qwen35WeightMap {
  Qwen35TensorBinding token_embedding;
  Qwen35TensorBinding output_norm;
  Qwen35TensorBinding output;
  bool output_tied_to_token_embedding{false};

  std::vector<Qwen35LayerBindings> layers;
  std::size_t full_attention_layers{0};
  std::size_t linear_attention_layers{0};
  std::size_t mtp_layers{0};
};

[[nodiscard]] const char* qwen35_layer_kind_name(Qwen35LayerKind kind);
[[nodiscard]] Result<Qwen35WeightMap> build_qwen35_weight_map(const ModelConfig& config,
                                                              const GgufFile& gguf);
[[nodiscard]] std::string format_qwen35_weight_map_summary(const Qwen35WeightMap& map);

}  // namespace toyllm
