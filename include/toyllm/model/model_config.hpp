#pragma once

#include "toyllm/core/status.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace toyllm {

struct ModelConfig {
  std::string architecture;
  std::string model_type;
  std::string hidden_act;
  std::string torch_dtype;
  std::string transformers_version;

  bool attention_bias{false};
  bool tie_word_embeddings{false};
  bool use_cache{false};
  bool use_sliding_window{false};

  double attention_dropout{0.0};
  double initializer_range{0.0};
  double rms_norm_eps{0.0};
  double rope_theta{0.0};

  std::int64_t bos_token_id{0};
  std::int64_t eos_token_id{0};
  std::int64_t head_dim{0};
  std::int64_t hidden_size{0};
  std::int64_t intermediate_size{0};
  std::int64_t max_position_embeddings{0};
  std::int64_t max_window_layers{0};
  std::int64_t num_attention_heads{0};
  std::int64_t num_hidden_layers{0};
  std::int64_t num_key_value_heads{0};
  std::int64_t vocab_size{0};
};

struct GenerationConfig {
  bool do_sample{false};
  double temperature{1.0};
  double top_p{1.0};
  std::int64_t bos_token_id{0};
  std::int64_t pad_token_id{0};
  std::int64_t top_k{0};
  std::string transformers_version;
  std::vector<std::int64_t> eos_token_ids;
};

struct TokenizerInfo {
  bool available{false};
  std::uint64_t base_vocab_size{0};
  std::uint64_t added_tokens{0};
  std::uint64_t tokenizer_config_added_tokens{0};
  std::uint64_t total_vocab_size{0};
  std::uint64_t max_token_id{0};
};

struct ModelBundle {
  std::filesystem::path model_dir;
  ModelConfig model;
  GenerationConfig generation;
  TokenizerInfo tokenizer;
};

[[nodiscard]] Result<ModelBundle> load_model_bundle(const std::filesystem::path& model_dir);
[[nodiscard]] Status validate_model_bundle(const ModelBundle& bundle);
[[nodiscard]] std::string format_model_summary(const ModelBundle& bundle);

}  // namespace toyllm
