#pragma once

#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_kv_cache.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_weight_store.hpp"
#include "toyllm/model/model_config.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace toyllm::mpsgraph {

struct QwenMpsGraphModelInfo {
  std::filesystem::path model_dir;
  std::size_t tensor_count{0};
  std::uint64_t safetensors_file_size{0};
  std::uint64_t safetensors_header_size{0};
  std::int64_t hidden_size{0};
  std::int64_t num_hidden_layers{0};
  std::int64_t vocab_size{0};
  bool core_weights_uploaded{false};
  bool lm_head_uploaded{false};
  bool layer_weights_uploaded{false};
  std::size_t uploaded_layer_count{0};
  std::size_t device_tensor_count{0};
  std::uint64_t device_weight_bytes{0};
};

struct QwenMpsGraphLayerWeights {
  MpsGraphDeviceTensor input_layernorm;
  MpsGraphDeviceTensor post_attention_layernorm;
  MpsGraphDeviceTensor q_proj;
  MpsGraphDeviceTensor k_proj;
  MpsGraphDeviceTensor v_proj;
  MpsGraphDeviceTensor o_proj;
  MpsGraphDeviceTensor q_norm;
  MpsGraphDeviceTensor k_norm;
  MpsGraphDeviceTensor gate_proj;
  MpsGraphDeviceTensor up_proj;
  MpsGraphDeviceTensor down_proj;
};

struct QwenMpsGraphRunState {
  MpsGraphBuffer hidden;
  MpsGraphBuffer normed;
  MpsGraphBuffer q;
  MpsGraphBuffer q_scratch;
  MpsGraphBuffer k;
  MpsGraphBuffer k_scratch;
  MpsGraphBuffer v;
  MpsGraphBuffer attn_out;
  MpsGraphBuffer projected;
  MpsGraphBuffer gate;
  MpsGraphBuffer up;
  MpsGraphBuffer mlp;
  MpsGraphBuffer down;
  MpsGraphBuffer logits;
  MpsGraphBuffer next_token;
  MpsGraphBuffer generated_tokens;
  MpsGraphKvCache kv_cache;
  std::size_t capacity_tokens{0};
  std::size_t generated_capacity{0};
};

class QwenMpsGraphModel {
 public:
  QwenMpsGraphModel() = default;
  ~QwenMpsGraphModel() = default;

  QwenMpsGraphModel(const QwenMpsGraphModel&) = delete;
  QwenMpsGraphModel& operator=(const QwenMpsGraphModel&) = delete;

  QwenMpsGraphModel(QwenMpsGraphModel&& other) noexcept = default;
  QwenMpsGraphModel& operator=(QwenMpsGraphModel&& other) noexcept = default;

  [[nodiscard]] static Result<QwenMpsGraphModel> load_metadata(
    const std::filesystem::path& model_dir);
  [[nodiscard]] static Result<QwenMpsGraphModel> load_core_weights(
    const std::filesystem::path& model_dir, const MpsGraphContext& context);
  [[nodiscard]] static Result<QwenMpsGraphModel> load_all_weights(
    const std::filesystem::path& model_dir, const MpsGraphContext& context);

  [[nodiscard]] const ModelConfig& config() const;
  [[nodiscard]] const GenerationConfig& generation() const;
  [[nodiscard]] const QwenMpsGraphModelInfo& info() const;
  [[nodiscard]] bool core_weights_uploaded() const;
  [[nodiscard]] bool all_weights_uploaded() const;
  [[nodiscard]] Result<QwenMpsGraphRunState> create_run_state(
    const MpsGraphContext& context, std::size_t capacity_tokens) const;

  [[nodiscard]] Result<std::vector<float>> debug_embed_token(
    const MpsGraphContext& context, std::int64_t token) const;
  [[nodiscard]] Status forward_token(const MpsGraphContext& context, std::int64_t token,
                                     std::size_t position,
                                     QwenMpsGraphRunState& state) const;
  [[nodiscard]] Status forward_next_token(const MpsGraphContext& context,
                                          std::size_t position,
                                          QwenMpsGraphRunState& state) const;
  [[nodiscard]] Status prefill_token_ids(const MpsGraphContext& context,
                                         const std::vector<std::int64_t>& tokens,
                                         QwenMpsGraphRunState& state) const;
  [[nodiscard]] Status greedy_next_token(const MpsGraphContext& context,
                                         QwenMpsGraphRunState& state) const;
  [[nodiscard]] Status record_next_token(const MpsGraphContext& context,
                                         std::size_t step,
                                         QwenMpsGraphRunState& state) const;
  [[nodiscard]] Result<std::vector<float>> debug_forward_token(
    const MpsGraphContext& context, std::int64_t token, std::size_t position,
    QwenMpsGraphRunState& state) const;
  [[nodiscard]] Result<std::int32_t> debug_greedy_next_token(
    const MpsGraphContext& context, QwenMpsGraphRunState& state) const;

 private:
  [[nodiscard]] Status upload_core_weights(const MpsGraphContext& context);
  [[nodiscard]] Status upload_lm_head(const MpsGraphContext& context);
  [[nodiscard]] Status upload_layer_weights(const MpsGraphContext& context);
  [[nodiscard]] Status apply_layer(const MpsGraphContext& context,
                                   const QwenMpsGraphLayerWeights& layer,
                                   std::size_t layer_index, std::size_t position,
                                   QwenMpsGraphRunState& state) const;
  [[nodiscard]] Status compute_logits(const MpsGraphContext& context,
                                      QwenMpsGraphRunState& state) const;
  [[nodiscard]] bool forward_weights_uploaded() const;
  void refresh_info();

  ModelBundle bundle_;
  MpsGraphWeightStore weights_;
  QwenMpsGraphModelInfo info_;
  MpsGraphDeviceTensor embedding_;
  MpsGraphDeviceTensor lm_head_;
  MpsGraphDeviceTensor final_norm_;
  std::vector<QwenMpsGraphLayerWeights> layers_;
};

}  // namespace toyllm::mpsgraph
