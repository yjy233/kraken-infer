#pragma once

#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
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
  std::size_t device_tensor_count{0};
  std::uint64_t device_weight_bytes{0};
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

  [[nodiscard]] const ModelConfig& config() const;
  [[nodiscard]] const QwenMpsGraphModelInfo& info() const;
  [[nodiscard]] bool core_weights_uploaded() const;

  [[nodiscard]] Result<std::vector<float>> debug_embed_token(
    const MpsGraphContext& context, std::int64_t token) const;

 private:
  [[nodiscard]] Status upload_core_weights(const MpsGraphContext& context);
  void refresh_info();

  ModelBundle bundle_;
  MpsGraphWeightStore weights_;
  QwenMpsGraphModelInfo info_;
  MpsGraphDeviceTensor embedding_;
  MpsGraphDeviceTensor final_norm_;
};

}  // namespace toyllm::mpsgraph
