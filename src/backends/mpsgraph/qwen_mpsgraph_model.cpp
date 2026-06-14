#include "toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace toyllm::mpsgraph {

namespace {

Result<std::uint64_t> f32_bytes(std::uint64_t elements) {
  if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
    return Status::invalid_argument("MPSGraph f32 byte count overflow");
  }
  return elements * sizeof(float);
}

}  // namespace

Result<QwenMpsGraphModel> QwenMpsGraphModel::load_metadata(
  const std::filesystem::path& model_dir) {
  auto bundle = load_model_bundle(model_dir);
  if (!bundle.is_ok()) {
    return bundle.status();
  }
  auto weights = MpsGraphWeightStore::load_metadata(model_dir / "model.safetensors");
  if (!weights.is_ok()) {
    return weights.status();
  }
  const auto shape_status = weights.value().validate_qwen3_shapes(bundle.value().model);
  if (!shape_status.is_ok()) {
    return shape_status;
  }

  QwenMpsGraphModel model;
  model.bundle_ = std::move(bundle.value());
  model.weights_ = std::move(weights.value());
  model.refresh_info();
  return Result<QwenMpsGraphModel>(std::move(model));
}

Result<QwenMpsGraphModel> QwenMpsGraphModel::load_core_weights(
  const std::filesystem::path& model_dir, const MpsGraphContext& context) {
  auto model = load_metadata(model_dir);
  if (!model.is_ok()) {
    return model.status();
  }
  const auto upload_status = model.value().upload_core_weights(context);
  if (!upload_status.is_ok()) {
    return upload_status;
  }
  return Result<QwenMpsGraphModel>(std::move(model.value()));
}

const ModelConfig& QwenMpsGraphModel::config() const {
  return bundle_.model;
}

const QwenMpsGraphModelInfo& QwenMpsGraphModel::info() const {
  return info_;
}

bool QwenMpsGraphModel::core_weights_uploaded() const {
  return info_.core_weights_uploaded;
}

Result<std::vector<float>> QwenMpsGraphModel::debug_embed_token(
  const MpsGraphContext& context, std::int64_t token) const {
  if (!embedding_.buffer.valid()) {
    return Status::invalid_argument("MPSGraph embedding weight is not uploaded");
  }
  if (token < 0 || token >= bundle_.model.vocab_size) {
    return Status::invalid_argument("MPSGraph debug embedding token is out of range");
  }
  const auto hidden = static_cast<std::size_t>(bundle_.model.hidden_size);
  auto output = context.make_buffer(hidden * sizeof(float));
  if (!output.is_ok()) {
    return output.status();
  }
  auto output_buffer = std::move(output.value());
  const auto status = context.embedding_f32(
    embedding_.buffer, static_cast<std::size_t>(bundle_.model.vocab_size), hidden, token,
    output_buffer);
  if (!status.is_ok()) {
    return status;
  }
  std::vector<float> values(hidden);
  const auto read_status =
    context.copy_from_buffer(output_buffer, values.data(), values.size() * sizeof(float));
  if (!read_status.is_ok()) {
    return read_status;
  }
  return values;
}

Status QwenMpsGraphModel::upload_core_weights(const MpsGraphContext& context) {
  auto embedding = weights_.upload_tensor_f32(context, "model.embed_tokens.weight");
  if (!embedding.is_ok()) {
    return embedding.status();
  }
  auto final_norm = weights_.upload_tensor_f32(context, "model.norm.weight");
  if (!final_norm.is_ok()) {
    return final_norm.status();
  }
  embedding_ = std::move(embedding.value());
  final_norm_ = std::move(final_norm.value());
  refresh_info();
  return Status::ok();
}

void QwenMpsGraphModel::refresh_info() {
  info_.model_dir = bundle_.model_dir;
  info_.tensor_count = weights_.tensors().size();
  info_.safetensors_file_size = weights_.file_size();
  info_.safetensors_header_size = weights_.header_size();
  info_.hidden_size = bundle_.model.hidden_size;
  info_.num_hidden_layers = bundle_.model.num_hidden_layers;
  info_.vocab_size = bundle_.model.vocab_size;
  info_.core_weights_uploaded = embedding_.buffer.valid() && final_norm_.buffer.valid();
  info_.device_tensor_count = 0;
  info_.device_weight_bytes = 0;

  if (embedding_.buffer.valid()) {
    ++info_.device_tensor_count;
    const auto bytes = f32_bytes(embedding_.elements);
    if (bytes.is_ok()) {
      info_.device_weight_bytes += bytes.value();
    }
  }
  if (final_norm_.buffer.valid()) {
    ++info_.device_tensor_count;
    const auto bytes = f32_bytes(final_norm_.elements);
    if (bytes.is_ok()) {
      info_.device_weight_bytes += bytes.value();
    }
  }
}

}  // namespace toyllm::mpsgraph
