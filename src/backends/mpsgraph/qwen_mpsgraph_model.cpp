#include "toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp"

#include "toyllm/runtime/qwen_tokenizer.hpp"

#include <algorithm>
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

Status append_tensor_bytes(const MpsGraphDeviceTensor& tensor,
                           std::uint64_t& total_bytes,
                           std::size_t& total_tensors) {
  if (!tensor.buffer.valid()) {
    return Status::ok();
  }
  auto bytes = f32_bytes(tensor.elements);
  if (!bytes.is_ok()) {
    return bytes.status();
  }
  total_bytes += bytes.value();
  ++total_tensors;
  return Status::ok();
}

bool layer_weights_valid(const QwenMpsGraphLayerWeights& layer) {
  return layer.input_layernorm.buffer.valid() &&
         layer.post_attention_layernorm.buffer.valid() &&
         layer.q_proj.buffer.valid() &&
         layer.k_proj.buffer.valid() &&
         layer.v_proj.buffer.valid() &&
         layer.o_proj.buffer.valid() &&
         layer.q_norm.buffer.valid() &&
         layer.k_norm.buffer.valid() &&
         layer.gate_proj.buffer.valid() &&
         layer.up_proj.buffer.valid() &&
         layer.down_proj.buffer.valid();
}

bool checked_mul_size(std::size_t lhs, std::size_t rhs, std::size_t& output) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  output = lhs * rhs;
  return true;
}

Result<std::size_t> f32_bytes_size(std::size_t elements, std::string_view name) {
  std::size_t bytes = 0;
  if (!checked_mul_size(elements, sizeof(float), bytes)) {
    return Status::invalid_argument("MPSGraph " + std::string{name} +
                                    " byte count overflow");
  }
  return bytes;
}

Status make_state_buffer(const MpsGraphContext& context, std::size_t elements,
                         std::string_view name, MpsGraphBuffer& target) {
  auto bytes = f32_bytes_size(elements, name);
  if (!bytes.is_ok()) {
    return bytes.status();
  }
  auto buffer = context.make_buffer(bytes.value());
  if (!buffer.is_ok()) {
    return buffer.status();
  }
  target = std::move(buffer.value());
  return Status::ok();
}

ScopedProfileSpan profile_span(RequestProfiler* profiler, std::string_view name) {
  return profiler == nullptr ? ScopedProfileSpan{} : profiler->scoped(name);
}

ScopedProfileSpan profile_layer_span(RequestProfiler* profiler, std::size_t layer) {
  return profiler == nullptr
           ? ScopedProfileSpan{}
           : profiler->scoped("mpsgraph.layer", {ProfileField{"layer", std::to_string(layer)}});
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

Result<QwenMpsGraphModel> QwenMpsGraphModel::load_all_weights(
  const std::filesystem::path& model_dir, const MpsGraphContext& context) {
  auto model = load_metadata(model_dir);
  if (!model.is_ok()) {
    return model.status();
  }
  auto upload_status = model.value().upload_core_weights(context);
  if (!upload_status.is_ok()) {
    return upload_status;
  }
  upload_status = model.value().upload_lm_head(context);
  if (!upload_status.is_ok()) {
    return upload_status;
  }
  upload_status = model.value().upload_layer_weights(context);
  if (!upload_status.is_ok()) {
    return upload_status;
  }
  return Result<QwenMpsGraphModel>(std::move(model.value()));
}

const ModelConfig& QwenMpsGraphModel::config() const {
  return bundle_.model;
}

const GenerationConfig& QwenMpsGraphModel::generation() const {
  return bundle_.generation;
}

const QwenMpsGraphModelInfo& QwenMpsGraphModel::info() const {
  return info_;
}

bool QwenMpsGraphModel::core_weights_uploaded() const {
  return info_.core_weights_uploaded;
}

bool QwenMpsGraphModel::all_weights_uploaded() const {
  return info_.core_weights_uploaded && info_.lm_head_uploaded &&
         info_.layer_weights_uploaded &&
         info_.uploaded_layer_count ==
           static_cast<std::size_t>(bundle_.model.num_hidden_layers);
}

Result<QwenMpsGraphRunState> QwenMpsGraphModel::create_run_state(
  const MpsGraphContext& context, std::size_t capacity_tokens) const {
  if (!forward_weights_uploaded()) {
    return Status::invalid_argument("MPSGraph Qwen weights are not fully uploaded");
  }
  if (capacity_tokens == 0) {
    return Status::invalid_argument("MPSGraph run state capacity must be positive");
  }

  const auto hidden_size = static_cast<std::size_t>(bundle_.model.hidden_size);
  const auto head_dim = static_cast<std::size_t>(bundle_.model.head_dim);
  const auto heads = static_cast<std::size_t>(bundle_.model.num_attention_heads);
  const auto kv_heads = static_cast<std::size_t>(bundle_.model.num_key_value_heads);
  const auto layers = static_cast<std::size_t>(bundle_.model.num_hidden_layers);
  const auto intermediate = static_cast<std::size_t>(bundle_.model.intermediate_size);
  const auto vocab = static_cast<std::size_t>(bundle_.model.vocab_size);
  std::size_t attn_dim = 0;
  if (!checked_mul_size(heads, head_dim, attn_dim)) {
    return Status::invalid_argument("MPSGraph Qwen attention dimension overflow");
  }
  std::size_t kv_dim = 0;
  if (!checked_mul_size(kv_heads, head_dim, kv_dim)) {
    return Status::invalid_argument("MPSGraph Qwen KV dimension overflow");
  }

  QwenMpsGraphRunState state;
  auto status = make_state_buffer(context, hidden_size, "hidden", state.hidden);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, hidden_size, "normed", state.normed);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, attn_dim, "q", state.q);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, attn_dim, "q scratch", state.q_scratch);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, kv_dim, "k", state.k);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, kv_dim, "k scratch", state.k_scratch);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, kv_dim, "v", state.v);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, attn_dim, "attention output", state.attn_out);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, hidden_size, "projected", state.projected);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, intermediate, "gate", state.gate);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, intermediate, "up", state.up);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, intermediate, "mlp", state.mlp);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, hidden_size, "down", state.down);
  if (!status.is_ok()) {
    return status;
  }
  status = make_state_buffer(context, vocab, "logits", state.logits);
  if (!status.is_ok()) {
    return status;
  }
  auto next_token = context.make_buffer(sizeof(std::int32_t));
  if (!next_token.is_ok()) {
    return next_token.status();
  }
  state.next_token = std::move(next_token.value());
  auto generated_tokens =
    context.make_buffer(capacity_tokens * sizeof(std::int32_t));
  if (!generated_tokens.is_ok()) {
    return generated_tokens.status();
  }
  state.generated_tokens = std::move(generated_tokens.value());
  std::vector<std::int32_t> zero_generated(capacity_tokens, 0);
  status = context.copy_to_buffer(state.generated_tokens, zero_generated.data(),
                                  zero_generated.size() * sizeof(std::int32_t));
  if (!status.is_ok()) {
    return status;
  }
  auto generation_status = context.make_buffer(3U * sizeof(std::int32_t));
  if (!generation_status.is_ok()) {
    return generation_status.status();
  }
  state.generation_status = std::move(generation_status.value());
  status = context.reset_generation_status_i32(state.generation_status);
  if (!status.is_ok()) {
    return status;
  }
  status = state.kv_cache.reset(context, layers, capacity_tokens, kv_heads, head_dim);
  if (!status.is_ok()) {
    return status;
  }
  state.capacity_tokens = capacity_tokens;
  state.generated_capacity = capacity_tokens;
  return Result<QwenMpsGraphRunState>(std::move(state));
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

Status QwenMpsGraphModel::forward_token(const MpsGraphContext& context, std::int64_t token,
                                        std::size_t position,
                                        QwenMpsGraphRunState& state,
                                        RequestProfiler* profiler) const {
  if (!forward_weights_uploaded()) {
    return Status::invalid_argument("MPSGraph Qwen weights are not fully uploaded");
  }
  if (position >= state.capacity_tokens) {
    return Status::invalid_argument("MPSGraph Qwen position exceeds run state capacity");
  }
  if (!state.kv_cache.allocated()) {
    return Status::invalid_argument("MPSGraph Qwen KV cache is not allocated");
  }
  if (token < 0 || token >= bundle_.model.vocab_size) {
    return Status::invalid_argument("MPSGraph Qwen token is out of range");
  }

  const auto hidden_size = static_cast<std::size_t>(bundle_.model.hidden_size);
  Status status;
  {
    auto span = profile_span(profiler, "mpsgraph.forward.embedding");
    status = context.embedding_f32(
      embedding_.buffer, static_cast<std::size_t>(bundle_.model.vocab_size), hidden_size,
      token, state.hidden);
    if (!status.is_ok()) {
      return status;
    }
    (void)span;
  }
  for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
    status = apply_layer(context, layers_[layer_index], layer_index, position, state,
                         profiler);
    if (!status.is_ok()) {
      return status;
    }
  }
  {
    auto span = profile_span(profiler, "mpsgraph.forward.final_norm");
    status = context.rms_norm_f32(state.hidden, final_norm_.buffer, hidden_size,
                                  static_cast<float>(bundle_.model.rms_norm_eps),
                                  state.normed);
    if (!status.is_ok()) {
      return status;
    }
    (void)span;
  }
  return Status::ok();
}

Status QwenMpsGraphModel::forward_next_token(const MpsGraphContext& context,
                                             std::size_t position,
                                             QwenMpsGraphRunState& state,
                                             RequestProfiler* profiler) const {
  if (!forward_weights_uploaded()) {
    return Status::invalid_argument("MPSGraph Qwen weights are not fully uploaded");
  }
  if (position >= state.capacity_tokens) {
    return Status::invalid_argument("MPSGraph Qwen position exceeds run state capacity");
  }
  if (!state.kv_cache.allocated()) {
    return Status::invalid_argument("MPSGraph Qwen KV cache is not allocated");
  }

  const auto hidden_size = static_cast<std::size_t>(bundle_.model.hidden_size);
  Status status;
  {
    auto span = profile_span(profiler, "mpsgraph.forward.embedding_from_token");
    status = context.embedding_from_token_f32(
      embedding_.buffer, static_cast<std::size_t>(bundle_.model.vocab_size), hidden_size,
      state.next_token, state.hidden);
    if (!status.is_ok()) {
      return status;
    }
    (void)span;
  }
  for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
    status = apply_layer(context, layers_[layer_index], layer_index, position, state,
                         profiler);
    if (!status.is_ok()) {
      return status;
    }
  }
  auto span = profile_span(profiler, "mpsgraph.forward.final_norm");
  status = context.rms_norm_f32(state.hidden, final_norm_.buffer, hidden_size,
                                static_cast<float>(bundle_.model.rms_norm_eps),
                                state.normed);
  (void)span;
  return status;
}

Status QwenMpsGraphModel::greedy_next_token(const MpsGraphContext& context,
                                            QwenMpsGraphRunState& state,
                                            RequestProfiler* profiler) const {
  if (!all_weights_uploaded()) {
    return Status::invalid_argument("MPSGraph Qwen all weights are not uploaded");
  }
  auto status = compute_logits(context, state, profiler);
  if (!status.is_ok()) {
    return status;
  }
  auto span = profile_span(profiler, "mpsgraph.logits.argmax");
  status = context.argmax_i32(state.logits, static_cast<std::size_t>(bundle_.model.vocab_size),
                              state.next_token);
  (void)span;
  return status;
}

Status QwenMpsGraphModel::record_next_token(const MpsGraphContext& context,
                                            std::size_t step,
                                            QwenMpsGraphRunState& state) const {
  if (!state.generated_tokens.valid()) {
    return Status::invalid_argument("MPSGraph generated token buffer is not allocated");
  }
  return context.write_i32_token(state.next_token, state.generated_tokens, step,
                                 state.generated_capacity);
}

Status QwenMpsGraphModel::update_generation_status(
  const MpsGraphContext& context, std::size_t step, bool final_step,
  QwenMpsGraphRunState& state) const {
  if (!state.generation_status.valid()) {
    return Status::invalid_argument("MPSGraph generation status buffer is not allocated");
  }
  std::vector<std::int64_t> eos_tokens = bundle_.generation.eos_token_ids;
  eos_tokens.push_back(bundle_.model.eos_token_id);
  eos_tokens.push_back(kQwenEndOfText);
  std::sort(eos_tokens.begin(), eos_tokens.end());
  eos_tokens.erase(std::unique(eos_tokens.begin(), eos_tokens.end()), eos_tokens.end());
  return context.update_generation_status_i32(
    state.next_token, eos_tokens.data(), eos_tokens.size(), step, final_step,
    state.generation_status);
}

Status QwenMpsGraphModel::prefill_token_ids(
  const MpsGraphContext& context, const std::vector<std::int64_t>& tokens,
  QwenMpsGraphRunState& state, RequestProfiler* profiler) const {
  if (tokens.empty()) {
    return Status::invalid_argument("MPSGraph Qwen prefill tokens must not be empty");
  }
  if (tokens.size() > state.capacity_tokens) {
    return Status::invalid_argument("MPSGraph Qwen prefill exceeds run state capacity");
  }
  for (std::size_t position = 0; position < tokens.size(); ++position) {
    auto span =
      profiler == nullptr
        ? ScopedProfileSpan{}
        : profiler->scoped("mpsgraph.prefill.forward_token",
                           {ProfileField{"position", std::to_string(position)}});
    auto status = forward_token(context, tokens[position], position, state, profiler);
    if (!status.is_ok()) {
      return status;
    }
    (void)span;
  }
  return Status::ok();
}

Result<std::vector<float>> QwenMpsGraphModel::debug_forward_token(
  const MpsGraphContext& context, std::int64_t token, std::size_t position,
  QwenMpsGraphRunState& state) const {
  const auto status = forward_token(context, token, position, state);
  if (!status.is_ok()) {
    return status;
  }

  const auto hidden_size = static_cast<std::size_t>(bundle_.model.hidden_size);
  std::vector<float> values(hidden_size);
  const auto read_status =
    context.copy_from_buffer(state.normed, values.data(), values.size() * sizeof(float));
  if (!read_status.is_ok()) {
    return read_status;
  }
  return values;
}

Result<std::int32_t> QwenMpsGraphModel::debug_greedy_next_token(
  const MpsGraphContext& context, QwenMpsGraphRunState& state) const {
  const auto status = greedy_next_token(context, state);
  if (!status.is_ok()) {
    return status;
  }

  std::int32_t token = 0;
  const auto read_status = context.copy_from_buffer(state.next_token, &token, sizeof(token));
  if (!read_status.is_ok()) {
    return read_status;
  }
  return token;
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

Status QwenMpsGraphModel::upload_lm_head(const MpsGraphContext& context) {
  auto lm_head = weights_.upload_tensor_f32(context, "lm_head.weight");
  if (!lm_head.is_ok()) {
    return lm_head.status();
  }
  lm_head_ = std::move(lm_head.value());
  refresh_info();
  return Status::ok();
}

Status QwenMpsGraphModel::upload_layer_weights(const MpsGraphContext& context) {
  const auto layer_count = static_cast<std::size_t>(bundle_.model.num_hidden_layers);
  layers_.clear();
  layers_.resize(layer_count);
  for (std::size_t i = 0; i < layer_count; ++i) {
    const auto prefix = "model.layers." + std::to_string(i) + ".";
    auto& layer = layers_[i];

    auto upload_named = [&](MpsGraphDeviceTensor& target,
                            const std::string& name) -> Status {
      auto tensor = weights_.upload_tensor_f32(context, name);
      if (!tensor.is_ok()) {
        return tensor.status();
      }
      target = std::move(tensor.value());
      return Status::ok();
    };

    auto status = upload_named(layer.input_layernorm, prefix + "input_layernorm.weight");
    if (!status.is_ok()) {
      return status;
    }
    status =
      upload_named(layer.post_attention_layernorm, prefix + "post_attention_layernorm.weight");
    if (!status.is_ok()) {
      return status;
    }
    status = upload_named(layer.q_proj, prefix + "self_attn.q_proj.weight");
    if (!status.is_ok()) {
      return status;
    }
    status = upload_named(layer.k_proj, prefix + "self_attn.k_proj.weight");
    if (!status.is_ok()) {
      return status;
    }
    status = upload_named(layer.v_proj, prefix + "self_attn.v_proj.weight");
    if (!status.is_ok()) {
      return status;
    }
    status = upload_named(layer.o_proj, prefix + "self_attn.o_proj.weight");
    if (!status.is_ok()) {
      return status;
    }
    status = upload_named(layer.q_norm, prefix + "self_attn.q_norm.weight");
    if (!status.is_ok()) {
      return status;
    }
    status = upload_named(layer.k_norm, prefix + "self_attn.k_norm.weight");
    if (!status.is_ok()) {
      return status;
    }
    status = upload_named(layer.gate_proj, prefix + "mlp.gate_proj.weight");
    if (!status.is_ok()) {
      return status;
    }
    status = upload_named(layer.up_proj, prefix + "mlp.up_proj.weight");
    if (!status.is_ok()) {
      return status;
    }
    status = upload_named(layer.down_proj, prefix + "mlp.down_proj.weight");
    if (!status.is_ok()) {
      return status;
    }
  }

  refresh_info();
  return Status::ok();
}

Status QwenMpsGraphModel::apply_layer(const MpsGraphContext& context,
                                      const QwenMpsGraphLayerWeights& layer,
                                      std::size_t layer_index, std::size_t position,
                                      QwenMpsGraphRunState& state,
                                      RequestProfiler* profiler) const {
  auto layer_span = profile_layer_span(profiler, layer_index);
  const auto hidden_size = static_cast<std::size_t>(bundle_.model.hidden_size);
  const auto head_dim = static_cast<std::size_t>(bundle_.model.head_dim);
  const auto heads = static_cast<std::size_t>(bundle_.model.num_attention_heads);
  const auto kv_heads = static_cast<std::size_t>(bundle_.model.num_key_value_heads);
  const auto intermediate = static_cast<std::size_t>(bundle_.model.intermediate_size);
  const auto eps = static_cast<float>(bundle_.model.rms_norm_eps);
  const auto theta = static_cast<float>(bundle_.model.rope_theta);
  std::size_t attn_dim = 0;
  if (!checked_mul_size(heads, head_dim, attn_dim)) {
    return Status::invalid_argument("MPSGraph Qwen attention dimension overflow");
  }
  std::size_t kv_dim = 0;
  if (!checked_mul_size(kv_heads, head_dim, kv_dim)) {
    return Status::invalid_argument("MPSGraph Qwen KV dimension overflow");
  }

  auto run = [&](std::string_view name, auto&& fn) -> Status {
    auto span = profile_span(profiler, name);
    auto status = fn();
    (void)span;
    return status;
  };

  auto status = run("mpsgraph.layer.input_rms_norm", [&] {
    return context.rms_norm_f32(state.hidden, layer.input_layernorm.buffer, hidden_size, eps,
                                state.normed);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.q_proj", [&] {
    return context.matvec_f32(layer.q_proj.buffer, attn_dim, hidden_size, state.normed,
                              state.q);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.k_proj", [&] {
    return context.matvec_f32(layer.k_proj.buffer, kv_dim, hidden_size, state.normed,
                              state.k);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.v_proj", [&] {
    return context.matvec_f32(layer.v_proj.buffer, kv_dim, hidden_size, state.normed,
                              state.v);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.q_norm", [&] {
    return context.qk_norm_f32(state.q, layer.q_norm.buffer, heads, head_dim, eps,
                               state.q_scratch);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.q_rope", [&] {
    return context.rope_f32(state.q_scratch, heads, head_dim, position, theta, state.q);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.k_norm", [&] {
    return context.qk_norm_f32(state.k, layer.k_norm.buffer, kv_heads, head_dim, eps,
                               state.k_scratch);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.k_rope", [&] {
    return context.rope_f32(state.k_scratch, kv_heads, head_dim, position, theta, state.k);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.kv_store", [&] {
    return state.kv_cache.store(context, layer_index, position, state.k, state.v);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.attention", [&] {
    return context.attention_f32(
      state.q, state.kv_cache.key_buffer(), state.kv_cache.value_buffer(), layer_index,
      position, state.capacity_tokens, heads, kv_heads, head_dim, state.attn_out);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.o_proj", [&] {
    return context.matvec_f32(layer.o_proj.buffer, hidden_size, attn_dim,
                              state.attn_out, state.projected);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.attn_residual", [&] {
    return context.add_f32(state.hidden, state.projected, hidden_size, state.normed);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.post_attention_rms_norm", [&] {
    return context.rms_norm_f32(state.normed, layer.post_attention_layernorm.buffer,
                                hidden_size, eps, state.hidden);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.gate_proj", [&] {
    return context.matvec_f32(layer.gate_proj.buffer, intermediate, hidden_size,
                              state.hidden, state.gate);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.up_proj", [&] {
    return context.matvec_f32(layer.up_proj.buffer, intermediate, hidden_size,
                              state.hidden, state.up);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.silu_mul", [&] {
    return context.silu_mul_f32(state.gate, state.up, intermediate, state.mlp);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.down_proj", [&] {
    return context.matvec_f32(layer.down_proj.buffer, hidden_size, intermediate,
                              state.mlp, state.down);
  });
  if (!status.is_ok()) {
    return status;
  }
  status = run("mpsgraph.layer.mlp_residual", [&] {
    return context.add_f32(state.normed, state.down, hidden_size, state.hidden);
  });
  (void)layer_span;
  return status;
}

Status QwenMpsGraphModel::compute_logits(const MpsGraphContext& context,
                                         QwenMpsGraphRunState& state,
                                         RequestProfiler* profiler) const {
  if (!lm_head_.buffer.valid()) {
    return Status::invalid_argument("MPSGraph Qwen lm_head weight is not uploaded");
  }
  auto span = profile_span(profiler, "mpsgraph.logits.lm_head");
  auto status = context.matvec_f32(
    lm_head_.buffer, static_cast<std::size_t>(bundle_.model.vocab_size),
    static_cast<std::size_t>(bundle_.model.hidden_size), state.normed, state.logits);
  (void)span;
  return status;
}

bool QwenMpsGraphModel::forward_weights_uploaded() const {
  return info_.core_weights_uploaded && info_.layer_weights_uploaded &&
         info_.uploaded_layer_count ==
           static_cast<std::size_t>(bundle_.model.num_hidden_layers);
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
  info_.lm_head_uploaded = lm_head_.buffer.valid();
  info_.layer_weights_uploaded = false;
  info_.uploaded_layer_count = 0;
  for (const auto& layer : layers_) {
    if (!layer_weights_valid(layer)) {
      continue;
    }
    ++info_.uploaded_layer_count;
  }
  info_.layer_weights_uploaded =
    info_.uploaded_layer_count == static_cast<std::size_t>(bundle_.model.num_hidden_layers);
  info_.device_tensor_count = 0;
  info_.device_weight_bytes = 0;

  (void)append_tensor_bytes(embedding_, info_.device_weight_bytes, info_.device_tensor_count);
  (void)append_tensor_bytes(lm_head_, info_.device_weight_bytes, info_.device_tensor_count);
  (void)append_tensor_bytes(final_norm_, info_.device_weight_bytes, info_.device_tensor_count);
  for (const auto& layer : layers_) {
    (void)append_tensor_bytes(layer.input_layernorm, info_.device_weight_bytes,
                              info_.device_tensor_count);
    (void)append_tensor_bytes(layer.post_attention_layernorm, info_.device_weight_bytes,
                              info_.device_tensor_count);
    (void)append_tensor_bytes(layer.q_proj, info_.device_weight_bytes,
                              info_.device_tensor_count);
    (void)append_tensor_bytes(layer.k_proj, info_.device_weight_bytes,
                              info_.device_tensor_count);
    (void)append_tensor_bytes(layer.v_proj, info_.device_weight_bytes,
                              info_.device_tensor_count);
    (void)append_tensor_bytes(layer.o_proj, info_.device_weight_bytes,
                              info_.device_tensor_count);
    (void)append_tensor_bytes(layer.q_norm, info_.device_weight_bytes,
                              info_.device_tensor_count);
    (void)append_tensor_bytes(layer.k_norm, info_.device_weight_bytes,
                              info_.device_tensor_count);
    (void)append_tensor_bytes(layer.gate_proj, info_.device_weight_bytes,
                              info_.device_tensor_count);
    (void)append_tensor_bytes(layer.up_proj, info_.device_weight_bytes,
                              info_.device_tensor_count);
    (void)append_tensor_bytes(layer.down_proj, info_.device_weight_bytes,
                              info_.device_tensor_count);
  }
}

}  // namespace toyllm::mpsgraph
