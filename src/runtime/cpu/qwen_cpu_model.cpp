#include "qwen_cpu_model.hpp"

#include "debug_dump.hpp"
#include "kv_cache.hpp"
#include "safetensors.hpp"
#include "tokenizer.hpp"
#include "tokens.hpp"
#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace toyllm::cpu {

namespace {

struct LayerWeights {
  TensorView input_norm;
  TensorView q_proj;
  TensorView q_norm;
  TensorView k_proj;
  TensorView k_norm;
  TensorView v_proj;
  TensorView o_proj;
  TensorView post_norm;
  TensorView gate_proj;
  TensorView up_proj;
  TensorView down_proj;
};

struct MpsLayerWeights {
  mps::MpsBuffer input_norm;
  mps::MpsBuffer q_proj;
  mps::MpsBuffer q_norm;
  mps::MpsBuffer k_proj;
  mps::MpsBuffer k_norm;
  mps::MpsBuffer v_proj;
  mps::MpsBuffer o_proj;
  mps::MpsBuffer post_norm;
  mps::MpsBuffer gate_proj;
  mps::MpsBuffer up_proj;
  mps::MpsBuffer down_proj;
};

struct MpsWorkspace {
  mps::MpsBuffer hidden;
  mps::MpsBuffer normed;
  mps::MpsBuffer q;
  mps::MpsBuffer k;
  mps::MpsBuffer v;
  mps::MpsBuffer attn_out;
  mps::MpsBuffer projected;
  mps::MpsBuffer gate;
  mps::MpsBuffer up;
  mps::MpsBuffer down;
  mps::MpsBuffer logits;
  mps::MpsBuffer key_cache;
  mps::MpsBuffer value_cache;
  std::size_t capacity_tokens{0};
  std::size_t used_tokens{0};
};

struct EffectiveSamplingConfig {
  bool do_sample{false};
  double temperature{1.0};
  std::size_t top_k{0};
  double top_p{1.0};
  std::uint64_t seed{0};
};

EffectiveSamplingConfig make_effective_sampling(const GenerationConfig& generation,
                                                const CpuSamplingConfig& request) {
  EffectiveSamplingConfig result;
  result.do_sample = request.do_sample;
  result.temperature = request.temperature_set ? request.temperature : generation.temperature;
  result.top_k = request.top_k_set ? request.top_k : static_cast<std::size_t>(generation.top_k);
  result.top_p = request.top_p_set ? request.top_p : generation.top_p;
  result.seed =
    request.seed_set ? request.seed : static_cast<std::uint64_t>(std::random_device{}());

  if (result.temperature <= 0.0) {
    throw std::runtime_error("sampling temperature must be positive");
  }
  if (result.top_p <= 0.0 || result.top_p > 1.0) {
    throw std::runtime_error("sampling top_p must be in (0, 1]");
  }
  return result;
}

class DumperScope {
 public:
  DumperScope(DebugDumper*& target, DebugDumper* dumper)
      : target_(target), previous_(target) {
    target_ = dumper;
  }
  ~DumperScope() { target_ = previous_; }

  DumperScope(const DumperScope&) = delete;
  DumperScope& operator=(const DumperScope&) = delete;

 private:
  DebugDumper*& target_;
  DebugDumper* previous_{nullptr};
};

class ProfilerScope {
 public:
  ProfilerScope(RequestProfiler*& target, RequestProfiler* profiler)
      : target_(target), previous_(target) {
    target_ = profiler;
  }
  ~ProfilerScope() { target_ = previous_; }

  ProfilerScope(const ProfilerScope&) = delete;
  ProfilerScope& operator=(const ProfilerScope&) = delete;

 private:
  RequestProfiler*& target_;
  RequestProfiler* previous_{nullptr};
};

std::string position_tensor_name(std::size_t position, std::string_view suffix) {
  std::ostringstream output;
  output << "position." << position << "." << suffix;
  return output.str();
}

std::string layer_tensor_name(std::size_t position, std::size_t layer, std::string_view suffix) {
  std::ostringstream output;
  output << "position." << position << ".layer." << layer << "." << suffix;
  return output.str();
}

class CpuQwenModel {
 public:
  static CpuQwenModel load(const std::filesystem::path& model_dir) {
    CpuQwenModel model;
    auto bundle = load_model_bundle(model_dir);
    if (!bundle.is_ok()) {
      throw std::runtime_error(bundle.status().message());
    }
    model.config_ = bundle.value().model;
    model.generation_ = bundle.value().generation;
    model.tokenizer_ = QwenTokenizer::load(model_dir);
    model.weights_ = SafeTensorMap::load(model_dir / "model.safetensors");
    model.bind_weights();
    return model;
  }

  CpuModelOutput generate(const std::vector<ChatMessage>& messages, std::size_t max_new_tokens,
                          bool enable_thinking, const std::filesystem::path& debug_dump_dir,
                          bool verify_kv_cache, Device compute_device,
                          const CpuSamplingConfig& sampling,
                          const std::function<void(std::string_view)>& stream_token,
                          RequestProfiler* profiler) {
    DebugDumper dumper(debug_dump_dir);
    DumperScope dumper_scope(dumper_, dumper.enabled() ? &dumper : nullptr);
    ProfilerScope profiler_scope(profiler_, profiler);
    if (compute_device.kind == DeviceKind::mps) {
      return generate_mps(messages, max_new_tokens, enable_thinking, verify_kv_cache,
                          compute_device, sampling, stream_token);
    }

    const auto effective_sampling = make_effective_sampling(generation_, sampling);
    std::mt19937_64 rng(effective_sampling.seed);

    std::vector<std::int64_t> prompt_tokens;
    {
      auto span = profile_span("request.tokenize");
      prompt_tokens = tokenizer_.encode_chat_messages(messages, enable_thinking);
      if (prompt_tokens.empty()) {
        throw std::runtime_error("tokenizer produced no prompt tokens");
      }
    }
    if (profiler_ != nullptr) {
      profiler_->set_metadata("prompt_tokens", prompt_tokens.size());
    }
    dump_i64("prompt_tokens", {static_cast<std::uint64_t>(prompt_tokens.size())}, prompt_tokens);

    const auto total_tokens = prompt_tokens.size() + max_new_tokens + 1U;
    {
      auto span = profile_span("request.reset_kv_cache");
      reset_kv_cache(total_tokens);
    }

    std::vector<float> hidden;
    std::size_t position = 0;
    {
      auto prefill = profile_span("request.prefill");
      for (const auto token : prompt_tokens) {
        hidden = forward_token(token, position);
        ++position;
      }
    }

    std::vector<std::int64_t> generated;
    generated.reserve(max_new_tokens);
    auto decode_span = profile_span("request.decode");
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
      const auto logits = [&]() {
        auto span = profile_span("decode_step.logits");
        return compute_logits(hidden, false);
      }();
      dump_f32(position_tensor_name(position - 1U, "logits"),
               {static_cast<std::uint64_t>(config_.vocab_size)}, logits);
      const auto next_token = [&]() {
        auto span = profile_span("decode_step.sample");
        return select_next_token(logits, rng, effective_sampling);
      }();
      if (is_eos(next_token)) {
        break;
      }
      generated.push_back(next_token);
      if (stream_token) {
        auto span = profile_span("decode_step.stream_emit");
        const auto token_text = tokenizer_.decode(std::vector<std::int64_t>{next_token});
        stream_token(token_text);
      }
      {
        auto span = profile_span("decode_step.forward_next_token");
        hidden = forward_token(next_token, position);
        ++position;
      }
    }
    (void)decode_span;
    dump_i64("generated_tokens", {static_cast<std::uint64_t>(generated.size())}, generated);
    const auto cached_stats = kv_cache_.stats();
    bool verified = false;
    if (verify_kv_cache) {
      const auto recomputed =
        generate_recomputed_tokens(prompt_tokens, max_new_tokens, effective_sampling,
                                   false);
      if (recomputed != generated) {
        throw std::runtime_error("KV cache verification failed: cached decode differs from "
                                 "full-prefix recompute");
      }
      verified = true;
    }
    if (profiler_ != nullptr) {
      profiler_->set_metadata("generated_tokens", generated.size());
      profiler_->set_metadata("device", compute_device.to_string());
      profiler_->set_status("ok");
    }
    return CpuModelOutput{tokenizer_.decode(generated), cached_stats, verified,
                          prompt_tokens.size(), generated.size()};
  }

 private:
  void reset_kv_cache(std::size_t total_tokens) {
    const auto layers = static_cast<std::size_t>(config_.num_hidden_layers);
    const auto kv_heads = static_cast<std::size_t>(config_.num_key_value_heads);
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    kv_cache_.reset(layers, total_tokens, kv_heads, head_dim);
  }

  std::vector<std::int64_t> generate_recomputed_tokens(
    const std::vector<std::int64_t>& prompt_tokens, std::size_t max_new_tokens,
    const EffectiveSamplingConfig& sampling, bool use_mps_lm_head) {
    DumperScope disable_dump(dumper_, nullptr);
    std::mt19937_64 rng(sampling.seed);
    std::vector<std::int64_t> tokens = prompt_tokens;
    std::vector<std::int64_t> generated;
    generated.reserve(max_new_tokens);

    for (std::size_t step = 0; step < max_new_tokens; ++step) {
      reset_kv_cache(tokens.size() + 1U);
      std::vector<float> hidden;
      for (std::size_t position = 0; position < tokens.size(); ++position) {
        hidden = forward_token(tokens[position], position);
      }
      const auto logits = compute_logits(hidden, use_mps_lm_head);
      const auto next_token = select_next_token(logits, rng, sampling);
      if (is_eos(next_token)) {
        break;
      }
      generated.push_back(next_token);
      tokens.push_back(next_token);
    }
    return generated;
  }

  void bind_weights() {
    embedding_ = weights_.at("model.embed_tokens.weight");
    lm_head_ = weights_.at("lm_head.weight");
    final_norm_ = weights_.at("model.norm.weight");
    layers_.resize(static_cast<std::size_t>(config_.num_hidden_layers));
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      const auto prefix = "model.layers." + std::to_string(i) + ".";
      auto& layer = layers_[i];
      layer.input_norm = weights_.at(prefix + "input_layernorm.weight");
      layer.q_proj = weights_.at(prefix + "self_attn.q_proj.weight");
      layer.q_norm = weights_.at(prefix + "self_attn.q_norm.weight");
      layer.k_proj = weights_.at(prefix + "self_attn.k_proj.weight");
      layer.k_norm = weights_.at(prefix + "self_attn.k_norm.weight");
      layer.v_proj = weights_.at(prefix + "self_attn.v_proj.weight");
      layer.o_proj = weights_.at(prefix + "self_attn.o_proj.weight");
      layer.post_norm = weights_.at(prefix + "post_attention_layernorm.weight");
      layer.gate_proj = weights_.at(prefix + "mlp.gate_proj.weight");
      layer.up_proj = weights_.at(prefix + "mlp.up_proj.weight");
      layer.down_proj = weights_.at(prefix + "mlp.down_proj.weight");
    }
  }

  CpuModelOutput generate_mps(const std::vector<ChatMessage>& messages,
                              std::size_t max_new_tokens, bool enable_thinking,
                              bool verify_kv_cache, Device compute_device,
                              const CpuSamplingConfig& sampling,
                              const std::function<void(std::string_view)>& stream_token) {
    if (compute_device.index != 0) {
      throw std::runtime_error("only mps:0 is supported");
    }
    const auto effective_sampling = make_effective_sampling(generation_, sampling);
    std::mt19937_64 rng(effective_sampling.seed);
    std::vector<std::int64_t> prompt_tokens;
    {
      auto span = profile_span("request.tokenize");
      prompt_tokens = tokenizer_.encode_chat_messages(messages, enable_thinking);
    }
    if (prompt_tokens.empty()) {
      throw std::runtime_error("tokenizer produced no prompt tokens");
    }
    if (profiler_ != nullptr) {
      profiler_->set_metadata("prompt_tokens", prompt_tokens.size());
      profiler_->set_metadata("device", compute_device.to_string());
    }
    dump_i64("prompt_tokens", {static_cast<std::uint64_t>(prompt_tokens.size())}, prompt_tokens);

    const auto total_tokens = prompt_tokens.size() + max_new_tokens + 1U;
    {
      auto span = profile_span("request.reset_kv_cache");
      prepare_mps_full_forward(total_tokens);
    }

    std::size_t position = 0;
    {
      auto span = profile_span("request.prefill");
      for (const auto token : prompt_tokens) {
        forward_token_mps(token, position);
        ++position;
      }
    }

    std::vector<std::int64_t> generated;
    generated.reserve(max_new_tokens);
    {
      auto span = profile_span("request.decode");
      for (std::size_t step = 0; step < max_new_tokens; ++step) {
        const auto logits = compute_logits_mps_device();
        dump_f32(position_tensor_name(position - 1U, "logits"),
                 {static_cast<std::uint64_t>(config_.vocab_size)}, logits);
        const auto next_token = select_next_token(logits, rng, effective_sampling);
        if (is_eos(next_token)) {
          break;
        }
        generated.push_back(next_token);
        if (stream_token) {
          const auto token_text = tokenizer_.decode(std::vector<std::int64_t>{next_token});
          stream_token(token_text);
        }
        forward_token_mps(next_token, position);
        ++position;
      }
    }
    dump_i64("generated_tokens", {static_cast<std::uint64_t>(generated.size())}, generated);

    const auto cached_stats = mps_kv_stats();
    bool verified = false;
    if (verify_kv_cache) {
      const auto recomputed =
        generate_mps_recomputed_tokens(prompt_tokens, max_new_tokens, effective_sampling);
      if (recomputed != generated) {
        throw std::runtime_error("MPS KV cache verification failed: cached decode differs from "
                                 "full-prefix recompute");
      }
      verified = true;
    }

    if (profiler_ != nullptr) {
      profiler_->set_metadata("generated_tokens", generated.size());
      profiler_->set_status("ok");
    }
    return CpuModelOutput{tokenizer_.decode(generated), cached_stats, verified,
                          prompt_tokens.size(), generated.size()};
  }

  std::vector<std::int64_t> generate_mps_recomputed_tokens(
    const std::vector<std::int64_t>& prompt_tokens, std::size_t max_new_tokens,
    const EffectiveSamplingConfig& sampling) {
    DumperScope disable_dump(dumper_, nullptr);
    std::mt19937_64 rng(sampling.seed);
    std::vector<std::int64_t> tokens = prompt_tokens;
    std::vector<std::int64_t> generated;
    generated.reserve(max_new_tokens);

    for (std::size_t step = 0; step < max_new_tokens; ++step) {
      reset_mps_workspace(tokens.size() + 1U);
      for (std::size_t position = 0; position < tokens.size(); ++position) {
        forward_token_mps(tokens[position], position);
      }
      const auto logits = compute_logits_mps_device();
      const auto next_token = select_next_token(logits, rng, sampling);
      if (is_eos(next_token)) {
        break;
      }
      generated.push_back(next_token);
      tokens.push_back(next_token);
    }
    return generated;
  }

  void ensure_mps_context() {
    if (mps_context_ != nullptr) {
      return;
    }
    auto context_result = mps::MpsContext::create();
    if (!context_result.is_ok()) {
      throw std::runtime_error("MPS initialization failed: " +
                               context_result.status().message());
    }
    auto context = std::move(context_result.value());
    mps_context_ = std::make_unique<mps::MpsContext>(std::move(context));
  }

  std::size_t f32_bytes(std::size_t values) const {
    if (values > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::runtime_error("MPS f32 buffer byte count overflow");
    }
    return values * sizeof(float);
  }

  mps::MpsBuffer make_mps_buffer(std::size_t byte_size, std::string_view name) {
    auto result = mps_context_->make_buffer(byte_size);
    if (!result.is_ok()) {
      throw std::runtime_error("MPS " + std::string{name} + " allocation failed: " +
                               result.status().message());
    }
    return std::move(result.value());
  }

  mps::MpsBuffer upload_tensor_to_mps(const TensorView& tensor, std::string_view name) {
    auto buffer = make_mps_buffer(static_cast<std::size_t>(tensor.byte_size), name);
    const auto status =
      mps_context_->copy_to_buffer(buffer, tensor.data, static_cast<std::size_t>(tensor.byte_size));
    if (!status.is_ok()) {
      throw std::runtime_error("MPS " + std::string{name} + " upload failed: " +
                               status.message());
    }
    return buffer;
  }

  void prepare_mps_full_forward(std::size_t capacity_tokens) {
    ensure_mps_context();
    if (!mps_weights_ready_) {
      mps_embedding_ = upload_tensor_to_mps(embedding_, "embedding");
      mps_final_norm_ = upload_tensor_to_mps(final_norm_, "final_norm");
      mps_lm_head_ = upload_tensor_to_mps(lm_head_, "lm_head");
      mps_layers_.resize(layers_.size());
      for (std::size_t i = 0; i < layers_.size(); ++i) {
        auto& target = mps_layers_[i];
        const auto prefix = "layer." + std::to_string(i) + ".";
        target.input_norm = upload_tensor_to_mps(layers_[i].input_norm, prefix + "input_norm");
        target.q_proj = upload_tensor_to_mps(layers_[i].q_proj, prefix + "q_proj");
        target.q_norm = upload_tensor_to_mps(layers_[i].q_norm, prefix + "q_norm");
        target.k_proj = upload_tensor_to_mps(layers_[i].k_proj, prefix + "k_proj");
        target.k_norm = upload_tensor_to_mps(layers_[i].k_norm, prefix + "k_norm");
        target.v_proj = upload_tensor_to_mps(layers_[i].v_proj, prefix + "v_proj");
        target.o_proj = upload_tensor_to_mps(layers_[i].o_proj, prefix + "o_proj");
        target.post_norm = upload_tensor_to_mps(layers_[i].post_norm, prefix + "post_norm");
        target.gate_proj = upload_tensor_to_mps(layers_[i].gate_proj, prefix + "gate_proj");
        target.up_proj = upload_tensor_to_mps(layers_[i].up_proj, prefix + "up_proj");
        target.down_proj = upload_tensor_to_mps(layers_[i].down_proj, prefix + "down_proj");
      }
      mps_weights_ready_ = true;
      mps_lm_head_ready_ = true;
    }
    reset_mps_workspace(capacity_tokens);
  }

  void reset_mps_workspace(std::size_t capacity_tokens) {
    if (capacity_tokens == 0) {
      throw std::runtime_error("MPS KV cache capacity must be positive");
    }
    if (mps_workspace_.capacity_tokens >= capacity_tokens && mps_workspace_.hidden.valid()) {
      mps_workspace_.used_tokens = 0;
      return;
    }

    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    const auto heads = static_cast<std::size_t>(config_.num_attention_heads);
    const auto kv_heads = static_cast<std::size_t>(config_.num_key_value_heads);
    const auto layers = static_cast<std::size_t>(config_.num_hidden_layers);
    const auto intermediate = static_cast<std::size_t>(config_.intermediate_size);
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto attn_dim = heads * head_dim;
    const auto kv_dim = kv_heads * head_dim;

    MpsWorkspace workspace;
    workspace.hidden = make_mps_buffer(f32_bytes(hidden_size), "hidden");
    workspace.normed = make_mps_buffer(f32_bytes(hidden_size), "normed");
    workspace.q = make_mps_buffer(f32_bytes(attn_dim), "q");
    workspace.k = make_mps_buffer(f32_bytes(kv_dim), "k");
    workspace.v = make_mps_buffer(f32_bytes(kv_dim), "v");
    workspace.attn_out = make_mps_buffer(f32_bytes(attn_dim), "attention_out");
    workspace.projected = make_mps_buffer(f32_bytes(hidden_size), "projected");
    workspace.gate = make_mps_buffer(f32_bytes(intermediate), "gate");
    workspace.up = make_mps_buffer(f32_bytes(intermediate), "up");
    workspace.down = make_mps_buffer(f32_bytes(hidden_size), "down");
    workspace.logits = make_mps_buffer(f32_bytes(vocab), "logits");
    workspace.key_cache =
      make_mps_buffer(f32_bytes(layers * capacity_tokens * kv_dim), "key_cache");
    workspace.value_cache =
      make_mps_buffer(f32_bytes(layers * capacity_tokens * kv_dim), "value_cache");
    workspace.capacity_tokens = capacity_tokens;
    workspace.used_tokens = 0;
    mps_workspace_ = std::move(workspace);
  }

  KvCacheStats mps_kv_stats() const {
    const auto layers = static_cast<std::size_t>(config_.num_hidden_layers);
    const auto kv_heads = static_cast<std::size_t>(config_.num_key_value_heads);
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    const auto kv_dim = kv_heads * head_dim;
    const auto values = layers * mps_workspace_.capacity_tokens * kv_dim;
    const auto key_bytes = static_cast<std::uint64_t>(f32_bytes(values));
    const auto value_bytes = static_cast<std::uint64_t>(f32_bytes(values));
    return KvCacheStats{true,
                        layers,
                        kv_heads,
                        head_dim,
                        kv_dim,
                        mps_workspace_.capacity_tokens,
                        mps_workspace_.used_tokens,
                        key_bytes,
                        value_bytes,
                        key_bytes + value_bytes};
  }

  void expect_mps(Status status, std::string_view operation) const {
    if (!status.is_ok()) {
      throw std::runtime_error("MPS " + std::string{operation} + " failed: " +
                               status.message());
    }
  }

  void dump_mps_f32(std::string_view name, const std::vector<std::uint64_t>& shape,
                    const mps::MpsBuffer& buffer, std::size_t values) const {
    if (dumper_ == nullptr) {
      return;
    }
    std::vector<float> output(values);
    expect_mps(mps_context_->copy_from_buffer(buffer, output.data(), f32_bytes(values)),
               "debug dump readback");
    dumper_->write_f32(name, shape, output);
  }

  void forward_token_mps(std::int64_t token, std::size_t position) {
    if (position >= mps_workspace_.capacity_tokens) {
      throw std::runtime_error("MPS KV cache position exceeds capacity");
    }
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    expect_mps(mps_context_->embedding_bf16_f32(mps_embedding_, token, hidden_size,
                                               mps_workspace_.hidden),
               "embedding");
    dump_mps_f32(position_tensor_name(position, "embedding"),
                 {static_cast<std::uint64_t>(hidden_size)}, mps_workspace_.hidden,
                 hidden_size);

    for (std::size_t layer_index = 0; layer_index < mps_layers_.size(); ++layer_index) {
      apply_layer_mps(mps_layers_[layer_index], layer_index, position);
    }

    expect_mps(mps_context_->rms_norm_f32_bf16(
                 mps_workspace_.hidden, mps_final_norm_, hidden_size,
                 static_cast<float>(config_.rms_norm_eps), mps_workspace_.normed),
               "final_norm");
    dump_mps_f32(position_tensor_name(position, "final_norm"),
                 {static_cast<std::uint64_t>(hidden_size)}, mps_workspace_.normed,
                 hidden_size);
    mps_workspace_.used_tokens = std::max(mps_workspace_.used_tokens, position + 1U);
  }

  void apply_layer_mps(const MpsLayerWeights& layer, std::size_t layer_index,
                       std::size_t position) {
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    const auto heads = static_cast<std::size_t>(config_.num_attention_heads);
    const auto kv_heads = static_cast<std::size_t>(config_.num_key_value_heads);
    const auto intermediate = static_cast<std::size_t>(config_.intermediate_size);
    const auto attn_dim = heads * head_dim;
    const auto kv_dim = kv_heads * head_dim;
    const auto eps = static_cast<float>(config_.rms_norm_eps);

    {
      auto span = profile_span("mps.input_rms_norm",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->rms_norm_f32_bf16(mps_workspace_.hidden, layer.input_norm,
                                                 hidden_size, eps, mps_workspace_.normed),
                 "input_norm");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "input_norm"),
                 {static_cast<std::uint64_t>(hidden_size)}, mps_workspace_.normed,
                 hidden_size);
    {
      auto span = profile_span("mps.qkv_proj",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->matvec_bf16_f32_device(layer.q_proj, attn_dim, hidden_size,
                                                      mps_workspace_.normed, mps_workspace_.q),
                 "q_proj");
      expect_mps(mps_context_->matvec_bf16_f32_device(layer.k_proj, kv_dim, hidden_size,
                                                      mps_workspace_.normed, mps_workspace_.k),
                 "k_proj");
      expect_mps(mps_context_->matvec_bf16_f32_device(layer.v_proj, kv_dim, hidden_size,
                                                      mps_workspace_.normed, mps_workspace_.v),
                 "v_proj");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "q_proj"),
                 {static_cast<std::uint64_t>(heads), static_cast<std::uint64_t>(head_dim)},
                 mps_workspace_.q, attn_dim);
    dump_mps_f32(layer_tensor_name(position, layer_index, "k_proj"),
                 {static_cast<std::uint64_t>(kv_heads), static_cast<std::uint64_t>(head_dim)},
                 mps_workspace_.k, kv_dim);
    dump_mps_f32(layer_tensor_name(position, layer_index, "v_proj"),
                 {static_cast<std::uint64_t>(kv_heads), static_cast<std::uint64_t>(head_dim)},
                 mps_workspace_.v, kv_dim);
    {
      auto span = profile_span("mps.qk_norm",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->qk_norm_f32_bf16(mps_workspace_.q, layer.q_norm, heads,
                                                head_dim, eps),
                 "q_norm");
      expect_mps(mps_context_->qk_norm_f32_bf16(mps_workspace_.k, layer.k_norm, kv_heads,
                                                head_dim, eps),
                 "k_norm");
    }
    {
      auto span = profile_span("mps.rope",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->rope_f32(mps_workspace_.q, heads, head_dim, position,
                                        static_cast<float>(config_.rope_theta)),
                 "q_rope");
      expect_mps(mps_context_->rope_f32(mps_workspace_.k, kv_heads, head_dim, position,
                                        static_cast<float>(config_.rope_theta)),
                 "k_rope");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "q_norm_rope"),
                 {static_cast<std::uint64_t>(heads), static_cast<std::uint64_t>(head_dim)},
                 mps_workspace_.q, attn_dim);
    dump_mps_f32(layer_tensor_name(position, layer_index, "k_norm_rope"),
                 {static_cast<std::uint64_t>(kv_heads), static_cast<std::uint64_t>(head_dim)},
                 mps_workspace_.k, kv_dim);

    const auto cache_offset =
      (layer_index * mps_workspace_.capacity_tokens + position) * kv_dim;
    {
      auto span = profile_span("mps.kv_store",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->copy_f32_region(mps_workspace_.k, mps_workspace_.key_cache, 0,
                                               cache_offset, kv_dim),
                 "store_k");
      expect_mps(mps_context_->copy_f32_region(mps_workspace_.v, mps_workspace_.value_cache, 0,
                                               cache_offset, kv_dim),
                 "store_v");
    }

    {
      auto span = profile_span("mps.attention",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->attention_f32(mps_workspace_.q, mps_workspace_.key_cache,
                                             mps_workspace_.value_cache, layer_index, position,
                                             mps_workspace_.capacity_tokens, heads, kv_heads,
                                             head_dim, mps_workspace_.attn_out),
                 "attention");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "attention_out"),
                 {static_cast<std::uint64_t>(heads), static_cast<std::uint64_t>(head_dim)},
                 mps_workspace_.attn_out, attn_dim);
    {
      auto span = profile_span("mps.o_proj",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->matvec_bf16_f32_device(layer.o_proj, hidden_size, attn_dim,
                                                      mps_workspace_.attn_out,
                                                      mps_workspace_.projected),
                 "o_proj");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "attention_projected"),
                 {static_cast<std::uint64_t>(hidden_size)}, mps_workspace_.projected,
                 hidden_size);
    {
      auto span = profile_span("mps.attention_residual",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->add_f32_in_place(mps_workspace_.hidden,
                                                mps_workspace_.projected, hidden_size),
                 "attention_residual");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "attention_residual"),
                 {static_cast<std::uint64_t>(hidden_size)}, mps_workspace_.hidden,
                 hidden_size);

    {
      auto span = profile_span("mps.post_attention_rms_norm",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->rms_norm_f32_bf16(mps_workspace_.hidden, layer.post_norm,
                                                 hidden_size, eps, mps_workspace_.normed),
                 "post_attention_norm");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "post_attention_norm"),
                 {static_cast<std::uint64_t>(hidden_size)}, mps_workspace_.normed,
                 hidden_size);
    {
      auto span = profile_span("mps.mlp_gate_up",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->matvec_bf16_f32_device(layer.gate_proj, intermediate,
                                                      hidden_size, mps_workspace_.normed,
                                                      mps_workspace_.gate),
                 "gate_proj");
      expect_mps(mps_context_->matvec_bf16_f32_device(layer.up_proj, intermediate,
                                                      hidden_size, mps_workspace_.normed,
                                                      mps_workspace_.up),
                 "up_proj");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "mlp_gate_proj"),
                 {static_cast<std::uint64_t>(intermediate)}, mps_workspace_.gate,
                 intermediate);
    dump_mps_f32(layer_tensor_name(position, layer_index, "mlp_up_proj"),
                 {static_cast<std::uint64_t>(intermediate)}, mps_workspace_.up,
                 intermediate);
    {
      auto span = profile_span("mps.mlp_silu_mul",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->silu_mul_f32_in_place(mps_workspace_.gate, mps_workspace_.up,
                                                     intermediate),
                 "silu_gate_mul");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "mlp_gate_silu_mul"),
                 {static_cast<std::uint64_t>(intermediate)}, mps_workspace_.gate,
                 intermediate);
    {
      auto span = profile_span("mps.mlp_down_proj",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->matvec_bf16_f32_device(layer.down_proj, hidden_size,
                                                      intermediate, mps_workspace_.gate,
                                                      mps_workspace_.down),
                 "down_proj");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "mlp_down_proj"),
                 {static_cast<std::uint64_t>(hidden_size)}, mps_workspace_.down,
                 hidden_size);
    {
      auto span = profile_span("mps.mlp_residual",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      expect_mps(mps_context_->add_f32_in_place(mps_workspace_.hidden, mps_workspace_.down,
                                                hidden_size),
                 "mlp_residual");
    }
    dump_mps_f32(layer_tensor_name(position, layer_index, "layer_output"),
                 {static_cast<std::uint64_t>(hidden_size)}, mps_workspace_.hidden,
                 hidden_size);
  }

  std::vector<float> compute_logits_mps_device() {
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    {
      auto span = profile_span("mps.lm_head");
      expect_mps(mps_context_->matvec_bf16_f32_device(mps_lm_head_, vocab, hidden_size,
                                                      mps_workspace_.normed,
                                                      mps_workspace_.logits),
                 "lm_head");
    }
    std::vector<float> logits(vocab);
    {
      auto span = profile_span("mps.logits_readback");
      expect_mps(mps_context_->copy_from_buffer(mps_workspace_.logits, logits.data(),
                                                f32_bytes(vocab)),
                 "logits readback");
    }
    return logits;
  }

  void prepare_mps_lm_head() {
    if (mps_lm_head_ready_) {
      return;
    }
    auto context_result = mps::MpsContext::create();
    if (!context_result.is_ok()) {
      throw std::runtime_error("MPS initialization failed: " +
                               context_result.status().message());
    }
    auto context = std::move(context_result.value());
    auto buffer_result = context.make_buffer(static_cast<std::size_t>(lm_head_.byte_size));
    if (!buffer_result.is_ok()) {
      throw std::runtime_error("MPS lm_head allocation failed: " +
                               buffer_result.status().message());
    }
    auto buffer = std::move(buffer_result.value());
    const auto copy_status =
      context.copy_to_buffer(buffer, lm_head_.data, static_cast<std::size_t>(lm_head_.byte_size));
    if (!copy_status.is_ok()) {
      throw std::runtime_error("MPS lm_head upload failed: " + copy_status.message());
    }
    auto workspace_result = context.make_matvec_workspace(
      static_cast<std::size_t>(config_.vocab_size),
      static_cast<std::size_t>(config_.hidden_size));
    if (!workspace_result.is_ok()) {
      throw std::runtime_error("MPS lm_head workspace allocation failed: " +
                               workspace_result.status().message());
    }
    mps_context_ = std::make_unique<mps::MpsContext>(std::move(context));
    mps_lm_head_ = std::move(buffer);
    mps_lm_head_workspace_ = std::move(workspace_result.value());
    mps_lm_head_ready_ = true;
  }

  std::vector<float> forward_token(std::int64_t token, std::size_t position) {
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    std::vector<float> hidden(hidden_size);
    {
      auto span = profile_span("forward_token.embedding",
                               {{"position", std::to_string(position)}});
      for (std::size_t i = 0; i < hidden_size; ++i) {
        hidden[i] = bf16_to_float(embedding_.data, static_cast<std::uint64_t>(token) *
                                                     static_cast<std::uint64_t>(hidden_size) + i);
      }
    }
    dump_f32(position_tensor_name(position, "embedding"), {static_cast<std::uint64_t>(hidden_size)},
             hidden);

    {
      auto span = profile_span("forward_token.layers",
                               {{"position", std::to_string(position)}});
      for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        apply_layer(layers_[layer_index], layer_index, position, hidden);
      }
    }

    std::vector<float> normed(hidden_size);
    {
      auto span = profile_span("forward_token.final_norm",
                               {{"position", std::to_string(position)}});
      rms_norm(hidden, final_norm_, normed);
    }
    dump_f32(position_tensor_name(position, "final_norm"),
             {static_cast<std::uint64_t>(hidden_size)}, normed);
    return normed;
  }

  void apply_layer(const LayerWeights& layer, std::size_t layer_index, std::size_t position,
                   std::vector<float>& hidden) {
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    const auto heads = static_cast<std::size_t>(config_.num_attention_heads);
    const auto kv_heads = static_cast<std::size_t>(config_.num_key_value_heads);
    const auto attn_dim = heads * head_dim;
    const auto kv_dim = kv_heads * head_dim;
    const auto intermediate = static_cast<std::size_t>(config_.intermediate_size);

    std::vector<float> normed(hidden_size);
    {
      auto span = profile_span("layer.input_rms_norm",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      rms_norm(hidden, layer.input_norm, normed);
    }
    dump_f32(layer_tensor_name(position, layer_index, "input_norm"),
             {static_cast<std::uint64_t>(hidden_size)}, normed);

    std::vector<float> q(attn_dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);
    {
      auto span = profile_span("layer.qkv_proj",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      matvec(layer.q_proj, normed, q);
      matvec(layer.k_proj, normed, k);
      matvec(layer.v_proj, normed, v);
    }
    dump_f32(layer_tensor_name(position, layer_index, "q_proj"),
             {static_cast<std::uint64_t>(heads), static_cast<std::uint64_t>(head_dim)}, q);
    dump_f32(layer_tensor_name(position, layer_index, "k_proj"),
             {static_cast<std::uint64_t>(kv_heads), static_cast<std::uint64_t>(head_dim)}, k);
    dump_f32(layer_tensor_name(position, layer_index, "v_proj"),
             {static_cast<std::uint64_t>(kv_heads), static_cast<std::uint64_t>(head_dim)}, v);
    {
      auto span = profile_span("layer.qk_norm",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      qk_norm(q, heads, layer.q_norm);
      qk_norm(k, kv_heads, layer.k_norm);
    }
    {
      auto span = profile_span("layer.rope",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      apply_rope(q, heads, position);
      apply_rope(k, kv_heads, position);
    }
    dump_f32(layer_tensor_name(position, layer_index, "q_norm_rope"),
             {static_cast<std::uint64_t>(heads), static_cast<std::uint64_t>(head_dim)}, q);
    dump_f32(layer_tensor_name(position, layer_index, "k_norm_rope"),
             {static_cast<std::uint64_t>(kv_heads), static_cast<std::uint64_t>(head_dim)}, k);
    {
      auto span = profile_span("layer.kv_store",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      store_kv(layer_index, position, k, v);
    }

    std::vector<float> attn_out(attn_dim);
    {
      auto span = profile_span("layer.attention",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      attention(layer_index, position, q, attn_out);
    }
    dump_f32(layer_tensor_name(position, layer_index, "attention_out"),
             {static_cast<std::uint64_t>(heads), static_cast<std::uint64_t>(head_dim)}, attn_out);
    std::vector<float> projected(hidden_size);
    {
      auto span = profile_span("layer.o_proj",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      matvec(layer.o_proj, attn_out, projected);
    }
    dump_f32(layer_tensor_name(position, layer_index, "attention_projected"),
             {static_cast<std::uint64_t>(hidden_size)}, projected);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      hidden[i] += projected[i];
    }
    dump_f32(layer_tensor_name(position, layer_index, "attention_residual"),
             {static_cast<std::uint64_t>(hidden_size)}, hidden);

    {
      auto span = profile_span("layer.post_attention_rms_norm",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      rms_norm(hidden, layer.post_norm, normed);
    }
    dump_f32(layer_tensor_name(position, layer_index, "post_attention_norm"),
             {static_cast<std::uint64_t>(hidden_size)}, normed);
    std::vector<float> gate(intermediate);
    std::vector<float> up(intermediate);
    {
      auto span = profile_span("layer.mlp_gate_up",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      matvec(layer.gate_proj, normed, gate);
      matvec(layer.up_proj, normed, up);
    }
    dump_f32(layer_tensor_name(position, layer_index, "mlp_gate_proj"),
             {static_cast<std::uint64_t>(intermediate)}, gate);
    dump_f32(layer_tensor_name(position, layer_index, "mlp_up_proj"),
             {static_cast<std::uint64_t>(intermediate)}, up);
    {
      auto span = profile_span("layer.mlp_silu_mul",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      for (std::size_t i = 0; i < intermediate; ++i) {
        gate[i] = silu(gate[i]) * up[i];
      }
    }
    dump_f32(layer_tensor_name(position, layer_index, "mlp_gate_silu_mul"),
             {static_cast<std::uint64_t>(intermediate)}, gate);
    std::vector<float> down(hidden_size);
    {
      auto span = profile_span("layer.mlp_down_proj",
                               {{"layer", std::to_string(layer_index)},
                                {"position", std::to_string(position)}});
      matvec(layer.down_proj, gate, down);
    }
    dump_f32(layer_tensor_name(position, layer_index, "mlp_down_proj"),
             {static_cast<std::uint64_t>(hidden_size)}, down);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      hidden[i] += down[i];
    }
    dump_f32(layer_tensor_name(position, layer_index, "layer_output"),
             {static_cast<std::uint64_t>(hidden_size)}, hidden);
  }

  void rms_norm(const std::vector<float>& input, const TensorView& weight,
                std::vector<float>& output) const {
    double mean_square = 0.0;
    for (const auto value : input) {
      mean_square += static_cast<double>(value) * static_cast<double>(value);
    }
    mean_square /= static_cast<double>(input.size());
    const auto scale = static_cast<float>(1.0 / std::sqrt(mean_square + config_.rms_norm_eps));
    for (std::size_t i = 0; i < input.size(); ++i) {
      output[i] = input[i] * scale * bf16_to_float(weight.data, i);
    }
  }

  void qk_norm(std::vector<float>& values, std::size_t heads, const TensorView& weight) const {
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    for (std::size_t head = 0; head < heads; ++head) {
      double mean_square = 0.0;
      const auto base = head * head_dim;
      for (std::size_t i = 0; i < head_dim; ++i) {
        mean_square += static_cast<double>(values[base + i]) * values[base + i];
      }
      mean_square /= static_cast<double>(head_dim);
      const auto scale = static_cast<float>(1.0 / std::sqrt(mean_square + config_.rms_norm_eps));
      for (std::size_t i = 0; i < head_dim; ++i) {
        values[base + i] *= scale * bf16_to_float(weight.data, i);
      }
    }
  }

  void apply_rope(std::vector<float>& values, std::size_t heads, std::size_t position) const {
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    const auto half_dim = head_dim / 2U;
    for (std::size_t head = 0; head < heads; ++head) {
      const auto base = head * head_dim;
      for (std::size_t i = 0; i < half_dim; ++i) {
        const auto freq =
          1.0 / std::pow(config_.rope_theta,
                         static_cast<double>(2U * i) / static_cast<double>(head_dim));
        const auto angle = static_cast<double>(position) * freq;
        const auto cos_v = static_cast<float>(std::cos(angle));
        const auto sin_v = static_cast<float>(std::sin(angle));
        const auto x0 = values[base + i];
        const auto x1 = values[base + half_dim + i];
        values[base + i] = x0 * cos_v - x1 * sin_v;
        values[base + half_dim + i] = x1 * cos_v + x0 * sin_v;
      }
    }
  }

  void matvec(const TensorView& weight, const std::vector<float>& input,
              std::vector<float>& output) const {
    const auto rows = static_cast<std::size_t>(weight.shape[0]);
    const auto cols = static_cast<std::size_t>(weight.shape[1]);
    for (std::size_t row = 0; row < rows; ++row) {
      float sum = 0.0F;
      const auto row_offset = static_cast<std::uint64_t>(row * cols);
      for (std::size_t col = 0; col < cols; ++col) {
        sum += bf16_to_float(weight.data, row_offset + col) * input[col];
      }
      output[row] = sum;
    }
  }

  void store_kv(std::size_t layer, std::size_t position, const std::vector<float>& k,
                const std::vector<float>& v) {
    kv_cache_.store(layer, position, k, v);
  }

  void attention(std::size_t layer, std::size_t position, const std::vector<float>& q,
                 std::vector<float>& output) const {
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    const auto heads = static_cast<std::size_t>(config_.num_attention_heads);
    const auto kv_heads = static_cast<std::size_t>(config_.num_key_value_heads);
    const auto group = heads / kv_heads;
    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(position + 1U);

    for (std::size_t head = 0; head < heads; ++head) {
      const auto kv_head = head / group;
      const auto q_base = head * head_dim;
      float max_score = -std::numeric_limits<float>::infinity();
      {
        auto span = profile_span("attention.score",
                                 {{"layer", std::to_string(layer)},
                                  {"position", std::to_string(position)},
                                  {"head", std::to_string(head)}});
        for (std::size_t t = 0; t <= position; ++t) {
          const auto* key = kv_cache_.key_ptr(layer, t, kv_head);
          float score = 0.0F;
          for (std::size_t d = 0; d < head_dim; ++d) {
            score += q[q_base + d] * key[d];
          }
          score *= scale;
          scores[t] = score;
          max_score = std::max(max_score, score);
        }
      }

      float denom = 0.0F;
      {
        auto span = profile_span("attention.softmax",
                                 {{"layer", std::to_string(layer)},
                                  {"position", std::to_string(position)},
                                  {"head", std::to_string(head)}});
        for (std::size_t t = 0; t <= position; ++t) {
          scores[t] = std::exp(scores[t] - max_score);
          denom += scores[t];
        }
      }
      {
        auto span = profile_span("attention.reduce",
                                 {{"layer", std::to_string(layer)},
                                  {"position", std::to_string(position)},
                                  {"head", std::to_string(head)}});
        for (std::size_t d = 0; d < head_dim; ++d) {
          float value = 0.0F;
          for (std::size_t t = 0; t <= position; ++t) {
            const auto* cached_value = kv_cache_.value_ptr(layer, t, kv_head);
            value += (scores[t] / denom) * cached_value[d];
          }
          output[q_base + d] = value;
        }
      }
    }
  }

  std::vector<float> compute_logits(const std::vector<float>& hidden,
                                    bool use_mps_lm_head) const {
    if (use_mps_lm_head) {
      return compute_logits_mps(hidden);
    }
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    std::vector<float> logits(vocab);
    {
      auto span = profile_span("lm_head");
      for (std::size_t row = 0; row < vocab; ++row) {
        float logit = 0.0F;
        const auto row_offset = static_cast<std::uint64_t>(row * hidden_size);
        for (std::size_t col = 0; col < hidden_size; ++col) {
          logit += bf16_to_float(lm_head_.data, row_offset + col) * hidden[col];
        }
        logits[row] = logit;
      }
    }
    return logits;
  }

  std::vector<float> compute_logits_mps(const std::vector<float>& hidden) const {
    if (mps_context_ == nullptr || !mps_lm_head_.valid()) {
      throw std::runtime_error("MPS lm_head is not initialized");
    }
    auto span = profile_span("lm_head");
    const auto logits_result =
      mps_context_->matvec_bf16_f32(mps_lm_head_, mps_lm_head_workspace_, hidden);
    if (!logits_result.is_ok()) {
      throw std::runtime_error("MPS lm_head matvec failed: " +
                               logits_result.status().message());
    }
    return logits_result.value();
  }

  static std::int64_t select_greedy_token(const std::vector<float>& logits) {
    float best = -std::numeric_limits<float>::infinity();
    std::int64_t best_id = kEndOfText;
    for (std::size_t row = 0; row < logits.size(); ++row) {
      if (logits[row] > best) {
        best = logits[row];
        best_id = static_cast<std::int64_t>(row);
      }
    }
    return best_id;
  }

  static std::int64_t select_next_token(const std::vector<float>& logits, std::mt19937_64& rng,
                                        const EffectiveSamplingConfig& sampling) {
    if (!sampling.do_sample) {
      return select_greedy_token(logits);
    }
    return sample_next_token(logits, rng, sampling);
  }

  static std::int64_t sample_next_token(const std::vector<float>& logits, std::mt19937_64& rng,
                                        const EffectiveSamplingConfig& sampling) {
    struct Candidate {
      std::int64_t id;
      float logit;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
      candidates.push_back(Candidate{static_cast<std::int64_t>(i), logits[i]});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) { return lhs.logit > rhs.logit; });

    if (sampling.top_k > 0 && sampling.top_k < candidates.size()) {
      candidates.resize(sampling.top_k);
    }

    const auto max_logit = static_cast<double>(candidates.front().logit);
    std::vector<double> weights;
    weights.reserve(candidates.size());
    double total_weight = 0.0;
    for (const auto& candidate : candidates) {
      const auto weight =
        std::exp((static_cast<double>(candidate.logit) - max_logit) / sampling.temperature);
      weights.push_back(weight);
      total_weight += weight;
    }

    std::size_t kept = candidates.size();
    if (sampling.top_p < 1.0) {
      double cumulative = 0.0;
      kept = 0;
      for (std::size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i] / total_weight;
        kept = i + 1U;
        if (cumulative >= sampling.top_p) {
          break;
        }
      }
      candidates.resize(kept);
      weights.resize(kept);
      total_weight = 0.0;
      for (const auto weight : weights) {
        total_weight += weight;
      }
    }

    std::uniform_real_distribution<double> distribution(0.0, total_weight);
    const auto target = distribution(rng);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
      cumulative += weights[i];
      if (target <= cumulative) {
        return candidates[i].id;
      }
    }
    return candidates.back().id;
  }

  bool is_eos(std::int64_t token) const {
    if (token == config_.eos_token_id || token == kEndOfText) {
      return true;
    }
    return std::find(generation_.eos_token_ids.begin(), generation_.eos_token_ids.end(), token) !=
           generation_.eos_token_ids.end();
  }

  static float silu(float value) { return value / (1.0F + std::exp(-value)); }

  ScopedProfileSpan profile_span(std::string_view name) const {
    if (profiler_ == nullptr) {
      return {};
    }
    return profiler_->scoped(name);
  }

  ScopedProfileSpan profile_span(std::string_view name, std::vector<ProfileField> fields) const {
    if (profiler_ == nullptr) {
      return {};
    }
    return profiler_->scoped(name, std::move(fields));
  }

  void dump_f32(std::string_view name, const std::vector<std::uint64_t>& shape,
                const std::vector<float>& values) const {
    if (dumper_ != nullptr) {
      dumper_->write_f32(name, shape, values);
    }
  }

  void dump_i64(std::string_view name, const std::vector<std::uint64_t>& shape,
                const std::vector<std::int64_t>& values) const {
    if (dumper_ != nullptr) {
      dumper_->write_i64(name, shape, values);
    }
  }

  ModelConfig config_;
  GenerationConfig generation_;
  QwenTokenizer tokenizer_;
  SafeTensorMap weights_;
  TensorView embedding_;
  TensorView lm_head_;
  TensorView final_norm_;
  std::vector<LayerWeights> layers_;
  KvCache kv_cache_;
  DebugDumper* dumper_{nullptr};
  RequestProfiler* profiler_{nullptr};
  std::unique_ptr<mps::MpsContext> mps_context_;
  mps::MpsBuffer mps_embedding_;
  mps::MpsBuffer mps_final_norm_;
  mps::MpsBuffer mps_lm_head_;
  mutable mps::MpsMatVecWorkspace mps_lm_head_workspace_;
  std::vector<MpsLayerWeights> mps_layers_;
  MpsWorkspace mps_workspace_;
  bool mps_weights_ready_{false};
  bool mps_lm_head_ready_{false};
};

CpuQwenModel& cached_model(const std::filesystem::path& model_dir) {
  static std::unique_ptr<CpuQwenModel> model;
  static std::filesystem::path loaded_dir;
  if (!model || loaded_dir != model_dir) {
    model = std::make_unique<CpuQwenModel>(CpuQwenModel::load(model_dir));
    loaded_dir = model_dir;
  }
  return *model;
}

std::string shape_to_string(const std::vector<std::uint64_t>& shape) {
  std::ostringstream output;
  output << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << shape[i];
  }
  output << ']';
  return output.str();
}

void expect_shape(const SafeTensorMap& weights, std::string_view name,
                  const std::vector<std::uint64_t>& expected) {
  const auto& tensor = weights.at(name);
  if (tensor.shape != expected) {
    std::ostringstream output;
    output << "shape mismatch for " << name << ": expected " << shape_to_string(expected)
           << ", got " << shape_to_string(tensor.shape);
    throw std::runtime_error(output.str());
  }
}

void validate_qwen3_weights(const ModelConfig& config, const SafeTensorMap& weights) {
  const auto hidden = static_cast<std::uint64_t>(config.hidden_size);
  const auto head_dim = static_cast<std::uint64_t>(config.head_dim);
  const auto attn_dim =
    static_cast<std::uint64_t>(config.num_attention_heads * config.head_dim);
  const auto kv_dim =
    static_cast<std::uint64_t>(config.num_key_value_heads * config.head_dim);
  const auto intermediate = static_cast<std::uint64_t>(config.intermediate_size);
  const auto vocab = static_cast<std::uint64_t>(config.vocab_size);

  expect_shape(weights, "model.embed_tokens.weight", {vocab, hidden});
  expect_shape(weights, "model.norm.weight", {hidden});
  if (weights.tensors().find("lm_head.weight") != weights.tensors().end()) {
    expect_shape(weights, "lm_head.weight", {vocab, hidden});
  } else if (!config.tie_word_embeddings) {
    throw std::runtime_error("missing tensor: lm_head.weight");
  }

  for (std::uint64_t layer = 0; layer < static_cast<std::uint64_t>(config.num_hidden_layers);
       ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    expect_shape(weights, prefix + "input_layernorm.weight", {hidden});
    expect_shape(weights, prefix + "self_attn.q_proj.weight", {attn_dim, hidden});
    expect_shape(weights, prefix + "self_attn.q_norm.weight", {head_dim});
    expect_shape(weights, prefix + "self_attn.k_proj.weight", {kv_dim, hidden});
    expect_shape(weights, prefix + "self_attn.k_norm.weight", {head_dim});
    expect_shape(weights, prefix + "self_attn.v_proj.weight", {kv_dim, hidden});
    expect_shape(weights, prefix + "self_attn.o_proj.weight", {hidden, attn_dim});
    expect_shape(weights, prefix + "post_attention_layernorm.weight", {hidden});
    expect_shape(weights, prefix + "mlp.gate_proj.weight", {intermediate, hidden});
    expect_shape(weights, prefix + "mlp.up_proj.weight", {intermediate, hidden});
    expect_shape(weights, prefix + "mlp.down_proj.weight", {hidden, intermediate});
  }
}

}  // namespace

CpuModelOutput generate_text(const std::filesystem::path& model_dir,
                             const std::vector<ChatMessage>& messages,
                             std::size_t max_new_tokens, bool enable_thinking,
                             const std::filesystem::path& debug_dump_dir,
                             bool verify_kv_cache, Device compute_device,
                             const CpuSamplingConfig& sampling,
                             const std::function<void(std::string_view)>& stream_token,
                             RequestProfiler* profiler) {
  return cached_model(model_dir)
    .generate(messages, max_new_tokens, enable_thinking, debug_dump_dir, verify_kv_cache,
              compute_device, sampling, stream_token, profiler);
}

std::string build_weight_summary(const std::filesystem::path& model_dir) {
  auto bundle = load_model_bundle(model_dir);
  if (!bundle.is_ok()) {
    throw std::runtime_error(bundle.status().message());
  }
  if (bundle.value().model.gguf) {
    auto gguf = read_gguf_file(bundle.value().model_file);
    if (!gguf.is_ok()) {
      throw std::runtime_error(gguf.status().message());
    }
    std::vector<std::string> tensor_names;
    tensor_names.reserve(gguf.value().tensors.size());
    std::uint64_t tensor_bytes = 0;
    for (const auto& tensor : gguf.value().tensors) {
      tensor_names.push_back(tensor.name);
      tensor_bytes += tensor.byte_size;
    }
    std::sort(tensor_names.begin(), tensor_names.end());

    std::ostringstream output;
    output << "Weights: ok\n";
    output << "Format: GGUF v" << gguf.value().version << '\n';
    output << "File: " << bundle.value().model_file.string() << '\n';
    output << "File size: " << gguf.value().file_size << " bytes\n";
    output << "Tensor data bytes: " << tensor_bytes << '\n';
    output << "Tensor count: " << gguf.value().tensor_count << '\n';
    output << "Metadata entries: " << gguf.value().metadata_count << '\n';
    output << "Alignment: " << gguf.value().alignment << '\n';
    output << "GGML types:\n";
    for (const auto& entry : gguf_tensor_type_counts(gguf.value())) {
      output << "- " << entry.first << ": " << entry.second << '\n';
    }
    output << "First tensors:\n";
    const auto preview_count = std::min<std::size_t>(tensor_names.size(), 12U);
    for (std::size_t i = 0; i < preview_count; ++i) {
      const auto it = std::find_if(gguf.value().tensors.begin(), gguf.value().tensors.end(),
                                   [&](const GgufTensorInfo& tensor) {
                                     return tensor.name == tensor_names[i];
                                   });
      if (it == gguf.value().tensors.end()) {
        continue;
      }
      output << "- " << it->name << " [";
      for (std::size_t dim = 0; dim < it->shape.size(); ++dim) {
        if (dim != 0) {
          output << ", ";
        }
        output << it->shape[dim];
      }
      output << "] " << ggml_type_name(it->type) << '\n';
    }
    output << "Qwen3.5 GGUF mapping: ok\n";
    output << "Validation: ok\n";
    return output.str();
  }

  const auto weights = SafeTensorMap::load(model_dir / "model.safetensors");
  validate_qwen3_weights(bundle.value().model, weights);

  std::unordered_map<std::string, std::size_t> dtype_counts;
  std::vector<std::string> names;
  names.reserve(weights.tensors().size());
  for (const auto& entry : weights.tensors()) {
    dtype_counts[entry.second.dtype] += 1U;
    names.push_back(entry.first);
  }
  std::sort(names.begin(), names.end());

  std::ostringstream output;
  output << "Weights: ok\n";
  output << "File: " << (model_dir / "model.safetensors").string() << '\n';
  output << "File size: " << weights.file_size() << " bytes\n";
  output << "Header size: " << weights.header_size() << " bytes\n";
  output << "Tensor count: " << weights.tensors().size() << '\n';
  output << "DTypes:\n";
  std::vector<std::string> dtype_names;
  dtype_names.reserve(dtype_counts.size());
  for (const auto& entry : dtype_counts) {
    dtype_names.push_back(entry.first);
  }
  std::sort(dtype_names.begin(), dtype_names.end());
  for (const auto& dtype : dtype_names) {
    output << "- " << dtype << ": " << dtype_counts[dtype] << '\n';
  }
  output << "First tensors:\n";
  const auto preview_count = std::min<std::size_t>(names.size(), 12U);
  for (std::size_t i = 0; i < preview_count; ++i) {
    const auto& tensor = weights.at(names[i]);
    output << "- " << tensor.name << " " << shape_to_string(tensor.shape) << ' ' << tensor.dtype
           << '\n';
  }
  output << "Qwen3 mapping: ok\n";
  output << "Validation: ok\n";
  return output.str();
}

}  // namespace toyllm::cpu
