#include "qwen_cpu_model.hpp"

#include "debug_dump.hpp"
#include "kv_cache.hpp"
#include "safetensors.hpp"
#include "tokenizer.hpp"
#include "tokens.hpp"
#include "toyllm/model/model_config.hpp"

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
                          bool verify_kv_cache, const CpuSamplingConfig& sampling,
                          const std::function<void(std::string_view)>& stream_token) {
    DebugDumper dumper(debug_dump_dir);
    DumperScope dumper_scope(dumper_, dumper.enabled() ? &dumper : nullptr);
    const auto effective_sampling = make_effective_sampling(generation_, sampling);
    std::mt19937_64 rng(effective_sampling.seed);

    const auto prompt_tokens = tokenizer_.encode_chat_messages(messages, enable_thinking);
    if (prompt_tokens.empty()) {
      throw std::runtime_error("tokenizer produced no prompt tokens");
    }
    dump_i64("prompt_tokens", {static_cast<std::uint64_t>(prompt_tokens.size())}, prompt_tokens);

    const auto total_tokens = prompt_tokens.size() + max_new_tokens + 1U;
    reset_kv_cache(total_tokens);

    std::vector<float> hidden;
    std::size_t position = 0;
    for (const auto token : prompt_tokens) {
      hidden = forward_token(token, position);
      ++position;
    }

    std::vector<std::int64_t> generated;
    generated.reserve(max_new_tokens);
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
      const auto logits = compute_logits(hidden);
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
      hidden = forward_token(next_token, position);
      ++position;
    }
    dump_i64("generated_tokens", {static_cast<std::uint64_t>(generated.size())}, generated);
    const auto cached_stats = kv_cache_.stats();
    bool verified = false;
    if (verify_kv_cache) {
      const auto recomputed =
        generate_recomputed_tokens(prompt_tokens, max_new_tokens, effective_sampling);
      if (recomputed != generated) {
        throw std::runtime_error("KV cache verification failed: cached decode differs from "
                                 "full-prefix recompute");
      }
      verified = true;
    }
    return CpuModelOutput{tokenizer_.decode(generated), cached_stats, verified};
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
    const EffectiveSamplingConfig& sampling) {
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
      const auto logits = compute_logits(hidden);
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

  std::vector<float> forward_token(std::int64_t token, std::size_t position) {
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    std::vector<float> hidden(hidden_size);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      hidden[i] = bf16_to_float(embedding_.data, static_cast<std::uint64_t>(token) *
                                                   static_cast<std::uint64_t>(hidden_size) + i);
    }
    dump_f32(position_tensor_name(position, "embedding"), {static_cast<std::uint64_t>(hidden_size)},
             hidden);

    for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
      apply_layer(layers_[layer_index], layer_index, position, hidden);
    }

    std::vector<float> normed(hidden_size);
    rms_norm(hidden, final_norm_, normed);
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
    rms_norm(hidden, layer.input_norm, normed);
    dump_f32(layer_tensor_name(position, layer_index, "input_norm"),
             {static_cast<std::uint64_t>(hidden_size)}, normed);

    std::vector<float> q(attn_dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);
    matvec(layer.q_proj, normed, q);
    matvec(layer.k_proj, normed, k);
    matvec(layer.v_proj, normed, v);
    dump_f32(layer_tensor_name(position, layer_index, "q_proj"),
             {static_cast<std::uint64_t>(heads), static_cast<std::uint64_t>(head_dim)}, q);
    dump_f32(layer_tensor_name(position, layer_index, "k_proj"),
             {static_cast<std::uint64_t>(kv_heads), static_cast<std::uint64_t>(head_dim)}, k);
    dump_f32(layer_tensor_name(position, layer_index, "v_proj"),
             {static_cast<std::uint64_t>(kv_heads), static_cast<std::uint64_t>(head_dim)}, v);
    qk_norm(q, heads, layer.q_norm);
    qk_norm(k, kv_heads, layer.k_norm);
    apply_rope(q, heads, position);
    apply_rope(k, kv_heads, position);
    dump_f32(layer_tensor_name(position, layer_index, "q_norm_rope"),
             {static_cast<std::uint64_t>(heads), static_cast<std::uint64_t>(head_dim)}, q);
    dump_f32(layer_tensor_name(position, layer_index, "k_norm_rope"),
             {static_cast<std::uint64_t>(kv_heads), static_cast<std::uint64_t>(head_dim)}, k);
    store_kv(layer_index, position, k, v);

    std::vector<float> attn_out(attn_dim);
    attention(layer_index, position, q, attn_out);
    dump_f32(layer_tensor_name(position, layer_index, "attention_out"),
             {static_cast<std::uint64_t>(heads), static_cast<std::uint64_t>(head_dim)}, attn_out);
    std::vector<float> projected(hidden_size);
    matvec(layer.o_proj, attn_out, projected);
    dump_f32(layer_tensor_name(position, layer_index, "attention_projected"),
             {static_cast<std::uint64_t>(hidden_size)}, projected);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      hidden[i] += projected[i];
    }
    dump_f32(layer_tensor_name(position, layer_index, "attention_residual"),
             {static_cast<std::uint64_t>(hidden_size)}, hidden);

    rms_norm(hidden, layer.post_norm, normed);
    dump_f32(layer_tensor_name(position, layer_index, "post_attention_norm"),
             {static_cast<std::uint64_t>(hidden_size)}, normed);
    std::vector<float> gate(intermediate);
    std::vector<float> up(intermediate);
    matvec(layer.gate_proj, normed, gate);
    matvec(layer.up_proj, normed, up);
    dump_f32(layer_tensor_name(position, layer_index, "mlp_gate_proj"),
             {static_cast<std::uint64_t>(intermediate)}, gate);
    dump_f32(layer_tensor_name(position, layer_index, "mlp_up_proj"),
             {static_cast<std::uint64_t>(intermediate)}, up);
    for (std::size_t i = 0; i < intermediate; ++i) {
      gate[i] = silu(gate[i]) * up[i];
    }
    dump_f32(layer_tensor_name(position, layer_index, "mlp_gate_silu_mul"),
             {static_cast<std::uint64_t>(intermediate)}, gate);
    std::vector<float> down(hidden_size);
    matvec(layer.down_proj, gate, down);
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

      float denom = 0.0F;
      for (std::size_t t = 0; t <= position; ++t) {
        scores[t] = std::exp(scores[t] - max_score);
        denom += scores[t];
      }
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

  std::vector<float> compute_logits(const std::vector<float>& hidden) const {
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    std::vector<float> logits(vocab);
    for (std::size_t row = 0; row < vocab; ++row) {
      float logit = 0.0F;
      const auto row_offset = static_cast<std::uint64_t>(row * hidden_size);
      for (std::size_t col = 0; col < hidden_size; ++col) {
        logit += bf16_to_float(lm_head_.data, row_offset + col) * hidden[col];
      }
      logits[row] = logit;
    }
    return logits;
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
                             bool verify_kv_cache, const CpuSamplingConfig& sampling,
                             const std::function<void(std::string_view)>& stream_token) {
  return cached_model(model_dir)
    .generate(messages, max_new_tokens, enable_thinking, debug_dump_dir, verify_kv_cache, sampling,
              stream_token);
}

std::string build_weight_summary(const std::filesystem::path& model_dir) {
  auto bundle = load_model_bundle(model_dir);
  if (!bundle.is_ok()) {
    throw std::runtime_error(bundle.status().message());
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
