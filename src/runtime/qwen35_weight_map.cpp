#include "toyllm/runtime/qwen35_weight_map.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace toyllm {

namespace {

using TensorIndex = std::unordered_map<std::string, const GgufTensorInfo*>;

std::string tensor_name(std::int64_t layer, std::string_view suffix) {
  std::ostringstream name;
  name << "blk." << layer << '.' << suffix;
  return name.str();
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

Result<std::uint64_t> checked_dim(std::int64_t value, std::string_view name) {
  if (value <= 0) {
    return Status::invalid_argument(std::string{name} + " must be positive");
  }
  return static_cast<std::uint64_t>(value);
}

TensorIndex make_tensor_index(const GgufFile& gguf) {
  TensorIndex index;
  index.reserve(gguf.tensors.size());
  for (const auto& tensor : gguf.tensors) {
    index.emplace(tensor.name, &tensor);
  }
  return index;
}

Qwen35TensorBinding bind_tensor(const GgufTensorInfo& tensor) {
  return Qwen35TensorBinding{
    tensor.name,
    tensor.shape,
    tensor.type,
    tensor.offset,
    tensor.absolute_offset,
    tensor.byte_size,
  };
}

Status bind_required(const TensorIndex& index, const std::string& name,
                     const std::vector<std::uint64_t>& expected_shape,
                     Qwen35TensorBinding& output) {
  const auto it = index.find(name);
  if (it == index.end()) {
    return Status::invalid_argument("missing Qwen3.5 GGUF tensor: " + name);
  }
  const auto& tensor = *it->second;
  if (tensor.shape != expected_shape) {
    std::ostringstream message;
    message << "Qwen3.5 GGUF tensor " << name << " has shape "
            << shape_to_string(tensor.shape) << ", expected "
            << shape_to_string(expected_shape);
    return Status::invalid_argument(message.str());
  }
  output = bind_tensor(tensor);
  return Status::ok();
}

Status bind_optional(const TensorIndex& index, const std::string& name,
                     const std::vector<std::uint64_t>& expected_shape, bool& present,
                     Qwen35TensorBinding& output) {
  const auto it = index.find(name);
  if (it == index.end()) {
    present = false;
    return Status::ok();
  }
  const auto& tensor = *it->second;
  if (tensor.shape != expected_shape) {
    std::ostringstream message;
    message << "Qwen3.5 GGUF tensor " << name << " has shape "
            << shape_to_string(tensor.shape) << ", expected "
            << shape_to_string(expected_shape);
    return Status::invalid_argument(message.str());
  }
  present = true;
  output = bind_tensor(tensor);
  return Status::ok();
}

Status require_qwen35_arch(const ModelConfig& config) {
  if (!config.gguf) {
    return Status::invalid_argument("Qwen3.5 weight map requires a GGUF model");
  }
  if (config.architecture != "qwen35") {
    return Status::invalid_argument("Qwen3.5 dense weight map does not support architecture: " +
                                    config.architecture);
  }
  return Status::ok();
}

Status bind_common_layer(const TensorIndex& index, const ModelConfig& config,
                         const std::uint64_t hidden, const std::uint64_t intermediate,
                         Qwen35LayerBindings& layer) {
  auto status = bind_required(index, tensor_name(layer.index, "attn_norm.weight"), {hidden},
                              layer.attn_norm);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer.index, "post_attention_norm.weight"), {hidden},
                         layer.attn_post_norm);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer.index, "ffn_gate.weight"),
                         {hidden, intermediate}, layer.ffn_gate);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer.index, "ffn_down.weight"),
                         {intermediate, hidden}, layer.ffn_down);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer.index, "ffn_up.weight"),
                         {hidden, intermediate}, layer.ffn_up);
  if (!status.is_ok()) {
    return status;
  }

  (void)config;
  return Status::ok();
}

Status bind_full_attention(const TensorIndex& index, std::int64_t layer_index,
                           const std::uint64_t hidden, const std::uint64_t head_dim,
                           const std::uint64_t attention_heads,
                           const std::uint64_t kv_heads,
                           Qwen35FullAttentionBindings& attention) {
  const auto q_dim = head_dim * attention_heads * 2U;
  const auto kv_dim = head_dim * kv_heads;
  auto status = bind_required(index, tensor_name(layer_index, "attn_q.weight"),
                              {hidden, q_dim}, attention.q);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "attn_k.weight"), {hidden, kv_dim},
                         attention.k);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "attn_v.weight"), {hidden, kv_dim},
                         attention.v);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "attn_output.weight"),
                         {head_dim * attention_heads, hidden}, attention.output);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "attn_q_norm.weight"), {head_dim},
                         attention.q_norm);
  if (!status.is_ok()) {
    return status;
  }
  return bind_required(index, tensor_name(layer_index, "attn_k_norm.weight"), {head_dim},
                       attention.k_norm);
}

Status bind_linear_attention(const TensorIndex& index, const ModelConfig& config,
                             std::int64_t layer_index, const std::uint64_t hidden,
                             Qwen35LinearAttentionBindings& attention) {
  auto key_head_dim = checked_dim(config.linear_key_head_dim, "ssm.state_size");
  if (!key_head_dim.is_ok()) {
    return key_head_dim.status();
  }
  auto key_heads = checked_dim(config.linear_num_key_heads, "ssm.group_count");
  if (!key_heads.is_ok()) {
    return key_heads.status();
  }
  auto value_heads = checked_dim(config.linear_num_value_heads, "ssm.time_step_rank");
  if (!value_heads.is_ok()) {
    return value_heads.status();
  }
  auto inner = checked_dim(config.linear_inner_size, "ssm.inner_size");
  if (!inner.is_ok()) {
    return inner.status();
  }
  auto conv_kernel = checked_dim(config.linear_conv_kernel_dim, "ssm.conv_kernel");
  if (!conv_kernel.is_ok()) {
    return conv_kernel.status();
  }
  if (inner.value() % value_heads.value() != 0) {
    return Status::invalid_argument("ssm.inner_size must be divisible by ssm.time_step_rank");
  }

  const auto key_dim = key_head_dim.value() * key_heads.value();
  const auto value_head_dim = inner.value() / value_heads.value();
  const auto value_dim = value_head_dim * value_heads.value();
  const auto conv_dim = 2U * key_dim + value_dim;

  auto status = bind_required(index, tensor_name(layer_index, "attn_qkv.weight"),
                              {hidden, conv_dim}, attention.qkv);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "attn_gate.weight"),
                         {hidden, value_dim}, attention.gate);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "ssm_conv1d.weight"),
                         {conv_kernel.value(), conv_dim}, attention.conv1d);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "ssm_dt.bias"),
                         {value_heads.value()}, attention.dt_bias);
  if (!status.is_ok()) {
    return status;
  }
  status =
    bind_required(index, tensor_name(layer_index, "ssm_a"), {value_heads.value()}, attention.a);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "ssm_beta.weight"),
                         {hidden, value_heads.value()}, attention.beta);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "ssm_alpha.weight"),
                         {hidden, value_heads.value()}, attention.alpha);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "ssm_norm.weight"),
                         {value_head_dim}, attention.norm);
  if (!status.is_ok()) {
    return status;
  }
  return bind_required(index, tensor_name(layer_index, "ssm_out.weight"), {value_dim, hidden},
                       attention.output);
}

Status bind_mtp(const TensorIndex& index, const ModelConfig& config, std::int64_t layer_index,
                const std::uint64_t hidden, const std::uint64_t vocab,
                const std::uint64_t head_dim, const std::uint64_t attention_heads,
                const std::uint64_t kv_heads, Qwen35MtpBindings& mtp) {
  auto status = bind_full_attention(index, layer_index, hidden, head_dim, attention_heads,
                                    kv_heads, mtp.attention);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, tensor_name(layer_index, "nextn.eh_proj.weight"),
                         {2U * hidden, hidden}, mtp.eh_proj);
  if (!status.is_ok()) {
    return status;
  }
  status =
    bind_required(index, tensor_name(layer_index, "nextn.enorm.weight"), {hidden}, mtp.enorm);
  if (!status.is_ok()) {
    return status;
  }
  status =
    bind_required(index, tensor_name(layer_index, "nextn.hnorm.weight"), {hidden}, mtp.hnorm);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_optional(index, tensor_name(layer_index, "nextn.embed_tokens.weight"),
                         {hidden, vocab}, mtp.has_embed_tokens, mtp.embed_tokens);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_optional(index, tensor_name(layer_index, "nextn.shared_head_head.weight"),
                         {hidden, vocab}, mtp.has_shared_head_head, mtp.shared_head_head);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_optional(index, tensor_name(layer_index, "nextn.shared_head_norm.weight"),
                         {hidden}, mtp.has_shared_head_norm, mtp.shared_head_norm);
  if (!status.is_ok()) {
    return status;
  }
  (void)config;
  return Status::ok();
}

}  // namespace

const char* qwen35_layer_kind_name(Qwen35LayerKind kind) {
  switch (kind) {
    case Qwen35LayerKind::full_attention:
      return "full_attention";
    case Qwen35LayerKind::linear_attention:
      return "linear_attention";
    case Qwen35LayerKind::mtp:
      return "mtp";
  }
  return "unknown";
}

Result<Qwen35WeightMap> build_qwen35_weight_map(const ModelConfig& config,
                                                const GgufFile& gguf) {
  auto arch_status = require_qwen35_arch(config);
  if (!arch_status.is_ok()) {
    return arch_status;
  }

  auto hidden = checked_dim(config.hidden_size, "embedding_length");
  if (!hidden.is_ok()) {
    return hidden.status();
  }
  auto vocab = checked_dim(config.vocab_size, "vocab_size");
  if (!vocab.is_ok()) {
    return vocab.status();
  }
  auto intermediate = checked_dim(config.intermediate_size, "feed_forward_length");
  if (!intermediate.is_ok()) {
    return intermediate.status();
  }
  auto head_dim = checked_dim(config.head_dim, "attention head dim");
  if (!head_dim.is_ok()) {
    return head_dim.status();
  }
  auto attention_heads = checked_dim(config.num_attention_heads, "attention.head_count");
  if (!attention_heads.is_ok()) {
    return attention_heads.status();
  }
  auto kv_heads = checked_dim(config.num_key_value_heads, "attention.head_count_kv");
  if (!kv_heads.is_ok()) {
    return kv_heads.status();
  }
  auto total_layers = checked_dim(config.total_layer_count, "block_count");
  if (!total_layers.is_ok()) {
    return total_layers.status();
  }
  auto main_layers = checked_dim(config.main_layer_count, "main layer count");
  if (!main_layers.is_ok()) {
    return main_layers.status();
  }
  if (main_layers.value() > total_layers.value()) {
    return Status::invalid_argument("main layer count exceeds total layer count");
  }
  if (config.attention_recurrent_layers.size() < main_layers.value()) {
    return Status::invalid_argument("attention_recurrent_layers shorter than main layers");
  }

  const auto index = make_tensor_index(gguf);
  Qwen35WeightMap map;
  auto status =
    bind_required(index, "token_embd.weight", {hidden.value(), vocab.value()},
                  map.token_embedding);
  if (!status.is_ok()) {
    return status;
  }
  status = bind_required(index, "output_norm.weight", {hidden.value()}, map.output_norm);
  if (!status.is_ok()) {
    return status;
  }
  const auto output_it = index.find("output.weight");
  if (output_it == index.end()) {
    map.output_tied_to_token_embedding = true;
    map.output = map.token_embedding;
  } else {
    if (output_it->second->shape != std::vector<std::uint64_t>{hidden.value(), vocab.value()}) {
      std::ostringstream message;
      message << "Qwen3.5 GGUF tensor output.weight has shape "
              << shape_to_string(output_it->second->shape) << ", expected "
              << shape_to_string({hidden.value(), vocab.value()});
      return Status::invalid_argument(message.str());
    }
    map.output = bind_tensor(*output_it->second);
  }

  map.layers.reserve(static_cast<std::size_t>(total_layers.value()));
  for (std::uint64_t i = 0; i < total_layers.value(); ++i) {
    if (i > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return Status::invalid_argument("Qwen3.5 layer index overflows int64");
    }
    Qwen35LayerBindings layer;
    layer.index = static_cast<std::int64_t>(i);
    if (i >= main_layers.value()) {
      layer.kind = Qwen35LayerKind::mtp;
    } else if (config.attention_recurrent_layers[static_cast<std::size_t>(i)] != 0) {
      layer.kind = Qwen35LayerKind::linear_attention;
    } else {
      layer.kind = Qwen35LayerKind::full_attention;
    }

    status = bind_common_layer(index, config, hidden.value(), intermediate.value(), layer);
    if (!status.is_ok()) {
      return status;
    }

    switch (layer.kind) {
      case Qwen35LayerKind::full_attention:
        status = bind_full_attention(index, layer.index, hidden.value(), head_dim.value(),
                                     attention_heads.value(), kv_heads.value(),
                                     layer.full_attention);
        if (!status.is_ok()) {
          return status;
        }
        ++map.full_attention_layers;
        break;
      case Qwen35LayerKind::linear_attention:
        status = bind_linear_attention(index, config, layer.index, hidden.value(),
                                       layer.linear_attention);
        if (!status.is_ok()) {
          return status;
        }
        ++map.linear_attention_layers;
        break;
      case Qwen35LayerKind::mtp:
        status = bind_mtp(index, config, layer.index, hidden.value(), vocab.value(),
                          head_dim.value(), attention_heads.value(), kv_heads.value(),
                          layer.mtp);
        if (!status.is_ok()) {
          return status;
        }
        ++map.mtp_layers;
        break;
    }
    map.layers.push_back(std::move(layer));
  }

  return map;
}

std::string format_qwen35_weight_map_summary(const Qwen35WeightMap& map) {
  std::ostringstream output;
  output << "Native Qwen3.5 weight map: ok\n";
  output << "Mapped layers: " << map.layers.size() << '\n';
  output << "- full_attention: " << map.full_attention_layers << '\n';
  output << "- linear_attention: " << map.linear_attention_layers << '\n';
  output << "- mtp: " << map.mtp_layers << '\n';
  output << "Output weight: "
         << (map.output_tied_to_token_embedding ? "tied token_embd.weight" : map.output.name)
         << '\n';
  return output.str();
}

}  // namespace toyllm
