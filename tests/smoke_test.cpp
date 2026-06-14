#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_kv_cache.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_weight_store.hpp"
#include "toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp"
#include "toyllm/core/tensor.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/cpu_inference.hpp"
#include "toyllm/runtime/runtime.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void test_tensor_metadata() {
  const toyllm::Tensor tensor{
    toyllm::TensorDesc{toyllm::DType::f16, toyllm::Shape{2, 3, 4}, toyllm::Device::mps()}};

  assert(tensor.numel() == 24);
  assert(tensor.byte_size() == 48);
  assert(tensor.device() == toyllm::Device::mps());
  assert(toyllm::dtype_to_string(tensor.dtype()) == "f16");
}

void test_invalid_shape() {
  bool threw = false;
  try {
    (void)toyllm::Tensor{
      toyllm::TensorDesc{toyllm::DType::f32, toyllm::Shape{1, -1}, toyllm::Device::cpu()}};
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}

void test_runtime_info() {
  const toyllm::Runtime runtime{toyllm::RuntimeConfig{}};
  const auto info = runtime.info();
  assert(info.selected_device.kind == toyllm::DeviceKind::cpu ||
         info.selected_device.kind == toyllm::DeviceKind::mps ||
         info.selected_device.kind == toyllm::DeviceKind::mpsgraph);
}

void test_mps_backend_query() {
  const auto info = toyllm::mps::query_backend();
  if (info.available) {
    assert(!info.device_name.empty());
  }
}

void test_mps_operator_smoke() {
  const auto info = toyllm::mps::query_backend();
  const auto status = toyllm::mps::run_operator_smoke_test();
  if (info.available && info.compute_ready) {
    assert(status.is_ok());
  } else {
    assert(!status.is_ok());
  }
}

void test_mpsgraph_backend_query() {
  const auto info = toyllm::mpsgraph::query_backend();
  if (info.available) {
    assert(!info.device_name.empty());
  }
}

void test_mpsgraph_operator_smoke() {
  const auto info = toyllm::mpsgraph::query_backend();
  const auto status = toyllm::mpsgraph::run_operator_smoke_test();
  if (info.available && info.graph_ready) {
    assert(status.is_ok());
  } else {
    assert(!status.is_ok());
  }
}

void test_mpsgraph_generation_does_not_fallback() {
  toyllm::CpuGenerationRequest request;
  request.compute_device = toyllm::Device::mpsgraph();
  request.prompt = "hello";
  request.max_new_tokens = 1;

  const auto result = toyllm::generate_cpu(request);
  assert(!result.is_ok());
  assert(result.status().code() == toyllm::StatusCode::unavailable);
}

void test_mpsgraph_kv_cache_layout() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  toyllm::mpsgraph::MpsGraphKvCache cache;
  assert(cache.reset(context, 2, 3, 2, 4).is_ok());
  assert(cache.allocated());
  assert(cache.key_buffer().valid());
  assert(cache.value_buffer().valid());
  assert(cache.value_offset(0, 0) == 0);
  assert(cache.value_offset(0, 2) == 16);
  assert(cache.value_offset(1, 0) == 24);

  auto stats = cache.stats();
  assert(stats.allocated);
  assert(stats.layers == 2);
  assert(stats.capacity_tokens == 3);
  assert(stats.kv_heads == 2);
  assert(stats.head_dim == 4);
  assert(stats.kv_dim == 8);
  assert(stats.key_bytes == 192);
  assert(stats.value_bytes == 192);
  assert(stats.total_bytes == 384);
  assert(stats.used_tokens == 0);

  assert(cache.mark_position_used(2).is_ok());
  stats = cache.stats();
  assert(stats.used_tokens == 3);
  assert(!cache.mark_position_used(3).is_ok());
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16U);
}

toyllm::mps::MpsBuffer make_f32_buffer(toyllm::mps::MpsContext& context,
                                       const std::vector<float>& values) {
  auto buffer_result = context.make_buffer(values.size() * sizeof(float));
  assert(buffer_result.is_ok());
  auto buffer = std::move(buffer_result.value());
  const auto status = context.copy_to_buffer(buffer, values.data(), values.size() * sizeof(float));
  assert(status.is_ok());
  return buffer;
}

toyllm::mps::MpsBuffer make_bf16_buffer(toyllm::mps::MpsContext& context,
                                        const std::vector<float>& values) {
  std::vector<std::uint16_t> bf16_values;
  bf16_values.reserve(values.size());
  for (const auto value : values) {
    bf16_values.push_back(float_to_bf16(value));
  }
  auto buffer_result = context.make_buffer(bf16_values.size() * sizeof(std::uint16_t));
  assert(buffer_result.is_ok());
  auto buffer = std::move(buffer_result.value());
  const auto status =
    context.copy_to_buffer(buffer, bf16_values.data(),
                           bf16_values.size() * sizeof(std::uint16_t));
  assert(status.is_ok());
  return buffer;
}

std::vector<float> read_f32_buffer(const toyllm::mps::MpsContext& context,
                                   const toyllm::mps::MpsBuffer& buffer,
                                   std::size_t values) {
  std::vector<float> output(values);
  const auto status = context.copy_from_buffer(buffer, output.data(), values * sizeof(float));
  assert(status.is_ok());
  return output;
}

void assert_close(float actual, float expected) {
  assert(std::abs(actual - expected) < 1e-5F);
}

toyllm::mpsgraph::MpsGraphBuffer make_mpsgraph_f32_buffer(
  const toyllm::mpsgraph::MpsGraphContext& context,
  const std::vector<float>& values) {
  auto buffer_result = context.make_buffer(values.size() * sizeof(float));
  assert(buffer_result.is_ok());
  auto buffer = std::move(buffer_result.value());
  const auto status = context.copy_to_buffer(buffer, values.data(),
                                            values.size() * sizeof(float));
  assert(status.is_ok());
  return buffer;
}

std::vector<float> read_mpsgraph_f32_buffer(
  const toyllm::mpsgraph::MpsGraphContext& context,
  const toyllm::mpsgraph::MpsGraphBuffer& buffer,
  std::size_t values) {
  std::vector<float> output(values);
  const auto status = context.copy_from_buffer(buffer, output.data(),
                                              values * sizeof(float));
  assert(status.is_ok());
  return output;
}

void test_mpsgraph_qk_norm_rope_and_attention_ops() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  auto qk_input = make_mpsgraph_f32_buffer(context, {3.0F, 4.0F, 0.0F, 2.0F});
  auto qk_weight = make_mpsgraph_f32_buffer(context, {1.0F, 2.0F});
  auto qk_output_result = context.make_buffer(4U * sizeof(float));
  assert(qk_output_result.is_ok());
  auto qk_output = std::move(qk_output_result.value());
  assert(context.qk_norm_f32(qk_input, qk_weight, 2, 2, 0.0F, qk_output).is_ok());
  auto output = read_mpsgraph_f32_buffer(context, qk_output, 4);
  assert(std::abs(output[0] - 3.0F / std::sqrt(12.5F)) < 1e-4F);
  assert(std::abs(output[1] - 8.0F / std::sqrt(12.5F)) < 1e-4F);
  assert(std::abs(output[2]) < 1e-4F);
  assert(std::abs(output[3] - 4.0F / std::sqrt(2.0F)) < 1e-4F);

  auto rope_input = make_mpsgraph_f32_buffer(context, {1.0F, 2.0F});
  auto rope_output_result = context.make_buffer(2U * sizeof(float));
  assert(rope_output_result.is_ok());
  auto rope_output = std::move(rope_output_result.value());
  assert(context.rope_f32(rope_input, 1, 2, 1, 10000.0F, rope_output).is_ok());
  output = read_mpsgraph_f32_buffer(context, rope_output, 2);
  assert(std::abs(output[0] - (std::cos(1.0F) - 2.0F * std::sin(1.0F))) < 1e-4F);
  assert(std::abs(output[1] - (2.0F * std::cos(1.0F) + std::sin(1.0F))) < 1e-4F);

  auto query = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F});
  auto key_cache = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F, 0.0F, 1.0F});
  auto value_cache = make_mpsgraph_f32_buffer(context, {5.0F, 6.0F, 7.0F, 8.0F});
  auto attention_output_result = context.make_buffer(2U * sizeof(float));
  assert(attention_output_result.is_ok());
  auto attention_output = std::move(attention_output_result.value());
  assert(context.attention_f32(query, key_cache, value_cache, 0, 1, 2, 1, 1, 2,
                               attention_output)
           .is_ok());
  output = read_mpsgraph_f32_buffer(context, attention_output, 2);
  const auto score0 = std::exp(1.0F / std::sqrt(2.0F));
  const auto weight0 = score0 / (score0 + 1.0F);
  const auto weight1 = 1.0F - weight0;
  assert(std::abs(output[0] - (weight0 * 5.0F + weight1 * 7.0F)) < 1e-4F);
  assert(std::abs(output[1] - (weight0 * 6.0F + weight1 * 8.0F)) < 1e-4F);
}

void test_mps_matvec_workspace_reuse() {
  auto context_result = toyllm::mps::MpsContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  const std::uint16_t weight[] = {
    float_to_bf16(1.0F),
    float_to_bf16(2.0F),
    float_to_bf16(3.0F),
    float_to_bf16(1.0F),
  };
  auto weight_buffer_result = context.make_buffer(sizeof(weight));
  assert(weight_buffer_result.is_ok());
  auto weight_buffer = std::move(weight_buffer_result.value());
  const auto copy_status = context.copy_to_buffer(weight_buffer, weight, sizeof(weight));
  assert(copy_status.is_ok());

  auto workspace_result = context.make_matvec_workspace(2, 2);
  assert(workspace_result.is_ok());
  auto workspace = std::move(workspace_result.value());
  assert(workspace.valid());
  assert(workspace.rows() == 2);
  assert(workspace.cols() == 2);

  const auto first = context.matvec_bf16_f32(weight_buffer, workspace, {3.0F, 4.0F});
  assert(first.is_ok());
  assert(first.value().size() == 2);
  assert(std::abs(first.value()[0] - 11.0F) < 1e-5F);
  assert(std::abs(first.value()[1] - 13.0F) < 1e-5F);

  const auto second = context.matvec_bf16_f32(weight_buffer, workspace, {1.0F, 1.0F});
  assert(second.is_ok());
  assert(second.value().size() == 2);
  assert(std::abs(second.value()[0] - 3.0F) < 1e-5F);
  assert(std::abs(second.value()[1] - 4.0F) < 1e-5F);
}

void test_mps_full_forward_operators() {
  auto context_result = toyllm::mps::MpsContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  auto embedding_weight = make_bf16_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F});
  auto embedding_output = make_f32_buffer(context, {0.0F, 0.0F});
  assert(context.embedding_bf16_f32(embedding_weight, 1, 2, embedding_output).is_ok());
  auto output = read_f32_buffer(context, embedding_output, 2);
  assert_close(output[0], 3.0F);
  assert_close(output[1], 4.0F);

  auto norm_input = make_f32_buffer(context, {1.0F, 1.0F});
  auto norm_weight = make_bf16_buffer(context, {2.0F, 3.0F});
  auto norm_output = make_f32_buffer(context, {0.0F, 0.0F});
  assert(context.rms_norm_f32_bf16(norm_input, norm_weight, 2, 0.0F, norm_output).is_ok());
  output = read_f32_buffer(context, norm_output, 2);
  assert_close(output[0], 2.0F);
  assert_close(output[1], 3.0F);

  auto q_values = make_f32_buffer(context, {1.0F, 1.0F});
  assert(context.qk_norm_f32_bf16(q_values, norm_weight, 1, 2, 0.0F).is_ok());
  assert(context.rope_f32(q_values, 1, 2, 0, 10000.0F).is_ok());
  output = read_f32_buffer(context, q_values, 2);
  assert_close(output[0], 2.0F);
  assert_close(output[1], 3.0F);

  auto target = make_f32_buffer(context, {1.0F, 2.0F});
  auto delta = make_f32_buffer(context, {3.0F, 4.0F});
  assert(context.add_f32_in_place(target, delta, 2).is_ok());
  output = read_f32_buffer(context, target, 2);
  assert_close(output[0], 4.0F);
  assert_close(output[1], 6.0F);

  auto gate = make_f32_buffer(context, {0.0F, 0.0F});
  auto up = make_f32_buffer(context, {5.0F, 6.0F});
  assert(context.silu_mul_f32_in_place(gate, up, 2).is_ok());
  output = read_f32_buffer(context, gate, 2);
  assert_close(output[0], 0.0F);
  assert_close(output[1], 0.0F);

  auto destination = make_f32_buffer(context, {0.0F, 0.0F, 0.0F});
  assert(context.copy_f32_region(delta, destination, 0, 1, 2).is_ok());
  output = read_f32_buffer(context, destination, 3);
  assert_close(output[0], 0.0F);
  assert_close(output[1], 3.0F);
  assert_close(output[2], 4.0F);

  auto query = make_f32_buffer(context, {1.0F, 0.0F});
  auto key_cache = make_f32_buffer(context, {1.0F, 0.0F});
  auto value_cache = make_f32_buffer(context, {5.0F, 6.0F});
  auto attention_output = make_f32_buffer(context, {0.0F, 0.0F});
  assert(context.attention_f32(query, key_cache, value_cache, 0, 0, 1, 1, 1, 2,
                               attention_output)
           .is_ok());
  output = read_f32_buffer(context, attention_output, 2);
  assert_close(output[0], 5.0F);
  assert_close(output[1], 6.0F);
}

void test_profile_artifacts() {
  const auto request_id = toyllm::make_gateway_request_id();
  assert(request_id.rfind("req-", 0) == 0);
  assert(toyllm::parse_profile_mode("summary").has_value());

  const auto temp_dir = std::filesystem::temp_directory_path() / "kraken-infer-profile-smoke";
  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);

  toyllm::ObservabilityConfig observability;
  observability.profile_mode = toyllm::ProfileMode::summary;
  observability.profile_output_dir = temp_dir;
  observability.request_id = request_id;
  toyllm::RequestProfiler profiler(observability);
  profiler.set_metadata("device", "cpu");
  profiler.set_metadata("model_dir", "models/qwen3-0.6b");
  profiler.set_metadata("prompt_tokens", static_cast<std::size_t>(1));
  profiler.set_metadata("generated_tokens", static_cast<std::size_t>(1));
  {
    auto request_span = profiler.scoped("request.total");
    {
      auto prefill_span = profiler.scoped("request.prefill");
      (void)prefill_span;
    }
    (void)request_span;
  }
  const auto artifacts = profiler.write_artifacts();
  assert(!artifacts.output_dir.empty());
  assert(std::filesystem::exists(artifacts.output_dir / "summary.txt"));
  assert(std::filesystem::exists(artifacts.output_dir / "summary.json"));
  assert(std::filesystem::exists(artifacts.output_dir / "manifest.json"));
  std::filesystem::remove_all(temp_dir, ec);
}

void test_qwen3_model_config() {
  const std::filesystem::path model_dir{"models/qwen3-0.6b"};
  if (!std::filesystem::exists(model_dir)) {
    return;
  }

  auto bundle = toyllm::load_model_bundle(model_dir);
  assert(bundle.is_ok());
  assert(bundle.value().model.architecture == "Qwen3ForCausalLM");
  assert(bundle.value().model.model_type == "qwen3");
  assert(bundle.value().model.num_hidden_layers == 28);
  assert(bundle.value().model.hidden_size == 1024);
  assert(bundle.value().model.num_attention_heads == 16);
  assert(bundle.value().model.num_key_value_heads == 8);
  assert(bundle.value().model.head_dim == 128);
  assert(bundle.value().generation.top_p > 0.94);
  assert(bundle.value().generation.top_p < 0.96);
  assert(bundle.value().tokenizer.total_vocab_size <=
         static_cast<std::uint64_t>(bundle.value().model.vocab_size));
  assert(bundle.value().tokenizer.max_token_id <
         static_cast<std::uint64_t>(bundle.value().model.vocab_size));
  assert(toyllm::format_model_summary(bundle.value()).find("Validation: ok") != std::string::npos);
}

void test_cpu_generation_entrypoint() {
  const std::filesystem::path model_dir{"models/qwen3-0.6b"};
  if (!std::filesystem::exists(model_dir)) {
    return;
  }

  toyllm::CpuGenerationRequest request;
  request.model_dir = model_dir;
  request.max_new_tokens = 1;
  const auto result = toyllm::generate_cpu(request);
  assert(!result.is_ok());
  assert(result.status().message().find("prompt") != std::string::npos);
}

void test_weight_summary() {
  const std::filesystem::path model_dir{"models/qwen3-0.6b"};
  if (!std::filesystem::exists(model_dir / "model.safetensors")) {
    return;
  }

  const auto summary = toyllm::format_weight_summary(model_dir);
  assert(summary.is_ok());
  assert(summary.value().find("Tensor count: 311") != std::string::npos);
  assert(summary.value().find("Qwen3 mapping: ok") != std::string::npos);
  assert(summary.value().find("Validation: ok") != std::string::npos);
}

void write_text_file(const std::filesystem::path& path, const std::string& content) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to create " + path.string());
  }
  output << content;
}

void write_fake_safetensors(const std::filesystem::path& path, const std::string& header,
                            std::size_t data_bytes) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create " + path.string());
  }
  const auto header_size = static_cast<std::uint64_t>(header.size());
  for (int index = 0; index < 8; ++index) {
    const auto byte = static_cast<char>((header_size >> (8U * static_cast<unsigned int>(index))) &
                                        0xFFU);
    output.put(byte);
  }
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (std::size_t index = 0; index < data_bytes; ++index) {
    output.put('\0');
  }
}

struct TinyTensorSpec {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<float> values;
};

void write_bf16_safetensors(const std::filesystem::path& path,
                            const std::vector<TinyTensorSpec>& tensors) {
  std::ostringstream header;
  std::vector<std::uint16_t> payload;
  std::uint64_t byte_offset = 0;

  header << "{";
  for (std::size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
    const auto& tensor = tensors[tensor_index];
    if (tensor_index != 0) {
      header << ",";
    }
    std::uint64_t elements = 1;
    for (const auto dim : tensor.shape) {
      elements *= dim;
    }
    assert(elements == static_cast<std::uint64_t>(tensor.values.size()));
    const auto begin = byte_offset;
    const auto bytes = elements * sizeof(std::uint16_t);
    const auto end = begin + bytes;
    header << "\"" << tensor.name << "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (std::size_t dim_index = 0; dim_index < tensor.shape.size(); ++dim_index) {
      if (dim_index != 0) {
        header << ",";
      }
      header << tensor.shape[dim_index];
    }
    header << "],\"data_offsets\":[" << begin << "," << end << "]}";
    for (const auto value : tensor.values) {
      payload.push_back(float_to_bf16(value));
    }
    byte_offset = end;
  }
  header << "}";

  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create " + path.string());
  }
  const auto header_text = header.str();
  const auto header_size = static_cast<std::uint64_t>(header_text.size());
  for (int index = 0; index < 8; ++index) {
    const auto byte = static_cast<char>((header_size >> (8U * static_cast<unsigned int>(index))) &
                                        0xFFU);
    output.put(byte);
  }
  output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
  output.write(reinterpret_cast<const char*>(payload.data()),
               static_cast<std::streamsize>(payload.size() * sizeof(std::uint16_t)));
}

std::filesystem::path create_tiny_model_dir(std::string_view name) {
  const auto model_dir = std::filesystem::temp_directory_path() / std::filesystem::path{name};
  std::filesystem::remove_all(model_dir);
  std::filesystem::create_directories(model_dir);

  write_text_file(
    model_dir / "config.json",
    R"({
      "architectures": ["Qwen3ForCausalLM"],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "bos_token_id": 0,
      "eos_token_id": 1,
      "head_dim": 1,
      "hidden_act": "silu",
      "hidden_size": 2,
      "initializer_range": 0.02,
      "intermediate_size": 3,
      "max_position_embeddings": 16,
      "max_window_layers": 1,
      "model_type": "qwen3",
      "num_attention_heads": 2,
      "num_hidden_layers": 1,
      "num_key_value_heads": 1,
      "rms_norm_eps": 1e-6,
      "rope_theta": 10000,
      "tie_word_embeddings": true,
      "torch_dtype": "bfloat16",
      "transformers_version": "test",
      "use_cache": true,
      "use_sliding_window": false,
      "vocab_size": 4
    })");
  write_text_file(model_dir / "generation_config.json",
                  R"({"bos_token_id":0,"do_sample":false,"eos_token_id":[1],"pad_token_id":0,"temperature":1.0,"top_k":0,"top_p":1.0})");
  write_text_file(model_dir / "tokenizer.json",
                  R"({"model":{"vocab":{"a":0,"b":1,"c":2,"d":3}},"added_tokens":[]})");
  write_text_file(model_dir / "tokenizer_config.json", R"({"added_tokens_decoder":{}})");
  write_text_file(model_dir / "vocab.json", R"({"a":0,"b":1,"c":2,"d":3})");
  return model_dir;
}

void write_tiny_qwen_mpsgraph_safetensors(const std::filesystem::path& path) {
  write_bf16_safetensors(
    path,
    {
      {"model.embed_tokens.weight", {4, 2}, {1.0F, 2.0F, 3.0F, 4.0F,
                                             5.0F, 6.0F, 7.0F, 8.0F}},
      {"lm_head.weight", {4, 2}, {1.0F, 1.0F, 1.0F, 1.0F,
                                  1.0F, 1.0F, 1.0F, 1.0F}},
      {"model.norm.weight", {2}, {1.0F, 1.0F}},
      {"model.layers.0.input_layernorm.weight", {2}, {1.0F, 1.0F}},
      {"model.layers.0.post_attention_layernorm.weight", {2}, {1.0F, 1.0F}},
      {"model.layers.0.self_attn.q_proj.weight", {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
      {"model.layers.0.self_attn.k_proj.weight", {1, 2}, {1.0F, 0.0F}},
      {"model.layers.0.self_attn.v_proj.weight", {1, 2}, {0.0F, 1.0F}},
      {"model.layers.0.self_attn.o_proj.weight", {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
      {"model.layers.0.self_attn.q_norm.weight", {1}, {1.0F}},
      {"model.layers.0.self_attn.k_norm.weight", {1}, {1.0F}},
      {"model.layers.0.mlp.gate_proj.weight", {3, 2}, {1.0F, 0.0F, 0.0F,
                                                       1.0F, 1.0F, 1.0F}},
      {"model.layers.0.mlp.up_proj.weight", {3, 2}, {1.0F, 1.0F, 1.0F,
                                                     1.0F, 1.0F, 1.0F}},
      {"model.layers.0.mlp.down_proj.weight", {2, 3}, {1.0F, 0.0F, 0.0F,
                                                       0.0F, 1.0F, 0.0F}},
    });
}

void test_mpsgraph_weight_store_metadata_and_upload() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  auto model_dir = create_tiny_model_dir("kraken-infer-mpsgraph-weight-store-smoke");
  write_tiny_qwen_mpsgraph_safetensors(model_dir / "model.safetensors");

  const auto bundle = toyllm::load_model_bundle(model_dir);
  assert(bundle.is_ok());
  auto store_result =
    toyllm::mpsgraph::MpsGraphWeightStore::load_metadata(model_dir / "model.safetensors");
  assert(store_result.is_ok());
  auto store = std::move(store_result.value());
  assert(store.tensors().size() == 14);
  assert(store.contains("model.embed_tokens.weight"));
  assert(store.validate_qwen3_shapes(bundle.value().model).is_ok());

  auto device_tensor = store.upload_tensor_f32(context, "model.embed_tokens.weight");
  assert(device_tensor.is_ok());
  assert(device_tensor.value().shape == std::vector<std::uint64_t>({4, 2}));
  assert(device_tensor.value().elements == 8);

  std::vector<float> output(8);
  const auto read_status =
    context.copy_from_buffer(device_tensor.value().buffer, output.data(),
                             output.size() * sizeof(float));
  assert(read_status.is_ok());
  for (std::size_t i = 0; i < output.size(); ++i) {
    assert_close(output[i], static_cast<float>(i + 1U));
  }

  std::error_code ec;
  std::filesystem::remove_all(model_dir, ec);
}

void test_qwen_mpsgraph_model_core_weight_load() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  auto model_dir = create_tiny_model_dir("kraken-infer-qwen-mpsgraph-model-smoke");
  write_tiny_qwen_mpsgraph_safetensors(model_dir / "model.safetensors");

  auto model = toyllm::mpsgraph::QwenMpsGraphModel::load_core_weights(model_dir, context);
  assert(model.is_ok());
  assert(model.value().config().hidden_size == 2);
  assert(model.value().info().tensor_count == 14);
  assert(model.value().info().core_weights_uploaded);
  assert(model.value().info().device_tensor_count == 2);

  const auto hidden = model.value().debug_embed_token(context, 2);
  assert(hidden.is_ok());
  assert(hidden.value().size() == 2);
  assert_close(hidden.value()[0], 5.0F);
  assert_close(hidden.value()[1], 6.0F);

  std::error_code ec;
  std::filesystem::remove_all(model_dir, ec);
}

void test_weight_summary_regressions() {
  auto invalid_header_dir = create_tiny_model_dir("kraken-infer-invalid-header-smoke");
  write_fake_safetensors(invalid_header_dir / "model.safetensors", R"({})", 0);
  const auto invalid_header = toyllm::format_weight_summary(invalid_header_dir);
  assert(!invalid_header.is_ok());
  assert(invalid_header.status().message().find("header size") != std::string::npos);

  auto missing_tensor_dir = create_tiny_model_dir("kraken-infer-missing-tensor-smoke");
  write_fake_safetensors(
    missing_tensor_dir / "model.safetensors",
    R"({"model.embed_tokens.weight":{"dtype":"BF16","shape":[4,2],"data_offsets":[0,16]}})",
    16);
  const auto missing_tensor = toyllm::format_weight_summary(missing_tensor_dir);
  assert(!missing_tensor.is_ok());
  assert(missing_tensor.status().message().find("missing tensor") != std::string::npos);

  auto shape_mismatch_dir = create_tiny_model_dir("kraken-infer-shape-mismatch-smoke");
  write_fake_safetensors(
    shape_mismatch_dir / "model.safetensors",
    R"({"model.embed_tokens.weight":{"dtype":"BF16","shape":[3,2],"data_offsets":[0,12]}})",
    12);
  const auto shape_mismatch = toyllm::format_weight_summary(shape_mismatch_dir);
  assert(!shape_mismatch.is_ok());
  assert(shape_mismatch.status().message().find("shape mismatch") != std::string::npos);
}

}  // namespace

int main() {
  test_tensor_metadata();
  test_invalid_shape();
  test_runtime_info();
  test_mps_backend_query();
  test_mps_operator_smoke();
  test_mpsgraph_backend_query();
  test_mpsgraph_operator_smoke();
  test_mpsgraph_generation_does_not_fallback();
  test_mpsgraph_kv_cache_layout();
  test_mpsgraph_qk_norm_rope_and_attention_ops();
  test_mps_matvec_workspace_reuse();
  test_mps_full_forward_operators();
  test_profile_artifacts();
  test_qwen3_model_config();
  test_cpu_generation_entrypoint();
  test_weight_summary();
  test_weight_summary_regressions();
  test_mpsgraph_weight_store_metadata_and_upload();
  test_qwen_mpsgraph_model_core_weight_load();

  std::cout << "smoke tests passed\n";
  return 0;
}
