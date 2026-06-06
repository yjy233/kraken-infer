#include "toyllm/backends/mps/mps_backend.hpp"
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
         info.selected_device.kind == toyllm::DeviceKind::mps);
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
  test_mps_matvec_workspace_reuse();
  test_mps_full_forward_operators();
  test_qwen3_model_config();
  test_cpu_generation_entrypoint();
  test_weight_summary();
  test_weight_summary_regressions();

  std::cout << "smoke tests passed\n";
  return 0;
}
