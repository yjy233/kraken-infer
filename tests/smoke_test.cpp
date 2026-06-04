#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/core/tensor.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/cpu_inference.hpp"
#include "toyllm/runtime/runtime.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

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
  auto invalid_header_dir = create_tiny_model_dir("toyllm-invalid-header-smoke");
  write_fake_safetensors(invalid_header_dir / "model.safetensors", R"({})", 0);
  const auto invalid_header = toyllm::format_weight_summary(invalid_header_dir);
  assert(!invalid_header.is_ok());
  assert(invalid_header.status().message().find("header size") != std::string::npos);

  auto missing_tensor_dir = create_tiny_model_dir("toyllm-missing-tensor-smoke");
  write_fake_safetensors(
    missing_tensor_dir / "model.safetensors",
    R"({"model.embed_tokens.weight":{"dtype":"BF16","shape":[4,2],"data_offsets":[0,16]}})",
    16);
  const auto missing_tensor = toyllm::format_weight_summary(missing_tensor_dir);
  assert(!missing_tensor.is_ok());
  assert(missing_tensor.status().message().find("missing tensor") != std::string::npos);

  auto shape_mismatch_dir = create_tiny_model_dir("toyllm-shape-mismatch-smoke");
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
  test_qwen3_model_config();
  test_cpu_generation_entrypoint();
  test_weight_summary();
  test_weight_summary_regressions();

  std::cout << "smoke tests passed\n";
  return 0;
}
