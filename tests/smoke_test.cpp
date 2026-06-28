#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_kv_cache.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_weight_store.hpp"
#include "toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp"
#include "toyllm/core/tensor.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/cpu_inference.hpp"
#include "toyllm/runtime/gguf_reader.hpp"
#include "toyllm/runtime/gguf_tokenizer.hpp"
#include "toyllm/runtime/qwen35_runtime.hpp"
#include "toyllm/runtime/qwen35_weight_map.hpp"
#include "toyllm/runtime/qwen_tokenizer.hpp"
#include "toyllm/runtime/runtime.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
  request.stream_token = [](std::string_view) {};

  const auto result = toyllm::generate_cpu(request);
  assert(!result.is_ok());
  assert(result.status().code() == toyllm::StatusCode::unavailable);
  assert(result.status().message().find("streaming") != std::string::npos);
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

toyllm::mps::MpsBuffer make_empty_buffer(toyllm::mps::MpsContext& context,
                                         std::size_t byte_size) {
  auto buffer_result = context.make_buffer(byte_size);
  assert(buffer_result.is_ok());
  return std::move(buffer_result.value());
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

toyllm::mps::MpsBuffer make_i32_buffer(toyllm::mps::MpsContext& context,
                                       const std::vector<std::int32_t>& values) {
  auto buffer_result = context.make_buffer(values.size() * sizeof(std::int32_t));
  assert(buffer_result.is_ok());
  auto buffer = std::move(buffer_result.value());
  const auto status =
    context.copy_to_buffer(buffer, values.data(), values.size() * sizeof(std::int32_t));
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
  if (std::abs(actual - expected) >= 1e-5F) {
    std::cerr << "assert_close failed: actual=" << actual
              << " expected=" << expected << '\n';
  }
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

void test_mpsgraph_kv_cache_store() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  toyllm::mpsgraph::MpsGraphKvCache cache;
  assert(cache.reset(context, 1, 2, 1, 2).is_ok());

  auto key = make_mpsgraph_f32_buffer(context, {1.0F, 2.0F});
  auto value = make_mpsgraph_f32_buffer(context, {3.0F, 4.0F});
  assert(cache.store(context, 0, 1, key, value).is_ok());

  auto key_values = read_mpsgraph_f32_buffer(context, cache.key_buffer(), 4);
  auto value_values = read_mpsgraph_f32_buffer(context, cache.value_buffer(), 4);
  assert_close(key_values[0], 0.0F);
  assert_close(key_values[1], 0.0F);
  assert_close(key_values[2], 1.0F);
  assert_close(key_values[3], 2.0F);
  assert_close(value_values[0], 0.0F);
  assert_close(value_values[1], 0.0F);
  assert_close(value_values[2], 3.0F);
  assert_close(value_values[3], 4.0F);
  assert(cache.stats().used_tokens == 2);
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

  auto q_norm_rope_input = make_mpsgraph_f32_buffer(context, {3.0F, 4.0F, 0.0F, 2.0F});
  auto k_norm_rope_input = make_mpsgraph_f32_buffer(context, {1.0F, 2.0F});
  auto q_norm_rope_weight = make_mpsgraph_f32_buffer(context, {1.0F, 2.0F});
  auto k_norm_rope_weight = make_mpsgraph_f32_buffer(context, {3.0F, 4.0F});
  auto q_rope_output_result = context.make_buffer(4U * sizeof(float));
  auto k_rope_output_result = context.make_buffer(2U * sizeof(float));
  assert(q_rope_output_result.is_ok());
  assert(k_rope_output_result.is_ok());
  auto q_rope_output = std::move(q_rope_output_result.value());
  auto k_rope_output = std::move(k_rope_output_result.value());
  assert(context.qk_norm_rope_f32(q_norm_rope_input, k_norm_rope_input,
                                  q_norm_rope_weight, k_norm_rope_weight, 2, 1, 2,
                                  1, 0.0F, 10000.0F, q_rope_output,
                                  k_rope_output)
           .is_ok());
  output = read_mpsgraph_f32_buffer(context, q_rope_output, 4);
  const auto q0_first = 3.0F / std::sqrt(12.5F);
  const auto q0_second = 8.0F / std::sqrt(12.5F);
  const auto q1_first = 0.0F;
  const auto q1_second = 4.0F / std::sqrt(2.0F);
  assert(std::abs(output[0] - (q0_first * std::cos(1.0F) -
                               q0_second * std::sin(1.0F))) < 1e-4F);
  assert(std::abs(output[1] - (q0_second * std::cos(1.0F) +
                               q0_first * std::sin(1.0F))) < 1e-4F);
  assert(std::abs(output[2] - (q1_first * std::cos(1.0F) -
                               q1_second * std::sin(1.0F))) < 1e-4F);
  assert(std::abs(output[3] - (q1_second * std::cos(1.0F) +
                               q1_first * std::sin(1.0F))) < 1e-4F);
  output = read_mpsgraph_f32_buffer(context, k_rope_output, 2);
  const auto k_first = 3.0F / std::sqrt(2.5F);
  const auto k_second = 8.0F / std::sqrt(2.5F);
  assert(std::abs(output[0] - (k_first * std::cos(1.0F) -
                               k_second * std::sin(1.0F))) < 1e-4F);
  assert(std::abs(output[1] - (k_second * std::cos(1.0F) +
                               k_first * std::sin(1.0F))) < 1e-4F);

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

  auto gqa_query = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F, 0.0F, 1.0F});
  auto gqa_output_result = context.make_buffer(4U * sizeof(float));
  assert(gqa_output_result.is_ok());
  auto gqa_output = std::move(gqa_output_result.value());
  assert(context.attention_f32(gqa_query, key_cache, value_cache, 0, 1, 2, 2, 1, 2,
                               gqa_output)
           .is_ok());
  output = read_mpsgraph_f32_buffer(context, gqa_output, 4);
  assert(std::abs(output[0] - (weight0 * 5.0F + weight1 * 7.0F)) < 1e-4F);
  assert(std::abs(output[1] - (weight0 * 6.0F + weight1 * 8.0F)) < 1e-4F);
  assert(std::abs(output[2] - (weight1 * 5.0F + weight0 * 7.0F)) < 1e-4F);
  assert(std::abs(output[3] - (weight1 * 6.0F + weight0 * 8.0F)) < 1e-4F);

  auto argmax_input = make_mpsgraph_f32_buffer(context, {-1.0F, 2.0F, 5.0F, 3.0F});
  auto argmax_output_result = context.make_buffer(sizeof(std::int32_t));
  assert(argmax_output_result.is_ok());
  auto argmax_output = std::move(argmax_output_result.value());
  assert(context.argmax_i32(argmax_input, 4, argmax_output).is_ok());
  std::int32_t argmax = -1;
  assert(context.copy_from_buffer(argmax_output, &argmax, sizeof(argmax)).is_ok());
  assert(argmax == 2);

  auto embedding_weight = make_mpsgraph_f32_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F});
  auto embedding_token_result = context.make_buffer(sizeof(std::int32_t));
  assert(embedding_token_result.is_ok());
  auto embedding_token = std::move(embedding_token_result.value());
  std::int32_t token_id = 1;
  assert(context.copy_to_buffer(embedding_token, &token_id, sizeof(token_id)).is_ok());
  auto embedding_output_result = context.make_buffer(2U * sizeof(float));
  assert(embedding_output_result.is_ok());
  auto embedding_output = std::move(embedding_output_result.value());
  assert(context.embedding_from_token_f32(embedding_weight, 2, 2, embedding_token,
                                          embedding_output)
           .is_ok());
  output = read_mpsgraph_f32_buffer(context, embedding_output, 2);
  assert_close(output[0], 3.0F);
  assert_close(output[1], 4.0F);

  auto generated_result = context.make_buffer(3U * sizeof(std::int32_t));
  assert(generated_result.is_ok());
  auto generated = std::move(generated_result.value());
  std::array<std::int32_t, 3> zeros{0, 0, 0};
  assert(context.copy_to_buffer(generated, zeros.data(), zeros.size() * sizeof(std::int32_t))
           .is_ok());
  assert(context.write_i32_token(embedding_token, generated, 1, 3).is_ok());
  std::array<std::int32_t, 3> generated_values{0, 0, 0};
  assert(context.copy_from_buffer(generated, generated_values.data(),
                                  generated_values.size() * sizeof(std::int32_t))
           .is_ok());
  assert(generated_values[0] == 0);
  assert(generated_values[1] == 1);
  assert(generated_values[2] == 0);

  auto q_weight = make_mpsgraph_f32_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F});
  auto k_weight = make_mpsgraph_f32_buffer(context, {5.0F, 6.0F});
  auto v_weight = make_mpsgraph_f32_buffer(context, {7.0F, 8.0F});
  auto qkv_input = make_mpsgraph_f32_buffer(context, {2.0F, 3.0F});
  auto q_output_result = context.make_buffer(2U * sizeof(float));
  auto k_output_result = context.make_buffer(sizeof(float));
  auto v_output_result = context.make_buffer(sizeof(float));
  assert(q_output_result.is_ok());
  assert(k_output_result.is_ok());
  assert(v_output_result.is_ok());
  auto q_output = std::move(q_output_result.value());
  auto k_output = std::move(k_output_result.value());
  auto v_output = std::move(v_output_result.value());
  assert(context.qkv_matvec_f32(q_weight, k_weight, v_weight, 2, 1, 2, qkv_input,
                                q_output, k_output, v_output)
           .is_ok());
  output = read_mpsgraph_f32_buffer(context, q_output, 2);
  assert_close(output[0], 8.0F);
  assert_close(output[1], 18.0F);
  output = read_mpsgraph_f32_buffer(context, k_output, 1);
  assert_close(output[0], 28.0F);
  output = read_mpsgraph_f32_buffer(context, v_output, 1);
  assert_close(output[0], 38.0F);

  auto fused_hidden = make_mpsgraph_f32_buffer(context, {2.0F, 3.0F});
  auto fused_input_norm_weight = make_mpsgraph_f32_buffer(context, {1.0F, 1.0F});
  auto fused_q_weight = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F,
                                                           0.0F, 1.0F});
  auto fused_k_weight = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F,
                                                           0.0F, 1.0F});
  auto fused_v_weight = make_mpsgraph_f32_buffer(context, {2.0F, 0.0F,
                                                           0.0F, 3.0F});
  auto fused_q_norm_weight = make_mpsgraph_f32_buffer(context, {1.0F, 1.0F});
  auto fused_k_norm_weight = make_mpsgraph_f32_buffer(context, {1.0F, 1.0F});
  auto fused_normed_result = context.make_buffer(2U * sizeof(float));
  auto fused_q_result = context.make_buffer(2U * sizeof(float));
  auto fused_k_result = context.make_buffer(2U * sizeof(float));
  auto fused_v_result = context.make_buffer(2U * sizeof(float));
  assert(fused_normed_result.is_ok());
  assert(fused_q_result.is_ok());
  assert(fused_k_result.is_ok());
  assert(fused_v_result.is_ok());
  auto fused_normed = std::move(fused_normed_result.value());
  auto fused_q = std::move(fused_q_result.value());
  auto fused_k = std::move(fused_k_result.value());
  auto fused_v = std::move(fused_v_result.value());
  assert(context.input_norm_qkv_qk_rope_f32(
           fused_hidden, fused_input_norm_weight, fused_q_weight, fused_k_weight,
           fused_v_weight, fused_q_norm_weight, fused_k_norm_weight, 2, 1, 1,
           2, 0, 0.0F, 10000.0F, fused_normed, fused_q, fused_k, fused_v)
           .is_ok());
  output = read_mpsgraph_f32_buffer(context, fused_normed, 2);
  const auto fused_norm_scale = 1.0F / std::sqrt(6.5F);
  assert_close(output[0], 2.0F * fused_norm_scale);
  assert_close(output[1], 3.0F * fused_norm_scale);
  output = read_mpsgraph_f32_buffer(context, fused_q, 2);
  const auto fused_q_norm = 1.0F /
                            std::sqrt((4.0F + 9.0F) * fused_norm_scale *
                                       fused_norm_scale / 2.0F);
  assert_close(output[0], 2.0F * fused_norm_scale * fused_q_norm);
  assert_close(output[1], 3.0F * fused_norm_scale * fused_q_norm);
  output = read_mpsgraph_f32_buffer(context, fused_k, 2);
  assert_close(output[0], 2.0F * fused_norm_scale * fused_q_norm);
  assert_close(output[1], 3.0F * fused_norm_scale * fused_q_norm);
  output = read_mpsgraph_f32_buffer(context, fused_v, 2);
  assert_close(output[0], 4.0F * fused_norm_scale);
  assert_close(output[1], 9.0F * fused_norm_scale);

  auto gate_weight = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F, 0.0F, 1.0F});
  auto up_weight = make_mpsgraph_f32_buffer(context, {2.0F, 0.0F, 0.0F, 3.0F});
  auto gate_up_input = make_mpsgraph_f32_buffer(context, {4.0F, 5.0F});
  auto gate_output_result = context.make_buffer(2U * sizeof(float));
  auto up_output_result = context.make_buffer(2U * sizeof(float));
  assert(gate_output_result.is_ok());
  assert(up_output_result.is_ok());
  auto gate_output = std::move(gate_output_result.value());
  auto up_output = std::move(up_output_result.value());
  assert(context.gate_up_matvec_f32(gate_weight, up_weight, 2, 2, gate_up_input,
                                    gate_output, up_output)
           .is_ok());
  output = read_mpsgraph_f32_buffer(context, gate_output, 2);
  assert_close(output[0], 4.0F);
  assert_close(output[1], 5.0F);
  output = read_mpsgraph_f32_buffer(context, up_output, 2);
  assert_close(output[0], 8.0F);
  assert_close(output[1], 15.0F);

  auto attn_o_weight = make_mpsgraph_f32_buffer(context, {1.0F, 2.0F,
                                                          3.0F, 4.0F});
  auto attn_project_input = make_mpsgraph_f32_buffer(context, {2.0F, 3.0F});
  auto attn_residual = make_mpsgraph_f32_buffer(context, {10.0F, 20.0F});
  auto attn_norm_weight = make_mpsgraph_f32_buffer(context, {2.0F, 3.0F});
  auto attn_residual_output_result = context.make_buffer(2U * sizeof(float));
  auto attn_norm_output_result = context.make_buffer(2U * sizeof(float));
  assert(attn_residual_output_result.is_ok());
  assert(attn_norm_output_result.is_ok());
  auto attn_residual_output = std::move(attn_residual_output_result.value());
  auto attn_norm_output = std::move(attn_norm_output_result.value());
  assert(context.attn_project_residual_norm_f32(
           attn_o_weight, attn_project_input, attn_residual, attn_norm_weight,
           2, 2, 0.0F, attn_residual_output, attn_norm_output)
           .is_ok());
  output = read_mpsgraph_f32_buffer(context, attn_residual_output, 2);
  const auto attn_res0 = 10.0F + 8.0F;
  const auto attn_res1 = 20.0F + 18.0F;
  assert_close(output[0], attn_res0);
  assert_close(output[1], attn_res1);
  output = read_mpsgraph_f32_buffer(context, attn_norm_output, 2);
  const auto attn_norm_scale =
    1.0F / std::sqrt((attn_res0 * attn_res0 + attn_res1 * attn_res1) / 2.0F);
  assert_close(output[0], attn_res0 * attn_norm_scale * 2.0F);
  assert_close(output[1], attn_res1 * attn_norm_scale * 3.0F);

  auto swiglu_gate = make_mpsgraph_f32_buffer(context, {0.0F, 1.0F});
  auto swiglu_up = make_mpsgraph_f32_buffer(context, {5.0F, 6.0F});
  auto swiglu_down_weight = make_mpsgraph_f32_buffer(context, {1.0F, 2.0F,
                                                               3.0F, 4.0F});
  auto swiglu_residual = make_mpsgraph_f32_buffer(context, {10.0F, 20.0F});
  auto swiglu_output_result = context.make_buffer(2U * sizeof(float));
  assert(swiglu_output_result.is_ok());
  auto swiglu_output = std::move(swiglu_output_result.value());
  assert(context.swiglu_down_residual_f32(swiglu_gate, swiglu_up,
                                          swiglu_down_weight, swiglu_residual,
                                          2, 2, swiglu_output)
           .is_ok());
  output = read_mpsgraph_f32_buffer(context, swiglu_output, 2);
  const auto swiglu_0 = 0.0F;
  const auto swiglu_1 = 6.0F / (1.0F + std::exp(-1.0F));
  assert_close(output[0], 10.0F + swiglu_0 + 2.0F * swiglu_1);
  assert_close(output[1], 20.0F + 3.0F * swiglu_0 + 4.0F * swiglu_1);

  auto layer_hidden = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F});
  auto layer_norm_weight = make_mpsgraph_f32_buffer(context, {1.0F, 1.0F});
  auto layer_q_weight = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F,
                                                           0.0F, 1.0F});
  auto layer_k_weight = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F,
                                                           0.0F, 1.0F});
  auto layer_v_weight = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F,
                                                           0.0F, 1.0F});
  auto layer_o_weight = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F,
                                                           0.0F, 1.0F});
  auto layer_q_norm_weight = make_mpsgraph_f32_buffer(context, {1.0F, 1.0F});
  auto layer_k_norm_weight = make_mpsgraph_f32_buffer(context, {1.0F, 1.0F});
  auto layer_gate_weight = make_mpsgraph_f32_buffer(context, {0.0F, 0.0F,
                                                              0.0F, 0.0F});
  auto layer_up_weight = make_mpsgraph_f32_buffer(context, {0.0F, 0.0F,
                                                            0.0F, 0.0F});
  auto layer_down_weight = make_mpsgraph_f32_buffer(context, {0.0F, 0.0F,
                                                              0.0F, 0.0F});
  auto layer_key_cache = make_mpsgraph_f32_buffer(context, {0.0F, 0.0F});
  auto layer_value_cache = make_mpsgraph_f32_buffer(context, {0.0F, 0.0F});
  assert(context.transformer_layer_f32(
           layer_norm_weight, layer_q_weight, layer_k_weight, layer_v_weight,
           layer_o_weight, layer_q_norm_weight, layer_k_norm_weight,
           layer_norm_weight, layer_gate_weight, layer_up_weight, layer_down_weight,
           0, 1, 0, 1, 2, 2, 1, 1, 2, 0.0F, 10000.0F, layer_hidden,
           layer_key_cache, layer_value_cache)
           .is_ok());
  const auto layer_norm = std::sqrt(2.0F);
  output = read_mpsgraph_f32_buffer(context, layer_hidden, 2);
  assert_close(output[0], 1.0F + layer_norm);
  assert_close(output[1], 0.0F);
  output = read_mpsgraph_f32_buffer(context, layer_key_cache, 2);
  assert_close(output[0], layer_norm);
  assert_close(output[1], 0.0F);
  output = read_mpsgraph_f32_buffer(context, layer_value_cache, 2);
  assert_close(output[0], layer_norm);
  assert_close(output[1], 0.0F);

  auto stack_hidden = make_mpsgraph_f32_buffer(context, {1.0F, 0.0F});
  auto stack_key_cache = make_mpsgraph_f32_buffer(context, {0.0F, 0.0F});
  auto stack_value_cache = make_mpsgraph_f32_buffer(context, {0.0F, 0.0F});
  const std::vector<toyllm::mpsgraph::MpsGraphTransformerLayerBuffers> stack_layers{
    {
      &layer_norm_weight,
      &layer_q_weight,
      &layer_k_weight,
      &layer_v_weight,
      &layer_o_weight,
      &layer_q_norm_weight,
      &layer_k_norm_weight,
      &layer_norm_weight,
      &layer_gate_weight,
      &layer_up_weight,
      &layer_down_weight,
    },
  };
  assert(context.transformer_stack_f32(
           stack_layers, 0, 1, 0, 1, 2, 2, 1, 1, 2, 0.0F, 10000.0F,
           stack_hidden, stack_key_cache, stack_value_cache)
           .is_ok());
  output = read_mpsgraph_f32_buffer(context, stack_hidden, 2);
  assert_close(output[0], 1.0F + layer_norm);
  assert_close(output[1], 0.0F);
  output = read_mpsgraph_f32_buffer(context, stack_key_cache, 2);
  assert_close(output[0], layer_norm);
  assert_close(output[1], 0.0F);
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

  auto lhs_f32 = make_f32_buffer(context, {1.0F, 2.0F, 3.0F,
                                           4.0F, 5.0F, 6.0F});
  auto rhs_f32 = make_f32_buffer(context, {7.0F, 8.0F,
                                           9.0F, 10.0F,
                                           11.0F, 12.0F});
  auto matmul_output = make_f32_buffer(context, std::vector<float>(4, 0.0F));
  assert(context.matmul_f32_f32_device(lhs_f32, rhs_f32, 2, 3, 2,
                                       matmul_output)
           .is_ok());
  auto matmul_values = read_f32_buffer(context, matmul_output, 4);
  assert_close(matmul_values[0], 58.0F);
  assert_close(matmul_values[1], 64.0F);
  assert_close(matmul_values[2], 139.0F);
  assert_close(matmul_values[3], 154.0F);

  std::array<std::uint8_t, 144> q4_k_block{};
  q4_k_block[0] = 0x00;
  q4_k_block[1] = 0x3c;
  std::fill(q4_k_block.begin() + 4, q4_k_block.begin() + 16, 1);
  std::fill(q4_k_block.begin() + 16, q4_k_block.end(), 0x21);
  auto q4_weight_buffer_result = context.make_buffer(q4_k_block.size());
  assert(q4_weight_buffer_result.is_ok());
  auto q4_weight_buffer = std::move(q4_weight_buffer_result.value());
  assert(context.copy_to_buffer(q4_weight_buffer, q4_k_block.data(),
                                q4_k_block.size())
           .is_ok());
  const std::vector<float> q4_input(256, 1.0F);
  const auto q4_output = context.matvec_q4_k_f32(q4_weight_buffer, 1, 256, q4_input);
  assert(q4_output.is_ok());
  assert(q4_output.value().size() == 1);
  assert_close(q4_output.value()[0], 384.0F);
  auto q4_row_output = make_f32_buffer(context, std::vector<float>(256, 0.0F));
  assert(context.dequantize_row_q4_k_f32(q4_weight_buffer, 0, 256, q4_row_output).is_ok());
  auto q_row = read_f32_buffer(context, q4_row_output, 256);
  assert_close(q_row[0], 1.0F);
  assert_close(q_row[31], 1.0F);
  assert_close(q_row[32], 2.0F);
  assert_close(q_row[255], 2.0F);

  std::array<std::uint8_t, 176> q5_k_block{};
  q5_k_block[0] = 0x00;
  q5_k_block[1] = 0x3c;
  std::fill(q5_k_block.begin() + 4, q5_k_block.begin() + 16, 1);
  std::fill(q5_k_block.begin() + 16, q5_k_block.begin() + 48, 0xff);
  std::fill(q5_k_block.begin() + 48, q5_k_block.end(), 0x21);
  auto q5_weight_buffer_result = context.make_buffer(q5_k_block.size());
  assert(q5_weight_buffer_result.is_ok());
  auto q5_weight_buffer = std::move(q5_weight_buffer_result.value());
  assert(context.copy_to_buffer(q5_weight_buffer, q5_k_block.data(),
                                q5_k_block.size())
           .is_ok());
  const auto q5_output = context.matvec_q5_k_f32(q5_weight_buffer, 1, 256, q4_input);
  assert(q5_output.is_ok());
  assert(q5_output.value().size() == 1);
  assert_close(q5_output.value()[0], 4480.0F);
  auto q5_row_output = make_f32_buffer(context, std::vector<float>(256, 0.0F));
  assert(context.dequantize_row_q5_k_f32(q5_weight_buffer, 0, 256, q5_row_output).is_ok());
  q_row = read_f32_buffer(context, q5_row_output, 256);
  assert_close(q_row[0], 17.0F);
  assert_close(q_row[31], 17.0F);
  assert_close(q_row[32], 18.0F);
  assert_close(q_row[255], 18.0F);

  std::array<std::uint8_t, 210> q6_k_block{};
  std::fill(q6_k_block.begin() + 192, q6_k_block.begin() + 208, 1);
  q6_k_block[208] = 0x00;
  q6_k_block[209] = 0x3c;
  auto q6_weight_buffer_result = context.make_buffer(q6_k_block.size());
  assert(q6_weight_buffer_result.is_ok());
  auto q6_weight_buffer = std::move(q6_weight_buffer_result.value());
  assert(context.copy_to_buffer(q6_weight_buffer, q6_k_block.data(),
                                q6_k_block.size())
           .is_ok());
  const auto q6_output = context.matvec_q6_k_f32(q6_weight_buffer, 1, 256, q4_input);
  assert(q6_output.is_ok());
  assert(q6_output.value().size() == 1);
  assert_close(q6_output.value()[0], -8192.0F);
  auto q6_row_output = make_f32_buffer(context, std::vector<float>(256, 0.0F));
  assert(context.dequantize_row_q6_k_f32(q6_weight_buffer, 0, 256, q6_row_output).is_ok());
  q_row = read_f32_buffer(context, q6_row_output, 256);
  assert_close(q_row[0], -32.0F);
  assert_close(q_row[127], -32.0F);
  assert_close(q_row[128], -32.0F);
  assert_close(q_row[255], -32.0F);

  std::vector<float> batched_input(512, 1.0F);
  std::fill(batched_input.begin() + 256, batched_input.end(), 2.0F);
  auto batched_input_buffer = make_f32_buffer(context, batched_input);
  auto q4_matmul_output = make_f32_buffer(context, std::vector<float>(2, 0.0F));
  auto q5_matmul_output = make_f32_buffer(context, std::vector<float>(2, 0.0F));
  auto q6_matmul_output = make_f32_buffer(context, std::vector<float>(2, 0.0F));

  std::vector<std::uint8_t> q4_two_rows;
  q4_two_rows.insert(q4_two_rows.end(), q4_k_block.begin(), q4_k_block.end());
  q4_two_rows.insert(q4_two_rows.end(), q4_k_block.begin(), q4_k_block.end());
  auto q4_two_rows_result = context.make_buffer(q4_two_rows.size());
  assert(q4_two_rows_result.is_ok());
  auto q4_two_rows_buffer = std::move(q4_two_rows_result.value());
  assert(context.copy_to_buffer(q4_two_rows_buffer, q4_two_rows.data(),
                                q4_two_rows.size())
           .is_ok());

  std::vector<std::uint8_t> q5_two_rows;
  q5_two_rows.insert(q5_two_rows.end(), q5_k_block.begin(), q5_k_block.end());
  q5_two_rows.insert(q5_two_rows.end(), q5_k_block.begin(), q5_k_block.end());
  auto q5_two_rows_result = context.make_buffer(q5_two_rows.size());
  assert(q5_two_rows_result.is_ok());
  auto q5_two_rows_buffer = std::move(q5_two_rows_result.value());
  assert(context.copy_to_buffer(q5_two_rows_buffer, q5_two_rows.data(),
                                q5_two_rows.size())
           .is_ok());

  std::vector<std::uint8_t> q6_two_rows;
  q6_two_rows.insert(q6_two_rows.end(), q6_k_block.begin(), q6_k_block.end());
  q6_two_rows.insert(q6_two_rows.end(), q6_k_block.begin(), q6_k_block.end());
  auto q6_two_rows_result = context.make_buffer(q6_two_rows.size());
  assert(q6_two_rows_result.is_ok());
  auto q6_two_rows_buffer = std::move(q6_two_rows_result.value());
  assert(context.copy_to_buffer(q6_two_rows_buffer, q6_two_rows.data(),
                                q6_two_rows.size())
           .is_ok());

  auto row_ids = make_i32_buffer(context, {1, 0});
  auto q4_rows_output = make_f32_buffer(context, std::vector<float>(512, 0.0F));
  auto q5_rows_output = make_f32_buffer(context, std::vector<float>(512, 0.0F));
  auto q6_rows_output = make_f32_buffer(context, std::vector<float>(512, 0.0F));

  assert(context.begin_graph().is_ok());
  assert(context.matmul_q4_k_f32_device(q4_weight_buffer, 1, 256, 2,
                                        batched_input_buffer, q4_matmul_output)
           .is_ok());
  assert(context.matmul_q5_k_f32_device(q5_weight_buffer, 1, 256, 2,
                                        batched_input_buffer, q5_matmul_output)
           .is_ok());
  assert(context.matmul_q6_k_f32_device(q6_weight_buffer, 1, 256, 2,
                                        batched_input_buffer, q6_matmul_output)
           .is_ok());
  assert(context.dequantize_rows_q4_k_f32(q4_two_rows_buffer, 2, row_ids, 2,
                                          256, q4_rows_output)
           .is_ok());
  assert(context.dequantize_rows_q5_k_f32(q5_two_rows_buffer, 2, row_ids, 2,
                                          256, q5_rows_output)
           .is_ok());
  assert(context.dequantize_rows_q6_k_f32(q6_two_rows_buffer, 2, row_ids, 2,
                                          256, q6_rows_output)
           .is_ok());
  assert(context.commit_graph().is_ok());

  auto batched_output = read_f32_buffer(context, q4_matmul_output, 2);
  assert_close(batched_output[0], 384.0F);
  assert_close(batched_output[1], 768.0F);
  batched_output = read_f32_buffer(context, q5_matmul_output, 2);
  assert_close(batched_output[0], 4480.0F);
  assert_close(batched_output[1], 8960.0F);
  batched_output = read_f32_buffer(context, q6_matmul_output, 2);
  assert_close(batched_output[0], -8192.0F);
  assert_close(batched_output[1], -16384.0F);

  auto rows_output = read_f32_buffer(context, q4_rows_output, 512);
  assert_close(rows_output[0], 1.0F);
  assert_close(rows_output[32], 2.0F);
  assert_close(rows_output[256], 1.0F);
  assert_close(rows_output[256 + 32], 2.0F);
  rows_output = read_f32_buffer(context, q5_rows_output, 512);
  assert_close(rows_output[0], 17.0F);
  assert_close(rows_output[32], 18.0F);
  assert_close(rows_output[256], 17.0F);
  assert_close(rows_output[256 + 32], 18.0F);
  rows_output = read_f32_buffer(context, q6_rows_output, 512);
  assert_close(rows_output[0], -32.0F);
  assert_close(rows_output[127], -32.0F);
  assert_close(rows_output[256], -32.0F);
  assert_close(rows_output[256 + 255], -32.0F);

  constexpr std::size_t simd_rows = 65;
  constexpr std::size_t simd_tokens = 33;
  auto repeat_block = [](const auto& block, std::size_t rows) {
    std::vector<std::uint8_t> repeated;
    repeated.reserve(block.size() * rows);
    for (std::size_t row = 0; row < rows; ++row) {
      repeated.insert(repeated.end(), block.begin(), block.end());
    }
    return repeated;
  };
  auto make_device_bytes = [&context](const std::vector<std::uint8_t>& bytes) {
    auto buffer_result = context.make_buffer(bytes.size());
    assert(buffer_result.is_ok());
    auto buffer = std::move(buffer_result.value());
    assert(context.copy_to_buffer(buffer, bytes.data(), bytes.size()).is_ok());
    return buffer;
  };

  auto q4_simd_weight = make_device_bytes(repeat_block(q4_k_block, simd_rows));
  auto q5_simd_weight = make_device_bytes(repeat_block(q5_k_block, simd_rows));
  auto q6_simd_weight = make_device_bytes(repeat_block(q6_k_block, simd_rows));

  auto check_batched_quant_matmul = [&](std::size_t tokens) {
    std::vector<float> input(tokens * 256U, 0.0F);
    for (std::size_t token = 0; token < tokens; ++token) {
      const float value = static_cast<float>((token % 7U) + 1U);
      std::fill(input.begin() + static_cast<std::ptrdiff_t>(token * 256U),
                input.begin() + static_cast<std::ptrdiff_t>((token + 1U) * 256U),
                value);
    }
    auto input_buffer = make_f32_buffer(context, input);
    auto q4_output =
      make_f32_buffer(context, std::vector<float>(tokens * simd_rows, 0.0F));
    auto q5_output =
      make_f32_buffer(context, std::vector<float>(tokens * simd_rows, 0.0F));
    auto q6_output =
      make_f32_buffer(context, std::vector<float>(tokens * simd_rows, 0.0F));

    assert(context.begin_graph().is_ok());
    assert(context.matmul_q4_k_f32_device(q4_simd_weight, simd_rows, 256,
                                          tokens, input_buffer, q4_output)
             .is_ok());
    assert(context.matmul_q5_k_f32_device(q5_simd_weight, simd_rows, 256,
                                          tokens, input_buffer, q5_output)
             .is_ok());
    assert(context.matmul_q6_k_f32_device(q6_simd_weight, simd_rows, 256,
                                          tokens, input_buffer, q6_output)
             .is_ok());
    assert(context.commit_graph().is_ok());

    auto assert_outputs = [&](const std::vector<float>& values, float base) {
      for (std::size_t token : std::array<std::size_t, 2>{0U, tokens - 1U}) {
        const float scale = static_cast<float>((token % 7U) + 1U);
        assert_close(values[token * simd_rows], base * scale);
        assert_close(values[token * simd_rows + 63U], base * scale);
        assert_close(values[token * simd_rows + simd_rows - 1U], base * scale);
      }
    };
    assert_outputs(read_f32_buffer(context, q4_output, tokens * simd_rows),
                   384.0F);
    assert_outputs(read_f32_buffer(context, q5_output, tokens * simd_rows),
                   4480.0F);
    assert_outputs(read_f32_buffer(context, q6_output, tokens * simd_rows),
                   -8192.0F);
  };

  check_batched_quant_matmul(4);
  check_batched_quant_matmul(5);
  check_batched_quant_matmul(6);

  std::vector<float> simd_input(simd_tokens * 256U, 0.0F);
  for (std::size_t token = 0; token < simd_tokens; ++token) {
    const float value = static_cast<float>((token % 7U) + 1U);
    std::fill(simd_input.begin() + static_cast<std::ptrdiff_t>(token * 256U),
              simd_input.begin() + static_cast<std::ptrdiff_t>((token + 1U) * 256U),
              value);
  }
  auto simd_input_buffer = make_f32_buffer(context, simd_input);
  auto q4_simd_output =
    make_f32_buffer(context, std::vector<float>(simd_tokens * simd_rows, 0.0F));
  auto q5_simd_output =
    make_f32_buffer(context, std::vector<float>(simd_tokens * simd_rows, 0.0F));
  auto q6_simd_output =
    make_f32_buffer(context, std::vector<float>(simd_tokens * simd_rows, 0.0F));

  assert(context.begin_graph().is_ok());
  assert(context.matmul_q4_k_f32_device(q4_simd_weight, simd_rows, 256,
                                        simd_tokens, simd_input_buffer,
                                        q4_simd_output)
           .is_ok());
  assert(context.matmul_q5_k_f32_device(q5_simd_weight, simd_rows, 256,
                                        simd_tokens, simd_input_buffer,
                                        q5_simd_output)
           .is_ok());
  assert(context.matmul_q6_k_f32_device(q6_simd_weight, simd_rows, 256,
                                        simd_tokens, simd_input_buffer,
                                        q6_simd_output)
           .is_ok());
  assert(context.commit_graph().is_ok());

  auto assert_simd_outputs = [](const std::vector<float>& values, float base) {
    for (std::size_t token : {0U, 31U, 32U}) {
      const float scale = static_cast<float>((token % 7U) + 1U);
      assert_close(values[token * simd_rows], base * scale);
      assert_close(values[token * simd_rows + 63U], base * scale);
      assert_close(values[token * simd_rows + simd_rows - 1U], base * scale);
    }
  };
  assert_simd_outputs(read_f32_buffer(context, q4_simd_output,
                                      simd_tokens * simd_rows),
                      384.0F);
  assert_simd_outputs(read_f32_buffer(context, q5_simd_output,
                                      simd_tokens * simd_rows),
                      4480.0F);
  assert_simd_outputs(read_f32_buffer(context, q6_simd_output,
                                      simd_tokens * simd_rows),
                      -8192.0F);
}

void test_mps_graph_scope_batches_ops() {
  auto context_result = toyllm::mps::MpsContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  auto target = make_f32_buffer(context, {1.0F, 2.0F});
  auto delta = make_f32_buffer(context, {3.0F, 4.0F});
  auto scale = make_f32_buffer(context, {2.0F, 3.0F});

  assert(context.begin_graph().is_ok());
  assert(context.add_f32_in_place(target, delta, 2).is_ok());
  assert(context.mul_f32_in_place(target, scale, 2).is_ok());
  assert(context.commit_graph().is_ok());

  const auto output = read_f32_buffer(context, target, 2);
  assert_close(output[0], 8.0F);
  assert_close(output[1], 18.0F);
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

  auto zeroed = make_f32_buffer(context, {7.0F, 8.0F});
  assert(context.zero_buffer(zeroed, 2U * sizeof(float)).is_ok());
  output = read_f32_buffer(context, zeroed, 2);
  assert_close(output[0], 0.0F);
  assert_close(output[1], 0.0F);

  auto norm_input = make_f32_buffer(context, {1.0F, 1.0F});
  auto norm_weight = make_bf16_buffer(context, {2.0F, 3.0F});
  auto norm_output = make_f32_buffer(context, {0.0F, 0.0F});
  assert(context.rms_norm_f32_bf16(norm_input, norm_weight, 2, 0.0F, norm_output).is_ok());
  output = read_f32_buffer(context, norm_output, 2);
  assert_close(output[0], 2.0F);
  assert_close(output[1], 3.0F);

  auto norm_weight_f32 = make_f32_buffer(context, {2.0F, 3.0F});
  auto norm_output_f32 = make_f32_buffer(context, {0.0F, 0.0F});
  assert(context.rms_norm_f32_f32(norm_input, norm_weight_f32, 2, 0.0F,
                                  norm_output_f32)
           .is_ok());
  output = read_f32_buffer(context, norm_output_f32, 2);
  assert_close(output[0], 2.0F);
  assert_close(output[1], 3.0F);

  auto batched_norm_input = make_f32_buffer(context, {1.0F, 1.0F, 3.0F, 4.0F});
  auto batched_norm_output = make_f32_buffer(context, std::vector<float>(4, 0.0F));
  auto batched_qk_values = make_f32_buffer(context, {1.0F, 1.0F, 3.0F, 4.0F});
  assert(context.begin_graph().is_ok());
  assert(context.rms_norm_f32_f32_batched(batched_norm_input, norm_weight_f32,
                                          2, 2, 0.0F, batched_norm_output)
           .is_ok());
  assert(context.qk_norm_f32_f32_batched(batched_qk_values, norm_weight_f32,
                                         2, 1, 2, 0.0F)
           .is_ok());
  assert(context.commit_graph().is_ok());
  output = read_f32_buffer(context, batched_norm_output, 4);
  assert_close(output[0], 2.0F);
  assert_close(output[1], 3.0F);
  const auto norm_scale_34 = 1.0F / std::sqrt(12.5F);
  assert_close(output[2], 3.0F * norm_scale_34 * 2.0F);
  assert_close(output[3], 4.0F * norm_scale_34 * 3.0F);
  output = read_f32_buffer(context, batched_qk_values, 4);
  assert_close(output[0], 2.0F);
  assert_close(output[1], 3.0F);
  assert_close(output[2], 3.0F * norm_scale_34 * 2.0F);
  assert_close(output[3], 4.0F * norm_scale_34 * 3.0F);

  auto norm_gated_values = make_f32_buffer(context, {1.0F, 1.0F, 3.0F, 4.0F});
  auto norm_gated_gate = make_f32_buffer(context, {0.0F, 1.0F, -1.0F, 2.0F});
  assert(context.qwen35_norm_gated_f32_in_place(
                  norm_gated_values, norm_weight_f32, norm_gated_gate, 2, 1, 2, 0.0F)
           .is_ok());
  output = read_f32_buffer(context, norm_gated_values, 4);
  const auto silu_1 = 1.0F / (1.0F + std::exp(-1.0F));
  const auto silu_neg_1 = -1.0F / (1.0F + std::exp(1.0F));
  const auto silu_2 = 2.0F / (1.0F + std::exp(-2.0F));
  assert_close(output[0], 0.0F);
  assert_close(output[1], 3.0F * silu_1);
  assert_close(output[2], 3.0F * norm_scale_34 * 2.0F * silu_neg_1);
  assert_close(output[3], 4.0F * norm_scale_34 * 3.0F * silu_2);

  auto q_values = make_f32_buffer(context, {1.0F, 1.0F});
  assert(context.qk_norm_f32_bf16(q_values, norm_weight, 1, 2, 0.0F).is_ok());
  assert(context.rope_f32(q_values, 1, 2, 0, 10000.0F).is_ok());
  output = read_f32_buffer(context, q_values, 2);
  assert_close(output[0], 2.0F);
  assert_close(output[1], 3.0F);

  auto mrope_values = make_f32_buffer(context, {1.0F, 1.0F, 1.0F, 1.0F,
                                                0.0F, 0.0F, 0.0F, 0.0F});
  auto mrope_positions = make_i32_buffer(context, {1, 2, 3, 4});
  assert(context.mrope_f32_in_place(mrope_values, mrope_positions, 1, 1, 8, 8,
                                    1, 1, 1, 1, 10000.0F)
           .is_ok());
  output = read_f32_buffer(context, mrope_values, 8);
  const float angle0 = 1.0F;
  const float angle1 = 2.0F / 10.0F;
  const float angle2 = 3.0F / 100.0F;
  const float angle3 = 4.0F / 1000.0F;
  assert_close(output[0], std::cos(angle0));
  assert_close(output[1], std::cos(angle1));
  assert_close(output[2], std::cos(angle2));
  assert_close(output[3], std::cos(angle3));
  assert_close(output[4], std::sin(angle0));
  assert_close(output[5], std::sin(angle1));
  assert_close(output[6], std::sin(angle2));
  assert_close(output[7], std::sin(angle3));

  auto q_values_f32 = make_f32_buffer(context, {1.0F, 1.0F});
  assert(context.qk_norm_f32_f32(q_values_f32, norm_weight_f32, 1, 2, 0.0F).is_ok());
  output = read_f32_buffer(context, q_values_f32, 2);
  assert_close(output[0], 2.0F);
  assert_close(output[1], 3.0F);

  auto l2_values = make_f32_buffer(context, {3.0F, 4.0F, 0.0F, 2.0F});
  assert(context.l2_norm_f32_in_place(l2_values, 2, 2, 0.0F).is_ok());
  output = read_f32_buffer(context, l2_values, 4);
  assert_close(output[0], 0.6F);
  assert_close(output[1], 0.8F);
  assert_close(output[2], 0.0F);
  assert_close(output[3], 1.0F);

  auto split_qkv_source = make_f32_buffer(
    context, {3.0F, 4.0F, 0.0F, 2.0F,
              5.0F, 12.0F, 8.0F, 15.0F,
              1.0F, 2.0F, 3.0F, 4.0F,
              1.0F, 0.0F, 2.0F, 0.0F,
              0.0F, 6.0F, 7.0F, 24.0F,
              9.0F, 10.0F, 11.0F, 12.0F});
  auto split_q = make_f32_buffer(context, std::vector<float>(8, 0.0F));
  auto split_k = make_f32_buffer(context, std::vector<float>(8, 0.0F));
  auto split_v = make_f32_buffer(context, std::vector<float>(8, 0.0F));
  assert(context.split_qkv_l2_norm_f32_qwen35(
                  split_qkv_source, split_q, split_k, split_v, 2, 2, 2, 2, 0.0F)
           .is_ok());
  output = read_f32_buffer(context, split_q, 8);
  assert_close(output[0], 0.6F);
  assert_close(output[1], 0.8F);
  assert_close(output[2], 0.0F);
  assert_close(output[3], 1.0F);
  assert_close(output[4], 1.0F);
  assert_close(output[5], 0.0F);
  assert_close(output[6], 1.0F);
  assert_close(output[7], 0.0F);
  output = read_f32_buffer(context, split_k, 8);
  assert_close(output[0], 5.0F / 13.0F);
  assert_close(output[1], 12.0F / 13.0F);
  assert_close(output[2], 8.0F / 17.0F);
  assert_close(output[3], 15.0F / 17.0F);
  assert_close(output[4], 0.0F);
  assert_close(output[5], 1.0F);
  assert_close(output[6], 7.0F / 25.0F);
  assert_close(output[7], 24.0F / 25.0F);
  output = read_f32_buffer(context, split_v, 8);
  assert_close(output[0], 1.0F);
  assert_close(output[1], 2.0F);
  assert_close(output[2], 3.0F);
  assert_close(output[3], 4.0F);
  assert_close(output[4], 9.0F);
  assert_close(output[5], 10.0F);
  assert_close(output[6], 11.0F);
  assert_close(output[7], 12.0F);

  auto ssm_input = make_f32_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
                                             10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F});
  auto ssm_kernel = make_f32_buffer(context, {1.0F, 0.0F, 0.0F, 0.0F,
                                              0.0F, 1.0F, 0.0F, 0.0F});
  auto ssm_output = make_f32_buffer(context, {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  assert(context.ssm_conv_f32(ssm_input, ssm_kernel, 4, 2, 3, 1, ssm_output).is_ok());
  output = read_f32_buffer(context, ssm_output, 6);
  assert_close(output[0], 1.0F);
  assert_close(output[1], 20.0F);
  assert_close(output[2], 2.0F);
  assert_close(output[3], 30.0F);
  assert_close(output[4], 3.0F);
  assert_close(output[5], 40.0F);

  auto ssm_state = make_f32_buffer(context, {1.0F, 2.0F, 3.0F,
                                             10.0F, 20.0F, 30.0F});
  auto ssm_step = make_f32_buffer(context, {4.0F, 40.0F});
  auto ssm_stateful_kernel = make_f32_buffer(context, {1.0F, 1.0F, 1.0F, 1.0F,
                                                       0.0F, 1.0F, 0.0F, 1.0F});
  auto ssm_step_output = make_f32_buffer(context, {0.0F, 0.0F});
  assert(context.ssm_conv1_f32_stateful(ssm_state, ssm_step, ssm_stateful_kernel,
                                        4, 2, ssm_step_output)
           .is_ok());
  output = read_f32_buffer(context, ssm_step_output, 2);
  assert_close(output[0], 10.0F);
  assert_close(output[1], 60.0F);
  output = read_f32_buffer(context, ssm_state, 6);
  assert_close(output[0], 2.0F);
  assert_close(output[1], 3.0F);
  assert_close(output[2], 4.0F);
  assert_close(output[3], 20.0F);
  assert_close(output[4], 30.0F);
  assert_close(output[5], 40.0F);

  auto chunk_state = make_f32_buffer(context, {10.0F, 20.0F, 30.0F,
                                               40.0F, 50.0F, 60.0F});
  auto chunk_input = make_f32_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F});
  auto chunk_conv_input = make_f32_buffer(context, std::vector<float>(10, 0.0F));
  auto chunk_kernel = make_f32_buffer(context, std::vector<float>(8, 1.0F));
  auto chunk_conv_output = make_f32_buffer(context, std::vector<float>(4, 0.0F));
  assert(context.begin_graph().is_ok());
  assert(context.build_ssm_conv_state_f32(chunk_state, 0, chunk_input, 4, 2, 2,
                                          chunk_conv_input)
           .is_ok());
  assert(context.ssm_conv_f32(chunk_conv_input, chunk_kernel, 4, 2, 2, 1,
                              chunk_conv_output)
           .is_ok());
  assert(context.commit_graph().is_ok());
  output = read_f32_buffer(context, chunk_conv_input, 10);
  assert_close(output[0], 10.0F);
  assert_close(output[1], 20.0F);
  assert_close(output[2], 30.0F);
  assert_close(output[3], 1.0F);
  assert_close(output[4], 3.0F);
  assert_close(output[5], 40.0F);
  assert_close(output[6], 50.0F);
  assert_close(output[7], 60.0F);
  assert_close(output[8], 2.0F);
  assert_close(output[9], 4.0F);
  output = read_f32_buffer(context, chunk_state, 6);
  assert_close(output[0], 30.0F);
  assert_close(output[1], 1.0F);
  assert_close(output[2], 3.0F);
  assert_close(output[3], 60.0F);
  assert_close(output[4], 2.0F);
  assert_close(output[5], 4.0F);
  output = read_f32_buffer(context, chunk_conv_output, 4);
  assert_close(output[0], 61.0F);
  assert_close(output[1], 152.0F);
  assert_close(output[2], 54.0F);
  assert_close(output[3], 116.0F);

  auto gdn_query = make_f32_buffer(context, {1.0F, 0.0F});
  auto gdn_key = make_f32_buffer(context, {0.5F, 0.25F});
  auto gdn_value = make_f32_buffer(context, {10.0F, 20.0F});
  auto gdn_gate = make_f32_buffer(context, {0.0F});
  auto gdn_beta = make_f32_buffer(context, {1.0F});
  auto gdn_state = make_f32_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F});
  auto gdn_output = make_f32_buffer(context, {0.0F, 0.0F});
  assert(context.gated_delta_net_f32_in_place(gdn_query, gdn_key, gdn_value,
                                              gdn_gate, gdn_beta, gdn_state,
                                              1, 1, 2, gdn_output)
           .is_ok());
  output = read_f32_buffer(context, gdn_output, 2);
  assert_close(output[0], 5.5F / std::sqrt(2.0F));
  assert_close(output[1], 11.75F / std::sqrt(2.0F));
  output = read_f32_buffer(context, gdn_state, 4);
  assert_close(output[0], 5.5F);
  assert_close(output[1], 4.25F);
  assert_close(output[2], 11.75F);
  assert_close(output[3], 8.375F);

  auto batched_gdn_query = make_f32_buffer(context, {1.0F, 0.0F, 0.0F, 1.0F});
  auto batched_gdn_key = make_f32_buffer(context, {0.5F, 0.25F, 0.25F, 0.5F});
  auto batched_gdn_value = make_f32_buffer(context, {10.0F, 20.0F, 1.0F, 2.0F});
  auto batched_gdn_gate = make_f32_buffer(context, {0.0F, 0.0F});
  auto batched_gdn_beta = make_f32_buffer(context, {1.0F, 0.5F});
  auto batched_gdn_state = make_f32_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F});
  auto batched_gdn_output = make_f32_buffer(context, std::vector<float>(4, 0.0F));
  assert(context.begin_graph().is_ok());
  assert(context.gated_delta_net_f32_batched_in_place(
                  batched_gdn_query, batched_gdn_key, batched_gdn_value,
                  batched_gdn_gate, batched_gdn_beta, batched_gdn_state,
                  2, 1, 1, 2, batched_gdn_output)
           .is_ok());
  assert(context.commit_graph().is_ok());
  output = read_f32_buffer(context, batched_gdn_output, 4);
  assert_close(output[0], 5.5F / std::sqrt(2.0F));
  assert_close(output[1], 11.75F / std::sqrt(2.0F));
  assert_close(output[2], 3.625F / std::sqrt(2.0F));
  assert_close(output[3], 7.09375F / std::sqrt(2.0F));
  output = read_f32_buffer(context, batched_gdn_state, 4);
  assert_close(output[0], 5.1875F);
  assert_close(output[1], 3.625F);
  assert_close(output[2], 11.109375F);
  assert_close(output[3], 7.09375F);

  constexpr std::size_t qwen_gdn_tokens = 2;
  constexpr std::size_t qwen_gdn_heads = 1;
  constexpr std::size_t qwen_gdn_head_dim = 128;
  const auto qwen_gdn_state_values = qwen_gdn_heads * qwen_gdn_head_dim *
                                     qwen_gdn_head_dim;
  std::vector<float> qwen_gdn_query(qwen_gdn_tokens * qwen_gdn_head_dim);
  std::vector<float> qwen_gdn_key(qwen_gdn_tokens * qwen_gdn_head_dim);
  std::vector<float> qwen_gdn_value(qwen_gdn_tokens * qwen_gdn_head_dim);
  std::vector<float> qwen_gdn_gate(qwen_gdn_tokens, 0.0F);
  std::vector<float> qwen_gdn_beta{0.75F, 0.5F};
  std::vector<float> qwen_gdn_state(qwen_gdn_state_values);
  for (std::size_t i = 0; i < qwen_gdn_query.size(); ++i) {
    qwen_gdn_query[i] = 0.001F * static_cast<float>(1U + i % 13U);
    qwen_gdn_key[i] = 0.0007F * static_cast<float>(1U + i % 17U);
    qwen_gdn_value[i] = 0.01F * static_cast<float>(1U + i % 19U);
  }
  for (std::size_t i = 0; i < qwen_gdn_state.size(); ++i) {
    qwen_gdn_state[i] = 0.0002F * static_cast<float>(1U + i % 23U);
  }
  auto qwen_gdn_expected_state = qwen_gdn_state;
  std::vector<float> qwen_gdn_expected_output(qwen_gdn_value.size(), 0.0F);
  for (std::size_t token = 0; token < qwen_gdn_tokens; ++token) {
    const auto qk_base = token * qwen_gdn_head_dim;
    const auto value_base = token * qwen_gdn_head_dim;
    const auto gate_exp = std::exp(qwen_gdn_gate[token]);
    for (std::size_t row = 0; row < qwen_gdn_head_dim; ++row) {
      const auto state_base = row * qwen_gdn_head_dim;
      float sk = 0.0F;
      for (std::size_t dim = 0; dim < qwen_gdn_head_dim; ++dim) {
        qwen_gdn_expected_state[state_base + dim] *= gate_exp;
        sk += qwen_gdn_expected_state[state_base + dim] *
              qwen_gdn_key[qk_base + dim];
      }
      const auto delta =
        (qwen_gdn_value[value_base + row] - sk) * qwen_gdn_beta[token];
      float y = 0.0F;
      for (std::size_t dim = 0; dim < qwen_gdn_head_dim; ++dim) {
        qwen_gdn_expected_state[state_base + dim] +=
          qwen_gdn_key[qk_base + dim] * delta;
        y += qwen_gdn_expected_state[state_base + dim] *
             qwen_gdn_query[qk_base + dim];
      }
      qwen_gdn_expected_output[value_base + row] =
        y / std::sqrt(static_cast<float>(qwen_gdn_head_dim));
    }
  }
  auto qwen_gdn_query_buffer = make_f32_buffer(context, qwen_gdn_query);
  auto qwen_gdn_key_buffer = make_f32_buffer(context, qwen_gdn_key);
  auto qwen_gdn_value_buffer = make_f32_buffer(context, qwen_gdn_value);
  auto qwen_gdn_gate_buffer = make_f32_buffer(context, qwen_gdn_gate);
  auto qwen_gdn_beta_buffer = make_f32_buffer(context, qwen_gdn_beta);
  auto qwen_gdn_state_buffer = make_f32_buffer(context, qwen_gdn_state);
  auto qwen_gdn_output_buffer =
    make_f32_buffer(context, std::vector<float>(qwen_gdn_value.size(), 0.0F));
  assert(context.gated_delta_net_f32_batched_in_place(
                  qwen_gdn_query_buffer, qwen_gdn_key_buffer,
                  qwen_gdn_value_buffer, qwen_gdn_gate_buffer,
                  qwen_gdn_beta_buffer, qwen_gdn_state_buffer,
                  qwen_gdn_tokens, qwen_gdn_heads, qwen_gdn_heads,
                  qwen_gdn_head_dim, qwen_gdn_output_buffer)
           .is_ok());
  output = read_f32_buffer(context, qwen_gdn_output_buffer,
                           qwen_gdn_expected_output.size());
  for (std::size_t i = 0; i < output.size(); ++i) {
    assert(std::abs(output[i] - qwen_gdn_expected_output[i]) < 5e-5F);
  }
  output = read_f32_buffer(context, qwen_gdn_state_buffer,
                           qwen_gdn_expected_state.size());
  for (std::size_t i = 0; i < output.size(); ++i) {
    assert(std::abs(output[i] - qwen_gdn_expected_state[i]) < 5e-5F);
  }

  auto target = make_f32_buffer(context, {1.0F, 2.0F});
  auto delta = make_f32_buffer(context, {3.0F, 4.0F});
  assert(context.add_f32_in_place(target, delta, 2).is_ok());
  output = read_f32_buffer(context, target, 2);
  assert_close(output[0], 4.0F);
  assert_close(output[1], 6.0F);

  auto mul_target = make_f32_buffer(context, {2.0F, 3.0F});
  auto mul_rhs = make_f32_buffer(context, {4.0F, 5.0F});
  assert(context.mul_f32_in_place(mul_target, mul_rhs, 2).is_ok());
  output = read_f32_buffer(context, mul_target, 2);
  assert_close(output[0], 8.0F);
  assert_close(output[1], 15.0F);

  auto rowwise_values = make_f32_buffer(context, {1.0F, 2.0F, 3.0F,
                                                  4.0F, 5.0F, 6.0F});
  auto rowwise_bias = make_f32_buffer(context, {10.0F, 20.0F, 30.0F});
  auto rowwise_scale = make_f32_buffer(context, {1.0F, 0.5F, 2.0F});
  assert(context.add_f32_row_in_place(rowwise_values, rowwise_bias, 2, 3).is_ok());
  assert(context.mul_f32_row_in_place(rowwise_values, rowwise_scale, 2, 3).is_ok());
  output = read_f32_buffer(context, rowwise_values, 6);
  assert_close(output[0], 11.0F);
  assert_close(output[1], 11.0F);
  assert_close(output[2], 66.0F);
  assert_close(output[3], 14.0F);
  assert_close(output[4], 12.5F);
  assert_close(output[5], 72.0F);

  auto sigmoid_values = make_f32_buffer(context, {-1.0F, 0.0F});
  assert(context.sigmoid_f32_in_place(sigmoid_values, 2).is_ok());
  output = read_f32_buffer(context, sigmoid_values, 2);
  assert_close(output[0], 1.0F / (1.0F + std::exp(1.0F)));
  assert_close(output[1], 0.5F);

  auto softplus_values = make_f32_buffer(context, {0.0F, 2.0F});
  assert(context.softplus_f32_in_place(softplus_values, 2).is_ok());
  output = read_f32_buffer(context, softplus_values, 2);
  assert_close(output[0], std::log(2.0F));
  assert_close(output[1], std::log(1.0F + std::exp(2.0F)));

  auto prepared_gdn_gate = make_f32_buffer(context, {0.0F, 2.0F, -1.0F, 1.0F});
  auto prepared_gdn_beta = make_f32_buffer(context, {-1.0F, 0.0F, 1.0F, 2.0F});
  auto prepared_gdn_gate_bias = make_f32_buffer(context, {0.5F, -0.5F});
  auto prepared_gdn_gate_scale = make_f32_buffer(context, {2.0F, -3.0F});
  assert(context.prepare_qwen35_gdn_gate_beta_f32(
                  prepared_gdn_gate, prepared_gdn_beta, prepared_gdn_gate_bias,
                  prepared_gdn_gate_scale, 2, 2)
           .is_ok());
  output = read_f32_buffer(context, prepared_gdn_gate, 4);
  assert_close(output[0], std::log(1.0F + std::exp(0.5F)) * 2.0F);
  assert_close(output[1], std::log(1.0F + std::exp(1.5F)) * -3.0F);
  assert_close(output[2], std::log(1.0F + std::exp(-0.5F)) * 2.0F);
  assert_close(output[3], std::log(1.0F + std::exp(0.5F)) * -3.0F);
  output = read_f32_buffer(context, prepared_gdn_beta, 4);
  assert_close(output[0], 1.0F / (1.0F + std::exp(1.0F)));
  assert_close(output[1], 0.5F);
  assert_close(output[2], 1.0F / (1.0F + std::exp(-1.0F)));
  assert_close(output[3], 1.0F / (1.0F + std::exp(-2.0F)));

  auto silu_values = make_f32_buffer(context, {0.0F, 1.0F});
  assert(context.silu_f32_in_place(silu_values, 2).is_ok());
  output = read_f32_buffer(context, silu_values, 2);
  assert_close(output[0], 0.0F);
  assert_close(output[1], 1.0F / (1.0F + std::exp(-1.0F)));

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

  auto row_source = make_f32_buffer(context, {0.0F, 1.0F, 2.0F, 3.0F,
                                              4.0F, 5.0F, 6.0F, 7.0F,
                                              8.0F, 9.0F, 10.0F, 11.0F});
  auto row_destination = make_f32_buffer(context, std::vector<float>(9, -1.0F));
  assert(context.copy_f32_rows(row_source, row_destination, 3, 2, 4, 1, 3, 0).is_ok());
  output = read_f32_buffer(context, row_destination, 9);
  assert_close(output[0], 1.0F);
  assert_close(output[1], 2.0F);
  assert_close(output[2], -1.0F);
  assert_close(output[3], 5.0F);
  assert_close(output[4], 6.0F);
  assert_close(output[5], -1.0F);
  assert_close(output[6], 9.0F);
  assert_close(output[7], 10.0F);
  assert_close(output[8], -1.0F);

  auto argmax_values = make_f32_buffer(context, {-1.0F, 5.0F, 2.0F});
  auto argmax_output_result = context.make_buffer(sizeof(std::int32_t));
  assert(argmax_output_result.is_ok());
  auto argmax_output = std::move(argmax_output_result.value());
  assert(context.argmax_f32_i32(argmax_values, 3, argmax_output).is_ok());
  std::int32_t argmax = -1;
  assert(context.copy_from_buffer(argmax_output, &argmax, sizeof(argmax)).is_ok());
  assert(argmax == 1);

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

  auto f16_key_cache = make_empty_buffer(context, 2U * sizeof(std::uint16_t));
  auto f16_value_cache = make_empty_buffer(context, 2U * sizeof(std::uint16_t));
  assert(context.copy_f32_rows_to_f16(key_cache, f16_key_cache, 1, 2, 2, 0,
                                      2, 0)
           .is_ok());
  assert(context.copy_f32_rows_to_f16(value_cache, f16_value_cache, 1, 2, 2,
                                      0, 2, 0)
           .is_ok());
  auto f16_attention_output = make_f32_buffer(context, {0.0F, 0.0F});
  assert(context.attention_f32_f16_kv(query, f16_key_cache, f16_value_cache,
                                      0, 0, 1, 1, 1, 2,
                                      f16_attention_output)
           .is_ok());
  output = read_f32_buffer(context, f16_attention_output, 2);
  assert_close(output[0], 5.0F);
  assert_close(output[1], 6.0F);

  auto batched_query = make_f32_buffer(context, {0.0F, 0.0F, 0.0F, 0.0F});
  auto batched_key_cache = make_f32_buffer(context, {1.0F, 0.0F,
                                                     0.0F, 1.0F,
                                                     1.0F, 1.0F});
  auto batched_value_cache = make_f32_buffer(context, {5.0F, 6.0F,
                                                       7.0F, 8.0F,
                                                       9.0F, 10.0F});
  auto batched_attention_output = make_f32_buffer(context, std::vector<float>(4, 0.0F));
  assert(context.begin_graph().is_ok());
  assert(context.attention_f32_batched(batched_query, batched_key_cache,
                                       batched_value_cache, 0, 1, 2, 3,
                                       1, 1, 2, batched_attention_output)
           .is_ok());
  assert(context.commit_graph().is_ok());
  output = read_f32_buffer(context, batched_attention_output, 4);
  assert_close(output[0], 6.0F);
  assert_close(output[1], 7.0F);
  assert_close(output[2], 7.0F);
  assert_close(output[3], 8.0F);

  constexpr std::size_t tiled_attention_tokens = 16;
  std::vector<float> tiled_query(tiled_attention_tokens * 2U, 0.0F);
  std::vector<float> tiled_key_cache(tiled_attention_tokens * 2U, 0.0F);
  std::vector<float> tiled_value_cache(tiled_attention_tokens * 2U, 0.0F);
  for (std::size_t token = 0; token < tiled_attention_tokens; ++token) {
    tiled_value_cache[token * 2U] = static_cast<float>(token);
    tiled_value_cache[token * 2U + 1U] = static_cast<float>(token + 1U);
  }
  auto tiled_query_buffer = make_f32_buffer(context, tiled_query);
  auto tiled_key_cache_buffer = make_f32_buffer(context, tiled_key_cache);
  auto tiled_value_cache_buffer = make_f32_buffer(context, tiled_value_cache);
  auto tiled_attention_output =
    make_f32_buffer(context, std::vector<float>(tiled_attention_tokens * 2U, 0.0F));
  const char* previous_tiled_attention =
    std::getenv("KRAKEN_QWEN35_TILED_ATTENTION");
  const bool had_tiled_attention_env = previous_tiled_attention != nullptr;
  const std::string previous_tiled_attention_value =
    had_tiled_attention_env ? std::string{previous_tiled_attention} : std::string{};
  assert(::setenv("KRAKEN_QWEN35_TILED_ATTENTION", "1", 1) == 0);
  assert(context.attention_f32_batched(
           tiled_query_buffer, tiled_key_cache_buffer, tiled_value_cache_buffer,
           0, 0, tiled_attention_tokens, tiled_attention_tokens, 1, 1, 2,
           tiled_attention_output)
           .is_ok());
  if (had_tiled_attention_env) {
    assert(::setenv("KRAKEN_QWEN35_TILED_ATTENTION",
                    previous_tiled_attention_value.c_str(), 1) == 0);
  } else {
    assert(::unsetenv("KRAKEN_QWEN35_TILED_ATTENTION") == 0);
  }
  output = read_f32_buffer(context, tiled_attention_output,
                           tiled_attention_tokens * 2U);
  assert_close(output[0], 0.0F);
  assert_close(output[1], 1.0F);
  assert_close(output[14U * 2U], 7.0F);
  assert_close(output[14U * 2U + 1U], 8.0F);
  assert_close(output[15U * 2U], 7.5F);
  assert_close(output[15U * 2U + 1U], 8.5F);

  constexpr std::size_t real_attention_tokens = 64;
  constexpr std::size_t real_attention_start = 0;
  constexpr std::size_t real_attention_capacity =
    real_attention_start + real_attention_tokens;
  constexpr std::size_t real_attention_heads = 8;
  constexpr std::size_t real_attention_kv_heads = 2;
  constexpr std::size_t real_attention_head_dim = 256;
  constexpr std::size_t real_attention_attn_dim =
    real_attention_heads * real_attention_head_dim;
  constexpr std::size_t real_attention_kv_dim =
    real_attention_kv_heads * real_attention_head_dim;
  std::vector<float> real_query(real_attention_tokens *
                                real_attention_attn_dim);
  std::vector<float> real_key_cache(real_attention_capacity *
                                    real_attention_kv_dim);
  std::vector<float> real_value_cache(real_attention_capacity *
                                      real_attention_kv_dim);
  for (std::size_t i = 0; i < real_query.size(); ++i) {
    real_query[i] =
      static_cast<float>(static_cast<int>((i * 13U) % 31U) - 15) * 0.015F;
  }
  for (std::size_t i = 0; i < real_key_cache.size(); ++i) {
    real_key_cache[i] =
      static_cast<float>(static_cast<int>((i * 7U) % 29U) - 14) * 0.012F;
    real_value_cache[i] =
      static_cast<float>(static_cast<int>((i * 11U) % 37U) - 18) * 0.025F;
  }

  auto real_query_buffer = make_f32_buffer(context, real_query);
  auto real_key_cache_buffer = make_f32_buffer(context, real_key_cache);
  auto real_value_cache_buffer = make_f32_buffer(context, real_value_cache);
  auto default_attention_output =
    make_f32_buffer(context, std::vector<float>(real_query.size(), 0.0F));
  auto tiled_real_attention_output =
    make_f32_buffer(context, std::vector<float>(real_query.size(), 0.0F));
  auto flash_real_attention_output =
    make_f32_buffer(context, std::vector<float>(real_query.size(), 0.0F));
  previous_tiled_attention = std::getenv("KRAKEN_QWEN35_TILED_ATTENTION");
  const char* previous_tiled_attention_tile =
    std::getenv("KRAKEN_QWEN35_TILED_ATTENTION_TILE");
  const char* previous_flash_attention =
    std::getenv("KRAKEN_QWEN35_FLASH_ATTENTION");
  const bool had_real_tiled_attention_env = previous_tiled_attention != nullptr;
  const bool had_real_tiled_attention_tile_env =
    previous_tiled_attention_tile != nullptr;
  const bool had_real_flash_attention_env = previous_flash_attention != nullptr;
  const std::string previous_real_tiled_attention_value =
    had_real_tiled_attention_env ? std::string{previous_tiled_attention}
                                 : std::string{};
  const std::string previous_real_tiled_attention_tile_value =
    had_real_tiled_attention_tile_env
      ? std::string{previous_tiled_attention_tile}
      : std::string{};
  const std::string previous_real_flash_attention_value =
    had_real_flash_attention_env ? std::string{previous_flash_attention}
                                 : std::string{};
  assert(::unsetenv("KRAKEN_QWEN35_TILED_ATTENTION") == 0);
  assert(::unsetenv("KRAKEN_QWEN35_TILED_ATTENTION_TILE") == 0);
  assert(::setenv("KRAKEN_QWEN35_FLASH_ATTENTION", "0", 1) == 0);
  assert(context.attention_f32_batched(
           real_query_buffer, real_key_cache_buffer, real_value_cache_buffer,
           0, real_attention_start, real_attention_tokens,
           real_attention_capacity, real_attention_heads,
           real_attention_kv_heads, real_attention_head_dim,
           default_attention_output)
           .is_ok());
  assert(::setenv("KRAKEN_QWEN35_TILED_ATTENTION", "1", 1) == 0);
  assert(::setenv("KRAKEN_QWEN35_TILED_ATTENTION_TILE", "32", 1) == 0);
  assert(::setenv("KRAKEN_QWEN35_FLASH_ATTENTION", "0", 1) == 0);
  assert(context.attention_f32_batched(
           real_query_buffer, real_key_cache_buffer, real_value_cache_buffer,
           0, real_attention_start, real_attention_tokens,
           real_attention_capacity, real_attention_heads,
           real_attention_kv_heads, real_attention_head_dim,
           tiled_real_attention_output)
           .is_ok());
  assert(::unsetenv("KRAKEN_QWEN35_TILED_ATTENTION") == 0);
  assert(::unsetenv("KRAKEN_QWEN35_TILED_ATTENTION_TILE") == 0);
  assert(::setenv("KRAKEN_QWEN35_FLASH_ATTENTION", "1", 1) == 0);
  assert(context.attention_f32_batched(
           real_query_buffer, real_key_cache_buffer, real_value_cache_buffer,
           0, real_attention_start, real_attention_tokens,
           real_attention_capacity, real_attention_heads,
           real_attention_kv_heads, real_attention_head_dim,
           flash_real_attention_output)
           .is_ok());

  constexpr std::size_t unaligned_attention_tokens = 65;
  constexpr std::size_t unaligned_attention_heads = 2;
  constexpr std::size_t unaligned_attention_kv_heads = 1;
  constexpr std::size_t unaligned_attention_head_dim = 256;
  constexpr std::size_t unaligned_attention_attn_dim =
    unaligned_attention_heads * unaligned_attention_head_dim;
  constexpr std::size_t unaligned_attention_kv_dim =
    unaligned_attention_kv_heads * unaligned_attention_head_dim;
  std::vector<float> unaligned_query(unaligned_attention_tokens *
                                     unaligned_attention_attn_dim);
  std::vector<float> unaligned_key_cache(unaligned_attention_tokens *
                                         unaligned_attention_kv_dim);
  std::vector<float> unaligned_value_cache(unaligned_attention_tokens *
                                           unaligned_attention_kv_dim);
  for (std::size_t i = 0; i < unaligned_query.size(); ++i) {
    unaligned_query[i] =
      static_cast<float>(static_cast<int>((i * 17U) % 43U) - 21) * 0.01F;
  }
  for (std::size_t i = 0; i < unaligned_key_cache.size(); ++i) {
    unaligned_key_cache[i] =
      static_cast<float>(static_cast<int>((i * 5U) % 41U) - 20) * 0.011F;
    unaligned_value_cache[i] =
      static_cast<float>(static_cast<int>((i * 19U) % 47U) - 23) * 0.017F;
  }
  auto unaligned_query_buffer = make_f32_buffer(context, unaligned_query);
  auto unaligned_key_cache_buffer = make_f32_buffer(context,
                                                    unaligned_key_cache);
  auto unaligned_value_cache_buffer = make_f32_buffer(context,
                                                      unaligned_value_cache);
  auto unaligned_key_cache_f16_buffer =
    make_empty_buffer(context, unaligned_key_cache.size() * sizeof(std::uint16_t));
  auto unaligned_value_cache_f16_buffer =
    make_empty_buffer(context, unaligned_value_cache.size() * sizeof(std::uint16_t));
  assert(context.copy_f32_rows_to_f16(
           unaligned_key_cache_buffer, unaligned_key_cache_f16_buffer,
           unaligned_attention_tokens, unaligned_attention_kv_dim,
           unaligned_attention_kv_dim, 0, unaligned_attention_kv_dim, 0)
           .is_ok());
  assert(context.copy_f32_rows_to_f16(
           unaligned_value_cache_buffer, unaligned_value_cache_f16_buffer,
           unaligned_attention_tokens, unaligned_attention_kv_dim,
           unaligned_attention_kv_dim, 0, unaligned_attention_kv_dim, 0)
           .is_ok());
  auto unaligned_default_attention_output =
    make_f32_buffer(context, std::vector<float>(unaligned_query.size(), 0.0F));
  auto unaligned_flash_attention_output =
    make_f32_buffer(context, std::vector<float>(unaligned_query.size(), 0.0F));
  auto unaligned_f16_default_attention_output =
    make_f32_buffer(context, std::vector<float>(unaligned_query.size(), 0.0F));
  auto unaligned_f16_flash_attention_output =
    make_f32_buffer(context, std::vector<float>(unaligned_query.size(), 0.0F));
  assert(::setenv("KRAKEN_QWEN35_FLASH_ATTENTION", "0", 1) == 0);
  assert(context.attention_f32_batched(
           unaligned_query_buffer, unaligned_key_cache_buffer,
           unaligned_value_cache_buffer, 0, 0, unaligned_attention_tokens,
           unaligned_attention_tokens, unaligned_attention_heads,
           unaligned_attention_kv_heads, unaligned_attention_head_dim,
           unaligned_default_attention_output)
           .is_ok());
  assert(context.attention_f32_batched_f16_kv(
           unaligned_query_buffer, unaligned_key_cache_f16_buffer,
           unaligned_value_cache_f16_buffer, 0, 0, unaligned_attention_tokens,
           unaligned_attention_tokens, unaligned_attention_heads,
           unaligned_attention_kv_heads, unaligned_attention_head_dim,
           unaligned_f16_default_attention_output)
           .is_ok());
  assert(::setenv("KRAKEN_QWEN35_FLASH_ATTENTION", "1", 1) == 0);
  assert(context.attention_f32_batched(
           unaligned_query_buffer, unaligned_key_cache_buffer,
           unaligned_value_cache_buffer, 0, 0, unaligned_attention_tokens,
           unaligned_attention_tokens, unaligned_attention_heads,
           unaligned_attention_kv_heads, unaligned_attention_head_dim,
           unaligned_flash_attention_output)
           .is_ok());
  assert(context.attention_f32_batched_f16_kv(
           unaligned_query_buffer, unaligned_key_cache_f16_buffer,
           unaligned_value_cache_f16_buffer, 0, 0, unaligned_attention_tokens,
           unaligned_attention_tokens, unaligned_attention_heads,
           unaligned_attention_kv_heads, unaligned_attention_head_dim,
           unaligned_f16_flash_attention_output)
           .is_ok());
  if (had_real_tiled_attention_env) {
    assert(::setenv("KRAKEN_QWEN35_TILED_ATTENTION",
                    previous_real_tiled_attention_value.c_str(), 1) == 0);
  } else {
    assert(::unsetenv("KRAKEN_QWEN35_TILED_ATTENTION") == 0);
  }
  if (had_real_tiled_attention_tile_env) {
    assert(::setenv("KRAKEN_QWEN35_TILED_ATTENTION_TILE",
                    previous_real_tiled_attention_tile_value.c_str(), 1) == 0);
  } else {
    assert(::unsetenv("KRAKEN_QWEN35_TILED_ATTENTION_TILE") == 0);
  }
  if (had_real_flash_attention_env) {
    assert(::setenv("KRAKEN_QWEN35_FLASH_ATTENTION",
                    previous_real_flash_attention_value.c_str(), 1) == 0);
  } else {
    assert(::unsetenv("KRAKEN_QWEN35_FLASH_ATTENTION") == 0);
  }
  const auto default_attention =
    read_f32_buffer(context, default_attention_output, real_query.size());
  const auto tiled_attention =
    read_f32_buffer(context, tiled_real_attention_output, real_query.size());
  const auto flash_attention =
    read_f32_buffer(context, flash_real_attention_output, real_query.size());
  const auto unaligned_default_attention =
    read_f32_buffer(context, unaligned_default_attention_output,
                    unaligned_query.size());
  const auto unaligned_flash_attention =
    read_f32_buffer(context, unaligned_flash_attention_output,
                    unaligned_query.size());
  const auto unaligned_f16_default_attention =
    read_f32_buffer(context, unaligned_f16_default_attention_output,
                    unaligned_query.size());
  const auto unaligned_f16_flash_attention =
    read_f32_buffer(context, unaligned_f16_flash_attention_output,
                    unaligned_query.size());
  for (std::size_t i = 0; i < real_query.size(); ++i) {
    if (std::abs(default_attention[i] - tiled_attention[i]) >= 1e-5F) {
      std::cerr << "tiled attention mismatch at " << i
                << ": default=" << default_attention[i]
                << " tiled=" << tiled_attention[i] << '\n';
    }
    assert(std::abs(default_attention[i] - tiled_attention[i]) < 1e-5F);
    if (std::abs(default_attention[i] - flash_attention[i]) >= 1e-5F) {
      std::cerr << "flash attention mismatch at " << i
                << ": default=" << default_attention[i]
                << " flash=" << flash_attention[i] << '\n';
    }
    assert(std::abs(default_attention[i] - flash_attention[i]) < 1e-5F);
  }
  for (std::size_t i = 0; i < unaligned_query.size(); ++i) {
    if (std::abs(unaligned_default_attention[i] -
                 unaligned_flash_attention[i]) >= 1e-5F) {
      std::cerr << "unaligned flash attention mismatch at " << i
                << ": default=" << unaligned_default_attention[i]
                << " flash=" << unaligned_flash_attention[i] << '\n';
    }
    assert(std::abs(unaligned_default_attention[i] -
                    unaligned_flash_attention[i]) < 1e-5F);
    if (std::abs(unaligned_default_attention[i] -
                 unaligned_f16_flash_attention[i]) >= 5e-3F) {
      std::cerr << "unaligned F16 flash attention mismatch at " << i
                << ": default=" << unaligned_default_attention[i]
                << " f16_flash=" << unaligned_f16_flash_attention[i]
                << '\n';
    }
    assert(std::abs(unaligned_default_attention[i] -
                    unaligned_f16_flash_attention[i]) < 5e-3F);
    if (std::abs(unaligned_f16_default_attention[i] -
                 unaligned_f16_flash_attention[i]) >= 5e-3F) {
      std::cerr << "unaligned F16 flash self mismatch at " << i
                << ": f16_default=" << unaligned_f16_default_attention[i]
                << " f16_flash=" << unaligned_f16_flash_attention[i]
                << '\n';
    }
    assert(std::abs(unaligned_f16_default_attention[i] -
                    unaligned_f16_flash_attention[i]) < 5e-3F);
  }
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

void test_qwen35_matmul_bench_format() {
  toyllm::Qwen35MatmulBenchResult result;
  result.gguf_path = "models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf";
  result.tensor_name = "blk.0.attn_qkv.weight";
  result.type_name = "Q5_K";
  result.dispatch = "mul_mm_simd_64x32";
  result.type = 13;
  result.tokens = 1024;
  result.rows = 6144;
  result.cols = 1024;
  result.warmup_iterations = 1;
  result.timed_iterations = 3;
  result.output_values_per_iteration = 6291456;
  result.total_seconds = 0.03;
  result.average_milliseconds = 10.0;
  result.token_iterations_per_second = 102400.0;
  result.output_megavalues_per_second = 629.0;
  result.sample_checksum = 1.25;

  const auto summary = toyllm::format_qwen35_matmul_bench_result(result);
  assert(summary.find("Qwen3.5 Metal matmul bench: ok") != std::string::npos);
  assert(summary.find("blk.0.attn_qkv.weight") != std::string::npos);
  assert(summary.find("Dispatch: mul_mm_simd_64x32") != std::string::npos);
  assert(summary.find("rows=6144") != std::string::npos);
  assert(summary.find("Output Mvalues/s") != std::string::npos);
}

void test_qwen35_gdn_bench_format() {
  toyllm::Qwen35GdnBenchResult result;
  result.dispatch = "qwen35_simdgroup_4rows";
  result.tokens = 1024;
  result.key_heads = 16;
  result.value_heads = 16;
  result.head_dim = 128;
  result.warmup_iterations = 2;
  result.timed_iterations = 10;
  result.state_values = 262144;
  result.output_values_per_iteration = 2097152;
  result.total_seconds = 0.1;
  result.average_milliseconds = 10.0;
  result.token_iterations_per_second = 102400.0;
  result.output_megavalues_per_second = 209.0;
  result.sample_checksum = 0.5;

  const auto summary = toyllm::format_qwen35_gdn_bench_result(result);
  assert(summary.find("Qwen3.5 Metal GDN bench: ok") != std::string::npos);
  assert(summary.find("Dispatch: qwen35_simdgroup_4rows") != std::string::npos);
  assert(summary.find("tokens=1024") != std::string::npos);
  assert(summary.find("State values: 262144") != std::string::npos);
}

void test_qwen35_attention_bench_format() {
  toyllm::Qwen35AttentionBenchResult result;
  result.dispatch = "flash256_f16_kv+tail";
  result.tokens = 760;
  result.start_position = 10240;
  result.capacity_tokens = 11000;
  result.key_count = 11000;
  result.heads = 8;
  result.kv_heads = 2;
  result.head_dim = 256;
  result.warmup_iterations = 2;
  result.timed_iterations = 10;
  result.cache_values = 5632000;
  result.output_values_per_iteration = 1556480;
  result.total_seconds = 0.02;
  result.average_milliseconds = 2.0;
  result.token_iterations_per_second = 380000.0;
  result.output_megavalues_per_second = 778.0;
  result.sample_checksum = 0.25;
  result.f16_kv = true;

  const auto summary = toyllm::format_qwen35_attention_bench_result(result);
  assert(summary.find("Qwen3.5 Metal attention bench: ok") != std::string::npos);
  assert(summary.find("Dispatch: flash256_f16_kv+tail") != std::string::npos);
  assert(summary.find("KV cache: f16") != std::string::npos);
  assert(summary.find("start_position=10240") != std::string::npos);
  assert(summary.find("key_count=11000") != std::string::npos);
}

void test_qwen35_gguf_model_config() {
  const std::filesystem::path model_dir{"models/qwen3.5-0.8b"};
  const auto model_file = model_dir / "Qwen3.5-0.8B-Q4_K_M.gguf";
  if (!std::filesystem::exists(model_file)) {
    return;
  }

  auto bundle = toyllm::load_model_bundle(model_dir);
  assert(bundle.is_ok());
  assert(bundle.value().model.gguf);
  assert(bundle.value().model_file == model_file);
  assert(bundle.value().model.architecture == "qwen35");
  assert(bundle.value().model.hidden_size == 1024);
  assert(bundle.value().model.num_hidden_layers == 24);
  assert(bundle.value().model.main_layer_count == 24);
  assert(bundle.value().model.vocab_size == 248320);
  assert(bundle.value().tokenizer.available);
  assert(bundle.value().tokenizer.model == "gpt2");
  assert(bundle.value().tokenizer.pre == "qwen35");

  const auto gguf = toyllm::read_gguf_file(model_file);
  assert(gguf.is_ok());
  const auto weight_map = toyllm::build_qwen35_weight_map(bundle.value().model, gguf.value());
  assert(weight_map.is_ok());
  assert(weight_map.value().layers.size() == 24);
  assert(weight_map.value().linear_attention_layers == 18);
  assert(weight_map.value().full_attention_layers == 6);
  assert(weight_map.value().mtp_layers == 0);
  assert(weight_map.value().output_tied_to_token_embedding);
  assert(weight_map.value().layers[0].kind == toyllm::Qwen35LayerKind::linear_attention);
  assert(weight_map.value().layers[3].kind == toyllm::Qwen35LayerKind::full_attention);
  assert(weight_map.value().layers[23].kind == toyllm::Qwen35LayerKind::full_attention);
  toyllm::Qwen35RuntimeOptions plan_options;
  plan_options.decode_tokens = 1;
  plan_options.prefill_chunk_tokens = 512;
  const auto plan = toyllm::build_qwen35_execution_plan(bundle.value().model,
                                                        weight_map.value(), 11000,
                                                        plan_options);
  assert(plan.is_ok());
  assert(plan.value().main_layers == 24);
  assert(plan.value().cache.full_attention_layers == 6);
  assert(plan.value().cache.linear_attention_layers == 18);
  assert(plan.value().cache.kv_dim == 512);
  assert(plan.value().cache.recurrent_r_elements_per_layer == 18432);
  assert(plan.value().cache.recurrent_s_elements_per_layer == 262144);
  assert(plan.value().cache.attention_capacity_tokens == 11001);
  assert(plan.value().prefill.chunk_count == 22);
  assert(plan.value().prefill.final_chunk_tokens == 248);
  assert(plan.value().prefill.output_only_last_token);
  assert(plan.value().cache.kv_cache_bytes == 270360576ULL);
  assert(plan.value().cache.recurrent_r_cache_bytes == 1327104ULL);
  assert(plan.value().cache.recurrent_s_cache_bytes == 18874368ULL);
  assert(plan.value().cache.total_cache_bytes == 290562048ULL);
  const auto plan_summary = toyllm::format_qwen35_execution_plan(plan.value());
  assert(plan_summary.find("Qwen3.5 execution plan: ok") != std::string::npos);
  assert(plan_summary.find("prompt_tokens=11000") != std::string::npos);
  const auto* output_norm_tensor = toyllm::find_gguf_tensor(gguf.value(), "output_norm.weight");
  assert(output_norm_tensor != nullptr);
  assert(output_norm_tensor->type == 0);
  assert(output_norm_tensor->byte_size == 4096);
  const auto mapped_data = toyllm::GgufMappedData::open(gguf.value());
  assert(mapped_data.is_ok());
  assert(mapped_data.value().valid());
  const auto output_norm_bytes = mapped_data.value().tensor_bytes(*output_norm_tensor);
  assert(output_norm_bytes.is_ok());
  assert(output_norm_bytes.value().size == output_norm_tensor->byte_size);
  assert(output_norm_bytes.value().data != nullptr);

  const auto tokenizer = toyllm::load_gguf_tokenizer(gguf.value());
  assert(tokenizer.is_ok());
  assert(tokenizer.value().model == "gpt2");
  assert(tokenizer.value().pre == "qwen35");
  assert(tokenizer.value().tokens.size() == 248320);
  assert(!tokenizer.value().merges.empty());
  assert(!tokenizer.value().chat_template.empty());
  const auto im_start = toyllm::gguf_token_id(tokenizer.value(), "<|im_start|>");
  const auto im_end = toyllm::gguf_token_id(tokenizer.value(), "<|im_end|>");
  assert(im_start.has_value());
  assert(im_end.has_value());
  assert(tokenizer.value().eos_token_id == *im_end);
  assert(toyllm::gguf_token_is_control(tokenizer.value(), *im_start));
  assert(toyllm::gguf_token_is_control(tokenizer.value(), *im_end));
  const auto raw_controls =
    toyllm::gguf_decode_token_text(tokenizer.value(), {*im_start, *im_end}, false);
  assert(raw_controls.is_ok());
  assert(raw_controls.value() == "<|im_start|><|im_end|>");
  const auto skipped_controls =
    toyllm::gguf_decode_token_text(tokenizer.value(), {*im_start, *im_end}, true);
  assert(skipped_controls.is_ok());
  assert(skipped_controls.value().empty());
  const auto chat_prompt = toyllm::format_qwen35_chat_prompt(
    tokenizer.value(), {toyllm::ChatMessage{"user", "hello"}}, true, false);
  assert(chat_prompt.is_ok());
  assert(chat_prompt.value() ==
         "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
         "<think>\n\n</think>\n\n");
  const auto thinking_chat_prompt = toyllm::format_qwen35_chat_prompt(
    tokenizer.value(), {toyllm::ChatMessage{"user", "hello"}}, true, true);
  assert(thinking_chat_prompt.is_ok());
  assert(thinking_chat_prompt.value() ==
         "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
         "<think>\n");
  const auto hello_tokens = toyllm::gguf_encode_text(tokenizer.value(), "hello", false, false);
  assert(hello_tokens.is_ok());
  assert((hello_tokens.value() == std::vector<std::int64_t>{14556}));
  const auto hello_text =
    toyllm::gguf_decode_token_text(tokenizer.value(), hello_tokens.value(), true);
  assert(hello_text.is_ok());
  assert(hello_text.value() == "hello");
  const auto hello_world_tokens =
    toyllm::gguf_encode_text(tokenizer.value(), "Hello world", false, false);
  assert(hello_world_tokens.is_ok());
  assert((hello_world_tokens.value() == std::vector<std::int64_t>{9419, 1814}));
  const auto digit_tokens = toyllm::gguf_encode_text(tokenizer.value(), "abc123", false, false);
  assert(digit_tokens.is_ok());
  assert((digit_tokens.value() == std::vector<std::int64_t>{13290, 16, 17, 18}));
  const auto chat_tokens =
    toyllm::gguf_encode_text(tokenizer.value(), chat_prompt.value(), false, true);
  assert(chat_tokens.is_ok());
  assert((chat_tokens.value() ==
          std::vector<std::int64_t>{248045, 846, 198, 14556, 248046, 198,
                                    248045, 74455, 198, 248068, 271,
                                    248069, 271}));
  const auto thinking_chat_tokens =
    toyllm::gguf_encode_text(tokenizer.value(), thinking_chat_prompt.value(), false, true);
  assert(thinking_chat_tokens.is_ok());
  assert((thinking_chat_tokens.value() ==
          std::vector<std::int64_t>{248045, 846, 198, 14556, 248046, 198,
                                    248045, 74455, 198, 248068, 198}));
  const auto chat_tokens_via_helper =
    toyllm::gguf_encode_qwen35_chat_prompt(
      tokenizer.value(), {toyllm::ChatMessage{"user", "hello"}}, true, false);
  assert(chat_tokens_via_helper.is_ok());
  assert(chat_tokens_via_helper.value() == chat_tokens.value());

  const auto summary = toyllm::format_model_summary(bundle.value());
  assert(summary.find("Architecture: qwen35") != std::string::npos);
  assert(summary.find("Tokenizer pre: qwen35") != std::string::npos);
  assert(summary.find("Validation: ok") != std::string::npos);

  const auto weights = toyllm::format_weight_summary(model_dir);
  assert(weights.is_ok());
  assert(weights.value().find("Format: GGUF v3") != std::string::npos);
  assert(weights.value().find("Native Qwen3.5 weight map: ok") != std::string::npos);
  assert(weights.value().find("Qwen3.5 GGUF mapping: ok") != std::string::npos);
  assert(weights.value().find("- Q4_K:") != std::string::npos);
}

void test_qwen35_gguf_generation_uses_native_runtime() {
  const std::filesystem::path model_dir{"models/qwen3.5-0.8b"};
  if (!std::filesystem::exists(model_dir / "Qwen3.5-0.8B-Q4_K_M.gguf")) {
    return;
  }

  toyllm::CpuGenerationRequest request;
  request.model_dir = model_dir;
  request.prompt = "hello";
  request.max_new_tokens = 2;
  request.compute_device = toyllm::Device::mps();

  const auto result = toyllm::generate_cpu(request);
  assert(result.is_ok());
  assert(result.value().implemented);
  assert(result.value().prompt_tokens == 1);
  assert(result.value().generated_tokens == 2);
  assert(result.value().finish_reason == "length");
  assert(result.value().kv_cache.available);
  assert(result.value().kv_cache.layers == 6);
  assert(result.value().kv_cache.kv_heads == 2);
  assert(result.value().kv_cache.head_dim == 256);
  assert(result.value().kv_cache.kv_dim == 512);
  assert(result.value().kv_cache.capacity_tokens == 3);
  assert(result.value().kv_cache.used_tokens == 2);

  toyllm::CpuGenerationRequest logits_request;
  logits_request.model_dir = model_dir;
  logits_request.prompt = "hello";
  logits_request.max_new_tokens = 1;
  logits_request.logits_top_k = 3;
  logits_request.compute_device = toyllm::Device::mps();
  const auto logits_result = toyllm::generate_cpu(logits_request);
  assert(logits_result.is_ok());
  assert(logits_result.value().generated_tokens == 1);
  assert(logits_result.value().logits_top.size() == 3);
  assert(logits_result.value().logits_top[0].token_id == 11);
  assert(logits_result.value().logits_top[0].logit > 12.0F);

  const std::string uk_question =
    "\xE7" "\xAE" "\x80" "\xE5" "\x8D" "\x95" "\xE4" "\xBB"
    "\x8B" "\xE7" "\xBB" "\x8D" "\xE4" "\xB8" "\x80" "\xE4"
    "\xB8" "\x8B" "\xE8" "\x8B" "\xB1" "\xE5" "\x9B" "\xBD";
  toyllm::CpuGenerationRequest cached_decode_request;
  cached_decode_request.model_dir = model_dir;
  cached_decode_request.prompt =
    std::string{"<|im_start|>user\n"} + uk_question +
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
  cached_decode_request.max_new_tokens = 2;
  cached_decode_request.parse_special_prompt = true;
  cached_decode_request.compute_device = toyllm::Device::mps();
  const auto cached_decode = toyllm::generate_cpu(cached_decode_request);
  assert(cached_decode.is_ok());
  assert(cached_decode.value().prompt_tokens == 15);
  assert(cached_decode.value().generated_tokens == 2);
  assert(cached_decode.value().text ==
         "\xE8" "\x8B" "\xB1" "\xE5" "\x9B" "\xBD" "\xEF" "\xBC" "\x88");

  request.sampling.do_sample = true;
  request.sampling.top_k_set = true;
  request.sampling.top_k = 1;
  request.sampling.seed_set = true;
  request.sampling.seed = 1;
  const auto sampled = toyllm::generate_cpu(request);
  assert(sampled.is_ok());
  assert(sampled.value().implemented);
  assert(sampled.value().generated_tokens == 2);
  assert(sampled.value().finish_reason == "length");
  assert(sampled.value().text == result.value().text);
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

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::filesystem::path unique_temp_dir(std::string_view name) {
  static std::uint64_t counter = 0;
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto suffix = std::to_string(ticks) + "-" + std::to_string(counter++);
  return std::filesystem::temp_directory_path() /
         std::filesystem::path{std::string{name} + "-" + suffix};
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
  const auto model_dir = unique_temp_dir(name);
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

std::filesystem::path create_tiny_forward_model_dir(std::string_view name) {
  const auto model_dir = unique_temp_dir(name);
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
      "head_dim": 2,
      "hidden_act": "silu",
      "hidden_size": 4,
      "initializer_range": 0.02,
      "intermediate_size": 5,
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

std::filesystem::path create_tiny_runtime_model_dir(std::string_view name) {
  const auto model_dir = unique_temp_dir(name);
  std::filesystem::remove_all(model_dir);
  std::filesystem::create_directories(model_dir);

  write_text_file(
    model_dir / "config.json",
    R"({
      "architectures": ["Qwen3ForCausalLM"],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "bos_token_id": 0,
      "eos_token_id": 151643,
      "head_dim": 2,
      "hidden_act": "silu",
      "hidden_size": 4,
      "initializer_range": 0.02,
      "intermediate_size": 5,
      "max_position_embeddings": 32,
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
      "vocab_size": 151669
    })");
  write_text_file(
    model_dir / "generation_config.json",
    R"({"bos_token_id":0,"do_sample":false,"eos_token_id":[151643],"pad_token_id":0,"temperature":1.0,"top_k":0,"top_p":1.0})");
  write_text_file(
    model_dir / "tokenizer.json",
    R"({"model":{"vocab":{"a":0,"b":1,"c":2,"d":3,"h":4,"e":5,"l":6,"o":7,"\n":8,"s":9,"r":10,"u":11,"t":12,"n":13,"i":14}},"added_tokens":[{"id":151643,"content":"<|endoftext|>"},{"id":151644,"content":"<|im_start|>"},{"id":151645,"content":"<|im_end|>"},{"id":151667,"content":"<think>"},{"id":151668,"content":"</think>"}]})");
  write_text_file(
    model_dir / "tokenizer_config.json",
    R"({"added_tokens_decoder":{"151643":{"content":"<|endoftext|>"},"151644":{"content":"<|im_start|>"},"151645":{"content":"<|im_end|>"},"151667":{"content":"<think>"},"151668":{"content":"</think>"}}})");
  write_text_file(
    model_dir / "vocab.json",
    R"({"a":0,"b":1,"c":2,"d":3,"h":4,"e":5,"l":6,"o":7,"\n":8,"s":9,"r":10,"u":11,"t":12,"n":13,"i":14})");
  write_text_file(model_dir / "merges.txt", "#version: 0.2\n");
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

void write_tiny_qwen_mpsgraph_forward_safetensors(const std::filesystem::path& path) {
  write_bf16_safetensors(
    path,
    {
      {"model.embed_tokens.weight", {4, 4}, {0.0F, 0.0F, 0.0F, 0.0F,
                                             1.0F, 1.0F, 1.0F, 1.0F,
                                             2.0F, 2.0F, 2.0F, 2.0F,
                                             3.0F, 3.0F, 3.0F, 3.0F}},
      {"lm_head.weight", {4, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                  0.0F, 1.0F, 0.0F, 0.0F,
                                  0.0F, 0.0F, 1.0F, 0.0F,
                                  0.0F, 0.0F, 0.0F, 1.0F}},
      {"model.norm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}},
      {"model.layers.0.input_layernorm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}},
      {"model.layers.0.post_attention_layernorm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}},
      {"model.layers.0.self_attn.q_proj.weight", {4, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F,
                                                          0.0F, 0.0F, 1.0F, 0.0F,
                                                          0.0F, 0.0F, 0.0F, 1.0F}},
      {"model.layers.0.self_attn.k_proj.weight", {2, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F}},
      {"model.layers.0.self_attn.v_proj.weight", {2, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F}},
      {"model.layers.0.self_attn.o_proj.weight", {4, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F,
                                                          0.0F, 0.0F, 1.0F, 0.0F,
                                                          0.0F, 0.0F, 0.0F, 1.0F}},
      {"model.layers.0.self_attn.q_norm.weight", {2}, {1.0F, 1.0F}},
      {"model.layers.0.self_attn.k_norm.weight", {2}, {1.0F, 1.0F}},
      {"model.layers.0.mlp.gate_proj.weight", {5, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 1.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F}},
      {"model.layers.0.mlp.up_proj.weight", {5, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 1.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F}},
      {"model.layers.0.mlp.down_proj.weight", {4, 5}, {1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F}},
    });
}

std::vector<float> make_zero_values(std::size_t count) {
  return std::vector<float>(count, 0.0F);
}

void set_row(std::vector<float>& values, std::size_t row, std::size_t cols,
             const std::array<float, 4>& row_values) {
  assert(cols == row_values.size());
  const auto offset = row * cols;
  assert(offset + cols <= values.size());
  for (std::size_t col = 0; col < cols; ++col) {
    values[offset + col] = row_values[col];
  }
}

void write_tiny_qwen_mpsgraph_runtime_safetensors(const std::filesystem::path& path) {
  constexpr std::uint64_t kVocab = 151669;
  constexpr std::uint64_t kHidden = 4;
  auto embeddings = make_zero_values(static_cast<std::size_t>(kVocab * kHidden));
  auto lm_head = make_zero_values(static_cast<std::size_t>(kVocab * kHidden));
  set_row(embeddings, 0, kHidden, {1.0F, 1.0F, 1.0F, 1.0F});
  set_row(embeddings, static_cast<std::size_t>(toyllm::kQwenImStart), kHidden,
          {1.0F, 1.0F, 1.0F, 1.0F});
  set_row(embeddings, static_cast<std::size_t>(toyllm::kQwenImEnd), kHidden,
          {1.0F, 1.0F, 1.0F, 1.0F});
  set_row(embeddings, static_cast<std::size_t>(toyllm::kQwenThinkStart), kHidden,
          {1.0F, 1.0F, 1.0F, 1.0F});
  set_row(embeddings, static_cast<std::size_t>(toyllm::kQwenThinkEnd), kHidden,
          {1.0F, 1.0F, 1.0F, 1.0F});
  set_row(lm_head, 0, kHidden, {10.0F, 10.0F, 10.0F, 10.0F});

  write_bf16_safetensors(
    path,
    {
      {"model.embed_tokens.weight", {kVocab, kHidden}, std::move(embeddings)},
      {"lm_head.weight", {kVocab, kHidden}, std::move(lm_head)},
      {"model.norm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}},
      {"model.layers.0.input_layernorm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}},
      {"model.layers.0.post_attention_layernorm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}},
      {"model.layers.0.self_attn.q_proj.weight", {4, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F,
                                                          0.0F, 0.0F, 1.0F, 0.0F,
                                                          0.0F, 0.0F, 0.0F, 1.0F}},
      {"model.layers.0.self_attn.k_proj.weight", {2, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F}},
      {"model.layers.0.self_attn.v_proj.weight", {2, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F}},
      {"model.layers.0.self_attn.o_proj.weight", {4, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F,
                                                          0.0F, 0.0F, 1.0F, 0.0F,
                                                          0.0F, 0.0F, 0.0F, 1.0F}},
      {"model.layers.0.self_attn.q_norm.weight", {2}, {1.0F, 1.0F}},
      {"model.layers.0.self_attn.k_norm.weight", {2}, {1.0F, 1.0F}},
      {"model.layers.0.mlp.gate_proj.weight", {5, 4}, {0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F}},
      {"model.layers.0.mlp.up_proj.weight", {5, 4}, {0.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F}},
      {"model.layers.0.mlp.down_proj.weight", {4, 5}, {0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F}},
    });
}

void write_tiny_qwen_mpsgraph_eos_runtime_safetensors(const std::filesystem::path& path) {
  constexpr std::uint64_t kVocab = 151669;
  constexpr std::uint64_t kHidden = 4;
  auto embeddings = make_zero_values(static_cast<std::size_t>(kVocab * kHidden));
  auto lm_head = make_zero_values(static_cast<std::size_t>(kVocab * kHidden));
  set_row(embeddings, 0, kHidden, {1.0F, 1.0F, 1.0F, 1.0F});
  set_row(embeddings, static_cast<std::size_t>(toyllm::kQwenImStart), kHidden,
          {1.0F, 1.0F, 1.0F, 1.0F});
  set_row(embeddings, static_cast<std::size_t>(toyllm::kQwenImEnd), kHidden,
          {1.0F, 1.0F, 1.0F, 1.0F});
  set_row(embeddings, static_cast<std::size_t>(toyllm::kQwenThinkStart), kHidden,
          {1.0F, 1.0F, 1.0F, 1.0F});
  set_row(embeddings, static_cast<std::size_t>(toyllm::kQwenThinkEnd), kHidden,
          {1.0F, 1.0F, 1.0F, 1.0F});
  set_row(lm_head, static_cast<std::size_t>(toyllm::kQwenEndOfText), kHidden,
          {10.0F, 10.0F, 10.0F, 10.0F});

  write_bf16_safetensors(
    path,
    {
      {"model.embed_tokens.weight", {kVocab, kHidden}, std::move(embeddings)},
      {"lm_head.weight", {kVocab, kHidden}, std::move(lm_head)},
      {"model.norm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}},
      {"model.layers.0.input_layernorm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}},
      {"model.layers.0.post_attention_layernorm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}},
      {"model.layers.0.self_attn.q_proj.weight", {4, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F,
                                                          0.0F, 0.0F, 1.0F, 0.0F,
                                                          0.0F, 0.0F, 0.0F, 1.0F}},
      {"model.layers.0.self_attn.k_proj.weight", {2, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F}},
      {"model.layers.0.self_attn.v_proj.weight", {2, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F}},
      {"model.layers.0.self_attn.o_proj.weight", {4, 4}, {1.0F, 0.0F, 0.0F, 0.0F,
                                                          0.0F, 1.0F, 0.0F, 0.0F,
                                                          0.0F, 0.0F, 1.0F, 0.0F,
                                                          0.0F, 0.0F, 0.0F, 1.0F}},
      {"model.layers.0.self_attn.q_norm.weight", {2}, {1.0F, 1.0F}},
      {"model.layers.0.self_attn.k_norm.weight", {2}, {1.0F, 1.0F}},
      {"model.layers.0.mlp.gate_proj.weight", {5, 4}, {0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F}},
      {"model.layers.0.mlp.up_proj.weight", {5, 4}, {0.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F,
                                                     0.0F, 0.0F, 0.0F, 0.0F}},
      {"model.layers.0.mlp.down_proj.weight", {4, 5}, {0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F}},
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

void test_qwen_mpsgraph_model_all_weight_load() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  auto model_dir = create_tiny_model_dir("kraken-infer-qwen-mpsgraph-all-weight-smoke");
  write_tiny_qwen_mpsgraph_safetensors(model_dir / "model.safetensors");

  auto model = toyllm::mpsgraph::QwenMpsGraphModel::load_all_weights(model_dir, context);
  assert(model.is_ok());
  assert(model.value().all_weights_uploaded());
  assert(model.value().info().core_weights_uploaded);
  assert(model.value().info().lm_head_uploaded);
  assert(model.value().info().layer_weights_uploaded);
  assert(model.value().info().uploaded_layer_count == 1);
  assert(model.value().info().device_tensor_count == 14);
  assert(model.value().info().device_weight_bytes == 216);

  std::error_code ec;
  std::filesystem::remove_all(model_dir, ec);
}

void test_qwen_mpsgraph_model_forward_token() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  auto model_dir = create_tiny_forward_model_dir("kraken-infer-qwen-mpsgraph-forward-smoke");
  write_tiny_qwen_mpsgraph_forward_safetensors(model_dir / "model.safetensors");

  auto model = toyllm::mpsgraph::QwenMpsGraphModel::load_all_weights(model_dir, context);
  assert(model.is_ok());
  auto state = model.value().create_run_state(context, 1);
  assert(state.is_ok());
  assert(model.value().forward_token(context, 1, 0, state.value()).is_ok());
  assert(model.value().greedy_next_token(context, state.value()).is_ok());

  const auto hidden = model.value().debug_forward_token(context, 1, 0, state.value());
  assert(hidden.is_ok());
  assert(hidden.value().size() == 4);

  const auto eps = 1e-6F;
  const auto one_norm = 1.0F / std::sqrt(1.0F + eps);
  const auto qk_norm = one_norm / std::sqrt(one_norm * one_norm + eps);
  const auto attention_residual = 1.0F + one_norm;
  const auto post_mean_square = attention_residual * attention_residual;
  const auto post_norm = attention_residual / std::sqrt(post_mean_square + eps);
  const auto silu_post_norm = post_norm / (1.0F + std::exp(-post_norm));
  const auto residual = attention_residual + silu_post_norm * post_norm;
  const auto mean_square =
    (2.0F * residual * residual + 2.0F * attention_residual * attention_residual) /
    4.0F;
  const auto scale = 1.0F / std::sqrt(mean_square + eps);
  assert(std::abs(hidden.value()[0] - residual * scale) < 1e-4F);
  assert(std::abs(hidden.value()[1] - residual * scale) < 1e-4F);
  assert(std::abs(hidden.value()[2] - attention_residual * scale) < 1e-4F);
  assert(std::abs(hidden.value()[3] - attention_residual * scale) < 1e-4F);
  assert(state.value().kv_cache.stats().used_tokens == 1);

  const auto key_cache = read_mpsgraph_f32_buffer(context, state.value().kv_cache.key_buffer(), 2);
  const auto value_cache =
    read_mpsgraph_f32_buffer(context, state.value().kv_cache.value_buffer(), 2);
  assert(std::abs(key_cache[0] - qk_norm) < 1e-4F);
  assert(std::abs(key_cache[1] - qk_norm) < 1e-4F);
  assert(std::abs(value_cache[0] - one_norm) < 1e-4F);
  assert(std::abs(value_cache[1] - one_norm) < 1e-4F);

  const auto next_token = model.value().debug_greedy_next_token(context, state.value());
  assert(next_token.is_ok());
  assert(next_token.value() == 0);

  std::error_code ec;
  std::filesystem::remove_all(model_dir, ec);
}

void test_qwen_mpsgraph_model_prefill_token_ids() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }
  auto context = std::move(context_result.value());

  auto model_dir = create_tiny_forward_model_dir("kraken-infer-qwen-mpsgraph-prefill-smoke");
  write_tiny_qwen_mpsgraph_forward_safetensors(model_dir / "model.safetensors");

  auto model = toyllm::mpsgraph::QwenMpsGraphModel::load_all_weights(model_dir, context);
  assert(model.is_ok());
  auto state = model.value().create_run_state(context, 2);
  assert(state.is_ok());
  assert(model.value().prefill_token_ids(context, {1, 2}, state.value()).is_ok());
  assert(state.value().kv_cache.stats().used_tokens == 2);

  const auto next_token = model.value().debug_greedy_next_token(context, state.value());
  assert(next_token.is_ok());
  assert(next_token.value() >= 0);
  assert(next_token.value() < 4);

  std::error_code ec;
  std::filesystem::remove_all(model_dir, ec);
}

void test_mpsgraph_generation_initializes_without_fallback() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }

  auto model_dir = create_tiny_runtime_model_dir("kraken-infer-mpsgraph-runtime-smoke");
  write_tiny_qwen_mpsgraph_runtime_safetensors(model_dir / "model.safetensors");

  toyllm::CpuGenerationRequest request;
  request.compute_device = toyllm::Device::mpsgraph();
  request.model_dir = model_dir;
  request.prompt = "hello";
  request.max_new_tokens = 2;
  const auto profile_root =
    std::filesystem::temp_directory_path() / "kraken-infer-mpsgraph-profile-smoke";
  std::error_code ec;
  std::filesystem::remove_all(profile_root, ec);
  request.observability.profile_mode = toyllm::ProfileMode::summary;
  request.observability.profile_output_dir = profile_root;
  request.observability.request_id = "mpsgraph-profile-smoke";

  const auto result = toyllm::generate_cpu(request);
  assert(result.is_ok());
  assert(result.value().implemented);
  assert(result.value().text == "aa");
  assert(result.value().finish_reason == "length");
  assert(result.value().generated_tokens == 2);
  assert(result.value().prompt_tokens > 0);
  assert(result.value().kv_cache.available);
  assert(result.value().kv_cache.used_tokens > 0);
  assert(std::filesystem::exists(result.value().profile_dir / "summary.json"));
  const auto summary_json = read_text_file(result.value().profile_dir / "summary.json");
  assert(summary_json.find("mpsgraph.load_weights") != std::string::npos);
  assert(summary_json.find("mpsgraph.decode.argmax") != std::string::npos);
  assert(summary_json.find("mpsgraph.decode.update_generation_status") != std::string::npos);
  assert(summary_json.find("mpsgraph.layer.full") != std::string::npos);
  assert(summary_json.find("mpsgraph.logits.lm_head") != std::string::npos);
  assert(summary_json.find("mpsgraph.decode.read_generation_status") != std::string::npos);
  assert(summary_json.find("mpsgraph.final_readback.generated_ids") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_d2h_calls\":\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_finish_reason\":\"length\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_decode_steps\":\"2\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_status_readbacks\":\"2\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_early_break\":\"true\"") == std::string::npos);
  assert(summary_json.find("\"mpsgraph_early_break_mode\":\"host_status_poll\"") !=
         std::string::npos);
  assert(summary_json.find("\"mpsgraph_graph_build_calls\":\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_graph_execute_calls\":\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_graph_compile_calls\":\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_executable_cache_hit_count\":\"") !=
         std::string::npos);
  assert(summary_json.find("\"mpsgraph_executable_cache_miss_count\":\"") !=
         std::string::npos);
  assert(summary_json.find("\"mpsgraph_executable_cache_entry_count\":\"") !=
         std::string::npos);

  auto cpu_request = request;
  cpu_request.compute_device = toyllm::Device::cpu();
  cpu_request.observability.profile_mode = toyllm::ProfileMode::off;
  const auto cpu_result = toyllm::generate_cpu(cpu_request);
  assert(cpu_result.is_ok());
  assert(cpu_result.value().text == result.value().text);
  assert(cpu_result.value().finish_reason == "length");
  assert(cpu_result.value().generated_tokens == 2);

  request.sampling.do_sample = true;
  const auto sampled = toyllm::generate_cpu(request);
  assert(!sampled.is_ok());
  assert(sampled.status().code() == toyllm::StatusCode::unavailable);
  assert(sampled.status().message().find("greedy") != std::string::npos);

  std::filesystem::remove_all(profile_root, ec);
  std::filesystem::remove_all(model_dir, ec);
}

void test_mpsgraph_generation_reuses_runtime_cache() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }

  auto model_dir = create_tiny_runtime_model_dir("kraken-infer-mpsgraph-cache-smoke");
  write_tiny_qwen_mpsgraph_runtime_safetensors(model_dir / "model.safetensors");

  const auto profile_root =
    std::filesystem::temp_directory_path() / "kraken-infer-mpsgraph-cache-profile-smoke";
  std::error_code ec;
  std::filesystem::remove_all(profile_root, ec);

  toyllm::CpuGenerationRequest request;
  request.compute_device = toyllm::Device::mpsgraph();
  request.model_dir = model_dir;
  request.prompt = "hello";
  request.max_new_tokens = 1;
  request.observability.profile_mode = toyllm::ProfileMode::summary;
  request.observability.profile_output_dir = profile_root;

  request.observability.request_id = "mpsgraph-cache-cold-smoke";
  const auto cold = toyllm::generate_cpu(request);
  assert(cold.is_ok());
  assert(cold.value().text == "a");
  const auto cold_summary = read_text_file(cold.value().profile_dir / "summary.json");
  assert(cold_summary.find("\"mpsgraph_model_cache\":\"miss\"") != std::string::npos);
  assert(cold_summary.find("mpsgraph.load_weights") != std::string::npos);

  request.observability.request_id = "mpsgraph-cache-warm-smoke";
  const auto warm = toyllm::generate_cpu(request);
  assert(warm.is_ok());
  assert(warm.value().text == "a");
  const auto warm_summary = read_text_file(warm.value().profile_dir / "summary.json");
  assert(warm_summary.find("\"mpsgraph_model_cache\":\"hit\"") != std::string::npos);
  assert(warm_summary.find("mpsgraph.load_weights") == std::string::npos);
  assert(warm_summary.find("mpsgraph.create_context") == std::string::npos);
  assert(warm_summary.find("\"mpsgraph_graph_execute_calls\":\"") != std::string::npos);
  assert(warm_summary.find("\"mpsgraph_executable_cache_miss_count\":\"") !=
         std::string::npos);

  std::filesystem::remove_all(profile_root, ec);
  std::filesystem::remove_all(model_dir, ec);
}

void test_mpsgraph_generation_device_side_eos_status() {
  auto context_result = toyllm::mpsgraph::MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return;
  }

  auto model_dir = create_tiny_runtime_model_dir("kraken-infer-mpsgraph-eos-smoke");
  write_tiny_qwen_mpsgraph_eos_runtime_safetensors(model_dir / "model.safetensors");

  const auto profile_root =
    std::filesystem::temp_directory_path() / "kraken-infer-mpsgraph-eos-profile-smoke";
  std::error_code ec;
  std::filesystem::remove_all(profile_root, ec);

  toyllm::CpuGenerationRequest request;
  request.compute_device = toyllm::Device::mpsgraph();
  request.model_dir = model_dir;
  request.prompt = "hello";
  request.max_new_tokens = 2;
  request.observability.profile_mode = toyllm::ProfileMode::summary;
  request.observability.profile_output_dir = profile_root;
  request.observability.request_id = "mpsgraph-eos-profile-smoke";

  const auto result = toyllm::generate_cpu(request);
  assert(result.is_ok());
  assert(result.value().implemented);
  assert(result.value().text.empty());
  assert(result.value().finish_reason == "stop");
  assert(result.value().generated_tokens == 0);
  assert(result.value().prompt_tokens > 0);
  assert(std::filesystem::exists(result.value().profile_dir / "summary.json"));
  const auto summary_json = read_text_file(result.value().profile_dir / "summary.json");
  assert(summary_json.find("\"generated_tokens\":0") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_finish_reason\":\"stop\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_d2h_calls\":\"1\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_decode_steps\":\"1\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_decode_forward_steps\":\"0\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_status_readbacks\":\"1\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_early_break\":\"true\"") != std::string::npos);
  assert(summary_json.find("\"mpsgraph_eos_stop_step\":\"0\"") != std::string::npos);
  assert(summary_json.find("mpsgraph.final_readback.generated_ids") == std::string::npos);

  std::filesystem::remove_all(profile_root, ec);
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
  test_mpsgraph_kv_cache_store();
  test_mpsgraph_qk_norm_rope_and_attention_ops();
  test_mps_matvec_workspace_reuse();
  test_mps_graph_scope_batches_ops();
  test_mps_full_forward_operators();
  test_profile_artifacts();
  test_qwen3_model_config();
  test_qwen35_matmul_bench_format();
  test_qwen35_gdn_bench_format();
  test_qwen35_attention_bench_format();
  test_qwen35_gguf_model_config();
  test_qwen35_gguf_generation_uses_native_runtime();
  test_cpu_generation_entrypoint();
  test_weight_summary();
  test_weight_summary_regressions();
  test_mpsgraph_weight_store_metadata_and_upload();
  test_qwen_mpsgraph_model_core_weight_load();
  test_qwen_mpsgraph_model_all_weight_load();
  test_qwen_mpsgraph_model_forward_token();
  test_qwen_mpsgraph_model_prefill_token_ids();
  test_mpsgraph_generation_initializes_without_fallback();
  test_mpsgraph_generation_reuses_runtime_cache();
  test_mpsgraph_generation_device_side_eos_status();

  std::cout << "smoke tests passed\n";
  return 0;
}
