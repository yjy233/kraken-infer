#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_kv_cache.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_weight_store.hpp"
#include "toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp"
#include "toyllm/core/tensor.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/cpu_inference.hpp"
#include "toyllm/runtime/qwen_tokenizer.hpp"
#include "toyllm/runtime/runtime.hpp"

#include <array>
#include <cassert>
#include <chrono>
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
  test_mps_full_forward_operators();
  test_profile_artifacts();
  test_qwen3_model_config();
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
