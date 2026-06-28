#pragma once

#include "toyllm/core/status.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/cpu_inference.hpp"
#include "toyllm/runtime/qwen35_weight_map.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace toyllm {

struct Qwen35RuntimeOptions {
  std::size_t context_tokens{0};
  std::size_t decode_tokens{1};
  std::size_t prefill_chunk_tokens{1024};
  std::size_t sequence_slots{1};
  std::size_t recurrent_snapshot_count{0};
  bool kv_cache_f16{true};
  bool output_all_prompt_logits{false};
};

struct Qwen35CachePlan {
  std::size_t attention_capacity_tokens{0};
  std::size_t recurrent_slots{0};
  std::size_t recurrent_rows{0};
  std::size_t full_attention_layers{0};
  std::size_t linear_attention_layers{0};
  std::size_t kv_dim{0};
  std::size_t kv_cache_element_bytes{sizeof(std::uint16_t)};
  bool kv_cache_f16{true};
  std::size_t recurrent_r_elements_per_layer{0};
  std::size_t recurrent_s_elements_per_layer{0};
  std::uint64_t kv_cache_bytes{0};
  std::uint64_t recurrent_r_cache_bytes{0};
  std::uint64_t recurrent_s_cache_bytes{0};
  std::uint64_t total_cache_bytes{0};
};

struct Qwen35PrefillPlan {
  std::size_t prompt_tokens{0};
  std::size_t chunk_tokens{0};
  std::size_t chunk_count{0};
  std::size_t final_chunk_tokens{0};
  bool output_only_last_token{true};
};

struct Qwen35ExecutionPlan {
  std::size_t hidden_size{0};
  std::size_t main_layers{0};
  std::size_t total_layers{0};
  std::size_t head_dim{0};
  std::size_t attention_heads{0};
  std::size_t kv_heads{0};
  std::size_t linear_key_heads{0};
  std::size_t linear_value_heads{0};
  std::size_t linear_key_head_dim{0};
  std::size_t linear_value_head_dim{0};
  std::size_t linear_inner_size{0};
  std::size_t linear_conv_kernel{0};
  std::size_t linear_conv_channels{0};
  std::vector<Qwen35LayerKind> layer_kinds;
  Qwen35CachePlan cache;
  Qwen35PrefillPlan prefill;
};

[[nodiscard]] Result<Qwen35ExecutionPlan> build_qwen35_execution_plan(
  const ModelConfig& config, const Qwen35WeightMap& weights,
  std::size_t prompt_tokens, const Qwen35RuntimeOptions& options);
[[nodiscard]] std::string format_qwen35_execution_plan(const Qwen35ExecutionPlan& plan);

struct Qwen35MatmulBenchConfig {
  std::filesystem::path model_dir;
  std::string tensor_name;
  std::size_t tokens{1024};
  std::size_t warmup_iterations{2};
  std::size_t timed_iterations{10};
};

struct Qwen35MatmulBenchResult {
  std::filesystem::path gguf_path;
  std::string tensor_name;
  std::string type_name;
  std::string dispatch;
  std::uint32_t type{0};
  std::size_t tokens{0};
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t warmup_iterations{0};
  std::size_t timed_iterations{0};
  std::uint64_t output_values_per_iteration{0};
  double total_seconds{0.0};
  double average_milliseconds{0.0};
  double token_iterations_per_second{0.0};
  double output_megavalues_per_second{0.0};
  double sample_checksum{0.0};
};

[[nodiscard]] Result<Qwen35MatmulBenchResult> benchmark_qwen35_metal_matmul(
  const Qwen35MatmulBenchConfig& config);
[[nodiscard]] std::string format_qwen35_matmul_bench_result(
  const Qwen35MatmulBenchResult& result);

struct Qwen35GdnBenchConfig {
  std::size_t tokens{1024};
  std::size_t key_heads{16};
  std::size_t value_heads{16};
  std::size_t head_dim{128};
  std::size_t warmup_iterations{2};
  std::size_t timed_iterations{10};
};

struct Qwen35GdnBenchResult {
  std::string dispatch;
  std::size_t tokens{0};
  std::size_t key_heads{0};
  std::size_t value_heads{0};
  std::size_t head_dim{0};
  std::size_t warmup_iterations{0};
  std::size_t timed_iterations{0};
  std::uint64_t state_values{0};
  std::uint64_t output_values_per_iteration{0};
  double total_seconds{0.0};
  double average_milliseconds{0.0};
  double token_iterations_per_second{0.0};
  double output_megavalues_per_second{0.0};
  double sample_checksum{0.0};
};

[[nodiscard]] Result<Qwen35GdnBenchResult> benchmark_qwen35_metal_gdn(
  const Qwen35GdnBenchConfig& config);
[[nodiscard]] std::string format_qwen35_gdn_bench_result(
  const Qwen35GdnBenchResult& result);

struct Qwen35AttentionBenchConfig {
  std::size_t tokens{1024};
  std::size_t start_position{0};
  std::size_t capacity_tokens{0};
  std::size_t heads{8};
  std::size_t kv_heads{2};
  std::size_t head_dim{256};
  std::size_t warmup_iterations{2};
  std::size_t timed_iterations{10};
  bool f16_kv{true};
};

struct Qwen35AttentionBenchResult {
  std::string dispatch;
  std::size_t tokens{0};
  std::size_t start_position{0};
  std::size_t capacity_tokens{0};
  std::size_t key_count{0};
  std::size_t heads{0};
  std::size_t kv_heads{0};
  std::size_t head_dim{0};
  std::size_t warmup_iterations{0};
  std::size_t timed_iterations{0};
  std::uint64_t cache_values{0};
  std::uint64_t output_values_per_iteration{0};
  double total_seconds{0.0};
  double average_milliseconds{0.0};
  double token_iterations_per_second{0.0};
  double output_megavalues_per_second{0.0};
  double sample_checksum{0.0};
  bool f16_kv{true};
};

[[nodiscard]] Result<Qwen35AttentionBenchResult> benchmark_qwen35_metal_attention(
  const Qwen35AttentionBenchConfig& config);
[[nodiscard]] std::string format_qwen35_attention_bench_result(
  const Qwen35AttentionBenchResult& result);

[[nodiscard]] Result<CpuGenerationResult> generate_qwen35_metal(
  const CpuGenerationRequest& request);

}  // namespace toyllm
