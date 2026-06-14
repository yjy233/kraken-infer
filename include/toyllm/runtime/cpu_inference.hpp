#pragma once

#include "toyllm/core/device.hpp"
#include "toyllm/core/status.hpp"
#include "toyllm/runtime/chat_message.hpp"
#include "toyllm/runtime/profiling.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace toyllm {

struct CpuSamplingConfig {
  bool do_sample{false};
  bool temperature_set{false};
  bool top_k_set{false};
  bool top_p_set{false};
  bool seed_set{false};
  double temperature{1.0};
  std::size_t top_k{0};
  double top_p{1.0};
  std::uint64_t seed{0};
};

struct CpuGenerationRequest {
  std::filesystem::path model_dir{"models/qwen3-0.6b"};
  std::string prompt;
  std::size_t max_new_tokens{16};
  bool enable_thinking{false};
  std::vector<ChatMessage> messages;
  std::filesystem::path debug_dump_dir;
  bool verify_kv_cache{false};
  Device compute_device{Device::cpu()};
  CpuSamplingConfig sampling;
  std::function<void(std::string_view)> stream_token;
  ObservabilityConfig observability;
};

struct CpuKvCacheReport {
  bool available{false};
  std::size_t layers{0};
  std::size_t kv_heads{0};
  std::size_t head_dim{0};
  std::size_t kv_dim{0};
  std::size_t capacity_tokens{0};
  std::size_t used_tokens{0};
  std::uint64_t key_bytes{0};
  std::uint64_t value_bytes{0};
  std::uint64_t total_bytes{0};
};

struct CpuGenerationResult {
  bool implemented{false};
  std::string text;
  std::string request_id;
  std::filesystem::path profile_dir;
  std::vector<std::string> missing_dependencies;
  CpuKvCacheReport kv_cache;
  bool kv_cache_verified{false};
};

[[nodiscard]] Result<CpuGenerationResult> generate_cpu(const CpuGenerationRequest& request);
[[nodiscard]] std::string format_cpu_generation_result(const CpuGenerationResult& result);
[[nodiscard]] Result<std::string> format_weight_summary(const std::filesystem::path& model_dir);

}  // namespace toyllm
