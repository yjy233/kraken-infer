#pragma once

#include "toyllm/runtime/cpu_inference.hpp"
#include "toyllm/runtime/profiling.hpp"

#include "kv_cache.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace toyllm::cpu {

struct CpuModelOutput {
  std::string text;
  KvCacheStats kv_cache;
  bool kv_cache_verified{false};
  std::size_t prompt_tokens{0};
  std::size_t generated_tokens{0};
};

CpuModelOutput generate_text(const std::filesystem::path& model_dir,
                             const std::vector<ChatMessage>& messages,
                             std::size_t max_new_tokens, bool enable_thinking,
                             const std::filesystem::path& debug_dump_dir,
                             bool verify_kv_cache, Device compute_device,
                             const CpuSamplingConfig& sampling,
                             const std::function<void(std::string_view)>& stream_token,
                             RequestProfiler* profiler = nullptr);
std::string build_weight_summary(const std::filesystem::path& model_dir);

}  // namespace toyllm::cpu
