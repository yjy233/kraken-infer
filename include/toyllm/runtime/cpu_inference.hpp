#pragma once

#include "toyllm/core/status.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace toyllm {

struct ChatMessage {
  std::string role;
  std::string content;
};

struct CpuGenerationRequest {
  std::filesystem::path model_dir{"models/qwen3-0.6b"};
  std::string prompt;
  std::size_t max_new_tokens{16};
  bool enable_thinking{false};
  std::vector<ChatMessage> messages;
};

struct CpuGenerationResult {
  bool implemented{false};
  std::string text;
  std::vector<std::string> missing_dependencies;
};

[[nodiscard]] Result<CpuGenerationResult> generate_cpu(const CpuGenerationRequest& request);
[[nodiscard]] std::string format_cpu_generation_result(const CpuGenerationResult& result);
[[nodiscard]] Result<std::string> format_weight_summary(const std::filesystem::path& model_dir);

}  // namespace toyllm
