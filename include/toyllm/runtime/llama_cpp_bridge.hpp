#pragma once

#include "toyllm/core/device.hpp"
#include "toyllm/core/status.hpp"
#include "toyllm/runtime/chat_message.hpp"
#include "toyllm/runtime/profiling.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace toyllm {

struct LlamaCppServerOptions {
  std::filesystem::path model_path;
  std::filesystem::path executable_path;
  std::filesystem::path llama_cpp_dir{"/Users/bill/code/llama.cpp"};
  std::filesystem::path mmproj_path;
  std::string model_alias{"kraken-infer-gguf"};
  std::string host{"127.0.0.1"};
  int port{0};
  Device compute_device{Device::cpu()};
  std::size_t context_size{0};
  std::size_t parallel_slots{1};
  std::size_t batch_size{2048};
  std::size_t ubatch_size{512};
  bool enable_mtp{true};
  bool enable_continuous_batching{true};
  bool enable_flash_attention{true};
  std::vector<std::string> extra_args;
};

struct LlamaCppHttpResponse {
  int status{0};
  std::string content_type;
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;
};

struct LlamaCppGenerationRequest {
  LlamaCppServerOptions server;
  std::string prompt;
  std::vector<ChatMessage> messages;
  std::size_t max_new_tokens{16};
  double temperature{0.8};
  double top_p{0.95};
  std::uint64_t seed{0};
  bool seed_set{false};
  bool stream{false};
  std::function<void(std::string_view)> stream_token;
  ObservabilityConfig observability;
};

struct LlamaCppGenerationResult {
  std::string text;
  std::string finish_reason{"stop"};
  std::size_t prompt_tokens{0};
  std::size_t generated_tokens{0};
};

[[nodiscard]] bool is_gguf_model_path(const std::filesystem::path& path);
[[nodiscard]] Result<std::filesystem::path> resolve_llama_cpp_server_executable(
  const LlamaCppServerOptions& options);
[[nodiscard]] Result<LlamaCppHttpResponse> forward_openai_request_to_llama_cpp(
  const LlamaCppServerOptions& options, std::string_view method, std::string_view path,
  std::string_view body, const std::vector<std::pair<std::string, std::string>>& headers);
[[nodiscard]] Result<LlamaCppGenerationResult> generate_with_llama_cpp(
  const LlamaCppGenerationRequest& request);

}  // namespace toyllm
