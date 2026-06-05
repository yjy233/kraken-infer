#include "toyllm/runtime/cpu_inference.hpp"

#include "cpu/qwen_cpu_model.hpp"

#include <sstream>
#include <stdexcept>

namespace toyllm {

namespace {

CpuKvCacheReport to_public_report(const cpu::KvCacheStats& stats) {
  return CpuKvCacheReport{
    stats.available,
    stats.layers,
    stats.kv_heads,
    stats.head_dim,
    stats.kv_dim,
    stats.capacity_tokens,
    stats.used_tokens,
    stats.key_bytes,
    stats.value_bytes,
    stats.total_bytes,
  };
}

}  // namespace

Result<CpuGenerationResult> generate_cpu(const CpuGenerationRequest& request) {
  if (request.prompt.empty() && request.messages.empty()) {
    return Status::invalid_argument("prompt must not be empty");
  }
  if (request.max_new_tokens == 0) {
    return Status::invalid_argument("max_new_tokens must be greater than zero");
  }

  try {
    auto messages = request.messages;
    if (messages.empty()) {
      messages.push_back(ChatMessage{"user", request.prompt});
    }
    if (messages.empty()) {
      return Status::invalid_argument("messages must not be empty");
    }

    CpuGenerationResult result{};
    result.implemented = true;
    const auto output =
      cpu::generate_text(request.model_dir, messages, request.max_new_tokens,
                         request.enable_thinking, request.debug_dump_dir, request.verify_kv_cache,
                         request.sampling, request.stream_token);
    result.text = output.text;
    result.kv_cache = to_public_report(output.kv_cache);
    result.kv_cache_verified = output.kv_cache_verified;
    return result;
  } catch (const std::exception& error) {
    return Status::internal_error(error.what());
  }
}

std::string format_cpu_generation_result(const CpuGenerationResult& result) {
  if (result.implemented) {
    return result.text + '\n';
  }

  std::ostringstream output;
  output << result.text;
  if (!result.missing_dependencies.empty()) {
    output << "Missing dependencies:\n";
    for (const auto& dependency : result.missing_dependencies) {
      output << "- " << dependency << '\n';
    }
  }
  return output.str();
}

Result<std::string> format_weight_summary(const std::filesystem::path& model_dir) {
  try {
    return cpu::build_weight_summary(model_dir);
  } catch (const std::exception& error) {
    return Status::internal_error(error.what());
  }
}

}  // namespace toyllm
