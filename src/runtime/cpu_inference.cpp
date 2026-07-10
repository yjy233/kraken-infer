#include "toyllm/runtime/cpu_inference.hpp"

#include "toyllm/runtime/gguf_reader.hpp"
#include "toyllm/runtime/mpsgraph_inference.hpp"
#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/runtime/qwen35_runtime.hpp"
#include "cpu/qwen_cpu_model.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace toyllm {

namespace {

std::string escape_debug_text(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped.push_back(character);
        break;
    }
  }
  return escaped;
}

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

std::string format_mtp_position_counts(const CpuMtpReport& report) {
  if (report.verified_by_position.empty() && report.accepted_by_position.empty()) {
    return {};
  }
  const auto positions = std::max(report.verified_by_position.size(),
                                  report.accepted_by_position.size());
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < positions; ++index) {
    if (index > 0) {
      output << ", ";
    }
    const auto accepted = index < report.accepted_by_position.size()
                            ? report.accepted_by_position[index]
                            : std::size_t{0};
    const auto verified = index < report.verified_by_position.size()
                            ? report.verified_by_position[index]
                            : std::size_t{0};
    output << accepted << '/' << verified;
  }
  output << ']';
  return output.str();
}

}  // namespace

Result<CpuGenerationResult> generate_cpu(const CpuGenerationRequest& request) {
  if (resolve_gguf_model_path(request.model_dir).is_ok()) {
    return generate_qwen35_metal(request);
  }
  if (chat_messages_have_image_content(request.messages)) {
    return Status::unavailable(
      "image input is only wired for the native Qwen3.5 GGUF path");
  }
  if (request.prompt.empty() && request.messages.empty()) {
    return Status::invalid_argument("prompt must not be empty");
  }
  if (request.max_new_tokens == 0) {
    return Status::invalid_argument("max_new_tokens must be greater than zero");
  }
  if (request.compute_device.kind == DeviceKind::mps) {
    const auto backend = mps::query_backend();
    if (!backend.compute_ready) {
      auto message = backend.failure_reason.empty() ? std::string{"MPS compute is not ready"}
                                                    : backend.failure_reason;
      return Status::unavailable(message);
    }
  }
  if (request.compute_device.kind == DeviceKind::mpsgraph) {
    return generate_mpsgraph(request);
  }

  try {
    RequestProfiler profiler(request.observability);
    auto messages = request.messages;
    if (messages.empty()) {
      messages.push_back(ChatMessage{"user", request.prompt});
    }
    if (messages.empty()) {
      return Status::invalid_argument("messages must not be empty");
    }

    profiler.set_metadata("model_dir", request.model_dir.string());
    profiler.set_metadata("device", request.compute_device.to_string());
    if (!request.observability.client_request_id.empty()) {
      profiler.set_metadata("client_request_id", request.observability.client_request_id);
    }

    CpuGenerationResult result{};
    result.implemented = true;
    const auto output =
      cpu::generate_text(request.model_dir, messages, request.max_new_tokens,
                         request.enable_thinking, request.debug_dump_dir, request.verify_kv_cache,
                         request.compute_device, request.sampling, request.stream_token,
                         &profiler);
    result.text = output.text;
    result.request_id = profiler.request_id();
    result.finish_reason = output.generated_tokens >= request.max_new_tokens ? "length" : "stop";
    result.prompt_tokens = output.prompt_tokens;
    result.generated_tokens = output.generated_tokens;
    result.kv_cache = to_public_report(output.kv_cache);
    result.kv_cache_verified = output.kv_cache_verified;
    profiler.set_metadata("prompt_tokens", output.prompt_tokens);
    profiler.set_metadata("generated_tokens", output.generated_tokens);
    if (request.compute_device.kind == DeviceKind::mps) {
      profiler.set_metadata("device", request.compute_device.to_string());
    }
    const auto artifacts = profiler.write_artifacts();
    if (!artifacts.output_dir.empty()) {
      result.profile_dir = artifacts.output_dir;
    }
    return result;
  } catch (const std::exception& error) {
    return Status::internal_error(error.what());
  }
}

std::string format_cpu_generation_result(const CpuGenerationResult& result) {
  if (result.implemented) {
    std::ostringstream output;
    output << result.text << '\n';
    if (!result.logits_top.empty()) {
      output << "logits_top:\n";
      for (std::size_t index = 0; index < result.logits_top.size(); ++index) {
        const auto& entry = result.logits_top[index];
        output << index << " token_id=" << entry.token_id
               << " logit=" << entry.logit
               << " text=\"" << escape_debug_text(entry.text) << "\"\n";
      }
    }
    if (result.mtp.available) {
      output << "mtp: " << (result.mtp.enabled ? "enabled" : "disabled")
             << ", layers=" << result.mtp.layers
             << ", draft_tokens=" << result.mtp.draft_tokens
             << ", p_min=" << result.mtp.p_min
             << ", drafted=" << result.mtp.drafted_tokens
             << ", accepted=" << result.mtp.accepted_tokens
             << ", verify_steps=" << result.mtp.verify_steps
             << ", confidence_stops=" << result.mtp.confidence_stops;
      if (result.mtp.enabled &&
          (result.mtp.adaptive_budget != result.mtp.draft_tokens ||
           result.mtp.adaptive_changes != 0U)) {
        output << ", adaptive_budget=" << result.mtp.adaptive_budget
               << ", adaptive_changes=" << result.mtp.adaptive_changes;
      }
      const auto position_counts = format_mtp_position_counts(result.mtp);
      if (!position_counts.empty()) {
        output << ", accepted_by_position=" << position_counts;
      }
      if (!result.mtp.enabled && !result.mtp.disabled_reason.empty()) {
        output << ", reason=" << result.mtp.disabled_reason;
      }
      output << '\n';
    }
    return output.str();
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
