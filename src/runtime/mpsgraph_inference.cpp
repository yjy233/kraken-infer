#include "toyllm/runtime/mpsgraph_inference.hpp"

#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp"
#include "toyllm/runtime/qwen_tokenizer.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace toyllm {

namespace {

CpuKvCacheReport to_public_report(const mpsgraph::MpsGraphKvCacheStats& stats) {
  return CpuKvCacheReport{
    stats.allocated,
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

bool is_eos(std::int64_t token, const GenerationConfig& generation,
            const ModelConfig& model) {
  if (token == model.eos_token_id || token == kQwenEndOfText) {
    return true;
  }
  return std::find(generation.eos_token_ids.begin(), generation.eos_token_ids.end(), token) !=
         generation.eos_token_ids.end();
}

}  // namespace

Result<CpuGenerationResult> generate_mpsgraph(const CpuGenerationRequest& request) {
  const auto backend = mpsgraph::query_backend();
  if (!backend.available || !backend.graph_ready) {
    const auto message = backend.failure_reason.empty()
                           ? std::string{"MPSGraph backend is not ready"}
                           : backend.failure_reason;
    return Status::unavailable(message);
  }

  if (request.stream_token) {
    return Status::unavailable(
      "MPSGraph backend does not support streaming in strict no-readback mode yet");
  }
  if (request.sampling.do_sample) {
    return Status::unavailable("MPSGraph backend currently supports greedy decoding only");
  }
  if (!request.debug_dump_dir.empty()) {
    return Status::unavailable(
      "MPSGraph backend does not support debug tensor dumps in strict mode");
  }
  if (request.verify_kv_cache) {
    return Status::unavailable("MPSGraph backend does not support KV cache verification yet");
  }
  if (request.prompt.empty() && request.messages.empty()) {
    return Status::invalid_argument("prompt must not be empty");
  }

  try {
    RequestProfiler profiler(request.observability);
    profiler.set_metadata("model_dir", request.model_dir.string());
    profiler.set_metadata("device", request.compute_device.to_string());
    if (!request.observability.client_request_id.empty()) {
      profiler.set_metadata("client_request_id", request.observability.client_request_id);
    }

    std::vector<ChatMessage> messages = request.messages;
    if (messages.empty()) {
      messages.push_back(ChatMessage{"user", request.prompt});
    }
    if (messages.empty()) {
      return Status::invalid_argument("messages must not be empty");
    }

    QwenTokenizer tokenizer;
    std::vector<std::int64_t> prompt_tokens;
    {
      auto span = profiler.scoped("request.tokenize");
      tokenizer = QwenTokenizer::load(request.model_dir);
      prompt_tokens = tokenizer.encode_chat_messages(messages, request.enable_thinking);
      (void)span;
    }
    if (prompt_tokens.empty()) {
      return Status::invalid_argument("tokenizer produced no prompt tokens");
    }
    profiler.set_metadata("prompt_tokens", prompt_tokens.size());

    auto context = mpsgraph::MpsGraphContext::create();
    if (!context.is_ok()) {
      return context.status();
    }

    auto model = mpsgraph::QwenMpsGraphModel::load_all_weights(request.model_dir,
                                                              context.value());
    if (!model.is_ok()) {
      return model.status();
    }

    if (request.max_new_tokens == std::numeric_limits<std::size_t>::max() ||
        prompt_tokens.size() > std::numeric_limits<std::size_t>::max() -
          request.max_new_tokens - 1U) {
      return Status::invalid_argument("MPSGraph token capacity exceeds supported range");
    }
    const auto total_capacity = prompt_tokens.size() + request.max_new_tokens + 1U;
    auto state = model.value().create_run_state(context.value(), total_capacity);
    if (!state.is_ok()) {
      return state.status();
    }

    std::size_t position = 0;
    {
      auto span = profiler.scoped("request.prefill");
      auto status = model.value().prefill_token_ids(context.value(), prompt_tokens,
                                                    state.value());
      if (!status.is_ok()) {
        return status;
      }
      position = prompt_tokens.size();
      (void)span;
    }

    std::size_t generated_count = 0;
    {
      auto decode_span = profiler.scoped("request.decode");
      for (std::size_t step = 0; step < request.max_new_tokens; ++step) {
        auto step_span =
          profiler.scoped("decode_step", {ProfileField{"step", std::to_string(step)}});
        auto status = model.value().greedy_next_token(context.value(), state.value());
        if (!status.is_ok()) {
          return status;
        }
        status = model.value().record_next_token(context.value(), step, state.value());
        if (!status.is_ok()) {
          return status;
        }
        ++generated_count;
        if (step + 1U == request.max_new_tokens) {
          break;
        }
        status = model.value().forward_next_token(context.value(), position, state.value());
        if (!status.is_ok()) {
          return status;
        }
        ++position;
        (void)step_span;
      }
      (void)decode_span;
    }

    std::vector<std::int32_t> generated_i32(generated_count);
    if (generated_count > 0) {
      const auto read_status =
        context.value().copy_from_buffer(state.value().generated_tokens, generated_i32.data(),
                                         generated_i32.size() * sizeof(std::int32_t));
      if (!read_status.is_ok()) {
        return read_status;
      }
    }

    std::vector<std::int64_t> generated;
    generated.reserve(generated_i32.size());
    for (const auto token : generated_i32) {
      if (is_eos(token, model.value().generation(), model.value().config())) {
        break;
      }
      generated.push_back(token);
    }

    CpuGenerationResult result{};
    result.implemented = true;
    result.text = tokenizer.decode(generated);
    result.request_id = profiler.request_id();
    result.kv_cache = to_public_report(state.value().kv_cache.stats());
    result.kv_cache_verified = false;
    profiler.set_metadata("generated_tokens", generated.size());
    profiler.set_status("ok");
    const auto artifacts = profiler.write_artifacts();
    if (!artifacts.output_dir.empty()) {
      result.profile_dir = artifacts.output_dir;
    }
    return result;
  } catch (const std::exception& error) {
    return Status::internal_error(error.what());
  }
}

}  // namespace toyllm
