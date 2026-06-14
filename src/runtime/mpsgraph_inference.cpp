#include "toyllm/runtime/mpsgraph_inference.hpp"

#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp"
#include "toyllm/runtime/qwen_tokenizer.hpp"

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

std::uint64_t device_to_host_delta(const mpsgraph::MpsGraphTransferStats& before,
                                   const mpsgraph::MpsGraphTransferStats& after) {
  return after.device_to_host_calls >= before.device_to_host_calls
           ? after.device_to_host_calls - before.device_to_host_calls
           : 0U;
}

std::uint64_t transfer_call_delta(const mpsgraph::MpsGraphTransferStats& before,
                                  const mpsgraph::MpsGraphTransferStats& after) {
  const auto host_to_device = after.host_to_device_calls >= before.host_to_device_calls
                                ? after.host_to_device_calls - before.host_to_device_calls
                                : 0U;
  return host_to_device + device_to_host_delta(before, after);
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

    mpsgraph::MpsGraphContext context;
    {
      auto span = profiler.scoped("mpsgraph.create_context");
      auto context_result = mpsgraph::MpsGraphContext::create();
      if (!context_result.is_ok()) {
        return context_result.status();
      }
      context = std::move(context_result.value());
      (void)span;
    }

    mpsgraph::QwenMpsGraphModel model;
    {
      auto span = profiler.scoped("mpsgraph.load_weights");
      auto model_result = mpsgraph::QwenMpsGraphModel::load_all_weights(request.model_dir,
                                                                       context);
      if (!model_result.is_ok()) {
        return model_result.status();
      }
      model = std::move(model_result.value());
      (void)span;
    }

    if (request.max_new_tokens == std::numeric_limits<std::size_t>::max() ||
        prompt_tokens.size() > std::numeric_limits<std::size_t>::max() -
          request.max_new_tokens - 1U) {
      return Status::invalid_argument("MPSGraph token capacity exceeds supported range");
    }
    const auto total_capacity = prompt_tokens.size() + request.max_new_tokens + 1U;
    mpsgraph::QwenMpsGraphRunState state;
    {
      auto span = profiler.scoped("mpsgraph.create_run_state");
      auto state_result = model.create_run_state(context, total_capacity);
      if (!state_result.is_ok()) {
        return state_result.status();
      }
      state = std::move(state_result.value());
      (void)span;
    }

    std::size_t position = 0;
    {
      auto span = profiler.scoped("request.prefill");
      const auto transfer_before = context.transfer_stats();
      auto status = model.prefill_token_ids(context, prompt_tokens, state);
      if (!status.is_ok()) {
        return status;
      }
      const auto transfer_after = context.transfer_stats();
      if (device_to_host_delta(transfer_before, transfer_after) != 0U) {
        return Status::internal_error("MPSGraph prefill performed host readback");
      }
      position = prompt_tokens.size();
      (void)span;
    }

    {
      auto decode_span = profiler.scoped("request.decode");
      const auto transfer_before = context.transfer_stats();
      for (std::size_t step = 0; step < request.max_new_tokens; ++step) {
        auto step_span =
          profiler.scoped("decode_step", {ProfileField{"step", std::to_string(step)}});
        {
          auto span = profiler.scoped("mpsgraph.decode.argmax");
          auto status = model.greedy_next_token(context, state);
          if (!status.is_ok()) {
            return status;
          }
          (void)span;
        }
        {
          auto span = profiler.scoped("mpsgraph.decode.record_token");
          auto status = model.record_next_token(context, step, state);
          if (!status.is_ok()) {
            return status;
          }
          (void)span;
        }
        {
          auto span = profiler.scoped("mpsgraph.decode.update_generation_status");
          auto status = model.update_generation_status(
            context, step, step + 1U == request.max_new_tokens, state);
          if (!status.is_ok()) {
            return status;
          }
          (void)span;
        }
        if (step + 1U == request.max_new_tokens) {
          break;
        }
        {
          auto span = profiler.scoped("mpsgraph.decode.forward_next_token");
          auto status = model.forward_next_token(context, position, state);
          if (!status.is_ok()) {
            return status;
          }
          (void)span;
        }
        ++position;
        (void)step_span;
      }
      const auto transfer_after = context.transfer_stats();
      if (transfer_call_delta(transfer_before, transfer_after) != 0U) {
        return Status::internal_error("MPSGraph decode performed host/device transfer");
      }
      (void)decode_span;
    }

    std::vector<std::int32_t> generation_status(3);
    {
      auto span = profiler.scoped("mpsgraph.final_readback.generation_status");
      const auto read_status =
        context.copy_from_buffer(state.generation_status, generation_status.data(),
                                 generation_status.size() * sizeof(std::int32_t));
      if (!read_status.is_ok()) {
        return read_status;
      }
      (void)span;
    }
    if (generation_status[0] < 0) {
      return Status::internal_error("MPSGraph generated count is negative");
    }
    auto generated_count = static_cast<std::size_t>(generation_status[0]);
    if (generated_count > request.max_new_tokens) {
      return Status::internal_error("MPSGraph generated count exceeds max_new_tokens");
    }

    std::vector<std::int32_t> generated_i32(generated_count);
    if (generated_count > 0) {
      auto span = profiler.scoped("mpsgraph.final_readback.generated_ids");
      const auto read_status =
        context.copy_from_buffer(state.generated_tokens, generated_i32.data(),
                                 generated_i32.size() * sizeof(std::int32_t));
      if (!read_status.is_ok()) {
        return read_status;
      }
      (void)span;
    }

    std::vector<std::int64_t> generated;
    generated.reserve(generated_i32.size());
    for (const auto token : generated_i32) {
      generated.push_back(token);
    }

    CpuGenerationResult result{};
    result.implemented = true;
    result.text = tokenizer.decode(generated);
    result.request_id = profiler.request_id();
    result.finish_reason = generation_status[1] == 1 ? "stop" : "length";
    result.prompt_tokens = prompt_tokens.size();
    result.generated_tokens = generated.size();
    result.kv_cache = to_public_report(state.kv_cache.stats());
    result.kv_cache_verified = false;
    profiler.set_metadata("generated_tokens", generated.size());
    profiler.set_metadata("mpsgraph_finish_reason", result.finish_reason);
    const auto transfers = context.transfer_stats();
    profiler.set_metadata("mpsgraph_h2d_calls",
                          static_cast<std::size_t>(transfers.host_to_device_calls));
    profiler.set_metadata("mpsgraph_d2h_calls",
                          static_cast<std::size_t>(transfers.device_to_host_calls));
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
