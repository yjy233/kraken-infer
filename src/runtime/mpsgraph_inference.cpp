#include "toyllm/runtime/mpsgraph_inference.hpp"

#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp"
#include "toyllm/runtime/qwen_tokenizer.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
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

std::uint64_t host_to_device_delta(const mpsgraph::MpsGraphTransferStats& before,
                                   const mpsgraph::MpsGraphTransferStats& after) {
  return after.host_to_device_calls >= before.host_to_device_calls
           ? after.host_to_device_calls - before.host_to_device_calls
           : 0U;
}

std::uint64_t graph_stat_delta(std::uint64_t before, std::uint64_t after) {
  return after >= before ? after - before : 0U;
}

std::filesystem::path canonical_model_key(const std::filesystem::path& model_dir) {
  std::error_code ec;
  auto canonical = std::filesystem::weakly_canonical(model_dir, ec);
  if (!ec) {
    return canonical;
  }
  auto absolute = std::filesystem::absolute(model_dir, ec);
  return ec ? model_dir : absolute.lexically_normal();
}

struct MpsGraphRuntimeCache {
  std::filesystem::path model_key;
  mpsgraph::MpsGraphContext context;
  mpsgraph::QwenMpsGraphModel model;
  QwenTokenizer tokenizer;
  std::size_t warmed_decode_capacity{0};
  std::size_t warmed_max_new_tokens{0};
};

std::mutex& runtime_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unique_ptr<MpsGraphRuntimeCache>& runtime_cache_slot() {
  static std::unique_ptr<MpsGraphRuntimeCache> cache;
  return cache;
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

    std::unique_lock<std::mutex> cache_lock(runtime_cache_mutex());
    auto& cache = runtime_cache_slot();
    const auto model_key = canonical_model_key(request.model_dir);
    const auto cache_hit = cache != nullptr && cache->model_key == model_key;
    profiler.set_metadata("mpsgraph_model_cache", cache_hit ? "hit" : "miss");
    profiler.set_metadata("mpsgraph_tokenizer_cache", cache_hit ? "hit" : "miss");
    mpsgraph::MpsGraphTransferStats request_transfer_before{};
    mpsgraph::MpsGraphGraphStats request_graph_before{};
    if (!cache_hit) {
      auto new_cache = std::make_unique<MpsGraphRuntimeCache>();
      new_cache->model_key = model_key;
      {
        auto span = profiler.scoped("mpsgraph.create_context");
        auto context_result = mpsgraph::MpsGraphContext::create();
        if (!context_result.is_ok()) {
          return context_result.status();
        }
        new_cache->context = std::move(context_result.value());
        (void)span;
      }
      request_transfer_before = new_cache->context.transfer_stats();
      request_graph_before = new_cache->context.graph_stats();
      {
        auto span = profiler.scoped("request.tokenize.load");
        new_cache->tokenizer = QwenTokenizer::load(request.model_dir);
        (void)span;
      }
      {
        auto span = profiler.scoped("mpsgraph.load_weights");
        auto model_result = mpsgraph::QwenMpsGraphModel::load_all_weights(
          request.model_dir, new_cache->context);
        if (!model_result.is_ok()) {
          return model_result.status();
        }
        new_cache->model = std::move(model_result.value());
        (void)span;
      }
      cache = std::move(new_cache);
    }
    auto& context = cache->context;
    if (cache_hit) {
      request_transfer_before = context.transfer_stats();
      request_graph_before = context.graph_stats();
    }

    std::vector<std::int64_t> prompt_tokens;
    {
      auto tokenize_span = profiler.scoped("request.tokenize");
      auto encode_span = profiler.scoped("request.tokenize.encode");
      prompt_tokens = cache->tokenizer.encode_chat_messages(messages, request.enable_thinking);
      (void)encode_span;
      (void)tokenize_span;
    }
    if (prompt_tokens.empty()) {
      return Status::invalid_argument("tokenizer produced no prompt tokens");
    }
    profiler.set_metadata("prompt_tokens", prompt_tokens.size());
    const auto weight_info = cache->model.info();
    profiler.set_metadata("mpsgraph_device_weight_bytes",
                          static_cast<std::size_t>(weight_info.device_weight_bytes));
    profiler.set_metadata("mpsgraph_device_tensor_count", weight_info.device_tensor_count);

    if (request.max_new_tokens == std::numeric_limits<std::size_t>::max() ||
        prompt_tokens.size() > std::numeric_limits<std::size_t>::max() -
          request.max_new_tokens - 1U) {
      return Status::invalid_argument("MPSGraph token capacity exceeds supported range");
    }
    const auto total_capacity = prompt_tokens.size() + request.max_new_tokens + 1U;
    const auto decode_cache_hit =
      cache->warmed_decode_capacity >= total_capacity &&
      cache->warmed_max_new_tokens >= request.max_new_tokens;
    profiler.set_metadata("mpsgraph_decode_cache", decode_cache_hit ? "hit" : "miss");
    profiler.set_metadata("mpsgraph_decode_cache_capacity", cache->warmed_decode_capacity);
    profiler.set_metadata("mpsgraph_decode_cache_max_new_tokens",
                          cache->warmed_max_new_tokens);
    mpsgraph::QwenMpsGraphRunState state;
    {
      auto span = profiler.scoped("mpsgraph.create_run_state");
      auto state_result = cache->model.create_run_state(context, total_capacity);
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
      auto status = cache->model.prefill_token_ids(context, prompt_tokens, state,
                                                   &profiler);
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

    std::vector<std::int32_t> generation_status(3);
    bool generation_status_available = false;
    bool decode_stopped_early = false;
    std::size_t decode_steps = 0;
    std::size_t decode_forward_steps = 0;
    std::size_t eos_stop_step = std::numeric_limits<std::size_t>::max();
    std::size_t status_readbacks = 0;
    {
      auto decode_span = profiler.scoped("request.decode");
      const auto transfer_before = context.transfer_stats();
      for (std::size_t step = 0; step < request.max_new_tokens; ++step) {
        auto step_span =
          profiler.scoped("decode_step", {ProfileField{"step", std::to_string(step)}});
        {
          auto span = profiler.scoped("mpsgraph.decode.argmax");
          auto status = cache->model.greedy_next_token(context, state, &profiler);
          if (!status.is_ok()) {
            return status;
          }
          (void)span;
        }
        {
          auto span = profiler.scoped("mpsgraph.decode.record_token");
          auto status = cache->model.record_next_token(context, step, state);
          if (!status.is_ok()) {
            return status;
          }
          (void)span;
        }
        {
          auto span = profiler.scoped("mpsgraph.decode.update_generation_status");
          auto status = cache->model.update_generation_status(
            context, step, step + 1U == request.max_new_tokens, state);
          if (!status.is_ok()) {
            return status;
          }
          (void)span;
        }
        {
          auto span = profiler.scoped("mpsgraph.decode.read_generation_status");
          const auto read_status =
            context.copy_from_buffer(state.generation_status, generation_status.data(),
                                     generation_status.size() * sizeof(std::int32_t));
          if (!read_status.is_ok()) {
            return read_status;
          }
          ++status_readbacks;
          generation_status_available = true;
          (void)span;
        }
        decode_steps = step + 1U;
        if (generation_status[2] != 0) {
          decode_stopped_early = step + 1U < request.max_new_tokens;
          eos_stop_step = step;
          break;
        }
        if (step + 1U == request.max_new_tokens) {
          break;
        }
        {
          auto span = profiler.scoped("mpsgraph.decode.forward_next_token");
          auto status = cache->model.forward_next_token(context, position, state, &profiler);
          if (!status.is_ok()) {
            return status;
          }
          (void)span;
        }
        ++position;
        ++decode_forward_steps;
        (void)step_span;
      }
      const auto transfer_after = context.transfer_stats();
      if (host_to_device_delta(transfer_before, transfer_after) != 0U) {
        return Status::internal_error("MPSGraph decode performed host write");
      }
      (void)decode_span;
    }

    if (!generation_status_available) {
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
    result.text = cache->tokenizer.decode(generated);
    result.request_id = profiler.request_id();
    result.finish_reason = generation_status[1] == 1 ? "stop" : "length";
    result.prompt_tokens = prompt_tokens.size();
    result.generated_tokens = generated.size();
    result.kv_cache = to_public_report(state.kv_cache.stats());
    result.kv_cache_verified = false;
    profiler.set_metadata("generated_tokens", generated.size());
    profiler.set_metadata("mpsgraph_finish_reason", result.finish_reason);
    profiler.set_metadata("mpsgraph_decode_steps", decode_steps);
    profiler.set_metadata("mpsgraph_decode_forward_steps", decode_forward_steps);
    profiler.set_metadata("mpsgraph_status_readbacks", status_readbacks);
    profiler.set_metadata("mpsgraph_early_break", decode_stopped_early ? "true" : "false");
    profiler.set_metadata("mpsgraph_early_break_mode", "host_status_poll");
    profiler.set_metadata("mpsgraph_generation_status_count",
                          static_cast<std::size_t>(generation_status[0]));
    profiler.set_metadata("mpsgraph_generation_status_reason",
                          static_cast<std::size_t>(generation_status[1]));
    profiler.set_metadata("mpsgraph_generation_status_finished",
                          static_cast<std::size_t>(generation_status[2]));
    if (eos_stop_step != std::numeric_limits<std::size_t>::max()) {
      profiler.set_metadata("mpsgraph_eos_stop_step", eos_stop_step);
    }
    const auto transfers = context.transfer_stats();
    profiler.set_metadata("mpsgraph_h2d_calls",
                          static_cast<std::size_t>(
                            host_to_device_delta(request_transfer_before, transfers)));
    profiler.set_metadata("mpsgraph_d2h_calls",
                          static_cast<std::size_t>(
                            device_to_host_delta(request_transfer_before, transfers)));
    const auto graph_stats = context.graph_stats();
    profiler.set_metadata("mpsgraph_graph_build_calls",
                          static_cast<std::size_t>(graph_stat_delta(
                            request_graph_before.graph_build_calls,
                            graph_stats.graph_build_calls)));
    profiler.set_metadata("mpsgraph_graph_build_ns",
                          static_cast<std::size_t>(graph_stat_delta(
                            request_graph_before.graph_build_ns,
                            graph_stats.graph_build_ns)));
    profiler.set_metadata("mpsgraph_graph_compile_calls",
                          static_cast<std::size_t>(graph_stat_delta(
                            request_graph_before.graph_compile_calls,
                            graph_stats.graph_compile_calls)));
    profiler.set_metadata("mpsgraph_graph_compile_ns",
                          static_cast<std::size_t>(graph_stat_delta(
                            request_graph_before.graph_compile_ns,
                            graph_stats.graph_compile_ns)));
    profiler.set_metadata("mpsgraph_graph_execute_calls",
                          static_cast<std::size_t>(graph_stat_delta(
                            request_graph_before.graph_execute_calls,
                            graph_stats.graph_execute_calls)));
    profiler.set_metadata("mpsgraph_graph_execute_ns",
                          static_cast<std::size_t>(graph_stat_delta(
                            request_graph_before.graph_execute_ns,
                            graph_stats.graph_execute_ns)));
    profiler.set_metadata("mpsgraph_executable_cache_hit_count",
                          static_cast<std::size_t>(graph_stat_delta(
                            request_graph_before.executable_cache_hits,
                            graph_stats.executable_cache_hits)));
    profiler.set_metadata("mpsgraph_executable_cache_miss_count",
                          static_cast<std::size_t>(graph_stat_delta(
                            request_graph_before.executable_cache_misses,
                            graph_stats.executable_cache_misses)));
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

Status warmup_mpsgraph(const std::filesystem::path& model_dir,
                       std::size_t max_new_tokens) {
  if (max_new_tokens == 0) {
    return Status::invalid_argument("MPSGraph warmup max_new_tokens must be positive");
  }
  CpuGenerationRequest request;
  request.model_dir = model_dir;
  request.prompt = "hello";
  request.max_new_tokens = max_new_tokens;
  request.compute_device = Device::mpsgraph();
  request.observability.profile_mode = ProfileMode::off;
  const auto result = generate_mpsgraph(request);
  if (!result.is_ok()) {
    return result.status();
  }
  std::unique_lock<std::mutex> cache_lock(runtime_cache_mutex());
  auto& cache = runtime_cache_slot();
  const auto model_key = canonical_model_key(model_dir);
  if (cache != nullptr && cache->model_key == model_key) {
    cache->warmed_decode_capacity = std::max(cache->warmed_decode_capacity,
                                             result.value().kv_cache.capacity_tokens);
    cache->warmed_max_new_tokens =
      std::max(cache->warmed_max_new_tokens, max_new_tokens);
  }
  return Status::ok();
}

}  // namespace toyllm
