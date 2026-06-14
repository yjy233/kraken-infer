#include "toyllm/runtime/mpsgraph_inference.hpp"

#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/backends/mpsgraph/qwen_mpsgraph_model.hpp"

#include <limits>

namespace toyllm {

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

  auto context = mpsgraph::MpsGraphContext::create();
  if (!context.is_ok()) {
    return context.status();
  }

  auto model = mpsgraph::QwenMpsGraphModel::load_all_weights(request.model_dir,
                                                            context.value());
  if (!model.is_ok()) {
    return model.status();
  }

  if (request.max_new_tokens == std::numeric_limits<std::size_t>::max()) {
    return Status::invalid_argument("MPSGraph max_new_tokens exceeds supported capacity");
  }
  auto state = model.value().create_run_state(context.value(), request.max_new_tokens + 1U);
  if (!state.is_ok()) {
    return state.status();
  }

  return Status::unavailable("MPSGraph Qwen3 prefill/decode loop is not implemented yet");
}

}  // namespace toyllm
