#include "toyllm/runtime/mpsgraph_inference.hpp"

#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"

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

  return Status::unavailable("MPSGraph Qwen3 inference is not implemented yet");
}

}  // namespace toyllm
