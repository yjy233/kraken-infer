#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"

#include <sstream>

namespace toyllm::mpsgraph {

BackendInfo query_backend() {
  BackendInfo info{};
  info.failure_reason = "MPSGraph backend was not compiled for this build";
  return info;
}

std::string format_backend_info(const BackendInfo& info) {
  std::ostringstream output;
  output << "MPSGraph compiled: " << (info.compiled ? "yes" : "no") << '\n';
  output << "MPSGraph available: " << (info.available ? "yes" : "no") << '\n';
  output << "MPSGraph ready: " << (info.graph_ready ? "yes" : "no") << '\n';
  if (!info.device_name.empty()) {
    output << "Device: " << info.device_name << '\n';
  }
  if (!info.failure_reason.empty()) {
    output << "Reason: " << info.failure_reason << '\n';
  }
  return output.str();
}

Status run_operator_smoke_test() {
  return Status::unavailable("MPSGraph backend was not compiled for this build");
}

}  // namespace toyllm::mpsgraph
