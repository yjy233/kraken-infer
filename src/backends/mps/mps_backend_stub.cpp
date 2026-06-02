#include "toyllm/backends/mps/mps_backend.hpp"

#include <sstream>

namespace toyllm::mps {

BackendInfo query_backend() {
  BackendInfo info{};
  info.failure_reason = "MPS backend was not compiled for this build";
  return info;
}

std::string format_backend_info(const BackendInfo& info) {
  std::ostringstream output;
  output << "MPS compiled: " << (info.compiled ? "yes" : "no") << '\n';
  output << "MPS available: " << (info.available ? "yes" : "no") << '\n';
  if (!info.failure_reason.empty()) {
    output << "Reason: " << info.failure_reason << '\n';
  }
  return output.str();
}

}  // namespace toyllm::mps
