#pragma once

#include <cstdint>
#include <string>

namespace toyllm::mps {

struct BackendInfo {
  bool compiled{false};
  bool available{false};
  std::string device_name;
  std::uint64_t recommended_max_working_set_size{0};
  bool low_power{false};
  bool headless{false};
  bool removable{false};
  std::string failure_reason;
};

[[nodiscard]] BackendInfo query_backend();
[[nodiscard]] std::string format_backend_info(const BackendInfo& info);

}  // namespace toyllm::mps
