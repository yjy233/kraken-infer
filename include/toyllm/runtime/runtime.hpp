#pragma once

#include "toyllm/core/device.hpp"

#include <string>

namespace toyllm {

struct RuntimeConfig {
  Device preferred_device{Device::mps()};
};

struct RuntimeInfo {
  Device selected_device{Device::cpu()};
  bool accelerator_available{false};
  std::string accelerator_name;
};

class Runtime {
 public:
  explicit Runtime(RuntimeConfig config);

  [[nodiscard]] const RuntimeConfig& config() const;
  [[nodiscard]] RuntimeInfo info() const;

 private:
  RuntimeConfig config_;
};

}  // namespace toyllm
