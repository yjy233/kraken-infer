#include "toyllm/runtime/runtime.hpp"

#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/backends/mps/mps_backend.hpp"

namespace toyllm {

Runtime::Runtime(RuntimeConfig config) : config_(config) {}

const RuntimeConfig& Runtime::config() const {
  return config_;
}

RuntimeInfo Runtime::info() const {
  RuntimeInfo info{};
  info.selected_device = Device::cpu();

  if (config_.preferred_device.kind == DeviceKind::mps) {
    const auto backend = mps::query_backend();
    info.accelerator_available = backend.available;
    info.accelerator_name = backend.device_name;
    if (backend.available) {
      info.selected_device = config_.preferred_device;
    }
  } else if (config_.preferred_device.kind == DeviceKind::mpsgraph) {
    const auto backend = mpsgraph::query_backend();
    info.accelerator_available = backend.available;
    info.accelerator_name = backend.device_name;
    if (backend.available) {
      info.selected_device = config_.preferred_device;
    }
  }

  return info;
}

}  // namespace toyllm
