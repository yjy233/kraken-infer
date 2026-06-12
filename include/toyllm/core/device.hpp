#pragma once

#include <cstdint>
#include <string>

namespace toyllm {

enum class DeviceKind {
  cpu,
  mps,
  mpsgraph,
};

struct Device {
  DeviceKind kind{DeviceKind::cpu};
  std::int32_t index{0};

  [[nodiscard]] static constexpr Device cpu() { return Device{DeviceKind::cpu, 0}; }
  [[nodiscard]] static constexpr Device mps(std::int32_t index = 0) {
    return Device{DeviceKind::mps, index};
  }
  [[nodiscard]] static constexpr Device mpsgraph() {
    return Device{DeviceKind::mpsgraph, 0};
  }

  [[nodiscard]] std::string to_string() const {
    switch (kind) {
      case DeviceKind::cpu:
        return "cpu";
      case DeviceKind::mps:
        return "mps:" + std::to_string(index);
      case DeviceKind::mpsgraph:
        return "mpsgraph";
    }
    return "unknown";
  }
};

[[nodiscard]] inline bool operator==(const Device& lhs, const Device& rhs) {
  return lhs.kind == rhs.kind && lhs.index == rhs.index;
}

[[nodiscard]] inline bool operator!=(const Device& lhs, const Device& rhs) {
  return !(lhs == rhs);
}

}  // namespace toyllm
