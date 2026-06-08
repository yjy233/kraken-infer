#pragma once

#include "toyllm/core/device.hpp"
#include "toyllm/core/status.hpp"
#include "toyllm/runtime/profiling.hpp"

#include <cstddef>
#include <filesystem>
#include <string>

namespace toyllm {

struct OpenAIGatewayConfig {
  std::string host{"127.0.0.1"};
  int port{8080};
  std::filesystem::path model_dir{"models/qwen3-0.6b"};
  std::string model_id{"kraken-infer-qwen3-0.6b"};
  Device compute_device{Device::cpu()};
  std::size_t default_max_tokens{16};
  ObservabilityConfig observability;
};

[[nodiscard]] Status serve_openai_gateway(const OpenAIGatewayConfig& config);

}  // namespace toyllm
