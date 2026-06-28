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
  std::filesystem::path mmproj_path;
  std::string model_id{"kraken-infer-qwen3-0.6b"};
  Device compute_device{Device::cpu()};
  std::size_t default_max_tokens{16};
  std::size_t context_size{0};
  std::size_t parallel_slots{1};
  bool enable_mtp{true};
  bool mpsgraph_warmup{false};
  ObservabilityConfig observability;
};

[[nodiscard]] Status serve_openai_gateway(const OpenAIGatewayConfig& config);

}  // namespace toyllm
