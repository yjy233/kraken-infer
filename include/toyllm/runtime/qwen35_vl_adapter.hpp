#pragma once

#include "toyllm/runtime/cpu_inference.hpp"

namespace toyllm {

[[nodiscard]] Result<CpuGenerationResult> generate_qwen35_vl_with_llama_mtmd(
  const CpuGenerationRequest& request);

}  // namespace toyllm
