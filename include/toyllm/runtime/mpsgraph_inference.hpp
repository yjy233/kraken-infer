#pragma once

#include "toyllm/runtime/cpu_inference.hpp"

namespace toyllm {

[[nodiscard]] Result<CpuGenerationResult> generate_mpsgraph(const CpuGenerationRequest& request);

}  // namespace toyllm
