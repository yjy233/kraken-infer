#pragma once

#include "toyllm/runtime/cpu_inference.hpp"

#include <filesystem>

namespace toyllm {

[[nodiscard]] Result<CpuGenerationResult> generate_mpsgraph(const CpuGenerationRequest& request);
[[nodiscard]] Status warmup_mpsgraph(const std::filesystem::path& model_dir,
                                     std::size_t max_new_tokens);

}  // namespace toyllm
