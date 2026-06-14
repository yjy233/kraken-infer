#pragma once

#include "toyllm/runtime/qwen_tokenizer.hpp"

namespace toyllm::cpu {

constexpr std::int64_t kEndOfText = toyllm::kQwenEndOfText;
constexpr std::int64_t kImStart = toyllm::kQwenImStart;
constexpr std::int64_t kImEnd = toyllm::kQwenImEnd;
constexpr std::int64_t kThinkStart = toyllm::kQwenThinkStart;
constexpr std::int64_t kThinkEnd = toyllm::kQwenThinkEnd;

}  // namespace toyllm::cpu
