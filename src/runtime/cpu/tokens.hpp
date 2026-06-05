#pragma once

#include <cstdint>

namespace toyllm::cpu {

constexpr std::int64_t kEndOfText = 151643;
constexpr std::int64_t kImStart = 151644;
constexpr std::int64_t kImEnd = 151645;
constexpr std::int64_t kThinkStart = 151667;
constexpr std::int64_t kThinkEnd = 151668;

}  // namespace toyllm::cpu
