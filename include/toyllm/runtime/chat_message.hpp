#pragma once

#include <string>

namespace toyllm {

struct ChatMessage {
  std::string role;
  std::string content;
};

}  // namespace toyllm
