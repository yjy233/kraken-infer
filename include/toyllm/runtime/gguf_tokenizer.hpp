#pragma once

#include "toyllm/core/status.hpp"
#include "toyllm/runtime/chat_message.hpp"
#include "toyllm/runtime/gguf_reader.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace toyllm {

enum class GgufTokenType {
  undefined = 0,
  normal = 1,
  unknown = 2,
  control = 3,
  user_defined = 4,
  unused = 5,
  byte = 6,
};

struct GgufTokenizer {
  std::string model;
  std::string pre;
  std::string chat_template;
  std::vector<std::string> tokens;
  std::vector<std::int64_t> token_types;
  std::vector<std::string> merges;
  std::unordered_map<std::string, std::int64_t> token_to_id;
  std::unordered_map<std::string, int> merge_ranks;
  std::int64_t bos_token_id{-1};
  std::int64_t eos_token_id{-1};
  std::int64_t pad_token_id{-1};
  bool add_bos_token{false};
  bool add_eos_token{false};
};

[[nodiscard]] const char* gguf_token_type_name(GgufTokenType type);
[[nodiscard]] Result<GgufTokenizer> load_gguf_tokenizer(const GgufFile& gguf);
[[nodiscard]] std::optional<std::int64_t> gguf_token_id(const GgufTokenizer& tokenizer,
                                                        std::string_view token);
[[nodiscard]] bool gguf_token_is_control(const GgufTokenizer& tokenizer, std::int64_t id);
[[nodiscard]] Result<std::vector<std::int64_t>> gguf_encode_text(
  const GgufTokenizer& tokenizer, std::string_view text, bool add_special,
  bool parse_special);
[[nodiscard]] Result<std::vector<std::int64_t>> gguf_encode_qwen35_chat_prompt(
  const GgufTokenizer& tokenizer, const std::vector<ChatMessage>& messages,
  bool add_generation_prompt, bool enable_thinking);
[[nodiscard]] Result<std::string> gguf_decode_token_text(const GgufTokenizer& tokenizer,
                                                         const std::vector<std::int64_t>& ids,
                                                         bool skip_control);
[[nodiscard]] Result<std::string> format_qwen35_chat_prompt(
  const GgufTokenizer& tokenizer, const std::vector<ChatMessage>& messages,
  bool add_generation_prompt, bool enable_thinking);

}  // namespace toyllm
