#include "toyllm/runtime/gguf_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <string>
#include <variant>

namespace toyllm {

namespace {

struct CodepointSpan {
  std::uint32_t codepoint{0};
  std::size_t byte_offset{0};
  std::size_t byte_length{0};
};

struct UnicodeFlags {
  bool is_number{false};
  bool is_letter{false};
  bool is_accent_mark{false};
  bool is_whitespace{false};

  [[nodiscard]] bool any() const {
    return is_number || is_letter || is_accent_mark || is_whitespace;
  }
};

struct Fragment {
  bool token{false};
  std::int64_t token_id{-1};
  std::size_t offset{0};
  std::size_t length{0};
};

Result<std::vector<std::string>> get_string_array(const GgufFile& gguf,
                                                  const std::string& key) {
  const auto* value = find_gguf_metadata(gguf, key);
  if (value == nullptr) {
    return Status::invalid_argument("missing GGUF metadata key: " + key);
  }
  if (!std::holds_alternative<std::vector<std::string>>(value->value)) {
    return Status::invalid_argument("GGUF metadata key is not string array: " + key);
  }
  return std::get<std::vector<std::string>>(value->value);
}

std::vector<std::int64_t> optional_i64_array(const GgufFile& gguf, const std::string& key,
                                             std::size_t fallback_size) {
  auto value = gguf_get_i64_array(gguf, key);
  if (value.is_ok()) {
    return value.value();
  }
  return std::vector<std::int64_t>(fallback_size,
                                  static_cast<std::int64_t>(GgufTokenType::normal));
}

std::string optional_string(const GgufFile& gguf, const std::string& key) {
  auto value = gguf_get_string(gguf, key);
  if (!value.is_ok()) {
    return {};
  }
  return value.value();
}

std::int64_t optional_i64(const GgufFile& gguf, const std::string& key,
                          std::int64_t fallback) {
  auto value = gguf_get_i64(gguf, key);
  if (!value.is_ok()) {
    return fallback;
  }
  return value.value();
}

bool optional_bool(const GgufFile& gguf, const std::string& key, bool fallback) {
  auto value = gguf_get_bool(gguf, key);
  if (!value.is_ok()) {
    return fallback;
  }
  return value.value();
}

bool valid_token_id(const GgufTokenizer& tokenizer, std::int64_t id) {
  return id >= 0 && static_cast<std::size_t>(id) < tokenizer.tokens.size();
}

std::string pair_key(const std::string& first, const std::string& second) {
  std::string key;
  key.reserve(first.size() + second.size() + 1U);
  key.append(first);
  key.push_back('\x1f');
  key.append(second);
  return key;
}

void append_codepoint_utf8(std::string& output, std::uint32_t codepoint) {
  if (codepoint <= 0x7FU) {
    output.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FFU) {
    output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else if (codepoint <= 0xFFFFU) {
    output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else {
    output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  }
}

std::vector<CodepointSpan> utf8_codepoint_spans(std::string_view text) {
  std::vector<CodepointSpan> spans;
  spans.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    const auto byte0 = static_cast<unsigned char>(text[i]);
    if ((byte0 & 0x80U) == 0U) {
      spans.push_back(CodepointSpan{byte0, i, 1});
      ++i;
      continue;
    }
    if ((byte0 & 0xE0U) == 0xC0U && i + 1U < text.size()) {
      const auto byte1 = static_cast<unsigned char>(text[i + 1U]);
      if ((byte1 & 0xC0U) == 0x80U) {
        const auto cp = (static_cast<std::uint32_t>(byte0 & 0x1FU) << 6U) |
                        static_cast<std::uint32_t>(byte1 & 0x3FU);
        spans.push_back(CodepointSpan{cp, i, 2});
        i += 2U;
        continue;
      }
    }
    if ((byte0 & 0xF0U) == 0xE0U && i + 2U < text.size()) {
      const auto byte1 = static_cast<unsigned char>(text[i + 1U]);
      const auto byte2 = static_cast<unsigned char>(text[i + 2U]);
      if ((byte1 & 0xC0U) == 0x80U && (byte2 & 0xC0U) == 0x80U) {
        const auto cp = (static_cast<std::uint32_t>(byte0 & 0x0FU) << 12U) |
                        (static_cast<std::uint32_t>(byte1 & 0x3FU) << 6U) |
                        static_cast<std::uint32_t>(byte2 & 0x3FU);
        spans.push_back(CodepointSpan{cp, i, 3});
        i += 3U;
        continue;
      }
    }
    if ((byte0 & 0xF8U) == 0xF0U && i + 3U < text.size()) {
      const auto byte1 = static_cast<unsigned char>(text[i + 1U]);
      const auto byte2 = static_cast<unsigned char>(text[i + 2U]);
      const auto byte3 = static_cast<unsigned char>(text[i + 3U]);
      if ((byte1 & 0xC0U) == 0x80U && (byte2 & 0xC0U) == 0x80U &&
          (byte3 & 0xC0U) == 0x80U) {
        const auto cp = (static_cast<std::uint32_t>(byte0 & 0x07U) << 18U) |
                        (static_cast<std::uint32_t>(byte1 & 0x3FU) << 12U) |
                        (static_cast<std::uint32_t>(byte2 & 0x3FU) << 6U) |
                        static_cast<std::uint32_t>(byte3 & 0x3FU);
        spans.push_back(CodepointSpan{cp, i, 4});
        i += 4U;
        continue;
      }
    }
    spans.push_back(CodepointSpan{0xFFFDU, i, 1});
    ++i;
  }
  return spans;
}

std::uint32_t ascii_tolower(std::uint32_t codepoint) {
  if (codepoint >= 'A' && codepoint <= 'Z') {
    return codepoint + ('a' - 'A');
  }
  return codepoint;
}

bool in_range(std::uint32_t value, std::uint32_t first, std::uint32_t last) {
  return value >= first && value <= last;
}

bool is_unicode_number(std::uint32_t codepoint) {
  return in_range(codepoint, '0', '9') || in_range(codepoint, 0xFF10U, 0xFF19U);
}

bool is_unicode_accent_mark(std::uint32_t codepoint) {
  return in_range(codepoint, 0x0300U, 0x036FU) ||
         in_range(codepoint, 0x1AB0U, 0x1AFFU) ||
         in_range(codepoint, 0x1DC0U, 0x1DFFU) ||
         in_range(codepoint, 0x20D0U, 0x20FFU) ||
         in_range(codepoint, 0xFE20U, 0xFE2FU);
}

bool is_unicode_whitespace(std::uint32_t codepoint) {
  switch (codepoint) {
    case 0x0009U:
    case 0x000AU:
    case 0x000BU:
    case 0x000CU:
    case 0x000DU:
    case 0x0020U:
    case 0x0085U:
    case 0x00A0U:
    case 0x1680U:
    case 0x2000U:
    case 0x2001U:
    case 0x2002U:
    case 0x2003U:
    case 0x2004U:
    case 0x2005U:
    case 0x2006U:
    case 0x2007U:
    case 0x2008U:
    case 0x2009U:
    case 0x200AU:
    case 0x2028U:
    case 0x2029U:
    case 0x202FU:
    case 0x205FU:
    case 0x3000U:
      return true;
    default:
      return false;
  }
}

bool is_unicode_letter(std::uint32_t codepoint) {
  if (in_range(codepoint, 'A', 'Z') || in_range(codepoint, 'a', 'z')) {
    return true;
  }
  if (in_range(codepoint, 0x00C0U, 0x02AFU)) {
    return codepoint != 0x00D7U && codepoint != 0x00F7U;
  }
  return in_range(codepoint, 0x0370U, 0x03FFU) ||
         in_range(codepoint, 0x0400U, 0x052FU) ||
         in_range(codepoint, 0x0530U, 0x058FU) ||
         in_range(codepoint, 0x0590U, 0x05FFU) ||
         in_range(codepoint, 0x0600U, 0x06FFU) ||
         in_range(codepoint, 0x0900U, 0x097FU) ||
         in_range(codepoint, 0x3040U, 0x30FFU) ||
         in_range(codepoint, 0x3400U, 0x4DBFU) ||
         in_range(codepoint, 0x4E00U, 0x9FFFU) ||
         in_range(codepoint, 0xAC00U, 0xD7AFU) ||
         in_range(codepoint, 0xF900U, 0xFAFFU) ||
         in_range(codepoint, 0x20000U, 0x2FA1FU);
}

UnicodeFlags unicode_flags(std::uint32_t codepoint) {
  return UnicodeFlags{
    is_unicode_number(codepoint),
    is_unicode_letter(codepoint),
    is_unicode_accent_mark(codepoint),
    is_unicode_whitespace(codepoint),
  };
}

std::size_t span_end_byte(const std::vector<CodepointSpan>& spans, std::size_t index,
                          std::string_view text) {
  if (index >= spans.size()) {
    return text.size();
  }
  return spans[index].byte_offset;
}

std::string byte_encoded(std::string_view text,
                         const std::array<std::string, 256>& byte_encoder) {
  std::string output;
  output.reserve(text.size());
  for (const auto ch : text) {
    output += byte_encoder[static_cast<unsigned char>(ch)];
  }
  return output;
}

std::array<std::string, 256> build_byte_encoder() {
  std::vector<int> bytes;
  for (int i = 33; i <= 126; ++i) {
    bytes.push_back(i);
  }
  for (int i = 161; i <= 172; ++i) {
    bytes.push_back(i);
  }
  for (int i = 174; i <= 255; ++i) {
    bytes.push_back(i);
  }

  std::vector<int> codepoints = bytes;
  int extra = 0;
  for (int byte = 0; byte < 256; ++byte) {
    if (std::find(bytes.begin(), bytes.end(), byte) == bytes.end()) {
      bytes.push_back(byte);
      codepoints.push_back(256 + extra);
      ++extra;
    }
  }

  std::array<std::string, 256> encoder{};
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    std::string token;
    append_codepoint_utf8(token, static_cast<std::uint32_t>(codepoints[i]));
    encoder[static_cast<std::size_t>(bytes[i])] = token;
  }
  return encoder;
}

std::unordered_map<std::uint32_t, unsigned char> build_byte_decoder() {
  const auto encoder = build_byte_encoder();
  std::unordered_map<std::uint32_t, unsigned char> decoder;
  for (std::size_t byte = 0; byte < encoder.size(); ++byte) {
    const auto spans = utf8_codepoint_spans(encoder[byte]);
    if (spans.size() == 1U) {
      decoder.emplace(spans[0].codepoint, static_cast<unsigned char>(byte));
    }
  }
  return decoder;
}

const std::array<std::string, 256>& byte_encoder() {
  static const auto encoder = build_byte_encoder();
  return encoder;
}

const std::unordered_map<std::uint32_t, unsigned char>& byte_decoder() {
  static const auto decoder = build_byte_decoder();
  return decoder;
}

std::vector<std::string> utf8_codepoint_strings(std::string_view text) {
  const auto spans = utf8_codepoint_spans(text);
  std::vector<std::string> output;
  output.reserve(spans.size());
  for (const auto& span : spans) {
    output.emplace_back(text.substr(span.byte_offset, span.byte_length));
  }
  return output;
}

std::vector<std::string> split_qwen35_words(std::string_view text) {
  static constexpr std::uint32_t out_of_range = 0xFFFFFFFFU;

  const auto spans = utf8_codepoint_spans(text);
  std::vector<std::string> words;
  words.reserve(spans.size());

  auto get_cpt = [&](std::size_t pos) -> std::uint32_t {
    return pos < spans.size() ? spans[pos].codepoint : out_of_range;
  };
  auto get_flags = [&](std::size_t pos) -> UnicodeFlags {
    return pos < spans.size() ? unicode_flags(spans[pos].codepoint) : UnicodeFlags{};
  };
  auto add_word = [&](std::size_t start, std::size_t end) {
    if (end <= start) {
      return;
    }
    const auto byte_start = spans[start].byte_offset;
    const auto byte_end = span_end_byte(spans, end, text);
    words.push_back(byte_encoded(text.substr(byte_start, byte_end - byte_start),
                                 byte_encoder()));
  };

  for (std::size_t pos = 0; pos < spans.size();) {
    const std::size_t token_start = pos;
    const auto cpt = get_cpt(pos);
    const auto flags = get_flags(pos);

    if (cpt == '\'' && pos + 1U < spans.size()) {
      const auto next = ascii_tolower(get_cpt(pos + 1U));
      if (next == 's' || next == 't' || next == 'm' || next == 'd') {
        pos += 2U;
        add_word(token_start, pos);
        continue;
      }
      if (pos + 2U < spans.size()) {
        const auto next_next = ascii_tolower(get_cpt(pos + 2U));
        if ((next == 'r' && next_next == 'e') ||
            (next == 'v' && next_next == 'e') ||
            (next == 'l' && next_next == 'l')) {
          pos += 3U;
          add_word(token_start, pos);
          continue;
        }
      }
    }

    if (cpt != '\r' && cpt != '\n' && !flags.is_number) {
      const auto next_flags = get_flags(pos + 1U);
      if (flags.is_letter || flags.is_accent_mark || next_flags.is_accent_mark ||
          next_flags.is_letter) {
        ++pos;
        while (get_flags(pos).is_letter || get_flags(pos).is_accent_mark) {
          ++pos;
        }
        add_word(token_start, pos);
        continue;
      }
    }

    if (flags.is_number) {
      ++pos;
      add_word(token_start, pos);
      continue;
    }

    auto flags2 = cpt == ' ' ? get_flags(pos + 1U) : flags;
    if (!(flags2.is_whitespace || flags2.is_letter || flags2.is_accent_mark ||
          flags2.is_number) &&
        flags.any()) {
      if (cpt == ' ') {
        ++pos;
      }
      while (!(flags2.is_whitespace || flags2.is_letter || flags2.is_accent_mark ||
               flags2.is_number) &&
             flags2.any()) {
        ++pos;
        flags2 = get_flags(pos);
      }
      while (get_cpt(pos) == '\r' || get_cpt(pos) == '\n') {
        ++pos;
      }
      add_word(token_start, pos);
      continue;
    }

    std::size_t num_whitespaces = 0;
    std::size_t last_end_r_or_n = 0;
    while (get_flags(pos + num_whitespaces).is_whitespace) {
      const auto cpt2 = get_cpt(pos + num_whitespaces);
      if (cpt2 == '\r' || cpt2 == '\n') {
        last_end_r_or_n = pos + num_whitespaces + 1U;
      }
      ++num_whitespaces;
    }

    if (last_end_r_or_n > 0U) {
      pos = last_end_r_or_n;
      add_word(token_start, pos);
      continue;
    }

    if (num_whitespaces > 1U && get_cpt(pos + num_whitespaces) != out_of_range) {
      pos += num_whitespaces - 1U;
      add_word(token_start, pos);
      continue;
    }

    if (num_whitespaces > 0U) {
      pos += num_whitespaces;
      add_word(token_start, pos);
      continue;
    }

    ++pos;
    add_word(token_start, pos);
  }

  return words;
}

std::vector<std::int64_t> bpe_encode_word(const GgufTokenizer& tokenizer,
                                          std::string_view encoded_word) {
  auto word = utf8_codepoint_strings(encoded_word);
  if (word.empty()) {
    return {};
  }

  while (word.size() > 1U) {
    int best_rank = std::numeric_limits<int>::max();
    std::size_t best_index = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i + 1U < word.size(); ++i) {
      const auto it = tokenizer.merge_ranks.find(pair_key(word[i], word[i + 1U]));
      if (it != tokenizer.merge_ranks.end() && it->second < best_rank) {
        best_rank = it->second;
        best_index = i;
      }
    }
    if (best_index == std::numeric_limits<std::size_t>::max()) {
      break;
    }

    std::vector<std::string> merged;
    merged.reserve(word.size() - 1U);
    for (std::size_t i = 0; i < word.size();) {
      if (i == best_index && i + 1U < word.size()) {
        merged.push_back(word[i] + word[i + 1U]);
        i += 2U;
      } else {
        merged.push_back(word[i]);
        ++i;
      }
    }
    word = std::move(merged);
  }

  std::vector<std::int64_t> ids;
  ids.reserve(word.size());
  for (const auto& token : word) {
    const auto it = tokenizer.token_to_id.find(token);
    if (it != tokenizer.token_to_id.end()) {
      ids.push_back(it->second);
      continue;
    }
    for (const auto ch : token) {
      const std::string byte_token(1, ch);
      const auto byte_it = tokenizer.token_to_id.find(byte_token);
      if (byte_it != tokenizer.token_to_id.end()) {
        ids.push_back(byte_it->second);
      }
    }
  }
  return ids;
}

bool token_type_matches_special(GgufTokenType type, bool parse_special) {
  if (type == GgufTokenType::user_defined) {
    return true;
  }
  if (!parse_special) {
    return false;
  }
  return type == GgufTokenType::control || type == GgufTokenType::unknown;
}

std::vector<std::int64_t> special_token_ids(const GgufTokenizer& tokenizer,
                                            bool parse_special) {
  std::vector<std::int64_t> ids;
  for (std::size_t i = 0; i < tokenizer.tokens.size(); ++i) {
    const auto type = static_cast<GgufTokenType>(tokenizer.token_types[i]);
    if (!tokenizer.tokens[i].empty() && token_type_matches_special(type, parse_special)) {
      ids.push_back(static_cast<std::int64_t>(i));
    }
  }
  std::sort(ids.begin(), ids.end(), [&](std::int64_t left, std::int64_t right) {
    const auto& left_text = tokenizer.tokens[static_cast<std::size_t>(left)];
    const auto& right_text = tokenizer.tokens[static_cast<std::size_t>(right)];
    if (left_text.size() != right_text.size()) {
      return left_text.size() > right_text.size();
    }
    return left < right;
  });
  return ids;
}

std::vector<Fragment> partition_special_tokens(const GgufTokenizer& tokenizer,
                                               const std::string& text,
                                               bool parse_special) {
  const auto specials = special_token_ids(tokenizer, parse_special);
  std::vector<Fragment> fragments;
  std::size_t raw_start = 0;
  std::size_t position = 0;

  while (position < text.size()) {
    std::int64_t match_id = -1;
    std::size_t match_length = 0;
    for (const auto id : specials) {
      const auto& token = tokenizer.tokens[static_cast<std::size_t>(id)];
      if (token.size() <= match_length || position + token.size() > text.size()) {
        continue;
      }
      if (text.compare(position, token.size(), token) == 0) {
        match_id = id;
        match_length = token.size();
      }
    }
    if (match_id < 0) {
      ++position;
      continue;
    }
    if (position > raw_start) {
      fragments.push_back(Fragment{false, -1, raw_start, position - raw_start});
    }
    fragments.push_back(Fragment{true, match_id, position, match_length});
    position += match_length;
    raw_start = position;
  }

  if (raw_start < text.size()) {
    fragments.push_back(Fragment{false, -1, raw_start, text.size() - raw_start});
  }
  return fragments;
}

Status append_special_bos(const GgufTokenizer& tokenizer, std::vector<std::int64_t>& ids) {
  if (!tokenizer.add_bos_token) {
    return Status::ok();
  }
  if (!valid_token_id(tokenizer, tokenizer.bos_token_id)) {
    return Status::invalid_argument("GGUF tokenizer add_bos_token is true but BOS id is invalid");
  }
  ids.push_back(tokenizer.bos_token_id);
  return Status::ok();
}

Status append_special_eos(const GgufTokenizer& tokenizer, std::vector<std::int64_t>& ids) {
  if (!tokenizer.add_eos_token) {
    return Status::ok();
  }
  if (!valid_token_id(tokenizer, tokenizer.eos_token_id)) {
    return Status::invalid_argument("GGUF tokenizer add_eos_token is true but EOS id is invalid");
  }
  ids.push_back(tokenizer.eos_token_id);
  return Status::ok();
}

void append_chat_message(std::ostringstream& prompt, const ChatMessage& message) {
  prompt << "<|im_start|>" << message.role << '\n' << message.content << "<|im_end|>\n";
}

}  // namespace

const char* gguf_token_type_name(GgufTokenType type) {
  switch (type) {
    case GgufTokenType::undefined:
      return "undefined";
    case GgufTokenType::normal:
      return "normal";
    case GgufTokenType::unknown:
      return "unknown";
    case GgufTokenType::control:
      return "control";
    case GgufTokenType::user_defined:
      return "user_defined";
    case GgufTokenType::unused:
      return "unused";
    case GgufTokenType::byte:
      return "byte";
  }
  return "unknown";
}

Result<GgufTokenizer> load_gguf_tokenizer(const GgufFile& gguf) {
  auto tokens = get_string_array(gguf, "tokenizer.ggml.tokens");
  if (!tokens.is_ok()) {
    return tokens.status();
  }

  GgufTokenizer tokenizer;
  tokenizer.model = optional_string(gguf, "tokenizer.ggml.model");
  tokenizer.pre = optional_string(gguf, "tokenizer.ggml.pre");
  tokenizer.chat_template = optional_string(gguf, "tokenizer.chat_template");
  tokenizer.tokens = tokens.value();
  tokenizer.token_types =
    optional_i64_array(gguf, "tokenizer.ggml.token_type", tokenizer.tokens.size());
  tokenizer.bos_token_id = optional_i64(gguf, "tokenizer.ggml.bos_token_id", -1);
  tokenizer.eos_token_id = optional_i64(gguf, "tokenizer.ggml.eos_token_id", -1);
  tokenizer.pad_token_id = optional_i64(gguf, "tokenizer.ggml.padding_token_id", -1);
  tokenizer.add_bos_token = optional_bool(gguf, "tokenizer.ggml.add_bos_token", false);
  tokenizer.add_eos_token = optional_bool(gguf, "tokenizer.ggml.add_eos_token", false);

  if (tokenizer.token_types.size() != tokenizer.tokens.size()) {
    return Status::invalid_argument("tokenizer.ggml.token_type size differs from tokens");
  }

  if (auto merges = get_string_array(gguf, "tokenizer.ggml.merges"); merges.is_ok()) {
    tokenizer.merges = merges.value();
  }
  tokenizer.merge_ranks.reserve(tokenizer.merges.size());
  for (std::size_t i = 0; i < tokenizer.merges.size(); ++i) {
    const auto& merge = tokenizer.merges[i];
    const auto split = merge.find(' ', 1);
    if (split == std::string::npos) {
      continue;
    }
    if (i > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      return Status::invalid_argument("GGUF tokenizer merge rank overflows int");
    }
    tokenizer.merge_ranks.emplace(pair_key(merge.substr(0, split), merge.substr(split + 1U)),
                                  static_cast<int>(i));
  }

  tokenizer.token_to_id.reserve(tokenizer.tokens.size());
  for (std::size_t id = 0; id < tokenizer.tokens.size(); ++id) {
    if (id > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
      return Status::invalid_argument("GGUF tokenizer token id overflows int64");
    }
    tokenizer.token_to_id.emplace(tokenizer.tokens[id], static_cast<std::int64_t>(id));
  }

  return tokenizer;
}

std::optional<std::int64_t> gguf_token_id(const GgufTokenizer& tokenizer,
                                          std::string_view token) {
  const auto it = tokenizer.token_to_id.find(std::string{token});
  if (it == tokenizer.token_to_id.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool gguf_token_is_control(const GgufTokenizer& tokenizer, std::int64_t id) {
  if (!valid_token_id(tokenizer, id)) {
    return false;
  }
  const auto type =
    static_cast<GgufTokenType>(tokenizer.token_types[static_cast<std::size_t>(id)]);
  return type == GgufTokenType::control || type == GgufTokenType::unused;
}

Result<std::vector<std::int64_t>> gguf_encode_text(const GgufTokenizer& tokenizer,
                                                   std::string_view text,
                                                   bool add_special,
                                                   bool parse_special) {
  if (tokenizer.model != "gpt2") {
    return Status::invalid_argument("GGUF encode currently supports gpt2/BPE tokenizers only");
  }
  if (tokenizer.pre != "qwen35") {
    return Status::invalid_argument("GGUF encode currently supports tokenizer pre qwen35 only");
  }

  std::vector<std::int64_t> ids;
  if (add_special) {
    const auto status = append_special_bos(tokenizer, ids);
    if (!status.is_ok()) {
      return status;
    }
  }

  const std::string owned_text{text};
  const auto fragments = partition_special_tokens(tokenizer, owned_text, parse_special);
  for (const auto& fragment : fragments) {
    if (fragment.token) {
      ids.push_back(fragment.token_id);
      continue;
    }
    const auto fragment_text =
      std::string_view{owned_text}.substr(fragment.offset, fragment.length);
    for (const auto& word : split_qwen35_words(fragment_text)) {
      const auto word_ids = bpe_encode_word(tokenizer, word);
      ids.insert(ids.end(), word_ids.begin(), word_ids.end());
    }
  }

  if (add_special) {
    const auto status = append_special_eos(tokenizer, ids);
    if (!status.is_ok()) {
      return status;
    }
  }
  return ids;
}

Result<std::vector<std::int64_t>> gguf_encode_qwen35_chat_prompt(
  const GgufTokenizer& tokenizer, const std::vector<ChatMessage>& messages,
  bool add_generation_prompt, bool enable_thinking) {
  auto prompt = format_qwen35_chat_prompt(tokenizer, messages, add_generation_prompt,
                                          enable_thinking);
  if (!prompt.is_ok()) {
    return prompt.status();
  }
  return gguf_encode_text(tokenizer, prompt.value(), false, true);
}

Result<std::string> gguf_decode_token_text(const GgufTokenizer& tokenizer,
                                           const std::vector<std::int64_t>& ids,
                                           bool skip_control) {
  std::string output;
  const auto& decoder = byte_decoder();
  for (const auto id : ids) {
    if (!valid_token_id(tokenizer, id)) {
      return Status::invalid_argument("GGUF token id out of range: " + std::to_string(id));
    }
    if (skip_control && gguf_token_is_control(tokenizer, id)) {
      continue;
    }
    const auto& token = tokenizer.tokens[static_cast<std::size_t>(id)];
    const auto spans = utf8_codepoint_spans(token);
    for (const auto& span : spans) {
      const auto it = decoder.find(span.codepoint);
      if (it != decoder.end()) {
        output.push_back(static_cast<char>(it->second));
      } else {
        output += token.substr(span.byte_offset, span.byte_length);
      }
    }
  }
  return output;
}
// 真理prompt 插入 <im_start>user <|im_endqq|>
Result<std::string> format_qwen35_chat_prompt(const GgufTokenizer& tokenizer,
                                              const std::vector<ChatMessage>& messages,
                                              bool add_generation_prompt,
                                              bool enable_thinking) {
  if (tokenizer.pre != "qwen35") {
    return Status::invalid_argument("Qwen3.5 chat formatter requires tokenizer pre qwen35");
  }
  if (!gguf_token_id(tokenizer, "<|im_start|>").has_value() ||
      !gguf_token_id(tokenizer, "<|im_end|>").has_value()) {
    return Status::invalid_argument("Qwen3.5 tokenizer is missing chat control tokens");
  }

  std::ostringstream prompt;
  for (const auto& message : messages) {
    if (message.role != "system" && message.role != "user" &&
        message.role != "assistant") {
      return Status::invalid_argument("unsupported chat message role: " + message.role);
    }
    append_chat_message(prompt, message);
  }
  // 就基本固定时true， 就时添加一个开头<|im_start|>user
  // 你好<|im_end|>
  // <|im_start|>assistant
  if (add_generation_prompt) {
    prompt << "<|im_start|>assistant\n";
    if (enable_thinking) {
      prompt << "<think>\n";
    } else {
      prompt << "<think>\n\n</think>\n\n";
    }
  }
  return prompt.str();
}

}  // namespace toyllm
