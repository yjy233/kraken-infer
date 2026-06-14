#include "toyllm/runtime/qwen_tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace toyllm {

namespace {

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string pair_key(const std::string& first, const std::string& second) {
  std::string key;
  key.reserve(first.size() + second.size() + 1);
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
  } else {
    output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  }
}

std::vector<std::uint32_t> utf8_codepoints(std::string_view text) {
  std::vector<std::uint32_t> result;
  for (std::size_t i = 0; i < text.size();) {
    const auto ch = static_cast<unsigned char>(text[i]);
    if ((ch & 0x80U) == 0U) {
      result.push_back(ch);
      ++i;
    } else if ((ch & 0xE0U) == 0xC0U && i + 1 < text.size()) {
      const auto cp =
        ((static_cast<std::uint32_t>(ch & 0x1FU)) << 6U) |
        static_cast<std::uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU);
      result.push_back(cp);
      i += 2;
    } else if ((ch & 0xF0U) == 0xE0U && i + 2 < text.size()) {
      const auto cp =
        ((static_cast<std::uint32_t>(ch & 0x0FU)) << 12U) |
        ((static_cast<std::uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU))
         << 6U) |
        static_cast<std::uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3FU);
      result.push_back(cp);
      i += 3;
    } else {
      result.push_back('?');
      ++i;
    }
  }
  return result;
}

std::pair<std::array<std::string, 256>, std::unordered_map<std::uint32_t, unsigned char>>
build_byte_maps() {
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

  std::array<std::string, 256> byte_encoder{};
  std::unordered_map<std::uint32_t, unsigned char> byte_decoder;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    std::string token;
    append_codepoint_utf8(token, static_cast<std::uint32_t>(codepoints[i]));
    byte_encoder[static_cast<std::size_t>(bytes[i])] = token;
    byte_decoder.emplace(static_cast<std::uint32_t>(codepoints[i]),
                         static_cast<unsigned char>(bytes[i]));
  }
  return {byte_encoder, byte_decoder};
}

}  // namespace

class QwenTokenizer::JsonScanner {
 public:
  explicit JsonScanner(std::string_view input) : input_(input) {}

  void skip_ws() {
    while (position_ < input_.size()) {
      const auto ch = static_cast<unsigned char>(input_[position_]);
      if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') {
        break;
      }
      ++position_;
    }
  }

  void expect(char expected) {
    skip_ws();
    if (position_ >= input_.size() || input_[position_] != expected) {
      std::string message = "expected JSON character ";
      message.push_back(expected);
      fail(message);
    }
    ++position_;
  }

  bool consume(char expected) {
    skip_ws();
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  std::string parse_string() {
    expect('"');
    std::string result;
    while (position_ < input_.size()) {
      const char ch = input_[position_++];
      if (ch == '"') {
        return result;
      }
      if (ch != '\\') {
        result.push_back(ch);
        continue;
      }
      if (position_ >= input_.size()) {
        fail("unterminated JSON escape");
      }
      const char escaped = input_[position_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          result.push_back(escaped);
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u':
          append_unicode_escape(result);
          break;
        default:
          fail("unsupported JSON escape");
      }
    }
    fail("unterminated JSON string");
  }

  std::uint64_t parse_uint() {
    skip_ws();
    if (position_ >= input_.size() || input_[position_] < '0' ||
        input_[position_] > '9') {
      fail("expected unsigned JSON integer");
    }
    std::uint64_t value = 0;
    while (position_ < input_.size() && input_[position_] >= '0' &&
           input_[position_] <= '9') {
      const auto digit = static_cast<std::uint64_t>(input_[position_] - '0');
      if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
        fail("JSON integer overflow");
      }
      value = value * 10U + digit;
      ++position_;
    }
    return value;
  }

  void skip_value() {
    skip_ws();
    if (position_ >= input_.size()) {
      fail("unexpected end of JSON value");
    }
    const char ch = input_[position_];
    if (ch == '"') {
      (void)parse_string();
      return;
    }
    if (ch == '{') {
      expect('{');
      if (consume('}')) {
        return;
      }
      while (true) {
        (void)parse_string();
        expect(':');
        skip_value();
        if (consume('}')) {
          return;
        }
        expect(',');
      }
    }
    if (ch == '[') {
      expect('[');
      if (consume(']')) {
        return;
      }
      while (true) {
        skip_value();
        if (consume(']')) {
          return;
        }
        expect(',');
      }
    }
    if ((ch >= '0' && ch <= '9') || ch == '-') {
      if (ch == '-') {
        ++position_;
      }
      while (position_ < input_.size()) {
        const char current = input_[position_];
        if ((current >= '0' && current <= '9') || current == '.' || current == 'e' ||
            current == 'E' || current == '+' || current == '-') {
          ++position_;
          continue;
        }
        break;
      }
      return;
    }
    if (input_.substr(position_, 4) == "true") {
      position_ += 4;
      return;
    }
    if (input_.substr(position_, 5) == "false") {
      position_ += 5;
      return;
    }
    if (input_.substr(position_, 4) == "null") {
      position_ += 4;
      return;
    }
    fail("unsupported JSON value");
  }

 private:
  [[noreturn]] void fail(std::string_view message) const {
    std::ostringstream output;
    output << message << " at byte " << position_;
    throw std::runtime_error(output.str());
  }

  void append_unicode_escape(std::string& result) {
    if (position_ + 4 > input_.size()) {
      fail("invalid JSON unicode escape");
    }
    unsigned int codepoint = 0;
    for (int i = 0; i < 4; ++i) {
      const char ch = input_[position_++];
      codepoint <<= 4U;
      if (ch >= '0' && ch <= '9') {
        codepoint += static_cast<unsigned int>(ch - '0');
      } else if (ch >= 'a' && ch <= 'f') {
        codepoint += static_cast<unsigned int>(ch - 'a' + 10);
      } else if (ch >= 'A' && ch <= 'F') {
        codepoint += static_cast<unsigned int>(ch - 'A' + 10);
      } else {
        fail("invalid JSON unicode escape");
      }
    }
    append_codepoint_utf8(result, codepoint);
  }

  std::string_view input_;
  std::size_t position_{0};
};

QwenTokenizer QwenTokenizer::load(const std::filesystem::path& model_dir) {
  QwenTokenizer tokenizer;
  tokenizer.load_vocab(model_dir / "vocab.json");
  tokenizer.load_added_tokens(model_dir / "tokenizer_config.json");
  tokenizer.load_merges(model_dir / "merges.txt");
  auto maps = build_byte_maps();
  tokenizer.byte_encoder_ = std::move(maps.first);
  tokenizer.byte_decoder_ = std::move(maps.second);
  return tokenizer;
}

std::vector<std::int64_t> QwenTokenizer::encode_chat_messages(
  const std::vector<ChatMessage>& messages, bool enable_thinking) {
  std::vector<std::int64_t> tokens;
  for (const auto& message : messages) {
    if (message.role != "system" && message.role != "user" && message.role != "assistant") {
      throw std::runtime_error("unsupported chat message role: " + message.role);
    }
    tokens.push_back(kQwenImStart);
    append_tokens(tokens, encode_text(message.role + "\n" + message.content));
    tokens.push_back(kQwenImEnd);
    append_tokens(tokens, encode_text("\n"));
  }
  tokens.push_back(kQwenImStart);
  append_tokens(tokens, encode_text("assistant\n"));
  if (!enable_thinking) {
    tokens.push_back(kQwenThinkStart);
    append_tokens(tokens, encode_text("\n\n"));
    tokens.push_back(kQwenThinkEnd);
    append_tokens(tokens, encode_text("\n\n"));
  }
  return tokens;
}

std::string QwenTokenizer::decode(const std::vector<std::int64_t>& ids) const {
  std::string bytes;
  for (const auto id : ids) {
    if (id == kQwenEndOfText || id == kQwenImStart || id == kQwenImEnd) {
      continue;
    }
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
      continue;
    }
    for (const auto cp : utf8_codepoints(id_to_token_[static_cast<std::size_t>(id)])) {
      const auto it = byte_decoder_.find(cp);
      if (it != byte_decoder_.end()) {
        bytes.push_back(static_cast<char>(it->second));
      }
    }
  }
  return bytes;
}

void QwenTokenizer::load_vocab(const std::filesystem::path& path) {
  const auto text = read_text_file(path);
  JsonScanner scanner(text);
  scanner.expect('{');
  if (scanner.consume('}')) {
    return;
  }
  while (true) {
    const std::string token = scanner.parse_string();
    scanner.expect(':');
    const auto id = scanner.parse_uint();
    add_token(static_cast<std::int64_t>(id), token);
    if (scanner.consume('}')) {
      return;
    }
    scanner.expect(',');
  }
}

void QwenTokenizer::load_added_tokens(const std::filesystem::path& path) {
  const auto text = read_text_file(path);
  JsonScanner scanner(text);
  scanner.expect('{');
  if (scanner.consume('}')) {
    return;
  }
  while (true) {
    const auto key = scanner.parse_string();
    scanner.expect(':');
    if (key == "added_tokens_decoder") {
      parse_added_tokens_decoder(scanner);
    } else {
      scanner.skip_value();
    }
    if (scanner.consume('}')) {
      return;
    }
    scanner.expect(',');
  }
}

void QwenTokenizer::parse_added_tokens_decoder(JsonScanner& scanner) {
  scanner.expect('{');
  if (scanner.consume('}')) {
    return;
  }
  while (true) {
    const auto id_text = scanner.parse_string();
    const auto id = parse_token_id(id_text);
    std::string content;
    scanner.expect(':');
    scanner.expect('{');
    while (true) {
      const auto key = scanner.parse_string();
      scanner.expect(':');
      if (key == "content") {
        content = scanner.parse_string();
      } else {
        scanner.skip_value();
      }
      if (scanner.consume('}')) {
        break;
      }
      scanner.expect(',');
    }
    if (!content.empty()) {
      add_token(id, content);
    }
    if (scanner.consume('}')) {
      return;
    }
    scanner.expect(',');
  }
}

std::int64_t QwenTokenizer::parse_token_id(const std::string& text) {
  std::size_t parsed = 0;
  const auto value = std::stoll(text, &parsed, 10);
  if (parsed != text.size()) {
    throw std::runtime_error("invalid tokenizer added token id: " + text);
  }
  return value;
}

void QwenTokenizer::add_token(std::int64_t id, const std::string& token) {
  if (id < 0) {
    throw std::runtime_error("negative tokenizer token id");
  }
  token_to_id_[token] = id;
  const auto index = static_cast<std::size_t>(id);
  if (index >= id_to_token_.size()) {
    id_to_token_.resize(index + 1U);
  }
  id_to_token_[index] = token;
}

void QwenTokenizer::load_merges(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::string line;
  int rank = 0;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto split = line.find(' ');
    if (split == std::string::npos) {
      continue;
    }
    merge_ranks_.emplace(pair_key(line.substr(0, split), line.substr(split + 1)), rank++);
  }
}

std::vector<std::int64_t> QwenTokenizer::encode_text(const std::string& text) {
  std::vector<std::string> word;
  word.reserve(text.size());
  for (const auto ch : text) {
    word.push_back(byte_encoder_[static_cast<unsigned char>(ch)]);
  }
  return bpe(word);
}

std::vector<std::int64_t> QwenTokenizer::bpe(std::vector<std::string> word) {
  if (word.empty()) {
    return {};
  }

  while (word.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    std::size_t best_index = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i + 1 < word.size(); ++i) {
      const auto it = merge_ranks_.find(pair_key(word[i], word[i + 1]));
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_index = i;
      }
    }
    if (best_index == std::numeric_limits<std::size_t>::max()) {
      break;
    }

    std::vector<std::string> merged;
    merged.reserve(word.size() - 1);
    for (std::size_t i = 0; i < word.size();) {
      if (i + 1 < word.size() && i == best_index) {
        merged.push_back(word[i] + word[i + 1]);
        i += 2;
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
    const auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
      ids.push_back(it->second);
    }
  }
  return ids;
}

void QwenTokenizer::append_tokens(std::vector<std::int64_t>& target,
                                  const std::vector<std::int64_t>& source) {
  target.insert(target.end(), source.begin(), source.end());
}

}  // namespace toyllm
