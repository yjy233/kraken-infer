#include "json_scanner.hpp"

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace toyllm::cpu {

JsonScanner::JsonScanner(std::string_view input) : input_(input) {}

void JsonScanner::skip_ws() {
  while (position_ < input_.size()) {
    const auto ch = static_cast<unsigned char>(input_[position_]);
    if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') {
      break;
    }
    ++position_;
  }
}

bool JsonScanner::eof() {
  skip_ws();
  return position_ == input_.size();
}

void JsonScanner::expect(char expected) {
  skip_ws();
  if (position_ >= input_.size() || input_[position_] != expected) {
    std::string message = "expected JSON character ";
    message.push_back(expected);
    fail(message);
  }
  ++position_;
}

bool JsonScanner::consume(char expected) {
  skip_ws();
  if (position_ < input_.size() && input_[position_] == expected) {
    ++position_;
    return true;
  }
  return false;
}

std::string JsonScanner::parse_string() {
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

std::uint64_t JsonScanner::parse_uint() {
  skip_ws();
  if (position_ >= input_.size() || input_[position_] < '0' || input_[position_] > '9') {
    fail("expected unsigned JSON integer");
  }
  std::uint64_t value = 0;
  while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
    const auto digit = static_cast<std::uint64_t>(input_[position_] - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
      fail("JSON integer overflow");
    }
    value = value * 10U + digit;
    ++position_;
  }
  return value;
}

std::vector<std::uint64_t> JsonScanner::parse_uint_array() {
  std::vector<std::uint64_t> result;
  expect('[');
  if (consume(']')) {
    return result;
  }
  while (true) {
    result.push_back(parse_uint());
    if (consume(']')) {
      return result;
    }
    expect(',');
  }
}

void JsonScanner::skip_value() {
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

void JsonScanner::fail(std::string_view message) const {
  std::ostringstream output;
  output << message << " at byte " << position_;
  throw std::runtime_error(output.str());
}

void JsonScanner::append_unicode_escape(std::string& result) {
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
  append_utf8(result, codepoint);
}

void JsonScanner::append_utf8(std::string& result, unsigned int codepoint) {
  if (codepoint <= 0x7FU) {
    result.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FFU) {
    result.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
    result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else {
    result.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
    result.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  }
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

}  // namespace toyllm::cpu
