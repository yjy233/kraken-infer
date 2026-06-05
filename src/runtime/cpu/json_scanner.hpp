#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace toyllm::cpu {

class JsonScanner {
 public:
  explicit JsonScanner(std::string_view input);

  void skip_ws();
  bool eof();
  void expect(char expected);
  bool consume(char expected);
  std::string parse_string();
  std::uint64_t parse_uint();
  std::vector<std::uint64_t> parse_uint_array();
  void skip_value();

 private:
  [[noreturn]] void fail(std::string_view message) const;
  void append_unicode_escape(std::string& result);
  static void append_utf8(std::string& result, unsigned int codepoint);

  std::string_view input_;
  std::size_t position_{0};
};

std::string read_text_file(const std::filesystem::path& path);

}  // namespace toyllm::cpu
