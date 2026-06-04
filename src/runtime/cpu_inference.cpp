#include "toyllm/runtime/cpu_inference.hpp"

#include "toyllm/model/model_config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace toyllm {

namespace {

constexpr std::int64_t kEndOfText = 151643;
constexpr std::int64_t kImStart = 151644;
constexpr std::int64_t kImEnd = 151645;
constexpr std::int64_t kThinkStart = 151667;
constexpr std::int64_t kThinkEnd = 151668;

struct TensorView {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  const std::byte* data{nullptr};
  std::uint64_t data_offset_begin{0};
  std::uint64_t data_offset_end{0};
  std::uint64_t byte_size{0};
};

class JsonScanner {
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

  bool eof() {
    skip_ws();
    return position_ == input_.size();
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
    if (position_ >= input_.size() ||
        input_[position_] < '0' || input_[position_] > '9') {
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

  std::vector<std::uint64_t> parse_uint_array() {
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
    append_utf8(result, codepoint);
  }

  static void append_utf8(std::string& result, unsigned int codepoint) {
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

  std::string_view input_;
  std::size_t position_{0};
};

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
      const auto cp = ((static_cast<std::uint32_t>(ch & 0x1FU)) << 6U) |
                      static_cast<std::uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU);
      result.push_back(cp);
      i += 2;
    } else if ((ch & 0xF0U) == 0xE0U && i + 2 < text.size()) {
      const auto cp = ((static_cast<std::uint32_t>(ch & 0x0FU)) << 12U) |
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

class QwenTokenizer {
 public:
  static QwenTokenizer load(const std::filesystem::path& model_dir) {
    QwenTokenizer tokenizer;
    tokenizer.load_vocab(model_dir / "vocab.json");
    tokenizer.load_added_tokens(model_dir / "tokenizer_config.json");
    tokenizer.load_merges(model_dir / "merges.txt");
    auto maps = build_byte_maps();
    tokenizer.byte_encoder_ = std::move(maps.first);
    tokenizer.byte_decoder_ = std::move(maps.second);
    return tokenizer;
  }

  std::vector<std::int64_t> encode_chat_messages(const std::vector<ChatMessage>& messages,
                                                 bool enable_thinking) {
    std::vector<std::int64_t> tokens;
    for (const auto& message : messages) {
      if (message.role != "system" && message.role != "user" && message.role != "assistant") {
        throw std::runtime_error("unsupported chat message role: " + message.role);
      }
      tokens.push_back(kImStart);
      append_tokens(tokens, encode_text(message.role + "\n" + message.content));
      tokens.push_back(kImEnd);
      append_tokens(tokens, encode_text("\n"));
    }
    tokens.push_back(kImStart);
    append_tokens(tokens, encode_text("assistant\n"));
    if (!enable_thinking) {
      tokens.push_back(kThinkStart);
      append_tokens(tokens, encode_text("\n\n"));
      tokens.push_back(kThinkEnd);
      append_tokens(tokens, encode_text("\n\n"));
    }
    return tokens;
  }

  std::string decode(const std::vector<std::int64_t>& ids) const {
    std::string bytes;
    for (const auto id : ids) {
      if (id == kEndOfText || id == kImStart || id == kImEnd) {
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

 private:
  void load_vocab(const std::filesystem::path& path) {
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

  void load_added_tokens(const std::filesystem::path& path) {
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

  void parse_added_tokens_decoder(JsonScanner& scanner) {
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

  static std::int64_t parse_token_id(const std::string& text) {
    std::size_t parsed = 0;
    const auto value = std::stoll(text, &parsed, 10);
    if (parsed != text.size()) {
      throw std::runtime_error("invalid tokenizer added token id: " + text);
    }
    return value;
  }

  void add_token(std::int64_t id, const std::string& token) {
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

  void load_merges(const std::filesystem::path& path) {
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

  std::vector<std::int64_t> encode_text(const std::string& text) {
    std::vector<std::string> word;
    word.reserve(text.size());
    for (const auto ch : text) {
      word.push_back(byte_encoder_[static_cast<unsigned char>(ch)]);
    }
    return bpe(word);
  }

  std::vector<std::int64_t> bpe(std::vector<std::string> word) {
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

  static void append_tokens(std::vector<std::int64_t>& target,
                            const std::vector<std::int64_t>& source) {
    target.insert(target.end(), source.begin(), source.end());
  }

  std::array<std::string, 256> byte_encoder_{};
  std::unordered_map<std::uint32_t, unsigned char> byte_decoder_;
  std::unordered_map<std::string, std::int64_t> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merge_ranks_;
};

float bf16_to_float(const std::byte* data, std::uint64_t index) {
  const auto* raw = reinterpret_cast<const std::uint16_t*>(data);
  const std::uint32_t bits = static_cast<std::uint32_t>(raw[index]) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

std::uint64_t product(const std::vector<std::uint64_t>& shape) {
  std::uint64_t result = 1;
  for (const auto dim : shape) {
    if (dim != 0 && result > std::numeric_limits<std::uint64_t>::max() / dim) {
      throw std::runtime_error("shape product overflow");
    }
    result *= dim;
  }
  return result;
}

class SafeTensorMap {
 public:
  static SafeTensorMap load(const std::filesystem::path& path) {
    SafeTensorMap map;
    map.path_ = path;
    map.fd_ = ::open(path.c_str(), O_RDONLY);
    if (map.fd_ < 0) {
      throw std::runtime_error("failed to open " + path.string());
    }

    struct stat file_stat {};
    if (::fstat(map.fd_, &file_stat) != 0 || file_stat.st_size <= 8) {
      throw std::runtime_error("invalid safetensors file size");
    }
    map.file_size_ = static_cast<std::uint64_t>(file_stat.st_size);
    void* mapped = ::mmap(nullptr, static_cast<std::size_t>(map.file_size_), PROT_READ, MAP_PRIVATE,
                          map.fd_, 0);
    if (mapped == MAP_FAILED) {
      throw std::runtime_error("failed to mmap " + path.string());
    }
    map.mapped_data_ = static_cast<const std::byte*>(mapped);

    std::uint64_t header_size = 0;
    for (int i = 0; i < 8; ++i) {
      header_size |= static_cast<std::uint64_t>(
                       static_cast<unsigned char>(map.mapped_data_[static_cast<std::size_t>(i)]))
                     << (8U * static_cast<unsigned int>(i));
    }
    if (header_size == 0 || 8U + header_size >= map.file_size_) {
      throw std::runtime_error("invalid safetensors header size");
    }
    map.header_size_ = header_size;
    map.data_start_ = 8U + header_size;
    const auto* header_begin = reinterpret_cast<const char*>(map.mapped_data_ + 8);
    map.parse_header(std::string_view(header_begin, static_cast<std::size_t>(header_size)));
    map.validate_tensor_ranges();
    return map;
  }

  SafeTensorMap() = default;
  ~SafeTensorMap() { close(); }

  SafeTensorMap(const SafeTensorMap&) = delete;
  SafeTensorMap& operator=(const SafeTensorMap&) = delete;

  SafeTensorMap(SafeTensorMap&& other) noexcept { move_from(std::move(other)); }

  SafeTensorMap& operator=(SafeTensorMap&& other) noexcept {
    if (this != &other) {
      close();
      move_from(std::move(other));
    }
    return *this;
  }

  const TensorView& at(std::string_view name) const {
    const auto it = tensors_.find(std::string(name));
    if (it == tensors_.end()) {
      throw std::runtime_error("missing tensor: " + std::string(name));
    }
    return it->second;
  }

  std::uint64_t file_size() const { return file_size_; }

  std::uint64_t header_size() const { return header_size_; }

  const std::unordered_map<std::string, TensorView>& tensors() const { return tensors_; }

 private:
  void parse_header(std::string_view header) {
    JsonScanner scanner(header);
    scanner.expect('{');
    if (scanner.consume('}')) {
      return;
    }
    while (true) {
      const std::string name = scanner.parse_string();
      scanner.expect(':');
      if (name == "__metadata__") {
        scanner.skip_value();
      } else {
        TensorView tensor;
        tensor.name = name;
        scanner.expect('{');
        while (true) {
          const std::string key = scanner.parse_string();
          scanner.expect(':');
          if (key == "dtype") {
            tensor.dtype = scanner.parse_string();
            if (tensor.dtype != "BF16") {
              throw std::runtime_error("only BF16 safetensors are supported currently");
            }
          } else if (key == "shape") {
            tensor.shape = scanner.parse_uint_array();
          } else if (key == "data_offsets") {
            const auto offsets = scanner.parse_uint_array();
            if (offsets.size() != 2 || offsets[1] < offsets[0]) {
              throw std::runtime_error("invalid data_offsets for " + tensor.name);
            }
            tensor.byte_size = offsets[1] - offsets[0];
            if (data_start_ + offsets[1] > file_size_) {
              throw std::runtime_error("tensor data offset exceeds file size for " + tensor.name);
            }
            tensor.data_offset_begin = offsets[0];
            tensor.data_offset_end = offsets[1];
            tensor.data = mapped_data_ + data_start_ + offsets[0];
          } else {
            scanner.skip_value();
          }
          if (scanner.consume('}')) {
            break;
          }
          scanner.expect(',');
        }
        if (tensor.dtype.empty()) {
          throw std::runtime_error("missing tensor dtype for " + tensor.name);
        }
        if (product(tensor.shape) * 2U != tensor.byte_size) {
          throw std::runtime_error("tensor byte size mismatch for " + tensor.name);
        }
        tensors_.emplace(tensor.name, std::move(tensor));
      }
      if (scanner.consume('}')) {
        return;
      }
      scanner.expect(',');
    }
  }

  void validate_tensor_ranges() const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(tensors_.size());
    for (const auto& entry : tensors_) {
      ranges.push_back({entry.second.data_offset_begin, entry.second.data_offset_end});
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i) {
      if (ranges[i].first < ranges[i - 1].second) {
        throw std::runtime_error("safetensors tensor data offsets overlap");
      }
    }
  }

  void close() {
    if (mapped_data_ != nullptr) {
      (void)::munmap(const_cast<std::byte*>(mapped_data_), static_cast<std::size_t>(file_size_));
      mapped_data_ = nullptr;
    }
    if (fd_ >= 0) {
      (void)::close(fd_);
      fd_ = -1;
    }
  }

  void move_from(SafeTensorMap&& other) {
    path_ = std::move(other.path_);
    fd_ = other.fd_;
    mapped_data_ = other.mapped_data_;
    file_size_ = other.file_size_;
    header_size_ = other.header_size_;
    data_start_ = other.data_start_;
    tensors_ = std::move(other.tensors_);
    other.fd_ = -1;
    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
    other.header_size_ = 0;
    other.data_start_ = 0;
  }

  std::filesystem::path path_;
  int fd_{-1};
  const std::byte* mapped_data_{nullptr};
  std::uint64_t file_size_{0};
  std::uint64_t header_size_{0};
  std::uint64_t data_start_{0};
  std::unordered_map<std::string, TensorView> tensors_;
};

struct LayerWeights {
  TensorView input_norm;
  TensorView q_proj;
  TensorView q_norm;
  TensorView k_proj;
  TensorView k_norm;
  TensorView v_proj;
  TensorView o_proj;
  TensorView post_norm;
  TensorView gate_proj;
  TensorView up_proj;
  TensorView down_proj;
};

class CpuQwenModel {
 public:
  static CpuQwenModel load(const std::filesystem::path& model_dir) {
    CpuQwenModel model;
    auto bundle = load_model_bundle(model_dir);
    if (!bundle.is_ok()) {
      throw std::runtime_error(bundle.status().message());
    }
    model.config_ = bundle.value().model;
    model.generation_ = bundle.value().generation;
    model.tokenizer_ = QwenTokenizer::load(model_dir);
    model.weights_ = SafeTensorMap::load(model_dir / "model.safetensors");
    model.bind_weights();
    return model;
  }

  std::string generate(const std::vector<ChatMessage>& messages, std::size_t max_new_tokens,
                       bool enable_thinking) {
    const auto prompt_tokens = tokenizer_.encode_chat_messages(messages, enable_thinking);
    if (prompt_tokens.empty()) {
      throw std::runtime_error("tokenizer produced no prompt tokens");
    }
    const auto total_tokens = prompt_tokens.size() + max_new_tokens + 1U;
    const auto layers = static_cast<std::size_t>(config_.num_hidden_layers);
    const auto kv_dim = static_cast<std::size_t>(config_.num_key_value_heads * config_.head_dim);
    key_cache_.assign(layers * total_tokens * kv_dim, 0.0F);
    value_cache_.assign(layers * total_tokens * kv_dim, 0.0F);
    max_seq_len_ = total_tokens;

    std::vector<float> hidden;
    std::size_t position = 0;
    for (const auto token : prompt_tokens) {
      hidden = forward_token(token, position);
      ++position;
    }

    std::vector<std::int64_t> generated;
    generated.reserve(max_new_tokens);
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
      const auto next_token = select_next_token(hidden);
      if (is_eos(next_token)) {
        break;
      }
      generated.push_back(next_token);
      hidden = forward_token(next_token, position);
      ++position;
    }
    return tokenizer_.decode(generated);
  }

 private:
  void bind_weights() {
    embedding_ = weights_.at("model.embed_tokens.weight");
    lm_head_ = weights_.at("lm_head.weight");
    final_norm_ = weights_.at("model.norm.weight");
    layers_.resize(static_cast<std::size_t>(config_.num_hidden_layers));
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      const auto prefix = "model.layers." + std::to_string(i) + ".";
      auto& layer = layers_[i];
      layer.input_norm = weights_.at(prefix + "input_layernorm.weight");
      layer.q_proj = weights_.at(prefix + "self_attn.q_proj.weight");
      layer.q_norm = weights_.at(prefix + "self_attn.q_norm.weight");
      layer.k_proj = weights_.at(prefix + "self_attn.k_proj.weight");
      layer.k_norm = weights_.at(prefix + "self_attn.k_norm.weight");
      layer.v_proj = weights_.at(prefix + "self_attn.v_proj.weight");
      layer.o_proj = weights_.at(prefix + "self_attn.o_proj.weight");
      layer.post_norm = weights_.at(prefix + "post_attention_layernorm.weight");
      layer.gate_proj = weights_.at(prefix + "mlp.gate_proj.weight");
      layer.up_proj = weights_.at(prefix + "mlp.up_proj.weight");
      layer.down_proj = weights_.at(prefix + "mlp.down_proj.weight");
    }
  }

  std::vector<float> forward_token(std::int64_t token, std::size_t position) {
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    std::vector<float> hidden(hidden_size);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      hidden[i] = bf16_to_float(embedding_.data, static_cast<std::uint64_t>(token) *
                                                   static_cast<std::uint64_t>(hidden_size) + i);
    }

    for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
      apply_layer(layers_[layer_index], layer_index, position, hidden);
    }

    std::vector<float> normed(hidden_size);
    rms_norm(hidden, final_norm_, normed);
    return normed;
  }

  void apply_layer(const LayerWeights& layer, std::size_t layer_index, std::size_t position,
                   std::vector<float>& hidden) {
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    const auto heads = static_cast<std::size_t>(config_.num_attention_heads);
    const auto kv_heads = static_cast<std::size_t>(config_.num_key_value_heads);
    const auto attn_dim = heads * head_dim;
    const auto kv_dim = kv_heads * head_dim;
    const auto intermediate = static_cast<std::size_t>(config_.intermediate_size);

    std::vector<float> normed(hidden_size);
    rms_norm(hidden, layer.input_norm, normed);

    std::vector<float> q(attn_dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);
    matvec(layer.q_proj, normed, q);
    matvec(layer.k_proj, normed, k);
    matvec(layer.v_proj, normed, v);
    qk_norm(q, heads, layer.q_norm);
    qk_norm(k, kv_heads, layer.k_norm);
    apply_rope(q, heads, position);
    apply_rope(k, kv_heads, position);
    store_kv(layer_index, position, k, v);

    std::vector<float> attn_out(attn_dim);
    attention(layer_index, position, q, attn_out);
    std::vector<float> projected(hidden_size);
    matvec(layer.o_proj, attn_out, projected);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      hidden[i] += projected[i];
    }

    rms_norm(hidden, layer.post_norm, normed);
    std::vector<float> gate(intermediate);
    std::vector<float> up(intermediate);
    matvec(layer.gate_proj, normed, gate);
    matvec(layer.up_proj, normed, up);
    for (std::size_t i = 0; i < intermediate; ++i) {
      gate[i] = silu(gate[i]) * up[i];
    }
    std::vector<float> down(hidden_size);
    matvec(layer.down_proj, gate, down);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      hidden[i] += down[i];
    }
  }

  void rms_norm(const std::vector<float>& input, const TensorView& weight,
                std::vector<float>& output) const {
    double mean_square = 0.0;
    for (const auto value : input) {
      mean_square += static_cast<double>(value) * static_cast<double>(value);
    }
    mean_square /= static_cast<double>(input.size());
    const auto scale = static_cast<float>(1.0 / std::sqrt(mean_square + config_.rms_norm_eps));
    for (std::size_t i = 0; i < input.size(); ++i) {
      output[i] = input[i] * scale * bf16_to_float(weight.data, i);
    }
  }

  void qk_norm(std::vector<float>& values, std::size_t heads, const TensorView& weight) const {
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    for (std::size_t head = 0; head < heads; ++head) {
      double mean_square = 0.0;
      const auto base = head * head_dim;
      for (std::size_t i = 0; i < head_dim; ++i) {
        mean_square += static_cast<double>(values[base + i]) * values[base + i];
      }
      mean_square /= static_cast<double>(head_dim);
      const auto scale = static_cast<float>(1.0 / std::sqrt(mean_square + config_.rms_norm_eps));
      for (std::size_t i = 0; i < head_dim; ++i) {
        values[base + i] *= scale * bf16_to_float(weight.data, i);
      }
    }
  }

  void apply_rope(std::vector<float>& values, std::size_t heads, std::size_t position) const {
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    const auto half_dim = head_dim / 2U;
    for (std::size_t head = 0; head < heads; ++head) {
      const auto base = head * head_dim;
      for (std::size_t i = 0; i < half_dim; ++i) {
        const auto freq =
          1.0 / std::pow(config_.rope_theta,
                         static_cast<double>(2U * i) / static_cast<double>(head_dim));
        const auto angle = static_cast<double>(position) * freq;
        const auto cos_v = static_cast<float>(std::cos(angle));
        const auto sin_v = static_cast<float>(std::sin(angle));
        const auto x0 = values[base + i];
        const auto x1 = values[base + half_dim + i];
        values[base + i] = x0 * cos_v - x1 * sin_v;
        values[base + half_dim + i] = x1 * cos_v + x0 * sin_v;
      }
    }
  }

  void matvec(const TensorView& weight, const std::vector<float>& input,
              std::vector<float>& output) const {
    const auto rows = static_cast<std::size_t>(weight.shape[0]);
    const auto cols = static_cast<std::size_t>(weight.shape[1]);
    for (std::size_t row = 0; row < rows; ++row) {
      float sum = 0.0F;
      const auto row_offset = static_cast<std::uint64_t>(row * cols);
      for (std::size_t col = 0; col < cols; ++col) {
        sum += bf16_to_float(weight.data, row_offset + col) * input[col];
      }
      output[row] = sum;
    }
  }

  void store_kv(std::size_t layer, std::size_t position, const std::vector<float>& k,
                const std::vector<float>& v) {
    const auto kv_dim = static_cast<std::size_t>(config_.num_key_value_heads * config_.head_dim);
    const auto offset = (layer * max_seq_len_ + position) * kv_dim;
    std::copy(k.begin(), k.end(), key_cache_.begin() + static_cast<std::ptrdiff_t>(offset));
    std::copy(v.begin(), v.end(), value_cache_.begin() + static_cast<std::ptrdiff_t>(offset));
  }

  void attention(std::size_t layer, std::size_t position, const std::vector<float>& q,
                 std::vector<float>& output) const {
    const auto head_dim = static_cast<std::size_t>(config_.head_dim);
    const auto heads = static_cast<std::size_t>(config_.num_attention_heads);
    const auto kv_heads = static_cast<std::size_t>(config_.num_key_value_heads);
    const auto group = heads / kv_heads;
    const auto kv_dim = kv_heads * head_dim;
    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(position + 1U);

    for (std::size_t head = 0; head < heads; ++head) {
      const auto kv_head = head / group;
      const auto q_base = head * head_dim;
      float max_score = -std::numeric_limits<float>::infinity();
      for (std::size_t t = 0; t <= position; ++t) {
        const auto cache_base = (layer * max_seq_len_ + t) * kv_dim + kv_head * head_dim;
        float score = 0.0F;
        for (std::size_t d = 0; d < head_dim; ++d) {
          score += q[q_base + d] * key_cache_[cache_base + d];
        }
        score *= scale;
        scores[t] = score;
        max_score = std::max(max_score, score);
      }

      float denom = 0.0F;
      for (std::size_t t = 0; t <= position; ++t) {
        scores[t] = std::exp(scores[t] - max_score);
        denom += scores[t];
      }
      for (std::size_t d = 0; d < head_dim; ++d) {
        float value = 0.0F;
        for (std::size_t t = 0; t <= position; ++t) {
          const auto cache_base = (layer * max_seq_len_ + t) * kv_dim + kv_head * head_dim;
          value += (scores[t] / denom) * value_cache_[cache_base + d];
        }
        output[q_base + d] = value;
      }
    }
  }

  std::int64_t select_next_token(const std::vector<float>& hidden) const {
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto hidden_size = static_cast<std::size_t>(config_.hidden_size);
    float best = -std::numeric_limits<float>::infinity();
    std::int64_t best_id = kEndOfText;
    for (std::size_t row = 0; row < vocab; ++row) {
      float logit = 0.0F;
      const auto row_offset = static_cast<std::uint64_t>(row * hidden_size);
      for (std::size_t col = 0; col < hidden_size; ++col) {
        logit += bf16_to_float(lm_head_.data, row_offset + col) * hidden[col];
      }
      if (logit > best) {
        best = logit;
        best_id = static_cast<std::int64_t>(row);
      }
    }
    return best_id;
  }

  bool is_eos(std::int64_t token) const {
    if (token == config_.eos_token_id || token == kEndOfText) {
      return true;
    }
    return std::find(generation_.eos_token_ids.begin(), generation_.eos_token_ids.end(), token) !=
           generation_.eos_token_ids.end();
  }

  static float silu(float value) { return value / (1.0F + std::exp(-value)); }

  ModelConfig config_;
  GenerationConfig generation_;
  QwenTokenizer tokenizer_;
  SafeTensorMap weights_;
  TensorView embedding_;
  TensorView lm_head_;
  TensorView final_norm_;
  std::vector<LayerWeights> layers_;
  std::vector<float> key_cache_;
  std::vector<float> value_cache_;
  std::size_t max_seq_len_{0};
};

CpuQwenModel& cached_model(const std::filesystem::path& model_dir) {
  static std::unique_ptr<CpuQwenModel> model;
  static std::filesystem::path loaded_dir;
  if (!model || loaded_dir != model_dir) {
    model = std::make_unique<CpuQwenModel>(CpuQwenModel::load(model_dir));
    loaded_dir = model_dir;
  }
  return *model;
}

std::string shape_to_string(const std::vector<std::uint64_t>& shape) {
  std::ostringstream output;
  output << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << shape[i];
  }
  output << ']';
  return output.str();
}

void expect_shape(const SafeTensorMap& weights, std::string_view name,
                  const std::vector<std::uint64_t>& expected) {
  const auto& tensor = weights.at(name);
  if (tensor.shape != expected) {
    std::ostringstream output;
    output << "shape mismatch for " << name << ": expected " << shape_to_string(expected)
           << ", got " << shape_to_string(tensor.shape);
    throw std::runtime_error(output.str());
  }
}

void validate_qwen3_weights(const ModelConfig& config, const SafeTensorMap& weights) {
  const auto hidden = static_cast<std::uint64_t>(config.hidden_size);
  const auto head_dim = static_cast<std::uint64_t>(config.head_dim);
  const auto attn_dim =
    static_cast<std::uint64_t>(config.num_attention_heads * config.head_dim);
  const auto kv_dim =
    static_cast<std::uint64_t>(config.num_key_value_heads * config.head_dim);
  const auto intermediate = static_cast<std::uint64_t>(config.intermediate_size);
  const auto vocab = static_cast<std::uint64_t>(config.vocab_size);

  expect_shape(weights, "model.embed_tokens.weight", {vocab, hidden});
  expect_shape(weights, "model.norm.weight", {hidden});
  if (weights.tensors().find("lm_head.weight") != weights.tensors().end()) {
    expect_shape(weights, "lm_head.weight", {vocab, hidden});
  } else if (!config.tie_word_embeddings) {
    throw std::runtime_error("missing tensor: lm_head.weight");
  }

  for (std::uint64_t layer = 0; layer < static_cast<std::uint64_t>(config.num_hidden_layers);
       ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    expect_shape(weights, prefix + "input_layernorm.weight", {hidden});
    expect_shape(weights, prefix + "self_attn.q_proj.weight", {attn_dim, hidden});
    expect_shape(weights, prefix + "self_attn.q_norm.weight", {head_dim});
    expect_shape(weights, prefix + "self_attn.k_proj.weight", {kv_dim, hidden});
    expect_shape(weights, prefix + "self_attn.k_norm.weight", {head_dim});
    expect_shape(weights, prefix + "self_attn.v_proj.weight", {kv_dim, hidden});
    expect_shape(weights, prefix + "self_attn.o_proj.weight", {hidden, attn_dim});
    expect_shape(weights, prefix + "post_attention_layernorm.weight", {hidden});
    expect_shape(weights, prefix + "mlp.gate_proj.weight", {intermediate, hidden});
    expect_shape(weights, prefix + "mlp.up_proj.weight", {intermediate, hidden});
    expect_shape(weights, prefix + "mlp.down_proj.weight", {hidden, intermediate});
  }
}

std::string build_weight_summary(const std::filesystem::path& model_dir) {
  auto bundle = load_model_bundle(model_dir);
  if (!bundle.is_ok()) {
    throw std::runtime_error(bundle.status().message());
  }

  const auto weights = SafeTensorMap::load(model_dir / "model.safetensors");
  validate_qwen3_weights(bundle.value().model, weights);

  std::unordered_map<std::string, std::size_t> dtype_counts;
  std::vector<std::string> names;
  names.reserve(weights.tensors().size());
  for (const auto& entry : weights.tensors()) {
    dtype_counts[entry.second.dtype] += 1U;
    names.push_back(entry.first);
  }
  std::sort(names.begin(), names.end());

  std::ostringstream output;
  output << "Weights: ok\n";
  output << "File: " << (model_dir / "model.safetensors").string() << '\n';
  output << "File size: " << weights.file_size() << " bytes\n";
  output << "Header size: " << weights.header_size() << " bytes\n";
  output << "Tensor count: " << weights.tensors().size() << '\n';
  output << "DTypes:\n";
  std::vector<std::string> dtype_names;
  dtype_names.reserve(dtype_counts.size());
  for (const auto& entry : dtype_counts) {
    dtype_names.push_back(entry.first);
  }
  std::sort(dtype_names.begin(), dtype_names.end());
  for (const auto& dtype : dtype_names) {
    output << "- " << dtype << ": " << dtype_counts[dtype] << '\n';
  }
  output << "First tensors:\n";
  const auto preview_count = std::min<std::size_t>(names.size(), 12U);
  for (std::size_t i = 0; i < preview_count; ++i) {
    const auto& tensor = weights.at(names[i]);
    output << "- " << tensor.name << " " << shape_to_string(tensor.shape) << ' ' << tensor.dtype
           << '\n';
  }
  output << "Qwen3 mapping: ok\n";
  output << "Validation: ok\n";
  return output.str();
}

}  // namespace

Result<CpuGenerationResult> generate_cpu(const CpuGenerationRequest& request) {
  if (request.prompt.empty() && request.messages.empty()) {
    return Status::invalid_argument("prompt must not be empty");
  }
  if (request.max_new_tokens == 0) {
    return Status::invalid_argument("max_new_tokens must be greater than zero");
  }

  try {
    auto messages = request.messages;
    if (messages.empty()) {
      messages.push_back(ChatMessage{"user", request.prompt});
    }
    if (messages.empty()) {
      return Status::invalid_argument("messages must not be empty");
    }

    CpuGenerationResult result{};
    result.implemented = true;
    result.text = cached_model(request.model_dir)
                    .generate(messages, request.max_new_tokens, request.enable_thinking);
    return result;
  } catch (const std::exception& error) {
    return Status::internal_error(error.what());
  }
}

std::string format_cpu_generation_result(const CpuGenerationResult& result) {
  if (result.implemented) {
    return result.text + '\n';
  }

  std::ostringstream output;
  output << result.text;
  if (!result.missing_dependencies.empty()) {
    output << "Missing dependencies:\n";
    for (const auto& dependency : result.missing_dependencies) {
      output << "- " << dependency << '\n';
    }
  }
  return output.str();
}

Result<std::string> format_weight_summary(const std::filesystem::path& model_dir) {
  try {
    return build_weight_summary(model_dir);
  } catch (const std::exception& error) {
    return Status::internal_error(error.what());
  }
}

}  // namespace toyllm
