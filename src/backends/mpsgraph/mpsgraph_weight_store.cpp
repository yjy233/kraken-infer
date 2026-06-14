#include "toyllm/backends/mpsgraph/mpsgraph_weight_store.hpp"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace toyllm::mpsgraph {

namespace {

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
    if (position_ >= input_.size() || input_[position_] < '0' || input_[position_] > '9') {
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
      return;
    }
    if (codepoint <= 0x7FFU) {
      result.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
      result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
      return;
    }
    result.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
    result.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  }

  std::string_view input_;
  std::size_t position_{0};
};

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

Status checked_expected_shape(const MpsGraphTensorInfo& tensor,
                              const std::vector<std::uint64_t>& expected) {
  if (tensor.shape == expected) {
    return Status::ok();
  }
  std::ostringstream output;
  output << "MPSGraph tensor shape mismatch for " << tensor.name << ": expected [";
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << expected[i];
  }
  output << "], got [";
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << tensor.shape[i];
  }
  output << "]";
  return Status::invalid_argument(output.str());
}

float bf16_to_float(const std::byte* data) {
  std::uint16_t value = 0;
  std::memcpy(&value, data, sizeof(value));
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float output = 0.0F;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

}  // namespace

Result<MpsGraphWeightStore> MpsGraphWeightStore::load_metadata(
  const std::filesystem::path& path) {
  try {
    return load_metadata_impl(path);
  } catch (const std::exception& error) {
    return Status::invalid_argument(error.what());
  }
}

Result<MpsGraphWeightStore> MpsGraphWeightStore::load_metadata_impl(
  const std::filesystem::path& path) {
  MpsGraphWeightStore store;
  store.path_ = path;
  store.fd_ = ::open(path.c_str(), O_RDONLY);
  if (store.fd_ < 0) {
    return Status::unavailable("failed to open " + path.string());
  }

  struct stat file_stat {};
  if (::fstat(store.fd_, &file_stat) != 0 || file_stat.st_size <= 8) {
    return Status::invalid_argument("invalid safetensors file size");
  }
  store.file_size_ = static_cast<std::uint64_t>(file_stat.st_size);
  void* mapped = ::mmap(nullptr, static_cast<std::size_t>(store.file_size_), PROT_READ,
                        MAP_PRIVATE, store.fd_, 0);
  if (mapped == MAP_FAILED) {
    return Status::unavailable("failed to mmap " + path.string());
  }
  store.mapped_data_ = static_cast<const std::byte*>(mapped);

  std::uint64_t header_size = 0;
  for (int i = 0; i < 8; ++i) {
    header_size |= static_cast<std::uint64_t>(
                     static_cast<unsigned char>(store.mapped_data_[static_cast<std::size_t>(i)]))
                   << (8U * static_cast<unsigned int>(i));
  }
  if (header_size == 0 || 8U + header_size >= store.file_size_) {
    return Status::invalid_argument("invalid safetensors header size");
  }
  store.header_size_ = header_size;
  store.data_start_ = 8U + header_size;
  const auto* header_begin = reinterpret_cast<const char*>(store.mapped_data_ + 8);
  store.parse_header(std::string_view(header_begin, static_cast<std::size_t>(header_size)));
  const auto range_status = store.validate_tensor_ranges();
  if (!range_status.is_ok()) {
    return range_status;
  }
  return store;
}

MpsGraphWeightStore::~MpsGraphWeightStore() { close(); }

MpsGraphWeightStore::MpsGraphWeightStore(MpsGraphWeightStore&& other) noexcept {
  move_from(std::move(other));
}

MpsGraphWeightStore& MpsGraphWeightStore::operator=(MpsGraphWeightStore&& other) noexcept {
  if (this != &other) {
    close();
    move_from(std::move(other));
  }
  return *this;
}

const MpsGraphTensorInfo& MpsGraphWeightStore::at(std::string_view name) const {
  const auto it = tensors_.find(std::string{name});
  if (it == tensors_.end()) {
    throw std::runtime_error("missing MPSGraph tensor: " + std::string{name});
  }
  return it->second;
}

bool MpsGraphWeightStore::contains(std::string_view name) const {
  return tensors_.find(std::string{name}) != tensors_.end();
}

std::uint64_t MpsGraphWeightStore::file_size() const { return file_size_; }

std::uint64_t MpsGraphWeightStore::header_size() const { return header_size_; }

const std::unordered_map<std::string, MpsGraphTensorInfo>& MpsGraphWeightStore::tensors()
  const {
  return tensors_;
}

Status MpsGraphWeightStore::validate_qwen3_shapes(const ModelConfig& config) const {
  if (config.hidden_size <= 0 || config.vocab_size <= 0 || config.num_hidden_layers <= 0 ||
      config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 ||
      config.head_dim <= 0 || config.intermediate_size <= 0) {
    return Status::invalid_argument("MPSGraph Qwen3 config dimensions must be positive");
  }

  const auto hidden = static_cast<std::uint64_t>(config.hidden_size);
  const auto vocab = static_cast<std::uint64_t>(config.vocab_size);
  const auto layers = static_cast<std::uint64_t>(config.num_hidden_layers);
  const auto heads = static_cast<std::uint64_t>(config.num_attention_heads);
  const auto kv_heads = static_cast<std::uint64_t>(config.num_key_value_heads);
  const auto head_dim = static_cast<std::uint64_t>(config.head_dim);
  const auto intermediate = static_cast<std::uint64_t>(config.intermediate_size);
  const auto attn_dim = heads * head_dim;
  const auto kv_dim = kv_heads * head_dim;

  auto expect = [&](std::string_view name, std::vector<std::uint64_t> shape) -> Status {
    if (!contains(name)) {
      return Status::invalid_argument("missing MPSGraph tensor: " + std::string{name});
    }
    const auto& tensor = at(name);
    if (tensor.dtype != "BF16") {
      return Status::invalid_argument("MPSGraph tensor must be BF16: " + std::string{name});
    }
    return checked_expected_shape(tensor, shape);
  };

  auto status = expect("model.embed_tokens.weight", {vocab, hidden});
  if (!status.is_ok()) {
    return status;
  }
  status = expect("lm_head.weight", {vocab, hidden});
  if (!status.is_ok()) {
    return status;
  }
  status = expect("model.norm.weight", {hidden});
  if (!status.is_ok()) {
    return status;
  }

  for (std::uint64_t layer = 0; layer < layers; ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    status = expect(prefix + "input_layernorm.weight", {hidden});
    if (!status.is_ok()) {
      return status;
    }
    status = expect(prefix + "post_attention_layernorm.weight", {hidden});
    if (!status.is_ok()) {
      return status;
    }
    status = expect(prefix + "self_attn.q_proj.weight", {attn_dim, hidden});
    if (!status.is_ok()) {
      return status;
    }
    status = expect(prefix + "self_attn.k_proj.weight", {kv_dim, hidden});
    if (!status.is_ok()) {
      return status;
    }
    status = expect(prefix + "self_attn.v_proj.weight", {kv_dim, hidden});
    if (!status.is_ok()) {
      return status;
    }
    status = expect(prefix + "self_attn.o_proj.weight", {hidden, attn_dim});
    if (!status.is_ok()) {
      return status;
    }
    status = expect(prefix + "self_attn.q_norm.weight", {head_dim});
    if (!status.is_ok()) {
      return status;
    }
    status = expect(prefix + "self_attn.k_norm.weight", {head_dim});
    if (!status.is_ok()) {
      return status;
    }
    status = expect(prefix + "mlp.gate_proj.weight", {intermediate, hidden});
    if (!status.is_ok()) {
      return status;
    }
    status = expect(prefix + "mlp.up_proj.weight", {intermediate, hidden});
    if (!status.is_ok()) {
      return status;
    }
    status = expect(prefix + "mlp.down_proj.weight", {hidden, intermediate});
    if (!status.is_ok()) {
      return status;
    }
  }

  return Status::ok();
}

Result<MpsGraphDeviceTensor> MpsGraphWeightStore::upload_tensor_f32(
  const MpsGraphContext& context, std::string_view name) const {
  if (mapped_data_ == nullptr) {
    return Status::invalid_argument("MPSGraph weight store is not loaded");
  }
  const auto& tensor = at(name);
  if (tensor.dtype != "BF16") {
    return Status::invalid_argument("MPSGraph f32 upload currently requires BF16 source");
  }
  if (tensor.byte_size % sizeof(std::uint16_t) != 0) {
    return Status::invalid_argument("MPSGraph BF16 tensor byte size is not aligned");
  }
  const auto elements = tensor.byte_size / sizeof(std::uint16_t);
  if (elements != product(tensor.shape)) {
    return Status::invalid_argument("MPSGraph tensor element count mismatch");
  }
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return Status::invalid_argument("MPSGraph f32 upload byte count overflow");
  }

  auto buffer_result = context.make_buffer(static_cast<std::size_t>(elements) * sizeof(float));
  if (!buffer_result.is_ok()) {
    return buffer_result.status();
  }
  auto buffer = std::move(buffer_result.value());

  constexpr std::size_t chunk_elements = 1U << 20U;
  std::vector<float> converted;
  converted.resize(std::min<std::uint64_t>(chunk_elements, elements));
  const auto* source = mapped_data_ + data_start_ + tensor.data_offset_begin;
  std::uint64_t offset = 0;
  while (offset < elements) {
    const auto current =
      static_cast<std::size_t>(std::min<std::uint64_t>(converted.size(), elements - offset));
    for (std::size_t i = 0; i < current; ++i) {
      converted[i] = bf16_to_float(source + (offset + i) * sizeof(std::uint16_t));
    }
    const auto copy_status = context.copy_to_buffer_at(
      buffer, static_cast<std::size_t>(offset) * sizeof(float), converted.data(),
      current * sizeof(float));
    if (!copy_status.is_ok()) {
      return copy_status;
    }
    offset += current;
  }

  MpsGraphDeviceTensor output;
  output.buffer = std::move(buffer);
  output.shape = tensor.shape;
  output.source_dtype = tensor.dtype;
  output.elements = elements;
  return output;
}

void MpsGraphWeightStore::parse_header(std::string_view header) {
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
      MpsGraphTensorInfo tensor;
      tensor.name = name;
      scanner.expect('{');
      while (true) {
        const std::string key = scanner.parse_string();
        scanner.expect(':');
        if (key == "dtype") {
          tensor.dtype = scanner.parse_string();
        } else if (key == "shape") {
          tensor.shape = scanner.parse_uint_array();
        } else if (key == "data_offsets") {
          const auto offsets = scanner.parse_uint_array();
          if (offsets.size() != 2 || offsets[1] < offsets[0]) {
            throw std::runtime_error("invalid data_offsets for " + tensor.name);
          }
          if (data_start_ + offsets[1] > file_size_) {
            throw std::runtime_error("tensor data offset exceeds file size for " +
                                     tensor.name);
          }
          tensor.data_offset_begin = offsets[0];
          tensor.data_offset_end = offsets[1];
          tensor.byte_size = offsets[1] - offsets[0];
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
      if (tensor.dtype != "BF16") {
        throw std::runtime_error("MPSGraph weight store only supports BF16 safetensors");
      }
      if (product(tensor.shape) * sizeof(std::uint16_t) != tensor.byte_size) {
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

Status MpsGraphWeightStore::validate_tensor_ranges() const {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  ranges.reserve(tensors_.size());
  for (const auto& entry : tensors_) {
    ranges.push_back({entry.second.data_offset_begin, entry.second.data_offset_end});
  }
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].first < ranges[i - 1].second) {
      return Status::invalid_argument("MPSGraph safetensors tensor data offsets overlap");
    }
  }
  return Status::ok();
}

void MpsGraphWeightStore::close() {
  if (mapped_data_ != nullptr) {
    (void)::munmap(const_cast<std::byte*>(mapped_data_), static_cast<std::size_t>(file_size_));
    mapped_data_ = nullptr;
  }
  if (fd_ >= 0) {
    (void)::close(fd_);
    fd_ = -1;
  }
}

void MpsGraphWeightStore::move_from(MpsGraphWeightStore&& other) noexcept {
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

}  // namespace toyllm::mpsgraph
