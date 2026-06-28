#include "toyllm/runtime/gguf_reader.hpp"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace toyllm {

namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747U;
constexpr std::uint64_t kDefaultAlignment = 32;

class MappedFile {
 public:
  explicit MappedFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("failed to open " + path_.string());
    }

    struct stat file_stat {};
    if (::fstat(fd_, &file_stat) != 0 || file_stat.st_size <= 0) {
      throw std::runtime_error("invalid file size for " + path_.string());
    }
    size_ = static_cast<std::uint64_t>(file_stat.st_size);
    void* mapped = ::mmap(nullptr, static_cast<std::size_t>(size_), PROT_READ, MAP_PRIVATE,
                          fd_, 0);
    if (mapped == MAP_FAILED) {
      throw std::runtime_error("failed to mmap " + path_.string());
    }
    data_ = static_cast<const std::byte*>(mapped);
  }

  ~MappedFile() {
    if (data_ != nullptr) {
      (void)::munmap(const_cast<std::byte*>(data_), static_cast<std::size_t>(size_));
    }
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  [[nodiscard]] const std::byte* data() const { return data_; }
  [[nodiscard]] std::uint64_t size() const { return size_; }

 private:
  std::filesystem::path path_;
  int fd_{-1};
  const std::byte* data_{nullptr};
  std::uint64_t size_{0};
};

class Reader {
 public:
  Reader(const std::byte* data, std::uint64_t size) : data_(data), size_(size) {}

  [[nodiscard]] std::uint64_t position() const { return position_; }

  std::uint8_t u8() { return read_pod<std::uint8_t>(); }
  std::int8_t i8() { return read_pod<std::int8_t>(); }
  std::uint16_t u16() { return read_pod<std::uint16_t>(); }
  std::int16_t i16() { return read_pod<std::int16_t>(); }
  std::uint32_t u32() { return read_pod<std::uint32_t>(); }
  std::int32_t i32() { return read_pod<std::int32_t>(); }
  std::uint64_t u64() { return read_pod<std::uint64_t>(); }
  std::int64_t i64() { return read_pod<std::int64_t>(); }
  float f32() { return read_pod<float>(); }
  double f64() { return read_pod<double>(); }

  bool boolean() {
    const auto value = u8();
    if (value > 1) {
      fail("invalid GGUF bool value");
    }
    return value != 0;
  }

  std::string string() {
    const auto size = u64();
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      fail("GGUF string is too large");
    }
    ensure(size);
    const auto* begin = reinterpret_cast<const char*>(data_ + position_);
    position_ += size;
    return std::string(begin, static_cast<std::size_t>(size));
  }

  void skip(std::uint64_t count) {
    ensure(count);
    position_ += count;
  }

 private:
  template <typename T>
  T read_pod() {
    ensure(sizeof(T));
    T value{};
    std::memcpy(&value, data_ + position_, sizeof(T));
    position_ += sizeof(T);
    return value;
  }

  void ensure(std::uint64_t count) const {
    if (count > size_ || position_ > size_ - count) {
      fail("unexpected end of GGUF file");
    }
  }

  [[noreturn]] void fail(std::string_view message) const {
    std::ostringstream output;
    output << message << " at byte " << position_;
    throw std::runtime_error(output.str());
  }

  const std::byte* data_{nullptr};
  std::uint64_t size_{0};
  std::uint64_t position_{0};
};

GgufValueKind to_value_kind(std::uint32_t type) {
  switch (type) {
    case 0:
      return GgufValueKind::uint8;
    case 1:
      return GgufValueKind::int8;
    case 2:
      return GgufValueKind::uint16;
    case 3:
      return GgufValueKind::int16;
    case 4:
      return GgufValueKind::uint32;
    case 5:
      return GgufValueKind::int32;
    case 6:
      return GgufValueKind::float32;
    case 7:
      return GgufValueKind::bool_value;
    case 8:
      return GgufValueKind::string;
    case 9:
      return GgufValueKind::array;
    case 10:
      return GgufValueKind::uint64;
    case 11:
      return GgufValueKind::int64;
    case 12:
      return GgufValueKind::float64;
    default:
      throw std::runtime_error("unsupported GGUF metadata value type: " + std::to_string(type));
  }
}

bool is_array_element_kind(GgufValueKind kind) {
  return kind != GgufValueKind::array;
}

GgufMetadataValue read_scalar_value(Reader& reader, GgufValueKind kind) {
  GgufMetadataValue value;
  value.kind = kind;
  switch (kind) {
    case GgufValueKind::uint8:
      value.value = static_cast<std::uint64_t>(reader.u8());
      break;
    case GgufValueKind::int8:
      value.value = static_cast<std::int64_t>(reader.i8());
      break;
    case GgufValueKind::uint16:
      value.value = static_cast<std::uint64_t>(reader.u16());
      break;
    case GgufValueKind::int16:
      value.value = static_cast<std::int64_t>(reader.i16());
      break;
    case GgufValueKind::uint32:
      value.value = static_cast<std::uint64_t>(reader.u32());
      break;
    case GgufValueKind::int32:
      value.value = static_cast<std::int64_t>(reader.i32());
      break;
    case GgufValueKind::float32:
      value.value = static_cast<double>(reader.f32());
      break;
    case GgufValueKind::bool_value:
      value.value = reader.boolean();
      break;
    case GgufValueKind::string:
      value.value = reader.string();
      break;
    case GgufValueKind::uint64:
      value.value = reader.u64();
      break;
    case GgufValueKind::int64:
      value.value = reader.i64();
      break;
    case GgufValueKind::float64:
      value.value = reader.f64();
      break;
    case GgufValueKind::array:
      throw std::runtime_error("nested GGUF arrays are not supported");
  }
  return value;
}

GgufMetadataValue read_value(Reader& reader) {
  const auto kind = to_value_kind(reader.u32());
  if (kind != GgufValueKind::array) {
    return read_scalar_value(reader, kind);
  }

  const auto array_kind = to_value_kind(reader.u32());
  if (!is_array_element_kind(array_kind)) {
    throw std::runtime_error("nested GGUF arrays are not supported");
  }
  const auto size = reader.u64();
  GgufMetadataValue value;
  value.kind = GgufValueKind::array;
  value.array_kind = array_kind;

  switch (array_kind) {
    case GgufValueKind::uint8:
    case GgufValueKind::uint16:
    case GgufValueKind::uint32:
    case GgufValueKind::uint64: {
      std::vector<std::uint64_t> values;
      values.reserve(static_cast<std::size_t>(size));
      for (std::uint64_t i = 0; i < size; ++i) {
        values.push_back(std::get<std::uint64_t>(read_scalar_value(reader, array_kind).value));
      }
      value.value = std::move(values);
      break;
    }
    case GgufValueKind::int8:
    case GgufValueKind::int16:
    case GgufValueKind::int32:
    case GgufValueKind::int64: {
      std::vector<std::int64_t> values;
      values.reserve(static_cast<std::size_t>(size));
      for (std::uint64_t i = 0; i < size; ++i) {
        values.push_back(std::get<std::int64_t>(read_scalar_value(reader, array_kind).value));
      }
      value.value = std::move(values);
      break;
    }
    case GgufValueKind::float32:
    case GgufValueKind::float64: {
      std::vector<double> values;
      values.reserve(static_cast<std::size_t>(size));
      for (std::uint64_t i = 0; i < size; ++i) {
        values.push_back(std::get<double>(read_scalar_value(reader, array_kind).value));
      }
      value.value = std::move(values);
      break;
    }
    case GgufValueKind::bool_value: {
      std::vector<bool> values;
      values.reserve(static_cast<std::size_t>(size));
      for (std::uint64_t i = 0; i < size; ++i) {
        values.push_back(std::get<bool>(read_scalar_value(reader, array_kind).value));
      }
      value.value = std::move(values);
      break;
    }
    case GgufValueKind::string: {
      std::vector<std::string> values;
      values.reserve(static_cast<std::size_t>(size));
      for (std::uint64_t i = 0; i < size; ++i) {
        values.push_back(std::get<std::string>(read_scalar_value(reader, array_kind).value));
      }
      value.value = std::move(values);
      break;
    }
    case GgufValueKind::array:
      throw std::runtime_error("nested GGUF arrays are not supported");
  }

  return value;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  if (alignment == 0) {
    throw std::runtime_error("GGUF alignment must be non-zero");
  }
  const auto remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const auto delta = alignment - remainder;
  if (value > std::numeric_limits<std::uint64_t>::max() - delta) {
    throw std::runtime_error("GGUF alignment overflow");
  }
  return value + delta;
}

std::uint64_t product(const std::vector<std::uint64_t>& shape) {
  std::uint64_t result = 1;
  for (const auto dim : shape) {
    if (dim != 0 && result > std::numeric_limits<std::uint64_t>::max() / dim) {
      throw std::runtime_error("GGUF tensor shape product overflow");
    }
    result *= dim;
  }
  return result;
}

struct TypeLayout {
  std::uint64_t block_size;
  std::uint64_t type_size;
};

TypeLayout ggml_type_layout(std::uint32_t type) {
  switch (type) {
    case 0:
      return {1, 4};    // F32
    case 1:
      return {1, 2};    // F16
    case 2:
      return {32, 18};  // Q4_0
    case 3:
      return {32, 20};  // Q4_1
    case 6:
      return {32, 22};  // Q5_0
    case 7:
      return {32, 24};  // Q5_1
    case 8:
      return {32, 34};  // Q8_0
    case 9:
      return {32, 36};  // Q8_1
    case 10:
      return {256, 84};   // Q2_K
    case 11:
      return {256, 110};  // Q3_K
    case 12:
      return {256, 144};  // Q4_K
    case 13:
      return {256, 176};  // Q5_K
    case 14:
      return {256, 210};  // Q6_K
    case 15:
      return {256, 292};  // Q8_K
    case 16:
      return {256, 66};   // IQ2_XXS
    case 17:
      return {256, 74};   // IQ2_XS
    case 18:
      return {256, 98};   // IQ3_XXS
    case 19:
      return {256, 50};   // IQ1_S
    case 20:
      return {32, 18};    // IQ4_NL
    case 21:
      return {256, 110};  // IQ3_S
    case 22:
      return {256, 82};   // IQ2_S
    case 23:
      return {256, 136};  // IQ4_XS
    case 24:
      return {1, 1};      // I8
    case 25:
      return {1, 2};      // I16
    case 26:
      return {1, 4};      // I32
    case 27:
      return {1, 8};      // I64
    case 28:
      return {1, 8};      // F64
    case 29:
      return {256, 56};   // IQ1_M
    case 30:
      return {1, 2};      // BF16
    case 34:
      return {256, 54};   // TQ1_0
    case 35:
      return {256, 66};   // TQ2_0
    case 39:
      return {32, 17};    // MXFP4
    case 40:
      return {64, 36};    // NVFP4
    case 41:
      return {128, 18};   // Q1_0
    default:
      throw std::runtime_error("unsupported GGML tensor type: " + std::to_string(type));
  }
}

bool has_exact_layout(std::uint32_t type) {
  return type <= 30 || type == 34 || type == 35 || type == 39 || type == 40 || type == 41;
}

std::uint64_t tensor_nbytes(const GgufTensorInfo& tensor) {
  const auto elements = product(tensor.shape);
  const auto layout = ggml_type_layout(tensor.type);
  if (!has_exact_layout(tensor.type)) {
    return 0;
  }
  if (elements % layout.block_size != 0) {
    throw std::runtime_error("GGUF tensor element count is not divisible by block size: " +
                             tensor.name);
  }
  const auto blocks = elements / layout.block_size;
  if (blocks > std::numeric_limits<std::uint64_t>::max() / layout.type_size) {
    throw std::runtime_error("GGUF tensor byte size overflow: " + tensor.name);
  }
  return blocks * layout.type_size;
}

template <typename T>
Result<T> missing_key(const std::string& key) {
  return Status::invalid_argument("missing GGUF metadata key: " + key);
}

template <typename T>
std::string join_values(const std::vector<T>& values) {
  std::ostringstream output;
  output << '[';
  const auto limit = std::min<std::size_t>(values.size(), 8U);
  for (std::size_t i = 0; i < limit; ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << values[i];
  }
  if (values.size() > limit) {
    output << ", ... (" << values.size() << ')';
  }
  output << ']';
  return output.str();
}

}  // namespace

struct GgufMappedData::Impl {
  std::filesystem::path path;
  int fd{-1};
  const std::byte* data{nullptr};
  std::uint64_t size{0};

  ~Impl() {
    if (data != nullptr) {
      (void)::munmap(const_cast<std::byte*>(data), static_cast<std::size_t>(size));
    }
    if (fd >= 0) {
      (void)::close(fd);
    }
  }
};

GgufMappedData::GgufMappedData() = default;
GgufMappedData::~GgufMappedData() = default;
GgufMappedData::GgufMappedData(GgufMappedData&& other) noexcept = default;
GgufMappedData& GgufMappedData::operator=(GgufMappedData&& other) noexcept = default;
GgufMappedData::GgufMappedData(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Result<GgufMappedData> GgufMappedData::open(const GgufFile& file) {
  auto impl = std::make_unique<Impl>();
  impl->path = file.path;
  impl->fd = ::open(impl->path.c_str(), O_RDONLY);
  if (impl->fd < 0) {
    return Status::invalid_argument("failed to open GGUF data file: " +
                                    impl->path.string());
  }

  struct stat file_stat {};
  if (::fstat(impl->fd, &file_stat) != 0 || file_stat.st_size <= 0) {
    return Status::invalid_argument("invalid GGUF data file size: " +
                                    impl->path.string());
  }
  impl->size = static_cast<std::uint64_t>(file_stat.st_size);
  if (impl->size != file.file_size) {
    return Status::invalid_argument("GGUF data file size changed after metadata read: " +
                                    impl->path.string());
  }
  void* mapped =
    ::mmap(nullptr, static_cast<std::size_t>(impl->size), PROT_READ, MAP_PRIVATE,
           impl->fd, 0);
  if (mapped == MAP_FAILED) {
    return Status::invalid_argument("failed to mmap GGUF data file: " +
                                    impl->path.string());
  }
  impl->data = static_cast<const std::byte*>(mapped);
  return GgufMappedData(std::move(impl));
}

bool GgufMappedData::valid() const {
  return impl_ != nullptr && impl_->data != nullptr && impl_->size > 0;
}

std::uint64_t GgufMappedData::size() const {
  return impl_ == nullptr ? 0 : impl_->size;
}

Result<GgufTensorBytes> GgufMappedData::tensor_bytes(const GgufTensorInfo& tensor) const {
  if (!valid()) {
    return Status::invalid_argument("GGUF mapped data is not initialized");
  }
  if (tensor.byte_size > std::numeric_limits<std::size_t>::max()) {
    return Status::invalid_argument("GGUF tensor byte size exceeds size_t: " +
                                    tensor.name);
  }
  if (tensor.absolute_offset > impl_->size ||
      tensor.byte_size > impl_->size - tensor.absolute_offset) {
    return Status::invalid_argument("GGUF tensor byte range exceeds mapped file: " +
                                    tensor.name);
  }
  return GgufTensorBytes{impl_->data + tensor.absolute_offset,
                         static_cast<std::size_t>(tensor.byte_size)};
}

Result<std::filesystem::path> resolve_gguf_model_path(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    return Status::unavailable("model path does not exist: " + path.string());
  }
  if (std::filesystem::is_regular_file(path)) {
    if (path.extension() == ".gguf") {
      return path;
    }
    return Status::invalid_argument("model file is not a GGUF file: " + path.string());
  }
  if (!std::filesystem::is_directory(path)) {
    return Status::invalid_argument("model path is not a file or directory: " + path.string());
  }

  std::vector<std::filesystem::path> candidates;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
      candidates.push_back(entry.path());
    }
  }
  if (candidates.empty()) {
    return Status::unavailable("no GGUF file found in " + path.string());
  }
  std::sort(candidates.begin(), candidates.end());
  return candidates.front();
}

Result<GgufFile> read_gguf_file(const std::filesystem::path& path) {
  try {
    MappedFile mapped(path);
    Reader reader(mapped.data(), mapped.size());

    const auto magic = reader.u32();
    if (magic != kGgufMagic) {
      return Status::invalid_argument("not a GGUF file: " + path.string());
    }

    GgufFile file;
    file.path = path;
    file.file_size = mapped.size();
    file.version = reader.u32();
    if (file.version < 2 || file.version > 3) {
      return Status::invalid_argument("unsupported GGUF version: " +
                                      std::to_string(file.version));
    }
    file.tensor_count = reader.u64();
    file.metadata_count = reader.u64();

    for (std::uint64_t i = 0; i < file.metadata_count; ++i) {
      const auto key = reader.string();
      auto value = read_value(reader);
      file.metadata.emplace(key, std::move(value));
    }

    if (auto* alignment = find_gguf_metadata(file, "general.alignment");
        alignment != nullptr && std::holds_alternative<std::uint64_t>(alignment->value)) {
      file.alignment = std::get<std::uint64_t>(alignment->value);
    } else {
      file.alignment = kDefaultAlignment;
    }

    file.tensors.reserve(static_cast<std::size_t>(file.tensor_count));
    for (std::uint64_t i = 0; i < file.tensor_count; ++i) {
      GgufTensorInfo tensor;
      tensor.name = reader.string();
      const auto n_dims = reader.u32();
      if (n_dims == 0 || n_dims > 4) {
        return Status::invalid_argument("invalid GGUF tensor rank for " + tensor.name);
      }
      tensor.shape.reserve(n_dims);
      for (std::uint32_t dim = 0; dim < n_dims; ++dim) {
        tensor.shape.push_back(reader.u64());
      }
      tensor.type = reader.u32();
      tensor.offset = reader.u64();
      tensor.byte_size = tensor_nbytes(tensor);
      file.tensors.push_back(std::move(tensor));
    }

    file.data_offset = align_up(reader.position(), file.alignment);
    if (file.data_offset > file.file_size) {
      return Status::invalid_argument("GGUF data offset exceeds file size");
    }
    for (auto& tensor : file.tensors) {
      if (tensor.offset > file.file_size - file.data_offset) {
        return Status::invalid_argument("GGUF tensor offset exceeds file size: " + tensor.name);
      }
      tensor.absolute_offset = file.data_offset + tensor.offset;
      if (tensor.byte_size != 0 &&
          tensor.byte_size > file.file_size - tensor.absolute_offset) {
        return Status::invalid_argument("GGUF tensor byte range exceeds file size: " +
                                        tensor.name);
      }
    }

    return file;
  } catch (const std::exception& error) {
    return Status::invalid_argument(error.what());
  }
}

const GgufMetadataValue* find_gguf_metadata(const GgufFile& file, const std::string& key) {
  const auto it = file.metadata.find(key);
  if (it == file.metadata.end()) {
    return nullptr;
  }
  return &it->second;
}

const GgufTensorInfo* find_gguf_tensor(const GgufFile& file, const std::string& name) {
  const auto it = std::find_if(file.tensors.begin(), file.tensors.end(),
                               [&](const GgufTensorInfo& tensor) {
                                 return tensor.name == name;
                               });
  if (it == file.tensors.end()) {
    return nullptr;
  }
  return &*it;
}

std::string gguf_value_kind_name(GgufValueKind kind) {
  switch (kind) {
    case GgufValueKind::uint8:
      return "uint8";
    case GgufValueKind::int8:
      return "int8";
    case GgufValueKind::uint16:
      return "uint16";
    case GgufValueKind::int16:
      return "int16";
    case GgufValueKind::uint32:
      return "uint32";
    case GgufValueKind::int32:
      return "int32";
    case GgufValueKind::float32:
      return "float32";
    case GgufValueKind::bool_value:
      return "bool";
    case GgufValueKind::string:
      return "string";
    case GgufValueKind::array:
      return "array";
    case GgufValueKind::uint64:
      return "uint64";
    case GgufValueKind::int64:
      return "int64";
    case GgufValueKind::float64:
      return "float64";
  }
  return "unknown";
}

std::string ggml_type_name(std::uint32_t type) {
  switch (type) {
    case 0:
      return "F32";
    case 1:
      return "F16";
    case 2:
      return "Q4_0";
    case 3:
      return "Q4_1";
    case 6:
      return "Q5_0";
    case 7:
      return "Q5_1";
    case 8:
      return "Q8_0";
    case 9:
      return "Q8_1";
    case 10:
      return "Q2_K";
    case 11:
      return "Q3_K";
    case 12:
      return "Q4_K";
    case 13:
      return "Q5_K";
    case 14:
      return "Q6_K";
    case 15:
      return "Q8_K";
    case 16:
      return "IQ2_XXS";
    case 17:
      return "IQ2_XS";
    case 18:
      return "IQ3_XXS";
    case 19:
      return "IQ1_S";
    case 20:
      return "IQ4_NL";
    case 21:
      return "IQ3_S";
    case 22:
      return "IQ2_S";
    case 23:
      return "IQ4_XS";
    case 24:
      return "I8";
    case 25:
      return "I16";
    case 26:
      return "I32";
    case 27:
      return "I64";
    case 28:
      return "F64";
    case 29:
      return "IQ1_M";
    case 30:
      return "BF16";
    case 34:
      return "TQ1_0";
    case 35:
      return "TQ2_0";
    case 39:
      return "MXFP4";
    case 40:
      return "NVFP4";
    case 41:
      return "Q1_0";
    default:
      return "UNKNOWN(" + std::to_string(type) + ")";
  }
}

std::string gguf_value_to_string(const GgufMetadataValue& value) {
  if (std::holds_alternative<std::uint64_t>(value.value)) {
    return std::to_string(std::get<std::uint64_t>(value.value));
  }
  if (std::holds_alternative<std::int64_t>(value.value)) {
    return std::to_string(std::get<std::int64_t>(value.value));
  }
  if (std::holds_alternative<double>(value.value)) {
    std::ostringstream output;
    output << std::get<double>(value.value);
    return output.str();
  }
  if (std::holds_alternative<bool>(value.value)) {
    return std::get<bool>(value.value) ? "true" : "false";
  }
  if (std::holds_alternative<std::string>(value.value)) {
    return std::get<std::string>(value.value);
  }
  if (std::holds_alternative<std::vector<std::uint64_t>>(value.value)) {
    return join_values(std::get<std::vector<std::uint64_t>>(value.value));
  }
  if (std::holds_alternative<std::vector<std::int64_t>>(value.value)) {
    return join_values(std::get<std::vector<std::int64_t>>(value.value));
  }
  if (std::holds_alternative<std::vector<double>>(value.value)) {
    return join_values(std::get<std::vector<double>>(value.value));
  }
  if (std::holds_alternative<std::vector<bool>>(value.value)) {
    const auto& values = std::get<std::vector<bool>>(value.value);
    std::ostringstream output;
    output << '[';
    const auto limit = std::min<std::size_t>(values.size(), 8U);
    for (std::size_t i = 0; i < limit; ++i) {
      if (i != 0) {
        output << ", ";
      }
      output << (values[i] ? "true" : "false");
    }
    if (values.size() > limit) {
      output << ", ... (" << values.size() << ')';
    }
    output << ']';
    return output.str();
  }
  if (std::holds_alternative<std::vector<std::string>>(value.value)) {
    const auto& values = std::get<std::vector<std::string>>(value.value);
    std::ostringstream output;
    output << '[';
    const auto limit = std::min<std::size_t>(values.size(), 4U);
    for (std::size_t i = 0; i < limit; ++i) {
      if (i != 0) {
        output << ", ";
      }
      output << '"' << values[i] << '"';
    }
    if (values.size() > limit) {
      output << ", ... (" << values.size() << ')';
    }
    output << ']';
    return output.str();
  }
  return "";
}

std::vector<std::pair<std::string, std::uint64_t>> gguf_tensor_type_counts(
  const GgufFile& file) {
  std::map<std::string, std::uint64_t> counts;
  for (const auto& tensor : file.tensors) {
    counts[ggml_type_name(tensor.type)] += 1;
  }
  return {counts.begin(), counts.end()};
}

Result<std::string> gguf_get_string(const GgufFile& file, const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr) {
    return missing_key<std::string>(key);
  }
  if (!std::holds_alternative<std::string>(value->value)) {
    return Status::invalid_argument("GGUF metadata key is not string: " + key);
  }
  return std::get<std::string>(value->value);
}

Result<std::int64_t> gguf_get_i64(const GgufFile& file, const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr) {
    return missing_key<std::int64_t>(key);
  }
  if (std::holds_alternative<std::int64_t>(value->value)) {
    return std::get<std::int64_t>(value->value);
  }
  if (std::holds_alternative<std::uint64_t>(value->value)) {
    const auto unsigned_value = std::get<std::uint64_t>(value->value);
    if (unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return Status::invalid_argument("GGUF metadata integer overflows int64: " + key);
    }
    return static_cast<std::int64_t>(unsigned_value);
  }
  return Status::invalid_argument("GGUF metadata key is not integer: " + key);
}

Result<double> gguf_get_f64(const GgufFile& file, const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr) {
    return missing_key<double>(key);
  }
  if (std::holds_alternative<double>(value->value)) {
    return std::get<double>(value->value);
  }
  if (std::holds_alternative<std::int64_t>(value->value)) {
    return static_cast<double>(std::get<std::int64_t>(value->value));
  }
  if (std::holds_alternative<std::uint64_t>(value->value)) {
    return static_cast<double>(std::get<std::uint64_t>(value->value));
  }
  return Status::invalid_argument("GGUF metadata key is not numeric: " + key);
}

Result<bool> gguf_get_bool(const GgufFile& file, const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr) {
    return missing_key<bool>(key);
  }
  if (!std::holds_alternative<bool>(value->value)) {
    return Status::invalid_argument("GGUF metadata key is not bool: " + key);
  }
  return std::get<bool>(value->value);
}

Result<std::vector<std::int64_t>> gguf_get_i64_array(const GgufFile& file,
                                                     const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr) {
    return missing_key<std::vector<std::int64_t>>(key);
  }
  if (std::holds_alternative<std::vector<std::int64_t>>(value->value)) {
    return std::get<std::vector<std::int64_t>>(value->value);
  }
  if (std::holds_alternative<std::vector<std::uint64_t>>(value->value)) {
    std::vector<std::int64_t> result;
    const auto& source = std::get<std::vector<std::uint64_t>>(value->value);
    result.reserve(source.size());
    for (const auto item : source) {
      if (item > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return Status::invalid_argument("GGUF metadata array item overflows int64: " + key);
      }
      result.push_back(static_cast<std::int64_t>(item));
    }
    return result;
  }
  return Status::invalid_argument("GGUF metadata key is not an integer array: " + key);
}

Result<std::uint64_t> gguf_get_array_size(const GgufFile& file, const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr) {
    return missing_key<std::uint64_t>(key);
  }
  if (std::holds_alternative<std::vector<std::uint64_t>>(value->value)) {
    return static_cast<std::uint64_t>(std::get<std::vector<std::uint64_t>>(value->value).size());
  }
  if (std::holds_alternative<std::vector<std::int64_t>>(value->value)) {
    return static_cast<std::uint64_t>(std::get<std::vector<std::int64_t>>(value->value).size());
  }
  if (std::holds_alternative<std::vector<double>>(value->value)) {
    return static_cast<std::uint64_t>(std::get<std::vector<double>>(value->value).size());
  }
  if (std::holds_alternative<std::vector<bool>>(value->value)) {
    return static_cast<std::uint64_t>(std::get<std::vector<bool>>(value->value).size());
  }
  if (std::holds_alternative<std::vector<std::string>>(value->value)) {
    return static_cast<std::uint64_t>(std::get<std::vector<std::string>>(value->value).size());
  }
  return Status::invalid_argument("GGUF metadata key is not array: " + key);
}

}  // namespace toyllm
