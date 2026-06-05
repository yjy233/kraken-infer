#include "safetensors.hpp"

#include "json_scanner.hpp"

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

namespace toyllm::cpu {

namespace {

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

}  // namespace

float bf16_to_float(const std::byte* data, std::uint64_t index) {
  const auto* raw = reinterpret_cast<const std::uint16_t*>(data);
  const std::uint32_t bits = static_cast<std::uint32_t>(raw[index]) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

SafeTensorMap SafeTensorMap::load(const std::filesystem::path& path) {
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

SafeTensorMap::~SafeTensorMap() { close(); }

SafeTensorMap::SafeTensorMap(SafeTensorMap&& other) noexcept { move_from(std::move(other)); }

SafeTensorMap& SafeTensorMap::operator=(SafeTensorMap&& other) noexcept {
  if (this != &other) {
    close();
    move_from(std::move(other));
  }
  return *this;
}

const TensorView& SafeTensorMap::at(std::string_view name) const {
  const auto it = tensors_.find(std::string(name));
  if (it == tensors_.end()) {
    throw std::runtime_error("missing tensor: " + std::string(name));
  }
  return it->second;
}

std::uint64_t SafeTensorMap::file_size() const { return file_size_; }

std::uint64_t SafeTensorMap::header_size() const { return header_size_; }

const std::unordered_map<std::string, TensorView>& SafeTensorMap::tensors() const {
  return tensors_;
}

void SafeTensorMap::parse_header(std::string_view header) {
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

void SafeTensorMap::validate_tensor_ranges() const {
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

void SafeTensorMap::close() {
  if (mapped_data_ != nullptr) {
    (void)::munmap(const_cast<std::byte*>(mapped_data_), static_cast<std::size_t>(file_size_));
    mapped_data_ = nullptr;
  }
  if (fd_ >= 0) {
    (void)::close(fd_);
    fd_ = -1;
  }
}

void SafeTensorMap::move_from(SafeTensorMap&& other) {
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

}  // namespace toyllm::cpu
