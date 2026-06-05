#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace toyllm::cpu {

struct TensorView {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  const std::byte* data{nullptr};
  std::uint64_t data_offset_begin{0};
  std::uint64_t data_offset_end{0};
  std::uint64_t byte_size{0};
};

class SafeTensorMap {
 public:
  static SafeTensorMap load(const std::filesystem::path& path);

  SafeTensorMap() = default;
  ~SafeTensorMap();

  SafeTensorMap(const SafeTensorMap&) = delete;
  SafeTensorMap& operator=(const SafeTensorMap&) = delete;

  SafeTensorMap(SafeTensorMap&& other) noexcept;
  SafeTensorMap& operator=(SafeTensorMap&& other) noexcept;

  const TensorView& at(std::string_view name) const;
  std::uint64_t file_size() const;
  std::uint64_t header_size() const;
  const std::unordered_map<std::string, TensorView>& tensors() const;

 private:
  void parse_header(std::string_view header);
  void validate_tensor_ranges() const;
  void close();
  void move_from(SafeTensorMap&& other);

  std::filesystem::path path_;
  int fd_{-1};
  const std::byte* mapped_data_{nullptr};
  std::uint64_t file_size_{0};
  std::uint64_t header_size_{0};
  std::uint64_t data_start_{0};
  std::unordered_map<std::string, TensorView> tensors_;
};

float bf16_to_float(const std::byte* data, std::uint64_t index);

}  // namespace toyllm::cpu
