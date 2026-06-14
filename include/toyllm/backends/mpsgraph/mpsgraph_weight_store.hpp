#pragma once

#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/model/model_config.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace toyllm::mpsgraph {

struct MpsGraphTensorInfo {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t data_offset_begin{0};
  std::uint64_t data_offset_end{0};
  std::uint64_t byte_size{0};
};

struct MpsGraphDeviceTensor {
  MpsGraphBuffer buffer;
  std::vector<std::uint64_t> shape;
  std::string source_dtype;
  std::uint64_t elements{0};
};

class MpsGraphWeightStore {
 public:
  [[nodiscard]] static Result<MpsGraphWeightStore> load_metadata(
    const std::filesystem::path& path);

  MpsGraphWeightStore() = default;
  ~MpsGraphWeightStore();

  MpsGraphWeightStore(const MpsGraphWeightStore&) = delete;
  MpsGraphWeightStore& operator=(const MpsGraphWeightStore&) = delete;

  MpsGraphWeightStore(MpsGraphWeightStore&& other) noexcept;
  MpsGraphWeightStore& operator=(MpsGraphWeightStore&& other) noexcept;

  [[nodiscard]] const MpsGraphTensorInfo& at(std::string_view name) const;
  [[nodiscard]] bool contains(std::string_view name) const;
  [[nodiscard]] std::uint64_t file_size() const;
  [[nodiscard]] std::uint64_t header_size() const;
  [[nodiscard]] const std::unordered_map<std::string, MpsGraphTensorInfo>& tensors() const;

  [[nodiscard]] Status validate_qwen3_shapes(const ModelConfig& config) const;
  [[nodiscard]] Result<MpsGraphDeviceTensor> upload_tensor_f32(
    const MpsGraphContext& context, std::string_view name) const;

 private:
  [[nodiscard]] static Result<MpsGraphWeightStore> load_metadata_impl(
    const std::filesystem::path& path);

  void parse_header(std::string_view header);
  [[nodiscard]] Status validate_tensor_ranges() const;
  void close();
  void move_from(MpsGraphWeightStore&& other) noexcept;

  std::filesystem::path path_;
  int fd_{-1};
  const std::byte* mapped_data_{nullptr};
  std::uint64_t file_size_{0};
  std::uint64_t header_size_{0};
  std::uint64_t data_start_{0};
  std::unordered_map<std::string, MpsGraphTensorInfo> tensors_;
};

}  // namespace toyllm::mpsgraph
