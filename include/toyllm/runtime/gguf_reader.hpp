#pragma once

#include "toyllm/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace toyllm {

enum class GgufValueKind {
  uint8,
  int8,
  uint16,
  int16,
  uint32,
  int32,
  float32,
  bool_value,
  string,
  array,
  uint64,
  int64,
  float64,
};

struct GgufMetadataValue {
  GgufValueKind kind{GgufValueKind::string};
  GgufValueKind array_kind{GgufValueKind::string};
  std::variant<std::monostate, std::uint64_t, std::int64_t, double, bool, std::string,
               std::vector<std::uint64_t>, std::vector<std::int64_t>, std::vector<double>,
               std::vector<bool>, std::vector<std::string>>
    value;
};

struct GgufTensorInfo {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::uint32_t type{0};
  std::uint64_t offset{0};
  std::uint64_t absolute_offset{0};
  std::uint64_t byte_size{0};
};

struct GgufFile {
  std::filesystem::path path;
  std::uint32_t version{0};
  std::uint64_t tensor_count{0};
  std::uint64_t metadata_count{0};
  std::uint64_t file_size{0};
  std::uint64_t alignment{32};
  std::uint64_t data_offset{0};
  std::vector<GgufTensorInfo> tensors;
  std::unordered_map<std::string, GgufMetadataValue> metadata;
};

struct GgufTensorBytes {
  const std::byte* data{nullptr};
  std::size_t size{0};
};

class GgufMappedData {
 public:
  GgufMappedData();
  ~GgufMappedData();

  GgufMappedData(const GgufMappedData&) = delete;
  GgufMappedData& operator=(const GgufMappedData&) = delete;

  GgufMappedData(GgufMappedData&& other) noexcept;
  GgufMappedData& operator=(GgufMappedData&& other) noexcept;

  [[nodiscard]] static Result<GgufMappedData> open(const GgufFile& file);
  [[nodiscard]] bool valid() const;
  [[nodiscard]] std::uint64_t size() const;
  [[nodiscard]] Result<GgufTensorBytes> tensor_bytes(const GgufTensorInfo& tensor) const;

 private:
  struct Impl;

  explicit GgufMappedData(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] Result<std::filesystem::path> resolve_gguf_model_path(
  const std::filesystem::path& path);
[[nodiscard]] Result<GgufFile> read_gguf_file(const std::filesystem::path& path);

[[nodiscard]] const GgufTensorInfo* find_gguf_tensor(const GgufFile& file,
                                                     const std::string& name);
[[nodiscard]] const GgufMetadataValue* find_gguf_metadata(const GgufFile& file,
                                                          const std::string& key);
[[nodiscard]] std::string gguf_value_kind_name(GgufValueKind kind);
[[nodiscard]] std::string ggml_type_name(std::uint32_t type);
[[nodiscard]] std::string gguf_value_to_string(const GgufMetadataValue& value);
[[nodiscard]] std::vector<std::pair<std::string, std::uint64_t>> gguf_tensor_type_counts(
  const GgufFile& file);

[[nodiscard]] Result<std::string> gguf_get_string(const GgufFile& file,
                                                  const std::string& key);
[[nodiscard]] Result<std::int64_t> gguf_get_i64(const GgufFile& file,
                                                const std::string& key);
[[nodiscard]] Result<double> gguf_get_f64(const GgufFile& file, const std::string& key);
[[nodiscard]] Result<bool> gguf_get_bool(const GgufFile& file, const std::string& key);
[[nodiscard]] Result<std::vector<std::int64_t>> gguf_get_i64_array(
  const GgufFile& file, const std::string& key);
[[nodiscard]] Result<std::uint64_t> gguf_get_array_size(const GgufFile& file,
                                                        const std::string& key);

}  // namespace toyllm
