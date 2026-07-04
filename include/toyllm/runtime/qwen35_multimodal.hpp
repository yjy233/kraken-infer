#pragma once

#include "toyllm/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace toyllm {

struct Qwen35MmprojMetadata {
  std::filesystem::path path;
  std::string architecture;
  std::string projector_type;
  std::string vision_projector_type;
  std::int64_t spatial_merge_size{0};
  std::int64_t patch_size{0};
  std::int64_t vision_block_count{0};
  std::int64_t vision_embedding_length{0};
  std::size_t tensor_count{0};
  std::size_t metadata_count{0};
  std::uint64_t file_size{0};
  std::size_t deepstack_layer_count{0};
  std::uint64_t projector_output_width{0};
  bool qwen3vl_required_tensors_present{false};
  std::vector<std::string> missing_required_tensors;
};

struct Qwen35ImageDataUrl {
  std::string mime_type;
  std::vector<std::uint8_t> bytes;
  std::uint32_t width{0};
  std::uint32_t height{0};
};

struct Qwen35ImageDimensions {
  std::uint32_t width{0};
  std::uint32_t height{0};
};

[[nodiscard]] Result<Qwen35MmprojMetadata> load_qwen35_mmproj_metadata(
  const std::filesystem::path& path);
[[nodiscard]] bool qwen35_mmproj_is_qwen3vl_merger(
  const Qwen35MmprojMetadata& metadata);
[[nodiscard]] Status validate_qwen35_mmproj_text_embedding_compatibility(
  const Qwen35MmprojMetadata& metadata, std::int64_t text_embedding_length);
[[nodiscard]] std::string format_qwen35_mmproj_metadata_summary(
  const Qwen35MmprojMetadata& metadata);
[[nodiscard]] bool qwen35_image_url_is_data_url(std::string_view url);
[[nodiscard]] std::uint64_t qwen35_image_content_fingerprint(
  std::string_view image_url, std::string_view image_mime_type,
  std::string_view detail, const std::vector<std::uint8_t>& image_bytes);
[[nodiscard]] Result<Qwen35ImageDimensions> infer_qwen35_image_dimensions(
  std::string_view mime_type, const std::vector<std::uint8_t>& image_bytes);
[[nodiscard]] Result<Qwen35ImageDataUrl> parse_qwen35_image_data_url(
  std::string_view url);

}  // namespace toyllm
