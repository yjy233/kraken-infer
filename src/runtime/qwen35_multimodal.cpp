#include "toyllm/runtime/qwen35_multimodal.hpp"

#include "toyllm/runtime/gguf_reader.hpp"
#include "toyllm/runtime/gguf_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <cmath>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>
#include <variant>

namespace toyllm {

namespace {

constexpr std::uint64_t kImageFingerprintRoot = 1469598103934665603ULL;
constexpr std::uint64_t kImageFingerprintPrime = 1099511628211ULL;
constexpr std::size_t kQwen35VlDefaultMinImageTokens = 8;
constexpr std::size_t kQwen35VlDefaultMaxImageTokens = 4096;

std::uint64_t fingerprint_mix(std::uint64_t hash, std::uint64_t value) {
  hash ^= value;
  hash *= kImageFingerprintPrime;
  return hash;
}

std::uint64_t fingerprint_mix_bytes(std::uint64_t hash,
                                    const std::uint8_t* data,
                                    std::size_t size) {
  hash = fingerprint_mix(hash, static_cast<std::uint64_t>(size));
  for (std::size_t index = 0; index < size; ++index) {
    hash = fingerprint_mix(hash, data[index]);
  }
  return hash;
}

std::uint64_t fingerprint_mix_string(std::uint64_t hash, std::string_view value) {
  return fingerprint_mix_bytes(hash,
                               reinterpret_cast<const std::uint8_t*>(value.data()),
                               value.size());
}

std::string optional_gguf_string(const GgufFile& file, const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr || !std::holds_alternative<std::string>(value->value)) {
    return {};
  }
  return std::get<std::string>(value->value);
}

std::int64_t optional_gguf_i64(const GgufFile& file, const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr) {
    return 0;
  }
  if (std::holds_alternative<std::int64_t>(value->value)) {
    return std::get<std::int64_t>(value->value);
  }
  if (std::holds_alternative<std::uint64_t>(value->value)) {
    return static_cast<std::int64_t>(std::get<std::uint64_t>(value->value));
  }
  return 0;
}

double optional_gguf_f64(const GgufFile& file, const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr) {
    return 0.0;
  }
  if (std::holds_alternative<double>(value->value)) {
    return std::get<double>(value->value);
  }
  return 0.0;
}

std::vector<bool> optional_gguf_bool_array(const GgufFile& file,
                                           const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr ||
      !std::holds_alternative<std::vector<bool>>(value->value)) {
    return {};
  }
  return std::get<std::vector<bool>>(value->value);
}

Result<std::array<float, 3>> required_gguf_f32x3(const GgufFile& file,
                                                 const std::string& key) {
  const auto* value = find_gguf_metadata(file, key);
  if (value == nullptr || !std::holds_alternative<std::vector<double>>(value->value)) {
    return Status::invalid_argument("mmproj GGUF is missing float array metadata: " +
                                    key);
  }
  const auto& values = std::get<std::vector<double>>(value->value);
  if (values.size() != 3U) {
    return Status::invalid_argument("mmproj GGUF float array metadata must have 3 values: " +
                                    key);
  }
  std::array<float, 3> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<float>(values[index]);
  }
  return result;
}

std::vector<std::size_t> qwen3vl_deepstack_indices_from_flags(
  const std::vector<bool>& flags) {
  std::vector<std::size_t> indices;
  for (std::size_t index = 0; index < flags.size(); ++index) {
    if (flags[index]) {
      indices.push_back(index);
    }
  }
  return indices;
}

std::string ascii_lower(std::string value) {
  for (auto& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

bool has_gguf_tensor(const GgufFile& file, std::string_view name) {
  return find_gguf_tensor(file, std::string{name}) != nullptr;
}

void append_missing_tensor(const GgufFile& file, std::string_view name,
                           std::vector<std::string>& missing) {
  if (!has_gguf_tensor(file, name)) {
    missing.emplace_back(name);
  }
}

std::string join_names(const std::vector<std::string>& names) {
  std::ostringstream output;
  const auto limit = std::min<std::size_t>(names.size(), 12U);
  for (std::size_t index = 0; index < limit; ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << names[index];
  }
  if (names.size() > limit) {
    output << ", ... (" << names.size() << " total)";
  }
  return output.str();
}

bool parse_qwen3vl_deepstack_index(std::string_view tensor_name,
                                   std::size_t& layer_index) {
  constexpr std::string_view prefix = "v.deepstack.";
  if (tensor_name.size() <= prefix.size() ||
      tensor_name.substr(0, prefix.size()) != prefix) {
    return false;
  }
  std::size_t value = 0;
  std::size_t cursor = prefix.size();
  bool saw_digit = false;
  while (cursor < tensor_name.size() &&
         std::isdigit(static_cast<unsigned char>(tensor_name[cursor])) != 0) {
    const auto digit = static_cast<std::size_t>(tensor_name[cursor] - '0');
    if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
      return false;
    }
    value = value * 10U + digit;
    ++cursor;
    saw_digit = true;
  }
  if (!saw_digit || cursor >= tensor_name.size() || tensor_name[cursor] != '.') {
    return false;
  }
  layer_index = value;
  return true;
}

std::vector<std::size_t> qwen3vl_deepstack_layer_indices(const GgufFile& file) {
  std::vector<std::size_t> indices;
  for (const auto& tensor : file.tensors) {
    std::size_t index = 0;
    if (parse_qwen3vl_deepstack_index(tensor.name, index)) {
      indices.push_back(index);
    }
  }
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  return indices;
}

std::string qwen3vl_deepstack_tensor_name(std::size_t layer_index,
                                          std::string_view suffix) {
  std::ostringstream output;
  output << "v.deepstack." << layer_index << '.' << suffix;
  return output.str();
}

std::vector<std::string> qwen3vl_missing_required_tensors(
  const GgufFile& file, const std::vector<bool>& deepstack_layer_flags,
  std::size_t& deepstack_layers) {
  std::vector<std::string> missing;
  constexpr std::array<std::string_view, 8> required{
    "v.patch_embd.weight",
    "v.patch_embd.weight.1",
    "v.patch_embd.bias",
    "v.position_embd.weight",
    "mm.0.weight",
    "mm.0.bias",
    "mm.2.weight",
    "mm.2.bias",
  };
  for (const auto name : required) {
    append_missing_tensor(file, name, missing);
  }

  auto deepstack_indices =
    qwen3vl_deepstack_indices_from_flags(deepstack_layer_flags);
  auto tensor_deepstack_indices = qwen3vl_deepstack_layer_indices(file);
  deepstack_indices.insert(deepstack_indices.end(), tensor_deepstack_indices.begin(),
                           tensor_deepstack_indices.end());
  std::sort(deepstack_indices.begin(), deepstack_indices.end());
  deepstack_indices.erase(
    std::unique(deepstack_indices.begin(), deepstack_indices.end()),
    deepstack_indices.end());
  deepstack_layers = deepstack_indices.size();
  constexpr std::array<std::string_view, 6> deepstack_suffixes{
    "norm.weight",
    "norm.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
  };
  for (const auto layer_index : deepstack_indices) {
    for (const auto suffix : deepstack_suffixes) {
      append_missing_tensor(file, qwen3vl_deepstack_tensor_name(layer_index, suffix),
                            missing);
    }
  }
  return missing;
}

std::uint64_t qwen3vl_projector_output_width(const GgufFile& file,
                                             std::size_t deepstack_layers) {
  const auto* bias = find_gguf_tensor(file, "mm.2.bias");
  if (bias == nullptr || bias->shape.empty()) {
    return 0;
  }
  const auto multiplier = static_cast<std::uint64_t>(deepstack_layers + 1U);
  if (bias->shape[0] > std::numeric_limits<std::uint64_t>::max() / multiplier) {
    return 0;
  }
  return bias->shape[0] * multiplier;
}

std::string format_shape(const std::vector<std::uint64_t>& shape) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      output << 'x';
    }
    output << shape[index];
  }
  output << ']';
  return output.str();
}

Qwen35VisionTensorPlan make_vision_tensor_plan(const GgufTensorInfo& tensor) {
  Qwen35VisionTensorPlan plan;
  plan.name = tensor.name;
  plan.shape = tensor.shape;
  plan.type = tensor.type;
  plan.byte_size = tensor.byte_size;
  return plan;
}

Result<std::uint64_t> checked_u64_mul(std::uint64_t lhs, std::uint64_t rhs,
                                      std::string_view label);
Result<std::size_t> checked_size_from_u64(std::uint64_t value,
                                          std::string_view label);

Result<Qwen35VisionTensorPlan> require_tensor_shape(
  const GgufFile& file, std::string_view name,
  const std::vector<std::uint64_t>& expected_shape) {
  const auto* tensor = find_gguf_tensor(file, std::string{name});
  if (tensor == nullptr) {
    return Status::invalid_argument("qwen3vl_merger mmproj is missing tensor: " +
                                    std::string{name});
  }
  if (tensor->shape != expected_shape) {
    return Status::invalid_argument(
      "qwen3vl_merger tensor " + std::string{name} + " shape " +
      format_shape(tensor->shape) + " does not match expected " +
      format_shape(expected_shape));
  }
  return make_vision_tensor_plan(*tensor);
}

Result<std::uint64_t> required_positive_u64_from_i64(std::int64_t value,
                                                     std::string_view label) {
  if (value <= 0) {
    return Status::invalid_argument(std::string{label} + " must be positive");
  }
  return static_cast<std::uint64_t>(value);
}

Status append_required_tensor(
  const GgufFile& file, std::string_view name,
  const std::vector<std::uint64_t>& expected_shape,
  std::vector<Qwen35VisionTensorPlan>& tensors) {
  auto tensor = require_tensor_shape(file, name, expected_shape);
  if (!tensor.is_ok()) {
    return tensor.status();
  }
  tensors.push_back(std::move(tensor.value()));
  return Status::ok();
}

Result<std::size_t> tensor_element_count(const std::vector<std::uint64_t>& shape,
                                         std::string_view label) {
  std::uint64_t elements = 1;
  for (const auto dim : shape) {
    auto multiplied = checked_u64_mul(elements, dim, label);
    if (!multiplied.is_ok()) {
      return multiplied.status();
    }
    elements = multiplied.value();
  }
  return checked_size_from_u64(elements, label);
}

float bf16_to_f32(std::uint16_t value) {
  const auto bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

Result<std::vector<float>> tensor_bytes_to_f32(const GgufTensorInfo& tensor,
                                               const GgufTensorBytes& bytes) {
  auto elements = tensor_element_count(tensor.shape, tensor.name);
  if (!elements.is_ok()) {
    return elements.status();
  }
  std::vector<float> values(elements.value());
  if (tensor.type == 0U) {
    const auto expected_bytes = checked_u64_mul(
      static_cast<std::uint64_t>(elements.value()), sizeof(float),
      "GGUF F32 tensor bytes");
    if (!expected_bytes.is_ok()) {
      return expected_bytes.status();
    }
    if (bytes.size != expected_bytes.value()) {
      return Status::invalid_argument("GGUF F32 tensor byte size mismatch: " +
                                      tensor.name);
    }
    std::memcpy(values.data(), bytes.data, bytes.size);
    return values;
  }
  if (tensor.type == 30U) {
    const auto expected_bytes = checked_u64_mul(
      static_cast<std::uint64_t>(elements.value()), sizeof(std::uint16_t),
      "GGUF BF16 tensor bytes");
    if (!expected_bytes.is_ok()) {
      return expected_bytes.status();
    }
    if (bytes.size != expected_bytes.value()) {
      return Status::invalid_argument("GGUF BF16 tensor byte size mismatch: " +
                                      tensor.name);
    }
    for (std::size_t index = 0; index < elements.value(); ++index) {
      std::uint16_t raw = 0;
      std::memcpy(&raw, bytes.data + index * sizeof(raw), sizeof(raw));
      values[index] = bf16_to_f32(raw);
    }
    return values;
  }
  return Status::invalid_argument(
    "Qwen3.5 CPU vision input stage supports F32/BF16 tensors only: " +
    tensor.name + " has " + ggml_type_name(tensor.type));
}

Result<std::vector<float>> load_tensor_f32(const GgufFile& file,
                                           const GgufMappedData& mapped,
                                           std::string_view name,
                                           const std::vector<std::uint64_t>& shape) {
  auto plan = require_tensor_shape(file, name, shape);
  if (!plan.is_ok()) {
    return plan.status();
  }
  const auto* tensor = find_gguf_tensor(file, std::string{name});
  if (tensor == nullptr) {
    return Status::invalid_argument("missing tensor: " + std::string{name});
  }
  auto bytes = mapped.tensor_bytes(*tensor);
  if (!bytes.is_ok()) {
    return bytes.status();
  }
  return tensor_bytes_to_f32(*tensor, bytes.value());
}

std::string vision_block_tensor_name(std::size_t layer_index,
                                     std::string_view stem,
                                     std::string_view suffix) {
  std::ostringstream output;
  output << "v.blk." << layer_index << '.' << stem;
  if (!suffix.empty()) {
    output << '.' << suffix;
  }
  return output.str();
}

std::uint16_t read_be_u16(const std::vector<std::uint8_t>& bytes,
                          std::size_t offset) {
  return static_cast<std::uint16_t>(
    (static_cast<std::uint16_t>(bytes[offset]) << 8U) |
    static_cast<std::uint16_t>(bytes[offset + 1U]));
}

std::uint32_t read_be_u32(const std::vector<std::uint8_t>& bytes,
                          std::size_t offset) {
  return (static_cast<std::uint32_t>(bytes[offset]) << 24U) |
         (static_cast<std::uint32_t>(bytes[offset + 1U]) << 16U) |
         (static_cast<std::uint32_t>(bytes[offset + 2U]) << 8U) |
         static_cast<std::uint32_t>(bytes[offset + 3U]);
}

Result<Qwen35ImageDimensions> infer_png_dimensions(
  const std::vector<std::uint8_t>& image_bytes) {
  constexpr std::array<std::uint8_t, 8> signature{
    0x89U, 0x50U, 0x4EU, 0x47U, 0x0DU, 0x0AU, 0x1AU, 0x0AU,
  };
  if (image_bytes.size() < 24U ||
      !std::equal(signature.begin(), signature.end(), image_bytes.begin())) {
    return Status::invalid_argument("PNG image header is missing signature/IHDR");
  }
  if (std::string_view{reinterpret_cast<const char*>(image_bytes.data() + 12U), 4} !=
      "IHDR") {
    return Status::invalid_argument("PNG image header is missing IHDR chunk");
  }
  const auto width = read_be_u32(image_bytes, 16U);
  const auto height = read_be_u32(image_bytes, 20U);
  if (width == 0 || height == 0) {
    return Status::invalid_argument("PNG image dimensions must be non-zero");
  }
  return Qwen35ImageDimensions{width, height};
}

bool is_jpeg_sof_marker(std::uint8_t marker) {
  switch (marker) {
    case 0xC0U:
    case 0xC1U:
    case 0xC2U:
    case 0xC3U:
    case 0xC5U:
    case 0xC6U:
    case 0xC7U:
    case 0xC9U:
    case 0xCAU:
    case 0xCBU:
    case 0xCDU:
    case 0xCEU:
    case 0xCFU:
      return true;
    default:
      return false;
  }
}

Result<Qwen35ImageDimensions> infer_jpeg_dimensions(
  const std::vector<std::uint8_t>& image_bytes) {
  if (image_bytes.size() < 4U || image_bytes[0] != 0xFFU ||
      image_bytes[1] != 0xD8U) {
    return Status::invalid_argument("JPEG image header is missing SOI marker");
  }
  std::size_t cursor = 2U;
  while (cursor + 3U < image_bytes.size()) {
    while (cursor < image_bytes.size() && image_bytes[cursor] != 0xFFU) {
      ++cursor;
    }
    if (cursor + 1U >= image_bytes.size()) {
      break;
    }
    while (cursor < image_bytes.size() && image_bytes[cursor] == 0xFFU) {
      ++cursor;
    }
    if (cursor >= image_bytes.size()) {
      break;
    }
    const auto marker = image_bytes[cursor++];
    if (marker == 0xD9U || marker == 0xDAU) {
      break;
    }
    if (marker >= 0xD0U && marker <= 0xD7U) {
      continue;
    }
    if (cursor + 2U > image_bytes.size()) {
      break;
    }
    const auto segment_length = read_be_u16(image_bytes, cursor);
    if (segment_length < 2U ||
        cursor + static_cast<std::size_t>(segment_length) > image_bytes.size()) {
      return Status::invalid_argument("JPEG image segment length is invalid");
    }
    if (is_jpeg_sof_marker(marker)) {
      if (segment_length < 7U) {
        return Status::invalid_argument("JPEG SOF segment is too short");
      }
      const auto height = read_be_u16(image_bytes, cursor + 3U);
      const auto width = read_be_u16(image_bytes, cursor + 5U);
      if (width == 0 || height == 0) {
        return Status::invalid_argument("JPEG image dimensions must be non-zero");
      }
      return Qwen35ImageDimensions{width, height};
    }
    cursor += static_cast<std::size_t>(segment_length);
  }
  return Status::invalid_argument("JPEG image header has no SOF dimensions");
}

Result<std::uint32_t> positive_u32_from_i64(std::int64_t value,
                                            std::string_view label) {
  if (value <= 0) {
    return Status::invalid_argument(std::string{label} + " must be positive");
  }
  if (static_cast<std::uint64_t>(value) >
      std::numeric_limits<std::uint32_t>::max()) {
    return Status::invalid_argument(std::string{label} + " exceeds uint32 range");
  }
  return static_cast<std::uint32_t>(value);
}

Result<std::uint64_t> checked_u64_mul(std::uint64_t lhs, std::uint64_t rhs,
                                      std::string_view label) {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    return Status::invalid_argument(std::string{label} + " overflow");
  }
  return lhs * rhs;
}

Result<std::size_t> checked_size_from_u64(std::uint64_t value,
                                          std::string_view label) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    return Status::invalid_argument(std::string{label} + " exceeds size_t range");
  }
  return static_cast<std::size_t>(value);
}

Result<std::size_t> checked_image_elements(std::uint32_t width,
                                           std::uint32_t height,
                                           std::uint32_t channels,
                                           std::string_view label) {
  auto pixels = checked_u64_mul(width, height, label);
  if (!pixels.is_ok()) {
    return pixels.status();
  }
  auto elements = checked_u64_mul(pixels.value(), channels, label);
  if (!elements.is_ok()) {
    return elements.status();
  }
  return checked_size_from_u64(elements.value(), label);
}

Result<std::uint64_t> qwen35_image_pixel_limit(
  const Qwen35MmprojMetadata& metadata, std::size_t token_limit) {
  auto patch_size = positive_u32_from_i64(metadata.patch_size,
                                          "clip.vision.patch_size");
  if (!patch_size.is_ok()) {
    return patch_size.status();
  }
  auto spatial_merge_size =
    metadata.spatial_merge_size > 0
      ? positive_u32_from_i64(metadata.spatial_merge_size,
                              "clip.vision.spatial_merge_size")
      : Result<std::uint32_t>{2U};
  if (!spatial_merge_size.is_ok()) {
    return spatial_merge_size.status();
  }
  auto align_size = checked_u64_mul(patch_size.value(),
                                    spatial_merge_size.value(),
                                    "Qwen3.5 image align size");
  if (!align_size.is_ok()) {
    return align_size.status();
  }
  auto patch_area = checked_u64_mul(align_size.value(), align_size.value(),
                                    "Qwen3.5 image patch area");
  if (!patch_area.is_ok()) {
    return patch_area.status();
  }
  return checked_u64_mul(patch_area.value(), token_limit,
                         "Qwen3.5 image pixel limit");
}

std::uint32_t round_by_factor(std::uint32_t value, std::uint32_t factor) {
  const auto rounded = std::round(static_cast<double>(value) /
                                  static_cast<double>(factor));
  return static_cast<std::uint32_t>(rounded) * factor;
}

std::uint32_t ceil_by_factor(double value, std::uint32_t factor) {
  const auto ceiled = std::ceil(value / static_cast<double>(factor));
  return static_cast<std::uint32_t>(ceiled) * factor;
}

std::uint32_t floor_by_factor(double value, std::uint32_t factor) {
  const auto floored = std::floor(value / static_cast<double>(factor));
  return static_cast<std::uint32_t>(floored) * factor;
}

Result<Qwen35ImageDimensions> qwen35_smart_resize(std::uint32_t width,
                                                  std::uint32_t height,
                                                  std::uint32_t align_size,
                                                  std::uint64_t min_pixels,
                                                  std::uint64_t max_pixels) {
  if (width == 0 || height == 0) {
    return Status::invalid_argument("image dimensions must be non-zero");
  }
  if (align_size == 0) {
    return Status::invalid_argument("Qwen3.5 image align size must be positive");
  }
  if (min_pixels == 0 || max_pixels == 0 || min_pixels > max_pixels) {
    return Status::invalid_argument("Qwen3.5 image pixel limits are invalid");
  }
  auto image_pixels = checked_u64_mul(width, height, "Qwen3.5 image pixels");
  if (!image_pixels.is_ok()) {
    return image_pixels.status();
  }

  auto aligned_width = std::max(align_size, round_by_factor(width, align_size));
  auto aligned_height = std::max(align_size, round_by_factor(height, align_size));
  auto aligned_pixels =
    checked_u64_mul(aligned_width, aligned_height, "Qwen3.5 aligned image pixels");
  if (!aligned_pixels.is_ok()) {
    return aligned_pixels.status();
  }

  if (aligned_pixels.value() > max_pixels) {
    const auto beta =
      std::sqrt(static_cast<double>(image_pixels.value()) /
                static_cast<double>(max_pixels));
    aligned_width = std::max(align_size, floor_by_factor(width / beta, align_size));
    aligned_height = std::max(align_size, floor_by_factor(height / beta, align_size));
  } else if (aligned_pixels.value() < min_pixels) {
    const auto beta =
      std::sqrt(static_cast<double>(min_pixels) /
                static_cast<double>(image_pixels.value()));
    aligned_width = ceil_by_factor(width * beta, align_size);
    aligned_height = ceil_by_factor(height * beta, align_size);
  }

  if (aligned_width == 0 || aligned_height == 0) {
    return Status::invalid_argument("Qwen3.5 image resize produced empty dimensions");
  }
  return Qwen35ImageDimensions{aligned_width, aligned_height};
}

float lerp(float lhs, float rhs, float weight) {
  return lhs + (rhs - lhs) * weight;
}

std::size_t rgb_offset(std::uint32_t width, std::uint32_t x, std::uint32_t y) {
  return (static_cast<std::size_t>(y) * width + x) * 3U;
}

std::vector<std::uint8_t> resize_rgb_bilinear(const Qwen35ImageRgb& src,
                                              std::uint32_t target_width,
                                              std::uint32_t target_height) {
  std::vector<std::uint8_t> dst(
    static_cast<std::size_t>(target_width) * target_height * 3U);
  const auto x_ratio =
    target_width > 1U
      ? static_cast<float>(src.width - 1U) / static_cast<float>(target_width - 1U)
      : 0.0F;
  const auto y_ratio =
    target_height > 1U
      ? static_cast<float>(src.height - 1U) / static_cast<float>(target_height - 1U)
      : 0.0F;

  for (std::uint32_t y = 0; y < target_height; ++y) {
    for (std::uint32_t x = 0; x < target_width; ++x) {
      const auto px = static_cast<float>(x) * x_ratio;
      const auto py = static_cast<float>(y) * y_ratio;
      const auto x0 = std::min(static_cast<std::uint32_t>(px), src.width - 1U);
      const auto y0 = std::min(static_cast<std::uint32_t>(py), src.height - 1U);
      const auto x1 = std::min(x0 + 1U, src.width - 1U);
      const auto y1 = std::min(y0 + 1U, src.height - 1U);
      const auto xf = px - static_cast<float>(x0);
      const auto yf = py - static_cast<float>(y0);

      const auto p00 = rgb_offset(src.width, x0, y0);
      const auto p10 = rgb_offset(src.width, x1, y0);
      const auto p01 = rgb_offset(src.width, x0, y1);
      const auto p11 = rgb_offset(src.width, x1, y1);
      const auto out = rgb_offset(target_width, x, y);
      for (std::size_t channel = 0; channel < 3U; ++channel) {
        const auto top = lerp(static_cast<float>(src.pixels[p00 + channel]),
                              static_cast<float>(src.pixels[p10 + channel]), xf);
        const auto bottom = lerp(static_cast<float>(src.pixels[p01 + channel]),
                                 static_cast<float>(src.pixels[p11 + channel]), xf);
        const auto value = std::clamp(lerp(top, bottom, yf), 0.0F, 255.0F);
        dst[out + channel] = static_cast<std::uint8_t>(value);
      }
    }
  }
  return dst;
}

std::vector<float> resize_position_embeddings_bilinear(
  const std::vector<float>& source, std::uint32_t source_grid,
  std::uint32_t target_width, std::uint32_t target_height,
  std::uint64_t embedding_length) {
  std::vector<float> resized(
    static_cast<std::size_t>(target_width) * target_height * embedding_length);
  const auto x_ratio =
    target_width > 1U
      ? static_cast<float>(source_grid - 1U) / static_cast<float>(target_width - 1U)
      : 0.0F;
  const auto y_ratio =
    target_height > 1U
      ? static_cast<float>(source_grid - 1U) / static_cast<float>(target_height - 1U)
      : 0.0F;
  const auto emb = static_cast<std::size_t>(embedding_length);
  for (std::uint32_t y = 0; y < target_height; ++y) {
    for (std::uint32_t x = 0; x < target_width; ++x) {
      const auto px = static_cast<float>(x) * x_ratio;
      const auto py = static_cast<float>(y) * y_ratio;
      const auto x0 = std::min(static_cast<std::uint32_t>(px), source_grid - 1U);
      const auto y0 = std::min(static_cast<std::uint32_t>(py), source_grid - 1U);
      const auto x1 = std::min(x0 + 1U, source_grid - 1U);
      const auto y1 = std::min(y0 + 1U, source_grid - 1U);
      const auto xf = px - static_cast<float>(x0);
      const auto yf = py - static_cast<float>(y0);
      const auto p00 = emb * (static_cast<std::size_t>(x0) + source_grid * y0);
      const auto p10 = emb * (static_cast<std::size_t>(x1) + source_grid * y0);
      const auto p01 = emb * (static_cast<std::size_t>(x0) + source_grid * y1);
      const auto p11 = emb * (static_cast<std::size_t>(x1) + source_grid * y1);
      const auto out = emb * (static_cast<std::size_t>(x) + target_width * y);
      for (std::size_t channel = 0; channel < emb; ++channel) {
        const auto top = lerp(source[p00 + channel], source[p10 + channel], xf);
        const auto bottom = lerp(source[p01 + channel], source[p11 + channel], xf);
        resized[out + channel] = lerp(top, bottom, yf);
      }
    }
  }
  return resized;
}

std::vector<float> qwen3vl_spatial_merge_ewh(const std::vector<float>& input,
                                             std::uint32_t patch_grid_x,
                                             std::uint32_t patch_grid_y,
                                             std::uint64_t embedding_length) {
  const auto emb = static_cast<std::size_t>(embedding_length);
  std::vector<float> output(
    static_cast<std::size_t>(patch_grid_x) * patch_grid_y * emb);
  std::size_t token = 0;
  for (std::uint32_t y_block = 0; y_block < patch_grid_y / 2U; ++y_block) {
    for (std::uint32_t x_block = 0; x_block < patch_grid_x / 2U; ++x_block) {
      for (std::uint32_t y_inner = 0; y_inner < 2U; ++y_inner) {
        for (std::uint32_t x_inner = 0; x_inner < 2U; ++x_inner) {
          const auto x = x_block * 2U + x_inner;
          const auto y = y_block * 2U + y_inner;
          const auto input_offset =
            emb * (static_cast<std::size_t>(x) + patch_grid_x * y);
          const auto output_offset = token * emb;
          std::copy(input.begin() + static_cast<std::ptrdiff_t>(input_offset),
                    input.begin() + static_cast<std::ptrdiff_t>(input_offset + emb),
                    output.begin() + static_cast<std::ptrdiff_t>(output_offset));
          ++token;
        }
      }
    }
  }
  return output;
}

Result<std::vector<float>> qwen3vl_patch_embedding_temporal_sum(
  const Qwen35ImagePreprocessResult& image, const std::vector<float>& weight_0,
  const std::vector<float>& weight_1, std::uint32_t patch_size,
  std::uint32_t patch_grid_x, std::uint32_t patch_grid_y,
  std::uint64_t embedding_length) {
  const auto emb = static_cast<std::size_t>(embedding_length);
  auto elements = checked_image_elements(
    patch_grid_x, patch_grid_y, static_cast<std::uint32_t>(embedding_length),
    "Qwen3.5 patch embedding output elements");
  if (!elements.is_ok()) {
    return elements.status();
  }
  std::vector<float> output(elements.value());
  for (std::uint32_t py = 0; py < patch_grid_y; ++py) {
    for (std::uint32_t px = 0; px < patch_grid_x; ++px) {
      for (std::size_t e = 0; e < emb; ++e) {
        float sum = 0.0F;
        for (std::uint32_t ky = 0; ky < patch_size; ++ky) {
          for (std::uint32_t kx = 0; kx < patch_size; ++kx) {
            const auto image_x = px * patch_size + kx;
            const auto image_y = py * patch_size + ky;
            const auto pixel_offset =
              (static_cast<std::size_t>(image_y) * image.width + image_x) * 3U;
            for (std::size_t channel = 0; channel < 3U; ++channel) {
              const auto weight_offset =
                static_cast<std::size_t>(kx) +
                patch_size * (static_cast<std::size_t>(ky) +
                patch_size * (channel + 3U * e));
              sum += image.pixels[pixel_offset + channel] *
                     (weight_0[weight_offset] + weight_1[weight_offset]);
            }
          }
        }
        output[e + emb * (static_cast<std::size_t>(px) + patch_grid_x * py)] = sum;
      }
    }
  }
  return output;
}

float qwen35_gelu(float value) {
  constexpr float kGeluCoefA = 0.044715F;
  constexpr float kSqrt2OverPi = 0.7978845608028654F;
  return 0.5F * value *
         (1.0F + std::tanh(kSqrt2OverPi * value *
                            (1.0F + kGeluCoefA * value * value)));
}

void qwen35_gelu_in_place(std::vector<float>& values) {
  for (auto& value : values) {
    value = qwen35_gelu(value);
  }
}

Result<std::vector<float>> qwen35_layer_norm_token_major(
  const std::vector<float>& input, std::size_t token_count,
  std::size_t width, const std::vector<float>& weight,
  const std::vector<float>& bias, double epsilon, std::string_view label) {
  const auto expected = token_count * width;
  if (input.size() != expected || weight.size() != width ||
      bias.size() != width) {
    return Status::invalid_argument(
      std::string{label} + " layer norm shape mismatch");
  }
  if (width == 0 || token_count == 0) {
    return Status::invalid_argument(
      std::string{label} + " layer norm requires non-empty input");
  }
  if (epsilon <= 0.0) {
    return Status::invalid_argument(
      std::string{label} + " layer norm epsilon must be positive");
  }
  std::vector<float> output(expected);
  for (std::size_t token = 0; token < token_count; ++token) {
    const auto offset = token * width;
    double mean = 0.0;
    for (std::size_t channel = 0; channel < width; ++channel) {
      mean += static_cast<double>(input[offset + channel]);
    }
    mean /= static_cast<double>(width);
    double variance = 0.0;
    for (std::size_t channel = 0; channel < width; ++channel) {
      const auto centered = static_cast<double>(input[offset + channel]) - mean;
      variance += centered * centered;
    }
    variance /= static_cast<double>(width);
    const auto inv_std = static_cast<float>(1.0 / std::sqrt(variance + epsilon));
    const auto mean_f = static_cast<float>(mean);
    for (std::size_t channel = 0; channel < width; ++channel) {
      output[offset + channel] =
        (input[offset + channel] - mean_f) * inv_std * weight[channel] +
        bias[channel];
    }
  }
  return output;
}

Result<std::vector<float>> qwen35_linear_token_major(
  const std::vector<float>& input, std::size_t token_count,
  std::size_t input_width, std::size_t output_width,
  const std::vector<float>& weight, const std::vector<float>& bias,
  std::string_view label) {
  if (input.size() != token_count * input_width ||
      weight.size() != input_width * output_width ||
      bias.size() != output_width) {
    return Status::invalid_argument(
      std::string{label} + " linear shape mismatch");
  }
  std::vector<float> output(token_count * output_width);
  for (std::size_t token = 0; token < token_count; ++token) {
    const auto input_offset = token * input_width;
    const auto output_offset = token * output_width;
    for (std::size_t out = 0; out < output_width; ++out) {
      double sum = static_cast<double>(bias[out]);
      for (std::size_t in = 0; in < input_width; ++in) {
        sum += static_cast<double>(input[input_offset + in]) *
               static_cast<double>(weight[in + input_width * out]);
      }
      output[output_offset + out] = static_cast<float>(sum);
    }
  }
  return output;
}

Status qwen35_add_in_place(std::vector<float>& lhs,
                           const std::vector<float>& rhs,
                           std::string_view label) {
  if (lhs.size() != rhs.size()) {
    return Status::invalid_argument(std::string{label} + " add shape mismatch");
  }
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    lhs[index] += rhs[index];
  }
  return Status::ok();
}

std::vector<std::int32_t> qwen3vl_vision_positions(std::uint32_t patch_grid_x,
                                                   std::uint32_t patch_grid_y) {
  const auto token_count =
    static_cast<std::size_t>(patch_grid_x) * patch_grid_y;
  std::vector<std::int32_t> positions(token_count * 4U);
  std::size_t token = 0;
  for (std::uint32_t y = 0; y < patch_grid_y; y += 2U) {
    for (std::uint32_t x = 0; x < patch_grid_x; x += 2U) {
      for (std::uint32_t dy = 0; dy < 2U; ++dy) {
        for (std::uint32_t dx = 0; dx < 2U; ++dx) {
          positions[token] = static_cast<std::int32_t>(y + dy);
          positions[token_count + token] = static_cast<std::int32_t>(x + dx);
          positions[2U * token_count + token] = static_cast<std::int32_t>(y + dy);
          positions[3U * token_count + token] = static_cast<std::int32_t>(x + dx);
          ++token;
        }
      }
    }
  }
  return positions;
}

void qwen3vl_apply_vision_rope_one(std::vector<float>& values,
                                   std::size_t token_count,
                                   std::size_t embedding_width,
                                   std::size_t head_count,
                                   std::size_t head_dim,
                                   const std::vector<std::int32_t>& positions) {
  constexpr float kFreqBase = 10000.0F;
  const auto rope_dims = head_dim / 2U;
  const auto section = head_dim / 4U;
  const auto section_w = section * 2U;
  const auto section_e = section * 3U;
  const auto section_dims = section * 4U;
  const auto theta_scale =
    static_cast<float>(std::pow(kFreqBase, -2.0 / static_cast<double>(rope_dims)));

  for (std::size_t token = 0; token < token_count; ++token) {
    const auto pos_t = static_cast<float>(positions[token]);
    const auto pos_h = static_cast<float>(positions[token_count + token]);
    const auto pos_w = static_cast<float>(positions[2U * token_count + token]);
    const auto pos_e = static_cast<float>(positions[3U * token_count + token]);
    for (std::size_t head = 0; head < head_count; ++head) {
      const auto base = token * embedding_width + head * head_dim;
      float theta_t = pos_t;
      float theta_h = pos_h;
      float theta_w = pos_w;
      float theta_e = pos_e;
      for (std::size_t i0 = 0; i0 < head_dim; i0 += 2U) {
        const auto sector = (i0 / 2U) % section_dims;
        if (sector == 0U) {
          theta_t = pos_t;
        } else if (sector == section) {
          theta_h = pos_h;
        } else if (sector == section_w) {
          theta_w = pos_w;
        } else if (sector == section_e) {
          theta_e = pos_e;
        }

        float theta = theta_t;
        if (sector >= section && sector < section_w) {
          theta = theta_h;
        } else if (sector >= section_w && sector < section_e) {
          theta = theta_w;
        } else if (sector >= section_e) {
          theta = theta_e;
        }

        const auto first = base + i0 / 2U;
        const auto second = first + rope_dims;
        const auto cos_theta = std::cos(theta);
        const auto sin_theta = std::sin(theta);
        const auto x0 = values[first];
        const auto x1 = values[second];
        values[first] = x0 * cos_theta - x1 * sin_theta;
        values[second] = x0 * sin_theta + x1 * cos_theta;

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
      }
    }
  }
}

Result<std::vector<float>> qwen3vl_attention_token_major(
  const std::vector<float>& qkv, std::size_t token_count,
  std::size_t embedding_width, std::size_t head_count,
  std::uint32_t patch_grid_x, std::uint32_t patch_grid_y,
  std::string_view label) {
  if (head_count == 0 || embedding_width % head_count != 0) {
    return Status::invalid_argument(
      std::string{label} + " attention head shape mismatch");
  }
  const auto head_dim = embedding_width / head_count;
  if (head_dim == 0 || head_dim % 4U != 0) {
    return Status::invalid_argument(
      std::string{label} + " vision RoPE requires head_dim divisible by 4");
  }
  if (qkv.size() != token_count * embedding_width * 3U) {
    return Status::invalid_argument(
      std::string{label} + " qkv shape mismatch");
  }

  std::vector<float> q(token_count * embedding_width);
  std::vector<float> k(token_count * embedding_width);
  std::vector<float> v(token_count * embedding_width);
  for (std::size_t token = 0; token < token_count; ++token) {
    const auto src = token * embedding_width * 3U;
    const auto dst = token * embedding_width;
    std::copy(qkv.begin() + static_cast<std::ptrdiff_t>(src),
              qkv.begin() + static_cast<std::ptrdiff_t>(src + embedding_width),
              q.begin() + static_cast<std::ptrdiff_t>(dst));
    std::copy(qkv.begin() + static_cast<std::ptrdiff_t>(src + embedding_width),
              qkv.begin() + static_cast<std::ptrdiff_t>(src + embedding_width * 2U),
              k.begin() + static_cast<std::ptrdiff_t>(dst));
    std::copy(qkv.begin() + static_cast<std::ptrdiff_t>(src + embedding_width * 2U),
              qkv.begin() + static_cast<std::ptrdiff_t>(src + embedding_width * 3U),
              v.begin() + static_cast<std::ptrdiff_t>(dst));
  }

  const auto positions = qwen3vl_vision_positions(patch_grid_x, patch_grid_y);
  qwen3vl_apply_vision_rope_one(q, token_count, embedding_width, head_count,
                                head_dim, positions);
  qwen3vl_apply_vision_rope_one(k, token_count, embedding_width, head_count,
                                head_dim, positions);

  const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> output(token_count * embedding_width);
  std::vector<float> scores(token_count);
  for (std::size_t query = 0; query < token_count; ++query) {
    for (std::size_t head = 0; head < head_count; ++head) {
      float max_score = -std::numeric_limits<float>::infinity();
      for (std::size_t key = 0; key < token_count; ++key) {
        double dot = 0.0;
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
          const auto q_index = query * embedding_width + head * head_dim + dim;
          const auto k_index = key * embedding_width + head * head_dim + dim;
          dot += static_cast<double>(q[q_index]) * static_cast<double>(k[k_index]);
        }
        const auto score = static_cast<float>(dot) * scale;
        scores[key] = score;
        max_score = std::max(max_score, score);
      }

      double denominator = 0.0;
      for (auto& score : scores) {
        score = static_cast<float>(std::exp(static_cast<double>(score - max_score)));
        denominator += static_cast<double>(score);
      }
      if (denominator == 0.0) {
        return Status::internal_error(
          std::string{label} + " attention softmax underflow");
      }
      for (std::size_t dim = 0; dim < head_dim; ++dim) {
        double sum = 0.0;
        for (std::size_t key = 0; key < token_count; ++key) {
          const auto v_index = key * embedding_width + head * head_dim + dim;
          sum += static_cast<double>(scores[key]) /
                 denominator * static_cast<double>(v[v_index]);
        }
        const auto out_index = query * embedding_width + head * head_dim + dim;
        output[out_index] = static_cast<float>(sum);
      }
    }
  }
  return output;
}

Result<std::vector<float>> qwen3vl_group_four_tokens(
  const std::vector<float>& input, std::size_t token_count,
  std::size_t width, std::string_view label) {
  if (token_count == 0 || token_count % 4U != 0 ||
      input.size() != token_count * width) {
    return Status::invalid_argument(
      std::string{label} + " requires token_count divisible by 4");
  }
  const auto output_tokens = token_count / 4U;
  const auto output_width = width * 4U;
  std::vector<float> output(output_tokens * output_width);
  for (std::size_t token = 0; token < output_tokens; ++token) {
    for (std::size_t group = 0; group < 4U; ++group) {
      const auto src = (token * 4U + group) * width;
      const auto dst = token * output_width + group * width;
      std::copy(input.begin() + static_cast<std::ptrdiff_t>(src),
                input.begin() + static_cast<std::ptrdiff_t>(src + width),
                output.begin() + static_cast<std::ptrdiff_t>(dst));
    }
  }
  return output;
}

Result<std::vector<float>> qwen3vl_projector_mlp_token_major(
  const std::vector<float>& input, std::size_t token_count,
  std::size_t input_width, std::size_t hidden_width,
  std::size_t output_width, const std::vector<float>& fc1_weight,
  const std::vector<float>& fc1_bias, const std::vector<float>& fc2_weight,
  const std::vector<float>& fc2_bias, std::string_view label) {
  auto hidden = qwen35_linear_token_major(
    input, token_count, input_width, hidden_width, fc1_weight, fc1_bias, label);
  if (!hidden.is_ok()) {
    return hidden.status();
  }
  qwen35_gelu_in_place(hidden.value());
  return qwen35_linear_token_major(
    hidden.value(), token_count, hidden_width, output_width,
    fc2_weight, fc2_bias, label);
}

Result<std::int32_t> qwen35_i32_position(std::size_t value,
                                         std::string_view label) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return Status::invalid_argument(std::string{label} + " exceeds int32 range");
  }
  return static_cast<std::int32_t>(value);
}

Status append_mixed_text_mrope_positions(std::vector<std::int32_t>& positions,
                                         std::size_t total_tokens,
                                         std::size_t token_offset,
                                         std::size_t position_start,
                                         std::size_t tokens) {
  for (std::size_t token = 0; token < tokens; ++token) {
    auto position = qwen35_i32_position(position_start + token,
                                        "Qwen3.5 text MRoPE position");
    if (!position.is_ok()) {
      return position.status();
    }
    for (std::size_t section = 0; section < 4U; ++section) {
      positions[section * total_tokens + token_offset + token] = position.value();
    }
  }
  return Status::ok();
}

Status append_mixed_image_mrope_positions(std::vector<std::int32_t>& positions,
                                          std::size_t total_tokens,
                                          std::size_t token_offset,
                                          std::size_t position_start,
                                          const Qwen35ImageEmbeddingPlan& image_plan) {
  if (image_plan.merge_grid_x == 0 || image_plan.merge_grid_y == 0 ||
      image_plan.image_tokens !=
        static_cast<std::size_t>(image_plan.merge_grid_x) *
          image_plan.merge_grid_y) {
    return Status::invalid_argument(
      "Qwen3.5 image MRoPE position plan has invalid grid");
  }
  for (std::uint32_t row = 0; row < image_plan.merge_grid_y; ++row) {
    for (std::uint32_t col = 0; col < image_plan.merge_grid_x; ++col) {
      const auto token =
        static_cast<std::size_t>(row) * image_plan.merge_grid_x + col;
      auto t = qwen35_i32_position(position_start,
                                   "Qwen3.5 image MRoPE t position");
      auto y = qwen35_i32_position(position_start + row,
                                   "Qwen3.5 image MRoPE y position");
      auto x = qwen35_i32_position(position_start + col,
                                   "Qwen3.5 image MRoPE x position");
      if (!t.is_ok()) {
        return t.status();
      }
      if (!y.is_ok()) {
        return y.status();
      }
      if (!x.is_ok()) {
        return x.status();
      }
      positions[token_offset + token] = t.value();
      positions[total_tokens + token_offset + token] = y.value();
      positions[2U * total_tokens + token_offset + token] = x.value();
      positions[3U * total_tokens + token_offset + token] = 0;
    }
  }
  return Status::ok();
}

bool qwen35_role_is_supported(std::string_view role) {
  return role == "system" || role == "user" || role == "assistant";
}

void add_multimodal_text_chunk(Qwen35MultimodalPromptPlan& plan,
                               std::string_view text,
                               std::size_t message_index,
                               std::size_t part_index) {
  if (text.empty()) {
    return;
  }
  if (!plan.chunks.empty() &&
      plan.chunks.back().kind == Qwen35MultimodalPromptChunkKind::text) {
    plan.chunks.back().text += text;
    return;
  }
  Qwen35MultimodalPromptChunk chunk;
  chunk.kind = Qwen35MultimodalPromptChunkKind::text;
  chunk.text = std::string{text};
  chunk.message_index = message_index;
  chunk.part_index = part_index;
  plan.chunks.push_back(std::move(chunk));
}

Result<Qwen35ImageDimensions> image_part_dimensions(const ChatContentPart& part) {
  if (part.image_width != 0 && part.image_height != 0) {
    return Qwen35ImageDimensions{part.image_width, part.image_height};
  }
  if (!part.image_mime_type.empty() && !part.image_bytes.empty()) {
    auto dimensions =
      infer_qwen35_image_dimensions(part.image_mime_type, part.image_bytes);
    if (dimensions.is_ok()) {
      return dimensions;
    }
    Qwen35ImageDataUrl image;
    image.mime_type = part.image_mime_type;
    image.bytes = part.image_bytes;
    auto decoded = decode_qwen35_image_rgb(image);
    if (!decoded.is_ok()) {
      return decoded.status();
    }
    return Qwen35ImageDimensions{decoded.value().width, decoded.value().height};
  }
  return Status::invalid_argument(
    "Qwen3.5 multimodal prompt image part is missing dimensions");
}

std::vector<const ChatContentPart*> collect_qwen35_image_parts(
  const std::vector<ChatMessage>& messages) {
  std::vector<const ChatContentPart*> image_parts;
  for (const auto& message : messages) {
    for (const auto& part : message.content_parts) {
      if (part.kind == ChatContentPartKind::image_url) {
        image_parts.push_back(&part);
      }
    }
  }
  return image_parts;
}

int base64_value(char ch) {
  if (ch >= 'A' && ch <= 'Z') {
    return ch - 'A';
  }
  if (ch >= 'a' && ch <= 'z') {
    return ch - 'a' + 26;
  }
  if (ch >= '0' && ch <= '9') {
    return ch - '0' + 52;
  }
  if (ch == '+') {
    return 62;
  }
  if (ch == '/') {
    return 63;
  }
  return -1;
}

Result<std::vector<std::uint8_t>> decode_base64(std::string_view input) {
  std::vector<std::uint8_t> output;
  output.reserve(input.size() * 3U / 4U);

  std::uint32_t buffer = 0;
  int bits = 0;
  bool seen_padding = false;
  for (char ch : input) {
    if (ch == '\r' || ch == '\n' || ch == '\t' || ch == ' ') {
      continue;
    }
    if (ch == '=') {
      seen_padding = true;
      continue;
    }
    if (seen_padding) {
      return Status::invalid_argument("base64 data has non-padding characters after padding");
    }
    const int value = base64_value(ch);
    if (value < 0) {
      return Status::invalid_argument("base64 data contains invalid character");
    }
    buffer = (buffer << 6U) | static_cast<std::uint32_t>(value);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      output.push_back(static_cast<std::uint8_t>((buffer >> bits) & 0xFFU));
    }
  }
  return output;
}

}  // namespace

Result<Qwen35MmprojMetadata> load_qwen35_mmproj_metadata(
  const std::filesystem::path& path) {
  if (path.empty()) {
    return Status::invalid_argument("mmproj path must not be empty");
  }
  const auto gguf = read_gguf_file(path);
  if (!gguf.is_ok()) {
    return gguf.status();
  }
  const auto architecture = gguf_get_string(gguf.value(), "general.architecture");
  if (!architecture.is_ok()) {
    return architecture.status();
  }
  if (architecture.value() != "clip") {
    return Status::invalid_argument("mmproj GGUF architecture must be clip; got " +
                                    architecture.value());
  }

  Qwen35MmprojMetadata metadata;
  metadata.path = path;
  metadata.architecture = architecture.value();
  metadata.projector_type = optional_gguf_string(gguf.value(), "clip.projector_type");
  metadata.vision_projector_type =
    optional_gguf_string(gguf.value(), "clip.vision.projector_type");
  metadata.spatial_merge_size =
    optional_gguf_i64(gguf.value(), "clip.vision.spatial_merge_size");
  metadata.patch_size = optional_gguf_i64(gguf.value(), "clip.vision.patch_size");
  const auto image_min_pixels =
    optional_gguf_i64(gguf.value(), "clip.vision.image_min_pixels");
  const auto image_max_pixels =
    optional_gguf_i64(gguf.value(), "clip.vision.image_max_pixels");
  if (image_min_pixels > 0) {
    metadata.image_min_pixels = static_cast<std::uint64_t>(image_min_pixels);
  }
  if (image_max_pixels > 0) {
    metadata.image_max_pixels = static_cast<std::uint64_t>(image_max_pixels);
  }
  metadata.image_size = optional_gguf_i64(gguf.value(), "clip.vision.image_size");
  metadata.projection_dim =
    optional_gguf_i64(gguf.value(), "clip.vision.projection_dim");
  metadata.vision_feed_forward_length =
    optional_gguf_i64(gguf.value(), "clip.vision.feed_forward_length");
  metadata.vision_attention_head_count =
    optional_gguf_i64(gguf.value(), "clip.vision.attention.head_count");
  metadata.vision_attention_layer_norm_epsilon =
    optional_gguf_f64(gguf.value(), "clip.vision.attention.layer_norm_epsilon");
  metadata.deepstack_layer_flags =
    optional_gguf_bool_array(gguf.value(), "clip.vision.is_deepstack_layers");
  metadata.vision_block_count =
    optional_gguf_i64(gguf.value(), "clip.vision.block_count");
  metadata.vision_embedding_length =
    optional_gguf_i64(gguf.value(), "clip.vision.embedding_length");
  metadata.tensor_count = static_cast<std::size_t>(gguf.value().tensor_count);
  metadata.metadata_count = static_cast<std::size_t>(gguf.value().metadata_count);
  metadata.file_size = gguf.value().file_size;

  if (metadata.projector_type.empty() && metadata.vision_projector_type.empty()) {
    return Status::invalid_argument(
      "mmproj GGUF is missing clip.projector_type / clip.vision.projector_type");
  }
  if (qwen35_mmproj_is_qwen3vl_merger(metadata)) {
    auto image_mean = required_gguf_f32x3(gguf.value(), "clip.vision.image_mean");
    if (!image_mean.is_ok()) {
      return image_mean.status();
    }
    auto image_std = required_gguf_f32x3(gguf.value(), "clip.vision.image_std");
    if (!image_std.is_ok()) {
      return image_std.status();
    }
    for (const auto value : image_std.value()) {
      if (value <= 0.0F) {
        return Status::invalid_argument("clip.vision.image_std values must be positive");
      }
    }
    metadata.image_mean = image_mean.value();
    metadata.image_std = image_std.value();
    metadata.image_mean_std_present = true;
    if (metadata.image_min_pixels == 0) {
      auto min_pixels = qwen35_image_pixel_limit(
        metadata, kQwen35VlDefaultMinImageTokens);
      if (!min_pixels.is_ok()) {
        return min_pixels.status();
      }
      metadata.image_min_pixels = min_pixels.value();
    }
    if (metadata.image_max_pixels == 0) {
      auto max_pixels = qwen35_image_pixel_limit(
        metadata, kQwen35VlDefaultMaxImageTokens);
      if (!max_pixels.is_ok()) {
        return max_pixels.status();
      }
      metadata.image_max_pixels = max_pixels.value();
    }
    metadata.missing_required_tensors = qwen3vl_missing_required_tensors(
      gguf.value(), metadata.deepstack_layer_flags, metadata.deepstack_layer_count);
    metadata.qwen3vl_required_tensors_present =
      metadata.missing_required_tensors.empty();
    metadata.projector_output_width =
      qwen3vl_projector_output_width(gguf.value(), metadata.deepstack_layer_count);
    if (!metadata.qwen3vl_required_tensors_present) {
      return Status::invalid_argument(
        "qwen3vl_merger mmproj is missing required tensors: " +
        join_names(metadata.missing_required_tensors));
    }
  }
  return metadata;
}

bool qwen35_mmproj_is_qwen3vl_merger(const Qwen35MmprojMetadata& metadata) {
  return metadata.projector_type == "qwen3vl_merger" ||
         metadata.vision_projector_type == "qwen3vl_merger";
}

Status validate_qwen35_mmproj_text_embedding_compatibility(
  const Qwen35MmprojMetadata& metadata, std::int64_t text_embedding_length) {
  if (!qwen35_mmproj_is_qwen3vl_merger(metadata)) {
    return Status::ok();
  }
  if (text_embedding_length <= 0) {
    return Status::invalid_argument(
      "Qwen3.5 text GGUF embedding_length must be positive when --mmproj is used");
  }
  if (metadata.projector_output_width == 0) {
    return Status::invalid_argument(
      "qwen3vl_merger mmproj projector_output_width could not be inferred");
  }
  if (metadata.projector_output_width !=
      static_cast<std::uint64_t>(text_embedding_length)) {
    std::ostringstream message;
    message << "qwen3vl_merger mmproj output width "
            << metadata.projector_output_width
            << " does not match Qwen3.5 text embedding_length "
            << text_embedding_length;
    return Status::invalid_argument(message.str());
  }
  return Status::ok();
}

std::string format_qwen35_mmproj_metadata_summary(
  const Qwen35MmprojMetadata& metadata) {
  std::ostringstream output;
  output << "mmproj path: " << metadata.path.string() << '\n';
  output << "mmproj architecture: " << metadata.architecture << '\n';
  output << "mmproj projector_type: " << metadata.projector_type << '\n';
  output << "mmproj vision_projector_type: " << metadata.vision_projector_type << '\n';
  output << "mmproj spatial_merge_size: " << metadata.spatial_merge_size << '\n';
  output << "mmproj patch_size: " << metadata.patch_size << '\n';
  output << "mmproj image_min_pixels: " << metadata.image_min_pixels << '\n';
  output << "mmproj image_max_pixels: " << metadata.image_max_pixels << '\n';
  output << "mmproj image_mean: [" << metadata.image_mean[0] << ", "
         << metadata.image_mean[1] << ", " << metadata.image_mean[2] << "]\n";
  output << "mmproj image_std: [" << metadata.image_std[0] << ", "
         << metadata.image_std[1] << ", " << metadata.image_std[2] << "]\n";
  output << "mmproj image_size: " << metadata.image_size << '\n';
  output << "mmproj projection_dim: " << metadata.projection_dim << '\n';
  output << "mmproj vision_feed_forward_length: "
         << metadata.vision_feed_forward_length << '\n';
  output << "mmproj vision_attention_head_count: "
         << metadata.vision_attention_head_count << '\n';
  output << "mmproj vision_attention_layer_norm_epsilon: "
         << metadata.vision_attention_layer_norm_epsilon << '\n';
  output << "mmproj vision_block_count: " << metadata.vision_block_count << '\n';
  output << "mmproj vision_embedding_length: " << metadata.vision_embedding_length << '\n';
  output << "mmproj qwen3vl_required_tensors_present: "
         << (metadata.qwen3vl_required_tensors_present ? "true" : "false") << '\n';
  output << "mmproj deepstack_layer_count: " << metadata.deepstack_layer_count << '\n';
  output << "mmproj projector_output_width: " << metadata.projector_output_width << '\n';
  if (!metadata.missing_required_tensors.empty()) {
    output << "mmproj missing_required_tensors: "
           << join_names(metadata.missing_required_tensors) << '\n';
  }
  return output.str();
}

Result<Qwen35VisionGraphPlan> plan_qwen35_vision_graph(
  const std::filesystem::path& mmproj_path) {
  auto metadata = load_qwen35_mmproj_metadata(mmproj_path);
  if (!metadata.is_ok()) {
    return metadata.status();
  }
  if (!qwen35_mmproj_is_qwen3vl_merger(metadata.value())) {
    return Status::invalid_argument(
      "Qwen3.5 native vision graph requires qwen3vl_merger mmproj");
  }
  auto gguf = read_gguf_file(mmproj_path);
  if (!gguf.is_ok()) {
    return gguf.status();
  }

  auto image_size = required_positive_u64_from_i64(
    metadata.value().image_size, "clip.vision.image_size");
  if (!image_size.is_ok()) {
    return image_size.status();
  }
  auto patch_size = required_positive_u64_from_i64(
    metadata.value().patch_size, "clip.vision.patch_size");
  if (!patch_size.is_ok()) {
    return patch_size.status();
  }
  auto spatial_merge_size = required_positive_u64_from_i64(
    metadata.value().spatial_merge_size, "clip.vision.spatial_merge_size");
  if (!spatial_merge_size.is_ok()) {
    return spatial_merge_size.status();
  }
  if (spatial_merge_size.value() != 2U) {
    return Status::invalid_argument(
      "Qwen3.5 native vision graph currently supports spatial_merge_size=2");
  }
  auto vision_embedding_length = required_positive_u64_from_i64(
    metadata.value().vision_embedding_length, "clip.vision.embedding_length");
  if (!vision_embedding_length.is_ok()) {
    return vision_embedding_length.status();
  }
  auto vision_feed_forward_length = required_positive_u64_from_i64(
    metadata.value().vision_feed_forward_length,
    "clip.vision.feed_forward_length");
  if (!vision_feed_forward_length.is_ok()) {
    return vision_feed_forward_length.status();
  }
  auto projection_dim = required_positive_u64_from_i64(
    metadata.value().projection_dim, "clip.vision.projection_dim");
  if (!projection_dim.is_ok()) {
    return projection_dim.status();
  }
  auto attention_head_count = required_positive_u64_from_i64(
    metadata.value().vision_attention_head_count,
    "clip.vision.attention.head_count");
  if (!attention_head_count.is_ok()) {
    return attention_head_count.status();
  }
  auto block_count_u64 = required_positive_u64_from_i64(
    metadata.value().vision_block_count, "clip.vision.block_count");
  if (!block_count_u64.is_ok()) {
    return block_count_u64.status();
  }
  auto block_count_size =
    checked_size_from_u64(block_count_u64.value(), "clip.vision.block_count");
  if (!block_count_size.is_ok()) {
    return block_count_size.status();
  }
  if (image_size.value() % patch_size.value() != 0) {
    return Status::invalid_argument(
      "clip.vision.image_size must be divisible by clip.vision.patch_size");
  }
  if (vision_embedding_length.value() % attention_head_count.value() != 0) {
    return Status::invalid_argument(
      "clip.vision.embedding_length must be divisible by attention.head_count");
  }
  const auto head_dim =
    vision_embedding_length.value() / attention_head_count.value();
  if (head_dim % 4U != 0) {
    return Status::invalid_argument(
      "Qwen3.5 native vision graph requires vision head_dim divisible by 4");
  }

  const auto image_grid = image_size.value() / patch_size.value();
  auto position_count = checked_u64_mul(image_grid, image_grid,
                                        "Qwen3.5 vision position count");
  if (!position_count.is_ok()) {
    return position_count.status();
  }
  auto merge_factor = checked_u64_mul(spatial_merge_size.value(),
                                      spatial_merge_size.value(),
                                      "Qwen3.5 vision merge factor");
  if (!merge_factor.is_ok()) {
    return merge_factor.status();
  }
  auto merged_embedding = checked_u64_mul(
    vision_embedding_length.value(), merge_factor.value(),
    "Qwen3.5 vision merged embedding width");
  if (!merged_embedding.is_ok()) {
    return merged_embedding.status();
  }

  Qwen35VisionGraphPlan plan;
  plan.path = mmproj_path;
  plan.image_size = image_size.value();
  plan.patch_size = patch_size.value();
  plan.spatial_merge_size = spatial_merge_size.value();
  plan.vision_embedding_length = vision_embedding_length.value();
  plan.vision_feed_forward_length = vision_feed_forward_length.value();
  plan.projection_dim = projection_dim.value();
  plan.vision_attention_head_count = attention_head_count.value();
  plan.vision_attention_layer_norm_epsilon =
    metadata.value().vision_attention_layer_norm_epsilon;
  plan.block_count = block_count_size.value();
  plan.projector_output_width = metadata.value().projector_output_width;

  auto append_input = [&](std::string_view name,
                          std::vector<std::uint64_t> shape) -> Status {
    return append_required_tensor(gguf.value(), name, shape, plan.input_tensors);
  };
  auto status = append_input(
    "v.patch_embd.weight",
    {patch_size.value(), patch_size.value(), 3U, vision_embedding_length.value()});
  if (!status.is_ok()) {
    return status;
  }
  status = append_input(
    "v.patch_embd.weight.1",
    {patch_size.value(), patch_size.value(), 3U, vision_embedding_length.value()});
  if (!status.is_ok()) {
    return status;
  }
  status = append_input("v.patch_embd.bias", {vision_embedding_length.value()});
  if (!status.is_ok()) {
    return status;
  }
  status = append_input("v.position_embd.weight",
                        {vision_embedding_length.value(), position_count.value()});
  if (!status.is_ok()) {
    return status;
  }

  status = append_required_tensor(
    gguf.value(), "v.post_ln.weight", {vision_embedding_length.value()},
    plan.output_norm_tensors);
  if (!status.is_ok()) {
    return status;
  }
  status = append_required_tensor(
    gguf.value(), "v.post_ln.bias", {vision_embedding_length.value()},
    plan.output_norm_tensors);
  if (!status.is_ok()) {
    return status;
  }

  status = append_required_tensor(
    gguf.value(), "mm.0.weight",
    {merged_embedding.value(), vision_feed_forward_length.value()},
    plan.projector_tensors);
  if (!status.is_ok()) {
    return status;
  }
  status = append_required_tensor(
    gguf.value(), "mm.0.bias", {vision_feed_forward_length.value()},
    plan.projector_tensors);
  if (!status.is_ok()) {
    return status;
  }
  status = append_required_tensor(
    gguf.value(), "mm.2.weight",
    {vision_feed_forward_length.value(), projection_dim.value()},
    plan.projector_tensors);
  if (!status.is_ok()) {
    return status;
  }
  status = append_required_tensor(
    gguf.value(), "mm.2.bias", {projection_dim.value()}, plan.projector_tensors);
  if (!status.is_ok()) {
    return status;
  }

  plan.deepstack_layer_indices =
    qwen3vl_deepstack_indices_from_flags(metadata.value().deepstack_layer_flags);
  auto tensor_deepstack_indices = qwen3vl_deepstack_layer_indices(gguf.value());
  plan.deepstack_layer_indices.insert(
    plan.deepstack_layer_indices.end(), tensor_deepstack_indices.begin(),
    tensor_deepstack_indices.end());
  std::sort(plan.deepstack_layer_indices.begin(),
            plan.deepstack_layer_indices.end());
  plan.deepstack_layer_indices.erase(
    std::unique(plan.deepstack_layer_indices.begin(),
                plan.deepstack_layer_indices.end()),
    plan.deepstack_layer_indices.end());
  for (const auto layer_index : plan.deepstack_layer_indices) {
    if (layer_index >= plan.block_count) {
      return Status::invalid_argument(
        "clip.vision.is_deepstack_layers contains out-of-range layer index");
    }
  }
  plan.deepstack_layer_count = plan.deepstack_layer_indices.size();
  if (plan.projector_output_width !=
      projection_dim.value() * static_cast<std::uint64_t>(plan.deepstack_layer_count + 1U)) {
    return Status::invalid_argument(
      "qwen3vl_merger projector_output_width does not match projection_dim and deepstack layers");
  }

  std::vector<bool> block_has_deepstack(plan.block_count, false);
  for (const auto layer_index : plan.deepstack_layer_indices) {
    block_has_deepstack[layer_index] = true;
  }
  plan.blocks.reserve(plan.block_count);
  for (std::size_t layer_index = 0; layer_index < plan.block_count; ++layer_index) {
    Qwen35VisionBlockPlan block;
    block.layer_index = layer_index;
    block.has_deepstack = block_has_deepstack[layer_index];
    auto append_block = [&](std::string_view stem, std::string_view suffix,
                            std::vector<std::uint64_t> shape) -> Status {
      return append_required_tensor(
        gguf.value(), vision_block_tensor_name(layer_index, stem, suffix),
        shape, block.tensors);
    };
    status = append_block("ln1", "weight", {vision_embedding_length.value()});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("ln1", "bias", {vision_embedding_length.value()});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("attn_qkv", "weight",
                          {vision_embedding_length.value(),
                           vision_embedding_length.value() * 3U});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("attn_qkv", "bias",
                          {vision_embedding_length.value() * 3U});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("attn_out", "weight",
                          {vision_embedding_length.value(),
                           vision_embedding_length.value()});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("attn_out", "bias", {vision_embedding_length.value()});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("ln2", "weight", {vision_embedding_length.value()});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("ln2", "bias", {vision_embedding_length.value()});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("ffn_up", "weight",
                          {vision_embedding_length.value(),
                           vision_feed_forward_length.value()});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("ffn_up", "bias",
                          {vision_feed_forward_length.value()});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("ffn_down", "weight",
                          {vision_feed_forward_length.value(),
                           vision_embedding_length.value()});
    if (!status.is_ok()) {
      return status;
    }
    status = append_block("ffn_down", "bias", {vision_embedding_length.value()});
    if (!status.is_ok()) {
      return status;
    }

    if (block.has_deepstack) {
      auto append_deepstack =
        [&](std::string_view suffix,
            std::vector<std::uint64_t> shape) -> Status {
        return append_required_tensor(
          gguf.value(), qwen3vl_deepstack_tensor_name(layer_index, suffix),
          shape, block.deepstack_tensors);
      };
      status = append_deepstack("norm.weight", {merged_embedding.value()});
      if (!status.is_ok()) {
        return status;
      }
      status = append_deepstack("norm.bias", {merged_embedding.value()});
      if (!status.is_ok()) {
        return status;
      }
      status = append_deepstack(
        "fc1.weight",
        {merged_embedding.value(), vision_feed_forward_length.value()});
      if (!status.is_ok()) {
        return status;
      }
      status = append_deepstack("fc1.bias",
                                {vision_feed_forward_length.value()});
      if (!status.is_ok()) {
        return status;
      }
      status = append_deepstack(
        "fc2.weight",
        {vision_feed_forward_length.value(), projection_dim.value()});
      if (!status.is_ok()) {
        return status;
      }
      status = append_deepstack("fc2.bias", {projection_dim.value()});
      if (!status.is_ok()) {
        return status;
      }
    }
    plan.required_tensor_count += block.tensors.size() + block.deepstack_tensors.size();
    plan.blocks.push_back(std::move(block));
  }
  plan.required_tensor_count += plan.input_tensors.size() +
                                plan.output_norm_tensors.size() +
                                plan.projector_tensors.size();
  return plan;
}

std::string format_qwen35_vision_graph_plan(const Qwen35VisionGraphPlan& plan) {
  std::ostringstream output;
  output << "vision graph path: " << plan.path.string() << '\n';
  output << "vision graph image_size: " << plan.image_size << '\n';
  output << "vision graph patch_size: " << plan.patch_size << '\n';
  output << "vision graph spatial_merge_size: " << plan.spatial_merge_size << '\n';
  output << "vision graph embedding_length: "
         << plan.vision_embedding_length << '\n';
  output << "vision graph feed_forward_length: "
         << plan.vision_feed_forward_length << '\n';
  output << "vision graph projection_dim: " << plan.projection_dim << '\n';
  output << "vision graph attention_head_count: "
         << plan.vision_attention_head_count << '\n';
  output << "vision graph attention_layer_norm_epsilon: "
         << plan.vision_attention_layer_norm_epsilon << '\n';
  output << "vision graph block_count: " << plan.block_count << '\n';
  output << "vision graph deepstack_layer_count: "
         << plan.deepstack_layer_count << '\n';
  output << "vision graph projector_output_width: "
         << plan.projector_output_width << '\n';
  output << "vision graph required_tensors: "
         << plan.required_tensor_count << '\n';
  output << "vision graph deepstack_layers:";
  if (plan.deepstack_layer_indices.empty()) {
    output << " none";
  } else {
    for (const auto layer_index : plan.deepstack_layer_indices) {
      output << ' ' << layer_index;
    }
  }
  output << '\n';
  return output.str();
}

Result<Qwen35VisionInputStageResult> run_qwen35_vision_input_stage_cpu(
  const std::filesystem::path& mmproj_path,
  const Qwen35ImagePreprocessResult& image) {
  auto graph_plan = plan_qwen35_vision_graph(mmproj_path);
  if (!graph_plan.is_ok()) {
    return graph_plan.status();
  }
  if (image.width == 0 || image.height == 0 || image.channels != 3U) {
    return Status::invalid_argument(
      "Qwen3.5 vision input stage requires a non-empty 3-channel image");
  }
  if (image.width % graph_plan.value().patch_size != 0 ||
      image.height % graph_plan.value().patch_size != 0) {
    return Status::invalid_argument(
      "Qwen3.5 preprocessed image is not divisible by vision patch_size");
  }
  const auto patch_grid_x =
    static_cast<std::uint32_t>(image.width / graph_plan.value().patch_size);
  const auto patch_grid_y =
    static_cast<std::uint32_t>(image.height / graph_plan.value().patch_size);
  if (patch_grid_x % graph_plan.value().spatial_merge_size != 0 ||
      patch_grid_y % graph_plan.value().spatial_merge_size != 0) {
    return Status::invalid_argument(
      "Qwen3.5 patch grid is not divisible by spatial_merge_size");
  }
  auto expected_pixels = checked_image_elements(
    image.width, image.height, 3U, "Qwen3.5 preprocessed image elements");
  if (!expected_pixels.is_ok()) {
    return expected_pixels.status();
  }
  if (image.pixels.size() != expected_pixels.value()) {
    return Status::invalid_argument(
      "Qwen3.5 preprocessed image buffer size does not match dimensions");
  }

  auto gguf = read_gguf_file(mmproj_path);
  if (!gguf.is_ok()) {
    return gguf.status();
  }
  auto mapped = GgufMappedData::open(gguf.value());
  if (!mapped.is_ok()) {
    return mapped.status();
  }
  const auto patch_size = static_cast<std::uint32_t>(graph_plan.value().patch_size);
  const auto emb = graph_plan.value().vision_embedding_length;
  const std::vector<std::uint64_t> patch_shape{
    graph_plan.value().patch_size, graph_plan.value().patch_size, 3U, emb};
  auto patch_0 = load_tensor_f32(gguf.value(), mapped.value(),
                                 "v.patch_embd.weight", patch_shape);
  if (!patch_0.is_ok()) {
    return patch_0.status();
  }
  auto patch_1 = load_tensor_f32(gguf.value(), mapped.value(),
                                 "v.patch_embd.weight.1", patch_shape);
  if (!patch_1.is_ok()) {
    return patch_1.status();
  }
  auto patch_bias = load_tensor_f32(gguf.value(), mapped.value(),
                                    "v.patch_embd.bias", {emb});
  if (!patch_bias.is_ok()) {
    return patch_bias.status();
  }
  const auto base_grid = graph_plan.value().image_size / graph_plan.value().patch_size;
  auto base_positions = checked_u64_mul(base_grid, base_grid,
                                        "Qwen3.5 position embedding grid");
  if (!base_positions.is_ok()) {
    return base_positions.status();
  }
  auto position_embedding = load_tensor_f32(
    gguf.value(), mapped.value(), "v.position_embd.weight",
    {emb, base_positions.value()});
  if (!position_embedding.is_ok()) {
    return position_embedding.status();
  }

  auto conv = qwen3vl_patch_embedding_temporal_sum(
    image, patch_0.value(), patch_1.value(), patch_size, patch_grid_x,
    patch_grid_y, emb);
  if (!conv.is_ok()) {
    return conv.status();
  }
  auto merged = qwen3vl_spatial_merge_ewh(
    conv.value(), patch_grid_x, patch_grid_y, emb);
  const auto emb_size = static_cast<std::size_t>(emb);
  for (std::size_t token = 0; token < merged.size() / emb_size; ++token) {
    const auto offset = token * emb_size;
    for (std::size_t channel = 0; channel < emb_size; ++channel) {
      merged[offset + channel] += patch_bias.value()[channel];
    }
  }

  const auto resized_positions = resize_position_embeddings_bilinear(
    position_embedding.value(), static_cast<std::uint32_t>(base_grid),
    patch_grid_x, patch_grid_y, emb);
  auto merged_positions = qwen3vl_spatial_merge_ewh(
    resized_positions, patch_grid_x, patch_grid_y, emb);
  if (merged_positions.size() != merged.size()) {
    return Status::internal_error(
      "Qwen3.5 merged position embedding size mismatch");
  }
  for (std::size_t index = 0; index < merged.size(); ++index) {
    merged[index] += merged_positions[index];
  }

  Qwen35VisionInputStageResult result;
  result.image_plan = image.plan;
  result.patch_grid_x = patch_grid_x;
  result.patch_grid_y = patch_grid_y;
  result.vision_embedding_length = emb;
  result.token_count = merged.size() / emb_size;
  result.embeddings = std::move(merged);
  return result;
}

std::string format_qwen35_vision_input_stage_result(
  const Qwen35VisionInputStageResult& result) {
  std::ostringstream output;
  output << "vision input patch_grid: " << result.patch_grid_x << 'x'
         << result.patch_grid_y << '\n';
  output << "vision input embedding_length: "
         << result.vision_embedding_length << '\n';
  output << "vision input tokens: " << result.token_count << '\n';
  output << "vision input values: " << result.embeddings.size() << '\n';
  return output.str();
}

Result<Qwen35VisionEncoderResult> run_qwen35_vision_encoder_cpu(
  const std::filesystem::path& mmproj_path,
  const Qwen35ImagePreprocessResult& image) {
  auto graph_plan = plan_qwen35_vision_graph(mmproj_path);
  if (!graph_plan.is_ok()) {
    return graph_plan.status();
  }
  auto input_stage = run_qwen35_vision_input_stage_cpu(mmproj_path, image);
  if (!input_stage.is_ok()) {
    return input_stage.status();
  }
  if (input_stage.value().token_count !=
      static_cast<std::size_t>(input_stage.value().patch_grid_x) *
        input_stage.value().patch_grid_y) {
    return Status::internal_error(
      "Qwen3.5 vision input stage token count does not match patch grid");
  }
  if (input_stage.value().token_count % 4U != 0) {
    return Status::invalid_argument(
      "Qwen3.5 vision encoder requires token_count divisible by 4");
  }

  auto emb_size = checked_size_from_u64(
    graph_plan.value().vision_embedding_length,
    "Qwen3.5 vision embedding length");
  if (!emb_size.is_ok()) {
    return emb_size.status();
  }
  auto ffn_size = checked_size_from_u64(
    graph_plan.value().vision_feed_forward_length,
    "Qwen3.5 vision feed-forward length");
  if (!ffn_size.is_ok()) {
    return ffn_size.status();
  }
  auto projection_size = checked_size_from_u64(
    graph_plan.value().projection_dim, "Qwen3.5 projection dimension");
  if (!projection_size.is_ok()) {
    return projection_size.status();
  }
  auto head_count = checked_size_from_u64(
    graph_plan.value().vision_attention_head_count,
    "Qwen3.5 vision attention head count");
  if (!head_count.is_ok()) {
    return head_count.status();
  }
  auto projector_output_width = checked_size_from_u64(
    graph_plan.value().projector_output_width,
    "Qwen3.5 projector output width");
  if (!projector_output_width.is_ok()) {
    return projector_output_width.status();
  }
  if (input_stage.value().vision_embedding_length !=
      graph_plan.value().vision_embedding_length) {
    return Status::internal_error(
      "Qwen3.5 vision input stage embedding length mismatch");
  }
  const auto merged_width = emb_size.value() * 4U;
  const auto image_token_count = input_stage.value().token_count / 4U;
  const auto epsilon =
    graph_plan.value().vision_attention_layer_norm_epsilon > 0.0
      ? graph_plan.value().vision_attention_layer_norm_epsilon
      : 1e-6;

  auto gguf = read_gguf_file(mmproj_path);
  if (!gguf.is_ok()) {
    return gguf.status();
  }
  auto mapped = GgufMappedData::open(gguf.value());
  if (!mapped.is_ok()) {
    return mapped.status();
  }

  std::vector<float> hidden = std::move(input_stage.value().embeddings);
  std::vector<std::vector<float>> deepstack_outputs;
  deepstack_outputs.reserve(graph_plan.value().deepstack_layer_count);

  for (const auto& block : graph_plan.value().blocks) {
    const auto layer = block.layer_index;
    const auto layer_prefix =
      std::string{"Qwen3.5 vision block "} + std::to_string(layer);
    auto ln1_weight = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "ln1", "weight"),
      {graph_plan.value().vision_embedding_length});
    if (!ln1_weight.is_ok()) {
      return ln1_weight.status();
    }
    auto ln1_bias = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "ln1", "bias"),
      {graph_plan.value().vision_embedding_length});
    if (!ln1_bias.is_ok()) {
      return ln1_bias.status();
    }
    auto qkv_weight = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "attn_qkv", "weight"),
      {graph_plan.value().vision_embedding_length,
       graph_plan.value().vision_embedding_length * 3U});
    if (!qkv_weight.is_ok()) {
      return qkv_weight.status();
    }
    auto qkv_bias = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "attn_qkv", "bias"),
      {graph_plan.value().vision_embedding_length * 3U});
    if (!qkv_bias.is_ok()) {
      return qkv_bias.status();
    }
    auto attn_out_weight = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "attn_out", "weight"),
      {graph_plan.value().vision_embedding_length,
       graph_plan.value().vision_embedding_length});
    if (!attn_out_weight.is_ok()) {
      return attn_out_weight.status();
    }
    auto attn_out_bias = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "attn_out", "bias"),
      {graph_plan.value().vision_embedding_length});
    if (!attn_out_bias.is_ok()) {
      return attn_out_bias.status();
    }
    auto ln2_weight = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "ln2", "weight"),
      {graph_plan.value().vision_embedding_length});
    if (!ln2_weight.is_ok()) {
      return ln2_weight.status();
    }
    auto ln2_bias = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "ln2", "bias"),
      {graph_plan.value().vision_embedding_length});
    if (!ln2_bias.is_ok()) {
      return ln2_bias.status();
    }
    auto ffn_up_weight = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "ffn_up", "weight"),
      {graph_plan.value().vision_embedding_length,
       graph_plan.value().vision_feed_forward_length});
    if (!ffn_up_weight.is_ok()) {
      return ffn_up_weight.status();
    }
    auto ffn_up_bias = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "ffn_up", "bias"),
      {graph_plan.value().vision_feed_forward_length});
    if (!ffn_up_bias.is_ok()) {
      return ffn_up_bias.status();
    }
    auto ffn_down_weight = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "ffn_down", "weight"),
      {graph_plan.value().vision_feed_forward_length,
       graph_plan.value().vision_embedding_length});
    if (!ffn_down_weight.is_ok()) {
      return ffn_down_weight.status();
    }
    auto ffn_down_bias = load_tensor_f32(
      gguf.value(), mapped.value(),
      vision_block_tensor_name(layer, "ffn_down", "bias"),
      {graph_plan.value().vision_embedding_length});
    if (!ffn_down_bias.is_ok()) {
      return ffn_down_bias.status();
    }

    auto normed = qwen35_layer_norm_token_major(
      hidden, input_stage.value().token_count, emb_size.value(),
      ln1_weight.value(), ln1_bias.value(), epsilon, layer_prefix);
    if (!normed.is_ok()) {
      return normed.status();
    }
    auto qkv = qwen35_linear_token_major(
      normed.value(), input_stage.value().token_count, emb_size.value(),
      emb_size.value() * 3U, qkv_weight.value(), qkv_bias.value(),
      layer_prefix);
    if (!qkv.is_ok()) {
      return qkv.status();
    }
    auto attention = qwen3vl_attention_token_major(
      qkv.value(), input_stage.value().token_count, emb_size.value(),
      head_count.value(), input_stage.value().patch_grid_x,
      input_stage.value().patch_grid_y, layer_prefix);
    if (!attention.is_ok()) {
      return attention.status();
    }
    auto attention_output = qwen35_linear_token_major(
      attention.value(), input_stage.value().token_count, emb_size.value(),
      emb_size.value(), attn_out_weight.value(), attn_out_bias.value(),
      layer_prefix);
    if (!attention_output.is_ok()) {
      return attention_output.status();
    }
    auto status = qwen35_add_in_place(hidden, attention_output.value(),
                                      layer_prefix);
    if (!status.is_ok()) {
      return status;
    }

    auto ffn_normed = qwen35_layer_norm_token_major(
      hidden, input_stage.value().token_count, emb_size.value(),
      ln2_weight.value(), ln2_bias.value(), epsilon, layer_prefix);
    if (!ffn_normed.is_ok()) {
      return ffn_normed.status();
    }
    auto ffn_hidden = qwen35_linear_token_major(
      ffn_normed.value(), input_stage.value().token_count, emb_size.value(),
      ffn_size.value(), ffn_up_weight.value(), ffn_up_bias.value(),
      layer_prefix);
    if (!ffn_hidden.is_ok()) {
      return ffn_hidden.status();
    }
    qwen35_gelu_in_place(ffn_hidden.value());
    auto ffn_output = qwen35_linear_token_major(
      ffn_hidden.value(), input_stage.value().token_count, ffn_size.value(),
      emb_size.value(), ffn_down_weight.value(), ffn_down_bias.value(),
      layer_prefix);
    if (!ffn_output.is_ok()) {
      return ffn_output.status();
    }
    status = qwen35_add_in_place(hidden, ffn_output.value(), layer_prefix);
    if (!status.is_ok()) {
      return status;
    }

    if (block.has_deepstack) {
      auto deepstack_input = qwen3vl_group_four_tokens(
        hidden, input_stage.value().token_count, emb_size.value(),
        layer_prefix);
      if (!deepstack_input.is_ok()) {
        return deepstack_input.status();
      }
      auto norm_weight = load_tensor_f32(
        gguf.value(), mapped.value(),
        qwen3vl_deepstack_tensor_name(layer, "norm.weight"),
        {graph_plan.value().vision_embedding_length * 4U});
      if (!norm_weight.is_ok()) {
        return norm_weight.status();
      }
      auto norm_bias = load_tensor_f32(
        gguf.value(), mapped.value(),
        qwen3vl_deepstack_tensor_name(layer, "norm.bias"),
        {graph_plan.value().vision_embedding_length * 4U});
      if (!norm_bias.is_ok()) {
        return norm_bias.status();
      }
      auto fc1_weight = load_tensor_f32(
        gguf.value(), mapped.value(),
        qwen3vl_deepstack_tensor_name(layer, "fc1.weight"),
        {graph_plan.value().vision_embedding_length * 4U,
         graph_plan.value().vision_feed_forward_length});
      if (!fc1_weight.is_ok()) {
        return fc1_weight.status();
      }
      auto fc1_bias = load_tensor_f32(
        gguf.value(), mapped.value(),
        qwen3vl_deepstack_tensor_name(layer, "fc1.bias"),
        {graph_plan.value().vision_feed_forward_length});
      if (!fc1_bias.is_ok()) {
        return fc1_bias.status();
      }
      auto fc2_weight = load_tensor_f32(
        gguf.value(), mapped.value(),
        qwen3vl_deepstack_tensor_name(layer, "fc2.weight"),
        {graph_plan.value().vision_feed_forward_length,
         graph_plan.value().projection_dim});
      if (!fc2_weight.is_ok()) {
        return fc2_weight.status();
      }
      auto fc2_bias = load_tensor_f32(
        gguf.value(), mapped.value(),
        qwen3vl_deepstack_tensor_name(layer, "fc2.bias"),
        {graph_plan.value().projection_dim});
      if (!fc2_bias.is_ok()) {
        return fc2_bias.status();
      }

      auto deepstack_normed = qwen35_layer_norm_token_major(
        deepstack_input.value(), image_token_count, merged_width,
        norm_weight.value(), norm_bias.value(), epsilon, layer_prefix);
      if (!deepstack_normed.is_ok()) {
        return deepstack_normed.status();
      }
      auto deepstack = qwen3vl_projector_mlp_token_major(
        deepstack_normed.value(), image_token_count, merged_width,
        ffn_size.value(), projection_size.value(), fc1_weight.value(),
        fc1_bias.value(), fc2_weight.value(), fc2_bias.value(), layer_prefix);
      if (!deepstack.is_ok()) {
        return deepstack.status();
      }
      deepstack_outputs.push_back(std::move(deepstack.value()));
    }
  }

  auto post_ln_weight = load_tensor_f32(
    gguf.value(), mapped.value(), "v.post_ln.weight",
    {graph_plan.value().vision_embedding_length});
  if (!post_ln_weight.is_ok()) {
    return post_ln_weight.status();
  }
  auto post_ln_bias = load_tensor_f32(
    gguf.value(), mapped.value(), "v.post_ln.bias",
    {graph_plan.value().vision_embedding_length});
  if (!post_ln_bias.is_ok()) {
    return post_ln_bias.status();
  }
  auto post_normed = qwen35_layer_norm_token_major(
    hidden, input_stage.value().token_count, emb_size.value(),
    post_ln_weight.value(), post_ln_bias.value(), epsilon,
    "Qwen3.5 vision post norm");
  if (!post_normed.is_ok()) {
    return post_normed.status();
  }
  auto projector_input = qwen3vl_group_four_tokens(
    post_normed.value(), input_stage.value().token_count, emb_size.value(),
    "Qwen3.5 vision projector input");
  if (!projector_input.is_ok()) {
    return projector_input.status();
  }
  auto mm0_weight = load_tensor_f32(
    gguf.value(), mapped.value(), "mm.0.weight",
    {graph_plan.value().vision_embedding_length * 4U,
     graph_plan.value().vision_feed_forward_length});
  if (!mm0_weight.is_ok()) {
    return mm0_weight.status();
  }
  auto mm0_bias = load_tensor_f32(
    gguf.value(), mapped.value(), "mm.0.bias",
    {graph_plan.value().vision_feed_forward_length});
  if (!mm0_bias.is_ok()) {
    return mm0_bias.status();
  }
  auto mm2_weight = load_tensor_f32(
    gguf.value(), mapped.value(), "mm.2.weight",
    {graph_plan.value().vision_feed_forward_length,
     graph_plan.value().projection_dim});
  if (!mm2_weight.is_ok()) {
    return mm2_weight.status();
  }
  auto mm2_bias = load_tensor_f32(
    gguf.value(), mapped.value(), "mm.2.bias",
    {graph_plan.value().projection_dim});
  if (!mm2_bias.is_ok()) {
    return mm2_bias.status();
  }
  auto main_projected = qwen3vl_projector_mlp_token_major(
    projector_input.value(), image_token_count, merged_width, ffn_size.value(),
    projection_size.value(), mm0_weight.value(), mm0_bias.value(),
    mm2_weight.value(), mm2_bias.value(), "Qwen3.5 vision projector");
  if (!main_projected.is_ok()) {
    return main_projected.status();
  }

  const auto expected_output_width =
    projection_size.value() * (deepstack_outputs.size() + 1U);
  if (expected_output_width != projector_output_width.value()) {
    return Status::invalid_argument(
      "Qwen3.5 vision projector output width does not match deepstack outputs");
  }
  std::vector<float> embeddings(image_token_count * projector_output_width.value());
  for (std::size_t token = 0; token < image_token_count; ++token) {
    const auto output_offset = token * projector_output_width.value();
    const auto main_offset = token * projection_size.value();
    std::copy(main_projected.value().begin() +
                static_cast<std::ptrdiff_t>(main_offset),
              main_projected.value().begin() +
                static_cast<std::ptrdiff_t>(main_offset + projection_size.value()),
              embeddings.begin() + static_cast<std::ptrdiff_t>(output_offset));
    for (std::size_t stack = 0; stack < deepstack_outputs.size(); ++stack) {
      const auto feature_offset = output_offset +
                                  (stack + 1U) * projection_size.value();
      const auto stack_offset = token * projection_size.value();
      if (deepstack_outputs[stack].size() !=
          image_token_count * projection_size.value()) {
        return Status::internal_error(
          "Qwen3.5 vision deepstack output shape mismatch");
      }
      std::copy(deepstack_outputs[stack].begin() +
                  static_cast<std::ptrdiff_t>(stack_offset),
                deepstack_outputs[stack].begin() +
                  static_cast<std::ptrdiff_t>(stack_offset + projection_size.value()),
                embeddings.begin() +
                  static_cast<std::ptrdiff_t>(feature_offset));
    }
  }

  Qwen35VisionEncoderResult result;
  result.image_plan = input_stage.value().image_plan;
  result.patch_grid_x = input_stage.value().patch_grid_x;
  result.patch_grid_y = input_stage.value().patch_grid_y;
  result.vision_embedding_length = graph_plan.value().vision_embedding_length;
  result.projection_dim = graph_plan.value().projection_dim;
  result.projector_output_width = graph_plan.value().projector_output_width;
  result.vision_token_count = input_stage.value().token_count;
  result.image_token_count = image_token_count;
  result.deepstack_layer_count = deepstack_outputs.size();
  result.embeddings = std::move(embeddings);
  return result;
}

std::string format_qwen35_vision_encoder_result(
  const Qwen35VisionEncoderResult& result) {
  std::ostringstream output;
  output << "vision encoder patch_grid: " << result.patch_grid_x << 'x'
         << result.patch_grid_y << '\n';
  output << "vision encoder embedding_length: "
         << result.vision_embedding_length << '\n';
  output << "vision encoder projection_dim: " << result.projection_dim << '\n';
  output << "vision encoder output_width: "
         << result.projector_output_width << '\n';
  output << "vision encoder vision_tokens: "
         << result.vision_token_count << '\n';
  output << "vision encoder image_tokens: "
         << result.image_token_count << '\n';
  output << "vision encoder deepstack_layers: "
         << result.deepstack_layer_count << '\n';
  output << "vision encoder values: " << result.embeddings.size() << '\n';
  return output.str();
}

bool qwen35_image_url_is_data_url(std::string_view url) {
  return url.size() >= 5U && url.substr(0, 5) == "data:";
}

std::uint64_t qwen35_image_content_fingerprint(
  std::string_view image_url, std::string_view image_mime_type,
  std::string_view detail, const std::vector<std::uint8_t>& image_bytes) {
  auto hash = kImageFingerprintRoot;
  hash = fingerprint_mix_string(hash, "qwen35-image-v1");
  hash = fingerprint_mix_string(hash, image_mime_type);
  hash = fingerprint_mix_string(hash, detail);
  if (!image_bytes.empty()) {
    hash = fingerprint_mix_string(hash, "bytes");
    hash = fingerprint_mix_bytes(hash, image_bytes.data(), image_bytes.size());
  } else {
    hash = fingerprint_mix_string(hash, "url");
    hash = fingerprint_mix_string(hash, image_url);
  }
  return hash == 0 ? 1 : hash;
}

Result<Qwen35ImageDimensions> infer_qwen35_image_dimensions(
  std::string_view mime_type, const std::vector<std::uint8_t>& image_bytes) {
  if (mime_type == "image/png") {
    return infer_png_dimensions(image_bytes);
  }
  if (mime_type == "image/jpeg" || mime_type == "image/jpg") {
    return infer_jpeg_dimensions(image_bytes);
  }
  return Status::invalid_argument("unsupported image MIME type for dimension inference: " +
                                  std::string{mime_type});
}

Result<Qwen35ImageDataUrl> parse_qwen35_image_data_url(std::string_view url) {
  if (!qwen35_image_url_is_data_url(url)) {
    return Status::invalid_argument("image URL is not a data URL");
  }
  const auto comma = url.find(',');
  if (comma == std::string_view::npos) {
    return Status::invalid_argument("image data URL is missing comma separator");
  }
  const auto metadata = url.substr(5, comma - 5U);
  const auto payload = url.substr(comma + 1U);

  std::string mime_type = "text/plain";
  bool base64 = false;
  std::size_t start = 0;
  bool first = true;
  while (start <= metadata.size()) {
    const auto semicolon = metadata.find(';', start);
    const auto end = semicolon == std::string_view::npos ? metadata.size() : semicolon;
    const auto token = metadata.substr(start, end - start);
    if (first && !token.empty() && token.find('=') == std::string_view::npos) {
      mime_type = ascii_lower(std::string{token});
    } else if (ascii_lower(std::string{token}) == "base64") {
      base64 = true;
    }
    first = false;
    if (semicolon == std::string_view::npos) {
      break;
    }
    start = semicolon + 1U;
  }
  if (!base64) {
    return Status::invalid_argument("image data URL must use base64 encoding");
  }
  if (mime_type.rfind("image/", 0) != 0) {
    return Status::invalid_argument("image data URL MIME type must start with image/");
  }
  auto bytes = decode_base64(payload);
  if (!bytes.is_ok()) {
    return bytes.status();
  }
  if (bytes.value().empty()) {
    return Status::invalid_argument("image data URL payload must not be empty");
  }
  Qwen35ImageDataUrl result{mime_type, std::move(bytes.value())};
  const auto dimensions = infer_qwen35_image_dimensions(result.mime_type, result.bytes);
  if (dimensions.is_ok()) {
    result.width = dimensions.value().width;
    result.height = dimensions.value().height;
  }
  return result;
}

Result<Qwen35ImageEmbeddingPlan> plan_qwen35_image_embeddings(
  const Qwen35MmprojMetadata& metadata, std::uint32_t image_width,
  std::uint32_t image_height) {
  if (!qwen35_mmproj_is_qwen3vl_merger(metadata)) {
    return Status::invalid_argument(
      "Qwen3.5 native image embedding plan requires qwen3vl_merger mmproj");
  }
  auto patch_size = positive_u32_from_i64(metadata.patch_size,
                                          "clip.vision.patch_size");
  if (!patch_size.is_ok()) {
    return patch_size.status();
  }
  auto spatial_merge_size =
    metadata.spatial_merge_size > 0
      ? positive_u32_from_i64(metadata.spatial_merge_size,
                              "clip.vision.spatial_merge_size")
      : Result<std::uint32_t>{2U};
  if (!spatial_merge_size.is_ok()) {
    return spatial_merge_size.status();
  }
  auto align_size = checked_u64_mul(patch_size.value(),
                                    spatial_merge_size.value(),
                                    "Qwen3.5 image align size");
  if (!align_size.is_ok()) {
    return align_size.status();
  }
  if (align_size.value() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::invalid_argument("Qwen3.5 image align size exceeds uint32 range");
  }
  auto patch_area = checked_u64_mul(align_size.value(), align_size.value(),
                                    "Qwen3.5 image patch area");
  if (!patch_area.is_ok()) {
    return patch_area.status();
  }
  auto min_pixels = metadata.image_min_pixels;
  if (min_pixels == 0) {
    auto default_min_pixels = checked_u64_mul(
      patch_area.value(), kQwen35VlDefaultMinImageTokens,
      "Qwen3.5 minimum image pixels");
    if (!default_min_pixels.is_ok()) {
      return default_min_pixels.status();
    }
    min_pixels = default_min_pixels.value();
  }
  auto max_pixels = metadata.image_max_pixels;
  if (max_pixels == 0) {
    auto default_max_pixels = checked_u64_mul(
      patch_area.value(), kQwen35VlDefaultMaxImageTokens,
      "Qwen3.5 maximum image pixels");
    if (!default_max_pixels.is_ok()) {
      return default_max_pixels.status();
    }
    max_pixels = default_max_pixels.value();
  }
  auto resized = qwen35_smart_resize(
    image_width, image_height, static_cast<std::uint32_t>(align_size.value()),
    min_pixels, max_pixels);
  if (!resized.is_ok()) {
    return resized.status();
  }
  if (resized.value().width % patch_size.value() != 0 ||
      resized.value().height % patch_size.value() != 0) {
    return Status::invalid_argument("Qwen3.5 resized image is not patch-aligned");
  }

  Qwen35ImageEmbeddingPlan plan;
  plan.original_width = image_width;
  plan.original_height = image_height;
  plan.resized_width = resized.value().width;
  plan.resized_height = resized.value().height;
  plan.patch_size = patch_size.value();
  plan.spatial_merge_size = spatial_merge_size.value();
  plan.patch_grid_x = plan.resized_width / plan.patch_size;
  plan.patch_grid_y = plan.resized_height / plan.patch_size;
  if (plan.patch_grid_x % plan.spatial_merge_size != 0 ||
      plan.patch_grid_y % plan.spatial_merge_size != 0) {
    return Status::invalid_argument(
      "Qwen3.5 resized image patch grid is not spatial-merge aligned");
  }
  plan.merge_grid_x = plan.patch_grid_x / plan.spatial_merge_size;
  plan.merge_grid_y = plan.patch_grid_y / plan.spatial_merge_size;
  plan.image_tokens =
    static_cast<std::size_t>(plan.merge_grid_x) * plan.merge_grid_y;
  plan.min_image_tokens = static_cast<std::size_t>(min_pixels / patch_area.value());
  plan.max_image_tokens = static_cast<std::size_t>(max_pixels / patch_area.value());
  if (plan.merge_grid_x == 0 || plan.merge_grid_y == 0 ||
      plan.image_tokens == 0) {
    return Status::invalid_argument("Qwen3.5 image embedding plan has no tokens");
  }
  return plan;
}

Result<Qwen35ImageEmbeddingPlan> plan_qwen35_image_embeddings(
  const Qwen35MmprojMetadata& metadata, const Qwen35ImageDataUrl& image) {
  return plan_qwen35_image_embeddings(metadata, image.width, image.height);
}

std::string format_qwen35_image_embedding_plan(
  const Qwen35ImageEmbeddingPlan& plan) {
  std::ostringstream output;
  output << "image original_size: " << plan.original_width << 'x'
         << plan.original_height << '\n';
  output << "image resized_size: " << plan.resized_width << 'x'
         << plan.resized_height << '\n';
  output << "image patch_size: " << plan.patch_size << '\n';
  output << "image spatial_merge_size: " << plan.spatial_merge_size << '\n';
  output << "image patch_grid: " << plan.patch_grid_x << 'x'
         << plan.patch_grid_y << '\n';
  output << "image merge_grid: " << plan.merge_grid_x << 'x'
         << plan.merge_grid_y << '\n';
  output << "image tokens: " << plan.image_tokens << '\n';
  output << "image token_limits: " << plan.min_image_tokens << ".."
         << plan.max_image_tokens << '\n';
  return output.str();
}

Result<Qwen35ImagePreprocessResult> preprocess_qwen35_image_for_vision(
  const Qwen35MmprojMetadata& metadata, const Qwen35ImageRgb& image) {
  if (!qwen35_mmproj_is_qwen3vl_merger(metadata)) {
    return Status::invalid_argument(
      "Qwen3.5 image preprocessing requires qwen3vl_merger mmproj");
  }
  if (!metadata.image_mean_std_present) {
    return Status::invalid_argument(
      "Qwen3.5 image preprocessing requires clip.vision.image_mean/std metadata");
  }
  if (image.width == 0 || image.height == 0) {
    return Status::invalid_argument("RGB image dimensions must be non-zero");
  }
  auto expected_elements = checked_image_elements(
    image.width, image.height, 3U, "RGB image element count");
  if (!expected_elements.is_ok()) {
    return expected_elements.status();
  }
  if (image.pixels.size() != expected_elements.value()) {
    return Status::invalid_argument("RGB image buffer size does not match dimensions");
  }
  auto plan = plan_qwen35_image_embeddings(metadata, image.width, image.height);
  if (!plan.is_ok()) {
    return plan.status();
  }
  auto resized_elements = checked_image_elements(
    plan.value().resized_width, plan.value().resized_height, 3U,
    "Qwen3.5 preprocessed image element count");
  if (!resized_elements.is_ok()) {
    return resized_elements.status();
  }

  const auto resized = resize_rgb_bilinear(
    image, plan.value().resized_width, plan.value().resized_height);
  Qwen35ImagePreprocessResult result;
  result.plan = plan.value();
  result.width = plan.value().resized_width;
  result.height = plan.value().resized_height;
  result.mean = metadata.image_mean;
  result.std = metadata.image_std;
  result.pixels.resize(resized_elements.value());
  for (std::size_t index = 0; index < resized_elements.value(); index += 3U) {
    result.pixels[index] =
      (static_cast<float>(resized[index]) / 255.0F - metadata.image_mean[0]) /
      metadata.image_std[0];
    result.pixels[index + 1U] =
      (static_cast<float>(resized[index + 1U]) / 255.0F - metadata.image_mean[1]) /
      metadata.image_std[1];
    result.pixels[index + 2U] =
      (static_cast<float>(resized[index + 2U]) / 255.0F - metadata.image_mean[2]) /
      metadata.image_std[2];
  }
  return result;
}

Result<Qwen35ImagePreprocessResult> preprocess_qwen35_image_for_vision(
  const Qwen35MmprojMetadata& metadata, const Qwen35ImageDataUrl& image) {
  auto decoded = decode_qwen35_image_rgb(image);
  if (!decoded.is_ok()) {
    return decoded.status();
  }
  return preprocess_qwen35_image_for_vision(metadata, decoded.value());
}

Result<Qwen35MultimodalPromptPlan> plan_qwen35_multimodal_prompt(
  const Qwen35MmprojMetadata& metadata, const std::vector<ChatMessage>& messages,
  bool add_generation_prompt, bool enable_thinking) {
  if (!qwen35_mmproj_is_qwen3vl_merger(metadata)) {
    return Status::invalid_argument(
      "Qwen3.5 multimodal prompt plan requires qwen3vl_merger mmproj");
  }
  Qwen35MultimodalPromptPlan plan;
  std::size_t image_index = 0;
  for (std::size_t message_index = 0; message_index < messages.size();
       ++message_index) {
    const auto& message = messages[message_index];
    if (!qwen35_role_is_supported(message.role)) {
      return Status::invalid_argument("unsupported chat message role: " +
                                      message.role);
    }
    {
      std::ostringstream prefix;
      prefix << "<|im_start|>" << message.role << '\n';
      add_multimodal_text_chunk(plan, prefix.str(), message_index, 0);
    }
    if (message.content_parts.empty()) {
      add_multimodal_text_chunk(plan, message.content, message_index, 0);
    } else {
      for (std::size_t part_index = 0; part_index < message.content_parts.size();
           ++part_index) {
        const auto& part = message.content_parts[part_index];
        if (part.kind == ChatContentPartKind::text) {
          add_multimodal_text_chunk(plan, part.text, message_index, part_index);
          continue;
        }
        if (part.kind != ChatContentPartKind::image_url) {
          return Status::invalid_argument("unsupported chat content part kind");
        }
        add_multimodal_text_chunk(plan, "<|vision_start|>", message_index,
                                  part_index);
        auto dimensions = image_part_dimensions(part);
        if (!dimensions.is_ok()) {
          return dimensions.status();
        }
        auto image_plan = plan_qwen35_image_embeddings(
          metadata, dimensions.value().width, dimensions.value().height);
        if (!image_plan.is_ok()) {
          return image_plan.status();
        }
        Qwen35MultimodalPromptChunk chunk;
        chunk.kind = Qwen35MultimodalPromptChunkKind::image;
        chunk.message_index = message_index;
        chunk.part_index = part_index;
        chunk.image_index = image_index++;
        chunk.image_fingerprint = part.image_fingerprint;
        chunk.image_plan = image_plan.value();
        plan.image_tokens += chunk.image_plan.image_tokens;
        plan.image_position_advance += std::max<std::size_t>(
          chunk.image_plan.merge_grid_x, chunk.image_plan.merge_grid_y);
        plan.chunks.push_back(std::move(chunk));
        add_multimodal_text_chunk(plan, "<|vision_end|>", message_index,
                                  part_index);
      }
    }
    add_multimodal_text_chunk(plan, "<|im_end|>\n", message_index, 0);
  }
  if (add_generation_prompt) {
    add_multimodal_text_chunk(plan, "<|im_start|>assistant\n", messages.size(), 0);
    if (enable_thinking) {
      add_multimodal_text_chunk(plan, "<think>\n", messages.size(), 0);
    } else {
      add_multimodal_text_chunk(plan, "<think>\n\n</think>\n\n",
                                messages.size(), 0);
    }
  }
  for (const auto& chunk : plan.chunks) {
    if (chunk.kind == Qwen35MultimodalPromptChunkKind::text) {
      ++plan.text_chunks;
    } else {
      ++plan.image_chunks;
    }
  }
  return plan;
}

std::string format_qwen35_multimodal_prompt_plan(
  const Qwen35MultimodalPromptPlan& plan) {
  std::ostringstream output;
  output << "multimodal chunks: " << plan.chunks.size() << '\n';
  output << "multimodal text_chunks: " << plan.text_chunks << '\n';
  output << "multimodal image_chunks: " << plan.image_chunks << '\n';
  output << "multimodal image_tokens: " << plan.image_tokens << '\n';
  output << "multimodal image_position_advance: "
         << plan.image_position_advance << '\n';
  for (const auto& chunk : plan.chunks) {
    output << "- ";
    if (chunk.kind == Qwen35MultimodalPromptChunkKind::text) {
      output << "text bytes=" << chunk.text.size();
    } else {
      output << "image index=" << chunk.image_index
             << " tokens=" << chunk.image_plan.image_tokens
             << " grid=" << chunk.image_plan.merge_grid_x << 'x'
             << chunk.image_plan.merge_grid_y
             << " fingerprint=" << chunk.image_fingerprint;
    }
    output << " message=" << chunk.message_index
           << " part=" << chunk.part_index << '\n';
  }
  return output.str();
}

Result<Qwen35MultimodalTokenPlan> tokenize_qwen35_multimodal_prompt(
  const GgufTokenizer& tokenizer, const Qwen35MultimodalPromptPlan& prompt_plan) {
  Qwen35MultimodalTokenPlan token_plan;
  token_plan.chunks.reserve(prompt_plan.chunks.size());
  for (const auto& prompt_chunk : prompt_plan.chunks) {
    Qwen35MultimodalTokenChunk chunk;
    chunk.kind = prompt_chunk.kind;
    chunk.message_index = prompt_chunk.message_index;
    chunk.part_index = prompt_chunk.part_index;
    chunk.image_index = prompt_chunk.image_index;
    chunk.image_fingerprint = prompt_chunk.image_fingerprint;
    chunk.image_plan = prompt_chunk.image_plan;
    if (prompt_chunk.kind == Qwen35MultimodalPromptChunkKind::text) {
      auto tokens = gguf_encode_text(tokenizer, prompt_chunk.text, false, true);
      if (!tokens.is_ok()) {
        return tokens.status();
      }
      chunk.text_tokens = std::move(tokens.value());
      chunk.token_count = chunk.text_tokens.size();
      chunk.position_advance = chunk.text_tokens.size();
      token_plan.text_tokens += chunk.text_tokens.size();
      ++token_plan.text_chunks;
    } else {
      chunk.token_count = chunk.image_plan.image_tokens;
      chunk.position_advance = std::max<std::size_t>(
        chunk.image_plan.merge_grid_x, chunk.image_plan.merge_grid_y);
      token_plan.image_tokens += chunk.image_plan.image_tokens;
      ++token_plan.image_chunks;
    }
    token_plan.total_tokens += chunk.token_count;
    token_plan.total_position_advance += chunk.position_advance;
    token_plan.chunks.push_back(std::move(chunk));
  }
  return token_plan;
}

Result<Qwen35MultimodalTokenPlan> tokenize_qwen35_multimodal_prompt(
  const GgufTokenizer& tokenizer, const Qwen35MmprojMetadata& metadata,
  const std::vector<ChatMessage>& messages, bool add_generation_prompt,
  bool enable_thinking) {
  auto prompt_plan = plan_qwen35_multimodal_prompt(
    metadata, messages, add_generation_prompt, enable_thinking);
  if (!prompt_plan.is_ok()) {
    return prompt_plan.status();
  }
  return tokenize_qwen35_multimodal_prompt(tokenizer, prompt_plan.value());
}

std::string format_qwen35_multimodal_token_plan(
  const Qwen35MultimodalTokenPlan& plan) {
  std::ostringstream output;
  output << "multimodal token chunks: " << plan.chunks.size() << '\n';
  output << "multimodal token text_chunks: " << plan.text_chunks << '\n';
  output << "multimodal token image_chunks: " << plan.image_chunks << '\n';
  output << "multimodal token text_tokens: " << plan.text_tokens << '\n';
  output << "multimodal token image_tokens: " << plan.image_tokens << '\n';
  output << "multimodal token total_tokens: " << plan.total_tokens << '\n';
  output << "multimodal token total_position_advance: "
         << plan.total_position_advance << '\n';
  for (const auto& chunk : plan.chunks) {
    output << "- ";
    if (chunk.kind == Qwen35MultimodalPromptChunkKind::text) {
      output << "text tokens=" << chunk.text_tokens.size();
    } else {
      output << "image index=" << chunk.image_index
             << " tokens=" << chunk.token_count
             << " pos_advance=" << chunk.position_advance
             << " grid=" << chunk.image_plan.merge_grid_x << 'x'
             << chunk.image_plan.merge_grid_y;
    }
    output << " message=" << chunk.message_index
           << " part=" << chunk.part_index << '\n';
  }
  return output.str();
}

Result<Qwen35MixedPrefillPlan> build_qwen35_mixed_prefill_plan(
  const GgufTokenizer& tokenizer, const Qwen35MmprojMetadata& metadata,
  const std::filesystem::path& mmproj_path,
  const std::vector<ChatMessage>& messages, bool add_generation_prompt,
  bool enable_thinking) {
  if (mmproj_path.empty()) {
    return Status::invalid_argument(
      "Qwen3.5 mixed prefill requires a qwen3vl_merger mmproj path");
  }
  if (!qwen35_mmproj_is_qwen3vl_merger(metadata)) {
    return Status::invalid_argument(
      "Qwen3.5 mixed prefill requires qwen3vl_merger mmproj");
  }
  if (metadata.projector_output_width == 0) {
    return Status::invalid_argument(
      "Qwen3.5 mixed prefill requires non-zero projector_output_width");
  }

  auto prompt_plan = plan_qwen35_multimodal_prompt(
    metadata, messages, add_generation_prompt, enable_thinking);
  if (!prompt_plan.is_ok()) {
    return prompt_plan.status();
  }
  auto token_plan =
    tokenize_qwen35_multimodal_prompt(tokenizer, prompt_plan.value());
  if (!token_plan.is_ok()) {
    return token_plan.status();
  }
  const auto image_parts = collect_qwen35_image_parts(messages);

  Qwen35MixedPrefillPlan mixed;
  mixed.embedding_width = metadata.projector_output_width;
  mixed.chunks.reserve(token_plan.value().chunks.size());
  for (const auto& token_chunk : token_plan.value().chunks) {
    Qwen35MixedPrefillChunk chunk;
    chunk.kind = token_chunk.kind;
    chunk.text_tokens = token_chunk.text_tokens;
    chunk.message_index = token_chunk.message_index;
    chunk.part_index = token_chunk.part_index;
    chunk.image_index = token_chunk.image_index;
    chunk.image_fingerprint = token_chunk.image_fingerprint;
    chunk.image_plan = token_chunk.image_plan;
    chunk.token_count = token_chunk.token_count;
    chunk.position_advance = token_chunk.position_advance;
    chunk.start_token = mixed.total_tokens;
    chunk.start_position = mixed.total_position_advance;

    if (token_chunk.kind == Qwen35MultimodalPromptChunkKind::text) {
      ++mixed.text_chunks;
      mixed.text_tokens += chunk.text_tokens.size();
    } else {
      if (chunk.image_index >= image_parts.size()) {
        return Status::invalid_argument(
          "Qwen3.5 mixed prefill image chunk index is out of range");
      }
      const auto* image_part = image_parts[chunk.image_index];
      if (image_part == nullptr || image_part->image_bytes.empty()) {
        return Status::unavailable(
          "Qwen3.5 mixed prefill currently requires data URL image bytes");
      }
      Qwen35ImageDataUrl image;
      image.mime_type = image_part->image_mime_type;
      image.bytes = image_part->image_bytes;
      image.width = image_part->image_width;
      image.height = image_part->image_height;
      auto preprocessed = preprocess_qwen35_image_for_vision(metadata, image);
      if (!preprocessed.is_ok()) {
        return preprocessed.status();
      }
      auto encoded = run_qwen35_vision_encoder_cpu(mmproj_path, preprocessed.value());
      if (!encoded.is_ok()) {
        return encoded.status();
      }
      if (encoded.value().image_token_count != chunk.token_count ||
          encoded.value().projector_output_width != mixed.embedding_width) {
        return Status::invalid_argument(
          "Qwen3.5 mixed prefill image encoder output does not match prompt plan");
      }
      chunk.image_plan = encoded.value().image_plan;
      chunk.image_embeddings = std::move(encoded.value().embeddings);
      ++mixed.image_chunks;
      mixed.image_tokens += chunk.token_count;
    }

    mixed.total_tokens += chunk.token_count;
    mixed.total_position_advance += chunk.position_advance;
    mixed.chunks.push_back(std::move(chunk));
  }
  if (mixed.total_tokens != token_plan.value().total_tokens ||
      mixed.total_position_advance != token_plan.value().total_position_advance ||
      mixed.text_tokens != token_plan.value().text_tokens ||
      mixed.image_tokens != token_plan.value().image_tokens) {
    return Status::internal_error(
      "Qwen3.5 mixed prefill counters do not match token plan");
  }
  mixed.mrope_positions.assign(mixed.total_tokens * 4U, 0);
  for (const auto& chunk : mixed.chunks) {
    Status status;
    if (chunk.kind == Qwen35MultimodalPromptChunkKind::text) {
      status = append_mixed_text_mrope_positions(
        mixed.mrope_positions, mixed.total_tokens, chunk.start_token,
        chunk.start_position, chunk.token_count);
    } else {
      status = append_mixed_image_mrope_positions(
        mixed.mrope_positions, mixed.total_tokens, chunk.start_token,
        chunk.start_position, chunk.image_plan);
    }
    if (!status.is_ok()) {
      return status;
    }
  }
  return mixed;
}

std::string format_qwen35_mixed_prefill_plan(
  const Qwen35MixedPrefillPlan& plan) {
  std::ostringstream output;
  output << "mixed prefill chunks: " << plan.chunks.size() << '\n';
  output << "mixed prefill text_chunks: " << plan.text_chunks << '\n';
  output << "mixed prefill image_chunks: " << plan.image_chunks << '\n';
  output << "mixed prefill text_tokens: " << plan.text_tokens << '\n';
  output << "mixed prefill image_tokens: " << plan.image_tokens << '\n';
  output << "mixed prefill total_tokens: " << plan.total_tokens << '\n';
  output << "mixed prefill total_position_advance: "
         << plan.total_position_advance << '\n';
  output << "mixed prefill embedding_width: " << plan.embedding_width << '\n';
  output << "mixed prefill mrope_positions: "
         << plan.mrope_positions.size() << '\n';
  for (const auto& chunk : plan.chunks) {
    output << "- ";
    if (chunk.kind == Qwen35MultimodalPromptChunkKind::text) {
      output << "text tokens=" << chunk.text_tokens.size();
    } else {
      output << "image index=" << chunk.image_index
             << " tokens=" << chunk.token_count
             << " embeddings=" << chunk.image_embeddings.size()
             << " grid=" << chunk.image_plan.merge_grid_x << 'x'
             << chunk.image_plan.merge_grid_y;
    }
    output << " start_token=" << chunk.start_token
           << " start_position=" << chunk.start_position
           << " pos_advance=" << chunk.position_advance
           << " message=" << chunk.message_index
           << " part=" << chunk.part_index << '\n';
  }
  return output.str();
}

}  // namespace toyllm
