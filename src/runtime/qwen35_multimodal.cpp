#include "toyllm/runtime/qwen35_multimodal.hpp"

#include "toyllm/runtime/gguf_reader.hpp"
#include "toyllm/runtime/gguf_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
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

std::vector<std::string> qwen3vl_missing_required_tensors(const GgufFile& file,
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

  const auto deepstack_indices = qwen3vl_deepstack_layer_indices(file);
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
    metadata.missing_required_tensors =
      qwen3vl_missing_required_tensors(gguf.value(), metadata.deepstack_layer_count);
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

}  // namespace toyllm
