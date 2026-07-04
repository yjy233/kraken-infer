#include "toyllm/runtime/qwen35_multimodal.hpp"

#include "toyllm/runtime/gguf_reader.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <sstream>
#include <string_view>
#include <variant>

namespace toyllm {

namespace {

constexpr std::uint64_t kImageFingerprintRoot = 1469598103934665603ULL;
constexpr std::uint64_t kImageFingerprintPrime = 1099511628211ULL;

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

}  // namespace toyllm
