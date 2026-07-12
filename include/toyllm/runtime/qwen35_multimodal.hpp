#pragma once

#include "toyllm/core/status.hpp"
#include "toyllm/runtime/chat_message.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace toyllm {

struct GgufTokenizer;

struct Qwen35MmprojMetadata {
  std::filesystem::path path;
  std::string architecture;
  std::string projector_type;
  std::string vision_projector_type;
  std::int64_t spatial_merge_size{0};
  std::int64_t patch_size{0};
  std::uint64_t image_min_pixels{0};
  std::uint64_t image_max_pixels{0};
  std::array<float, 3> image_mean{0.0F, 0.0F, 0.0F};
  std::array<float, 3> image_std{1.0F, 1.0F, 1.0F};
  bool image_mean_std_present{false};
  std::int64_t image_size{0};
  std::int64_t projection_dim{0};
  std::int64_t vision_feed_forward_length{0};
  std::int64_t vision_attention_head_count{0};
  double vision_attention_layer_norm_epsilon{0.0};
  std::vector<bool> deepstack_layer_flags;
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

struct Qwen35ImageEmbeddingPlan {
  std::uint32_t original_width{0};
  std::uint32_t original_height{0};
  std::uint32_t resized_width{0};
  std::uint32_t resized_height{0};
  std::uint32_t patch_grid_x{0};
  std::uint32_t patch_grid_y{0};
  std::uint32_t merge_grid_x{0};
  std::uint32_t merge_grid_y{0};
  std::uint32_t patch_size{0};
  std::uint32_t spatial_merge_size{0};
  std::size_t image_tokens{0};
  std::size_t min_image_tokens{0};
  std::size_t max_image_tokens{0};
};

struct Qwen35ImageRgb {
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::vector<std::uint8_t> pixels;
};

struct Qwen35ImagePreprocessResult {
  Qwen35ImageEmbeddingPlan plan;
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::uint32_t channels{3};
  std::array<float, 3> mean{0.0F, 0.0F, 0.0F};
  std::array<float, 3> std{1.0F, 1.0F, 1.0F};
  std::vector<float> pixels;
};

struct Qwen35VisionTensorPlan {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::uint32_t type{0};
  std::uint64_t byte_size{0};
};

struct Qwen35VisionBlockPlan {
  std::size_t layer_index{0};
  bool has_deepstack{false};
  std::vector<Qwen35VisionTensorPlan> tensors;
  std::vector<Qwen35VisionTensorPlan> deepstack_tensors;
};

struct Qwen35VisionGraphPlan {
  std::filesystem::path path;
  std::uint64_t image_size{0};
  std::uint64_t patch_size{0};
  std::uint64_t spatial_merge_size{0};
  std::uint64_t vision_embedding_length{0};
  std::uint64_t vision_feed_forward_length{0};
  std::uint64_t projection_dim{0};
  std::uint64_t vision_attention_head_count{0};
  double vision_attention_layer_norm_epsilon{0.0};
  std::size_t block_count{0};
  std::size_t deepstack_layer_count{0};
  std::uint64_t projector_output_width{0};
  std::size_t required_tensor_count{0};
  std::vector<std::size_t> deepstack_layer_indices;
  std::vector<Qwen35VisionTensorPlan> input_tensors;
  std::vector<Qwen35VisionTensorPlan> projector_tensors;
  std::vector<Qwen35VisionTensorPlan> output_norm_tensors;
  std::vector<Qwen35VisionBlockPlan> blocks;
};

struct Qwen35VisionInputStageResult {
  Qwen35ImageEmbeddingPlan image_plan;
  std::uint32_t patch_grid_x{0};
  std::uint32_t patch_grid_y{0};
  std::uint64_t vision_embedding_length{0};
  std::size_t token_count{0};
  std::vector<float> embeddings;
};

struct Qwen35VisionEncoderResult {
  Qwen35ImageEmbeddingPlan image_plan;
  std::uint32_t patch_grid_x{0};
  std::uint32_t patch_grid_y{0};
  std::uint64_t vision_embedding_length{0};
  std::uint64_t projection_dim{0};
  std::uint64_t projector_output_width{0};
  std::size_t vision_token_count{0};
  std::size_t image_token_count{0};
  std::size_t deepstack_layer_count{0};
  std::vector<float> embeddings;
};

enum class Qwen35MultimodalPromptChunkKind {
  text,
  image,
};

struct Qwen35MultimodalPromptChunk {
  Qwen35MultimodalPromptChunkKind kind{Qwen35MultimodalPromptChunkKind::text};
  std::string text;
  std::size_t message_index{0};
  std::size_t part_index{0};
  std::size_t image_index{0};
  std::uint64_t image_fingerprint{0};
  Qwen35ImageEmbeddingPlan image_plan;
};

struct Qwen35MultimodalPromptPlan {
  std::vector<Qwen35MultimodalPromptChunk> chunks;
  std::size_t text_chunks{0};
  std::size_t image_chunks{0};
  std::size_t image_tokens{0};
  std::size_t image_position_advance{0};
};

struct Qwen35MultimodalTokenChunk {
  Qwen35MultimodalPromptChunkKind kind{Qwen35MultimodalPromptChunkKind::text};
  std::vector<std::int64_t> text_tokens;
  std::size_t message_index{0};
  std::size_t part_index{0};
  std::size_t image_index{0};
  std::uint64_t image_fingerprint{0};
  Qwen35ImageEmbeddingPlan image_plan;
  std::size_t token_count{0};
  std::size_t position_advance{0};
};

struct Qwen35MultimodalTokenPlan {
  std::vector<Qwen35MultimodalTokenChunk> chunks;
  std::size_t text_chunks{0};
  std::size_t image_chunks{0};
  std::size_t text_tokens{0};
  std::size_t image_tokens{0};
  std::size_t total_tokens{0};
  std::size_t total_position_advance{0};
};

struct Qwen35MixedPrefillChunk {
  Qwen35MultimodalPromptChunkKind kind{Qwen35MultimodalPromptChunkKind::text};
  std::vector<std::int64_t> text_tokens;
  std::vector<float> image_embeddings;
  std::size_t message_index{0};
  std::size_t part_index{0};
  std::size_t image_index{0};
  std::uint64_t image_fingerprint{0};
  Qwen35ImageEmbeddingPlan image_plan;
  std::size_t token_count{0};
  std::size_t position_advance{0};
  std::size_t start_token{0};
  std::size_t start_position{0};
};

struct Qwen35MixedPrefillPlan {
  std::vector<Qwen35MixedPrefillChunk> chunks;
  std::size_t text_chunks{0};
  std::size_t image_chunks{0};
  std::size_t text_tokens{0};
  std::size_t image_tokens{0};
  std::size_t total_tokens{0};
  std::size_t total_position_advance{0};
  std::uint64_t embedding_width{0};
  std::vector<std::int32_t> mrope_positions;
};

[[nodiscard]] Result<Qwen35MmprojMetadata> load_qwen35_mmproj_metadata(
  const std::filesystem::path& path);
[[nodiscard]] bool qwen35_mmproj_is_qwen3vl_merger(
  const Qwen35MmprojMetadata& metadata);
[[nodiscard]] Status validate_qwen35_mmproj_text_embedding_compatibility(
  const Qwen35MmprojMetadata& metadata, std::int64_t text_embedding_length);
[[nodiscard]] std::string format_qwen35_mmproj_metadata_summary(
  const Qwen35MmprojMetadata& metadata);
[[nodiscard]] Result<Qwen35VisionGraphPlan> plan_qwen35_vision_graph(
  const std::filesystem::path& mmproj_path);
[[nodiscard]] std::string format_qwen35_vision_graph_plan(
  const Qwen35VisionGraphPlan& plan);
[[nodiscard]] Result<Qwen35VisionInputStageResult>
run_qwen35_vision_input_stage_cpu(const std::filesystem::path& mmproj_path,
                                  const Qwen35ImagePreprocessResult& image);
[[nodiscard]] std::string format_qwen35_vision_input_stage_result(
  const Qwen35VisionInputStageResult& result);
[[nodiscard]] Result<Qwen35VisionEncoderResult> run_qwen35_vision_encoder_cpu(
  const std::filesystem::path& mmproj_path,
  const Qwen35ImagePreprocessResult& image);
[[nodiscard]] std::string format_qwen35_vision_encoder_result(
  const Qwen35VisionEncoderResult& result);
[[nodiscard]] bool qwen35_image_url_is_data_url(std::string_view url);
[[nodiscard]] std::uint64_t qwen35_image_content_fingerprint(
  std::string_view image_url, std::string_view image_mime_type,
  std::string_view detail, const std::vector<std::uint8_t>& image_bytes);
[[nodiscard]] Result<Qwen35ImageDimensions> infer_qwen35_image_dimensions(
  std::string_view mime_type, const std::vector<std::uint8_t>& image_bytes);
[[nodiscard]] Result<Qwen35ImageDataUrl> parse_qwen35_image_data_url(
  std::string_view url);
[[nodiscard]] Result<Qwen35ImageRgb> decode_qwen35_image_rgb(
  const Qwen35ImageDataUrl& image);
[[nodiscard]] Result<Qwen35ImageEmbeddingPlan> plan_qwen35_image_embeddings(
  const Qwen35MmprojMetadata& metadata, std::uint32_t image_width,
  std::uint32_t image_height);
[[nodiscard]] Result<Qwen35ImageEmbeddingPlan> plan_qwen35_image_embeddings(
  const Qwen35MmprojMetadata& metadata, const Qwen35ImageDataUrl& image);
[[nodiscard]] std::string format_qwen35_image_embedding_plan(
  const Qwen35ImageEmbeddingPlan& plan);
[[nodiscard]] Result<Qwen35ImagePreprocessResult> preprocess_qwen35_image_for_vision(
  const Qwen35MmprojMetadata& metadata, const Qwen35ImageRgb& image);
[[nodiscard]] Result<Qwen35ImagePreprocessResult> preprocess_qwen35_image_for_vision(
  const Qwen35MmprojMetadata& metadata, const Qwen35ImageDataUrl& image);
[[nodiscard]] Result<Qwen35MultimodalPromptPlan> plan_qwen35_multimodal_prompt(
  const Qwen35MmprojMetadata& metadata, const std::vector<ChatMessage>& messages,
  bool add_generation_prompt, bool enable_thinking);
[[nodiscard]] std::string format_qwen35_multimodal_prompt_plan(
  const Qwen35MultimodalPromptPlan& plan);
[[nodiscard]] Result<Qwen35MultimodalTokenPlan> tokenize_qwen35_multimodal_prompt(
  const GgufTokenizer& tokenizer, const Qwen35MultimodalPromptPlan& prompt_plan);
[[nodiscard]] Result<Qwen35MultimodalTokenPlan> tokenize_qwen35_multimodal_prompt(
  const GgufTokenizer& tokenizer, const Qwen35MmprojMetadata& metadata,
  const std::vector<ChatMessage>& messages, bool add_generation_prompt,
  bool enable_thinking);
[[nodiscard]] std::string format_qwen35_multimodal_token_plan(
  const Qwen35MultimodalTokenPlan& plan);
[[nodiscard]] Result<Qwen35MixedPrefillPlan> build_qwen35_mixed_prefill_plan(
  const GgufTokenizer& tokenizer, const Qwen35MmprojMetadata& metadata,
  const std::filesystem::path& mmproj_path, const std::vector<ChatMessage>& messages,
  bool add_generation_prompt, bool enable_thinking);
[[nodiscard]] std::string format_qwen35_mixed_prefill_plan(
  const Qwen35MixedPrefillPlan& plan);

}  // namespace toyllm
