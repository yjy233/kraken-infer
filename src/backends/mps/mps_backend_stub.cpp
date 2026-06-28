#include "toyllm/backends/mps/mps_backend.hpp"

#include <sstream>
#include <utility>

namespace toyllm::mps {

struct MpsBuffer::Impl {};
struct MpsContext::Impl {};

MpsBuffer::MpsBuffer() = default;
MpsBuffer::~MpsBuffer() = default;
MpsBuffer::MpsBuffer(MpsBuffer&& other) noexcept = default;
MpsBuffer& MpsBuffer::operator=(MpsBuffer&& other) noexcept = default;
MpsBuffer::MpsBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool MpsBuffer::valid() const {
  return false;
}

std::size_t MpsBuffer::byte_size() const {
  return 0;
}

MpsMatVecWorkspace::MpsMatVecWorkspace() = default;
MpsMatVecWorkspace::~MpsMatVecWorkspace() = default;
MpsMatVecWorkspace::MpsMatVecWorkspace(MpsMatVecWorkspace&& other) noexcept = default;
MpsMatVecWorkspace& MpsMatVecWorkspace::operator=(MpsMatVecWorkspace&& other) noexcept =
  default;
MpsMatVecWorkspace::MpsMatVecWorkspace(std::size_t rows, std::size_t cols,
                                       MpsBuffer input, MpsBuffer output)
    : rows_(rows), cols_(cols), input_(std::move(input)), output_(std::move(output)) {}

bool MpsMatVecWorkspace::valid() const {
  return false;
}

std::size_t MpsMatVecWorkspace::rows() const {
  return rows_;
}

std::size_t MpsMatVecWorkspace::cols() const {
  return cols_;
}

MpsContext::MpsContext() = default;
MpsContext::~MpsContext() = default;
MpsContext::MpsContext(MpsContext&& other) noexcept = default;
MpsContext& MpsContext::operator=(MpsContext&& other) noexcept = default;
MpsContext::MpsContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Result<MpsContext> MpsContext::create() {
  return Status::unavailable("MPS backend was not compiled for this build");
}

bool MpsContext::valid() const {
  return false;
}

Status MpsContext::begin_graph() const {
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::commit_graph() const {
  return Status::unavailable("MPS backend was not compiled for this build");
}

void MpsContext::abort_graph() const {}

Result<MpsBuffer> MpsContext::make_buffer(std::size_t byte_size) const {
  (void)byte_size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::copy_to_buffer(MpsBuffer& buffer, const void* data,
                                  std::size_t byte_size) const {
  (void)buffer;
  (void)data;
  (void)byte_size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::copy_from_buffer(const MpsBuffer& buffer, void* data,
                                    std::size_t byte_size) const {
  (void)buffer;
  (void)data;
  (void)byte_size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::zero_buffer(MpsBuffer& buffer, std::size_t byte_size) const {
  (void)buffer;
  (void)byte_size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Result<MpsMatVecWorkspace> MpsContext::make_matvec_workspace(std::size_t rows,
                                                             std::size_t cols) const {
  (void)rows;
  (void)cols;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Result<std::vector<float>> MpsContext::matvec_bf16_f32(
  const MpsBuffer& weight, std::size_t rows, std::size_t cols,
  const std::vector<float>& input) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)input;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Result<std::vector<float>> MpsContext::matvec_bf16_f32(
  const MpsBuffer& weight, MpsMatVecWorkspace& workspace,
  const std::vector<float>& input) const {
  (void)weight;
  (void)workspace;
  (void)input;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::matvec_bf16_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)input;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::matmul_f32_f32_device(const MpsBuffer& lhs,
                                         const MpsBuffer& rhs,
                                         std::size_t lhs_rows,
                                         std::size_t inner_cols,
                                         std::size_t rhs_cols,
                                         MpsBuffer& output) const {
  (void)lhs;
  (void)rhs;
  (void)lhs_rows;
  (void)inner_cols;
  (void)rhs_cols;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Result<std::vector<float>> MpsContext::matvec_q4_k_f32(
  const MpsBuffer& weight, std::size_t rows, std::size_t cols,
  const std::vector<float>& input) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)input;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::matvec_q4_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)input;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::matmul_q4_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          std::size_t tokens,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)tokens;
  (void)input;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Result<std::vector<float>> MpsContext::matvec_q5_k_f32(
  const MpsBuffer& weight, std::size_t rows, std::size_t cols,
  const std::vector<float>& input) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)input;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::matvec_q5_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)input;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::matmul_q5_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          std::size_t tokens,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)tokens;
  (void)input;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Result<std::vector<float>> MpsContext::matvec_q6_k_f32(
  const MpsBuffer& weight, std::size_t rows, std::size_t cols,
  const std::vector<float>& input) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)input;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::matvec_q6_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)input;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::matmul_q6_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          std::size_t tokens,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)tokens;
  (void)input;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::dequantize_row_q4_k_f32(const MpsBuffer& weight, std::size_t row,
                                           std::size_t cols, MpsBuffer& output) const {
  (void)weight;
  (void)row;
  (void)cols;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::dequantize_rows_q4_k_f32(const MpsBuffer& weight,
                                            std::size_t rows,
                                            const MpsBuffer& row_ids,
                                            std::size_t tokens,
                                            std::size_t cols,
                                            MpsBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)row_ids;
  (void)tokens;
  (void)cols;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::dequantize_row_q5_k_f32(const MpsBuffer& weight, std::size_t row,
                                           std::size_t cols, MpsBuffer& output) const {
  (void)weight;
  (void)row;
  (void)cols;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::dequantize_rows_q5_k_f32(const MpsBuffer& weight,
                                            std::size_t rows,
                                            const MpsBuffer& row_ids,
                                            std::size_t tokens,
                                            std::size_t cols,
                                            MpsBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)row_ids;
  (void)tokens;
  (void)cols;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::dequantize_row_q6_k_f32(const MpsBuffer& weight, std::size_t row,
                                           std::size_t cols, MpsBuffer& output) const {
  (void)weight;
  (void)row;
  (void)cols;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::dequantize_rows_q6_k_f32(const MpsBuffer& weight,
                                            std::size_t rows,
                                            const MpsBuffer& row_ids,
                                            std::size_t tokens,
                                            std::size_t cols,
                                            MpsBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)row_ids;
  (void)tokens;
  (void)cols;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::embedding_bf16_f32(const MpsBuffer& weight, std::int64_t token,
                                      std::size_t hidden_size,
                                      MpsBuffer& output) const {
  (void)weight;
  (void)token;
  (void)hidden_size;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::rms_norm_f32_bf16(const MpsBuffer& input, const MpsBuffer& weight,
                                     std::size_t size, float eps,
                                     MpsBuffer& output) const {
  (void)input;
  (void)weight;
  (void)size;
  (void)eps;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::rms_norm_f32_f32(const MpsBuffer& input, const MpsBuffer& weight,
                                    std::size_t size, float eps,
                                    MpsBuffer& output) const {
  (void)input;
  (void)weight;
  (void)size;
  (void)eps;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::rms_norm_f32_f32_batched(const MpsBuffer& input,
                                            const MpsBuffer& weight,
                                            std::size_t tokens,
                                            std::size_t size,
                                            float eps,
                                            MpsBuffer& output) const {
  (void)input;
  (void)weight;
  (void)tokens;
  (void)size;
  (void)eps;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::qk_norm_f32_bf16(MpsBuffer& values, const MpsBuffer& weight,
                                    std::size_t heads, std::size_t head_dim,
                                    float eps) const {
  (void)values;
  (void)weight;
  (void)heads;
  (void)head_dim;
  (void)eps;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::qk_norm_f32_f32(MpsBuffer& values, const MpsBuffer& weight,
                                   std::size_t heads, std::size_t head_dim,
                                   float eps) const {
  (void)values;
  (void)weight;
  (void)heads;
  (void)head_dim;
  (void)eps;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::qk_norm_f32_f32_batched(MpsBuffer& values,
                                           const MpsBuffer& weight,
                                           std::size_t tokens,
                                           std::size_t heads,
                                           std::size_t head_dim,
                                           float eps) const {
  (void)values;
  (void)weight;
  (void)tokens;
  (void)heads;
  (void)head_dim;
  (void)eps;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::qwen35_norm_gated_f32_in_place(
    MpsBuffer& values, const MpsBuffer& weight, const MpsBuffer& gate,
    std::size_t tokens, std::size_t heads, std::size_t head_dim,
    float eps) const {
  (void)values;
  (void)weight;
  (void)gate;
  (void)tokens;
  (void)heads;
  (void)head_dim;
  (void)eps;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::l2_norm_f32_in_place(MpsBuffer& values, std::size_t rows,
                                        std::size_t row_size, float eps) const {
  (void)values;
  (void)rows;
  (void)row_size;
  (void)eps;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::split_qkv_l2_norm_f32_qwen35(
    const MpsBuffer& source, MpsBuffer& query, MpsBuffer& key,
    MpsBuffer& value, std::size_t tokens, std::size_t key_heads,
    std::size_t value_heads, std::size_t head_dim, float eps) const {
  (void)source;
  (void)query;
  (void)key;
  (void)value;
  (void)tokens;
  (void)key_heads;
  (void)value_heads;
  (void)head_dim;
  (void)eps;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::gated_delta_net_f32_in_place(
  const MpsBuffer& query, const MpsBuffer& key, const MpsBuffer& value,
  const MpsBuffer& gate, const MpsBuffer& beta, MpsBuffer& state,
  std::size_t key_heads, std::size_t value_heads, std::size_t head_dim,
  MpsBuffer& output) const {
  (void)query;
  (void)key;
  (void)value;
  (void)gate;
  (void)beta;
  (void)state;
  (void)key_heads;
  (void)value_heads;
  (void)head_dim;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::gated_delta_net_f32_in_place_at(
  const MpsBuffer& query, const MpsBuffer& key, const MpsBuffer& value,
  const MpsBuffer& gate, const MpsBuffer& beta, MpsBuffer& state,
  std::size_t state_offset, std::size_t key_heads, std::size_t value_heads,
  std::size_t head_dim, MpsBuffer& output) const {
  (void)query;
  (void)key;
  (void)value;
  (void)gate;
  (void)beta;
  (void)state;
  (void)state_offset;
  (void)key_heads;
  (void)value_heads;
  (void)head_dim;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::gated_delta_net_f32_batched_in_place(
  const MpsBuffer& query, const MpsBuffer& key, const MpsBuffer& value,
  const MpsBuffer& gate, const MpsBuffer& beta, MpsBuffer& state,
  std::size_t tokens, std::size_t key_heads, std::size_t value_heads,
  std::size_t head_dim, MpsBuffer& output) const {
  (void)query;
  (void)key;
  (void)value;
  (void)gate;
  (void)beta;
  (void)state;
  (void)tokens;
  (void)key_heads;
  (void)value_heads;
  (void)head_dim;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::gated_delta_net_f32_batched_in_place_at(
  const MpsBuffer& query, const MpsBuffer& key, const MpsBuffer& value,
  const MpsBuffer& gate, const MpsBuffer& beta, MpsBuffer& state,
  std::size_t state_offset, std::size_t tokens, std::size_t key_heads,
  std::size_t value_heads, std::size_t head_dim, MpsBuffer& output) const {
  (void)query;
  (void)key;
  (void)value;
  (void)gate;
  (void)beta;
  (void)state;
  (void)state_offset;
  (void)tokens;
  (void)key_heads;
  (void)value_heads;
  (void)head_dim;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::ssm_conv_f32(const MpsBuffer& input, const MpsBuffer& kernel,
                                std::size_t conv_kernel, std::size_t channels,
                                std::size_t tokens, std::size_t sequences,
                                MpsBuffer& output) const {
  (void)input;
  (void)kernel;
  (void)conv_kernel;
  (void)channels;
  (void)tokens;
  (void)sequences;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::build_ssm_conv_state_f32(MpsBuffer& state,
                                            std::size_t state_offset,
                                            const MpsBuffer& input,
                                            std::size_t conv_kernel,
                                            std::size_t channels,
                                            std::size_t tokens,
                                            MpsBuffer& conv_input) const {
  (void)state;
  (void)state_offset;
  (void)input;
  (void)conv_kernel;
  (void)channels;
  (void)tokens;
  (void)conv_input;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::ssm_conv1_f32_stateful(MpsBuffer& state,
                                          const MpsBuffer& input,
                                          const MpsBuffer& kernel,
                                          std::size_t conv_kernel,
                                          std::size_t channels,
                                          MpsBuffer& output) const {
  (void)state;
  (void)input;
  (void)kernel;
  (void)conv_kernel;
  (void)channels;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::ssm_conv1_f32_stateful_at(MpsBuffer& state,
                                             std::size_t state_offset,
                                             const MpsBuffer& input,
                                             const MpsBuffer& kernel,
                                             std::size_t conv_kernel,
                                             std::size_t channels,
                                             MpsBuffer& output) const {
  (void)state;
  (void)state_offset;
  (void)input;
  (void)kernel;
  (void)conv_kernel;
  (void)channels;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::rope_f32(MpsBuffer& values, std::size_t heads,
                            std::size_t head_dim, std::size_t position,
                            float theta) const {
  (void)values;
  (void)heads;
  (void)head_dim;
  (void)position;
  (void)theta;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::mrope_f32_in_place(MpsBuffer& values, const MpsBuffer& positions,
                                      std::size_t tokens, std::size_t heads,
                                      std::size_t head_dim, std::size_t n_dims,
                                      std::size_t section_0, std::size_t section_1,
                                      std::size_t section_2, std::size_t section_3,
                                      float theta) const {
  (void)values;
  (void)positions;
  (void)tokens;
  (void)heads;
  (void)head_dim;
  (void)n_dims;
  (void)section_0;
  (void)section_1;
  (void)section_2;
  (void)section_3;
  (void)theta;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::add_f32_in_place(MpsBuffer& target, const MpsBuffer& delta,
                                    std::size_t size) const {
  (void)target;
  (void)delta;
  (void)size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::mul_f32_in_place(MpsBuffer& target, const MpsBuffer& rhs,
                                    std::size_t size) const {
  (void)target;
  (void)rhs;
  (void)size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::add_f32_row_in_place(MpsBuffer& target, const MpsBuffer& row,
                                        std::size_t rows,
                                        std::size_t row_size) const {
  (void)target;
  (void)row;
  (void)rows;
  (void)row_size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::mul_f32_row_in_place(MpsBuffer& target, const MpsBuffer& row,
                                        std::size_t rows,
                                        std::size_t row_size) const {
  (void)target;
  (void)row;
  (void)rows;
  (void)row_size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::sigmoid_f32_in_place(MpsBuffer& values, std::size_t size) const {
  (void)values;
  (void)size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::softplus_f32_in_place(MpsBuffer& values, std::size_t size) const {
  (void)values;
  (void)size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::prepare_qwen35_gdn_gate_beta_f32(
    MpsBuffer& gate, MpsBuffer& beta, const MpsBuffer& gate_bias,
    const MpsBuffer& gate_scale, std::size_t rows,
    std::size_t row_size) const {
  (void)gate;
  (void)beta;
  (void)gate_bias;
  (void)gate_scale;
  (void)rows;
  (void)row_size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::silu_f32_in_place(MpsBuffer& values, std::size_t size) const {
  (void)values;
  (void)size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::silu_mul_f32_in_place(MpsBuffer& gate, const MpsBuffer& up,
                                         std::size_t size) const {
  (void)gate;
  (void)up;
  (void)size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::copy_f32_region(const MpsBuffer& source, MpsBuffer& destination,
                                   std::size_t source_offset,
                                   std::size_t destination_offset,
                                   std::size_t size) const {
  (void)source;
  (void)destination;
  (void)source_offset;
  (void)destination_offset;
  (void)size;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::copy_f32_rows(const MpsBuffer& source, MpsBuffer& destination,
                                 std::size_t rows, std::size_t row_size,
                                 std::size_t source_stride,
                                 std::size_t source_offset,
                                 std::size_t destination_stride,
                                 std::size_t destination_offset) const {
  (void)source;
  (void)destination;
  (void)rows;
  (void)row_size;
  (void)source_stride;
  (void)source_offset;
  (void)destination_stride;
  (void)destination_offset;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::copy_f32_rows_to_f16(
  const MpsBuffer& source, MpsBuffer& destination, std::size_t rows,
  std::size_t row_size, std::size_t source_stride,
  std::size_t source_offset, std::size_t destination_stride,
  std::size_t destination_offset) const {
  (void)source;
  (void)destination;
  (void)rows;
  (void)row_size;
  (void)source_stride;
  (void)source_offset;
  (void)destination_stride;
  (void)destination_offset;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::copy_f16_rows(const MpsBuffer& source, MpsBuffer& destination,
                                 std::size_t rows, std::size_t row_size,
                                 std::size_t source_stride,
                                 std::size_t source_offset,
                                 std::size_t destination_stride,
                                 std::size_t destination_offset) const {
  (void)source;
  (void)destination;
  (void)rows;
  (void)row_size;
  (void)source_stride;
  (void)source_offset;
  (void)destination_stride;
  (void)destination_offset;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::argmax_f32_i32(const MpsBuffer& values, std::size_t size,
                                  MpsBuffer& output) const {
  (void)values;
  (void)size;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::attention_f32(const MpsBuffer& query,
                                 const MpsBuffer& key_cache,
                                 const MpsBuffer& value_cache,
                                 std::size_t layer, std::size_t position,
                                 std::size_t capacity_tokens,
                                 std::size_t heads, std::size_t kv_heads,
                                 std::size_t head_dim,
                                 MpsBuffer& output) const {
  (void)query;
  (void)key_cache;
  (void)value_cache;
  (void)layer;
  (void)position;
  (void)capacity_tokens;
  (void)heads;
  (void)kv_heads;
  (void)head_dim;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::attention_f32_f16_kv(
  const MpsBuffer& query, const MpsBuffer& key_cache,
  const MpsBuffer& value_cache, std::size_t layer, std::size_t position,
  std::size_t capacity_tokens, std::size_t heads, std::size_t kv_heads,
  std::size_t head_dim, MpsBuffer& output) const {
  (void)query;
  (void)key_cache;
  (void)value_cache;
  (void)layer;
  (void)position;
  (void)capacity_tokens;
  (void)heads;
  (void)kv_heads;
  (void)head_dim;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::attention_f32_batched(const MpsBuffer& query,
                                         const MpsBuffer& key_cache,
                                         const MpsBuffer& value_cache,
                                         std::size_t layer,
                                         std::size_t start_position,
                                         std::size_t tokens,
                                         std::size_t capacity_tokens,
                                         std::size_t heads,
                                         std::size_t kv_heads,
                                         std::size_t head_dim,
                                         MpsBuffer& output) const {
  (void)query;
  (void)key_cache;
  (void)value_cache;
  (void)layer;
  (void)start_position;
  (void)tokens;
  (void)capacity_tokens;
  (void)heads;
  (void)kv_heads;
  (void)head_dim;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

Status MpsContext::attention_f32_batched_f16_kv(
  const MpsBuffer& query, const MpsBuffer& key_cache,
  const MpsBuffer& value_cache, std::size_t layer,
  std::size_t start_position, std::size_t tokens,
  std::size_t capacity_tokens, std::size_t heads, std::size_t kv_heads,
  std::size_t head_dim, MpsBuffer& output) const {
  (void)query;
  (void)key_cache;
  (void)value_cache;
  (void)layer;
  (void)start_position;
  (void)tokens;
  (void)capacity_tokens;
  (void)heads;
  (void)kv_heads;
  (void)head_dim;
  (void)output;
  return Status::unavailable("MPS backend was not compiled for this build");
}

BackendInfo query_backend() {
  BackendInfo info{};
  info.failure_reason = "MPS backend was not compiled for this build";
  return info;
}

std::string format_backend_info(const BackendInfo& info) {
  std::ostringstream output;
  output << "MPS compiled: " << (info.compiled ? "yes" : "no") << '\n';
  output << "MPS available: " << (info.available ? "yes" : "no") << '\n';
  output << "MPS compute ready: " << (info.compute_ready ? "yes" : "no") << '\n';
  output << "MPS full forward ready: " << (info.forward_ready ? "yes" : "no") << '\n';
  if (!info.failure_reason.empty()) {
    output << "Reason: " << info.failure_reason << '\n';
  }
  return output.str();
}

Status run_operator_smoke_test() {
  return Status::unavailable("MPS backend was not compiled for this build");
}

}  // namespace toyllm::mps
