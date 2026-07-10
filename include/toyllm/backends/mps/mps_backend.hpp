#pragma once

#include "toyllm/core/status.hpp"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace toyllm::mps {

struct BackendInfo {
  bool compiled{false};
  bool available{false};
  bool compute_ready{false};
  bool forward_ready{false};
  std::string device_name;
  std::uint64_t recommended_max_working_set_size{0};
  bool low_power{false};
  bool headless{false};
  bool removable{false};
  std::string failure_reason;
};

class MpsBuffer {
 public:
  MpsBuffer();
  ~MpsBuffer();

  MpsBuffer(const MpsBuffer&) = delete;
  MpsBuffer& operator=(const MpsBuffer&) = delete;

  MpsBuffer(MpsBuffer&& other) noexcept;
  MpsBuffer& operator=(MpsBuffer&& other) noexcept;

  [[nodiscard]] bool valid() const;
  [[nodiscard]] std::size_t byte_size() const;

 private:
  friend class MpsContext;
  struct Impl;

  explicit MpsBuffer(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

class MpsMatVecWorkspace {
 public:
  MpsMatVecWorkspace();
  ~MpsMatVecWorkspace();

  MpsMatVecWorkspace(const MpsMatVecWorkspace&) = delete;
  MpsMatVecWorkspace& operator=(const MpsMatVecWorkspace&) = delete;

  MpsMatVecWorkspace(MpsMatVecWorkspace&& other) noexcept;
  MpsMatVecWorkspace& operator=(MpsMatVecWorkspace&& other) noexcept;

  [[nodiscard]] bool valid() const;
  [[nodiscard]] std::size_t rows() const;
  [[nodiscard]] std::size_t cols() const;

 private:
  friend class MpsContext;

  MpsMatVecWorkspace(std::size_t rows, std::size_t cols, MpsBuffer input,
                     MpsBuffer output);

  std::size_t rows_{0};
  std::size_t cols_{0};
  MpsBuffer input_;
  MpsBuffer output_;
};

class MpsContext {
 public:
  MpsContext();
  ~MpsContext();

  MpsContext(const MpsContext&) = delete;
  MpsContext& operator=(const MpsContext&) = delete;

  MpsContext(MpsContext&& other) noexcept;
  MpsContext& operator=(MpsContext&& other) noexcept;

  [[nodiscard]] static Result<MpsContext> create();
  [[nodiscard]] bool valid() const;

  // Mirrors llama.cpp's graph execution boundary: callers set CPU inputs first,
  // then encode multiple kernels and synchronize once at the end.
  [[nodiscard]] Status begin_graph() const;
  [[nodiscard]] Status commit_graph() const;
  void abort_graph() const;

  [[nodiscard]] Result<MpsBuffer> make_buffer(std::size_t byte_size) const;
  [[nodiscard]] Status copy_to_buffer(MpsBuffer& buffer, const void* data,
                                      std::size_t byte_size) const;
  [[nodiscard]] Status copy_to_buffer_at(MpsBuffer& buffer,
                                         std::size_t destination_offset,
                                         const void* data,
                                         std::size_t byte_size) const;
  [[nodiscard]] Status copy_from_buffer(const MpsBuffer& buffer, void* data,
                                        std::size_t byte_size) const;
  [[nodiscard]] Status copy_from_buffer_at(const MpsBuffer& buffer,
                                           std::size_t source_offset,
                                           void* data,
                                           std::size_t byte_size) const;
  [[nodiscard]] Status copy_buffer_region(const MpsBuffer& source,
                                          MpsBuffer& destination,
                                          std::size_t source_offset,
                                          std::size_t destination_offset,
                                          std::size_t byte_size) const;
  [[nodiscard]] Status zero_buffer(MpsBuffer& buffer, std::size_t byte_size) const;
  [[nodiscard]] Result<MpsMatVecWorkspace> make_matvec_workspace(std::size_t rows,
                                                                 std::size_t cols) const;
  [[nodiscard]] Result<std::vector<float>> matvec_bf16_f32(const MpsBuffer& weight,
                                                           std::size_t rows,
                                                           std::size_t cols,
                                                           const std::vector<float>& input) const;
  [[nodiscard]] Result<std::vector<float>> matvec_bf16_f32(
    const MpsBuffer& weight, MpsMatVecWorkspace& workspace,
    const std::vector<float>& input) const;
  [[nodiscard]] Status matvec_bf16_f32_device(const MpsBuffer& weight,
                                              std::size_t rows, std::size_t cols,
                                              const MpsBuffer& input,
                                              MpsBuffer& output) const;
  [[nodiscard]] Status matmul_f32_f32_device(const MpsBuffer& lhs,
                                             const MpsBuffer& rhs,
                                             std::size_t lhs_rows,
                                             std::size_t inner_cols,
                                             std::size_t rhs_cols,
                                             MpsBuffer& output) const;
  [[nodiscard]] Result<std::vector<float>> matvec_q4_k_f32(
    const MpsBuffer& weight, std::size_t rows, std::size_t cols,
    const std::vector<float>& input) const;
  [[nodiscard]] Status matvec_q4_k_f32_device(const MpsBuffer& weight,
                                              std::size_t rows, std::size_t cols,
                                              const MpsBuffer& input,
                                              MpsBuffer& output) const;
  [[nodiscard]] Status matvec_q4_k_f32_top1(const MpsBuffer& weight,
                                            std::size_t rows, std::size_t cols,
                                            const MpsBuffer& input,
                                            MpsBuffer& token_output,
                                            MpsBuffer& probability_output) const;
  [[nodiscard]] Status matvec_q4_k_f32_argmax(const MpsBuffer& weight,
                                              std::size_t rows, std::size_t cols,
                                              const MpsBuffer& input,
                                              MpsBuffer& token_output) const;
  [[nodiscard]] Status matmul_q4_k_f32_device(const MpsBuffer& weight,
                                              std::size_t rows, std::size_t cols,
                                              std::size_t tokens,
                                              const MpsBuffer& input,
                                              MpsBuffer& output) const;
  [[nodiscard]] Result<std::vector<float>> matvec_q5_k_f32(
    const MpsBuffer& weight, std::size_t rows, std::size_t cols,
    const std::vector<float>& input) const;
  [[nodiscard]] Status matvec_q5_k_f32_device(const MpsBuffer& weight,
                                              std::size_t rows, std::size_t cols,
                                              const MpsBuffer& input,
                                              MpsBuffer& output) const;
  [[nodiscard]] Status matvec_q5_k_f32_top1(const MpsBuffer& weight,
                                            std::size_t rows, std::size_t cols,
                                            const MpsBuffer& input,
                                            MpsBuffer& token_output,
                                            MpsBuffer& probability_output) const;
  [[nodiscard]] Status matvec_q5_k_f32_argmax(const MpsBuffer& weight,
                                              std::size_t rows, std::size_t cols,
                                              const MpsBuffer& input,
                                              MpsBuffer& token_output) const;
  [[nodiscard]] Status matmul_q5_k_f32_device(const MpsBuffer& weight,
                                              std::size_t rows, std::size_t cols,
                                              std::size_t tokens,
                                              const MpsBuffer& input,
                                              MpsBuffer& output) const;
  [[nodiscard]] Result<std::vector<float>> matvec_q6_k_f32(
    const MpsBuffer& weight, std::size_t rows, std::size_t cols,
    const std::vector<float>& input) const;
  [[nodiscard]] Status matvec_q6_k_f32_device(const MpsBuffer& weight,
                                              std::size_t rows, std::size_t cols,
                                              const MpsBuffer& input,
                                              MpsBuffer& output) const;
  [[nodiscard]] Status matvec_q6_k_f32_top1(const MpsBuffer& weight,
                                            std::size_t rows, std::size_t cols,
                                            const MpsBuffer& input,
                                            MpsBuffer& token_output,
                                            MpsBuffer& probability_output) const;
  [[nodiscard]] Status matvec_q6_k_f32_argmax(const MpsBuffer& weight,
                                              std::size_t rows, std::size_t cols,
                                              const MpsBuffer& input,
                                              MpsBuffer& token_output) const;
  [[nodiscard]] Status matmul_q6_k_f32_device(const MpsBuffer& weight,
                                              std::size_t rows, std::size_t cols,
                                              std::size_t tokens,
                                              const MpsBuffer& input,
                                              MpsBuffer& output) const;
  [[nodiscard]] Status dequantize_row_q4_k_f32(const MpsBuffer& weight,
                                               std::size_t row,
                                               std::size_t cols,
                                               MpsBuffer& output) const;
  [[nodiscard]] Status dequantize_rows_q4_k_f32(const MpsBuffer& weight,
                                                std::size_t rows,
                                                const MpsBuffer& row_ids,
                                                std::size_t tokens,
                                                std::size_t cols,
                                                MpsBuffer& output) const;
  [[nodiscard]] Status dequantize_row_q5_k_f32(const MpsBuffer& weight,
                                               std::size_t row,
                                               std::size_t cols,
                                               MpsBuffer& output) const;
  [[nodiscard]] Status dequantize_rows_q5_k_f32(const MpsBuffer& weight,
                                                std::size_t rows,
                                                const MpsBuffer& row_ids,
                                                std::size_t tokens,
                                                std::size_t cols,
                                                MpsBuffer& output) const;
  [[nodiscard]] Status dequantize_row_q6_k_f32(const MpsBuffer& weight,
                                               std::size_t row,
                                               std::size_t cols,
                                               MpsBuffer& output) const;
  [[nodiscard]] Status dequantize_rows_q6_k_f32(const MpsBuffer& weight,
                                                std::size_t rows,
                                                const MpsBuffer& row_ids,
                                                std::size_t tokens,
                                                std::size_t cols,
                                                MpsBuffer& output) const;
  [[nodiscard]] Status embedding_bf16_f32(const MpsBuffer& weight, std::int64_t token,
                                          std::size_t hidden_size,
                                          MpsBuffer& output) const;
  [[nodiscard]] Status rms_norm_f32_bf16(const MpsBuffer& input,
                                         const MpsBuffer& weight,
                                         std::size_t size, float eps,
                                         MpsBuffer& output) const;
  [[nodiscard]] Status rms_norm_f32_f32(const MpsBuffer& input,
                                        const MpsBuffer& weight,
                                        std::size_t size, float eps,
                                        MpsBuffer& output) const;
  [[nodiscard]] Status rms_norm_f32_f32_batched(const MpsBuffer& input,
                                                const MpsBuffer& weight,
                                                std::size_t tokens,
                                                std::size_t size,
                                                float eps,
                                                MpsBuffer& output) const;
  [[nodiscard]] Status qk_norm_f32_bf16(MpsBuffer& values,
                                        const MpsBuffer& weight,
                                        std::size_t heads, std::size_t head_dim,
                                        float eps) const;
  [[nodiscard]] Status qk_norm_f32_f32(MpsBuffer& values,
                                       const MpsBuffer& weight,
                                       std::size_t heads, std::size_t head_dim,
                                       float eps) const;
  [[nodiscard]] Status qk_norm_f32_f32_batched(MpsBuffer& values,
                                               const MpsBuffer& weight,
                                               std::size_t tokens,
                                               std::size_t heads,
                                               std::size_t head_dim,
                                               float eps) const;
  [[nodiscard]] Status qwen35_norm_gated_f32_in_place(
    MpsBuffer& values, const MpsBuffer& weight, const MpsBuffer& gate,
    std::size_t tokens, std::size_t heads, std::size_t head_dim,
    float eps) const;
  [[nodiscard]] Status l2_norm_f32_in_place(MpsBuffer& values,
                                            std::size_t rows,
                                            std::size_t row_size,
                                            float eps) const;
  [[nodiscard]] Status split_qkv_l2_norm_f32_qwen35(
    const MpsBuffer& source, MpsBuffer& query, MpsBuffer& key,
    MpsBuffer& value, std::size_t tokens, std::size_t key_heads,
    std::size_t value_heads, std::size_t head_dim, float eps) const;
  [[nodiscard]] Status gated_delta_net_f32_in_place(const MpsBuffer& query,
                                                    const MpsBuffer& key,
                                                    const MpsBuffer& value,
                                                    const MpsBuffer& gate,
                                                    const MpsBuffer& beta,
                                                    MpsBuffer& state,
                                                    std::size_t key_heads,
                                                    std::size_t value_heads,
                                                    std::size_t head_dim,
                                                    MpsBuffer& output) const;
  [[nodiscard]] Status gated_delta_net_f32_in_place_at(const MpsBuffer& query,
                                                       const MpsBuffer& key,
                                                       const MpsBuffer& value,
                                                       const MpsBuffer& gate,
                                                       const MpsBuffer& beta,
                                                       MpsBuffer& state,
                                                       std::size_t state_offset,
                                                       std::size_t key_heads,
                                                       std::size_t value_heads,
                                                       std::size_t head_dim,
                                                       MpsBuffer& output) const;
  [[nodiscard]] Status gated_delta_net_f32_batched_in_place(
    const MpsBuffer& query, const MpsBuffer& key, const MpsBuffer& value,
    const MpsBuffer& gate, const MpsBuffer& beta, MpsBuffer& state,
    std::size_t tokens, std::size_t key_heads, std::size_t value_heads,
    std::size_t head_dim, MpsBuffer& output) const;
  [[nodiscard]] Status gated_delta_net_f32_batched_in_place_at(
    const MpsBuffer& query, const MpsBuffer& key, const MpsBuffer& value,
    const MpsBuffer& gate, const MpsBuffer& beta, MpsBuffer& state,
    std::size_t state_offset, std::size_t tokens, std::size_t key_heads,
    std::size_t value_heads, std::size_t head_dim, MpsBuffer& output) const;
  [[nodiscard]] Status ssm_conv_f32(const MpsBuffer& input,
                                    const MpsBuffer& kernel,
                                    std::size_t conv_kernel,
                                    std::size_t channels,
                                    std::size_t tokens,
                                    std::size_t sequences,
                                    MpsBuffer& output) const;
  [[nodiscard]] Status build_ssm_conv_state_f32(MpsBuffer& state,
                                                std::size_t state_offset,
                                                const MpsBuffer& input,
                                                std::size_t conv_kernel,
                                                std::size_t channels,
                                                std::size_t tokens,
                                                MpsBuffer& conv_input) const;
  [[nodiscard]] Status ssm_conv1_f32_stateful(MpsBuffer& state,
                                              const MpsBuffer& input,
                                              const MpsBuffer& kernel,
                                              std::size_t conv_kernel,
                                              std::size_t channels,
                                              MpsBuffer& output) const;
  [[nodiscard]] Status ssm_conv1_f32_stateful_at(MpsBuffer& state,
                                                 std::size_t state_offset,
                                                 const MpsBuffer& input,
                                                 const MpsBuffer& kernel,
                                                 std::size_t conv_kernel,
                                                 std::size_t channels,
                                                 MpsBuffer& output) const;
  [[nodiscard]] Status rope_f32(MpsBuffer& values, std::size_t heads,
                                std::size_t head_dim, std::size_t position,
                                float theta) const;
  [[nodiscard]] Status mrope_f32_in_place(MpsBuffer& values,
                                          const MpsBuffer& positions,
                                          std::size_t tokens,
                                          std::size_t heads,
                                          std::size_t head_dim,
                                          std::size_t n_dims,
                                          std::size_t section_0,
                                          std::size_t section_1,
                                          std::size_t section_2,
                                          std::size_t section_3,
                                          float theta) const;
  [[nodiscard]] Status add_f32_in_place(MpsBuffer& target, const MpsBuffer& delta,
                                        std::size_t size) const;
  [[nodiscard]] Status mul_f32_in_place(MpsBuffer& target, const MpsBuffer& rhs,
                                        std::size_t size) const;
  [[nodiscard]] Status add_f32_row_in_place(MpsBuffer& target,
                                            const MpsBuffer& row,
                                            std::size_t rows,
                                            std::size_t row_size) const;
  [[nodiscard]] Status mul_f32_row_in_place(MpsBuffer& target,
                                            const MpsBuffer& row,
                                            std::size_t rows,
                                            std::size_t row_size) const;
  [[nodiscard]] Status sigmoid_f32_in_place(MpsBuffer& values, std::size_t size) const;
  [[nodiscard]] Status softplus_f32_in_place(MpsBuffer& values, std::size_t size) const;
  [[nodiscard]] Status prepare_qwen35_gdn_gate_beta_f32(
    MpsBuffer& gate, MpsBuffer& beta, const MpsBuffer& gate_bias,
    const MpsBuffer& gate_scale, std::size_t rows, std::size_t row_size) const;
  [[nodiscard]] Status silu_f32_in_place(MpsBuffer& values, std::size_t size) const;
  [[nodiscard]] Status silu_mul_f32_in_place(MpsBuffer& gate, const MpsBuffer& up,
                                             std::size_t size) const;
  [[nodiscard]] Status copy_f32_region(const MpsBuffer& source, MpsBuffer& destination,
                                       std::size_t source_offset,
                                       std::size_t destination_offset,
                                       std::size_t size) const;
  [[nodiscard]] Status copy_f32_rows(const MpsBuffer& source, MpsBuffer& destination,
                                     std::size_t rows, std::size_t row_size,
                                     std::size_t source_stride,
                                     std::size_t source_offset,
                                     std::size_t destination_stride,
                                     std::size_t destination_offset) const;
  [[nodiscard]] Status copy_f32_rows_to_f16(
    const MpsBuffer& source, MpsBuffer& destination, std::size_t rows,
    std::size_t row_size, std::size_t source_stride,
    std::size_t source_offset, std::size_t destination_stride,
    std::size_t destination_offset) const;
  [[nodiscard]] Status copy_f16_rows(const MpsBuffer& source,
                                     MpsBuffer& destination,
                                     std::size_t rows, std::size_t row_size,
                                     std::size_t source_stride,
                                     std::size_t source_offset,
                                     std::size_t destination_stride,
                                     std::size_t destination_offset) const;
  [[nodiscard]] Status argmax_f32_i32(const MpsBuffer& values,
                                      std::size_t size,
                                      MpsBuffer& output) const;
  [[nodiscard]] Status argmax_prob_f32_i32(const MpsBuffer& values,
                                           std::size_t size,
                                           MpsBuffer& token_output,
                                           MpsBuffer& probability_output) const;
  [[nodiscard]] Status attention_f32(const MpsBuffer& query,
                                     const MpsBuffer& key_cache,
                                     const MpsBuffer& value_cache,
                                     std::size_t layer, std::size_t position,
                                     std::size_t capacity_tokens,
                                     std::size_t heads, std::size_t kv_heads,
                                     std::size_t head_dim,
                                     MpsBuffer& output) const;
  [[nodiscard]] Status attention_f32_f16_kv(
    const MpsBuffer& query, const MpsBuffer& key_cache,
    const MpsBuffer& value_cache, std::size_t layer, std::size_t position,
    std::size_t capacity_tokens, std::size_t heads, std::size_t kv_heads,
    std::size_t head_dim, MpsBuffer& output) const;
  [[nodiscard]] Status attention_f32_batched(const MpsBuffer& query,
                                             const MpsBuffer& key_cache,
                                             const MpsBuffer& value_cache,
                                             std::size_t layer,
                                             std::size_t start_position,
                                             std::size_t tokens,
                                             std::size_t capacity_tokens,
                                             std::size_t heads,
                                             std::size_t kv_heads,
                                             std::size_t head_dim,
                                             MpsBuffer& output) const;
  [[nodiscard]] Status attention_f32_batched_f16_kv(
    const MpsBuffer& query, const MpsBuffer& key_cache,
    const MpsBuffer& value_cache, std::size_t layer,
    std::size_t start_position, std::size_t tokens,
    std::size_t capacity_tokens, std::size_t heads, std::size_t kv_heads,
    std::size_t head_dim, MpsBuffer& output) const;

 private:
  struct Impl;

  explicit MpsContext(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] BackendInfo query_backend();
[[nodiscard]] std::string format_backend_info(const BackendInfo& info);
[[nodiscard]] Status run_operator_smoke_test();

}  // namespace toyllm::mps
