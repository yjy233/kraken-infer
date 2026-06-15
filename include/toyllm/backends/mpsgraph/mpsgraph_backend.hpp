#pragma once

#include "toyllm/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace toyllm::mpsgraph {

struct BackendInfo {
  bool compiled{false};
  bool available{false};
  bool graph_ready{false};
  std::string device_name;
  std::uint64_t recommended_max_working_set_size{0};
  bool low_power{false};
  bool headless{false};
  bool removable{false};
  std::string failure_reason;
};

struct MpsGraphTransferStats {
  std::uint64_t host_to_device_calls{0};
  std::uint64_t host_to_device_bytes{0};
  std::uint64_t device_to_host_calls{0};
  std::uint64_t device_to_host_bytes{0};
};

struct MpsGraphGraphStats {
  std::uint64_t graph_build_calls{0};
  std::uint64_t graph_build_ns{0};
  std::uint64_t graph_compile_calls{0};
  std::uint64_t graph_compile_ns{0};
  std::uint64_t graph_execute_calls{0};
  std::uint64_t graph_execute_ns{0};
  std::uint64_t executable_cache_hits{0};
  std::uint64_t executable_cache_misses{0};
};

class MpsGraphBuffer {
 public:
  MpsGraphBuffer();
  ~MpsGraphBuffer();

  MpsGraphBuffer(const MpsGraphBuffer&) = delete;
  MpsGraphBuffer& operator=(const MpsGraphBuffer&) = delete;

  MpsGraphBuffer(MpsGraphBuffer&& other) noexcept;
  MpsGraphBuffer& operator=(MpsGraphBuffer&& other) noexcept;

  [[nodiscard]] bool valid() const;
  [[nodiscard]] std::size_t byte_size() const;

 private:
  friend class MpsGraphContext;
  struct Impl;

  explicit MpsGraphBuffer(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

class MpsGraphContext {
 public:
  MpsGraphContext();
  ~MpsGraphContext();

  MpsGraphContext(const MpsGraphContext&) = delete;
  MpsGraphContext& operator=(const MpsGraphContext&) = delete;

  MpsGraphContext(MpsGraphContext&& other) noexcept;
  MpsGraphContext& operator=(MpsGraphContext&& other) noexcept;

  [[nodiscard]] static Result<MpsGraphContext> create();
  [[nodiscard]] bool valid() const;
  [[nodiscard]] MpsGraphTransferStats transfer_stats() const;
  [[nodiscard]] MpsGraphGraphStats graph_stats() const;

  [[nodiscard]] Result<MpsGraphBuffer> make_buffer(std::size_t byte_size) const;
  [[nodiscard]] Status copy_to_buffer(MpsGraphBuffer& buffer, const void* data,
                                      std::size_t byte_size) const;
  [[nodiscard]] Status copy_to_buffer_at(MpsGraphBuffer& buffer,
                                         std::size_t byte_offset,
                                         const void* data,
                                         std::size_t byte_size) const;
  [[nodiscard]] Status copy_from_buffer(const MpsGraphBuffer& buffer, void* data,
                                        std::size_t byte_size) const;

  [[nodiscard]] Status embedding_f32(const MpsGraphBuffer& weight,
                                     std::size_t vocab_size,
                                     std::size_t hidden_size,
                                     std::int64_t token,
                                     MpsGraphBuffer& output) const;
  [[nodiscard]] Status embedding_from_token_f32(const MpsGraphBuffer& weight,
                                                std::size_t vocab_size,
                                                std::size_t hidden_size,
                                                const MpsGraphBuffer& token,
                                                MpsGraphBuffer& output) const;
  [[nodiscard]] Status rms_norm_f32(const MpsGraphBuffer& input,
                                    const MpsGraphBuffer& weight,
                                    std::size_t size, float eps,
                                    MpsGraphBuffer& output) const;
  [[nodiscard]] Status qk_norm_f32(const MpsGraphBuffer& input,
                                   const MpsGraphBuffer& weight,
                                   std::size_t heads, std::size_t head_dim,
                                   float eps,
                                   MpsGraphBuffer& output) const;
  [[nodiscard]] Status rope_f32(const MpsGraphBuffer& input,
                                std::size_t heads, std::size_t head_dim,
                                std::size_t position, float theta,
                                MpsGraphBuffer& output) const;
  [[nodiscard]] Status qk_norm_rope_f32(const MpsGraphBuffer& q_input,
                                        const MpsGraphBuffer& k_input,
                                        const MpsGraphBuffer& q_weight,
                                        const MpsGraphBuffer& k_weight,
                                        std::size_t q_heads,
                                        std::size_t kv_heads,
                                        std::size_t head_dim,
                                        std::size_t position,
                                        float eps, float theta,
                                        MpsGraphBuffer& q_output,
                                        MpsGraphBuffer& k_output) const;
  [[nodiscard]] Status matvec_f32(const MpsGraphBuffer& weight,
                                  std::size_t rows, std::size_t cols,
                                  const MpsGraphBuffer& input,
                                  MpsGraphBuffer& output) const;
  [[nodiscard]] Status qkv_matvec_f32(const MpsGraphBuffer& q_weight,
                                      const MpsGraphBuffer& k_weight,
                                      const MpsGraphBuffer& v_weight,
                                      std::size_t q_rows, std::size_t kv_rows,
                                      std::size_t cols,
                                      const MpsGraphBuffer& input,
                                      MpsGraphBuffer& q_output,
                                      MpsGraphBuffer& k_output,
                                      MpsGraphBuffer& v_output) const;
  [[nodiscard]] Status input_norm_qkv_qk_rope_f32(
    const MpsGraphBuffer& hidden,
    const MpsGraphBuffer& input_norm_weight,
    const MpsGraphBuffer& q_weight,
    const MpsGraphBuffer& k_weight,
    const MpsGraphBuffer& v_weight,
    const MpsGraphBuffer& q_norm_weight,
    const MpsGraphBuffer& k_norm_weight,
    std::size_t hidden_size,
    std::size_t q_heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    std::size_t position,
    float eps,
    float theta,
    MpsGraphBuffer& normed_output,
    MpsGraphBuffer& q_output,
    MpsGraphBuffer& k_output,
    MpsGraphBuffer& v_output) const;
  [[nodiscard]] Status gate_up_matvec_f32(const MpsGraphBuffer& gate_weight,
                                          const MpsGraphBuffer& up_weight,
                                          std::size_t rows, std::size_t cols,
                                          const MpsGraphBuffer& input,
                                          MpsGraphBuffer& gate_output,
                                          MpsGraphBuffer& up_output) const;
  [[nodiscard]] Status attn_project_residual_norm_f32(
    const MpsGraphBuffer& o_weight,
    const MpsGraphBuffer& attn_output,
    const MpsGraphBuffer& residual,
    const MpsGraphBuffer& norm_weight,
    std::size_t hidden_size,
    std::size_t attn_dim,
    float eps,
    MpsGraphBuffer& residual_output,
    MpsGraphBuffer& norm_output) const;
  [[nodiscard]] Status silu_mul_f32(const MpsGraphBuffer& gate,
                                    const MpsGraphBuffer& up,
                                    std::size_t size,
                                    MpsGraphBuffer& output) const;
  [[nodiscard]] Status swiglu_down_residual_f32(const MpsGraphBuffer& gate,
                                                const MpsGraphBuffer& up,
                                                const MpsGraphBuffer& down_weight,
                                                const MpsGraphBuffer& residual,
                                                std::size_t hidden_size,
                                                std::size_t intermediate_size,
                                                MpsGraphBuffer& output) const;
  [[nodiscard]] Status add_f32(const MpsGraphBuffer& lhs,
                               const MpsGraphBuffer& rhs,
                               std::size_t size,
                               MpsGraphBuffer& output) const;
  [[nodiscard]] Status argmax_i32(const MpsGraphBuffer& input,
                                  std::size_t size,
                                  MpsGraphBuffer& output) const;
  [[nodiscard]] Status write_i32_token(const MpsGraphBuffer& token,
                                       MpsGraphBuffer& output,
                                       std::size_t index,
                                       std::size_t capacity) const;
  [[nodiscard]] Status reset_generation_status_i32(MpsGraphBuffer& status) const;
  [[nodiscard]] Status update_generation_status_i32(
    const MpsGraphBuffer& token, const std::int64_t* eos_tokens,
    std::size_t eos_token_count, std::size_t step, bool final_step,
    MpsGraphBuffer& status) const;
  [[nodiscard]] Status write_kv_cache_f32(const MpsGraphBuffer& source,
                                          MpsGraphBuffer& cache,
                                          std::size_t layer,
                                          std::size_t position,
                                          std::size_t layers,
                                          std::size_t capacity_tokens,
                                          std::size_t kv_heads,
                                          std::size_t head_dim) const;
  [[nodiscard]] Status write_kv_cache_pair_f32(const MpsGraphBuffer& key_source,
                                               const MpsGraphBuffer& value_source,
                                               MpsGraphBuffer& key_cache,
                                               MpsGraphBuffer& value_cache,
                                               std::size_t layer,
                                               std::size_t position,
                                               std::size_t layers,
                                               std::size_t capacity_tokens,
                                               std::size_t kv_heads,
                                               std::size_t head_dim) const;
  [[nodiscard]] Status attention_f32(const MpsGraphBuffer& query,
                                     const MpsGraphBuffer& key_cache,
                                     const MpsGraphBuffer& value_cache,
                                     std::size_t layer, std::size_t position,
                                     std::size_t capacity_tokens,
                                     std::size_t heads, std::size_t kv_heads,
                                     std::size_t head_dim,
                                     MpsGraphBuffer& output) const;

 private:
  struct Impl;

  explicit MpsGraphContext(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] BackendInfo query_backend();
[[nodiscard]] std::string format_backend_info(const BackendInfo& info);
[[nodiscard]] Status run_operator_smoke_test();

}  // namespace toyllm::mpsgraph
