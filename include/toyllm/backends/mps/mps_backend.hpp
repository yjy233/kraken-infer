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

  [[nodiscard]] Result<MpsBuffer> make_buffer(std::size_t byte_size) const;
  [[nodiscard]] Status copy_to_buffer(MpsBuffer& buffer, const void* data,
                                      std::size_t byte_size) const;
  [[nodiscard]] Status copy_from_buffer(const MpsBuffer& buffer, void* data,
                                        std::size_t byte_size) const;
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
  [[nodiscard]] Status embedding_bf16_f32(const MpsBuffer& weight, std::int64_t token,
                                          std::size_t hidden_size,
                                          MpsBuffer& output) const;
  [[nodiscard]] Status rms_norm_f32_bf16(const MpsBuffer& input,
                                         const MpsBuffer& weight,
                                         std::size_t size, float eps,
                                         MpsBuffer& output) const;
  [[nodiscard]] Status qk_norm_f32_bf16(MpsBuffer& values,
                                        const MpsBuffer& weight,
                                        std::size_t heads, std::size_t head_dim,
                                        float eps) const;
  [[nodiscard]] Status rope_f32(MpsBuffer& values, std::size_t heads,
                                std::size_t head_dim, std::size_t position,
                                float theta) const;
  [[nodiscard]] Status add_f32_in_place(MpsBuffer& target, const MpsBuffer& delta,
                                        std::size_t size) const;
  [[nodiscard]] Status silu_mul_f32_in_place(MpsBuffer& gate, const MpsBuffer& up,
                                             std::size_t size) const;
  [[nodiscard]] Status copy_f32_region(const MpsBuffer& source, MpsBuffer& destination,
                                       std::size_t source_offset,
                                       std::size_t destination_offset,
                                       std::size_t size) const;
  [[nodiscard]] Status attention_f32(const MpsBuffer& query,
                                     const MpsBuffer& key_cache,
                                     const MpsBuffer& value_cache,
                                     std::size_t layer, std::size_t position,
                                     std::size_t capacity_tokens,
                                     std::size_t heads, std::size_t kv_heads,
                                     std::size_t head_dim,
                                     MpsBuffer& output) const;

 private:
  struct Impl;

  explicit MpsContext(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] BackendInfo query_backend();
[[nodiscard]] std::string format_backend_info(const BackendInfo& info);
[[nodiscard]] Status run_operator_smoke_test();

}  // namespace toyllm::mps
