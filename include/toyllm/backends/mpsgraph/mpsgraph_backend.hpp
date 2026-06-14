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
  [[nodiscard]] Status matvec_f32(const MpsGraphBuffer& weight,
                                  std::size_t rows, std::size_t cols,
                                  const MpsGraphBuffer& input,
                                  MpsGraphBuffer& output) const;
  [[nodiscard]] Status silu_mul_f32(const MpsGraphBuffer& gate,
                                    const MpsGraphBuffer& up,
                                    std::size_t size,
                                    MpsGraphBuffer& output) const;
  [[nodiscard]] Status add_f32(const MpsGraphBuffer& lhs,
                               const MpsGraphBuffer& rhs,
                               std::size_t size,
                               MpsGraphBuffer& output) const;
  [[nodiscard]] Status write_kv_cache_f32(const MpsGraphBuffer& source,
                                          MpsGraphBuffer& cache,
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
