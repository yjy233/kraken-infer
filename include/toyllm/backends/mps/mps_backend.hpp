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
  [[nodiscard]] Result<std::vector<float>> matvec_bf16_f32(const MpsBuffer& weight,
                                                           std::size_t rows,
                                                           std::size_t cols,
                                                           const std::vector<float>& input) const;

 private:
  struct Impl;

  explicit MpsContext(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] BackendInfo query_backend();
[[nodiscard]] std::string format_backend_info(const BackendInfo& info);
[[nodiscard]] Status run_operator_smoke_test();

}  // namespace toyllm::mps
