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

Status MpsContext::add_f32_in_place(MpsBuffer& target, const MpsBuffer& delta,
                                    std::size_t size) const {
  (void)target;
  (void)delta;
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
