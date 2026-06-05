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

Result<std::vector<float>> MpsContext::matvec_bf16_f32(
  const MpsBuffer& weight, std::size_t rows, std::size_t cols,
  const std::vector<float>& input) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)input;
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
