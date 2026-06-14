#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"

#include <sstream>
#include <utility>

namespace toyllm::mpsgraph {

namespace {

constexpr const char* kUnavailable = "MPSGraph backend was not compiled for this build";

}  // namespace

struct MpsGraphBuffer::Impl {};
struct MpsGraphContext::Impl {};

MpsGraphBuffer::MpsGraphBuffer() = default;
MpsGraphBuffer::~MpsGraphBuffer() = default;
MpsGraphBuffer::MpsGraphBuffer(MpsGraphBuffer&& other) noexcept = default;
MpsGraphBuffer& MpsGraphBuffer::operator=(MpsGraphBuffer&& other) noexcept = default;
MpsGraphBuffer::MpsGraphBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool MpsGraphBuffer::valid() const {
  return false;
}

std::size_t MpsGraphBuffer::byte_size() const {
  return 0;
}

MpsGraphContext::MpsGraphContext() = default;
MpsGraphContext::~MpsGraphContext() = default;
MpsGraphContext::MpsGraphContext(MpsGraphContext&& other) noexcept = default;
MpsGraphContext& MpsGraphContext::operator=(MpsGraphContext&& other) noexcept = default;
MpsGraphContext::MpsGraphContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Result<MpsGraphContext> MpsGraphContext::create() {
  return Status::unavailable(kUnavailable);
}

bool MpsGraphContext::valid() const {
  return false;
}

Result<MpsGraphBuffer> MpsGraphContext::make_buffer(std::size_t byte_size) const {
  (void)byte_size;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::copy_to_buffer(MpsGraphBuffer& buffer, const void* data,
                                       std::size_t byte_size) const {
  (void)buffer;
  (void)data;
  (void)byte_size;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::copy_to_buffer_at(MpsGraphBuffer& buffer,
                                          std::size_t byte_offset,
                                          const void* data,
                                          std::size_t byte_size) const {
  (void)buffer;
  (void)byte_offset;
  (void)data;
  (void)byte_size;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::copy_from_buffer(const MpsGraphBuffer& buffer, void* data,
                                         std::size_t byte_size) const {
  (void)buffer;
  (void)data;
  (void)byte_size;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::embedding_f32(const MpsGraphBuffer& weight,
                                      std::size_t vocab_size,
                                      std::size_t hidden_size,
                                      std::int64_t token,
                                      MpsGraphBuffer& output) const {
  (void)weight;
  (void)vocab_size;
  (void)hidden_size;
  (void)token;
  (void)output;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::rms_norm_f32(const MpsGraphBuffer& input,
                                     const MpsGraphBuffer& weight,
                                     std::size_t size, float eps,
                                     MpsGraphBuffer& output) const {
  (void)input;
  (void)weight;
  (void)size;
  (void)eps;
  (void)output;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::qk_norm_f32(const MpsGraphBuffer& input,
                                    const MpsGraphBuffer& weight,
                                    std::size_t heads, std::size_t head_dim,
                                    float eps,
                                    MpsGraphBuffer& output) const {
  (void)input;
  (void)weight;
  (void)heads;
  (void)head_dim;
  (void)eps;
  (void)output;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::rope_f32(const MpsGraphBuffer& input,
                                 std::size_t heads, std::size_t head_dim,
                                 std::size_t position, float theta,
                                 MpsGraphBuffer& output) const {
  (void)input;
  (void)heads;
  (void)head_dim;
  (void)position;
  (void)theta;
  (void)output;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::matvec_f32(const MpsGraphBuffer& weight,
                                   std::size_t rows, std::size_t cols,
                                   const MpsGraphBuffer& input,
                                   MpsGraphBuffer& output) const {
  (void)weight;
  (void)rows;
  (void)cols;
  (void)input;
  (void)output;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::silu_mul_f32(const MpsGraphBuffer& gate,
                                     const MpsGraphBuffer& up,
                                     std::size_t size,
                                     MpsGraphBuffer& output) const {
  (void)gate;
  (void)up;
  (void)size;
  (void)output;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::add_f32(const MpsGraphBuffer& lhs,
                                const MpsGraphBuffer& rhs,
                                std::size_t size,
                                MpsGraphBuffer& output) const {
  (void)lhs;
  (void)rhs;
  (void)size;
  (void)output;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::argmax_i32(const MpsGraphBuffer& input,
                                   std::size_t size,
                                   MpsGraphBuffer& output) const {
  (void)input;
  (void)size;
  (void)output;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::write_kv_cache_f32(const MpsGraphBuffer& source,
                                           MpsGraphBuffer& cache,
                                           std::size_t layer,
                                           std::size_t position,
                                           std::size_t layers,
                                           std::size_t capacity_tokens,
                                           std::size_t kv_heads,
                                           std::size_t head_dim) const {
  (void)source;
  (void)cache;
  (void)layer;
  (void)position;
  (void)layers;
  (void)capacity_tokens;
  (void)kv_heads;
  (void)head_dim;
  return Status::unavailable(kUnavailable);
}

Status MpsGraphContext::attention_f32(const MpsGraphBuffer& query,
                                      const MpsGraphBuffer& key_cache,
                                      const MpsGraphBuffer& value_cache,
                                      std::size_t layer, std::size_t position,
                                      std::size_t capacity_tokens,
                                      std::size_t heads, std::size_t kv_heads,
                                      std::size_t head_dim,
                                      MpsGraphBuffer& output) const {
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
  return Status::unavailable(kUnavailable);
}

BackendInfo query_backend() {
  BackendInfo info{};
  info.failure_reason = kUnavailable;
  return info;
}

std::string format_backend_info(const BackendInfo& info) {
  std::ostringstream output;
  output << "MPSGraph compiled: " << (info.compiled ? "yes" : "no") << '\n';
  output << "MPSGraph available: " << (info.available ? "yes" : "no") << '\n';
  output << "MPSGraph ready: " << (info.graph_ready ? "yes" : "no") << '\n';
  if (!info.device_name.empty()) {
    output << "Device: " << info.device_name << '\n';
  }
  if (!info.failure_reason.empty()) {
    output << "Reason: " << info.failure_reason << '\n';
  }
  return output.str();
}

Status run_operator_smoke_test() {
  return Status::unavailable(kUnavailable);
}

}  // namespace toyllm::mpsgraph
