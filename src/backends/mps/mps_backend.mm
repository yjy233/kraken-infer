#include "toyllm/backends/mps/mps_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace toyllm::mps {

namespace {

constexpr const char* kKernelSource = R"metal(
#include <metal_stdlib>
using namespace metal;

static inline float bf16_to_float(ushort value) {
  return as_type<float>(static_cast<uint>(value) << 16);
}

struct SizeParams {
  uint size;
};

struct EmbeddingParams {
  uint token;
  uint hidden_size;
};

struct RmsNormParams {
  uint size;
  float eps;
  uint threads;
};

struct QkNormParams {
  uint heads;
  uint head_dim;
  float eps;
  uint threads;
};

struct RopeParams {
  uint heads;
  uint head_dim;
  uint position;
  float theta;
};

struct CopyRegionParams {
  uint source_offset;
  uint destination_offset;
  uint size;
};

struct AttentionParams {
  uint layer;
  uint position;
  uint capacity_tokens;
  uint heads;
  uint kv_heads;
  uint head_dim;
  uint group;
  float scale;
};

kernel void bf16_matvec(const device ushort* weight [[buffer(0)]],
                        const device float* input [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant uint& cols [[buffer(3)]],
                        constant uint& threads_per_row [[buffer(4)]],
                        uint3 group_id [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint row = group_id.x;
  float sum = 0.0f;
  const uint row_offset = row * cols;
  for (uint col = lane; col < cols; col += threads_per_row) {
    sum += bf16_to_float(weight[row_offset + col]) * input[col];
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads_per_row >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    output[row] = partials[0];
  }
}

kernel void embedding_bf16_f32(const device ushort* weight [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant EmbeddingParams& params [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  if (index >= params.hidden_size) {
    return;
  }
  output[index] = bf16_to_float(weight[params.token * params.hidden_size + index]);
}

kernel void rms_norm_f32_bf16(const device float* input [[buffer(0)]],
                              const device ushort* weight [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant RmsNormParams& params [[buffer(3)]],
                              uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  float sum = 0.0f;
  for (uint index = lane; index < params.size; index += params.threads) {
    const float value = input[index];
    sum += value * value;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale = rsqrt(partials[0] / static_cast<float>(params.size) + params.eps);
  for (uint index = lane; index < params.size; index += params.threads) {
    output[index] = input[index] * scale * bf16_to_float(weight[index]);
  }
}

kernel void qk_norm_f32_bf16(device float* values [[buffer(0)]],
                             const device ushort* weight [[buffer(1)]],
                             constant QkNormParams& params [[buffer(2)]],
                             uint3 group_id [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint head = group_id.x;
  if (head >= params.heads) {
    return;
  }

  const uint base = head * params.head_dim;
  float sum = 0.0f;
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    const float value = values[base + dim];
    sum += value * value;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale =
    rsqrt(partials[0] / static_cast<float>(params.head_dim) + params.eps);
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    values[base + dim] *= scale * bf16_to_float(weight[dim]);
  }
}

kernel void rope_f32(device float* values [[buffer(0)]],
                     constant RopeParams& params [[buffer(1)]],
                     uint index [[thread_position_in_grid]]) {
  const uint half_dim = params.head_dim >> 1U;
  const uint total = params.heads * half_dim;
  if (index >= total) {
    return;
  }

  const uint head = index / half_dim;
  const uint dim = index % half_dim;
  const uint base = head * params.head_dim;
  const float exponent = static_cast<float>(2U * dim) / static_cast<float>(params.head_dim);
  const float freq = 1.0f / pow(params.theta, exponent);
  const float angle = static_cast<float>(params.position) * freq;
  const float cos_value = cos(angle);
  const float sin_value = sin(angle);
  const float x0 = values[base + dim];
  const float x1 = values[base + half_dim + dim];
  values[base + dim] = x0 * cos_value - x1 * sin_value;
  values[base + half_dim + dim] = x1 * cos_value + x0 * sin_value;
}

kernel void add_f32_in_place(device float* target [[buffer(0)]],
                             const device float* delta [[buffer(1)]],
                             constant SizeParams& params [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
  if (index >= params.size) {
    return;
  }
  target[index] += delta[index];
}

kernel void silu_mul_f32_in_place(device float* gate [[buffer(0)]],
                                  const device float* up [[buffer(1)]],
                                  constant SizeParams& params [[buffer(2)]],
                                  uint index [[thread_position_in_grid]]) {
  if (index >= params.size) {
    return;
  }
  const float value = gate[index];
  gate[index] = (value / (1.0f + exp(-value))) * up[index];
}

kernel void copy_f32_region(const device float* source [[buffer(0)]],
                            device float* destination [[buffer(1)]],
                            constant CopyRegionParams& params [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  if (index >= params.size) {
    return;
  }
  destination[params.destination_offset + index] =
    source[params.source_offset + index];
}

kernel void attention_f32(const device float* query [[buffer(0)]],
                          const device float* key_cache [[buffer(1)]],
                          const device float* value_cache [[buffer(2)]],
                          device float* output [[buffer(3)]],
                          constant AttentionParams& params [[buffer(4)]],
                          uint index [[thread_position_in_grid]]) {
  const uint head_dim = params.head_dim;
  const uint total = params.heads * head_dim;
  if (index >= total) {
    return;
  }

  const uint head = index / head_dim;
  const uint dim = index % head_dim;
  const uint kv_head = head / params.group;
  const uint kv_dim = params.kv_heads * head_dim;
  float max_score = -INFINITY;

  for (uint token = 0; token <= params.position; ++token) {
    float score = 0.0f;
    const uint query_base = head * head_dim;
    const uint cache_base =
      ((params.layer * params.capacity_tokens + token) * kv_dim) +
      kv_head * head_dim;
    for (uint value_dim = 0; value_dim < head_dim; ++value_dim) {
      score += query[query_base + value_dim] * key_cache[cache_base + value_dim];
    }
    score *= params.scale;
    max_score = max(max_score, score);
  }

  float denom = 0.0f;
  float value = 0.0f;
  for (uint token = 0; token <= params.position; ++token) {
    float score = 0.0f;
    const uint query_base = head * head_dim;
    const uint cache_base =
      ((params.layer * params.capacity_tokens + token) * kv_dim) +
      kv_head * head_dim;
    for (uint value_dim = 0; value_dim < head_dim; ++value_dim) {
      score += query[query_base + value_dim] * key_cache[cache_base + value_dim];
    }
    const float weight = exp(score * params.scale - max_score);
    denom += weight;
    value += weight * value_cache[cache_base + dim];
  }
  output[index] = value / denom;
}
)metal";

std::string yes_no(bool value) {
  return value ? "yes" : "no";
}

std::string nsstring_to_string(NSString* value) {
  if (value == nil) {
    return {};
  }
  const char* utf8 = [value UTF8String];
  return utf8 == nullptr ? std::string{} : std::string{utf8};
}

std::string nserror_to_string(NSError* error) {
  if (error == nil) {
    return "unknown Metal error";
  }
  return nsstring_to_string([error localizedDescription]);
}

bool fits_nsuinteger(std::size_t value) {
  return value <= static_cast<std::size_t>(std::numeric_limits<NSUInteger>::max());
}

bool fits_uint32(std::size_t value) {
  return value <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t& result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16U);
}

NSUInteger choose_threadgroup_width(NSUInteger max_threads) {
  const NSUInteger limit = static_cast<NSUInteger>(256);
  const NSUInteger capped = std::min(max_threads, limit);
  NSUInteger width = static_cast<NSUInteger>(1);
  while (width < capped && width <= capped / static_cast<NSUInteger>(2)) {
    width *= static_cast<NSUInteger>(2);
  }
  return width;
}

struct MatVecLayout {
  std::size_t elements{0};
  std::size_t weight_bytes{0};
  std::size_t input_bytes{0};
  std::size_t output_bytes{0};
};

struct SizeParams {
  std::uint32_t size{0};
};

struct EmbeddingParams {
  std::uint32_t token{0};
  std::uint32_t hidden_size{0};
};

struct RmsNormParams {
  std::uint32_t size{0};
  float eps{0.0F};
  std::uint32_t threads{0};
};

struct QkNormParams {
  std::uint32_t heads{0};
  std::uint32_t head_dim{0};
  float eps{0.0F};
  std::uint32_t threads{0};
};

struct RopeParams {
  std::uint32_t heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t position{0};
  float theta{0.0F};
};

struct CopyRegionParams {
  std::uint32_t source_offset{0};
  std::uint32_t destination_offset{0};
  std::uint32_t size{0};
};

struct AttentionParams {
  std::uint32_t layer{0};
  std::uint32_t position{0};
  std::uint32_t capacity_tokens{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t group{0};
  float scale{0.0F};
};

Result<MatVecLayout> make_matvec_layout(std::size_t rows, std::size_t cols) {
  if (rows == 0 || cols == 0) {
    return Status::invalid_argument("MPS matvec rows and cols must be greater than zero");
  }
  if (!fits_uint32(cols)) {
    return Status::invalid_argument("MPS matvec cols exceeds uint32 range");
  }
  if (!fits_nsuinteger(rows)) {
    return Status::invalid_argument("MPS matvec rows exceeds NSUInteger range");
  }

  std::size_t elements = 0;
  if (!checked_mul(rows, cols, elements)) {
    return Status::invalid_argument("MPS matvec weight element count overflow");
  }
  if (!fits_uint32(elements)) {
    return Status::invalid_argument("MPS matvec weight element count exceeds kernel range");
  }

  std::size_t weight_bytes = 0;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  if (!checked_mul(elements, sizeof(std::uint16_t), weight_bytes)) {
    return Status::invalid_argument("MPS matvec weight byte count overflow");
  }
  if (!checked_mul(cols, sizeof(float), input_bytes) ||
      !checked_mul(rows, sizeof(float), output_bytes)) {
    return Status::invalid_argument("MPS matvec buffer byte count overflow");
  }
  return MatVecLayout{elements, weight_bytes, input_bytes, output_bytes};
}

Result<std::uint32_t> checked_u32(std::size_t value, const char* name) {
  if (!fits_uint32(value)) {
    return Status::invalid_argument(std::string{"MPS "} + name + " exceeds uint32 range");
  }
  return static_cast<std::uint32_t>(value);
}

Status validate_f32_buffer(const MpsBuffer& buffer, std::size_t values,
                           const char* name) {
  std::size_t bytes = 0;
  if (!checked_mul(values, sizeof(float), bytes)) {
    return Status::invalid_argument(std::string{"MPS "} + name + " byte count overflow");
  }
  if (!buffer.valid()) {
    return Status::invalid_argument(std::string{"MPS "} + name + " buffer is not initialized");
  }
  if (buffer.byte_size() < bytes) {
    return Status::invalid_argument(std::string{"MPS "} + name + " buffer is too small");
  }
  return Status::ok();
}

}  // namespace

struct MpsBuffer::Impl {
  id<MTLBuffer> buffer{nil};
  std::size_t byte_size{0};

  ~Impl() {
    if (buffer != nil) {
      [buffer release];
    }
  }
};

struct MpsContext::Impl {
  id<MTLDevice> device{nil};
  id<MTLCommandQueue> queue{nil};
  id<MTLComputePipelineState> matvec_pipeline{nil};
  id<MTLComputePipelineState> embedding_pipeline{nil};
  id<MTLComputePipelineState> rms_norm_pipeline{nil};
  id<MTLComputePipelineState> qk_norm_pipeline{nil};
  id<MTLComputePipelineState> rope_pipeline{nil};
  id<MTLComputePipelineState> add_pipeline{nil};
  id<MTLComputePipelineState> silu_mul_pipeline{nil};
  id<MTLComputePipelineState> copy_region_pipeline{nil};
  id<MTLComputePipelineState> attention_pipeline{nil};

  ~Impl() {
    if (attention_pipeline != nil) {
      [attention_pipeline release];
    }
    if (copy_region_pipeline != nil) {
      [copy_region_pipeline release];
    }
    if (silu_mul_pipeline != nil) {
      [silu_mul_pipeline release];
    }
    if (add_pipeline != nil) {
      [add_pipeline release];
    }
    if (rope_pipeline != nil) {
      [rope_pipeline release];
    }
    if (qk_norm_pipeline != nil) {
      [qk_norm_pipeline release];
    }
    if (rms_norm_pipeline != nil) {
      [rms_norm_pipeline release];
    }
    if (embedding_pipeline != nil) {
      [embedding_pipeline release];
    }
    if (matvec_pipeline != nil) {
      [matvec_pipeline release];
    }
    if (queue != nil) {
      [queue release];
    }
    if (device != nil) {
      [device release];
    }
  }
};

MpsBuffer::MpsBuffer() = default;
MpsBuffer::~MpsBuffer() = default;
MpsBuffer::MpsBuffer(MpsBuffer&& other) noexcept = default;
MpsBuffer& MpsBuffer::operator=(MpsBuffer&& other) noexcept = default;
MpsBuffer::MpsBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool MpsBuffer::valid() const {
  return impl_ != nullptr && impl_->buffer != nil;
}

std::size_t MpsBuffer::byte_size() const {
  return impl_ == nullptr ? 0 : impl_->byte_size;
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
  return rows_ > 0 && cols_ > 0 && input_.valid() && output_.valid();
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
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return Status::unavailable("Metal returned no default device");
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      [device release];
      return Status::unavailable("failed to create Metal command queue");
    }

    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->queue = queue;

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:kKernelSource];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
      return Status::internal_error("failed to compile Metal kernels: " +
                                    nserror_to_string(error));
    }

    auto make_pipeline = [&](const char* name) -> Result<id<MTLComputePipelineState>> {
      id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:name]];
      if (function == nil) {
        return Status::internal_error("failed to find Metal kernel " + std::string{name});
      }

      error = nil;
      id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
      [function release];
      if (pipeline == nil) {
        return Status::internal_error("failed to create " + std::string{name} +
                                      " pipeline: " + nserror_to_string(error));
      }
      return pipeline;
    };

    auto matvec_pipeline = make_pipeline("bf16_matvec");
    if (!matvec_pipeline.is_ok()) {
      [library release];
      return matvec_pipeline.status();
    }
    impl->matvec_pipeline = matvec_pipeline.value();

    auto embedding_pipeline = make_pipeline("embedding_bf16_f32");
    if (!embedding_pipeline.is_ok()) {
      [library release];
      return embedding_pipeline.status();
    }
    impl->embedding_pipeline = embedding_pipeline.value();

    auto rms_norm_pipeline = make_pipeline("rms_norm_f32_bf16");
    if (!rms_norm_pipeline.is_ok()) {
      [library release];
      return rms_norm_pipeline.status();
    }
    impl->rms_norm_pipeline = rms_norm_pipeline.value();

    auto qk_norm_pipeline = make_pipeline("qk_norm_f32_bf16");
    if (!qk_norm_pipeline.is_ok()) {
      [library release];
      return qk_norm_pipeline.status();
    }
    impl->qk_norm_pipeline = qk_norm_pipeline.value();

    auto rope_pipeline = make_pipeline("rope_f32");
    if (!rope_pipeline.is_ok()) {
      [library release];
      return rope_pipeline.status();
    }
    impl->rope_pipeline = rope_pipeline.value();

    auto add_pipeline = make_pipeline("add_f32_in_place");
    if (!add_pipeline.is_ok()) {
      [library release];
      return add_pipeline.status();
    }
    impl->add_pipeline = add_pipeline.value();

    auto silu_mul_pipeline = make_pipeline("silu_mul_f32_in_place");
    if (!silu_mul_pipeline.is_ok()) {
      [library release];
      return silu_mul_pipeline.status();
    }
    impl->silu_mul_pipeline = silu_mul_pipeline.value();

    auto copy_region_pipeline = make_pipeline("copy_f32_region");
    if (!copy_region_pipeline.is_ok()) {
      [library release];
      return copy_region_pipeline.status();
    }
    impl->copy_region_pipeline = copy_region_pipeline.value();

    auto attention_pipeline = make_pipeline("attention_f32");
    if (!attention_pipeline.is_ok()) {
      [library release];
      return attention_pipeline.status();
    }
    impl->attention_pipeline = attention_pipeline.value();
    [library release];

    return MpsContext(std::move(impl));
  }
}

bool MpsContext::valid() const {
  return impl_ != nullptr && impl_->device != nil && impl_->queue != nil &&
         impl_->matvec_pipeline != nil && impl_->embedding_pipeline != nil &&
         impl_->rms_norm_pipeline != nil && impl_->qk_norm_pipeline != nil &&
         impl_->rope_pipeline != nil && impl_->add_pipeline != nil &&
         impl_->silu_mul_pipeline != nil && impl_->copy_region_pipeline != nil &&
         impl_->attention_pipeline != nil;
}

Result<MpsBuffer> MpsContext::make_buffer(std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (byte_size == 0) {
    return Status::invalid_argument("MPS buffer size must be greater than zero");
  }
  if (!fits_nsuinteger(byte_size)) {
    return Status::invalid_argument("MPS buffer size exceeds NSUInteger range");
  }

  @autoreleasepool {
    id<MTLBuffer> buffer =
      [impl_->device newBufferWithLength:static_cast<NSUInteger>(byte_size)
                                 options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      return Status::unavailable("failed to allocate Metal buffer");
    }

    auto impl = std::make_unique<MpsBuffer::Impl>();
    impl->buffer = buffer;
    impl->byte_size = byte_size;
    return MpsBuffer(std::move(impl));
  }
}

Status MpsContext::copy_to_buffer(MpsBuffer& buffer, const void* data,
                                  std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!buffer.valid()) {
    return Status::invalid_argument("MPS buffer is not initialized");
  }
  if (data == nullptr) {
    return Status::invalid_argument("MPS copy source must not be null");
  }
  if (byte_size > buffer.byte_size()) {
    return Status::invalid_argument("MPS copy size exceeds destination buffer size");
  }

  std::memcpy([buffer.impl_->buffer contents], data, byte_size);
  return Status::ok();
}

Status MpsContext::copy_from_buffer(const MpsBuffer& buffer, void* data,
                                    std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!buffer.valid()) {
    return Status::invalid_argument("MPS buffer is not initialized");
  }
  if (data == nullptr) {
    return Status::invalid_argument("MPS copy destination must not be null");
  }
  if (byte_size > buffer.byte_size()) {
    return Status::invalid_argument("MPS copy size exceeds source buffer size");
  }

  std::memcpy(data, [buffer.impl_->buffer contents], byte_size);
  return Status::ok();
}

Result<MpsMatVecWorkspace> MpsContext::make_matvec_workspace(std::size_t rows,
                                                             std::size_t cols) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto layout_result = make_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();

  auto input_buffer_result = make_buffer(layout.input_bytes);
  if (!input_buffer_result.is_ok()) {
    return input_buffer_result.status();
  }
  auto input_buffer = std::move(input_buffer_result.value());

  auto output_buffer_result = make_buffer(layout.output_bytes);
  if (!output_buffer_result.is_ok()) {
    return output_buffer_result.status();
  }
  auto output_buffer = std::move(output_buffer_result.value());

  return MpsMatVecWorkspace(rows, cols, std::move(input_buffer), std::move(output_buffer));
}

Result<std::vector<float>> MpsContext::matvec_bf16_f32(const MpsBuffer& weight,
                                                       std::size_t rows,
                                                       std::size_t cols,
                                                       const std::vector<float>& input) const {
  auto workspace_result = make_matvec_workspace(rows, cols);
  if (!workspace_result.is_ok()) {
    return workspace_result.status();
  }
  auto workspace = std::move(workspace_result.value());
  return matvec_bf16_f32(weight, workspace, input);
}

Status MpsContext::matvec_bf16_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS weight buffer is not initialized");
  }

  auto layout_result = make_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  if (weight.byte_size() < layout.weight_bytes) {
    return Status::invalid_argument("MPS matvec weight buffer is smaller than rows * cols");
  }
  auto input_status = validate_f32_buffer(input, cols, "matvec input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, rows, "matvec output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    [encoder setComputePipelineState:impl_->matvec_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    const auto cols_u32 = static_cast<std::uint32_t>(cols);
    [encoder setBytes:&cols_u32 length:sizeof(cols_u32) atIndex:3];
    const NSUInteger max_threads = [impl_->matvec_pipeline maxTotalThreadsPerThreadgroup];
    if (max_threads == 0) {
      return Status::internal_error("MPS matvec pipeline reported zero max threads");
    }
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    const auto threads_per_group_u32 = static_cast<std::uint32_t>(threads_per_group);
    [encoder setBytes:&threads_per_group_u32
               length:sizeof(threads_per_group_u32)
              atIndex:4];

    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error("MPS matvec command failed: " +
                                    nserror_to_string([command_buffer error]));
    }
  }
  return Status::ok();
}

Result<std::vector<float>> MpsContext::matvec_bf16_f32(
  const MpsBuffer& weight, MpsMatVecWorkspace& workspace,
  const std::vector<float>& input) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS weight buffer is not initialized");
  }
  if (!workspace.valid()) {
    return Status::invalid_argument("MPS matvec workspace is not initialized");
  }
  const auto rows = workspace.rows();
  const auto cols = workspace.cols();
  if (input.size() != cols) {
    return Status::invalid_argument("MPS matvec input size does not match cols");
  }

  auto layout_result = make_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  if (weight.byte_size() < layout.weight_bytes) {
    return Status::invalid_argument("MPS matvec weight buffer is smaller than rows * cols");
  }
  if (workspace.input_.byte_size() < layout.input_bytes ||
      workspace.output_.byte_size() < layout.output_bytes) {
    return Status::invalid_argument("MPS matvec workspace buffers are too small");
  }
  const auto copy_status = copy_to_buffer(workspace.input_, input.data(), layout.input_bytes);
  if (!copy_status.is_ok()) {
    return copy_status;
  }

  const auto matvec_status =
    matvec_bf16_f32_device(weight, rows, cols, workspace.input_, workspace.output_);
  if (!matvec_status.is_ok()) {
    return matvec_status;
  }

  std::vector<float> output(rows);
  const auto read_status = copy_from_buffer(workspace.output_, output.data(), layout.output_bytes);
  if (!read_status.is_ok()) {
    return read_status;
  }
  return output;
}

Status MpsContext::embedding_bf16_f32(const MpsBuffer& weight, std::int64_t token,
                                      std::size_t hidden_size,
                                      MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS embedding weight buffer is not initialized");
  }
  if (token < 0) {
    return Status::invalid_argument("MPS embedding token must be non-negative");
  }
  auto hidden_u32 = checked_u32(hidden_size, "embedding hidden size");
  if (!hidden_u32.is_ok()) {
    return hidden_u32.status();
  }
  auto token_u32 = checked_u32(static_cast<std::size_t>(token), "embedding token");
  if (!token_u32.is_ok()) {
    return token_u32.status();
  }
  std::size_t token_end = 0;
  if (!checked_mul(static_cast<std::size_t>(token) + 1U, hidden_size, token_end) ||
      token_end > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
    return Status::invalid_argument("MPS embedding weight byte count overflow");
  }
  if (weight.byte_size() < token_end * sizeof(std::uint16_t)) {
    return Status::invalid_argument("MPS embedding weight buffer is too small");
  }
  auto output_status = validate_f32_buffer(output, hidden_size, "embedding output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    EmbeddingParams params{token_u32.value(), hidden_u32.value()};
    [encoder setComputePipelineState:impl_->embedding_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->embedding_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(hidden_size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error("MPS embedding command failed: " +
                                    nserror_to_string([command_buffer error]));
    }
  }
  return Status::ok();
}

Status MpsContext::rms_norm_f32_bf16(const MpsBuffer& input, const MpsBuffer& weight,
                                     std::size_t size, float eps,
                                     MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "rms norm size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto input_status = validate_f32_buffer(input, size, "rms norm input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, size, "rms norm output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  if (!weight.valid() || weight.byte_size() < size * sizeof(std::uint16_t)) {
    return Status::invalid_argument("MPS rms norm weight buffer is too small");
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->rms_norm_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    RmsNormParams params{size_u32.value(), eps,
                         static_cast<std::uint32_t>(threads_per_group)};
    [encoder setComputePipelineState:impl_->rms_norm_pipeline];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const MTLSize group_count = MTLSizeMake(1, 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error("MPS rms norm command failed: " +
                                    nserror_to_string([command_buffer error]));
    }
  }
  return Status::ok();
}

Status MpsContext::qk_norm_f32_bf16(MpsBuffer& values, const MpsBuffer& weight,
                                    std::size_t heads, std::size_t head_dim,
                                    float eps) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (heads == 0 || head_dim == 0) {
    return Status::invalid_argument("MPS qk norm dimensions must be positive");
  }
  auto heads_u32 = checked_u32(heads, "qk norm heads");
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  auto head_dim_u32 = checked_u32(head_dim, "qk norm head_dim");
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  auto values_status = validate_f32_buffer(values, heads * head_dim, "qk norm values");
  if (!values_status.is_ok()) {
    return values_status;
  }
  if (!weight.valid() || weight.byte_size() < head_dim * sizeof(std::uint16_t)) {
    return Status::invalid_argument("MPS qk norm weight buffer is too small");
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->qk_norm_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    QkNormParams params{heads_u32.value(), head_dim_u32.value(), eps,
                        static_cast<std::uint32_t>(threads_per_group)};
    [encoder setComputePipelineState:impl_->qk_norm_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(heads), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error("MPS qk norm command failed: " +
                                    nserror_to_string([command_buffer error]));
    }
  }
  return Status::ok();
}

Status MpsContext::rope_f32(MpsBuffer& values, std::size_t heads,
                            std::size_t head_dim, std::size_t position,
                            float theta) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (heads == 0 || head_dim == 0 || head_dim % 2U != 0) {
    return Status::invalid_argument("MPS RoPE dimensions are invalid");
  }
  auto heads_u32 = checked_u32(heads, "RoPE heads");
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  auto head_dim_u32 = checked_u32(head_dim, "RoPE head_dim");
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  auto position_u32 = checked_u32(position, "RoPE position");
  if (!position_u32.is_ok()) {
    return position_u32.status();
  }
  auto values_status = validate_f32_buffer(values, heads * head_dim, "RoPE values");
  if (!values_status.is_ok()) {
    return values_status;
  }
  const auto total = heads * (head_dim / 2U);

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    RopeParams params{heads_u32.value(), head_dim_u32.value(), position_u32.value(),
                      theta};
    [encoder setComputePipelineState:impl_->rope_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    const NSUInteger max_threads = [impl_->rope_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(total), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error("MPS RoPE command failed: " +
                                    nserror_to_string([command_buffer error]));
    }
  }
  return Status::ok();
}

Status MpsContext::add_f32_in_place(MpsBuffer& target, const MpsBuffer& delta,
                                    std::size_t size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "add size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto target_status = validate_f32_buffer(target, size, "add target");
  if (!target_status.is_ok()) {
    return target_status;
  }
  auto delta_status = validate_f32_buffer(delta, size, "add delta");
  if (!delta_status.is_ok()) {
    return delta_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    SizeParams params{size_u32.value()};
    [encoder setComputePipelineState:impl_->add_pipeline];
    [encoder setBuffer:target.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:delta.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->add_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error("MPS add command failed: " +
                                    nserror_to_string([command_buffer error]));
    }
  }
  return Status::ok();
}

Status MpsContext::silu_mul_f32_in_place(MpsBuffer& gate, const MpsBuffer& up,
                                         std::size_t size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "silu mul size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto gate_status = validate_f32_buffer(gate, size, "silu gate");
  if (!gate_status.is_ok()) {
    return gate_status;
  }
  auto up_status = validate_f32_buffer(up, size, "silu up");
  if (!up_status.is_ok()) {
    return up_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    SizeParams params{size_u32.value()};
    [encoder setComputePipelineState:impl_->silu_mul_pipeline];
    [encoder setBuffer:gate.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:up.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->silu_mul_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error("MPS silu mul command failed: " +
                                    nserror_to_string([command_buffer error]));
    }
  }
  return Status::ok();
}

Status MpsContext::copy_f32_region(const MpsBuffer& source, MpsBuffer& destination,
                                   std::size_t source_offset,
                                   std::size_t destination_offset,
                                   std::size_t size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto source_offset_u32 = checked_u32(source_offset, "copy source offset");
  if (!source_offset_u32.is_ok()) {
    return source_offset_u32.status();
  }
  auto destination_offset_u32 = checked_u32(destination_offset, "copy destination offset");
  if (!destination_offset_u32.is_ok()) {
    return destination_offset_u32.status();
  }
  auto size_u32 = checked_u32(size, "copy size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto source_status = validate_f32_buffer(source, source_offset + size, "copy source");
  if (!source_status.is_ok()) {
    return source_status;
  }
  auto destination_status =
    validate_f32_buffer(destination, destination_offset + size, "copy destination");
  if (!destination_status.is_ok()) {
    return destination_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    CopyRegionParams params{source_offset_u32.value(), destination_offset_u32.value(),
                            size_u32.value()};
    [encoder setComputePipelineState:impl_->copy_region_pipeline];
    [encoder setBuffer:source.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:destination.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->copy_region_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error("MPS copy region command failed: " +
                                    nserror_to_string([command_buffer error]));
    }
  }
  return Status::ok();
}

Status MpsContext::attention_f32(const MpsBuffer& query, const MpsBuffer& key_cache,
                                 const MpsBuffer& value_cache, std::size_t layer,
                                 std::size_t position, std::size_t capacity_tokens,
                                 std::size_t heads, std::size_t kv_heads,
                                 std::size_t head_dim, MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (heads == 0 || kv_heads == 0 || head_dim == 0 || capacity_tokens == 0) {
    return Status::invalid_argument("MPS attention dimensions must be positive");
  }
  if (heads % kv_heads != 0) {
    return Status::invalid_argument("MPS attention heads must be divisible by kv_heads");
  }
  if (position >= capacity_tokens) {
    return Status::invalid_argument("MPS attention position exceeds KV cache capacity");
  }
  auto layer_u32 = checked_u32(layer, "attention layer");
  auto position_u32 = checked_u32(position, "attention position");
  auto capacity_u32 = checked_u32(capacity_tokens, "attention capacity");
  auto heads_u32 = checked_u32(heads, "attention heads");
  auto kv_heads_u32 = checked_u32(kv_heads, "attention kv_heads");
  auto head_dim_u32 = checked_u32(head_dim, "attention head_dim");
  auto group_u32 = checked_u32(heads / kv_heads, "attention group");
  if (!layer_u32.is_ok()) {
    return layer_u32.status();
  }
  if (!position_u32.is_ok()) {
    return position_u32.status();
  }
  if (!capacity_u32.is_ok()) {
    return capacity_u32.status();
  }
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  if (!kv_heads_u32.is_ok()) {
    return kv_heads_u32.status();
  }
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  if (!group_u32.is_ok()) {
    return group_u32.status();
  }
  const auto attn_dim = heads * head_dim;
  const auto kv_dim = kv_heads * head_dim;
  const auto cache_values = (layer + 1U) * capacity_tokens * kv_dim;
  auto query_status = validate_f32_buffer(query, attn_dim, "attention query");
  if (!query_status.is_ok()) {
    return query_status;
  }
  auto key_status = validate_f32_buffer(key_cache, cache_values, "attention key cache");
  if (!key_status.is_ok()) {
    return key_status;
  }
  auto value_status = validate_f32_buffer(value_cache, cache_values, "attention value cache");
  if (!value_status.is_ok()) {
    return value_status;
  }
  auto output_status = validate_f32_buffer(output, attn_dim, "attention output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    AttentionParams params{
      layer_u32.value(),    position_u32.value(), capacity_u32.value(),
      heads_u32.value(),    kv_heads_u32.value(), head_dim_u32.value(),
      group_u32.value(),    scale,
    };
    [encoder setComputePipelineState:impl_->attention_pipeline];
    [encoder setBuffer:query.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:key_cache.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:value_cache.impl_->buffer offset:0 atIndex:2];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    const NSUInteger max_threads = [impl_->attention_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(attn_dim), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error("MPS attention command failed: " +
                                    nserror_to_string([command_buffer error]));
    }
  }
  return Status::ok();
}

BackendInfo query_backend() {
  BackendInfo info{};
  info.compiled = true;

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      info.failure_reason = "Metal returned no default device";
      return info;
    }

    info.available = true;
    info.compute_ready = true;
    info.forward_ready = true;

    NSString* name = [device name];
    if (name != nil) {
      info.device_name = [name UTF8String];
    }

    info.recommended_max_working_set_size =
      static_cast<std::uint64_t>([device recommendedMaxWorkingSetSize]);
    info.low_power = [device isLowPower];
    info.headless = [device isHeadless];
    info.removable = [device isRemovable];
  }

  return info;
}

std::string format_backend_info(const BackendInfo& info) {
  std::ostringstream output;
  output << "MPS compiled: " << yes_no(info.compiled) << '\n';
  output << "MPS available: " << yes_no(info.available) << '\n';
  output << "MPS compute ready: " << yes_no(info.compute_ready) << '\n';
  output << "MPS full forward ready: " << yes_no(info.forward_ready) << '\n';
  if (!info.device_name.empty()) {
    output << "Metal device: " << info.device_name << '\n';
  }
  if (info.recommended_max_working_set_size > 0) {
    output << "Recommended max working set: " << info.recommended_max_working_set_size
           << " bytes\n";
  }
  output << "Low power: " << yes_no(info.low_power) << '\n';
  output << "Headless: " << yes_no(info.headless) << '\n';
  output << "Removable: " << yes_no(info.removable) << '\n';
  if (!info.failure_reason.empty()) {
    output << "Reason: " << info.failure_reason << '\n';
  }
  return output.str();
}

Status run_operator_smoke_test() {
  auto context_result = MpsContext::create();
  if (!context_result.is_ok()) {
    return context_result.status();
  }
  auto context = std::move(context_result.value());

  const std::uint16_t weight[] = {
    float_to_bf16(1.0F),
    float_to_bf16(2.0F),
    float_to_bf16(3.0F),
    float_to_bf16(1.0F),
  };
  auto weight_buffer_result = context.make_buffer(sizeof(weight));
  if (!weight_buffer_result.is_ok()) {
    return weight_buffer_result.status();
  }
  auto weight_buffer = std::move(weight_buffer_result.value());
  const auto copy_status = context.copy_to_buffer(weight_buffer, weight, sizeof(weight));
  if (!copy_status.is_ok()) {
    return copy_status;
  }

  const std::vector<float> input{3.0F, 4.0F};
  auto output_result = context.matvec_bf16_f32(weight_buffer, 2, 2, input);
  if (!output_result.is_ok()) {
    return output_result.status();
  }
  const auto& output = output_result.value();
  if (output.size() != 2 || std::abs(output[0] - 11.0F) > 1e-5F ||
      std::abs(output[1] - 13.0F) > 1e-5F) {
    std::ostringstream message;
    message << "MPS operator smoke mismatch: got [";
    if (!output.empty()) {
      message << output[0];
    }
    if (output.size() > 1) {
      message << ", " << output[1];
    }
    message << ']';
    return Status::internal_error(message.str());
  }
  return Status::ok();
}

}  // namespace toyllm::mps
