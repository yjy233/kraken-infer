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
#include <vector>

namespace toyllm::mps {

namespace {

constexpr const char* kKernelSource = R"metal(
#include <metal_stdlib>
using namespace metal;

static inline float bf16_to_float(ushort value) {
  return as_type<float>(static_cast<uint>(value) << 16);
}

kernel void bf16_matvec(const device ushort* weight [[buffer(0)]],
                        const device float* input [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant uint& cols [[buffer(3)]],
                        uint row [[thread_position_in_grid]]) {
  float sum = 0.0f;
  const uint row_offset = row * cols;
  for (uint col = 0; col < cols; ++col) {
    sum += bf16_to_float(weight[row_offset + col]) * input[col];
  }
  output[row] = sum;
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

  ~Impl() {
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

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:kKernelSource];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
      [queue release];
      [device release];
      return Status::internal_error("failed to compile Metal kernels: " +
                                    nserror_to_string(error));
    }

    id<MTLFunction> matvec_function =
      [library newFunctionWithName:[NSString stringWithUTF8String:"bf16_matvec"]];
    if (matvec_function == nil) {
      [library release];
      [queue release];
      [device release];
      return Status::internal_error("failed to find Metal kernel bf16_matvec");
    }

    id<MTLComputePipelineState> matvec_pipeline =
      [device newComputePipelineStateWithFunction:matvec_function error:&error];
    [matvec_function release];
    [library release];
    if (matvec_pipeline == nil) {
      [queue release];
      [device release];
      return Status::internal_error("failed to create bf16_matvec pipeline: " +
                                    nserror_to_string(error));
    }

    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->queue = queue;
    impl->matvec_pipeline = matvec_pipeline;
    return MpsContext(std::move(impl));
  }
}

bool MpsContext::valid() const {
  return impl_ != nullptr && impl_->device != nil && impl_->queue != nil &&
         impl_->matvec_pipeline != nil;
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

Result<std::vector<float>> MpsContext::matvec_bf16_f32(const MpsBuffer& weight,
                                                       std::size_t rows,
                                                       std::size_t cols,
                                                       const std::vector<float>& input) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS weight buffer is not initialized");
  }
  if (rows == 0 || cols == 0) {
    return Status::invalid_argument("MPS matvec rows and cols must be greater than zero");
  }
  if (input.size() != cols) {
    return Status::invalid_argument("MPS matvec input size does not match cols");
  }
  if (cols > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return Status::invalid_argument("MPS matvec cols exceeds uint32 range");
  }

  std::size_t elements = 0;
  if (!checked_mul(rows, cols, elements)) {
    return Status::invalid_argument("MPS matvec weight element count overflow");
  }
  std::size_t weight_bytes = 0;
  if (!checked_mul(elements, sizeof(std::uint16_t), weight_bytes)) {
    return Status::invalid_argument("MPS matvec weight byte count overflow");
  }
  if (weight.byte_size() < weight_bytes) {
    return Status::invalid_argument("MPS matvec weight buffer is smaller than rows * cols");
  }

  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  if (!checked_mul(cols, sizeof(float), input_bytes) ||
      !checked_mul(rows, sizeof(float), output_bytes)) {
    return Status::invalid_argument("MPS matvec buffer byte count overflow");
  }
  if (!fits_nsuinteger(rows)) {
    return Status::invalid_argument("MPS matvec rows exceeds NSUInteger range");
  }

  auto input_buffer_result = make_buffer(input_bytes);
  if (!input_buffer_result.is_ok()) {
    return input_buffer_result.status();
  }
  auto input_buffer = std::move(input_buffer_result.value());
  const auto copy_status = copy_to_buffer(input_buffer, input.data(), input_bytes);
  if (!copy_status.is_ok()) {
    return copy_status;
  }

  auto output_buffer_result = make_buffer(output_bytes);
  if (!output_buffer_result.is_ok()) {
    return output_buffer_result.status();
  }
  auto output_buffer = std::move(output_buffer_result.value());

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
    [encoder setBuffer:input_buffer.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer.impl_->buffer offset:0 atIndex:2];
    const auto cols_u32 = static_cast<std::uint32_t>(cols);
    [encoder setBytes:&cols_u32 length:sizeof(cols_u32) atIndex:3];

    const NSUInteger max_threads = [impl_->matvec_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error("MPS matvec command failed: " +
                                    nserror_to_string([command_buffer error]));
    }
  }

  std::vector<float> output(rows);
  std::memcpy(output.data(), [output_buffer.impl_->buffer contents], output_bytes);
  return output;
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
    info.forward_ready = false;

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
