#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <sstream>

namespace toyllm::mpsgraph {

namespace {

BackendInfo build_backend_info() {
  BackendInfo info{};
  info.compiled = true;

  id<MTLDevice> metal_device = MTLCreateSystemDefaultDevice();
  if (metal_device == nil) {
    info.failure_reason = "Metal device is not available";
    return info;
  }

MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:metal_device];
  if (graph_device == nil) {
    info.failure_reason = "failed to create MPSGraphDevice from Metal device";
    return info;
  }

  id<MTLCommandQueue> command_queue = [metal_device newCommandQueue];
  if (command_queue == nil) {
    info.failure_reason = "failed to create Metal command queue for MPSGraph";
    return info;
  }

  info.available = true;
  info.graph_ready = true;
  info.device_name = metal_device.name != nil ? std::string{metal_device.name.UTF8String}
                                              : std::string{};
  info.recommended_max_working_set_size =
    static_cast<std::uint64_t>(metal_device.recommendedMaxWorkingSetSize);
  info.low_power = metal_device.lowPower;
  info.headless = metal_device.headless;
  info.removable = metal_device.removable;
  return info;
}

Status run_smoke_graph() {
  id<MTLDevice> metal_device = MTLCreateSystemDefaultDevice();
  if (metal_device == nil) {
    return Status::unavailable("Metal device is not available");
  }
  id<MTLCommandQueue> command_queue = [metal_device newCommandQueue];
  if (command_queue == nil) {
    return Status::unavailable("failed to create Metal command queue for MPSGraph");
  }

  MPSGraph* graph = [MPSGraph new];
  if (graph == nil) {
    return Status::unavailable("failed to create MPSGraph");
  }

  const float lhs_values[] = {1.0F, 2.0F};
  const float rhs_values[] = {3.0F, 4.0F};
  NSData* lhs_data = [NSData dataWithBytes:lhs_values length:sizeof(lhs_values)];
  NSData* rhs_data = [NSData dataWithBytes:rhs_values length:sizeof(rhs_values)];
  MPSShape* shape = @[ @2 ];

  MPSGraphTensor* lhs = [graph constantWithData:lhs_data shape:shape dataType:MPSDataTypeFloat32];
  MPSGraphTensor* rhs = [graph constantWithData:rhs_data shape:shape dataType:MPSDataTypeFloat32];
  MPSGraphTensor* sum = [graph additionWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
  MPSGraphTensor* product =
    [graph multiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
  MPSGraphTensor* argmax = [graph reductionArgMaximumWithTensor:product axis:0 name:nil];

  id<MTLBuffer> sum_buffer =
    [metal_device newBufferWithLength:sizeof(float) * 2U options:MTLResourceStorageModeShared];
  if (sum_buffer == nil) {
    return Status::unavailable("failed to allocate MPSGraph smoke sum buffer");
  }

  const auto argmax_byte_size = [&]() -> std::size_t {
    switch (argmax.dataType) {
      case MPSDataTypeInt64:
      case MPSDataTypeUInt64:
        return sizeof(std::uint64_t);
      case MPSDataTypeInt32:
      case MPSDataTypeUInt32:
        return sizeof(std::uint32_t);
      default:
        return 0U;
    }
  }();
  if (argmax_byte_size == 0U) {
    return Status::internal_error("MPSGraph smoke argmax returned unsupported dtype");
  }
  id<MTLBuffer> argmax_buffer =
    [metal_device newBufferWithLength:argmax_byte_size options:MTLResourceStorageModeShared];
  if (argmax_buffer == nil) {
    return Status::unavailable("failed to allocate MPSGraph smoke argmax buffer");
  }

  MPSGraphTensorData* sum_output_data =
    [[MPSGraphTensorData alloc] initWithMTLBuffer:sum_buffer
                                           shape:shape
                                        dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* argmax_output_data =
    [[MPSGraphTensorData alloc] initWithMTLBuffer:argmax_buffer
                                           shape:@[]
                                        dataType:argmax.dataType];
  if (sum_output_data == nil || argmax_output_data == nil) {
    return Status::unavailable("failed to bind MPSGraph smoke output buffers");
  }

  NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [@{
    sum : sum_output_data,
    argmax : argmax_output_data,
  } mutableCopy];
  [graph runWithMTLCommandQueue:command_queue
                          feeds:@{}
               targetOperations:nil
              resultsDictionary:results];
  if (results == nil) {
    return Status::internal_error("MPSGraph smoke execution returned no results");
  }

  const auto* sum_output = static_cast<const float*>(sum_buffer.contents);
  std::int64_t argmax_output = -1;
  const void* argmax_contents = argmax_buffer.contents;
  switch (argmax.dataType) {
    case MPSDataTypeInt64: {
      argmax_output = *static_cast<const std::int64_t*>(argmax_contents);
      break;
    }
    case MPSDataTypeInt32: {
      argmax_output = static_cast<std::int64_t>(*static_cast<const std::int32_t*>(
        argmax_contents));
      break;
    }
    case MPSDataTypeUInt64: {
      argmax_output = static_cast<std::int64_t>(*static_cast<const std::uint64_t*>(
        argmax_contents));
      break;
    }
    case MPSDataTypeUInt32: {
      argmax_output = static_cast<std::int64_t>(*static_cast<const std::uint32_t*>(
        argmax_contents));
      break;
    }
    default:
      return Status::internal_error("MPSGraph smoke argmax returned unsupported dtype");
  }

  if (sum_output[0] != 4.0F || sum_output[1] != 6.0F) {
    return Status::internal_error("MPSGraph smoke addition produced unexpected results");
  }
  if (argmax_output != 1) {
    return Status::internal_error("MPSGraph smoke argmax produced unexpected result");
  }
  return Status::ok();
}

}  // namespace

BackendInfo query_backend() { return build_backend_info(); }

std::string format_backend_info(const BackendInfo& info) {
  std::ostringstream output;
  output << "MPSGraph compiled: " << (info.compiled ? "yes" : "no") << '\n';
  output << "MPSGraph available: " << (info.available ? "yes" : "no") << '\n';
  output << "MPSGraph ready: " << (info.graph_ready ? "yes" : "no") << '\n';
  if (!info.device_name.empty()) {
    output << "Device: " << info.device_name << '\n';
  }
  if (info.recommended_max_working_set_size != 0) {
    output << "Recommended max working set size: "
           << info.recommended_max_working_set_size << '\n';
  }
  output << "Low power: " << (info.low_power ? "yes" : "no") << '\n';
  output << "Headless: " << (info.headless ? "yes" : "no") << '\n';
  output << "Removable: " << (info.removable ? "yes" : "no") << '\n';
  if (!info.failure_reason.empty()) {
    output << "Reason: " << info.failure_reason << '\n';
  }
  return output.str();
}

Status run_operator_smoke_test() {
  const auto info = build_backend_info();
  if (!info.available || !info.graph_ready) {
    const auto message = info.failure_reason.empty()
                           ? std::string{"MPSGraph backend is not ready"}
                           : info.failure_reason;
    return Status::unavailable(message);
  }
  return run_smoke_graph();
}

}  // namespace toyllm::mpsgraph
