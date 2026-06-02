#include "toyllm/backends/mps/mps_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <sstream>

namespace toyllm::mps {

namespace {

std::string yes_no(bool value) {
  return value ? "yes" : "no";
}

}  // namespace

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

}  // namespace toyllm::mps
