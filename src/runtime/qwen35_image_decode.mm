#include "toyllm/runtime/qwen35_multimodal.hpp"

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace toyllm {

Result<Qwen35ImageRgb> decode_qwen35_image_rgb(const Qwen35ImageDataUrl& image) {
  if (image.bytes.empty()) {
    return Status::invalid_argument("image data URL payload must not be empty");
  }
  if (image.bytes.size() > static_cast<std::size_t>(std::numeric_limits<CFIndex>::max())) {
    return Status::invalid_argument("image payload is too large for ImageIO");
  }

  CFDataRef data = CFDataCreate(
    kCFAllocatorDefault, image.bytes.data(), static_cast<CFIndex>(image.bytes.size()));
  if (data == nullptr) {
    return Status::internal_error("failed to create ImageIO data buffer");
  }

  CGImageSourceRef source = CGImageSourceCreateWithData(data, nullptr);
  CFRelease(data);
  if (source == nullptr) {
    return Status::invalid_argument("ImageIO could not parse image payload");
  }

  CGImageRef cg_image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
  CFRelease(source);
  if (cg_image == nullptr) {
    return Status::invalid_argument("ImageIO could not decode image payload");
  }

  const auto width = CGImageGetWidth(cg_image);
  const auto height = CGImageGetHeight(cg_image);
  if (width == 0 || height == 0 ||
      width > std::numeric_limits<std::uint32_t>::max() ||
      height > std::numeric_limits<std::uint32_t>::max()) {
    CGImageRelease(cg_image);
    return Status::invalid_argument("decoded image dimensions are invalid");
  }
  if (width > std::numeric_limits<std::size_t>::max() / 4U ||
      height > std::numeric_limits<std::size_t>::max() / (width * 4U)) {
    CGImageRelease(cg_image);
    return Status::invalid_argument("decoded image buffer size overflows");
  }

  const auto row_bytes = width * 4U;
  std::vector<std::uint8_t> rgba(height * row_bytes);
  CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
  if (color_space == nullptr) {
    CGImageRelease(cg_image);
    return Status::internal_error("failed to create device RGB color space");
  }

  const auto bitmap_info = static_cast<CGBitmapInfo>(
    static_cast<std::uint32_t>(kCGImageAlphaPremultipliedLast) |
    static_cast<std::uint32_t>(kCGBitmapByteOrder32Big));
  CGContextRef context = CGBitmapContextCreate(
    rgba.data(), width, height, 8, row_bytes, color_space, bitmap_info);
  CGColorSpaceRelease(color_space);
  if (context == nullptr) {
    CGImageRelease(cg_image);
    return Status::internal_error("failed to create RGB bitmap context");
  }

  CGContextDrawImage(
    context, CGRectMake(0.0, 0.0, static_cast<CGFloat>(width),
                        static_cast<CGFloat>(height)), cg_image);
  CGContextRelease(context);
  CGImageRelease(cg_image);

  if (width > std::numeric_limits<std::size_t>::max() / 3U ||
      height > std::numeric_limits<std::size_t>::max() / (width * 3U)) {
    return Status::invalid_argument("RGB image buffer size overflows");
  }
  std::vector<std::uint8_t> rgb(width * height * 3U);
  for (std::size_t pixel = 0; pixel < width * height; ++pixel) {
    rgb[pixel * 3U] = rgba[pixel * 4U];
    rgb[pixel * 3U + 1U] = rgba[pixel * 4U + 1U];
    rgb[pixel * 3U + 2U] = rgba[pixel * 4U + 2U];
  }

  Qwen35ImageRgb result;
  result.width = static_cast<std::uint32_t>(width);
  result.height = static_cast<std::uint32_t>(height);
  result.pixels = std::move(rgb);
  return result;
}

}  // namespace toyllm
