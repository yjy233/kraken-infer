#include "toyllm/runtime/qwen35_multimodal.hpp"

namespace toyllm {

Result<Qwen35ImageRgb> decode_qwen35_image_rgb(const Qwen35ImageDataUrl&) {
  return Status::unavailable(
    "Qwen3.5 image decoding requires Apple ImageIO on this build");
}

}  // namespace toyllm
