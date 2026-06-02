#pragma once

#include "toyllm/core/device.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace toyllm {

enum class DType {
  f32,
  f16,
  i32,
  i64,
  u8,
};

using Shape = std::vector<std::int64_t>;

struct TensorDesc {
  DType dtype{DType::f32};
  Shape shape;
  Device device{Device::cpu()};
};

[[nodiscard]] std::string dtype_to_string(DType dtype);
[[nodiscard]] std::size_t dtype_size_bytes(DType dtype);
[[nodiscard]] std::size_t element_count(const Shape& shape);

class Tensor {
 public:
  explicit Tensor(TensorDesc desc);

  [[nodiscard]] const TensorDesc& desc() const;
  [[nodiscard]] DType dtype() const;
  [[nodiscard]] const Shape& shape() const;
  [[nodiscard]] const Device& device() const;
  [[nodiscard]] std::size_t numel() const;
  [[nodiscard]] std::size_t byte_size() const;

 private:
  TensorDesc desc_;
};

}  // namespace toyllm
