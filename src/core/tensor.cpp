#include "toyllm/core/tensor.hpp"

#include <limits>
#include <stdexcept>

namespace toyllm {

std::string dtype_to_string(DType dtype) {
  switch (dtype) {
    case DType::f32:
      return "f32";
    case DType::f16:
      return "f16";
    case DType::i32:
      return "i32";
    case DType::i64:
      return "i64";
    case DType::u8:
      return "u8";
  }
  return "unknown";
}

std::size_t dtype_size_bytes(DType dtype) {
  switch (dtype) {
    case DType::f32:
    case DType::i32:
      return 4;
    case DType::f16:
      return 2;
    case DType::i64:
      return 8;
    case DType::u8:
      return 1;
  }
  throw std::invalid_argument("unknown dtype");
}

std::size_t element_count(const Shape& shape) {
  std::size_t total = 1;
  for (const auto dim : shape) {
    if (dim < 0) {
      throw std::invalid_argument("tensor shape dimensions must be non-negative");
    }
    const auto size_dim = static_cast<std::size_t>(dim);
    if (size_dim != 0 && total > std::numeric_limits<std::size_t>::max() / size_dim) {
      throw std::overflow_error("tensor shape element count overflow");
    }
    total *= size_dim;
  }
  return total;
}

Tensor::Tensor(TensorDesc desc) : desc_(std::move(desc)) {
  (void)element_count(desc_.shape);
  (void)dtype_size_bytes(desc_.dtype);
}

const TensorDesc& Tensor::desc() const {
  return desc_;
}

DType Tensor::dtype() const {
  return desc_.dtype;
}

const Shape& Tensor::shape() const {
  return desc_.shape;
}

const Device& Tensor::device() const {
  return desc_.device;
}

std::size_t Tensor::numel() const {
  return element_count(desc_.shape);
}

std::size_t Tensor::byte_size() const {
  const auto count = numel();
  const auto dtype_size = dtype_size_bytes(desc_.dtype);
  if (dtype_size != 0 && count > std::numeric_limits<std::size_t>::max() / dtype_size) {
    throw std::overflow_error("tensor byte size overflow");
  }
  return count * dtype_size;
}

}  // namespace toyllm
