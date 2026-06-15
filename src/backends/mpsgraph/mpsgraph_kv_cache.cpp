#include "toyllm/backends/mpsgraph/mpsgraph_kv_cache.hpp"

#include <algorithm>
#include <limits>

namespace toyllm::mpsgraph {

namespace {

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t& output) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  output = lhs * rhs;
  return true;
}

Result<std::size_t> cache_values(std::size_t layers, std::size_t capacity_tokens,
                                 std::size_t kv_heads, std::size_t head_dim) {
  std::size_t kv_dim = 0;
  if (!checked_mul(kv_heads, head_dim, kv_dim)) {
    return Status::invalid_argument("MPSGraph KV cache kv_dim overflow");
  }
  std::size_t layer_values = 0;
  if (!checked_mul(capacity_tokens, kv_dim, layer_values)) {
    return Status::invalid_argument("MPSGraph KV cache layer size overflow");
  }
  std::size_t total_values = 0;
  if (!checked_mul(layers, layer_values, total_values)) {
    return Status::invalid_argument("MPSGraph KV cache total size overflow");
  }
  return total_values;
}

Result<std::size_t> f32_bytes(std::size_t values) {
  std::size_t bytes = 0;
  if (!checked_mul(values, sizeof(float), bytes)) {
    return Status::invalid_argument("MPSGraph KV cache byte count overflow");
  }
  return bytes;
}

}  // namespace

Status MpsGraphKvCache::reset(const MpsGraphContext& context,
                              std::size_t layers,
                              std::size_t capacity_tokens,
                              std::size_t kv_heads,
                              std::size_t head_dim) {
  if (layers == 0 || capacity_tokens == 0 || kv_heads == 0 || head_dim == 0) {
    return Status::invalid_argument("MPSGraph KV cache dimensions must be positive");
  }

  auto values = cache_values(layers, capacity_tokens, kv_heads, head_dim);
  if (!values.is_ok()) {
    return values.status();
  }
  auto bytes = f32_bytes(values.value());
  if (!bytes.is_ok()) {
    return bytes.status();
  }

  auto key = context.make_buffer(bytes.value());
  if (!key.is_ok()) {
    return key.status();
  }
  auto value = context.make_buffer(bytes.value());
  if (!value.is_ok()) {
    return value.status();
  }

  key_cache_ = std::move(key.value());
  value_cache_ = std::move(value.value());
  layers_ = layers;
  capacity_tokens_ = capacity_tokens;
  kv_heads_ = kv_heads;
  head_dim_ = head_dim;
  kv_dim_ = kv_heads * head_dim;
  used_tokens_ = 0;
  key_bytes_ = static_cast<std::uint64_t>(bytes.value());
  value_bytes_ = static_cast<std::uint64_t>(bytes.value());
  return Status::ok();
}

Status MpsGraphKvCache::store(const MpsGraphContext& context,
                              std::size_t layer,
                              std::size_t position,
                              const MpsGraphBuffer& key,
                              const MpsGraphBuffer& value) {
  if (!allocated()) {
    return Status::invalid_argument("MPSGraph KV cache is not allocated");
  }
  if (layer >= layers_) {
    return Status::invalid_argument("MPSGraph KV cache layer exceeds capacity");
  }
  if (position >= capacity_tokens_) {
    return Status::invalid_argument("MPSGraph KV cache position exceeds capacity");
  }

  auto status = context.write_kv_cache_pair_f32(key, value, key_cache_, value_cache_, layer,
                                                position, layers_, capacity_tokens_,
                                                kv_heads_, head_dim_);
  if (!status.is_ok()) {
    return status;
  }
  return mark_position_used(position);
}

Status MpsGraphKvCache::mark_position_used(std::size_t position) {
  if (!allocated()) {
    return Status::invalid_argument("MPSGraph KV cache is not allocated");
  }
  if (position >= capacity_tokens_) {
    return Status::invalid_argument("MPSGraph KV cache position exceeds capacity");
  }
  used_tokens_ = std::max(used_tokens_, position + 1U);
  return Status::ok();
}

bool MpsGraphKvCache::allocated() const {
  return key_cache_.valid() && value_cache_.valid();
}

MpsGraphBuffer& MpsGraphKvCache::key_buffer() {
  return key_cache_;
}

MpsGraphBuffer& MpsGraphKvCache::value_buffer() {
  return value_cache_;
}

const MpsGraphBuffer& MpsGraphKvCache::key_buffer() const {
  return key_cache_;
}

const MpsGraphBuffer& MpsGraphKvCache::value_buffer() const {
  return value_cache_;
}

std::size_t MpsGraphKvCache::value_offset(std::size_t layer, std::size_t position) const {
  return (layer * capacity_tokens_ + position) * kv_dim_;
}

MpsGraphKvCacheStats MpsGraphKvCache::stats() const {
  MpsGraphKvCacheStats output;
  output.allocated = allocated();
  output.layers = layers_;
  output.kv_heads = kv_heads_;
  output.head_dim = head_dim_;
  output.kv_dim = kv_dim_;
  output.capacity_tokens = capacity_tokens_;
  output.used_tokens = used_tokens_;
  output.key_bytes = key_bytes_;
  output.value_bytes = value_bytes_;
  output.total_bytes = key_bytes_ + value_bytes_;
  return output;
}

}  // namespace toyllm::mpsgraph
