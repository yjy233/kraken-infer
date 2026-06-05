#include "kv_cache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace toyllm::cpu {

namespace {

std::uint64_t bytes_for(std::size_t values) {
  if (values > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
    throw std::runtime_error("KV cache byte size overflow");
  }
  return static_cast<std::uint64_t>(values * sizeof(float));
}

}  // namespace

void KvCache::reset(std::size_t layers, std::size_t capacity_tokens, std::size_t kv_heads,
                    std::size_t head_dim) {
  if (layers == 0 || capacity_tokens == 0 || kv_heads == 0 || head_dim == 0) {
    throw std::runtime_error("KV cache dimensions must be positive");
  }
  if (kv_heads > std::numeric_limits<std::size_t>::max() / head_dim) {
    throw std::runtime_error("KV cache dimension overflow");
  }
  const auto kv_dim = kv_heads * head_dim;
  if (layers > std::numeric_limits<std::size_t>::max() / capacity_tokens ||
      layers * capacity_tokens > std::numeric_limits<std::size_t>::max() / kv_dim) {
    throw std::runtime_error("KV cache allocation size overflow");
  }

  layers_ = layers;
  capacity_tokens_ = capacity_tokens;
  used_tokens_ = 0;
  kv_heads_ = kv_heads;
  head_dim_ = head_dim;
  kv_dim_ = kv_dim;
  keys_.assign(layers_ * capacity_tokens_ * kv_dim_, 0.0F);
  values_.assign(layers_ * capacity_tokens_ * kv_dim_, 0.0F);
}

void KvCache::store(std::size_t layer, std::size_t position, const std::vector<float>& key,
                    const std::vector<float>& value) {
  validate_address(layer, position, 0);
  if (key.size() != kv_dim_ || value.size() != kv_dim_) {
    throw std::runtime_error("KV cache key/value size mismatch");
  }

  const auto base = offset(layer, position, 0);
  std::copy(key.begin(), key.end(), keys_.begin() + static_cast<std::ptrdiff_t>(base));
  std::copy(value.begin(), value.end(), values_.begin() + static_cast<std::ptrdiff_t>(base));
  used_tokens_ = std::max(used_tokens_, position + 1U);
}

const float* KvCache::key_ptr(std::size_t layer, std::size_t position,
                              std::size_t kv_head) const {
  validate_address(layer, position, kv_head);
  return keys_.data() + offset(layer, position, kv_head);
}

const float* KvCache::value_ptr(std::size_t layer, std::size_t position,
                                std::size_t kv_head) const {
  validate_address(layer, position, kv_head);
  return values_.data() + offset(layer, position, kv_head);
}

std::size_t KvCache::capacity_tokens() const { return capacity_tokens_; }

std::size_t KvCache::used_tokens() const { return used_tokens_; }

std::size_t KvCache::kv_dim() const { return kv_dim_; }

KvCacheStats KvCache::stats() const {
  const auto key_values = keys_.size();
  const auto value_values = values_.size();
  const auto key_bytes = bytes_for(key_values);
  const auto value_bytes = bytes_for(value_values);
  return KvCacheStats{
    true, layers_, kv_heads_, head_dim_, kv_dim_, capacity_tokens_, used_tokens_,
    key_bytes, value_bytes, key_bytes + value_bytes};
}

std::size_t KvCache::offset(std::size_t layer, std::size_t position, std::size_t kv_head) const {
  return (layer * capacity_tokens_ + position) * kv_dim_ + kv_head * head_dim_;
}

void KvCache::validate_address(std::size_t layer, std::size_t position,
                               std::size_t kv_head) const {
  if (layer >= layers_) {
    throw std::runtime_error("KV cache layer index exceeds capacity");
  }
  if (position >= capacity_tokens_) {
    throw std::runtime_error("KV cache position exceeds capacity");
  }
  if (kv_head >= kv_heads_) {
    throw std::runtime_error("KV cache head index exceeds capacity");
  }
}

}  // namespace toyllm::cpu
