#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace toyllm::cpu {

struct KvCacheStats {
  bool available{false};
  std::size_t layers{0};
  std::size_t kv_heads{0};
  std::size_t head_dim{0};
  std::size_t kv_dim{0};
  std::size_t capacity_tokens{0};
  std::size_t used_tokens{0};
  std::uint64_t key_bytes{0};
  std::uint64_t value_bytes{0};
  std::uint64_t total_bytes{0};
};

class KvCache {
 public:
  void reset(std::size_t layers, std::size_t capacity_tokens, std::size_t kv_heads,
             std::size_t head_dim);

  void store(std::size_t layer, std::size_t position, const std::vector<float>& key,
             const std::vector<float>& value);

  const float* key_ptr(std::size_t layer, std::size_t position, std::size_t kv_head) const;
  const float* value_ptr(std::size_t layer, std::size_t position, std::size_t kv_head) const;

  std::size_t capacity_tokens() const;
  std::size_t used_tokens() const;
  std::size_t kv_dim() const;
  KvCacheStats stats() const;

 private:
  std::size_t offset(std::size_t layer, std::size_t position, std::size_t kv_head) const;
  void validate_address(std::size_t layer, std::size_t position, std::size_t kv_head) const;

  std::size_t layers_{0};
  std::size_t capacity_tokens_{0};
  std::size_t used_tokens_{0};
  std::size_t kv_heads_{0};
  std::size_t head_dim_{0};
  std::size_t kv_dim_{0};
  std::vector<float> keys_;
  std::vector<float> values_;
};

}  // namespace toyllm::cpu
