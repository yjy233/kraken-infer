#pragma once

#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"

#include <cstddef>
#include <cstdint>

namespace toyllm::mpsgraph {

struct MpsGraphKvCacheStats {
  bool allocated{false};
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

class MpsGraphKvCache {
 public:
  MpsGraphKvCache() = default;
  ~MpsGraphKvCache() = default;

  MpsGraphKvCache(const MpsGraphKvCache&) = delete;
  MpsGraphKvCache& operator=(const MpsGraphKvCache&) = delete;

  MpsGraphKvCache(MpsGraphKvCache&& other) noexcept = default;
  MpsGraphKvCache& operator=(MpsGraphKvCache&& other) noexcept = default;

  [[nodiscard]] Status reset(const MpsGraphContext& context,
                             std::size_t layers,
                             std::size_t capacity_tokens,
                             std::size_t kv_heads,
                             std::size_t head_dim);
  [[nodiscard]] Status store(const MpsGraphContext& context,
                             std::size_t layer,
                             std::size_t position,
                             const MpsGraphBuffer& key,
                             const MpsGraphBuffer& value);
  [[nodiscard]] Status mark_position_used(std::size_t position);

  [[nodiscard]] bool allocated() const;
  [[nodiscard]] const MpsGraphBuffer& key_buffer() const;
  [[nodiscard]] const MpsGraphBuffer& value_buffer() const;
  [[nodiscard]] std::size_t value_offset(std::size_t layer, std::size_t position) const;
  [[nodiscard]] MpsGraphKvCacheStats stats() const;

 private:
  MpsGraphBuffer key_cache_;
  MpsGraphBuffer value_cache_;
  std::size_t layers_{0};
  std::size_t capacity_tokens_{0};
  std::size_t kv_heads_{0};
  std::size_t head_dim_{0};
  std::size_t kv_dim_{0};
  std::size_t used_tokens_{0};
  std::uint64_t key_bytes_{0};
  std::uint64_t value_bytes_{0};
};

}  // namespace toyllm::mpsgraph
