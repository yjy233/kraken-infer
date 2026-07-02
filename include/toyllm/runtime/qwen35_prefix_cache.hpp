#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace toyllm {

struct Qwen35PrefixCacheConfig {
  bool enabled{false};
  std::size_t block_tokens{1024};
  std::size_t capacity_blocks{16};
  std::size_t min_reuse_tokens{0};
};

struct Qwen35PrefixCacheStats {
  bool enabled{false};
  std::size_t block_tokens{0};
  std::size_t capacity_blocks{0};
  std::size_t stored_blocks{0};
  std::size_t hit_tokens{0};
  std::size_t miss_tokens{0};
  std::size_t committed_tokens{0};
  std::size_t evicted_blocks{0};
};

struct Qwen35PrefixBlockMetadata {
  std::uint64_t hash{0};
  std::uint64_t parent_hash{0};
  std::size_t slot{0};
  std::size_t token_start{0};
  std::size_t token_count{0};
  std::vector<std::int64_t> tokens;
  std::uint64_t last_used_tick{0};
};

struct Qwen35PrefixCacheLookup {
  std::size_t hit_tokens{0};
  std::vector<std::uint64_t> block_hashes;
};

struct Qwen35PrefixCacheCommit {
  bool inserted{false};
  bool evicted{false};
  std::uint64_t hash{0};
  std::uint64_t evicted_hash{0};
  std::size_t slot{0};
};

class Qwen35PrefixCacheIndex {
 public:
  explicit Qwen35PrefixCacheIndex(Qwen35PrefixCacheConfig config);

  [[nodiscard]] const Qwen35PrefixCacheConfig& config() const;
  [[nodiscard]] Qwen35PrefixCacheStats stats() const;
  [[nodiscard]] Qwen35PrefixCacheLookup lookup(
    const std::vector<std::int64_t>& tokens);
  [[nodiscard]] Qwen35PrefixCacheCommit commit_block(
    const std::vector<std::int64_t>& tokens, std::size_t token_start);
  [[nodiscard]] const Qwen35PrefixBlockMetadata* block(std::uint64_t hash) const;
  [[nodiscard]] std::optional<std::uint64_t> hash_for_block(
    const std::vector<std::int64_t>& tokens, std::size_t token_start) const;

  void clear();

 private:
  [[nodiscard]] std::uint64_t compute_hash(
    const std::vector<std::int64_t>& tokens, std::size_t token_start,
    std::uint64_t parent_hash) const;
  [[nodiscard]] std::uint64_t parent_hash_for(
    const std::vector<std::int64_t>& tokens, std::size_t token_start) const;
  [[nodiscard]] std::size_t allocate_slot(std::uint64_t new_hash,
                                          std::uint64_t& evicted_hash);

  Qwen35PrefixCacheConfig config_;
  Qwen35PrefixCacheStats stats_;
  std::vector<Qwen35PrefixBlockMetadata> blocks_;
  std::uint64_t tick_{0};
};

}  // namespace toyllm
