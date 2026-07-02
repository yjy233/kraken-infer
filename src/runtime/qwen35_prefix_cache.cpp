#include "toyllm/runtime/qwen35_prefix_cache.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace toyllm {

namespace {

constexpr std::uint64_t kRootHash = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

std::uint64_t mix_u64(std::uint64_t hash, std::uint64_t value) {
  hash ^= value;
  hash *= kFnvPrime;
  return hash;
}

std::uint64_t token_bits(std::int64_t token) {
  return static_cast<std::uint64_t>(token);
}

}  // namespace

Qwen35PrefixCacheIndex::Qwen35PrefixCacheIndex(Qwen35PrefixCacheConfig config)
  : config_(config) {
  if (config_.block_tokens == 0) {
    config_.enabled = false;
  }
  stats_.enabled = config_.enabled;
  stats_.block_tokens = config_.block_tokens;
  stats_.capacity_blocks = config_.capacity_blocks;
}

const Qwen35PrefixCacheConfig& Qwen35PrefixCacheIndex::config() const {
  return config_;
}

Qwen35PrefixCacheStats Qwen35PrefixCacheIndex::stats() const {
  auto result = stats_;
  result.enabled = config_.enabled;
  result.block_tokens = config_.block_tokens;
  result.capacity_blocks = config_.capacity_blocks;
  result.stored_blocks = blocks_.size();
  return result;
}

Qwen35PrefixCacheLookup Qwen35PrefixCacheIndex::lookup(
  const std::vector<std::int64_t>& tokens) {
  Qwen35PrefixCacheLookup result;
  if (!config_.enabled || config_.block_tokens == 0 || config_.capacity_blocks == 0) {
    stats_.miss_tokens += tokens.size();
    return result;
  }

  std::uint64_t parent_hash = kRootHash;
  for (std::size_t token_start = 0;
       token_start + config_.block_tokens <= tokens.size();
       token_start += config_.block_tokens) {
    const auto hash = compute_hash(tokens, token_start, parent_hash);
    auto* metadata = const_cast<Qwen35PrefixBlockMetadata*>(block(hash));
    if (metadata == nullptr || metadata->parent_hash != parent_hash ||
        metadata->token_count != config_.block_tokens) {
      break;
    }
    bool tokens_match = true;
    for (std::size_t index = 0; index < config_.block_tokens; ++index) {
      if (metadata->tokens[index] != tokens[token_start + index]) {
        tokens_match = false;
        break;
      }
    }
    if (!tokens_match) {
      break;
    }
    ++tick_;
    metadata->last_used_tick = tick_;
    result.hit_tokens += config_.block_tokens;
    result.block_hashes.push_back(hash);
    parent_hash = hash;
  }

  if (result.hit_tokens < config_.min_reuse_tokens) {
    result.hit_tokens = 0;
    result.block_hashes.clear();
  }
  stats_.hit_tokens += result.hit_tokens;
  stats_.miss_tokens += tokens.size() - result.hit_tokens;
  return result;
}

Qwen35PrefixCacheCommit Qwen35PrefixCacheIndex::commit_block(
  const std::vector<std::int64_t>& tokens, std::size_t token_start) {
  Qwen35PrefixCacheCommit result;
  if (!config_.enabled || config_.block_tokens == 0 || config_.capacity_blocks == 0 ||
      token_start + config_.block_tokens > tokens.size()) {
    return result;
  }

  const auto parent_hash = parent_hash_for(tokens, token_start);
  const auto hash = compute_hash(tokens, token_start, parent_hash);
  result.hash = hash;

  if (auto* metadata = const_cast<Qwen35PrefixBlockMetadata*>(block(hash));
      metadata != nullptr) {
    ++tick_;
    metadata->last_used_tick = tick_;
    result.slot = metadata->slot;
    return result;
  }

  std::uint64_t evicted_hash = 0;
  const auto slot = allocate_slot(hash, evicted_hash);
  result.slot = slot;
  result.evicted_hash = evicted_hash;
  result.evicted = evicted_hash != 0;
  result.inserted = true;
  if (result.evicted) {
    ++stats_.evicted_blocks;
  }

  Qwen35PrefixBlockMetadata metadata;
  metadata.hash = hash;
  metadata.parent_hash = parent_hash;
  metadata.slot = slot;
  metadata.token_start = token_start;
  metadata.token_count = config_.block_tokens;
  metadata.tokens.insert(metadata.tokens.end(),
                         tokens.begin() + static_cast<std::ptrdiff_t>(token_start),
                         tokens.begin() + static_cast<std::ptrdiff_t>(
                           token_start + config_.block_tokens));
  ++tick_;
  metadata.last_used_tick = tick_;
  blocks_.push_back(std::move(metadata));
  stats_.committed_tokens += config_.block_tokens;
  return result;
}

const Qwen35PrefixBlockMetadata* Qwen35PrefixCacheIndex::block(
  std::uint64_t hash) const {
  const auto it = std::find_if(blocks_.begin(), blocks_.end(),
                               [hash](const Qwen35PrefixBlockMetadata& block) {
                                 return block.hash == hash;
                               });
  if (it == blocks_.end()) {
    return nullptr;
  }
  return &*it;
}

std::optional<std::uint64_t> Qwen35PrefixCacheIndex::hash_for_block(
  const std::vector<std::int64_t>& tokens, std::size_t token_start) const {
  if (config_.block_tokens == 0 ||
      token_start + config_.block_tokens > tokens.size() ||
      token_start % config_.block_tokens != 0) {
    return std::nullopt;
  }
  return compute_hash(tokens, token_start, parent_hash_for(tokens, token_start));
}

void Qwen35PrefixCacheIndex::clear() {
  blocks_.clear();
  tick_ = 0;
  stats_ = Qwen35PrefixCacheStats{};
  stats_.enabled = config_.enabled;
  stats_.block_tokens = config_.block_tokens;
  stats_.capacity_blocks = config_.capacity_blocks;
}

std::uint64_t Qwen35PrefixCacheIndex::compute_hash(
  const std::vector<std::int64_t>& tokens, std::size_t token_start,
  std::uint64_t parent_hash) const {
  std::uint64_t hash = parent_hash;
  hash = mix_u64(hash, static_cast<std::uint64_t>(config_.block_tokens));
  hash = mix_u64(hash, static_cast<std::uint64_t>(
                         token_start / config_.block_tokens));
  for (std::size_t index = 0; index < config_.block_tokens; ++index) {
    hash = mix_u64(hash, token_bits(tokens[token_start + index]));
  }
  return hash == 0 ? 1 : hash;
}

std::uint64_t Qwen35PrefixCacheIndex::parent_hash_for(
  const std::vector<std::int64_t>& tokens, std::size_t token_start) const {
  std::uint64_t parent_hash = kRootHash;
  for (std::size_t start = 0; start < token_start; start += config_.block_tokens) {
    parent_hash = compute_hash(tokens, start, parent_hash);
  }
  return parent_hash;
}

std::size_t Qwen35PrefixCacheIndex::allocate_slot(std::uint64_t new_hash,
                                                  std::uint64_t& evicted_hash) {
  evicted_hash = 0;
  if (blocks_.size() < config_.capacity_blocks) {
    return blocks_.size();
  }

  auto victim = blocks_.begin();
  for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
    if (it->last_used_tick < victim->last_used_tick) {
      victim = it;
    }
  }
  const auto slot = victim->slot;
  evicted_hash = victim->hash;
  blocks_.erase(victim);
  (void)new_hash;
  return slot;
}

}  // namespace toyllm
