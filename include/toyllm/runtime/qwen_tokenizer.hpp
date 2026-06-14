#pragma once

#include "toyllm/runtime/chat_message.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace toyllm {

class QwenTokenizer {
 public:
  static QwenTokenizer load(const std::filesystem::path& model_dir);

  [[nodiscard]] std::vector<std::int64_t> encode_chat_messages(
    const std::vector<ChatMessage>& messages, bool enable_thinking);
  [[nodiscard]] std::string decode(const std::vector<std::int64_t>& ids) const;

 private:
  class JsonScanner;

  void load_vocab(const std::filesystem::path& path);
  void load_added_tokens(const std::filesystem::path& path);
  void parse_added_tokens_decoder(JsonScanner& scanner);
  static std::int64_t parse_token_id(const std::string& text);
  void add_token(std::int64_t id, const std::string& token);
  void load_merges(const std::filesystem::path& path);
  [[nodiscard]] std::vector<std::int64_t> encode_text(const std::string& text);
  [[nodiscard]] std::vector<std::int64_t> bpe(std::vector<std::string> word);
  static void append_tokens(std::vector<std::int64_t>& target,
                            const std::vector<std::int64_t>& source);

  std::array<std::string, 256> byte_encoder_{};
  std::unordered_map<std::uint32_t, unsigned char> byte_decoder_;
  std::unordered_map<std::string, std::int64_t> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merge_ranks_;
};

constexpr std::int64_t kQwenEndOfText = 151643;
constexpr std::int64_t kQwenImStart = 151644;
constexpr std::int64_t kQwenImEnd = 151645;
constexpr std::int64_t kQwenThinkStart = 151667;
constexpr std::int64_t kQwenThinkEnd = 151668;

}  // namespace toyllm
