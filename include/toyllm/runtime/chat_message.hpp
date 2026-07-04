#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace toyllm {

enum class ChatContentPartKind {
  text,
  image_url,
};

struct ChatContentPart {
  ChatContentPart() = default;
  ChatContentPart(ChatContentPartKind kind_value, std::string text_value,
                  std::string image_url_value, std::string detail_value)
      : kind(kind_value),
        text(std::move(text_value)),
        image_url(std::move(image_url_value)),
        detail(std::move(detail_value)) {}

  ChatContentPartKind kind{ChatContentPartKind::text};
  std::string text;
  std::string image_url;
  std::string detail;
  std::string image_mime_type;
  std::vector<std::uint8_t> image_bytes;
  std::uint64_t image_fingerprint{0};
  std::uint32_t image_width{0};
  std::uint32_t image_height{0};
};

struct ChatMessage {
  ChatMessage() = default;
  ChatMessage(std::string role_value, std::string content_value)
      : role(std::move(role_value)), content(std::move(content_value)) {}
  ChatMessage(std::string role_value, std::string content_value,
              std::vector<ChatContentPart> parts_value)
      : role(std::move(role_value)),
        content(std::move(content_value)),
        content_parts(std::move(parts_value)) {}

  std::string role;
  std::string content;
  std::vector<ChatContentPart> content_parts;
};

[[nodiscard]] inline bool chat_message_has_image_content(const ChatMessage& message) {
  for (const auto& part : message.content_parts) {
    if (part.kind == ChatContentPartKind::image_url) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] inline bool chat_messages_have_image_content(
  const std::vector<ChatMessage>& messages) {
  for (const auto& message : messages) {
    if (chat_message_has_image_content(message)) {
      return true;
    }
  }
  return false;
}

}  // namespace toyllm
