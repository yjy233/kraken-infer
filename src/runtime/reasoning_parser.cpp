#include "toyllm/runtime/reasoning_parser.hpp"

#include <algorithm>
#include <utility>

namespace toyllm {

namespace {

constexpr std::string_view kThinkStart{"<think>"};
constexpr std::string_view kThinkEnd{"</think>"};

bool is_newline(char ch) {
  return ch == '\n' || ch == '\r';
}

void trim_trailing_newlines(std::string& text) {
  while (!text.empty() && is_newline(text.back())) {
    text.pop_back();
  }
}

std::size_t leading_newline_count(std::string_view text) {
  std::size_t count = 0;
  while (count < text.size() && is_newline(text[count])) {
    ++count;
  }
  return count;
}

std::size_t trailing_newline_count(std::string_view text) {
  std::size_t count = 0;
  while (count < text.size() && is_newline(text[text.size() - count - 1U])) {
    ++count;
  }
  return count;
}

std::size_t tag_prefix_suffix_size(std::string_view text, std::string_view tag) {
  const auto limit = std::min(text.size(), tag.size() - 1U);
  for (std::size_t size = limit; size > 0; --size) {
    if (text.substr(text.size() - size, size) == tag.substr(0, size)) {
      return size;
    }
  }
  return 0;
}

bool starts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

}  // namespace

Result<ReasoningFormat> parse_reasoning_format(std::string_view value) {
  if (value == "none") {
    return ReasoningFormat::none;
  }
  if (value == "auto") {
    return ReasoningFormat::auto_;
  }
  if (value == "deepseek") {
    return ReasoningFormat::deepseek;
  }
  return Status::invalid_argument(
    "unsupported reasoning_format: " + std::string{value});
}

const char* reasoning_format_name(ReasoningFormat format) {
  switch (format) {
    case ReasoningFormat::none:
      return "none";
    case ReasoningFormat::auto_:
      return "auto";
    case ReasoningFormat::deepseek:
      return "deepseek";
  }
  return "none";
}

bool reasoning_format_extracts_content(ReasoningFormat format) {
  return format == ReasoningFormat::auto_ || format == ReasoningFormat::deepseek;
}

ReasoningMessage split_reasoning_content(
  std::string_view text, bool enable_thinking, ReasoningFormat format) {
  if (!reasoning_format_extracts_content(format)) {
    return ReasoningMessage{.content = std::string{text}};
  }

  ReasoningMessage message;
  if (enable_thinking) {
    if (starts_with(text, kThinkStart)) {
      text.remove_prefix(kThinkStart.size());
      while (!text.empty() && is_newline(text.front())) {
        text.remove_prefix(1);
      }
    }

    const auto end = text.find(kThinkEnd);
    if (end == std::string_view::npos) {
      message.reasoning_content = std::string{text};
      trim_trailing_newlines(message.reasoning_content);
      return message;
    }

    message.reasoning_content = std::string{text.substr(0, end)};
    trim_trailing_newlines(message.reasoning_content);
    auto content = text.substr(end + kThinkEnd.size());
    while (!content.empty() && is_newline(content.front())) {
      content.remove_prefix(1);
    }
    message.content = std::string{content};
    return message;
  }

  const auto start = text.find(kThinkStart);
  if (start == std::string_view::npos) {
    message.content = std::string{text};
    return message;
  }

  message.content = std::string{text.substr(0, start)};
  auto rest = text.substr(start + kThinkStart.size());
  while (!rest.empty() && is_newline(rest.front())) {
    rest.remove_prefix(1);
  }

  const auto end = rest.find(kThinkEnd);
  if (end == std::string_view::npos) {
    message.reasoning_content = std::string{rest};
    trim_trailing_newlines(message.reasoning_content);
    return message;
  }

  message.reasoning_content = std::string{rest.substr(0, end)};
  trim_trailing_newlines(message.reasoning_content);
  auto content = rest.substr(end + kThinkEnd.size());
  while (!content.empty() && is_newline(content.front())) {
    content.remove_prefix(1);
  }
  message.content += std::string{content};
  return message;
}

ReasoningStreamParser::ReasoningStreamParser(bool enable_thinking,
                                             ReasoningFormat format)
  : format_(format),
    mode_(enable_thinking && reasoning_format_extracts_content(format)
            ? Mode::reasoning
            : Mode::content),
    skip_reasoning_prefix_newlines_(enable_thinking &&
                                    reasoning_format_extracts_content(format)) {}

std::vector<ReasoningDelta> ReasoningStreamParser::push(std::string_view chunk) {
  if (finished_) {
    return {};
  }
  buffer_ += chunk;
  return drain(false);
}

std::vector<ReasoningDelta> ReasoningStreamParser::finish() {
  if (finished_) {
    return {};
  }
  finished_ = true;
  return drain(true);
}

std::vector<ReasoningDelta> ReasoningStreamParser::drain(bool final) {
  std::vector<ReasoningDelta> deltas;
  if (!reasoning_format_extracts_content(format_)) {
    append_delta(deltas, ReasoningDelta::Kind::content, std::move(buffer_));
    buffer_.clear();
    return deltas;
  }

  while (!buffer_.empty()) {
    if (mode_ == Mode::reasoning) {
      if (skip_reasoning_prefix_newlines_) {
        const auto count = leading_newline_count(buffer_);
        buffer_.erase(0, count);
        skip_reasoning_prefix_newlines_ = false;
        if (buffer_.empty()) {
          break;
        }
      }

      if (starts_with(buffer_, kThinkStart)) {
        buffer_.erase(0, kThinkStart.size());
        skip_reasoning_prefix_newlines_ = true;
        continue;
      }

      const auto end = buffer_.find(kThinkEnd);
      if (end != std::string::npos) {
        auto reasoning = buffer_.substr(0, end);
        trim_trailing_newlines(reasoning);
        append_delta(deltas, ReasoningDelta::Kind::reasoning,
                     std::move(reasoning));
        buffer_.erase(0, end + kThinkEnd.size());
        mode_ = Mode::content;
        skip_content_prefix_newlines_ = true;
        continue;
      }

      const auto tag_hold = tag_prefix_suffix_size(buffer_, kThinkEnd);
      const auto newline_hold =
        trailing_newline_count(std::string_view{buffer_}.substr(0, buffer_.size() - tag_hold));
      auto hold = tag_hold + newline_hold;
      if (final) {
        hold = 0;
      }
      if (buffer_.size() == hold) {
        break;
      }
      const auto emit_size = buffer_.size() - hold;
      auto reasoning = buffer_.substr(0, emit_size);
      if (final) {
        trim_trailing_newlines(reasoning);
      }
      append_delta(deltas, ReasoningDelta::Kind::reasoning,
                   std::move(reasoning));
      buffer_.erase(0, emit_size);
      continue;
    }

    if (skip_content_prefix_newlines_) {
      const auto count = leading_newline_count(buffer_);
      buffer_.erase(0, count);
      if (buffer_.empty()) {
        break;
      }
      skip_content_prefix_newlines_ = false;
    }

    const auto start = buffer_.find(kThinkStart);
    if (start != std::string::npos) {
      append_delta(deltas, ReasoningDelta::Kind::content,
                   buffer_.substr(0, start));
      buffer_.erase(0, start + kThinkStart.size());
      mode_ = Mode::reasoning;
      skip_reasoning_prefix_newlines_ = true;
      continue;
    }

    auto hold = tag_prefix_suffix_size(buffer_, kThinkStart);
    if (final) {
      hold = 0;
    }
    if (buffer_.size() == hold) {
      break;
    }
    const auto emit_size = buffer_.size() - hold;
    append_delta(deltas, ReasoningDelta::Kind::content,
                 buffer_.substr(0, emit_size));
    buffer_.erase(0, emit_size);
  }

  return deltas;
}

void ReasoningStreamParser::append_delta(std::vector<ReasoningDelta>& deltas,
                                         ReasoningDelta::Kind kind,
                                         std::string text) const {
  if (text.empty()) {
    return;
  }
  if (!deltas.empty() && deltas.back().kind == kind) {
    deltas.back().text += text;
    return;
  }
  deltas.push_back(ReasoningDelta{.kind = kind, .text = std::move(text)});
}

}  // namespace toyllm
