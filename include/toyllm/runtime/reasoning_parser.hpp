#pragma once

#include "toyllm/core/status.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace toyllm {

enum class ReasoningFormat {
  none,
  auto_,
  deepseek,
};

struct ReasoningMessage {
  std::string reasoning_content;
  std::string content;
};

struct ReasoningDelta {
  enum class Kind {
    reasoning,
    content,
  };

  Kind kind{Kind::content};
  std::string text;
};

[[nodiscard]] Result<ReasoningFormat> parse_reasoning_format(std::string_view value);
[[nodiscard]] const char* reasoning_format_name(ReasoningFormat format);
[[nodiscard]] bool reasoning_format_extracts_content(ReasoningFormat format);

[[nodiscard]] ReasoningMessage split_reasoning_content(
  std::string_view text, bool enable_thinking, ReasoningFormat format);

class ReasoningStreamParser {
 public:
  ReasoningStreamParser(bool enable_thinking, ReasoningFormat format);

  [[nodiscard]] std::vector<ReasoningDelta> push(std::string_view chunk);
  [[nodiscard]] std::vector<ReasoningDelta> finish();

 private:
  enum class Mode {
    reasoning,
    content,
  };

  [[nodiscard]] std::vector<ReasoningDelta> drain(bool final);
  void append_delta(std::vector<ReasoningDelta>& deltas, ReasoningDelta::Kind kind,
                    std::string text) const;

  ReasoningFormat format_{ReasoningFormat::none};
  Mode mode_{Mode::content};
  bool skip_reasoning_prefix_newlines_{false};
  bool skip_content_prefix_newlines_{false};
  bool finished_{false};
  std::string buffer_;
};

}  // namespace toyllm

