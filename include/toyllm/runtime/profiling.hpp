#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace toyllm {

enum class LogLevel {
  error,
  warn,
  info,
  debug,
  trace,
};

enum class LogFormat {
  text,
  jsonl,
};

enum class ProfileMode {
  off,
  summary,
  trace,
  flamegraph,
  all,
};

struct ObservabilityConfig {
  LogLevel log_level{LogLevel::info};
  LogFormat log_format{LogFormat::text};
  ProfileMode profile_mode{ProfileMode::off};
  std::filesystem::path log_output;
  std::filesystem::path profile_output_dir;
  std::uint64_t min_duration_us{0};
  bool per_layer{true};
  bool per_operator{true};
  bool emit_summary_json{true};
  std::string request_id;
  std::string client_request_id;
};

struct ProfileField {
  std::string key;
  std::string value;
};

struct ProfileArtifacts {
  std::filesystem::path output_dir;
  bool wrote_summary{false};
  bool wrote_trace{false};
  bool wrote_folded{false};
  bool wrote_flamegraph{false};
};

class RequestProfiler;

class ScopedProfileSpan {
 public:
  ScopedProfileSpan() = default;
  ScopedProfileSpan(RequestProfiler* profiler, std::uint32_t span_id);
  ~ScopedProfileSpan();

  ScopedProfileSpan(const ScopedProfileSpan&) = delete;
  ScopedProfileSpan& operator=(const ScopedProfileSpan&) = delete;

  ScopedProfileSpan(ScopedProfileSpan&& other) noexcept;
  ScopedProfileSpan& operator=(ScopedProfileSpan&& other) noexcept;

 private:
  RequestProfiler* profiler_{nullptr};
  std::uint32_t span_id_{0};
};

class RequestProfiler {
 public:
  explicit RequestProfiler(ObservabilityConfig config);
  ~RequestProfiler();

  [[nodiscard]] bool enabled() const;
  [[nodiscard]] const std::string& request_id() const;
  [[nodiscard]] const ObservabilityConfig& config() const;

  ScopedProfileSpan scoped(std::string_view name);
  ScopedProfileSpan scoped(std::string_view name, std::vector<ProfileField> fields);

  void set_metadata(std::string key, std::string value);
  void set_metadata(std::string key, std::size_t value);
  void set_status(std::string status);
  [[nodiscard]] ProfileArtifacts write_artifacts();

 private:
  friend class ScopedProfileSpan;

  std::uint32_t begin_span(std::string name, std::vector<ProfileField> fields);
  void end_span(std::uint32_t span_id);

  ObservabilityConfig config_;
  std::string request_id_;
  struct Impl;
  Impl* impl_{nullptr};
};

[[nodiscard]] std::string profile_mode_to_string(ProfileMode mode);
[[nodiscard]] std::optional<ProfileMode> parse_profile_mode(std::string_view value);
[[nodiscard]] std::string make_cli_request_id();
[[nodiscard]] std::string make_gateway_request_id();

}  // namespace toyllm
