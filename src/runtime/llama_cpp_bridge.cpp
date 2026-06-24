#include "toyllm/runtime/llama_cpp_bridge.hpp"

#include "toyllm/runtime/gguf_reader.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace toyllm {

namespace {

constexpr int kDefaultLlamaServerPort = 18080;
constexpr int kHttpTimeoutSeconds = 600;

std::string lower_ascii(std::string value) {
  for (auto& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

bool write_all(int fd, std::string_view data) {
  std::size_t written = 0;
  while (written < data.size()) {
    const auto result = ::send(fd, data.data() + written, data.size() - written, 0);
    if (result <= 0) {
      return false;
    }
    written += static_cast<std::size_t>(result);
  }
  return true;
}

Result<int> connect_tcp(std::string_view host, int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return Status::internal_error("socket failed: " + std::string{std::strerror(errno)});
  }

  timeval timeout{};
  timeout.tv_sec = kHttpTimeoutSeconds;
  (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#ifdef SO_NOSIGPIPE
  const int no_sigpipe = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, std::string{host}.c_str(), &address.sin_addr) != 1) {
    ::close(fd);
    return Status::invalid_argument("llama.cpp bridge host must be an IPv4 address");
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    const auto message = std::string{std::strerror(errno)};
    ::close(fd);
    return Status::unavailable("connect failed: " + message);
  }
  return fd;
}

std::string strip_status_text(std::string line) {
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  return line;
}

Result<LlamaCppHttpResponse> parse_http_response(std::string response) {
  const auto header_end = response.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    return Status::internal_error("llama.cpp response missing HTTP header terminator");
  }

  std::istringstream header_stream(response.substr(0, header_end));
  std::string status_line;
  std::getline(header_stream, status_line);
  status_line = strip_status_text(std::move(status_line));
  std::istringstream status_parser(status_line);
  std::string http_version;
  int status = 0;
  status_parser >> http_version >> status;
  if (status <= 0) {
    return Status::internal_error("llama.cpp response has invalid HTTP status line");
  }

  LlamaCppHttpResponse parsed;
  parsed.status = status;
  std::string line;
  while (std::getline(header_stream, line)) {
    line = strip_status_text(std::move(line));
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    auto name = lower_ascii(line.substr(0, colon));
    auto value = line.substr(colon + 1U);
    while (!value.empty() && value.front() == ' ') {
      value.erase(value.begin());
    }
    if (name == "content-type") {
      parsed.content_type = value;
    }
    parsed.headers.emplace_back(std::move(name), std::move(value));
  }
  parsed.body = response.substr(header_end + 4U);
  if (parsed.content_type.empty()) {
    parsed.content_type = "application/json";
  }
  return parsed;
}

Result<LlamaCppHttpResponse> send_http_request(std::string_view host, int port,
                                               std::string_view method, std::string_view path,
                                               std::string_view body,
                                               const std::vector<std::pair<std::string,
                                                                           std::string>>& headers) {
  auto fd_result = connect_tcp(host, port);
  if (!fd_result.is_ok()) {
    return fd_result.status();
  }
  const int fd = fd_result.value();

  std::ostringstream request;
  request << method << ' ' << path << " HTTP/1.1\r\n";
  request << "Host: " << host << ':' << port << "\r\n";
  request << "Connection: close\r\n";
  request << "Content-Length: " << body.size() << "\r\n";
  bool has_content_type = false;
  for (const auto& header : headers) {
    const auto lower = lower_ascii(header.first);
    if (lower == "host" || lower == "connection" || lower == "content-length") {
      continue;
    }
    if (lower == "content-type") {
      has_content_type = true;
    }
    request << header.first << ": " << header.second << "\r\n";
  }
  if (!has_content_type) {
    request << "Content-Type: application/json\r\n";
  }
  request << "\r\n";
  request << body;

  if (!write_all(fd, request.str())) {
    ::close(fd);
    return Status::internal_error("failed to write HTTP request to llama.cpp");
  }

  std::string response;
  char chunk[16384];
  while (true) {
    const auto count = ::recv(fd, chunk, sizeof(chunk), 0);
    if (count == 0) {
      break;
    }
    if (count < 0) {
      const auto message = std::string{std::strerror(errno)};
      ::close(fd);
      return Status::internal_error("failed to read HTTP response from llama.cpp: " + message);
    }
    response.append(chunk, static_cast<std::size_t>(count));
  }
  ::close(fd);
  return parse_http_response(std::move(response));
}

std::filesystem::path getenv_path(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return {};
  }
  return std::filesystem::path{value};
}

std::filesystem::path find_executable_in_path(std::string_view name) {
  const char* value = std::getenv("PATH");
  if (value == nullptr) {
    return {};
  }
  std::string_view path_env{value};
  std::size_t start = 0;
  while (start <= path_env.size()) {
    const auto end = path_env.find(':', start);
    const auto part = path_env.substr(start, end == std::string_view::npos
                                               ? std::string_view::npos
                                               : end - start);
    if (!part.empty()) {
      auto candidate = std::filesystem::path{std::string{part}} /
                       std::filesystem::path{std::string{name}};
      if (std::filesystem::is_regular_file(candidate)) {
        return candidate;
      }
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1U;
  }
  return {};
}

std::vector<std::filesystem::path> executable_candidates(const LlamaCppServerOptions& options) {
  std::vector<std::filesystem::path> candidates;
  if (!options.executable_path.empty()) {
    candidates.push_back(options.executable_path);
  }
  if (auto from_env = getenv_path("KRAKEN_LLAMA_SERVER"); !from_env.empty()) {
    candidates.push_back(from_env);
  }
  if (auto from_env = getenv_path("LLAMA_SERVER"); !from_env.empty()) {
    candidates.push_back(from_env);
  }
  const auto& root = options.llama_cpp_dir;
  candidates.push_back(root / "build" / "bin" / "llama-server");
  candidates.push_back(root / "build" / "tools" / "server" / "llama-server");
  candidates.push_back(root / "build" / "llama-server");
  candidates.push_back(root / "llama-server");
  candidates.push_back(std::filesystem::path{"/opt/homebrew/bin/llama-server"});
  candidates.push_back(std::filesystem::path{"/usr/local/bin/llama-server"});
  candidates.push_back(std::filesystem::path{"llama-server"});
  return candidates;
}

std::string shell_quote(const std::filesystem::path& path) {
  std::string text = path.string();
  std::string output;
  output.reserve(text.size() + 2U);
  output.push_back('\'');
  for (const char ch : text) {
    if (ch == '\'') {
      output += "'\\''";
    } else {
      output.push_back(ch);
    }
  }
  output.push_back('\'');
  return output;
}

std::vector<std::string> build_server_args(const LlamaCppServerOptions& options,
                                           const std::filesystem::path& executable,
                                           int port) {
  auto model_path = options.model_path;
  if (auto resolved = resolve_gguf_model_path(model_path); resolved.is_ok()) {
    model_path = resolved.value();
  }
  std::vector<std::string> args;
  args.push_back(executable.string());
  args.push_back("-m");
  args.push_back(model_path.string());
  args.push_back("--host");
  args.push_back(options.host);
  args.push_back("--port");
  args.push_back(std::to_string(port));
  args.push_back("--alias");
  args.push_back(options.model_alias);
  args.push_back("--parallel");
  args.push_back(std::to_string(std::max<std::size_t>(options.parallel_slots, 1U)));
  args.push_back("--batch-size");
  args.push_back(std::to_string(options.batch_size));
  args.push_back("--ubatch-size");
  args.push_back(std::to_string(options.ubatch_size));
  if (options.context_size > 0) {
    args.push_back("--ctx-size");
    args.push_back(std::to_string(options.context_size));
  }
  if (options.compute_device.kind == DeviceKind::mps ||
      options.compute_device.kind == DeviceKind::mpsgraph) {
    args.push_back("--n-gpu-layers");
    args.push_back("auto");
  } else {
    args.push_back("--n-gpu-layers");
    args.push_back("0");
  }
  if (options.enable_flash_attention) {
    args.push_back("--flash-attn");
    args.push_back("auto");
  }
  if (!options.enable_continuous_batching) {
    args.push_back("--no-cont-batching");
  }
  if (options.enable_mtp) {
    args.push_back("--spec-type");
    args.push_back("draft-mtp");
  }
  if (!options.mmproj_path.empty()) {
    args.push_back("--mmproj");
    args.push_back(options.mmproj_path.string());
  }
  for (const auto& arg : options.extra_args) {
    args.push_back(arg);
  }
  return args;
}

Result<bool> start_server_process(const LlamaCppServerOptions& options,
                                  const std::filesystem::path& executable, int port) {
  const auto args = build_server_args(options, executable, port);
  const auto pid = ::fork();
  if (pid < 0) {
    return Status::internal_error("fork failed: " + std::string{std::strerror(errno)});
  }
  if (pid == 0) {
    (void)::setsid();
    ::signal(SIGPIPE, SIG_IGN);

    std::vector<std::string> owned = args;
    std::vector<char*> argv;
    argv.reserve(owned.size() + 1U);
    for (auto& arg : owned) {
      argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    ::execv(executable.c_str(), argv.data());
    _exit(127);
  }
  return true;
}

Result<bool> ensure_server_running(const LlamaCppServerOptions& options, int port) {
  const auto health = send_http_request(options.host, port, "GET", "/health", {},
                                        {{"Content-Type", "application/json"}});
  if (health.is_ok() && health.value().status >= 200 && health.value().status < 500) {
    return true;
  }

  auto executable = resolve_llama_cpp_server_executable(options);
  if (!executable.is_ok()) {
    return executable.status();
  }

  auto started = start_server_process(options, executable.value(), port);
  if (!started.is_ok()) {
    return started.status();
  }

  for (int attempt = 0; attempt < 240; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const auto probe = send_http_request(options.host, port, "GET", "/health", {},
                                         {{"Content-Type", "application/json"}});
    if (probe.is_ok() && probe.value().status >= 200 && probe.value().status < 500) {
      return true;
    }
  }
  std::ostringstream message;
  message << "llama.cpp server did not become ready on " << options.host << ':' << port
          << " after launch. Build it with `cmake -S "
          << shell_quote(options.llama_cpp_dir)
          << " -B " << shell_quote(options.llama_cpp_dir / "build")
          << " && cmake --build " << shell_quote(options.llama_cpp_dir / "build")
          << " --target llama-server`.";
  return Status::unavailable(message.str());
}

int effective_port(const LlamaCppServerOptions& options) {
  if (options.port > 0) {
    return options.port;
  }
  const char* value = std::getenv("KRAKEN_LLAMA_SERVER_PORT");
  if (value != nullptr && *value != '\0') {
    try {
      const auto parsed = std::stoi(value);
      if (parsed > 0 && parsed <= 65535) {
        return parsed;
      }
    } catch (const std::exception&) {
    }
  }
  return kDefaultLlamaServerPort;
}

std::string json_escape(std::string_view text) {
  std::string output;
  output.reserve(text.size() + 8U);
  for (const char ch : text) {
    switch (ch) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20U) {
          constexpr char kHex[] = "0123456789abcdef";
          output += "\\u00";
          output.push_back(kHex[(static_cast<unsigned char>(ch) >> 4U) & 0xFU]);
          output.push_back(kHex[static_cast<unsigned char>(ch) & 0xFU]);
        } else {
          output.push_back(ch);
        }
    }
  }
  return output;
}

std::string make_generation_body(const LlamaCppGenerationRequest& request) {
  std::ostringstream body;
  if (request.messages.empty()) {
    body << "{\"model\":\"" << json_escape(request.server.model_alias) << "\",\"prompt\":\""
         << json_escape(request.prompt) << "\",\"max_tokens\":" << request.max_new_tokens
         << ",\"temperature\":" << request.temperature << ",\"top_p\":" << request.top_p
         << ",\"stream\":" << (request.stream ? "true" : "false");
    if (request.seed_set) {
      body << ",\"seed\":" << request.seed;
    }
    body << '}';
    return body.str();
  }

  body << "{\"model\":\"" << json_escape(request.server.model_alias) << "\",\"messages\":[";
  for (std::size_t i = 0; i < request.messages.size(); ++i) {
    if (i != 0) {
      body << ',';
    }
    body << "{\"role\":\"" << json_escape(request.messages[i].role) << "\",\"content\":\""
         << json_escape(request.messages[i].content) << "\"}";
  }
  body << "],\"max_tokens\":" << request.max_new_tokens
       << ",\"temperature\":" << request.temperature << ",\"top_p\":" << request.top_p
       << ",\"stream\":" << (request.stream ? "true" : "false");
  if (request.seed_set) {
    body << ",\"seed\":" << request.seed;
  }
  body << '}';
  return body.str();
}

std::optional<std::string> extract_json_string(std::string_view body, std::string_view key,
                                               std::size_t start = 0) {
  const auto marker = "\"" + std::string{key} + "\"";
  const auto key_pos = body.find(marker, start);
  if (key_pos == std::string_view::npos) {
    return std::nullopt;
  }
  const auto colon = body.find(':', key_pos + marker.size());
  if (colon == std::string_view::npos) {
    return std::nullopt;
  }
  auto pos = colon + 1U;
  while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos])) != 0) {
    ++pos;
  }
  if (pos >= body.size() || body[pos] != '"') {
    return std::nullopt;
  }
  ++pos;

  std::string output;
  while (pos < body.size()) {
    const char ch = body[pos++];
    if (ch == '"') {
      return output;
    }
    if (ch != '\\') {
      output.push_back(ch);
      continue;
    }
    if (pos >= body.size()) {
      return std::nullopt;
    }
    const char escaped = body[pos++];
    switch (escaped) {
      case '"':
      case '\\':
      case '/':
        output.push_back(escaped);
        break;
      case 'b':
        output.push_back('\b');
        break;
      case 'f':
        output.push_back('\f');
        break;
      case 'n':
        output.push_back('\n');
        break;
      case 'r':
        output.push_back('\r');
        break;
      case 't':
        output.push_back('\t');
        break;
      default:
        break;
    }
  }
  return std::nullopt;
}

std::size_t extract_json_size(std::string_view body, std::string_view key) {
  const auto marker = "\"" + std::string{key} + "\"";
  const auto key_pos = body.find(marker);
  if (key_pos == std::string_view::npos) {
    return 0;
  }
  const auto colon = body.find(':', key_pos + marker.size());
  if (colon == std::string_view::npos) {
    return 0;
  }
  auto pos = colon + 1U;
  while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos])) != 0) {
    ++pos;
  }
  std::size_t value = 0;
  while (pos < body.size() && std::isdigit(static_cast<unsigned char>(body[pos])) != 0) {
    value = value * 10U + static_cast<std::size_t>(body[pos] - '0');
    ++pos;
  }
  return value;
}

Result<LlamaCppGenerationResult> parse_generation_response(const LlamaCppHttpResponse& response,
                                                           bool chat) {
  if (response.status < 200 || response.status >= 300) {
    return Status::internal_error("llama.cpp generation failed with HTTP " +
                                  std::to_string(response.status) + ": " + response.body);
  }
  LlamaCppGenerationResult result;
  if (chat) {
    const auto message_pos = response.body.find("\"message\"");
    auto content = extract_json_string(response.body, "content", message_pos);
    if (content.has_value()) {
      result.text = *content;
    }
  } else {
    auto text = extract_json_string(response.body, "text");
    if (text.has_value()) {
      result.text = *text;
    }
  }
  if (auto finish = extract_json_string(response.body, "finish_reason"); finish.has_value()) {
    result.finish_reason = *finish;
  }
  result.prompt_tokens = extract_json_size(response.body, "prompt_tokens");
  result.generated_tokens = extract_json_size(response.body, "completion_tokens");
  return result;
}

Result<LlamaCppGenerationResult> parse_stream_response(const LlamaCppHttpResponse& response) {
  if (response.status < 200 || response.status >= 300) {
    return Status::internal_error("llama.cpp streaming failed with HTTP " +
                                  std::to_string(response.status) + ": " + response.body);
  }
  LlamaCppGenerationResult result;
  std::size_t line_start = 0;
  while (line_start < response.body.size()) {
    const auto line_end = response.body.find('\n', line_start);
    std::string_view line{response.body.data() + line_start,
                          line_end == std::string::npos ? response.body.size() - line_start
                                                        : line_end - line_start};
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    if (line.rfind("data: ", 0) == 0 && line != "data: [DONE]") {
      const auto data = line.substr(6);
      auto token = extract_json_string(data, "content");
      if (!token.has_value()) {
        token = extract_json_string(data, "text");
      }
      if (token.has_value()) {
        result.text += *token;
      }
      if (auto finish = extract_json_string(data, "finish_reason"); finish.has_value()) {
        result.finish_reason = *finish;
      }
    }
    if (line_end == std::string::npos) {
      break;
    }
    line_start = line_end + 1U;
  }
  return result;
}

}  // namespace

bool is_gguf_model_path(const std::filesystem::path& path) {
  return resolve_gguf_model_path(path).is_ok();
}

Result<std::filesystem::path> resolve_llama_cpp_server_executable(
  const LlamaCppServerOptions& options) {
  for (const auto& candidate : executable_candidates(options)) {
    if (candidate.filename() == candidate && candidate.parent_path().empty()) {
      auto from_path = find_executable_in_path(candidate.string());
      if (!from_path.empty()) {
        return from_path;
      }
      continue;
    }
    if (std::filesystem::is_regular_file(candidate)) {
      return candidate;
    }
  }
  return Status::unavailable("llama-server executable was not found; set KRAKEN_LLAMA_SERVER "
                             "or build /Users/bill/code/llama.cpp");
}

Result<LlamaCppHttpResponse> forward_openai_request_to_llama_cpp(
  const LlamaCppServerOptions& options, std::string_view method, std::string_view path,
  std::string_view body, const std::vector<std::pair<std::string, std::string>>& headers) {
  const auto port = effective_port(options);
  auto ready = ensure_server_running(options, port);
  if (!ready.is_ok()) {
    return ready.status();
  }
  return send_http_request(options.host, port, method, path, body, headers);
}

Result<LlamaCppGenerationResult> generate_with_llama_cpp(
  const LlamaCppGenerationRequest& request) {
  auto body = make_generation_body(request);
  const auto path = request.messages.empty() ? "/v1/completions" : "/v1/chat/completions";
  auto response = forward_openai_request_to_llama_cpp(
    request.server, "POST", path, body, {{"Content-Type", "application/json"}});
  if (!response.is_ok()) {
    return response.status();
  }
  if (request.stream) {
    auto parsed = parse_stream_response(response.value());
    if (parsed.is_ok() && request.stream_token) {
      request.stream_token(parsed.value().text);
    }
    return parsed;
  }
  return parse_generation_response(response.value(), !request.messages.empty());
}

}  // namespace toyllm
