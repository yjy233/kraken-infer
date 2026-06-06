#include "toyllm/runtime/openai_gateway.hpp"

#include "toyllm/runtime/cpu_inference.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace toyllm {

namespace {

struct JsonValue {
  enum class Type {
    null,
    boolean,
    number,
    string,
    array,
    object,
  };

  Type type{Type::null};
  bool boolean{false};
  double number{0.0};
  std::string string;
  std::vector<JsonValue> array;
  std::vector<std::pair<std::string, JsonValue>> object;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  JsonValue parse() {
    auto value = parse_value();
    skip_ws();
    if (position_ != input_.size()) {
      fail("unexpected trailing JSON input");
    }
    return value;
  }

 private:
  JsonValue parse_value() {
    skip_ws();
    if (position_ >= input_.size()) {
      fail("unexpected end of JSON");
    }
    const char ch = input_[position_];
    if (ch == '"') {
      JsonValue value;
      value.type = JsonValue::Type::string;
      value.string = parse_string();
      return value;
    }
    if (ch == '{') {
      return parse_object();
    }
    if (ch == '[') {
      return parse_array();
    }
    if (ch == '-' || (ch >= '0' && ch <= '9')) {
      return parse_number();
    }
    if (consume_literal("true")) {
      JsonValue value;
      value.type = JsonValue::Type::boolean;
      value.boolean = true;
      return value;
    }
    if (consume_literal("false")) {
      JsonValue value;
      value.type = JsonValue::Type::boolean;
      value.boolean = false;
      return value;
    }
    if (consume_literal("null")) {
      return JsonValue{};
    }
    fail("unsupported JSON value");
  }

  JsonValue parse_object() {
    expect('{');
    JsonValue value;
    value.type = JsonValue::Type::object;
    if (consume('}')) {
      return value;
    }
    while (true) {
      const auto key = parse_string();
      expect(':');
      value.object.emplace_back(key, parse_value());
      if (consume('}')) {
        return value;
      }
      expect(',');
    }
  }

  JsonValue parse_array() {
    expect('[');
    JsonValue value;
    value.type = JsonValue::Type::array;
    if (consume(']')) {
      return value;
    }
    while (true) {
      value.array.push_back(parse_value());
      if (consume(']')) {
        return value;
      }
      expect(',');
    }
  }

  JsonValue parse_number() {
    const auto start = position_;
    if (position_ < input_.size() && input_[position_] == '-') {
      ++position_;
    }
    while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
      ++position_;
    }
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
        ++position_;
      }
    }
    if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
        ++position_;
      }
    }

    JsonValue value;
    value.type = JsonValue::Type::number;
    value.number = std::stod(std::string{input_.substr(start, position_ - start)});
    return value;
  }

  std::string parse_string() {
    expect('"');
    std::string output;
    while (position_ < input_.size()) {
      const char ch = input_[position_++];
      if (ch == '"') {
        return output;
      }
      if (ch != '\\') {
        output.push_back(ch);
        continue;
      }
      if (position_ >= input_.size()) {
        fail("unterminated JSON escape");
      }
      const char escaped = input_[position_++];
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
        case 'u':
          append_unicode_escape(output);
          break;
        default:
          fail("unsupported JSON escape");
      }
    }
    fail("unterminated JSON string");
  }

  void append_unicode_escape(std::string& output) {
    if (position_ + 4U > input_.size()) {
      fail("invalid JSON unicode escape");
    }
    unsigned int codepoint = 0;
    for (int index = 0; index < 4; ++index) {
      const char ch = input_[position_++];
      codepoint <<= 4U;
      if (ch >= '0' && ch <= '9') {
        codepoint += static_cast<unsigned int>(ch - '0');
      } else if (ch >= 'a' && ch <= 'f') {
        codepoint += static_cast<unsigned int>(ch - 'a' + 10);
      } else if (ch >= 'A' && ch <= 'F') {
        codepoint += static_cast<unsigned int>(ch - 'A' + 10);
      } else {
        fail("invalid JSON unicode escape");
      }
    }
    append_utf8(output, codepoint);
  }

  static void append_utf8(std::string& output, unsigned int codepoint) {
    if (codepoint <= 0x7FU) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
      output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
      output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
  }

  void skip_ws() {
    while (position_ < input_.size()) {
      const auto ch = static_cast<unsigned char>(input_[position_]);
      if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') {
        return;
      }
      ++position_;
    }
  }

  bool consume(char expected) {
    skip_ws();
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void expect(char expected) {
    skip_ws();
    if (position_ >= input_.size() || input_[position_] != expected) {
      std::string message = "expected JSON character ";
      message.push_back(expected);
      fail(message);
    }
    ++position_;
  }

  bool consume_literal(std::string_view literal) {
    if (input_.substr(position_, literal.size()) != literal) {
      return false;
    }
    position_ += literal.size();
    return true;
  }

  [[noreturn]] void fail(std::string_view message) const {
    std::ostringstream output;
    output << message << " at byte " << position_;
    throw std::runtime_error(output.str());
  }

  std::string_view input_;
  std::size_t position_{0};
};

const JsonValue* object_get(const JsonValue& value, std::string_view key) {
  if (value.type != JsonValue::Type::object) {
    return nullptr;
  }
  for (const auto& entry : value.object) {
    if (entry.first == key) {
      return &entry.second;
    }
  }
  return nullptr;
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
          output += "\\u00";
          constexpr char kHex[] = "0123456789abcdef";
          output.push_back(kHex[(static_cast<unsigned char>(ch) >> 4U) & 0xFU]);
          output.push_back(kHex[static_cast<unsigned char>(ch) & 0xFU]);
        } else {
          output.push_back(ch);
        }
    }
  }
  return output;
}

std::int64_t unix_time() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::seconds>(now).count();
}

std::string completion_id() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return "chatcmpl-" +
         std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

std::string text_completion_id() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return "cmpl-" +
         std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

std::string string_value(const JsonValue* value, std::string_view fallback = {}) {
  if (value == nullptr || value->type != JsonValue::Type::string) {
    return std::string{fallback};
  }
  return value->string;
}

bool bool_value(const JsonValue* value, bool fallback = false) {
  if (value == nullptr || value->type != JsonValue::Type::boolean) {
    return fallback;
  }
  return value->boolean;
}

std::optional<std::size_t> size_value(const JsonValue* value) {
  if (value == nullptr || value->type != JsonValue::Type::number || value->number < 0.0) {
    return std::nullopt;
  }
  if (value->number > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(value->number);
}

std::optional<std::uint64_t> u64_value(const JsonValue* value) {
  if (value == nullptr || value->type != JsonValue::Type::number || value->number < 0.0) {
    return std::nullopt;
  }
  if (value->number > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(value->number);
}

std::optional<double> double_value(const JsonValue* value) {
  if (value == nullptr || value->type != JsonValue::Type::number) {
    return std::nullopt;
  }
  return value->number;
}

std::string content_to_text(const JsonValue* value) {
  if (value == nullptr || value->type == JsonValue::Type::null) {
    return {};
  }
  if (value->type == JsonValue::Type::string) {
    return value->string;
  }
  if (value->type == JsonValue::Type::array) {
    std::string output;
    for (const auto& item : value->array) {
      if (item.type == JsonValue::Type::object) {
        const auto* text = object_get(item, "text");
        if (text != nullptr && text->type == JsonValue::Type::string) {
          output += text->string;
        }
      }
    }
    return output;
  }
  return {};
}

struct ToolSpec {
  std::string name;
};

struct ChatRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  std::size_t max_tokens{16};
  bool stream{false};
  CpuSamplingConfig sampling;
  Device compute_device{Device::cpu()};
  std::vector<ToolSpec> tools;
  std::optional<std::string> forced_tool;
};

struct CompletionRequest {
  std::string model;
  std::string prompt;
  std::size_t max_tokens{16};
  bool stream{false};
  CpuSamplingConfig sampling;
  Device compute_device{Device::cpu()};
};

std::vector<ToolSpec> parse_tools(const JsonValue* value) {
  std::vector<ToolSpec> tools;
  if (value == nullptr || value->type != JsonValue::Type::array) {
    return tools;
  }
  for (const auto& item : value->array) {
    const auto* function = object_get(item, "function");
    const auto* name = function == nullptr ? nullptr : object_get(*function, "name");
    if (name != nullptr && name->type == JsonValue::Type::string && !name->string.empty()) {
      tools.push_back(ToolSpec{name->string});
    }
  }
  return tools;
}

std::optional<std::string> parse_tool_choice(const JsonValue* value,
                                             const std::vector<ToolSpec>& tools) {
  if (value == nullptr || tools.empty()) {
    return std::nullopt;
  }
  if (value->type == JsonValue::Type::string) {
    if (value->string == "required") {
      return tools.front().name;
    }
    return std::nullopt;
  }
  const auto* function = object_get(*value, "function");
  const auto* name = function == nullptr ? nullptr : object_get(*function, "name");
  if (name != nullptr && name->type == JsonValue::Type::string && !name->string.empty()) {
    return name->string;
  }
  return std::nullopt;
}

std::vector<ChatMessage> parse_messages(const JsonValue* value) {
  if (value == nullptr || value->type != JsonValue::Type::array) {
    throw std::runtime_error("messages must be an array");
  }
  std::vector<ChatMessage> messages;
  for (const auto& item : value->array) {
    if (item.type != JsonValue::Type::object) {
      throw std::runtime_error("message entries must be objects");
    }
    const auto role = string_value(object_get(item, "role"));
    const auto content = content_to_text(object_get(item, "content"));
    if (role == "system" || role == "user" || role == "assistant") {
      messages.push_back(ChatMessage{role, content});
      continue;
    }
    if (role == "tool") {
      const auto tool_id = string_value(object_get(item, "tool_call_id"), "tool");
      messages.push_back(ChatMessage{"user", "Tool result " + tool_id + ":\n" + content});
      continue;
    }
    throw std::runtime_error("unsupported message role: " + role);
  }
  if (messages.empty()) {
    throw std::runtime_error("messages must not be empty");
  }
  return messages;
}

void parse_common_generation_options(const JsonValue& root, const OpenAIGatewayConfig& config,
                                     std::size_t& max_tokens, bool& stream,
                                     Device& compute_device, CpuSamplingConfig& sampling) {
  max_tokens = config.default_max_tokens;
  if (const auto parsed = size_value(object_get(root, "max_tokens")); parsed.has_value()) {
    max_tokens = *parsed;
  }
  if (const auto parsed = size_value(object_get(root, "max_completion_tokens"));
      parsed.has_value()) {
    max_tokens = *parsed;
  }
  stream = bool_value(object_get(root, "stream"), false);
  compute_device = config.compute_device;
  const auto device = string_value(object_get(root, "device"));
  if (device == "cpu") {
    compute_device = Device::cpu();
  } else if (device == "mps" || device == "mps:0") {
    compute_device = Device::mps();
  }
  if (const auto temperature = double_value(object_get(root, "temperature"));
      temperature.has_value()) {
    sampling.do_sample = true;
    sampling.temperature_set = true;
    sampling.temperature = *temperature;
  }
  if (const auto top_p = double_value(object_get(root, "top_p")); top_p.has_value()) {
    sampling.do_sample = true;
    sampling.top_p_set = true;
    sampling.top_p = *top_p;
  }
  if (const auto seed = u64_value(object_get(root, "seed")); seed.has_value()) {
    sampling.do_sample = true;
    sampling.seed_set = true;
    sampling.seed = *seed;
  }
}

std::string parse_prompt(const JsonValue* value) {
  if (value == nullptr) {
    throw std::runtime_error("prompt is required");
  }
  if (value->type == JsonValue::Type::string) {
    if (value->string.empty()) {
      throw std::runtime_error("prompt must not be empty");
    }
    return value->string;
  }
  if (value->type == JsonValue::Type::array) {
    std::string output;
    for (const auto& item : value->array) {
      if (item.type != JsonValue::Type::string) {
        throw std::runtime_error("prompt array entries must be strings");
      }
      if (!output.empty()) {
        output.push_back('\n');
      }
      output += item.string;
    }
    if (output.empty()) {
      throw std::runtime_error("prompt must not be empty");
    }
    return output;
  }
  throw std::runtime_error("prompt must be a string or string array");
}

JsonValue parse_request_object(std::string_view body) {
  const auto root = JsonParser(body).parse();
  if (root.type != JsonValue::Type::object) {
    throw std::runtime_error("request body must be a JSON object");
  }
  return root;
}

ChatRequest parse_chat_request(std::string_view body, const OpenAIGatewayConfig& config) {
  const auto root = parse_request_object(body);
  ChatRequest request;
  request.model = string_value(object_get(root, "model"), config.model_id);
  request.messages = parse_messages(object_get(root, "messages"));
  parse_common_generation_options(root, config, request.max_tokens, request.stream,
                                  request.compute_device, request.sampling);
  request.tools = parse_tools(object_get(root, "tools"));
  request.forced_tool = parse_tool_choice(object_get(root, "tool_choice"), request.tools);
  return request;
}

CompletionRequest parse_completion_request(std::string_view body,
                                           const OpenAIGatewayConfig& config) {
  const auto root = parse_request_object(body);
  CompletionRequest request;
  request.model = string_value(object_get(root, "model"), config.model_id);
  request.prompt = parse_prompt(object_get(root, "prompt"));
  parse_common_generation_options(root, config, request.max_tokens, request.stream,
                                  request.compute_device, request.sampling);
  return request;
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

std::string http_status_text(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 500:
      return "Internal Server Error";
    default:
      return "Error";
  }
}

void send_json_response(int fd, int status, std::string_view body) {
  std::ostringstream response;
  response << "HTTP/1.1 " << status << ' ' << http_status_text(status) << "\r\n";
  response << "Content-Type: application/json\r\n";
  response << "Content-Length: " << body.size() << "\r\n";
  response << "Connection: close\r\n\r\n";
  response << body;
  (void)write_all(fd, response.str());
}

void send_html_response(int fd, int status, std::string_view body) {
  std::ostringstream response;
  response << "HTTP/1.1 " << status << ' ' << http_status_text(status) << "\r\n";
  response << "Content-Type: text/html; charset=utf-8\r\n";
  response << "Cache-Control: no-store\r\n";
  response << "Content-Length: " << body.size() << "\r\n";
  response << "Connection: close\r\n\r\n";
  response << body;
  (void)write_all(fd, response.str());
}

void send_sse_headers(int fd) {
  const std::string headers =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n\r\n";
  (void)write_all(fd, headers);
}

void send_sse(int fd, std::string_view json) {
  (void)write_all(fd, "data: ");
  (void)write_all(fd, json);
  (void)write_all(fd, "\n\n");
}

std::string error_body(std::string_view message, std::string_view type = "invalid_request_error") {
  std::ostringstream output;
  output << "{\"error\":{\"message\":\"" << json_escape(message) << "\",\"type\":\""
         << json_escape(type) << "\",\"param\":null,\"code\":null}}";
  return output.str();
}

std::string models_body(const OpenAIGatewayConfig& config) {
  std::ostringstream output;
  output << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(config.model_id)
         << "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"kraken-infer\"}]}";
  return output.str();
}

std::string openapi_body(const OpenAIGatewayConfig& config) {
  std::ostringstream output;
  output << "{\"openapi\":\"3.0.3\",\"info\":{\"title\":\"kraken-infer "
            "OpenAI-compatible Gateway\",\"version\":\"0.1.0\"},\"servers\":[{\"url\":\"http://"
         << json_escape(config.host) << ':' << config.port
         << "\"}],\"paths\":{\"/health\":{\"get\":{\"responses\":{\"200\":{\"description\":\"Gateway "
            "health\"}}}},\"/chat_page\":{\"get\":{\"responses\":{\"200\":{\"description\":\"Browser "
            "chat page\",\"content\":{\"text/html\":{}}}}}},\"/v1/models\":{\"get\":{\"responses\":{\"200\":{\"description\":\"List "
            "models\"}}}},\"/v1/completions\":{\"post\":{\"requestBody\":{\"required\":true,"
            "\"content\":{\"application/json\":{\"schema\":{\"type\":\"object\",\"properties\":{"
            "\"model\":{\"type\":\"string\"},\"prompt\":{\"oneOf\":[{\"type\":\"string\"},"
            "{\"type\":\"array\",\"items\":{\"type\":\"string\"}}]},\"max_tokens\":{\"type\":"
            "\"integer\"},\"temperature\":{\"type\":\"number\"},\"top_p\":{\"type\":\"number\"},"
            "\"stream\":{\"type\":\"boolean\"},\"seed\":{\"type\":\"integer\"},\"device\":{\"type\":"
            "\"string\",\"enum\":[\"cpu\",\"mps\",\"mps:0\"]}},\"required\":[\"prompt\"]}}}},"
            "\"responses\":{\"200\":{\"description\":\"Text completion or SSE stream\"}}}},"
            "\"/v1/chat/completions\":{\"post\":{\"requestBody\":{\"required\":true,\"content\":{"
            "\"application/json\":{\"schema\":{\"type\":\"object\",\"properties\":{\"model\":{\"type\":"
            "\"string\"},\"messages\":{\"type\":\"array\"},\"tools\":{\"type\":\"array\"},"
            "\"tool_choice\":{},\"max_tokens\":{\"type\":\"integer\"},\"max_completion_tokens\":{"
            "\"type\":\"integer\"},\"temperature\":{\"type\":\"number\"},\"top_p\":{\"type\":"
            "\"number\"},\"stream\":{\"type\":\"boolean\"},\"seed\":{\"type\":\"integer\"},"
            "\"device\":{\"type\":\"string\",\"enum\":[\"cpu\",\"mps\",\"mps:0\"]}},\"required\":"
            "[\"messages\"]}}}},\"responses\":{\"200\":{\"description\":\"Chat completion, tool "
            "call, or SSE stream\"}}}}}}";
  return output.str();
}

std::string chat_page_body(const OpenAIGatewayConfig& config) {
  std::ostringstream output;
  output << R"HTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>kraken-infer Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d7dce2;
      --text: #18202a;
      --muted: #667384;
      --accent: #146c5d;
      --accent-dark: #0d4f44;
      --warn: #9b5b00;
      --user: #e4f4ef;
      --assistant: #ffffff;
      --error: #fff0f0;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      line-height: 1.2;
    }
    .status {
      min-width: 64px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      text-align: right;
    }
    .field {
      display: grid;
      gap: 6px;
    }
    label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    input,
    select,
    textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      outline: none;
    }
    input,
    select {
      height: 36px;
      padding: 0 10px;
    }
    textarea {
      min-height: 54px;
      max-height: 180px;
      padding: 10px;
      line-height: 1.4;
      resize: vertical;
    }
    input:focus,
    select:focus,
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(20, 108, 93, 0.14);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 36px;
      color: var(--text);
      font-size: 14px;
    }
    .toggle input {
      width: 18px;
      height: 18px;
    }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      min-height: 100vh;
    }
    .messages {
      overflow: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .empty {
      margin: auto;
      color: var(--muted);
      font-size: 14px;
    }
    .message {
      max-width: min(760px, 92%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: var(--assistant);
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .message.user {
      align-self: flex-end;
      background: var(--user);
      border-color: #b8ddd1;
    }
    .message.assistant {
      align-self: flex-start;
    }
    .message.error {
      align-self: flex-start;
      background: var(--error);
      border-color: #efc3c3;
      color: #7b1d1d;
    }
    form {
      border-top: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 10px;
      align-items: end;
    }
    button {
      height: 42px;
      min-width: 74px;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover {
      background: var(--accent-dark);
    }
    button.secondary {
      background: #fff;
      color: var(--text);
      border-color: var(--line);
    }
    button.secondary:hover {
      background: #f0f2f4;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    @media (max-width: 780px) {
      .app {
        grid-template-columns: 1fr;
      }
      aside {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      main {
        min-height: calc(100vh - 286px);
      }
      .messages {
        padding: 16px;
      }
      form {
        grid-template-columns: 1fr;
      }
      button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>kraken-infer</h1>
        <div class="status" id="status">Ready</div>
      </header>
      <div class="field">
        <label for="model">Model</label>
        <input id="model" autocomplete="off">
      </div>
      <div class="row">
        <div class="field">
          <label for="device">Device</label>
          <select id="device">
            <option value="cpu">cpu</option>
            <option value="mps:0">mps:0</option>
          </select>
        </div>
        <div class="field">
          <label for="tokens">Tokens</label>
          <input id="tokens" type="number" min="1" max="4096" step="1">
        </div>
      </div>
      <div class="field">
        <label for="temperature">Temperature</label>
        <input id="temperature" type="number" min="0" max="2" step="0.1" value="0.6">
      </div>
      <label class="toggle" for="stream">
        <span>Stream</span>
        <input id="stream" type="checkbox" checked>
      </label>
    </aside>
    <main>
      <section class="messages" id="messages">
        <div class="empty" id="empty">No messages</div>
      </section>
      <form id="composer">
        <textarea id="prompt" rows="2" autocomplete="off" spellcheck="true"></textarea>
        <button id="send" type="submit">Send</button>
        <button id="stop" class="secondary" type="button" disabled>Stop</button>
      </form>
    </main>
  </div>
  <script>
    const defaultModel = ")HTML"
         << json_escape(config.model_id) << R"HTML(";
    const defaultDevice = ")HTML"
         << json_escape(config.compute_device.to_string()) << R"HTML(";
    const defaultTokens = )HTML"
         << config.default_max_tokens << R"HTML(;

    const modelInput = document.getElementById("model");
    const deviceInput = document.getElementById("device");
    const tokensInput = document.getElementById("tokens");
    const temperatureInput = document.getElementById("temperature");
    const streamInput = document.getElementById("stream");
    const messagesEl = document.getElementById("messages");
    const emptyEl = document.getElementById("empty");
    const form = document.getElementById("composer");
    const promptInput = document.getElementById("prompt");
    const sendButton = document.getElementById("send");
    const stopButton = document.getElementById("stop");
    const statusEl = document.getElementById("status");

    const messages = [];
    let controller = null;

    modelInput.value = defaultModel;
    tokensInput.value = String(defaultTokens);
    deviceInput.value = defaultDevice === "mps" ? "mps:0" : defaultDevice;

    function setStatus(value) {
      statusEl.textContent = value;
    }

    function setBusy(value) {
      sendButton.disabled = value;
      stopButton.disabled = !value;
      promptInput.disabled = value;
      setStatus(value ? "Busy" : "Ready");
    }

    function scrollToBottom() {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function addBubble(role, text) {
      emptyEl.hidden = true;
      const node = document.createElement("div");
      node.className = "message " + role;
      node.textContent = text;
      messagesEl.appendChild(node);
      scrollToBottom();
      return node;
    }

    function requestBody(history) {
      const maxTokens = Math.max(1, Number.parseInt(tokensInput.value || "1", 10));
      const temperature = Number.parseFloat(temperatureInput.value);
      const body = {
        model: modelInput.value.trim() || defaultModel,
        messages: history,
        max_tokens: maxTokens,
        stream: streamInput.checked,
        device: deviceInput.value
      };
      if (Number.isFinite(temperature)) {
        body.temperature = temperature;
      }
      return body;
    }

    function contentFromPayload(payload) {
      return payload && payload.choices && payload.choices[0] &&
        payload.choices[0].message && payload.choices[0].message.content || "";
    }

    async function readStream(response, bubble) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let text = "";
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) {
          break;
        }
        buffer += decoder.decode(chunk.value, {stream: true});
        let split = buffer.indexOf("\n\n");
        while (split !== -1) {
          const eventText = buffer.slice(0, split);
          buffer = buffer.slice(split + 2);
          for (const rawLine of eventText.split("\n")) {
            const line = rawLine.trim();
            if (!line.startsWith("data:")) {
              continue;
            }
            const data = line.slice(5).trim();
            if (data === "[DONE]") {
              return text;
            }
            const payload = JSON.parse(data);
            const delta = payload && payload.choices && payload.choices[0] &&
              payload.choices[0].delta && payload.choices[0].delta.content || "";
            if (delta) {
              text += delta;
              bubble.textContent = text;
              scrollToBottom();
            }
          }
          split = buffer.indexOf("\n\n");
        }
      }
      return text;
    }

    async function sendMessage(text) {
      const nextMessages = messages.concat([{role: "user", content: text}]);
      addBubble("user", text);
      const assistantBubble = addBubble("assistant", "");
      controller = new AbortController();
      setBusy(true);
      try {
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(requestBody(nextMessages)),
          signal: controller.signal
        });
        if (!response.ok) {
          let errorText = "HTTP " + response.status;
          try {
            const payload = await response.json();
            errorText = payload.error && payload.error.message || errorText;
          } catch (_) {
          }
          throw new Error(errorText);
        }
        let answer = "";
        if (streamInput.checked) {
          answer = await readStream(response, assistantBubble);
        } else {
          const payload = await response.json();
          answer = contentFromPayload(payload);
          assistantBubble.textContent = answer;
        }
        messages.push({role: "user", content: text});
        messages.push({role: "assistant", content: answer});
        scrollToBottom();
      } catch (error) {
        assistantBubble.remove();
        addBubble("error", error.name === "AbortError" ? "Stopped" : error.message);
      } finally {
        controller = null;
        setBusy(false);
        promptInput.focus();
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = promptInput.value.trim();
      if (!text || controller) {
        return;
      }
      promptInput.value = "";
      void sendMessage(text);
    });

    stopButton.addEventListener("click", () => {
      if (controller) {
        controller.abort();
      }
    });

    promptInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
  </script>
</body>
</html>
)HTML";
  return output.str();
}

std::string completion_body(std::string_view id, std::int64_t created, std::string_view model,
                            std::string_view text) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"text_completion\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"text\":\"" << json_escape(text)
         << "\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,"
            "\"completion_tokens\":0,\"total_tokens\":0}}";
  return output.str();
}

std::string chat_completion_body(std::string_view id, std::int64_t created,
                                 std::string_view model, std::string_view content) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
         << json_escape(content)
         << "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,"
            "\"completion_tokens\":0,\"total_tokens\":0}}";
  return output.str();
}

std::string tool_call_body(std::string_view id, std::int64_t created, std::string_view model,
                           std::string_view tool_name) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
            "\"content\":null,\"tool_calls\":[{\"id\":\"call_0\",\"type\":\"function\","
            "\"function\":{\"name\":\""
         << json_escape(tool_name)
         << "\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}],"
            "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}";
  return output.str();
}

std::string completion_stream_chunk(std::string_view id, std::int64_t created,
                                    std::string_view model, std::string_view text,
                                    std::string_view finish_reason) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"text_completion\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"text\":\"" << json_escape(text)
         << "\",\"finish_reason\":";
  if (finish_reason.empty()) {
    output << "null";
  } else {
    output << '"' << json_escape(finish_reason) << '"';
  }
  output << "}]}";
  return output.str();
}

std::string stream_chunk(std::string_view id, std::int64_t created, std::string_view model,
                         std::string_view delta, bool role, std::string_view finish_reason) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion.chunk\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"delta\":{";
  bool wrote = false;
  if (role) {
    output << "\"role\":\"assistant\"";
    wrote = true;
  }
  if (!delta.empty()) {
    if (wrote) {
      output << ',';
    }
    output << "\"content\":\"" << json_escape(delta) << "\"";
  }
  output << "},\"finish_reason\":";
  if (finish_reason.empty()) {
    output << "null";
  } else {
    output << '"' << json_escape(finish_reason) << '"';
  }
  output << "}]}";
  return output.str();
}

std::string tool_stream_chunk(std::string_view id, std::int64_t created, std::string_view model,
                              std::string_view tool_name, bool finish) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion.chunk\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"delta\":";
  if (finish) {
    output << "{}";
  } else {
    output << "{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_0\","
              "\"type\":\"function\",\"function\":{\"name\":\""
           << json_escape(tool_name) << "\",\"arguments\":\"{}\"}}]}";
  }
  output << ",\"finish_reason\":";
  output << (finish ? "\"tool_calls\"" : "null") << "}]}";
  return output.str();
}

struct HttpRequest {
  std::string method;
  std::string path;
  std::string body;
};

std::string lower_ascii(std::string value) {
  for (auto& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

HttpRequest read_http_request(int fd) {
  std::string buffer;
  char chunk[4096];
  while (buffer.find("\r\n\r\n") == std::string::npos) {
    const auto count = ::recv(fd, chunk, sizeof(chunk), 0);
    if (count <= 0) {
      throw std::runtime_error("failed to read HTTP request");
    }
    buffer.append(chunk, static_cast<std::size_t>(count));
    if (buffer.size() > 1024U * 1024U) {
      throw std::runtime_error("HTTP headers exceed limit");
    }
  }

  const auto header_end = buffer.find("\r\n\r\n");
  const auto header_text = buffer.substr(0, header_end);
  std::istringstream header_stream(header_text);
  std::string request_line;
  std::getline(header_stream, request_line);
  if (!request_line.empty() && request_line.back() == '\r') {
    request_line.pop_back();
  }
  std::istringstream line_stream(request_line);
  HttpRequest request;
  line_stream >> request.method >> request.path;
  if (request.method.empty() || request.path.empty()) {
    throw std::runtime_error("invalid HTTP request line");
  }

  std::size_t content_length = 0;
  std::string line;
  while (std::getline(header_stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    auto name = lower_ascii(line.substr(0, colon));
    auto value = line.substr(colon + 1U);
    while (!value.empty() && value.front() == ' ') {
      value.erase(value.begin());
    }
    if (name == "content-length") {
      content_length = static_cast<std::size_t>(std::stoull(value));
    }
  }

  const auto body_start = header_end + 4U;
  while (buffer.size() < body_start + content_length) {
    const auto count = ::recv(fd, chunk, sizeof(chunk), 0);
    if (count <= 0) {
      throw std::runtime_error("failed to read HTTP body");
    }
    buffer.append(chunk, static_cast<std::size_t>(count));
    if (buffer.size() > 8U * 1024U * 1024U) {
      throw std::runtime_error("HTTP body exceeds limit");
    }
  }
  request.body = buffer.substr(body_start, content_length);
  const auto query = request.path.find('?');
  if (query != std::string::npos) {
    request.path.resize(query);
  }
  return request;
}

void send_forced_tool_response(int fd, const ChatRequest& request) {
  const auto id = completion_id();
  const auto created = unix_time();
  const auto tool_name = *request.forced_tool;
  if (!request.stream) {
    send_json_response(fd, 200, tool_call_body(id, created, request.model, tool_name));
    return;
  }
  send_sse_headers(fd);
  send_sse(fd, tool_stream_chunk(id, created, request.model, tool_name, false));
  send_sse(fd, tool_stream_chunk(id, created, request.model, tool_name, true));
  send_sse(fd, "[DONE]");
}

void send_completion_response(int fd, const OpenAIGatewayConfig& config,
                              const CompletionRequest& request) {
  CpuGenerationRequest generation;
  generation.model_dir = config.model_dir;
  generation.prompt = request.prompt;
  generation.max_new_tokens = request.max_tokens;
  generation.compute_device = request.compute_device;
  generation.sampling = request.sampling;

  const auto id = text_completion_id();
  const auto created = unix_time();
  if (!request.stream) {
    const auto result = generate_cpu(generation);
    if (!result.is_ok()) {
      send_json_response(fd, 500, error_body(result.status().message(), "server_error"));
      return;
    }
    send_json_response(fd, 200, completion_body(id, created, request.model, result.value().text));
    return;
  }

  send_sse_headers(fd);
  generation.stream_token = [&](std::string_view token) {
    send_sse(fd, completion_stream_chunk(id, created, request.model, token, {}));
  };
  const auto result = generate_cpu(generation);
  if (!result.is_ok()) {
    send_sse(fd, error_body(result.status().message(), "server_error"));
    send_sse(fd, "[DONE]");
    return;
  }
  send_sse(fd, completion_stream_chunk(id, created, request.model, {}, "stop"));
  send_sse(fd, "[DONE]");
}

void send_chat_response(int fd, const OpenAIGatewayConfig& config, const ChatRequest& request) {
  if (request.forced_tool.has_value()) {
    send_forced_tool_response(fd, request);
    return;
  }

  CpuGenerationRequest generation;
  generation.model_dir = config.model_dir;
  generation.messages = request.messages;
  generation.max_new_tokens = request.max_tokens;
  generation.compute_device = request.compute_device;
  generation.sampling = request.sampling;

  const auto id = completion_id();
  const auto created = unix_time();
  if (!request.stream) {
    const auto result = generate_cpu(generation);
    if (!result.is_ok()) {
      send_json_response(fd, 500, error_body(result.status().message(), "server_error"));
      return;
    }
    send_json_response(fd, 200,
                       chat_completion_body(id, created, request.model, result.value().text));
    return;
  }

  send_sse_headers(fd);
  send_sse(fd, stream_chunk(id, created, request.model, {}, true, {}));
  generation.stream_token = [&](std::string_view token) {
    send_sse(fd, stream_chunk(id, created, request.model, token, false, {}));
  };
  const auto result = generate_cpu(generation);
  if (!result.is_ok()) {
    send_sse(fd, error_body(result.status().message(), "server_error"));
    send_sse(fd, "[DONE]");
    return;
  }
  send_sse(fd, stream_chunk(id, created, request.model, {}, false, "stop"));
  send_sse(fd, "[DONE]");
}

void handle_client(int fd, const OpenAIGatewayConfig& config) {
  try {
    const auto request = read_http_request(fd);
    if (request.path == "/health" || request.path == "/v1/health") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_json_response(fd, 200, "{\"status\":\"ok\"}");
      return;
    }
    if (request.path == "/v1/models") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_json_response(fd, 200, models_body(config));
      return;
    }
    if (request.path == "/chat_page") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_html_response(fd, 200, chat_page_body(config));
      return;
    }
    if (request.path == "/openapi.json" || request.path == "/v1/openapi.json") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_json_response(fd, 200, openapi_body(config));
      return;
    }
    if (request.path == "/v1/completions") {
      if (request.method != "POST") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      const auto completion_request = parse_completion_request(request.body, config);
      send_completion_response(fd, config, completion_request);
      return;
    }
    if (request.path == "/v1/chat/completions") {
      if (request.method != "POST") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      const auto chat_request = parse_chat_request(request.body, config);
      send_chat_response(fd, config, chat_request);
      return;
    }
    send_json_response(fd, 404, error_body("route not found"));
  } catch (const std::exception& error) {
    send_json_response(fd, 400, error_body(error.what()));
  }
}

Status errno_status(std::string_view operation) {
  return Status::internal_error(std::string{operation} + ": " + std::strerror(errno));
}

}  // namespace

Status serve_openai_gateway(const OpenAIGatewayConfig& config) {
  if (config.port <= 0 || config.port > 65535) {
    return Status::invalid_argument("gateway port must be in 1..65535");
  }

  const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    return errno_status("socket failed");
  }

  const int reuse = 1;
  (void)::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(config.port));
  if (::inet_pton(AF_INET, config.host.c_str(), &address.sin_addr) != 1) {
    ::close(server_fd);
    return Status::invalid_argument("gateway host must be an IPv4 address");
  }
  if (::bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    const auto status = errno_status("bind failed");
    ::close(server_fd);
    return status;
  }
  if (::listen(server_fd, 16) != 0) {
    const auto status = errno_status("listen failed");
    ::close(server_fd);
    return status;
  }

  std::cout << "OpenAI-compatible gateway listening on http://" << config.host << ':'
            << config.port << '\n';
  std::cout << "model: " << config.model_id << '\n';
  std::cout << "device: " << config.compute_device.to_string() << '\n';

  while (true) {
    sockaddr_in client_address{};
    socklen_t client_length = sizeof(client_address);
    const int client_fd =
      ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_address), &client_length);
    if (client_fd < 0) {
      if (errno == EINTR) {
        continue;
      }
      const auto status = errno_status("accept failed");
      ::close(server_fd);
      return status;
    }
    handle_client(client_fd, config);
    ::close(client_fd);
  }
}

}  // namespace toyllm
