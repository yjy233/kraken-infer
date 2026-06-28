#include "toyllm/model/model_config.hpp"

#include "toyllm/runtime/gguf_reader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace toyllm {

namespace {

struct JsonValue {
  using Array = std::vector<JsonValue>;
  using Object = std::map<std::string, JsonValue>;
  using Value = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

  Value value{nullptr};
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  Result<JsonValue> parse() {
    try {
      JsonValue value = parse_value();
      skip_ws();
      if (position_ != input_.size()) {
        return Status::invalid_argument("unexpected trailing JSON content");
      }
      return value;
    } catch (const std::exception& error) {
      return Status::invalid_argument(error.what());
    }
  }

 private:
  JsonValue parse_value() {
    skip_ws();
    if (position_ >= input_.size()) {
      fail("unexpected end of JSON input");
    }

    const char ch = input_[position_];
    if (ch == '{') {
      return JsonValue{parse_object()};
    }
    if (ch == '[') {
      return JsonValue{parse_array()};
    }
    if (ch == '"') {
      return JsonValue{parse_string()};
    }
    if (ch == 't') {
      consume_literal("true");
      return JsonValue{true};
    }
    if (ch == 'f') {
      consume_literal("false");
      return JsonValue{false};
    }
    if (ch == 'n') {
      consume_literal("null");
      return JsonValue{nullptr};
    }
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch)) != 0) {
      return JsonValue{parse_number()};
    }

    fail("unexpected JSON value");
  }

  JsonValue::Object parse_object() {
    expect('{');
    JsonValue::Object object;
    skip_ws();
    if (try_consume('}')) {
      return object;
    }

    while (true) {
      skip_ws();
      if (position_ >= input_.size() || input_[position_] != '"') {
        fail("expected JSON object key");
      }
      std::string key = parse_string();
      skip_ws();
      expect(':');
      JsonValue value = parse_value();
      object.emplace(std::move(key), std::move(value));
      skip_ws();
      if (try_consume('}')) {
        return object;
      }
      expect(',');
    }
  }

  JsonValue::Array parse_array() {
    expect('[');
    JsonValue::Array array;
    skip_ws();
    if (try_consume(']')) {
      return array;
    }

    while (true) {
      array.push_back(parse_value());
      skip_ws();
      if (try_consume(']')) {
        return array;
      }
      expect(',');
    }
  }

  std::string parse_string() {
    expect('"');
    std::string result;
    while (position_ < input_.size()) {
      const char ch = input_[position_++];
      if (ch == '"') {
        return result;
      }
      if (ch != '\\') {
        result.push_back(ch);
        continue;
      }

      if (position_ >= input_.size()) {
        fail("unterminated JSON escape sequence");
      }
      const char escaped = input_[position_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          result.push_back(escaped);
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u':
          append_unicode_escape(result);
          break;
        default:
          fail("unsupported JSON escape sequence");
      }
    }
    fail("unterminated JSON string");
  }

  double parse_number() {
    const auto start = position_;
    if (try_consume('-')) {
      if (position_ >= input_.size()) {
        fail("invalid JSON number");
      }
    }

    if (position_ < input_.size() && input_[position_] == '0') {
      ++position_;
    } else {
      if (position_ >= input_.size() ||
          std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
        fail("invalid JSON number");
      }
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
        ++position_;
      }
    }

    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      if (position_ >= input_.size() ||
          std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
        fail("invalid JSON fraction");
      }
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
      if (position_ >= input_.size() ||
          std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
        fail("invalid JSON exponent");
      }
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
        ++position_;
      }
    }

    const auto number = std::string(input_.substr(start, position_ - start));
    return std::stod(number);
  }

  void append_unicode_escape(std::string& result) {
    if (position_ + 4 > input_.size()) {
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

    if (codepoint <= 0x7F) {
      result.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
      result.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
      result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
      result.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
      result.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
  }

  void consume_literal(std::string_view literal) {
    if (input_.substr(position_, literal.size()) != literal) {
      fail("invalid JSON literal");
    }
    position_ += literal.size();
  }

  void skip_ws() {
    while (position_ < input_.size()) {
      const auto ch = static_cast<unsigned char>(input_[position_]);
      if (std::isspace(ch) == 0) {
        break;
      }
      ++position_;
    }
  }

  void expect(char expected) {
    skip_ws();
    if (position_ >= input_.size() || input_[position_] != expected) {
      std::string message = "expected JSON character '";
      message.push_back(expected);
      message.push_back('\'');
      fail(message);
    }
    ++position_;
  }

  bool try_consume(char expected) {
    skip_ws();
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  [[noreturn]] void fail(std::string_view message) const {
    std::ostringstream output;
    output << message << " at byte " << position_;
    throw std::runtime_error(output.str());
  }

  std::string_view input_;
  std::size_t position_{0};
};

Result<std::string> read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return Status::unavailable("failed to open " + path.string());
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input.good() && !input.eof()) {
    return Status::internal_error("failed to read " + path.string());
  }

  return buffer.str();
}

Result<JsonValue> parse_json_file(const std::filesystem::path& path) {
  auto content = read_text_file(path);
  if (!content.is_ok()) {
    return content.status();
  }
  auto parsed = JsonParser(content.value()).parse();
  if (!parsed.is_ok()) {
    return Status::invalid_argument(path.string() + ": " + parsed.status().message());
  }
  return parsed.value();
}

const JsonValue::Object* as_object(const JsonValue& value) {
  return std::get_if<JsonValue::Object>(&value.value);
}

const JsonValue::Array* as_array(const JsonValue& value) {
  return std::get_if<JsonValue::Array>(&value.value);
}

const std::string* as_string(const JsonValue& value) {
  return std::get_if<std::string>(&value.value);
}

const bool* as_bool(const JsonValue& value) {
  return std::get_if<bool>(&value.value);
}

const double* as_number(const JsonValue& value) {
  return std::get_if<double>(&value.value);
}

const JsonValue* find_field(const JsonValue::Object& object, std::string_view key) {
  const auto iterator = object.find(std::string(key));
  if (iterator == object.end()) {
    return nullptr;
  }
  return &iterator->second;
}

Result<const JsonValue::Object*> require_object_root(const JsonValue& value, std::string_view path) {
  const auto* object = as_object(value);
  if (object == nullptr) {
    return Status::invalid_argument(std::string(path) + " must contain a JSON object");
  }
  return object;
}

Result<std::string> require_string(const JsonValue::Object& object, std::string_view key) {
  const JsonValue* field = find_field(object, key);
  if (field == nullptr) {
    return Status::invalid_argument("missing string field: " + std::string(key));
  }
  const auto* value = as_string(*field);
  if (value == nullptr) {
    return Status::invalid_argument("field must be a string: " + std::string(key));
  }
  return *value;
}

Result<std::int64_t> require_int(const JsonValue::Object& object, std::string_view key) {
  const JsonValue* field = find_field(object, key);
  if (field == nullptr) {
    return Status::invalid_argument("missing integer field: " + std::string(key));
  }
  const auto* value = as_number(*field);
  if (value == nullptr || std::floor(*value) != *value) {
    return Status::invalid_argument("field must be an integer: " + std::string(key));
  }
  return static_cast<std::int64_t>(*value);
}

Result<double> require_number(const JsonValue::Object& object, std::string_view key) {
  const JsonValue* field = find_field(object, key);
  if (field == nullptr) {
    return Status::invalid_argument("missing number field: " + std::string(key));
  }
  const auto* value = as_number(*field);
  if (value == nullptr) {
    return Status::invalid_argument("field must be a number: " + std::string(key));
  }
  return *value;
}

Result<bool> require_bool(const JsonValue::Object& object, std::string_view key) {
  const JsonValue* field = find_field(object, key);
  if (field == nullptr) {
    return Status::invalid_argument("missing bool field: " + std::string(key));
  }
  const auto* value = as_bool(*field);
  if (value == nullptr) {
    return Status::invalid_argument("field must be a bool: " + std::string(key));
  }
  return *value;
}

Result<std::string> optional_string(const JsonValue::Object& object, std::string_view key,
                                    std::string fallback) {
  const JsonValue* field = find_field(object, key);
  if (field == nullptr) {
    return fallback;
  }
  const auto* value = as_string(*field);
  if (value == nullptr) {
    return Status::invalid_argument("field must be a string: " + std::string(key));
  }
  return *value;
}

Result<std::vector<std::int64_t>> parse_eos_ids(const JsonValue::Object& object) {
  const JsonValue* field = find_field(object, "eos_token_id");
  if (field == nullptr) {
    return Status::invalid_argument("missing eos_token_id");
  }

  if (const auto* number = as_number(*field); number != nullptr) {
    if (std::floor(*number) != *number) {
      return Status::invalid_argument("eos_token_id must be an integer or integer array");
    }
    return std::vector<std::int64_t>{static_cast<std::int64_t>(*number)};
  }

  const auto* array = as_array(*field);
  if (array == nullptr) {
    return Status::invalid_argument("eos_token_id must be an integer or integer array");
  }

  std::vector<std::int64_t> result;
  result.reserve(array->size());
  for (const JsonValue& item : *array) {
    const auto* number = as_number(item);
    if (number == nullptr || std::floor(*number) != *number) {
      return Status::invalid_argument("eos_token_id array must contain only integers");
    }
    result.push_back(static_cast<std::int64_t>(*number));
  }
  return result;
}

Result<ModelConfig> parse_model_config(const JsonValue& json) {
  auto root = require_object_root(json, "config.json");
  if (!root.is_ok()) {
    return root.status();
  }
  const JsonValue::Object& object = *root.value();

  ModelConfig config{};

#define KRAKEN_INFER_ASSIGN_REQUIRED_STRING(field, key) \
  do {                                            \
    auto result = require_string(object, key);    \
    if (!result.is_ok()) {                        \
      return result.status();                     \
    }                                             \
    config.field = result.value();                \
  } while (false)

#define KRAKEN_INFER_ASSIGN_REQUIRED_INT(field, key) \
  do {                                         \
    auto result = require_int(object, key);    \
    if (!result.is_ok()) {                     \
      return result.status();                  \
    }                                          \
    config.field = result.value();             \
  } while (false)

#define KRAKEN_INFER_ASSIGN_REQUIRED_NUMBER(field, key) \
  do {                                            \
    auto result = require_number(object, key);    \
    if (!result.is_ok()) {                        \
      return result.status();                     \
    }                                             \
    config.field = result.value();                \
  } while (false)

#define KRAKEN_INFER_ASSIGN_REQUIRED_BOOL(field, key) \
  do {                                          \
    auto result = require_bool(object, key);    \
    if (!result.is_ok()) {                      \
      return result.status();                   \
    }                                           \
    config.field = result.value();              \
  } while (false)

  const JsonValue* architectures = find_field(object, "architectures");
  if (architectures == nullptr) {
    return Status::invalid_argument("missing architectures");
  }
  const auto* architecture_array = as_array(*architectures);
  if (architecture_array == nullptr || architecture_array->empty()) {
    return Status::invalid_argument("architectures must be a non-empty string array");
  }
  const auto* architecture = as_string(architecture_array->front());
  if (architecture == nullptr) {
    return Status::invalid_argument("architectures[0] must be a string");
  }
  config.architecture = *architecture;

  KRAKEN_INFER_ASSIGN_REQUIRED_BOOL(attention_bias, "attention_bias");
  KRAKEN_INFER_ASSIGN_REQUIRED_NUMBER(attention_dropout, "attention_dropout");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(bos_token_id, "bos_token_id");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(eos_token_id, "eos_token_id");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(head_dim, "head_dim");
  KRAKEN_INFER_ASSIGN_REQUIRED_STRING(hidden_act, "hidden_act");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(hidden_size, "hidden_size");
  KRAKEN_INFER_ASSIGN_REQUIRED_NUMBER(initializer_range, "initializer_range");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(intermediate_size, "intermediate_size");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(max_position_embeddings, "max_position_embeddings");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(max_window_layers, "max_window_layers");
  KRAKEN_INFER_ASSIGN_REQUIRED_STRING(model_type, "model_type");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(num_attention_heads, "num_attention_heads");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(num_hidden_layers, "num_hidden_layers");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(num_key_value_heads, "num_key_value_heads");
  KRAKEN_INFER_ASSIGN_REQUIRED_NUMBER(rms_norm_eps, "rms_norm_eps");
  KRAKEN_INFER_ASSIGN_REQUIRED_NUMBER(rope_theta, "rope_theta");
  KRAKEN_INFER_ASSIGN_REQUIRED_BOOL(tie_word_embeddings, "tie_word_embeddings");
  KRAKEN_INFER_ASSIGN_REQUIRED_STRING(torch_dtype, "torch_dtype");
  KRAKEN_INFER_ASSIGN_REQUIRED_STRING(transformers_version, "transformers_version");
  KRAKEN_INFER_ASSIGN_REQUIRED_BOOL(use_cache, "use_cache");
  KRAKEN_INFER_ASSIGN_REQUIRED_BOOL(use_sliding_window, "use_sliding_window");
  KRAKEN_INFER_ASSIGN_REQUIRED_INT(vocab_size, "vocab_size");

#undef KRAKEN_INFER_ASSIGN_REQUIRED_STRING
#undef KRAKEN_INFER_ASSIGN_REQUIRED_INT
#undef KRAKEN_INFER_ASSIGN_REQUIRED_NUMBER
#undef KRAKEN_INFER_ASSIGN_REQUIRED_BOOL

  return config;
}

Result<GenerationConfig> parse_generation_config(const JsonValue& json) {
  auto root = require_object_root(json, "generation_config.json");
  if (!root.is_ok()) {
    return root.status();
  }
  const JsonValue::Object& object = *root.value();

  GenerationConfig config{};

  auto bos = require_int(object, "bos_token_id");
  if (!bos.is_ok()) {
    return bos.status();
  }
  config.bos_token_id = bos.value();

  auto do_sample = require_bool(object, "do_sample");
  if (!do_sample.is_ok()) {
    return do_sample.status();
  }
  config.do_sample = do_sample.value();

  auto eos = parse_eos_ids(object);
  if (!eos.is_ok()) {
    return eos.status();
  }
  config.eos_token_ids = eos.value();

  auto pad = require_int(object, "pad_token_id");
  if (!pad.is_ok()) {
    return pad.status();
  }
  config.pad_token_id = pad.value();

  auto temperature = require_number(object, "temperature");
  if (!temperature.is_ok()) {
    return temperature.status();
  }
  config.temperature = temperature.value();

  auto top_k = require_int(object, "top_k");
  if (!top_k.is_ok()) {
    return top_k.status();
  }
  config.top_k = top_k.value();

  auto top_p = require_number(object, "top_p");
  if (!top_p.is_ok()) {
    return top_p.status();
  }
  config.top_p = top_p.value();

  auto transformers_version = optional_string(object, "transformers_version", "");
  if (!transformers_version.is_ok()) {
    return transformers_version.status();
  }
  config.transformers_version = transformers_version.value();

  return config;
}

Result<TokenizerInfo> parse_tokenizer_info(const JsonValue& tokenizer_json,
                                           const JsonValue& tokenizer_config_json,
                                           const JsonValue& vocab_json) {
  auto tokenizer_json_root = require_object_root(tokenizer_json, "tokenizer.json");
  if (!tokenizer_json_root.is_ok()) {
    return tokenizer_json_root.status();
  }

  auto tokenizer_config_root = require_object_root(tokenizer_config_json, "tokenizer_config.json");
  if (!tokenizer_config_root.is_ok()) {
    return tokenizer_config_root.status();
  }

  auto vocab_root = require_object_root(vocab_json, "vocab.json");
  if (!vocab_root.is_ok()) {
    return vocab_root.status();
  }

  const JsonValue* model_field = find_field(*tokenizer_json_root.value(), "model");
  if (model_field == nullptr) {
    return Status::invalid_argument("tokenizer.json missing model");
  }
  const auto* model_object = as_object(*model_field);
  if (model_object == nullptr) {
    return Status::invalid_argument("tokenizer.json model must be an object");
  }

  const JsonValue* model_vocab_field = find_field(*model_object, "vocab");
  if (model_vocab_field == nullptr) {
    return Status::invalid_argument("tokenizer.json model missing vocab");
  }
  const auto* model_vocab = as_object(*model_vocab_field);
  if (model_vocab == nullptr) {
    return Status::invalid_argument("tokenizer.json model.vocab must be an object");
  }

  TokenizerInfo info{};
  info.available = true;
  info.base_vocab_size = static_cast<std::uint64_t>(model_vocab->size());

  if (info.base_vocab_size != static_cast<std::uint64_t>(vocab_root.value()->size())) {
    return Status::invalid_argument("tokenizer.json model.vocab size differs from vocab.json");
  }

  for (const auto& [token, id_value] : *model_vocab) {
    (void)token;
    const auto* id = as_number(id_value);
    if (id == nullptr || std::floor(*id) != *id || *id < 0.0) {
      return Status::invalid_argument("tokenizer.json model.vocab ids must be non-negative integers");
    }
    info.max_token_id = std::max(info.max_token_id, static_cast<std::uint64_t>(*id));
  }

  const JsonValue* added_tokens = find_field(*tokenizer_json_root.value(), "added_tokens");
  if (added_tokens != nullptr) {
    const auto* added_array = as_array(*added_tokens);
    if (added_array == nullptr) {
      return Status::invalid_argument("tokenizer.json added_tokens must be an array");
    }
    info.added_tokens = static_cast<std::uint64_t>(added_array->size());
    for (const JsonValue& item : *added_array) {
      const auto* token = as_object(item);
      if (token == nullptr) {
        return Status::invalid_argument("tokenizer.json added_tokens entries must be objects");
      }
      auto id = require_int(*token, "id");
      if (!id.is_ok()) {
        return id.status();
      }
      if (id.value() < 0) {
        return Status::invalid_argument("tokenizer added token id must be non-negative");
      }
      info.max_token_id =
        std::max(info.max_token_id, static_cast<std::uint64_t>(id.value()));
    }
  }

  const JsonValue* added_decoder = find_field(*tokenizer_config_root.value(), "added_tokens_decoder");
  if (added_decoder != nullptr) {
    const auto* added_object = as_object(*added_decoder);
    if (added_object == nullptr) {
      return Status::invalid_argument("added_tokens_decoder must be an object");
    }
    info.tokenizer_config_added_tokens = static_cast<std::uint64_t>(added_object->size());
  }

  info.total_vocab_size = info.base_vocab_size + info.added_tokens;
  return info;
}

std::string bool_to_string(bool value) {
  return value ? "true" : "false";
}

std::string join_ints(const std::vector<std::int64_t>& values) {
  std::ostringstream output;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      output << ", ";
    }
    output << values[index];
  }
  return output.str();
}

std::string gguf_arch_key(const GgufFile& file, std::string_view suffix) {
  auto arch = gguf_get_string(file, "general.architecture");
  if (!arch.is_ok()) {
    return {};
  }
  std::string key = arch.value();
  key.push_back('.');
  key.append(suffix);
  return key;
}

std::int64_t gguf_optional_i64(const GgufFile& file, std::string_view suffix,
                               std::int64_t fallback) {
  const auto key = gguf_arch_key(file, suffix);
  if (key.empty()) {
    return fallback;
  }
  auto value = gguf_get_i64(file, key);
  if (!value.is_ok()) {
    return fallback;
  }
  return value.value();
}

double gguf_optional_f64(const GgufFile& file, std::string_view suffix, double fallback) {
  const auto key = gguf_arch_key(file, suffix);
  if (key.empty()) {
    return fallback;
  }
  auto value = gguf_get_f64(file, key);
  if (!value.is_ok()) {
    return fallback;
  }
  return value.value();
}

std::vector<std::int64_t> gguf_optional_i64_array(const GgufFile& file, std::string_view suffix,
                                                  std::vector<std::int64_t> fallback) {
  const auto key = gguf_arch_key(file, suffix);
  if (key.empty()) {
    return fallback;
  }
  auto value = gguf_get_i64_array(file, key);
  if (!value.is_ok()) {
    return fallback;
  }
  return value.value();
}

std::uint64_t gguf_optional_array_size(const GgufFile& file, const std::string& key,
                                       std::uint64_t fallback) {
  auto value = gguf_get_array_size(file, key);
  if (!value.is_ok()) {
    return fallback;
  }
  return value.value();
}

Result<ModelBundle> load_gguf_bundle(const std::filesystem::path& input_path) {
  auto gguf_path = resolve_gguf_model_path(input_path);
  if (!gguf_path.is_ok()) {
    return gguf_path.status();
  }
  auto gguf = read_gguf_file(gguf_path.value());
  if (!gguf.is_ok()) {
    return gguf.status();
  }

  auto arch = gguf_get_string(gguf.value(), "general.architecture");
  if (!arch.is_ok()) {
    return arch.status();
  }

  const std::string& arch_name = arch.value();
  if (arch_name != "qwen35" && arch_name != "qwen35moe" && arch_name != "qwen3vl" &&
      arch_name != "qwen3vlmoe" && arch_name != "clip") {
    return Status::invalid_argument("unsupported GGUF architecture: " + arch_name);
  }

  ModelBundle bundle{};
  bundle.model_dir = std::filesystem::is_directory(input_path) ? input_path : gguf_path.value().parent_path();
  bundle.model_file = gguf_path.value();
  bundle.gguf_version = gguf.value().version;
  bundle.gguf_tensor_count = gguf.value().tensor_count;
  bundle.gguf_metadata_count = gguf.value().metadata_count;
  bundle.gguf_file_size = gguf.value().file_size;

  ModelConfig model{};
  model.gguf = true;
  model.architecture = arch_name;
  model.model_type = arch_name;
  model.hidden_act = "silu";
  model.torch_dtype = "gguf";
  model.hidden_size = gguf_optional_i64(gguf.value(), "embedding_length", 0);
  model.intermediate_size = gguf_optional_i64(gguf.value(), "feed_forward_length", 0);
  model.vocab_size = gguf_optional_i64(gguf.value(), "vocab_size", 0);
  if (model.vocab_size == 0) {
    model.vocab_size = static_cast<std::int64_t>(
      gguf_optional_array_size(gguf.value(), "tokenizer.ggml.tokens", 0));
  }
  model.max_position_embeddings = gguf_optional_i64(gguf.value(), "context_length", 0);
  model.num_attention_heads = gguf_optional_i64(gguf.value(), "attention.head_count", 0);
  model.num_key_value_heads = gguf_optional_i64(gguf.value(), "attention.head_count_kv", 0);
  model.num_hidden_layers = gguf_optional_i64(gguf.value(), "block_count", 0);
  model.total_layer_count = model.num_hidden_layers;
  model.rms_norm_eps = gguf_optional_f64(gguf.value(), "attention.layer_norm_rms_epsilon", 0.0);
  model.rope_theta = gguf_optional_f64(gguf.value(), "rope.freq_base", 0.0);
  model.head_dim = gguf_optional_i64(gguf.value(), "attention.key_length", 0);
  if (model.head_dim == 0 && model.num_attention_heads > 0) {
    model.head_dim = model.hidden_size / model.num_attention_heads;
  }

  model.full_attention_interval = gguf_optional_i64(gguf.value(), "full_attention_interval", 4);
  model.linear_conv_kernel_dim = gguf_optional_i64(gguf.value(), "ssm.conv_kernel", 0);
  model.linear_key_head_dim = gguf_optional_i64(gguf.value(), "ssm.state_size", 0);
  model.linear_value_head_dim = model.linear_key_head_dim;
  model.linear_num_key_heads = gguf_optional_i64(gguf.value(), "ssm.group_count", 0);
  model.linear_num_value_heads = gguf_optional_i64(gguf.value(), "ssm.time_step_rank", 0);
  model.linear_inner_size = gguf_optional_i64(gguf.value(), "ssm.inner_size", 0);
  model.mtp_num_hidden_layers = gguf_optional_i64(gguf.value(), "nextn_predict_layers", 0);
  model.main_layer_count = model.num_hidden_layers - model.mtp_num_hidden_layers;
  if (model.main_layer_count < 0) {
    return Status::invalid_argument("GGUF nextn_predict_layers exceeds block_count");
  }
  model.expert_count = gguf_optional_i64(gguf.value(), "expert_count", 0);
  model.expert_used_count = gguf_optional_i64(gguf.value(), "expert_used_count", 0);
  model.expert_feed_forward_length =
    gguf_optional_i64(gguf.value(), "expert_feed_forward_length", 0);
  model.expert_shared_feed_forward_length =
    gguf_optional_i64(gguf.value(), "expert_shared_feed_forward_length", 0);
  model.rope_dimension_sections =
    gguf_optional_i64_array(gguf.value(), "rope.dimension_sections", {11, 11, 10, 0});
  model.attention_recurrent_layers =
    gguf_optional_i64_array(gguf.value(), "attention.recurrent_layers", {});
  if (model.attention_recurrent_layers.empty() && model.main_layer_count > 0) {
    model.attention_recurrent_layers.reserve(static_cast<std::size_t>(model.main_layer_count));
    for (std::int64_t layer = 0; layer < model.main_layer_count; ++layer) {
      model.attention_recurrent_layers.push_back(
        ((layer + 1) % model.full_attention_interval) != 0 ? 1 : 0);
    }
  }

  GenerationConfig generation{};
  generation.temperature = 1.0;
  generation.top_p = 1.0;
  generation.top_k = 0;
  if (auto bos = gguf_get_i64(gguf.value(), "tokenizer.ggml.bos_token_id"); bos.is_ok()) {
    generation.bos_token_id = bos.value();
    model.bos_token_id = bos.value();
  }
  if (auto eos = gguf_get_i64(gguf.value(), "tokenizer.ggml.eos_token_id"); eos.is_ok()) {
    generation.eos_token_ids.push_back(eos.value());
    generation.pad_token_id = eos.value();
    model.eos_token_id = eos.value();
  }
  if (generation.eos_token_ids.empty()) {
    generation.eos_token_ids.push_back(model.eos_token_id);
  }

  TokenizerInfo tokenizer{};
  tokenizer.available = find_gguf_metadata(gguf.value(), "tokenizer.ggml.tokens") != nullptr;
  if (auto tokenizer_model = gguf_get_string(gguf.value(), "tokenizer.ggml.model");
      tokenizer_model.is_ok()) {
    tokenizer.model = tokenizer_model.value();
  }
  if (auto tokenizer_pre = gguf_get_string(gguf.value(), "tokenizer.ggml.pre");
      tokenizer_pre.is_ok()) {
    tokenizer.pre = tokenizer_pre.value();
  }
  tokenizer.base_vocab_size =
    gguf_optional_array_size(gguf.value(), "tokenizer.ggml.tokens", 0);
  tokenizer.total_vocab_size = tokenizer.base_vocab_size;
  tokenizer.max_token_id = tokenizer.base_vocab_size == 0 ? 0 : tokenizer.base_vocab_size - 1;

  bundle.model = std::move(model);
  bundle.generation = std::move(generation);
  bundle.tokenizer = std::move(tokenizer);

  auto validation = validate_model_bundle(bundle);
  if (!validation.is_ok()) {
    return validation;
  }
  return bundle;
}

Status validate_positive(std::int64_t value, std::string_view name) {
  if (value <= 0) {
    return Status::invalid_argument(std::string(name) + " must be positive");
  }
  return Status::ok();
}

}  // namespace

Result<ModelBundle> load_model_bundle(const std::filesystem::path& model_dir) {
  if (!std::filesystem::exists(model_dir)) {
    return Status::unavailable("model directory does not exist: " + model_dir.string());
  }
  if (std::filesystem::is_regular_file(model_dir)) {
    if (model_dir.extension() == ".gguf") {
      return load_gguf_bundle(model_dir);
    }
    return Status::invalid_argument("model path is not a directory or GGUF file: " +
                                    model_dir.string());
  }
  if (!std::filesystem::is_directory(model_dir)) {
    return Status::invalid_argument("model path is not a directory: " + model_dir.string());
  }

  auto gguf_path = resolve_gguf_model_path(model_dir);
  if (gguf_path.is_ok()) {
    return load_gguf_bundle(model_dir);
  }

  auto config_json = parse_json_file(model_dir / "config.json");
  if (!config_json.is_ok()) {
    return config_json.status();
  }

  auto generation_json = parse_json_file(model_dir / "generation_config.json");
  if (!generation_json.is_ok()) {
    return generation_json.status();
  }

  auto tokenizer_config_json = parse_json_file(model_dir / "tokenizer_config.json");
  if (!tokenizer_config_json.is_ok()) {
    return tokenizer_config_json.status();
  }

  auto tokenizer_json = parse_json_file(model_dir / "tokenizer.json");
  if (!tokenizer_json.is_ok()) {
    return tokenizer_json.status();
  }

  auto vocab_json = parse_json_file(model_dir / "vocab.json");
  if (!vocab_json.is_ok()) {
    return vocab_json.status();
  }

  auto model = parse_model_config(config_json.value());
  if (!model.is_ok()) {
    return model.status();
  }

  auto generation = parse_generation_config(generation_json.value());
  if (!generation.is_ok()) {
    return generation.status();
  }

  auto tokenizer =
    parse_tokenizer_info(tokenizer_json.value(), tokenizer_config_json.value(), vocab_json.value());
  if (!tokenizer.is_ok()) {
    return tokenizer.status();
  }

  ModelBundle bundle{};
  bundle.model_dir = model_dir;
  bundle.model = model.value();
  bundle.generation = generation.value();
  bundle.tokenizer = tokenizer.value();

  Status validation = validate_model_bundle(bundle);
  if (!validation.is_ok()) {
    return validation;
  }

  return bundle;
}

Status validate_model_bundle(const ModelBundle& bundle) {
  const ModelConfig& model = bundle.model;
  const GenerationConfig& generation = bundle.generation;

  if (model.gguf) {
    if (model.architecture != "qwen35" && model.architecture != "qwen35moe" &&
        model.architecture != "qwen3vl" && model.architecture != "qwen3vlmoe" &&
        model.architecture != "clip") {
      return Status::invalid_argument("unsupported GGUF architecture: " + model.architecture);
    }
    if (model.architecture == "clip") {
      return Status::ok();
    }
    if (auto status = validate_positive(model.hidden_size, "embedding_length");
        !status.is_ok()) {
      return status;
    }
    if (auto status = validate_positive(model.num_attention_heads, "attention.head_count");
        !status.is_ok()) {
      return status;
    }
    if (auto status = validate_positive(model.num_key_value_heads, "attention.head_count_kv");
        !status.is_ok()) {
      return status;
    }
    if (auto status = validate_positive(model.main_layer_count, "main layer count");
        !status.is_ok()) {
      return status;
    }
    if (auto status = validate_positive(model.vocab_size, "vocab_size"); !status.is_ok()) {
      return status;
    }
    if (model.num_attention_heads % model.num_key_value_heads != 0) {
      return Status::invalid_argument("attention.head_count must be divisible by head_count_kv");
    }
    if (model.rms_norm_eps <= 0.0) {
      return Status::invalid_argument("attention.layer_norm_rms_epsilon must be positive");
    }
    if (model.rope_theta <= 0.0) {
      return Status::invalid_argument("rope.freq_base must be positive");
    }
    if (model.rope_dimension_sections.empty()) {
      return Status::invalid_argument("rope.dimension_sections must not be empty");
    }
    if (model.attention_recurrent_layers.size() <
        static_cast<std::size_t>(model.main_layer_count)) {
      return Status::invalid_argument("attention_recurrent_layers shorter than main layers");
    }
    return Status::ok();
  }

  if (model.architecture != "Qwen3ForCausalLM") {
    return Status::invalid_argument("unsupported architecture: " + model.architecture);
  }
  if (model.model_type != "qwen3") {
    return Status::invalid_argument("unsupported model_type: " + model.model_type);
  }

  if (auto status = validate_positive(model.hidden_size, "hidden_size"); !status.is_ok()) {
    return status;
  }
  if (auto status = validate_positive(model.head_dim, "head_dim"); !status.is_ok()) {
    return status;
  }
  if (auto status = validate_positive(model.num_attention_heads, "num_attention_heads");
      !status.is_ok()) {
    return status;
  }
  if (auto status = validate_positive(model.num_key_value_heads, "num_key_value_heads");
      !status.is_ok()) {
    return status;
  }
  if (auto status = validate_positive(model.num_hidden_layers, "num_hidden_layers");
      !status.is_ok()) {
    return status;
  }
  if (auto status = validate_positive(model.vocab_size, "vocab_size"); !status.is_ok()) {
    return status;
  }

  if (model.num_attention_heads % model.num_key_value_heads != 0) {
    return Status::invalid_argument("num_attention_heads must be divisible by num_key_value_heads");
  }
  if (model.intermediate_size <= model.hidden_size) {
    return Status::invalid_argument("intermediate_size must be larger than hidden_size");
  }
  if (model.rms_norm_eps <= 0.0) {
    return Status::invalid_argument("rms_norm_eps must be positive");
  }
  if (model.rope_theta <= 0.0) {
    return Status::invalid_argument("rope_theta must be positive");
  }

  if (bundle.tokenizer.available &&
      bundle.tokenizer.total_vocab_size > static_cast<std::uint64_t>(model.vocab_size)) {
    std::ostringstream message;
    message << "tokenizer explicit vocab exceeds config vocab_size: config=" << model.vocab_size
            << ", tokenizer total=" << bundle.tokenizer.total_vocab_size;
    return Status::invalid_argument(message.str());
  }
  if (bundle.tokenizer.available &&
      bundle.tokenizer.max_token_id >= static_cast<std::uint64_t>(model.vocab_size)) {
    std::ostringstream message;
    message << "tokenizer max token id " << bundle.tokenizer.max_token_id
            << " exceeds config vocab_size " << model.vocab_size;
    return Status::invalid_argument(message.str());
  }
  if (bundle.tokenizer.available &&
      bundle.tokenizer.added_tokens != bundle.tokenizer.tokenizer_config_added_tokens) {
    std::ostringstream message;
    message << "tokenizer added token count mismatch: tokenizer.json=" << bundle.tokenizer.added_tokens
            << ", tokenizer_config.json=" << bundle.tokenizer.tokenizer_config_added_tokens;
    return Status::invalid_argument(message.str());
  }

  if (generation.temperature <= 0.0) {
    return Status::invalid_argument("temperature must be positive");
  }
  if (generation.top_p <= 0.0 || generation.top_p > 1.0) {
    return Status::invalid_argument("top_p must be in (0, 1]");
  }
  if (generation.top_k < 0) {
    return Status::invalid_argument("top_k must be non-negative");
  }
  if (generation.eos_token_ids.empty()) {
    return Status::invalid_argument("eos_token_ids must not be empty");
  }

  return Status::ok();
}

std::string format_model_summary(const ModelBundle& bundle) {
  const ModelConfig& model = bundle.model;
  const GenerationConfig& generation = bundle.generation;
  const TokenizerInfo& tokenizer = bundle.tokenizer;

  if (model.gguf) {
    std::ostringstream output;
    const auto gqa_group_size = model.num_key_value_heads == 0
                                  ? 0
                                  : model.num_attention_heads / model.num_key_value_heads;
    output << "Model path: " << bundle.model_dir.string() << '\n';
    output << "Model file: " << bundle.model_file.string() << '\n';
    output << "Format: GGUF v" << bundle.gguf_version << '\n';
    output << "File size: " << bundle.gguf_file_size << " bytes\n";
    output << "Architecture: " << model.architecture << '\n';
    output << "Tensors: " << bundle.gguf_tensor_count << '\n';
    output << "Metadata entries: " << bundle.gguf_metadata_count << '\n';
    output << "Layers total: " << model.total_layer_count << '\n';
    output << "Layers main: " << model.main_layer_count << '\n';
    output << "MTP/NextN layers: " << model.mtp_num_hidden_layers << '\n';
    output << "Hidden size: " << model.hidden_size << '\n';
    output << "Attention heads: " << model.num_attention_heads << '\n';
    output << "KV heads: " << model.num_key_value_heads << '\n';
    output << "GQA group size: " << gqa_group_size << '\n';
    output << "Head dim: " << model.head_dim << '\n';
    output << "Intermediate size: " << model.intermediate_size << '\n';
    output << "Context length: " << model.max_position_embeddings << '\n';
    output << "Full attention interval: " << model.full_attention_interval << '\n';
    output << "Linear conv kernel: " << model.linear_conv_kernel_dim << '\n';
    output << "Linear state/head dim: " << model.linear_key_head_dim << '\n';
    output << "Linear key heads: " << model.linear_num_key_heads << '\n';
    output << "Linear value heads: " << model.linear_num_value_heads << '\n';
    output << "Linear inner size: " << model.linear_inner_size << '\n';
    output << "MRoPE sections: " << join_ints(model.rope_dimension_sections) << '\n';
    output << "Recurrent layer flags: " << model.attention_recurrent_layers.size() << '\n';
    output << "Expert count: " << model.expert_count << '\n';
    output << "Expert used count: " << model.expert_used_count << '\n';
    output << "Expert FFN size: " << model.expert_feed_forward_length << '\n';
    output << "Shared expert FFN size: " << model.expert_shared_feed_forward_length << '\n';
    output << "RMSNorm eps: " << model.rms_norm_eps << '\n';
    output << "RoPE theta: " << static_cast<std::int64_t>(model.rope_theta) << '\n';
    output << "Vocab size: " << model.vocab_size << '\n';
    output << "Tokenizer available: " << bool_to_string(tokenizer.available) << '\n';
    output << "Tokenizer model: " << tokenizer.model << '\n';
    output << "Tokenizer pre: " << tokenizer.pre << '\n';
    output << "Tokenizer tokens: " << tokenizer.base_vocab_size << '\n';
    output << "BOS token id: " << model.bos_token_id << '\n';
    output << "EOS token id: " << model.eos_token_id << '\n';
    output << "Generation eos ids: " << join_ints(generation.eos_token_ids) << '\n';
    output << "Validation: ok\n";
    return output.str();
  }

  const auto gqa_group_size = model.num_attention_heads / model.num_key_value_heads;
  const auto attention_projection_size = model.num_attention_heads * model.head_dim;
  const auto kv_projection_size = model.num_key_value_heads * model.head_dim;
  const auto reserved_vocab_slots =
    static_cast<std::uint64_t>(model.vocab_size) - tokenizer.total_vocab_size;

  std::ostringstream output;
  output << "Model path: " << bundle.model_dir.string() << '\n';
  output << "Architecture: " << model.architecture << '\n';
  output << "Model type: " << model.model_type << '\n';
  output << "Layers: " << model.num_hidden_layers << '\n';
  output << "Hidden size: " << model.hidden_size << '\n';
  output << "Attention heads: " << model.num_attention_heads << '\n';
  output << "KV heads: " << model.num_key_value_heads << '\n';
  output << "GQA group size: " << gqa_group_size << '\n';
  output << "Head dim: " << model.head_dim << '\n';
  output << "Attention projection size: " << attention_projection_size << '\n';
  output << "KV projection size: " << kv_projection_size << '\n';
  output << "Intermediate size: " << model.intermediate_size << '\n';
  output << "Max position embeddings: " << model.max_position_embeddings << '\n';
  output << "Max window layers: " << model.max_window_layers << '\n';
  output << "Sliding window enabled: " << bool_to_string(model.use_sliding_window) << '\n';
  output << "Attention bias: " << bool_to_string(model.attention_bias) << '\n';
  output << "Attention dropout: " << model.attention_dropout << '\n';
  output << "Activation: " << model.hidden_act << '\n';
  output << "RMSNorm eps: " << model.rms_norm_eps << '\n';
  output << "RoPE theta: " << static_cast<std::int64_t>(model.rope_theta) << '\n';
  output << "DType: " << model.torch_dtype << '\n';
  output << "Tie word embeddings: " << bool_to_string(model.tie_word_embeddings) << '\n';
  output << "Use cache: " << bool_to_string(model.use_cache) << '\n';
  output << "Vocab size: " << model.vocab_size << '\n';
  output << "Tokenizer base vocab: " << tokenizer.base_vocab_size << '\n';
  output << "Tokenizer added tokens: " << tokenizer.added_tokens << '\n';
  output << "Tokenizer total vocab: " << tokenizer.total_vocab_size << '\n';
  output << "Tokenizer max token id: " << tokenizer.max_token_id << '\n';
  output << "Reserved vocab slots: " << reserved_vocab_slots << '\n';
  output << "BOS token id: " << model.bos_token_id << '\n';
  output << "EOS token id: " << model.eos_token_id << '\n';
  output << "Generation do_sample: " << bool_to_string(generation.do_sample) << '\n';
  output << "Generation temperature: " << generation.temperature << '\n';
  output << "Generation top_k: " << generation.top_k << '\n';
  output << "Generation top_p: " << generation.top_p << '\n';
  output << "Generation eos ids: " << join_ints(generation.eos_token_ids) << '\n';
  output << "Validation: ok\n";
  return output.str();
}

}  // namespace toyllm
