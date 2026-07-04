#include "toyllm/runtime/qwen35_vl_adapter.hpp"

#include "toyllm/runtime/chat_message.hpp"
#include "toyllm/runtime/gguf_reader.hpp"
#include "toyllm/runtime/qwen35_multimodal.hpp"

#include <array>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace toyllm {

namespace {

constexpr std::string_view kDefaultMtmdCli =
  "/Users/bill/code/llama.cpp/build/bin/llama-mtmd-cli";
constexpr std::string_view kMtmdResponseMarker = "KRAKEN_MTMD_RESPONSE_BEGIN";

std::string shell_quote(std::string_view value) {
  std::string quoted{"'"};
  for (const char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(ch);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

std::string mtmd_cli_path() {
  if (const char* path = std::getenv("KRAKEN_LLAMA_MTMD_CLI");
      path != nullptr && path[0] != '\0') {
    return std::string{path};
  }
  return std::string{kDefaultMtmdCli};
}

std::string image_suffix(std::string_view mime_type) {
  if (mime_type == "image/jpeg" || mime_type == "image/jpg") {
    return ".jpg";
  }
  if (mime_type == "image/png") {
    return ".png";
  }
  if (mime_type == "image/webp") {
    return ".webp";
  }
  return ".img";
}

Result<std::filesystem::path> write_temp_image(const ChatContentPart& part,
                                               std::size_t index) {
  if (part.image_bytes.empty()) {
    return Status::invalid_argument(
      "Qwen3.5 VL adapter currently requires data:image/...;base64 image_url");
  }

  auto pattern = std::filesystem::temp_directory_path() /
                 ("kraken-qwen35-vl-" + std::to_string(::getpid()) + "-" +
                  std::to_string(index) + "-XXXXXX");
  std::string path_pattern = pattern.string();
  std::vector<char> mutable_path(path_pattern.begin(), path_pattern.end());
  mutable_path.push_back('\0');
  const int fd = ::mkstemp(mutable_path.data());
  if (fd < 0) {
    return Status::internal_error("failed to create temporary image file");
  }
  ::close(fd);

  std::filesystem::path path{mutable_path.data()};
  const auto final_path = path.string() + image_suffix(part.image_mime_type);
  std::error_code ec;
  std::filesystem::rename(path, final_path, ec);
  if (ec) {
    std::filesystem::remove(path);
    return Status::internal_error("failed to name temporary image file: " +
                                  ec.message());
  }

  std::ofstream output(final_path, std::ios::binary);
  if (!output) {
    std::filesystem::remove(final_path);
    return Status::internal_error("failed to open temporary image file");
  }
  output.write(reinterpret_cast<const char*>(part.image_bytes.data()),
               static_cast<std::streamsize>(part.image_bytes.size()));
  if (!output) {
    std::filesystem::remove(final_path);
    return Status::internal_error("failed to write temporary image file");
  }
  return std::filesystem::path{final_path};
}

void append_message_text(std::ostringstream& output, const ChatMessage& message) {
  if (!message.role.empty()) {
    output << message.role << ": ";
  }
  if (!message.content_parts.empty()) {
    bool wrote_text = false;
    for (const auto& part : message.content_parts) {
      if (part.kind != ChatContentPartKind::text || part.text.empty()) {
        continue;
      }
      if (wrote_text) {
        output << '\n';
      }
      output << part.text;
      wrote_text = true;
    }
    if (!wrote_text && !message.content.empty()) {
      output << message.content;
    }
  } else {
    output << message.content;
  }
  output << '\n';
}

std::string make_mtmd_prompt(const CpuGenerationRequest& request) {
  if (!request.messages.empty()) {
    std::ostringstream output;
    for (const auto& message : request.messages) {
      append_message_text(output, message);
    }
    output << "assistant:";
    return output.str();
  }
  return request.prompt;
}

Result<std::vector<std::filesystem::path>> write_request_images(
  const CpuGenerationRequest& request) {
  std::vector<std::filesystem::path> paths;
  for (const auto& message : request.messages) {
    for (const auto& part : message.content_parts) {
      if (part.kind != ChatContentPartKind::image_url) {
        continue;
      }
      if (!qwen35_image_url_is_data_url(part.image_url)) {
        return Status::unavailable(
          "Qwen3.5 VL adapter currently supports data URL images only");
      }
      auto path = write_temp_image(part, paths.size());
      if (!path.is_ok()) {
        return path.status();
      }
      paths.push_back(path.value());
    }
  }
  if (paths.empty()) {
    return Status::invalid_argument("Qwen3.5 VL request has no image content");
  }
  return paths;
}

std::string trim_mtmd_output(std::string output) {
  while (!output.empty() &&
         (output.back() == '\n' || output.back() == '\r' || output.back() == ' ')) {
    output.pop_back();
  }
  return output;
}

bool is_mtmd_log_line(std::string_view line) {
  // llama.cpp log lines start with timestamps such as
  // "0.01.036.607 W find_slot: ...". Generated text can contain digits, so keep
  // this narrow and only strip the structured log prefix.
  std::size_t cursor = 0;
  int dots = 0;
  while (cursor < line.size()) {
    const auto ch = static_cast<unsigned char>(line[cursor]);
    if (std::isdigit(ch) != 0) {
      ++cursor;
      continue;
    }
    if (line[cursor] == '.') {
      ++dots;
      ++cursor;
      continue;
    }
    break;
  }
  if (dots < 2 || cursor + 3U > line.size() || line[cursor] != ' ') {
    return false;
  }
  const char level = line[cursor + 1U];
  return line[cursor + 2U] == ' ' &&
         (level == 'I' || level == 'W' || level == 'E' || level == 'D');
}

std::string strip_mtmd_logs(std::string output) {
  const auto marker = output.find(kMtmdResponseMarker);
  if (marker != std::string::npos) {
    output.erase(0, marker + kMtmdResponseMarker.size());
  }

  std::istringstream input(output);
  std::ostringstream cleaned;
  std::string line;
  bool wrote_line = false;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line == kMtmdResponseMarker || is_mtmd_log_line(line)) {
      continue;
    }
    if (line.rfind("llama_perf_", 0) == 0 || line.rfind("load_", 0) == 0) {
      continue;
    }
    if (wrote_line) {
      cleaned << '\n';
    }
    cleaned << line;
    wrote_line = true;
  }
  return trim_mtmd_output(cleaned.str());
}

Result<std::string> run_command_capture_stdout(const std::string& command) {
  FILE* pipe = ::popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return Status::internal_error("failed to launch llama-mtmd-cli");
  }

  std::string output;
  std::array<char, 4096> buffer{};
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  const int status = ::pclose(pipe);
  if (status == -1) {
    return Status::internal_error("failed to wait for llama-mtmd-cli");
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::ostringstream message;
    message << "llama-mtmd-cli failed";
    if (WIFEXITED(status)) {
      message << " with exit code " << WEXITSTATUS(status);
    }
    if (!output.empty()) {
      message << ": " << trim_mtmd_output(output);
    }
    return Status::unavailable(message.str());
  }
  return strip_mtmd_logs(output);
}

std::string make_mtmd_command(const CpuGenerationRequest& request,
                              const std::filesystem::path& model_file,
                              const std::vector<std::filesystem::path>& images) {
  std::ostringstream command;
  command << "MTMD_TEST_RESPONSE_MARKER=" << shell_quote(kMtmdResponseMarker)
          << ' ' << shell_quote(mtmd_cli_path())
          << " --no-warmup"
          << " -m " << shell_quote(model_file.string())
          << " --mmproj " << shell_quote(request.mmproj_path.string())
          << " -ngl all"
          << " -n " << request.max_new_tokens
          << " -p " << shell_quote(make_mtmd_prompt(request));
  for (const auto& image : images) {
    command << " --image " << shell_quote(image.string());
  }
  if (request.sampling.seed_set) {
    command << " --seed " << request.sampling.seed;
  }
  if (request.sampling.temperature_set) {
    command << " --temp " << request.sampling.temperature;
  } else if (!request.sampling.do_sample) {
    command << " --temp 0";
  }
  if (request.sampling.top_k_set) {
    command << " --top-k " << request.sampling.top_k;
  }
  if (request.sampling.top_p_set) {
    command << " --top-p " << request.sampling.top_p;
  }
  command << " 2>&1";
  return command.str();
}

}  // namespace

Result<CpuGenerationResult> generate_qwen35_vl_with_llama_mtmd(
  const CpuGenerationRequest& request) {
  if (!chat_messages_have_image_content(request.messages)) {
    return Status::invalid_argument("Qwen3.5 VL adapter requires image content");
  }
  if (request.mmproj_path.empty()) {
    return Status::invalid_argument(
      "image input requires a Qwen3.5 conditional mmproj GGUF");
  }
  if (!std::filesystem::exists(mtmd_cli_path())) {
    return Status::unavailable(
      "llama-mtmd-cli was not found; set KRAKEN_LLAMA_MTMD_CLI or build "
      "~/code/llama.cpp target llama-mtmd-cli");
  }

  const auto metadata = load_qwen35_mmproj_metadata(request.mmproj_path);
  if (!metadata.is_ok()) {
    return metadata.status();
  }
  if (!qwen35_mmproj_is_qwen3vl_merger(metadata.value())) {
    return Status::invalid_argument(
      "Qwen3.5 image input requires a qwen3vl_merger mmproj");
  }
  const auto model_file = resolve_gguf_model_path(request.model_dir);
  if (!model_file.is_ok()) {
    return model_file.status();
  }

  auto images = write_request_images(request);
  if (!images.is_ok()) {
    return images.status();
  }

  const auto command = make_mtmd_command(request, model_file.value(), images.value());
  auto output = run_command_capture_stdout(command);
  for (const auto& image : images.value()) {
    std::error_code ec;
    std::filesystem::remove(image, ec);
  }
  if (!output.is_ok()) {
    return output.status();
  }

  CpuGenerationResult result;
  result.implemented = true;
  result.text = output.value();
  result.finish_reason = "stop";
  result.generated_tokens = 0;
  result.prompt_tokens = 0;
  if (request.stream_token) {
    request.stream_token(result.text);
  }
  return result;
}

}  // namespace toyllm
