#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/runtime.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kVersion = "0.1.0";
constexpr std::string_view kDefaultModelPath = "models/qwen3-0.6b";

void print_usage(std::string_view program) {
  std::cout << "Usage:\n";
  std::cout << "  " << program << " help\n";
  std::cout << "  " << program << " version\n";
  std::cout << "  " << program << " mps\n";
  std::cout << "  " << program << " inspect [model_dir]\n";
  std::cout << "  " << program << " doctor [model_dir]\n";
  std::cout << "  " << program << " run --prompt <text> [--model <model_dir>]\n\n";
  std::cout << "Compatibility flags:\n";
  std::cout << "  " << program << " --mps-info\n";
  std::cout << "  " << program << " --inspect-model <model_dir>\n";
  std::cout << "  " << program << " --model <model_dir> --prompt <text>\n\n";
  std::cout << "Default model_dir: " << kDefaultModelPath << '\n';
}

bool arg_equals(char const* arg, std::string_view expected) {
  return std::string_view(arg) == expected;
}

std::filesystem::path default_model_path() {
  return std::filesystem::path{kDefaultModelPath};
}

int inspect_model(const std::filesystem::path& model_path) {
  auto bundle = toyllm::load_model_bundle(model_path);
  if (!bundle.is_ok()) {
    std::cerr << "Failed to inspect model: " << bundle.status().message() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << toyllm::format_model_summary(bundle.value());
  return EXIT_SUCCESS;
}

int print_mps_info() {
  const auto info = toyllm::mps::query_backend();
  std::cout << toyllm::mps::format_backend_info(info);
  return EXIT_SUCCESS;
}

int run_doctor(const std::filesystem::path& model_path) {
  std::cout << "toyllm " << kVersion << '\n';
  std::cout << "\n== MPS ==\n";
  (void)print_mps_info();
  std::cout << "\n== Model ==\n";
  return inspect_model(model_path);
}

int run_placeholder(const std::string& model_path, const std::string& prompt) {
  if (model_path.empty() || prompt.empty()) {
    std::cerr << "run requires --prompt and a model path.\n";
    return EXIT_FAILURE;
  }

  const toyllm::Runtime runtime{toyllm::RuntimeConfig{}};
  const auto info = runtime.info();

  std::cout << "Model path: " << model_path << '\n';
  std::cout << "Selected device: " << info.selected_device.to_string() << '\n';
  if (info.accelerator_available) {
    std::cout << "Accelerator: " << info.accelerator_name << '\n';
  }
  std::cout << "Prompt: " << prompt << '\n';
  std::cout << "Inference is not implemented yet.\n";

  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc <= 1 || arg_equals(argv[1], "help") || arg_equals(argv[1], "--help") ||
      arg_equals(argv[1], "-h")) {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }

  if (arg_equals(argv[1], "version") || arg_equals(argv[1], "--version")) {
    std::cout << "toyllm " << kVersion << '\n';
    return EXIT_SUCCESS;
  }

  if (arg_equals(argv[1], "mps") || arg_equals(argv[1], "--mps-info")) {
    return print_mps_info();
  }

  if (arg_equals(argv[1], "inspect") || arg_equals(argv[1], "--inspect-model")) {
    if (argc > 3) {
      std::cerr << "inspect accepts at most one model directory.\n";
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    const auto model_path = argc == 3 ? std::filesystem::path{argv[2]} : default_model_path();
    return inspect_model(model_path);
  }

  if (arg_equals(argv[1], "doctor")) {
    if (argc > 3) {
      std::cerr << "doctor accepts at most one model directory.\n";
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    const auto model_path = argc == 3 ? std::filesystem::path{argv[2]} : default_model_path();
    return run_doctor(model_path);
  }

  const bool explicit_run = arg_equals(argv[1], "run");
  const int first_option = explicit_run ? 2 : 1;
  std::string model_path = std::string(kDefaultModelPath);
  std::string prompt;
  for (int index = first_option; index < argc; ++index) {
    if (arg_equals(argv[index], "--model") && index + 1 < argc) {
      model_path = argv[++index];
      continue;
    }
    if (arg_equals(argv[index], "--prompt") && index + 1 < argc) {
      prompt = argv[++index];
      continue;
    }

    std::cerr << "Unknown or incomplete argument: " << argv[index] << '\n';
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (!explicit_run && prompt.empty()) {
    std::cerr << "Unknown command: " << argv[1] << '\n';
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  return run_placeholder(model_path, prompt);
}
