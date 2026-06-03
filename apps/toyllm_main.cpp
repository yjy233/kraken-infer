#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/runtime.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kVersion = "0.1.0";

void print_usage(std::string_view program) {
  std::cout << "Usage:\n";
  std::cout << "  " << program << " --mps-info\n";
  std::cout << "  " << program << " --inspect-model <path>\n";
  std::cout << "  " << program << " --model <path> --prompt <text>\n";
  std::cout << "  " << program << " --help\n\n";
  std::cout << "This repository is initialized as a from-scratch LLM runtime skeleton. ";
  std::cout << "Qwen3 execution is intentionally a staged TODO after tokenizer, loader, ";
  std::cout << "kernels, and sampler land.\n";
}

bool arg_equals(char const* arg, std::string_view expected) {
  return std::string_view(arg) == expected;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc <= 1 || arg_equals(argv[1], "--help") || arg_equals(argv[1], "-h")) {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }

  if (arg_equals(argv[1], "--version")) {
    std::cout << "toyllm " << kVersion << '\n';
    return EXIT_SUCCESS;
  }

  if (arg_equals(argv[1], "--mps-info")) {
    const auto info = toyllm::mps::query_backend();
    std::cout << toyllm::mps::format_backend_info(info);
    return EXIT_SUCCESS;
  }

  if (arg_equals(argv[1], "--inspect-model")) {
    if (argc != 3) {
      std::cerr << "--inspect-model requires exactly one model directory.\n";
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }

    auto bundle = toyllm::load_model_bundle(argv[2]);
    if (!bundle.is_ok()) {
      std::cerr << "Failed to inspect model: " << bundle.status().message() << '\n';
      return EXIT_FAILURE;
    }

    std::cout << toyllm::format_model_summary(bundle.value());
    return EXIT_SUCCESS;
  }

  std::string model_path;
  std::string prompt;
  for (int index = 1; index < argc; ++index) {
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

  if (model_path.empty() || prompt.empty()) {
    std::cerr << "Both --model and --prompt are required for inference.\n";
    print_usage(argv[0]);
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
