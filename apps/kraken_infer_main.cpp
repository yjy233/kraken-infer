#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/cpu_inference.hpp"
#include "toyllm/runtime/openai_gateway.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kVersion = "0.1.0";
constexpr std::string_view kDefaultModelPath = "models/qwen3-0.6b";

void print_usage(std::string_view program) {
  std::cout << "Usage:\n";
  std::cout << "  " << program << " help\n";
  std::cout << "  " << program << " version\n";
  std::cout << "  " << program << " mps\n";
  std::cout << "  " << program << " mps-smoke\n";
  std::cout << "  " << program << " mpsgraph\n";
  std::cout << "  " << program << " mpsgraph-smoke\n";
  std::cout << "  " << program << " inspect [model_dir]\n";
  std::cout << "  " << program << " weights [model_dir]\n";
  std::cout << "  " << program << " doctor [model_dir]\n";
  std::cout << "  " << program
            << " infer --prompt <text> [--model <model_dir>] [--max-new-tokens N]"
               " [--device cpu|mps|mpsgraph]"
               " [--sample] [--temperature T] [--top-k K] [--top-p P] [--seed N]"
               " [--stream] [--dump-dir DIR] [--kv-cache-stats] [--verify-kv-cache]"
               " [--profile off|summary|trace|flamegraph|all] [--profile-dir DIR]"
               " [--profile-min-us N]\n";
  std::cout << "  " << program
            << " run --prompt <text> [--model <model_dir>] [--max-new-tokens N]"
               " [--device cpu|mps|mpsgraph]"
               " [--sample] [--temperature T] [--top-k K] [--top-p P] [--seed N]"
               " [--stream] [--dump-dir DIR] [--kv-cache-stats] [--verify-kv-cache]"
               " [--profile off|summary|trace|flamegraph|all] [--profile-dir DIR]"
               " [--profile-min-us N]\n";
  std::cout << "  " << program
            << " chat [model_dir] [--max-new-tokens N] [--enable-thinking] [--dump-dir DIR]"
               " [--device cpu|mps|mpsgraph]"
               " [--sample] [--temperature T] [--top-k K] [--top-p P] [--seed N]"
               " [--stream] [--kv-cache-stats] [--verify-kv-cache]"
               " [--profile off|summary|trace|flamegraph|all] [--profile-dir DIR]"
               " [--profile-min-us N]\n\n";
  std::cout << "  " << program
            << " serve [--host 127.0.0.1] [--port 8080] [--model <model_dir>]"
               " [--model-id ID] [--device cpu|mps|mpsgraph] [--max-new-tokens N]\n\n";
  std::cout << "       [--profile off|summary|trace|flamegraph|all] [--profile-dir DIR]"
               " [--profile-min-us N]\n\n";
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

std::optional<std::size_t> parse_size_arg(std::string_view value) {
  try {
    std::size_t parsed_chars = 0;
    const auto parsed = std::stoull(std::string(value), &parsed_chars, 10);
    if (parsed_chars != value.size() ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(parsed);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<std::uint64_t> parse_u64_arg(std::string_view value) {
  try {
    std::size_t parsed_chars = 0;
    const auto parsed = std::stoull(std::string(value), &parsed_chars, 10);
    if (parsed_chars != value.size()) {
      return std::nullopt;
    }
    return static_cast<std::uint64_t>(parsed);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<double> parse_double_arg(std::string_view value) {
  try {
    std::size_t parsed_chars = 0;
    const auto parsed = std::stod(std::string(value), &parsed_chars);
    if (parsed_chars != value.size()) {
      return std::nullopt;
    }
    return parsed;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<toyllm::Device> parse_device_arg(std::string_view value) {
  if (value == "cpu") {
    return toyllm::Device::cpu();
  }
  if (value == "mps" || value == "mps:0") {
    return toyllm::Device::mps();
  }
  if (value == "mpsgraph") {
    return toyllm::Device::mpsgraph();
  }
  return std::nullopt;
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

int run_mps_smoke() {
  const auto status = toyllm::mps::run_operator_smoke_test();
  if (!status.is_ok()) {
    std::cerr << "MPS operator smoke failed: " << status.message() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "MPS operator smoke: ok\n";
  return EXIT_SUCCESS;
}

int print_mpsgraph_info() {
  const auto info = toyllm::mpsgraph::query_backend();
  std::cout << toyllm::mpsgraph::format_backend_info(info);
  return EXIT_SUCCESS;
}

int run_mpsgraph_smoke() {
  const auto status = toyllm::mpsgraph::run_operator_smoke_test();
  if (!status.is_ok()) {
    std::cerr << "MPSGraph operator smoke failed: " << status.message() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "MPSGraph operator smoke: ok\n";
  return EXIT_SUCCESS;
}

int inspect_weights(const std::filesystem::path& model_path) {
  const auto summary = toyllm::format_weight_summary(model_path);
  if (!summary.is_ok()) {
    std::cerr << "Failed to inspect weights: " << summary.status().message() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << summary.value();
  return EXIT_SUCCESS;
}

int run_doctor(const std::filesystem::path& model_path) {
  std::cout << "kraken-infer " << kVersion << '\n';
  std::cout << "\n== MPS ==\n";
  (void)print_mps_info();
  const auto mps_smoke = toyllm::mps::run_operator_smoke_test();
  if (mps_smoke.is_ok()) {
    std::cout << "MPS operator smoke: ok\n";
  } else {
    std::cout << "MPS operator smoke: not ready: " << mps_smoke.message() << '\n';
  }
  std::cout << "\n== MPSGraph ==\n";
  (void)print_mpsgraph_info();
  const auto mpsgraph_smoke = toyllm::mpsgraph::run_operator_smoke_test();
  if (mpsgraph_smoke.is_ok()) {
    std::cout << "MPSGraph operator smoke: ok\n";
  } else {
    std::cout << "MPSGraph operator smoke: not ready: " << mpsgraph_smoke.message() << '\n';
  }
  std::cout << "\n== Model ==\n";
  const auto model_status = inspect_model(model_path);
  if (model_status != EXIT_SUCCESS) {
    return model_status;
  }
  std::cout << "\n== Weights ==\n";
  return inspect_weights(model_path);
}

void print_kv_cache_report(const toyllm::CpuKvCacheReport& report) {
  if (!report.available) {
    return;
  }

  std::cout << "KV cache:\n";
  std::cout << "- layers: " << report.layers << '\n';
  std::cout << "- kv_heads: " << report.kv_heads << '\n';
  std::cout << "- head_dim: " << report.head_dim << '\n';
  std::cout << "- kv_dim: " << report.kv_dim << '\n';
  std::cout << "- capacity_tokens: " << report.capacity_tokens << '\n';
  std::cout << "- used_tokens: " << report.used_tokens << '\n';
  std::cout << "- key_bytes: " << report.key_bytes << '\n';
  std::cout << "- value_bytes: " << report.value_bytes << '\n';
  std::cout << "- total_bytes: " << report.total_bytes << '\n';
}

int run_cpu_generation(const std::string& model_path, const std::string& prompt,
                       std::size_t max_new_tokens, bool enable_thinking,
                       const std::filesystem::path& debug_dump_dir, bool print_kv_cache_stats,
                       bool verify_kv_cache, toyllm::Device compute_device,
                       const toyllm::CpuSamplingConfig& sampling, bool stream,
                       const toyllm::ObservabilityConfig& observability) {
  if (model_path.empty() || prompt.empty()) {
    std::cerr << "generation requires --prompt and a model path.\n";
    return EXIT_FAILURE;
  }

  toyllm::CpuGenerationRequest request;
  request.model_dir = std::filesystem::path{model_path};
  request.prompt = prompt;
  request.max_new_tokens = max_new_tokens;
  request.enable_thinking = enable_thinking;
  request.debug_dump_dir = debug_dump_dir;
  request.verify_kv_cache = verify_kv_cache;
  request.compute_device = compute_device;
  request.sampling = sampling;
  request.observability = observability;
  if (stream) {
    request.stream_token = [](std::string_view chunk) {
      std::cout << chunk;
      std::cout.flush();
    };
  }
  const auto result = toyllm::generate_cpu(request);
  if (!result.is_ok()) {
    std::cerr << "generation failed: " << result.status().message() << '\n';
    return EXIT_FAILURE;
  }

  if (stream) {
    std::cout << '\n';
  } else {
    std::cout << toyllm::format_cpu_generation_result(result.value());
  }
  if (print_kv_cache_stats) {
    print_kv_cache_report(result.value().kv_cache);
  }
  if (verify_kv_cache) {
    std::cout << "KV cache verification: "
              << (result.value().kv_cache_verified ? "ok" : "not run") << '\n';
  }
  if (!result.value().request_id.empty()) {
    std::cout << "request_id: " << result.value().request_id << '\n';
  }
  if (!result.value().profile_dir.empty()) {
    std::cout << "profile_dir: " << result.value().profile_dir.string() << '\n';
  }
  return EXIT_SUCCESS;
}

int run_chat(const std::filesystem::path& model_path, std::size_t max_new_tokens,
             bool enable_thinking, const std::filesystem::path& debug_dump_dir,
             bool print_kv_cache_stats, bool verify_kv_cache, toyllm::Device compute_device,
             const toyllm::CpuSamplingConfig& sampling, bool stream,
             const toyllm::ObservabilityConfig& observability) {
  std::cout << "kraken-infer chat\n";
  std::cout << "model: " << model_path.string() << '\n';
  std::cout << "device: " << compute_device.to_string() << '\n';
  std::cout << "max_new_tokens: " << max_new_tokens << '\n';
  std::cout << "type /exit to quit\n\n";

  std::vector<toyllm::ChatMessage> messages;
  std::string line;
  while (true) {
    std::cout << "user> ";
    if (!std::getline(std::cin, line)) {
      std::cout << '\n';
      return EXIT_SUCCESS;
    }
    if (line == "/exit" || line == "/quit") {
      return EXIT_SUCCESS;
    }
    if (line.empty()) {
      continue;
    }

    messages.push_back(toyllm::ChatMessage{"user", line});

    toyllm::CpuGenerationRequest request;
    request.model_dir = model_path;
    request.max_new_tokens = max_new_tokens;
    request.enable_thinking = enable_thinking;
    request.messages = messages;
    request.debug_dump_dir = debug_dump_dir;
    request.verify_kv_cache = verify_kv_cache;
    request.compute_device = compute_device;
    request.sampling = sampling;
    request.observability = observability;
    if (stream) {
      std::cout << "assistant>\n";
      request.stream_token = [](std::string_view chunk) {
        std::cout << chunk;
        std::cout.flush();
      };
    }
    const auto result = toyllm::generate_cpu(request);
    if (!result.is_ok()) {
      std::cout << "assistant> error: " << result.status().message() << "\n\n";
      messages.pop_back();
      continue;
    }

    const auto answer = result.value().text;
    messages.push_back(toyllm::ChatMessage{"assistant", answer});
    if (stream) {
      std::cout << "\n\n";
    } else {
      std::cout << "assistant>\n" << toyllm::format_cpu_generation_result(result.value()) << '\n';
    }
    if (print_kv_cache_stats) {
      print_kv_cache_report(result.value().kv_cache);
      std::cout << '\n';
    }
    if (verify_kv_cache) {
      std::cout << "KV cache verification: "
                << (result.value().kv_cache_verified ? "ok" : "not run") << "\n\n";
    }
    if (!result.value().request_id.empty()) {
      std::cout << "request_id: " << result.value().request_id << '\n';
    }
    if (!result.value().profile_dir.empty()) {
      std::cout << "profile_dir: " << result.value().profile_dir.string() << '\n';
    }
  }
}

int run_gateway(const toyllm::OpenAIGatewayConfig& config) {
  const auto status = toyllm::serve_openai_gateway(config);
  if (!status.is_ok()) {
    std::cerr << "gateway failed: " << status.message() << '\n';
    return EXIT_FAILURE;
  }
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
    std::cout << "kraken-infer " << kVersion << '\n';
    return EXIT_SUCCESS;
  }

  if (arg_equals(argv[1], "mps") || arg_equals(argv[1], "--mps-info")) {
    return print_mps_info();
  }

  if (arg_equals(argv[1], "mps-smoke")) {
    return run_mps_smoke();
  }

  if (arg_equals(argv[1], "mpsgraph")) {
    return print_mpsgraph_info();
  }

  if (arg_equals(argv[1], "mpsgraph-smoke")) {
    return run_mpsgraph_smoke();
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

  if (arg_equals(argv[1], "weights")) {
    if (argc > 3) {
      std::cerr << "weights accepts at most one model directory.\n";
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    const auto model_path = argc == 3 ? std::filesystem::path{argv[2]} : default_model_path();
    return inspect_weights(model_path);
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

  if (arg_equals(argv[1], "serve")) {
    toyllm::OpenAIGatewayConfig config;
    config.model_dir = default_model_path();
    config.observability.profile_mode = toyllm::ProfileMode::summary;
    config.observability.profile_output_dir = std::filesystem::path{"build"} / "profiles";
    for (int index = 2; index < argc; ++index) {
      if (arg_equals(argv[index], "--host") && index + 1 < argc) {
        config.host = argv[++index];
        continue;
      }
      if (arg_equals(argv[index], "--port") && index + 1 < argc) {
        const auto parsed = parse_size_arg(argv[++index]);
        if (!parsed.has_value() || *parsed == 0 || *parsed > 65535U) {
          std::cerr << "--port must be a positive integer.\n";
          return EXIT_FAILURE;
        }
        config.port = static_cast<int>(*parsed);
        continue;
      }
      if (arg_equals(argv[index], "--model") && index + 1 < argc) {
        config.model_dir = std::filesystem::path{argv[++index]};
        continue;
      }
      if (arg_equals(argv[index], "--model-id") && index + 1 < argc) {
        config.model_id = argv[++index];
        continue;
      }
      if (arg_equals(argv[index], "--device") && index + 1 < argc) {
        const auto parsed = parse_device_arg(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--device must be cpu, mps, or mpsgraph.\n";
          return EXIT_FAILURE;
        }
        config.compute_device = *parsed;
        continue;
      }
      if (arg_equals(argv[index], "--max-new-tokens") && index + 1 < argc) {
        const auto parsed = parse_size_arg(argv[++index]);
        if (!parsed.has_value() || *parsed == 0) {
          std::cerr << "--max-new-tokens must be a positive integer.\n";
          return EXIT_FAILURE;
        }
        config.default_max_tokens = *parsed;
        continue;
      }
      if (arg_equals(argv[index], "--profile") && index + 1 < argc) {
        const auto parsed = toyllm::parse_profile_mode(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--profile must be off, summary, trace, flamegraph, or all.\n";
          return EXIT_FAILURE;
        }
        config.observability.profile_mode = *parsed;
        continue;
      }
      if (arg_equals(argv[index], "--profile-dir") && index + 1 < argc) {
        config.observability.profile_output_dir = std::filesystem::path{argv[++index]};
        continue;
      }
      if (arg_equals(argv[index], "--profile-min-us") && index + 1 < argc) {
        const auto parsed = parse_size_arg(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--profile-min-us must be a non-negative integer.\n";
          return EXIT_FAILURE;
        }
        config.observability.min_duration_us = static_cast<std::uint64_t>(*parsed);
        continue;
      }
      std::cerr << "Unknown serve option: " << argv[index] << '\n';
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    return run_gateway(config);
  }

  if (arg_equals(argv[1], "chat")) {
    std::filesystem::path model_path = default_model_path();
    std::size_t max_new_tokens = 16;
    bool enable_thinking = false;
    bool print_kv_cache_stats = false;
    bool verify_kv_cache = false;
    bool stream = false;
    toyllm::Device compute_device = toyllm::Device::cpu();
    toyllm::CpuSamplingConfig sampling;
    toyllm::ObservabilityConfig observability;
    std::filesystem::path debug_dump_dir;
    bool model_path_set = false;
    for (int index = 2; index < argc; ++index) {
      if (arg_equals(argv[index], "--model") && index + 1 < argc) {
        model_path = std::filesystem::path{argv[++index]};
        model_path_set = true;
        continue;
      }
      if (arg_equals(argv[index], "--max-new-tokens") && index + 1 < argc) {
        const auto parsed = parse_size_arg(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--max-new-tokens must be a non-negative integer.\n";
          return EXIT_FAILURE;
        }
        max_new_tokens = *parsed;
        continue;
      }
      if (arg_equals(argv[index], "--enable-thinking")) {
        enable_thinking = true;
        continue;
      }
      if (arg_equals(argv[index], "--device") && index + 1 < argc) {
        const auto parsed = parse_device_arg(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--device must be cpu, mps, or mpsgraph.\n";
          return EXIT_FAILURE;
        }
        compute_device = *parsed;
        continue;
      }
      if (arg_equals(argv[index], "--sample")) {
        sampling.do_sample = true;
        continue;
      }
      if (arg_equals(argv[index], "--temperature") && index + 1 < argc) {
        const auto parsed = parse_double_arg(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--temperature must be a number.\n";
          return EXIT_FAILURE;
        }
        sampling.do_sample = true;
        sampling.temperature_set = true;
        sampling.temperature = *parsed;
        continue;
      }
      if (arg_equals(argv[index], "--top-k") && index + 1 < argc) {
        const auto parsed = parse_size_arg(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--top-k must be a non-negative integer.\n";
          return EXIT_FAILURE;
        }
        sampling.do_sample = true;
        sampling.top_k_set = true;
        sampling.top_k = *parsed;
        continue;
      }
      if (arg_equals(argv[index], "--top-p") && index + 1 < argc) {
        const auto parsed = parse_double_arg(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--top-p must be a number.\n";
          return EXIT_FAILURE;
        }
        sampling.do_sample = true;
        sampling.top_p_set = true;
        sampling.top_p = *parsed;
        continue;
      }
      if (arg_equals(argv[index], "--seed") && index + 1 < argc) {
        const auto parsed = parse_u64_arg(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--seed must be a non-negative integer.\n";
          return EXIT_FAILURE;
        }
        sampling.do_sample = true;
        sampling.seed_set = true;
        sampling.seed = *parsed;
        continue;
      }
      if (arg_equals(argv[index], "--stream")) {
        stream = true;
        continue;
      }
      if (arg_equals(argv[index], "--profile") && index + 1 < argc) {
        const auto parsed = toyllm::parse_profile_mode(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--profile must be off, summary, trace, flamegraph, or all.\n";
          return EXIT_FAILURE;
        }
        observability.profile_mode = *parsed;
        continue;
      }
      if (arg_equals(argv[index], "--profile-dir") && index + 1 < argc) {
        observability.profile_output_dir = std::filesystem::path{argv[++index]};
        continue;
      }
      if (arg_equals(argv[index], "--profile-min-us") && index + 1 < argc) {
        const auto parsed = parse_size_arg(argv[++index]);
        if (!parsed.has_value()) {
          std::cerr << "--profile-min-us must be a non-negative integer.\n";
          return EXIT_FAILURE;
        }
        observability.min_duration_us = static_cast<std::uint64_t>(*parsed);
        continue;
      }
      if (arg_equals(argv[index], "--dump-dir") && index + 1 < argc) {
        debug_dump_dir = std::filesystem::path{argv[++index]};
        continue;
      }
      if (arg_equals(argv[index], "--kv-cache-stats")) {
        print_kv_cache_stats = true;
        continue;
      }
      if (arg_equals(argv[index], "--verify-kv-cache")) {
        verify_kv_cache = true;
        continue;
      }
      if (std::string_view(argv[index]).starts_with("-")) {
        std::cerr << "Unknown chat option: " << argv[index] << '\n';
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      if (model_path_set) {
        std::cerr << "chat accepts at most one model directory.\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      model_path = std::filesystem::path{argv[index]};
      model_path_set = true;
    }
    return run_chat(model_path, max_new_tokens, enable_thinking, debug_dump_dir,
                    print_kv_cache_stats, verify_kv_cache, compute_device, sampling, stream,
                    observability);
  }

  const bool explicit_generation = arg_equals(argv[1], "run") || arg_equals(argv[1], "infer");
  const bool legacy_generation = !explicit_generation;
  const int first_option = explicit_generation ? 2 : 1;
  std::size_t max_new_tokens = 16;
  bool enable_thinking = false;
  bool print_kv_cache_stats = false;
  bool verify_kv_cache = false;
  bool stream = false;
  toyllm::Device compute_device = toyllm::Device::cpu();
  toyllm::CpuSamplingConfig sampling;
  toyllm::ObservabilityConfig observability;
  std::filesystem::path debug_dump_dir;
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
    if (arg_equals(argv[index], "--max-new-tokens") && index + 1 < argc) {
      const auto parsed = parse_size_arg(argv[++index]);
      if (!parsed.has_value()) {
        std::cerr << "--max-new-tokens must be a non-negative integer.\n";
        return EXIT_FAILURE;
      }
      max_new_tokens = *parsed;
      continue;
    }
    if (arg_equals(argv[index], "--enable-thinking")) {
      enable_thinking = true;
      continue;
    }
    if (arg_equals(argv[index], "--device") && index + 1 < argc) {
      const auto parsed = parse_device_arg(argv[++index]);
      if (!parsed.has_value()) {
        std::cerr << "--device must be cpu, mps, or mpsgraph.\n";
        return EXIT_FAILURE;
      }
      compute_device = *parsed;
      continue;
    }
    if (arg_equals(argv[index], "--sample")) {
      sampling.do_sample = true;
      continue;
    }
    if (arg_equals(argv[index], "--temperature") && index + 1 < argc) {
      const auto parsed = parse_double_arg(argv[++index]);
      if (!parsed.has_value()) {
        std::cerr << "--temperature must be a number.\n";
        return EXIT_FAILURE;
      }
      sampling.do_sample = true;
      sampling.temperature_set = true;
      sampling.temperature = *parsed;
      continue;
    }
    if (arg_equals(argv[index], "--top-k") && index + 1 < argc) {
      const auto parsed = parse_size_arg(argv[++index]);
      if (!parsed.has_value()) {
        std::cerr << "--top-k must be a non-negative integer.\n";
        return EXIT_FAILURE;
      }
      sampling.do_sample = true;
      sampling.top_k_set = true;
      sampling.top_k = *parsed;
      continue;
    }
    if (arg_equals(argv[index], "--top-p") && index + 1 < argc) {
      const auto parsed = parse_double_arg(argv[++index]);
      if (!parsed.has_value()) {
        std::cerr << "--top-p must be a number.\n";
        return EXIT_FAILURE;
      }
      sampling.do_sample = true;
      sampling.top_p_set = true;
      sampling.top_p = *parsed;
      continue;
    }
    if (arg_equals(argv[index], "--seed") && index + 1 < argc) {
      const auto parsed = parse_u64_arg(argv[++index]);
      if (!parsed.has_value()) {
        std::cerr << "--seed must be a non-negative integer.\n";
        return EXIT_FAILURE;
      }
      sampling.do_sample = true;
      sampling.seed_set = true;
      sampling.seed = *parsed;
      continue;
    }
    if (arg_equals(argv[index], "--stream")) {
      stream = true;
      continue;
    }
    if (arg_equals(argv[index], "--profile") && index + 1 < argc) {
      const auto parsed = toyllm::parse_profile_mode(argv[++index]);
      if (!parsed.has_value()) {
        std::cerr << "--profile must be off, summary, trace, flamegraph, or all.\n";
        return EXIT_FAILURE;
      }
      observability.profile_mode = *parsed;
      continue;
    }
    if (arg_equals(argv[index], "--profile-dir") && index + 1 < argc) {
      observability.profile_output_dir = std::filesystem::path{argv[++index]};
      continue;
    }
    if (arg_equals(argv[index], "--profile-min-us") && index + 1 < argc) {
      const auto parsed = parse_size_arg(argv[++index]);
      if (!parsed.has_value()) {
        std::cerr << "--profile-min-us must be a non-negative integer.\n";
        return EXIT_FAILURE;
      }
      observability.min_duration_us = static_cast<std::uint64_t>(*parsed);
      continue;
    }
    if (arg_equals(argv[index], "--dump-dir") && index + 1 < argc) {
      debug_dump_dir = std::filesystem::path{argv[++index]};
      continue;
    }
    if (arg_equals(argv[index], "--kv-cache-stats")) {
      print_kv_cache_stats = true;
      continue;
    }
    if (arg_equals(argv[index], "--verify-kv-cache")) {
      verify_kv_cache = true;
      continue;
    }

    std::cerr << "Unknown or incomplete argument: " << argv[index] << '\n';
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (legacy_generation && prompt.empty()) {
    std::cerr << "Unknown command: " << argv[1] << '\n';
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  return run_cpu_generation(model_path, prompt, max_new_tokens, enable_thinking, debug_dump_dir,
                            print_kv_cache_stats, verify_kv_cache, compute_device, sampling,
                            stream, observability);
}
