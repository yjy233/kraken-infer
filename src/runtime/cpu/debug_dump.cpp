#include "debug_dump.hpp"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace toyllm::cpu {

namespace {

std::string sanitize_name(std::string_view name) {
  std::string output;
  output.reserve(name.size());
  for (const auto ch : name) {
    const auto byte = static_cast<unsigned char>(ch);
    if (std::isalnum(byte) != 0 || ch == '-' || ch == '_' || ch == '.') {
      output.push_back(ch);
    } else {
      output.push_back('_');
    }
  }
  return output;
}

std::string shape_json(const std::vector<std::uint64_t>& shape) {
  std::ostringstream output;
  output << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      output << ',';
    }
    output << shape[i];
  }
  output << ']';
  return output.str();
}

}  // namespace

DebugDumper::DebugDumper(std::filesystem::path output_dir) : output_dir_(std::move(output_dir)) {
  if (!enabled()) {
    return;
  }

  std::filesystem::create_directories(output_dir_);
  metadata_.open(output_dir_ / "metadata.jsonl", std::ios::out | std::ios::trunc);
  if (!metadata_) {
    throw std::runtime_error("failed to create debug dump metadata in " + output_dir_.string());
  }
}

bool DebugDumper::enabled() const { return !output_dir_.empty(); }

void DebugDumper::write_f32(std::string_view name, const std::vector<std::uint64_t>& shape,
                            const std::vector<float>& values) {
  if (!enabled()) {
    return;
  }
  const auto file_path = make_file_path(name, ".f32.bin");
  std::ofstream output(file_path, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to create debug dump tensor " + file_path.string());
  }
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  write_metadata(name, "f32", shape, file_path);
}

void DebugDumper::write_i64(std::string_view name, const std::vector<std::uint64_t>& shape,
                            const std::vector<std::int64_t>& values) {
  if (!enabled()) {
    return;
  }
  const auto file_path = make_file_path(name, ".i64.bin");
  std::ofstream output(file_path, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to create debug dump tensor " + file_path.string());
  }
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(std::int64_t)));
  write_metadata(name, "i64", shape, file_path);
}

std::filesystem::path DebugDumper::make_file_path(std::string_view name,
                                                  std::string_view extension) {
  std::ostringstream filename;
  filename << sequence_++ << "." << sanitize_name(name) << extension;
  return output_dir_ / filename.str();
}

void DebugDumper::write_metadata(std::string_view name, std::string_view dtype,
                                 const std::vector<std::uint64_t>& shape,
                                 const std::filesystem::path& file_path) {
  metadata_ << "{\"name\":\"" << name << "\",\"dtype\":\"" << dtype << "\",\"shape\":"
            << shape_json(shape) << ",\"file\":\"" << file_path.filename().string() << "\"}\n";
}

}  // namespace toyllm::cpu
