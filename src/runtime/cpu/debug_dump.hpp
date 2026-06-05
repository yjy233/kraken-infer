#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace toyllm::cpu {

class DebugDumper {
 public:
  explicit DebugDumper(std::filesystem::path output_dir = {});

  bool enabled() const;
  void write_f32(std::string_view name, const std::vector<std::uint64_t>& shape,
                 const std::vector<float>& values);
  void write_i64(std::string_view name, const std::vector<std::uint64_t>& shape,
                 const std::vector<std::int64_t>& values);

 private:
  std::filesystem::path make_file_path(std::string_view name, std::string_view extension);
  void write_metadata(std::string_view name, std::string_view dtype,
                      const std::vector<std::uint64_t>& shape,
                      const std::filesystem::path& file_path);

  std::filesystem::path output_dir_;
  std::ofstream metadata_;
  std::size_t sequence_{0};
};

}  // namespace toyllm::cpu
