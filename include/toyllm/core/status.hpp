#pragma once

#include <string>
#include <utility>

namespace toyllm {

enum class StatusCode {
  ok,
  invalid_argument,
  unavailable,
  internal_error,
};

class Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message);

  [[nodiscard]] static Status ok();
  [[nodiscard]] static Status invalid_argument(std::string message);
  [[nodiscard]] static Status unavailable(std::string message);
  [[nodiscard]] static Status internal_error(std::string message);

  [[nodiscard]] bool is_ok() const;
  [[nodiscard]] StatusCode code() const;
  [[nodiscard]] const std::string& message() const;

 private:
  StatusCode code_{StatusCode::ok};
  std::string message_;
};

template <typename T>
class Result {
 public:
  Result(T value) : value_(std::move(value)), status_(Status::ok()) {}
  Result(Status status) : status_(std::move(status)) {}

  [[nodiscard]] bool is_ok() const { return status_.is_ok(); }
  [[nodiscard]] const Status& status() const { return status_; }
  [[nodiscard]] const T& value() const { return value_; }
  [[nodiscard]] T& value() { return value_; }

 private:
  T value_{};
  Status status_;
};

}  // namespace toyllm
