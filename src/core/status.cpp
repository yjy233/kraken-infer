#include "toyllm/core/status.hpp"

namespace toyllm {

Status::Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

Status Status::ok() {
  return Status{};
}

Status Status::invalid_argument(std::string message) {
  return Status{StatusCode::invalid_argument, std::move(message)};
}

Status Status::unavailable(std::string message) {
  return Status{StatusCode::unavailable, std::move(message)};
}

Status Status::internal_error(std::string message) {
  return Status{StatusCode::internal_error, std::move(message)};
}

bool Status::is_ok() const {
  return code_ == StatusCode::ok;
}

StatusCode Status::code() const {
  return code_;
}

const std::string& Status::message() const {
  return message_;
}

}  // namespace toyllm
