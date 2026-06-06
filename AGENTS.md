# Repository Guidelines

## Project Structure & Module Organization

This is a C++20 toy LLM runtime focused on macOS Metal/MPS experiments. Public headers live in `include/toyllm/` and implementation files live in `src/`; keep matching concepts aligned, for example `include/toyllm/core/tensor.hpp` and `src/core/tensor.cpp`. CLI entry points belong in `apps/`, backend code in `src/backends/`, runtime orchestration in `src/runtime/`, smoke tests in `tests/`, architecture notes in `docs/`, and local model placeholders in `models/`.

## Build, Test, and Development Commands

- `cmake --preset debug`: configure a Debug build in `build/debug` with tests enabled.
- `cmake --build --preset debug`: build `kraken-infer`, `kraken_infer_core`, and the smoke test.
- `ctest --preset debug`: run CTest with failure output.
- `cmake --preset release && cmake --build --preset release`: produce an optimized Release build.
- `./build/debug/kraken-infer --mps-info`: inspect local Metal/MPS availability.
- `make test` and `make mps-info`: fallback commands using macOS `clang++` when CMake is unavailable.

## Coding Style & Naming Conventions

Use C++20 with no compiler extensions. Follow two-space indentation and order includes as local project headers first, then standard library headers. Use `toyllm` for core APIs and nested namespaces such as `toyllm::mps` for backends. Prefer `snake_case` for functions, files, and test helpers; `PascalCase` for classes and structs; and scoped enums such as `enum class DType` with concise lowercase values. Keep code warning-clean under `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`.

## Commenting Guidelines

Write comments for non-obvious implementation details, especially tensor shape assumptions, memory ownership, lifetime constraints, backend fallbacks, and Objective-C++/MPS bridging behavior. Prefer short comments that explain why a decision is made, not comments that restate the code. For new runtime or backend logic, document preconditions and invariants near the relevant function or type. For example, note whether a tensor is CPU-only, whether zero-sized shapes are valid, or why an MPS path falls back to the stub implementation.

## Testing Guidelines

Tests currently use a lightweight CTest smoke binary in `tests/smoke_test.cpp` with `assert`. Add or update unit-style smoke tests for behavior changes, especially tensor validation, runtime selection, config parsing, backend fallback paths, and error handling. Use focused helpers named `test_<behavior>()`, keep fixtures small and deterministic, and avoid requiring downloaded model weights or guaranteed MPS availability. When fixing a bug, add a regression test that fails before the fix where practical. Run `ctest --preset debug` before submitting C++ changes; use `make test` only as the fallback path.

## Commit & Pull Request Guidelines

The current history uses Conventional Commit style (`feat: init`). Continue with concise subjects such as `feat: add tokenizer config parser`, `fix: handle empty tensor shapes`, or `test: cover runtime device selection`. Pull requests should include a summary, changed areas, platform tested, commands run, and relevant CLI output for MPS/backend behavior.

## Security & Configuration Tips

Do not commit large model weights or machine-specific build products. Keep generated files under `build/` and local Qwen assets under `models/qwen3-0.6b/` unless documenting placeholders. Treat MPS availability as platform-dependent and keep CPU/stub paths buildable where possible.
