cmake_minimum_required(VERSION 3.24)

project(
  toy_llm_interface
  VERSION 0.1.0
  DESCRIPTION "A small from-scratch LLM inference runtime for macOS MPS experiments"
  LANGUAGES CXX
)

option(TOYLLM_ENABLE_MPS "Enable Metal Performance Shaders backend on macOS" ON)
option(TOYLLM_BUILD_TESTS "Build smoke tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectWarnings)

add_library(toyllm_core)
add_library(toyllm::core ALIAS toyllm_core)

target_sources(
  toyllm_core
  PRIVATE
    src/core/status.cpp
    src/core/tensor.cpp
    src/model/model_config.cpp
    src/runtime/runtime.cpp
)

target_include_directories(
  toyllm_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(toyllm_core PUBLIC cxx_std_20)
toyllm_set_project_warnings(toyllm_core)

if(APPLE AND TOYLLM_ENABLE_MPS)
  enable_language(OBJCXX)
  target_sources(toyllm_core PRIVATE src/backends/mps/mps_backend.mm)
  target_compile_definitions(toyllm_core PUBLIC TOYLLM_ENABLE_MPS=1)
  target_link_libraries(
    toyllm_core
    PUBLIC
      "-framework Foundation"
      "-framework Metal"
      "-framework MetalPerformanceShaders"
  )
else()
  target_sources(toyllm_core PRIVATE src/backends/mps/mps_backend_stub.cpp)
  target_compile_definitions(toyllm_core PUBLIC TOYLLM_ENABLE_MPS=0)
endif()

add_executable(toyllm apps/toyllm_main.cpp)
target_link_libraries(toyllm PRIVATE toyllm::core)
toyllm_set_project_warnings(toyllm)

if(TOYLLM_BUILD_TESTS)
  enable_testing()
  add_executable(toyllm_smoke_test tests/smoke_test.cpp)
  target_link_libraries(toyllm_smoke_test PRIVATE toyllm::core)
  toyllm_set_project_warnings(toyllm_smoke_test)
  add_test(NAME toyllm_smoke_test COMMAND toyllm_smoke_test)
endif()
