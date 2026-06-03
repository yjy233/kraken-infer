CXX ?= clang++
BUILD_DIR ?= build/manual

COMMON_FLAGS := -std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Iinclude
DEBUG_FLAGS := -O0 -g
RELEASE_FLAGS := -O3 -DNDEBUG
MPS_FLAGS := -DTOYLLM_ENABLE_MPS=1
MPS_LIBS := -framework Foundation -framework Metal -framework MetalPerformanceShaders

CORE_SRCS := \
	src/core/status.cpp \
	src/core/tensor.cpp \
	src/model/model_config.cpp \
	src/runtime/runtime.cpp \
	src/backends/mps/mps_backend.mm

.PHONY: all debug release test mps-info clean

all: debug

debug: $(BUILD_DIR)/toyllm $(BUILD_DIR)/toyllm_smoke_test

release:
	$(MAKE) BUILD_DIR=build/release EXTRA_FLAGS="$(RELEASE_FLAGS)" debug

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/toyllm: $(CORE_SRCS) apps/toyllm_main.cpp | $(BUILD_DIR)
	$(CXX) $(COMMON_FLAGS) $(DEBUG_FLAGS) $(EXTRA_FLAGS) $(MPS_FLAGS) $^ $(MPS_LIBS) -o $@

$(BUILD_DIR)/toyllm_smoke_test: $(CORE_SRCS) tests/smoke_test.cpp | $(BUILD_DIR)
	$(CXX) $(COMMON_FLAGS) $(DEBUG_FLAGS) $(EXTRA_FLAGS) $(MPS_FLAGS) $^ $(MPS_LIBS) -o $@

test: $(BUILD_DIR)/toyllm_smoke_test
	./$(BUILD_DIR)/toyllm_smoke_test

mps-info: $(BUILD_DIR)/toyllm
	./$(BUILD_DIR)/toyllm --mps-info

clean:
	rm -rf build
