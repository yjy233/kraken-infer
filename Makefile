CXX ?= clang++
BUILD_DIR ?= build/manual
MODEL ?= models/qwen3-0.6b
PROMPT ?= hello
CHAT_TOKENS ?= 16

COMMON_FLAGS := -std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Iinclude
DEBUG_FLAGS := -O0 -g
RELEASE_FLAGS := -O3 -DNDEBUG
MPS_FLAGS := -DTOYLLM_ENABLE_MPS=1
MPS_LIBS := -framework Foundation -framework Metal -framework MetalPerformanceShaders

CORE_SRCS := \
	src/core/status.cpp \
	src/core/tensor.cpp \
	src/model/model_config.cpp \
	src/runtime/cpu_inference.cpp \
	src/runtime/runtime.cpp \
	src/backends/mps/mps_backend.mm

.PHONY: all debug release test cli inspect weights doctor infer run chat mps-info clean

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

cli: $(BUILD_DIR)/toyllm
	./$(BUILD_DIR)/toyllm help

inspect: $(BUILD_DIR)/toyllm
	./$(BUILD_DIR)/toyllm inspect $(MODEL)

weights: $(BUILD_DIR)/toyllm
	./$(BUILD_DIR)/toyllm weights $(MODEL)

doctor: $(BUILD_DIR)/toyllm
	./$(BUILD_DIR)/toyllm doctor $(MODEL)

infer: $(BUILD_DIR)/toyllm
	./$(BUILD_DIR)/toyllm infer --model $(MODEL) --prompt "$(PROMPT)"

run: $(BUILD_DIR)/toyllm
	./$(BUILD_DIR)/toyllm run --model $(MODEL) --prompt "$(PROMPT)"

chat: $(BUILD_DIR)/toyllm
	./$(BUILD_DIR)/toyllm chat --model $(MODEL) --max-new-tokens $(CHAT_TOKENS)

mps-info: $(BUILD_DIR)/toyllm
	./$(BUILD_DIR)/toyllm mps

clean:
	rm -rf build
