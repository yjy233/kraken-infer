CXX ?= clang++
BUILD_DIR ?= build/manual
MODEL ?= models/qwen3-0.6b
PROMPT ?= hello
CHAT_TOKENS ?= 16
QWEN35_MODEL ?= models/qwen3.5-0.8b
QWEN35_MMPROJ ?= models/qwen3.5-0.8b/mmproj-Qwen3.5-0.8B-BF16.gguf
QWEN35_IMAGE ?= /Users/bill/code/llama.cpp/tools/mtmd/test-1.jpeg
BINARY := kraken-infer
SMOKE_TEST := kraken-infer-smoke-test

COMMON_FLAGS := -std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Iinclude
DEBUG_FLAGS := -O0 -g
RELEASE_FLAGS := -O3 -DNDEBUG
MPS_FLAGS := -DKRAKEN_INFER_ENABLE_MPS=1
MPSGRAPH_FLAGS := -DKRAKEN_INFER_ENABLE_MPSGRAPH=1
APPLE_FRAMEWORKS := -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph

CORE_SRCS := \
	src/core/status.cpp \
	src/core/tensor.cpp \
	src/model/model_config.cpp \
	src/runtime/cpu/debug_dump.cpp \
	src/runtime/cpu/json_scanner.cpp \
	src/runtime/cpu/kv_cache.cpp \
	src/runtime/cpu/qwen_cpu_model.cpp \
	src/runtime/cpu/safetensors.cpp \
	src/runtime/cpu/tokenizer.cpp \
	src/runtime/cpu_inference.cpp \
	src/runtime/gguf_reader.cpp \
	src/runtime/gguf_tokenizer.cpp \
	src/runtime/mpsgraph_inference.cpp \
	src/runtime/openai_gateway.cpp \
	src/runtime/profiling.cpp \
	src/runtime/qwen35_prefix_cache.cpp \
	src/runtime/qwen35_runtime.cpp \
	src/runtime/qwen35_multimodal.cpp \
	src/runtime/qwen35_vl_adapter.cpp \
	src/runtime/qwen35_weight_map.cpp \
	src/runtime/qwen_tokenizer.cpp \
	src/runtime/reasoning_parser.cpp \
	src/runtime/runtime.cpp \
	src/backends/mpsgraph/mpsgraph_kv_cache.cpp \
	src/backends/mpsgraph/qwen_mpsgraph_model.cpp \
	src/backends/mpsgraph/mpsgraph_weight_store.cpp \
	src/backends/mps/mps_backend.mm \
	src/backends/mpsgraph/mpsgraph_backend.mm

.PHONY: all debug release test qwen35-vl-test cli inspect weights doctor infer run chat serve compare-transformers mps-info clean

all: debug

debug: $(BUILD_DIR)/$(BINARY) $(BUILD_DIR)/$(SMOKE_TEST)

release:
	$(MAKE) BUILD_DIR=build/release EXTRA_FLAGS="$(RELEASE_FLAGS)" debug

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(BINARY): $(CORE_SRCS) apps/kraken_infer_main.cpp | $(BUILD_DIR)
	$(CXX) $(COMMON_FLAGS) $(DEBUG_FLAGS) $(EXTRA_FLAGS) $(MPS_FLAGS) $(MPSGRAPH_FLAGS) $^ $(APPLE_FRAMEWORKS) -o $@

$(BUILD_DIR)/$(SMOKE_TEST): $(CORE_SRCS) tests/smoke_test.cpp | $(BUILD_DIR)
	$(CXX) $(COMMON_FLAGS) $(DEBUG_FLAGS) $(EXTRA_FLAGS) $(MPS_FLAGS) $(MPSGRAPH_FLAGS) $^ $(APPLE_FRAMEWORKS) -o $@

test: $(BUILD_DIR)/$(SMOKE_TEST)
	./$(BUILD_DIR)/$(SMOKE_TEST)

qwen35-vl-test: $(BUILD_DIR)/$(BINARY)
	python3 scripts/test_qwen35_vl_gateway.py \
		--binary ./$(BUILD_DIR)/$(BINARY) \
		--model $(QWEN35_MODEL) \
		--mmproj $(QWEN35_MMPROJ) \
		--image $(QWEN35_IMAGE)

cli: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) help

inspect: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) inspect $(MODEL)

weights: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) weights $(MODEL)

doctor: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) doctor $(MODEL)

infer: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) infer --model $(MODEL) --prompt "$(PROMPT)"

run: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) run --model $(MODEL) --prompt "$(PROMPT)"

chat: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) chat --model $(MODEL) --max-new-tokens $(CHAT_TOKENS)

serve: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) serve --model $(MODEL)

compare-transformers: $(BUILD_DIR)/$(BINARY)
	python3 scripts/compare_cpu_transformers.py --binary ./$(BUILD_DIR)/$(BINARY) --model $(MODEL) --prompt "$(PROMPT)"

mps-info: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) mps

clean:
	rm -rf build
