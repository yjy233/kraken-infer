#include "toyllm/backends/mpsgraph/mpsgraph_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace toyllm::mpsgraph {

namespace {

using SteadyClock = std::chrono::steady_clock;

constexpr const char* kNotReady = "MPSGraph context is not initialized";

std::string nsstring_to_string(NSString* value) {
  return value == nil ? std::string{} : std::string{[value UTF8String]};
}

std::string exception_to_string(NSException* exception) {
  std::string message = "MPSGraph exception";
  if (exception.name != nil) {
    message += " ";
    message += [exception.name UTF8String];
  }
  if (exception.reason != nil) {
    message += ": ";
    message += [exception.reason UTF8String];
  }
  return message;
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t& output) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  output = lhs * rhs;
  return true;
}

Result<std::size_t> f32_bytes(std::size_t values, const char* name) {
  std::size_t bytes = 0;
  if (!checked_mul(values, sizeof(float), bytes)) {
    return Status::invalid_argument(std::string{"MPSGraph "} + name +
                                    " byte count overflow");
  }
  return bytes;
}

Status validate_f32_buffer(const MpsGraphBuffer& buffer, std::size_t values,
                           const char* name) {
  auto byte_count = f32_bytes(values, name);
  if (!byte_count.is_ok()) {
    return byte_count.status();
  }
  if (!buffer.valid()) {
    return Status::invalid_argument(std::string{"MPSGraph "} + name +
                                    " buffer is not initialized");
  }
  if (buffer.byte_size() < byte_count.value()) {
    return Status::invalid_argument(std::string{"MPSGraph "} + name +
                                    " buffer is too small");
  }
  return Status::ok();
}

Status validate_i32_buffer(const MpsGraphBuffer& buffer, std::size_t values,
                           const char* name) {
  std::size_t byte_count = 0;
  if (!checked_mul(values, sizeof(std::int32_t), byte_count)) {
    return Status::invalid_argument(std::string{"MPSGraph "} + name +
                                    " byte count overflow");
  }
  if (!buffer.valid()) {
    return Status::invalid_argument(std::string{"MPSGraph "} + name +
                                    " buffer is not initialized");
  }
  if (buffer.byte_size() < byte_count) {
    return Status::invalid_argument(std::string{"MPSGraph "} + name +
                                    " buffer is too small");
  }
  return Status::ok();
}

Status validate_positive_dim(std::size_t value, const char* name) {
  if (value == 0) {
    return Status::invalid_argument(std::string{"MPSGraph "} + name +
                                    " must be greater than zero");
  }
  if (value > static_cast<std::size_t>(std::numeric_limits<NSInteger>::max())) {
    return Status::invalid_argument(std::string{"MPSGraph "} + name +
                                    " exceeds NSInteger range");
  }
  return Status::ok();
}

MPSShape* make_shape(std::initializer_list<std::size_t> dims) {
  NSMutableArray<NSNumber*>* shape =
    [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(dims.size())];
  for (const auto dim : dims) {
    [shape addObject:@(static_cast<NSInteger>(dim))];
  }
  return shape;
}

MPSGraph* make_graph(MpsGraphGraphStats* stats) {
  const auto started = SteadyClock::now();
  MPSGraph* graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
    SteadyClock::now() - started);
  if (stats != nullptr) {
    ++stats->graph_build_calls;
    stats->graph_build_ns += static_cast<std::uint64_t>(elapsed.count());
  }
  return graph;
}

Status run_graph_with_results(MPSGraph* graph, id<MTLCommandQueue> queue,
                              NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                              NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results,
                              MpsGraphGraphStats* stats) {
  if (graph == nil || queue == nil || results == nil || results.count == 0) {
    return Status::unavailable("failed to bind MPSGraph execution");
  }

  @try {
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* mutable_results =
      [results mutableCopy];
    const auto started = SteadyClock::now();
    [graph runWithMTLCommandQueue:queue
                            feeds:feeds
                 targetOperations:nil
                resultsDictionary:mutable_results];
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      SteadyClock::now() - started);
    if (stats != nullptr) {
      ++stats->graph_execute_calls;
      stats->graph_execute_ns += static_cast<std::uint64_t>(elapsed.count());
      ++stats->executable_cache_misses;
    }
    [mutable_results release];
  } @catch (NSException* exception) {
    return Status::internal_error(exception_to_string(exception));
  }

  return Status::ok();
}

Status run_executable_with_results(
  MPSGraphExecutable* executable, id<MTLCommandQueue> queue,
  NSArray<MPSGraphTensor*>* feed_tensors,
  NSArray<MPSGraphTensor*>* target_tensors,
  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results,
  MpsGraphGraphStats* stats) {
  if (executable == nil || queue == nil || feed_tensors == nil ||
      target_tensors == nil || results == nil || results.count == 0) {
    return Status::unavailable("failed to bind MPSGraph executable execution");
  }

  NSMutableArray<MPSGraphTensorData*>* input_array =
    [NSMutableArray arrayWithCapacity:feed_tensors.count];
  for (MPSGraphTensor* tensor in feed_tensors) {
    MPSGraphTensorData* data = [feeds objectForKey:tensor];
    if (data == nil) {
      return Status::unavailable("failed to bind MPSGraph executable input");
    }
    [input_array addObject:data];
  }

  NSMutableArray<MPSGraphTensorData*>* result_array =
    [NSMutableArray arrayWithCapacity:target_tensors.count];
  for (MPSGraphTensor* tensor in target_tensors) {
    MPSGraphTensorData* data = [results objectForKey:tensor];
    if (data == nil) {
      return Status::unavailable("failed to bind MPSGraph executable output");
    }
    [result_array addObject:data];
  }

  @try {
    const auto started = SteadyClock::now();
    [executable runWithMTLCommandQueue:queue
                           inputsArray:input_array
                          resultsArray:result_array
                   executionDescriptor:nil];
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      SteadyClock::now() - started);
    if (stats != nullptr) {
      ++stats->graph_execute_calls;
      stats->graph_execute_ns += static_cast<std::uint64_t>(elapsed.count());
    }
  } @catch (NSException* exception) {
    return Status::internal_error(exception_to_string(exception));
  }

  return Status::ok();
}

Status run_graph_with_results(MPSGraph* graph, id<MTLCommandQueue> queue,
                              NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                              MPSGraphTensor* output_tensor,
                              MPSGraphTensorData* output_data,
                              MpsGraphGraphStats* stats) {
  if (output_tensor == nil || output_data == nil) {
    return Status::unavailable("failed to bind MPSGraph execution");
  }
  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
    output_tensor : output_data,
  };
  return run_graph_with_results(graph, queue, feeds, results, stats);
}

Status check_close(float actual, float expected, const char* name) {
  if (std::abs(actual - expected) > 1e-4F) {
    std::ostringstream output;
    output << "MPSGraph " << name << " expected " << expected << " but got " << actual;
    return Status::internal_error(output.str());
  }
  return Status::ok();
}

MPSGraphTensor* build_rms_norm(MPSGraph* graph, MPSGraphTensor* input_tensor,
                               MPSGraphTensor* weight_tensor, std::size_t size,
                               float eps) {
  MPSGraphTensor* squared = [graph squareWithTensor:input_tensor name:nil];
  MPSGraphTensor* sum = [graph reductionSumWithTensor:squared axis:0 name:nil];
  MPSGraphTensor* denom =
    [graph constantWithScalar:static_cast<double>(size)
                        shape:@[ @1 ]
                     dataType:MPSDataTypeFloat32];
  MPSGraphTensor* mean =
    [graph divisionWithPrimaryTensor:sum secondaryTensor:denom name:nil];
  MPSGraphTensor* eps_tensor =
    [graph constantWithScalar:static_cast<double>(eps)
                        shape:@[ @1 ]
                     dataType:MPSDataTypeFloat32];
  MPSGraphTensor* variance =
    [graph additionWithPrimaryTensor:mean secondaryTensor:eps_tensor name:nil];
  MPSGraphTensor* scale = [graph reciprocalSquareRootWithTensor:variance name:nil];
  MPSGraphTensor* normalized =
    [graph multiplicationWithPrimaryTensor:input_tensor secondaryTensor:scale name:nil];
  return [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:weight_tensor
                                           name:nil];
}

MPSGraphTensor* build_qk_norm(MPSGraph* graph, MPSGraphTensor* input_tensor,
                              MPSGraphTensor* weight_tensor, std::size_t heads,
                              std::size_t head_dim, float eps) {
  MPSShape* matrix_shape = make_shape({heads, head_dim});
  MPSShape* head_scale_shape = make_shape({heads, 1});
  MPSGraphTensor* squared = [graph squareWithTensor:input_tensor name:nil];
  MPSGraphTensor* sum = [graph reductionSumWithTensor:squared axis:1 name:nil];
  MPSGraphTensor* denom =
    [graph constantWithScalar:static_cast<double>(head_dim)
                        shape:@[ @1 ]
                     dataType:MPSDataTypeFloat32];
  MPSGraphTensor* mean =
    [graph divisionWithPrimaryTensor:sum secondaryTensor:denom name:nil];
  MPSGraphTensor* eps_tensor =
    [graph constantWithScalar:static_cast<double>(eps)
                        shape:@[ @1 ]
                     dataType:MPSDataTypeFloat32];
  MPSGraphTensor* variance =
    [graph additionWithPrimaryTensor:mean secondaryTensor:eps_tensor name:nil];
  MPSGraphTensor* scale = [graph reciprocalSquareRootWithTensor:variance name:nil];
  MPSGraphTensor* scale_column =
    [graph reshapeTensor:scale withShape:head_scale_shape name:nil];
  MPSGraphTensor* scale_matrix =
    [graph broadcastTensor:scale_column toShape:matrix_shape name:nil];
  MPSGraphTensor* weight_row =
    [graph reshapeTensor:weight_tensor withShape:make_shape({1, head_dim}) name:nil];
  MPSGraphTensor* weight_matrix =
    [graph broadcastTensor:weight_row toShape:matrix_shape name:nil];
  MPSGraphTensor* normalized =
    [graph multiplicationWithPrimaryTensor:input_tensor secondaryTensor:scale_matrix name:nil];
  return [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:weight_matrix
                                           name:nil];
}

MPSGraphTensor* build_rope(MPSGraph* graph, MPSGraphTensor* input_tensor,
                           std::size_t heads, std::size_t head_dim,
                           const std::vector<float>& cos_values,
                           const std::vector<float>& sin_values,
                           std::size_t trig_bytes) {
  const auto half_dim = head_dim / 2U;
  MPSShape* half_shape = make_shape({heads, half_dim});
  MPSShape* trig_shape = make_shape({1, half_dim});
  MPSGraphTensor* first_half =
    [graph sliceTensor:input_tensor dimension:1 start:0
                length:static_cast<NSInteger>(half_dim) name:nil];
  MPSGraphTensor* second_half =
    [graph sliceTensor:input_tensor dimension:1
                 start:static_cast<NSInteger>(half_dim)
                length:static_cast<NSInteger>(half_dim) name:nil];

  NSData* cos_data = [NSData dataWithBytes:cos_values.data() length:trig_bytes];
  NSData* sin_data = [NSData dataWithBytes:sin_values.data() length:trig_bytes];
  MPSGraphTensor* cos_row =
    [graph constantWithData:cos_data shape:trig_shape dataType:MPSDataTypeFloat32];
  MPSGraphTensor* sin_row =
    [graph constantWithData:sin_data shape:trig_shape dataType:MPSDataTypeFloat32];
  MPSGraphTensor* cos_matrix =
    [graph broadcastTensor:cos_row toShape:half_shape name:nil];
  MPSGraphTensor* sin_matrix =
    [graph broadcastTensor:sin_row toShape:half_shape name:nil];

  MPSGraphTensor* first_cos =
    [graph multiplicationWithPrimaryTensor:first_half secondaryTensor:cos_matrix name:nil];
  MPSGraphTensor* second_sin =
    [graph multiplicationWithPrimaryTensor:second_half secondaryTensor:sin_matrix name:nil];
  MPSGraphTensor* rotated_first =
    [graph subtractionWithPrimaryTensor:first_cos secondaryTensor:second_sin name:nil];
  MPSGraphTensor* second_cos =
    [graph multiplicationWithPrimaryTensor:second_half secondaryTensor:cos_matrix name:nil];
  MPSGraphTensor* first_sin =
    [graph multiplicationWithPrimaryTensor:first_half secondaryTensor:sin_matrix name:nil];
  MPSGraphTensor* rotated_second =
    [graph additionWithPrimaryTensor:second_cos secondaryTensor:first_sin name:nil];
  return [graph concatTensor:rotated_first withTensor:rotated_second dimension:1 name:nil];
}

MPSGraphTensor* build_grouped_attention_from_cache(
  MPSGraph* graph, MPSGraphTensor* query_tensor, MPSGraphTensor* key_cache_tensor,
  MPSGraphTensor* value_cache_tensor, std::size_t layer, std::size_t position,
  std::size_t capacity_tokens, std::size_t heads, std::size_t kv_heads,
  std::size_t head_dim) {
  const auto seq_len = position + 1U;
  const auto kv_group = heads / kv_heads;

  MPSGraphTensor* layer_keys =
    [graph sliceTensor:key_cache_tensor dimension:0
                 start:static_cast<NSInteger>(layer) length:1 name:nil];
  MPSGraphTensor* layer_values =
    [graph sliceTensor:value_cache_tensor dimension:0
                 start:static_cast<NSInteger>(layer) length:1 name:nil];
  layer_keys =
    [graph reshapeTensor:layer_keys
               withShape:make_shape({capacity_tokens, kv_heads, head_dim})
                    name:nil];
  layer_values =
    [graph reshapeTensor:layer_values
               withShape:make_shape({capacity_tokens, kv_heads, head_dim})
                    name:nil];
  MPSGraphTensor* visible_keys =
    [graph sliceTensor:layer_keys dimension:0 start:0
                length:static_cast<NSInteger>(seq_len) name:nil];
  MPSGraphTensor* visible_values =
    [graph sliceTensor:layer_values dimension:0 start:0
                length:static_cast<NSInteger>(seq_len) name:nil];

  NSMutableArray<MPSGraphTensor*>* group_outputs =
    [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(kv_heads)];
  MPSGraphTensor* scale_tensor =
    [graph constantWithScalar:1.0 / std::sqrt(static_cast<double>(head_dim))
                        shape:@[ @1 ]
                     dataType:MPSDataTypeFloat32];
  for (std::size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const auto query_head_start = kv_head * kv_group;
    MPSGraphTensor* query_group =
      [graph sliceTensor:query_tensor dimension:0
                   start:static_cast<NSInteger>(query_head_start)
                  length:static_cast<NSInteger>(kv_group)
                    name:nil];
    MPSGraphTensor* query_group_t =
      [graph transposeTensor:query_group dimension:0 withDimension:1 name:nil];

    MPSGraphTensor* key_head =
      [graph sliceTensor:visible_keys dimension:1
                   start:static_cast<NSInteger>(kv_head) length:1 name:nil];
    key_head =
      [graph reshapeTensor:key_head withShape:make_shape({seq_len, head_dim}) name:nil];
    MPSGraphTensor* scores =
      [graph matrixMultiplicationWithPrimaryTensor:key_head
                                   secondaryTensor:query_group_t
                                              name:nil];
    scores =
      [graph multiplicationWithPrimaryTensor:scores secondaryTensor:scale_tensor name:nil];
    MPSGraphTensor* probabilities = [graph softMaxWithTensor:scores axis:0 name:nil];
    probabilities =
      [graph transposeTensor:probabilities dimension:0 withDimension:1 name:nil];

    MPSGraphTensor* value_head =
      [graph sliceTensor:visible_values dimension:1
                   start:static_cast<NSInteger>(kv_head) length:1 name:nil];
    value_head =
      [graph reshapeTensor:value_head withShape:make_shape({seq_len, head_dim}) name:nil];
    MPSGraphTensor* group_output =
      [graph matrixMultiplicationWithPrimaryTensor:probabilities
                                   secondaryTensor:value_head
                                              name:nil];
    [group_outputs addObject:group_output];
  }

  return kv_heads == 1U
           ? [group_outputs objectAtIndex:0]
           : [graph concatTensors:group_outputs dimension:0 name:nil];
}

MPSGraphTensor* build_attention_from_cache(
  MPSGraph* graph, MPSGraphTensor* query_tensor, MPSGraphTensor* key_cache_tensor,
  MPSGraphTensor* value_cache_tensor, std::size_t layer, std::size_t position,
  std::size_t capacity_tokens, std::size_t heads, std::size_t kv_heads,
  std::size_t head_dim, bool use_sdpa) {
  const auto seq_len = position + 1U;
  const auto kv_group = heads / kv_heads;

  if (use_sdpa) {
    if (@available(macOS 15.0, *)) {
      MPSGraphTensor* layer_keys =
        [graph sliceTensor:key_cache_tensor dimension:0
                     start:static_cast<NSInteger>(layer) length:1 name:nil];
      MPSGraphTensor* layer_values =
        [graph sliceTensor:value_cache_tensor dimension:0
                     start:static_cast<NSInteger>(layer) length:1 name:nil];
      layer_keys =
        [graph reshapeTensor:layer_keys
                   withShape:make_shape({capacity_tokens, kv_heads, head_dim})
                        name:nil];
      layer_values =
        [graph reshapeTensor:layer_values
                   withShape:make_shape({capacity_tokens, kv_heads, head_dim})
                        name:nil];
      MPSGraphTensor* visible_keys =
        [graph sliceTensor:layer_keys dimension:0 start:0
                    length:static_cast<NSInteger>(seq_len) name:nil];
      MPSGraphTensor* visible_values =
        [graph sliceTensor:layer_values dimension:0 start:0
                    length:static_cast<NSInteger>(seq_len) name:nil];

      MPSGraphTensor* sdpa_query =
        [graph reshapeTensor:query_tensor
                   withShape:make_shape({1, heads, 1, head_dim})
                        name:nil];
      MPSGraphTensor* sdpa_keys =
        [graph transposeTensor:visible_keys permutation:@[ @1, @0, @2 ] name:nil];
      sdpa_keys = [graph reshapeTensor:sdpa_keys
                              withShape:make_shape({kv_heads, 1, seq_len, head_dim})
                                   name:nil];
      sdpa_keys = [graph tileTensor:sdpa_keys
                      withMultiplier:make_shape({1, kv_group, 1, 1})
                                name:nil];
      sdpa_keys = [graph reshapeTensor:sdpa_keys
                              withShape:make_shape({1, heads, seq_len, head_dim})
                                   name:nil];

      MPSGraphTensor* sdpa_values =
        [graph transposeTensor:visible_values permutation:@[ @1, @0, @2 ] name:nil];
      sdpa_values = [graph reshapeTensor:sdpa_values
                                withShape:make_shape({kv_heads, 1, seq_len, head_dim})
                                     name:nil];
      sdpa_values = [graph tileTensor:sdpa_values
                        withMultiplier:make_shape({1, kv_group, 1, 1})
                                  name:nil];
      sdpa_values = [graph reshapeTensor:sdpa_values
                                withShape:make_shape({1, heads, seq_len, head_dim})
                                     name:nil];

      const auto attention_scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
      MPSGraphTensor* sdpa_output =
        [graph scaledDotProductAttentionWithQueryTensor:sdpa_query
                                              keyTensor:sdpa_keys
                                            valueTensor:sdpa_values
                                                  scale:attention_scale
                                                   name:nil];
      return [graph reshapeTensor:sdpa_output
                        withShape:make_shape({heads, head_dim})
                             name:nil];
    }
  }

  return build_grouped_attention_from_cache(
    graph, query_tensor, key_cache_tensor, value_cache_tensor, layer, position,
    capacity_tokens, heads, kv_heads, head_dim);
}

struct TransformerLayerGraphOutputs {
  MPSGraphTensor* hidden{nil};
  MPSGraphTensor* key_cache{nil};
  MPSGraphTensor* value_cache{nil};
};

TransformerLayerGraphOutputs build_transformer_layer_block(
  MPSGraph* graph,
  MPSGraphTensor* hidden_tensor,
  MPSGraphTensor* input_norm_tensor,
  MPSGraphTensor* q_weight_tensor,
  MPSGraphTensor* k_weight_tensor,
  MPSGraphTensor* v_weight_tensor,
  MPSGraphTensor* o_weight_tensor,
  MPSGraphTensor* q_norm_tensor,
  MPSGraphTensor* k_norm_tensor,
  MPSGraphTensor* post_norm_tensor,
  MPSGraphTensor* gate_weight_tensor,
  MPSGraphTensor* up_weight_tensor,
  MPSGraphTensor* down_weight_tensor,
  MPSGraphTensor* key_cache_tensor,
  MPSGraphTensor* value_cache_tensor,
  std::size_t layer,
  std::size_t position,
  std::size_t capacity_tokens,
  std::size_t hidden_size,
  std::size_t intermediate_size,
  std::size_t heads,
  std::size_t kv_heads,
  std::size_t head_dim,
  float eps,
  const std::vector<float>& cos_values,
  const std::vector<float>& sin_values,
  std::size_t trig_bytes,
  bool use_sdpa_attention) {
  const auto attn_dim = heads * head_dim;
  MPSShape* hidden_shape = make_shape({hidden_size});
  MPSShape* hidden_column_shape = make_shape({hidden_size, 1});
  MPSShape* intermediate_shape = make_shape({intermediate_size});
  MPSShape* intermediate_column_shape = make_shape({intermediate_size, 1});
  MPSShape* q_shape = make_shape({heads, head_dim});
  MPSShape* k_shape = make_shape({kv_heads, head_dim});
  MPSShape* kv_update_shape = make_shape({1, 1, kv_heads, head_dim});

  MPSGraphTensor* normed =
    build_rms_norm(graph, hidden_tensor, input_norm_tensor, hidden_size, eps);
  MPSGraphTensor* normed_column =
    [graph reshapeTensor:normed withShape:hidden_column_shape name:nil];
  MPSGraphTensor* q_projected =
    [graph matrixMultiplicationWithPrimaryTensor:q_weight_tensor
                                 secondaryTensor:normed_column
                                            name:nil];
  MPSGraphTensor* k_projected =
    [graph matrixMultiplicationWithPrimaryTensor:k_weight_tensor
                                 secondaryTensor:normed_column
                                            name:nil];
  MPSGraphTensor* v_projected =
    [graph matrixMultiplicationWithPrimaryTensor:v_weight_tensor
                                 secondaryTensor:normed_column
                                            name:nil];
  MPSGraphTensor* q_matrix =
    [graph reshapeTensor:q_projected withShape:q_shape name:nil];
  MPSGraphTensor* k_matrix =
    [graph reshapeTensor:k_projected withShape:k_shape name:nil];
  MPSGraphTensor* v_matrix =
    [graph reshapeTensor:v_projected withShape:k_shape name:nil];
  MPSGraphTensor* q_normed =
    build_qk_norm(graph, q_matrix, q_norm_tensor, heads, head_dim, eps);
  MPSGraphTensor* k_normed =
    build_qk_norm(graph, k_matrix, k_norm_tensor, kv_heads, head_dim, eps);
  MPSGraphTensor* q_rope =
    build_rope(graph, q_normed, heads, head_dim, cos_values, sin_values,
               trig_bytes);
  MPSGraphTensor* k_rope =
    build_rope(graph, k_normed, kv_heads, head_dim, cos_values, sin_values,
               trig_bytes);

  MPSGraphTensor* key_update =
    [graph reshapeTensor:k_rope withShape:kv_update_shape name:nil];
  MPSGraphTensor* value_update =
    [graph reshapeTensor:v_matrix withShape:kv_update_shape name:nil];
  NSArray<NSNumber*>* cache_starts = @[
    @(static_cast<NSInteger>(layer)),
    @(static_cast<NSInteger>(position)),
    @0,
    @0,
  ];
  NSArray<NSNumber*>* cache_ends = @[
    @(static_cast<NSInteger>(layer + 1U)),
    @(static_cast<NSInteger>(position + 1U)),
    @(static_cast<NSInteger>(kv_heads)),
    @(static_cast<NSInteger>(head_dim)),
  ];
  NSArray<NSNumber*>* cache_strides = @[ @1, @1, @1, @1 ];
  MPSGraphTensor* key_cache_result =
    [graph sliceUpdateDataTensor:key_cache_tensor
                     updateTensor:key_update
                           starts:cache_starts
                             ends:cache_ends
                          strides:cache_strides
                        startMask:0
                          endMask:0
                      squeezeMask:0
                             name:nil];
  MPSGraphTensor* value_cache_result =
    [graph sliceUpdateDataTensor:value_cache_tensor
                     updateTensor:value_update
                           starts:cache_starts
                             ends:cache_ends
                          strides:cache_strides
                        startMask:0
                          endMask:0
                      squeezeMask:0
                             name:nil];

  MPSGraphTensor* attention =
    build_attention_from_cache(graph, q_rope, key_cache_result, value_cache_result,
                               layer, position, capacity_tokens, heads, kv_heads,
                               head_dim, use_sdpa_attention);
  MPSGraphTensor* attention_vector =
    [graph reshapeTensor:attention withShape:make_shape({attn_dim}) name:nil];
  MPSGraphTensor* attention_column =
    [graph reshapeTensor:attention_vector withShape:make_shape({attn_dim, 1}) name:nil];
  MPSGraphTensor* projected =
    [graph matrixMultiplicationWithPrimaryTensor:o_weight_tensor
                                 secondaryTensor:attention_column
                                            name:nil];
  MPSGraphTensor* projected_vector =
    [graph reshapeTensor:projected withShape:hidden_shape name:nil];
  MPSGraphTensor* attention_residual =
    [graph additionWithPrimaryTensor:hidden_tensor
                      secondaryTensor:projected_vector
                                 name:nil];
  MPSGraphTensor* post_norm =
    build_rms_norm(graph, attention_residual, post_norm_tensor, hidden_size, eps);
  MPSGraphTensor* post_norm_column =
    [graph reshapeTensor:post_norm withShape:hidden_column_shape name:nil];

  MPSGraphTensor* gate_product =
    [graph matrixMultiplicationWithPrimaryTensor:gate_weight_tensor
                                 secondaryTensor:post_norm_column
                                            name:nil];
  MPSGraphTensor* up_product =
    [graph matrixMultiplicationWithPrimaryTensor:up_weight_tensor
                                 secondaryTensor:post_norm_column
                                            name:nil];
  MPSGraphTensor* gate_vector =
    [graph reshapeTensor:gate_product withShape:intermediate_shape name:nil];
  MPSGraphTensor* up_vector =
    [graph reshapeTensor:up_product withShape:intermediate_shape name:nil];
  MPSGraphTensor* sigmoid = [graph sigmoidWithTensor:gate_vector name:nil];
  MPSGraphTensor* silu =
    [graph multiplicationWithPrimaryTensor:gate_vector secondaryTensor:sigmoid name:nil];
  MPSGraphTensor* gated =
    [graph multiplicationWithPrimaryTensor:silu secondaryTensor:up_vector name:nil];
  MPSGraphTensor* gated_column =
    [graph reshapeTensor:gated withShape:intermediate_column_shape name:nil];
  MPSGraphTensor* down_projected =
    [graph matrixMultiplicationWithPrimaryTensor:down_weight_tensor
                                 secondaryTensor:gated_column
                                            name:nil];
  MPSGraphTensor* down_vector =
    [graph reshapeTensor:down_projected withShape:hidden_shape name:nil];
  MPSGraphTensor* hidden_result =
    [graph additionWithPrimaryTensor:attention_residual
                      secondaryTensor:down_vector
                                 name:nil];

  return TransformerLayerGraphOutputs{hidden_result, key_cache_result,
                                      value_cache_result};
}

Result<MpsGraphBuffer> make_f32_buffer(const MpsGraphContext& context,
                                       const std::vector<float>& values) {
  auto buffer_result = context.make_buffer(values.size() * sizeof(float));
  if (!buffer_result.is_ok()) {
    return buffer_result.status();
  }
  auto buffer = std::move(buffer_result.value());
  const auto copy_status =
    context.copy_to_buffer(buffer, values.data(), values.size() * sizeof(float));
  if (!copy_status.is_ok()) {
    return copy_status;
  }
  return buffer;
}

Result<std::vector<float>> read_f32_buffer(const MpsGraphContext& context,
                                           const MpsGraphBuffer& buffer,
                                           std::size_t values) {
  std::vector<float> output(values);
  const auto status = context.copy_from_buffer(buffer, output.data(),
                                              values * sizeof(float));
  if (!status.is_ok()) {
    return status;
  }
  return output;
}

std::string transformer_layer_cache_key(std::size_t layer,
                                        std::size_t layers,
                                        std::size_t position,
                                        std::size_t capacity_tokens,
                                        std::size_t hidden_size,
                                        std::size_t intermediate_size,
                                        std::size_t heads,
                                        std::size_t kv_heads,
                                        std::size_t head_dim,
                                        bool use_sdpa_attention) {
  std::ostringstream key;
  key << "qwen.layer.full.f32.v1"
      << ":layer=" << layer
      << ":layers=" << layers
      << ":pos=" << position
      << ":cap=" << capacity_tokens
      << ":hidden=" << hidden_size
      << ":inter=" << intermediate_size
      << ":heads=" << heads
      << ":kv_heads=" << kv_heads
      << ":head_dim=" << head_dim
      << ":attn=" << (use_sdpa_attention ? "sdpa" : "grouped");
  return key.str();
}

MPSGraphShapedType* make_f32_shaped_type(MPSShape* shape) {
  return [[[MPSGraphShapedType alloc] initWithShape:shape
                                           dataType:MPSDataTypeFloat32] autorelease];
}

BackendInfo build_backend_info() {
  BackendInfo info{};
  info.compiled = true;

  @autoreleasepool {
    id<MTLDevice> metal_device = MTLCreateSystemDefaultDevice();
    if (metal_device == nil) {
      info.failure_reason = "Metal device is not available";
      return info;
    }

    MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:metal_device];
    if (graph_device == nil) {
      info.failure_reason = "failed to create MPSGraphDevice from Metal device";
      [metal_device release];
      return info;
    }

    id<MTLCommandQueue> command_queue = [metal_device newCommandQueue];
    if (command_queue == nil) {
      info.failure_reason = "failed to create Metal command queue for MPSGraph";
      [metal_device release];
      return info;
    }

    info.available = true;
    info.graph_ready = true;
    info.device_name = nsstring_to_string(metal_device.name);
    info.recommended_max_working_set_size =
      static_cast<std::uint64_t>(metal_device.recommendedMaxWorkingSetSize);
    info.low_power = metal_device.lowPower;
    info.headless = metal_device.headless;
    info.removable = metal_device.removable;

    [command_queue release];
    [metal_device release];
  }

  return info;
}

}  // namespace

struct MpsGraphBuffer::Impl {
  id<MTLBuffer> buffer{nil};
  std::size_t byte_size{0};

  ~Impl() {
    if (buffer != nil) {
      [buffer release];
    }
  }
};

struct MpsGraphContext::Impl {
  struct TransformerLayerExecutable {
    MPSGraph* graph{nil};
    MPSGraphExecutable* executable{nil};
    NSArray<MPSGraphTensor*>* feed_tensors{nil};
    NSArray<MPSGraphTensor*>* run_feed_tensors{nil};
    NSArray<MPSGraphTensor*>* target_tensors{nil};
    NSArray<MPSGraphTensor*>* run_target_tensors{nil};

    ~TransformerLayerExecutable() {
      if (run_target_tensors != nil) {
        [run_target_tensors release];
      }
      if (target_tensors != nil) {
        [target_tensors release];
      }
      if (run_feed_tensors != nil) {
        [run_feed_tensors release];
      }
      if (feed_tensors != nil) {
        [feed_tensors release];
      }
      if (executable != nil) {
        [executable release];
      }
      if (graph != nil) {
        [graph release];
      }
    }
  };

  id<MTLDevice> device{nil};
  id<MTLCommandQueue> queue{nil};
  MPSGraphDevice* graph_device{nil};
  mutable MpsGraphTransferStats transfer_stats;
  mutable MpsGraphGraphStats graph_stats;
  mutable bool sdpa_attention_disabled{false};
  mutable std::unordered_map<std::string, std::unique_ptr<TransformerLayerExecutable>>
    transformer_layer_executables;

  ~Impl() {
    if (graph_device != nil) {
      [graph_device release];
    }
    if (queue != nil) {
      [queue release];
    }
    if (device != nil) {
      [device release];
    }
  }
};

MpsGraphBuffer::MpsGraphBuffer() = default;
MpsGraphBuffer::~MpsGraphBuffer() = default;
MpsGraphBuffer::MpsGraphBuffer(MpsGraphBuffer&& other) noexcept = default;
MpsGraphBuffer& MpsGraphBuffer::operator=(MpsGraphBuffer&& other) noexcept = default;
MpsGraphBuffer::MpsGraphBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool MpsGraphBuffer::valid() const {
  return impl_ != nullptr && impl_->buffer != nil;
}

std::size_t MpsGraphBuffer::byte_size() const {
  return impl_ == nullptr ? 0 : impl_->byte_size;
}

MpsGraphContext::MpsGraphContext() = default;
MpsGraphContext::~MpsGraphContext() = default;
MpsGraphContext::MpsGraphContext(MpsGraphContext&& other) noexcept = default;
MpsGraphContext& MpsGraphContext::operator=(MpsGraphContext&& other) noexcept = default;
MpsGraphContext::MpsGraphContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Result<MpsGraphContext> MpsGraphContext::create() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return Status::unavailable("Metal returned no default device");
    }

    MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:device];
    if (graph_device == nil) {
      [device release];
      return Status::unavailable("failed to create MPSGraphDevice from Metal device");
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      [device release];
      return Status::unavailable("failed to create Metal command queue for MPSGraph");
    }

    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->queue = queue;
    impl->graph_device = [graph_device retain];
    return MpsGraphContext(std::move(impl));
  }
}

bool MpsGraphContext::valid() const {
  return impl_ != nullptr && impl_->device != nil && impl_->queue != nil &&
         impl_->graph_device != nil;
}

MpsGraphTransferStats MpsGraphContext::transfer_stats() const {
  return impl_ == nullptr ? MpsGraphTransferStats{} : impl_->transfer_stats;
}

MpsGraphGraphStats MpsGraphContext::graph_stats() const {
  return impl_ == nullptr ? MpsGraphGraphStats{} : impl_->graph_stats;
}

Result<MpsGraphBuffer> MpsGraphContext::make_buffer(std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  if (byte_size == 0) {
    return Status::invalid_argument("MPSGraph buffer size must be greater than zero");
  }
  if (byte_size > static_cast<std::size_t>(std::numeric_limits<NSUInteger>::max())) {
    return Status::invalid_argument("MPSGraph buffer size exceeds NSUInteger range");
  }

  @autoreleasepool {
    id<MTLBuffer> buffer =
      [impl_->device newBufferWithLength:static_cast<NSUInteger>(byte_size)
                                 options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      return Status::unavailable("failed to allocate MPSGraph Metal buffer");
    }

    auto impl = std::make_unique<MpsGraphBuffer::Impl>();
    impl->buffer = buffer;
    impl->byte_size = byte_size;
    return MpsGraphBuffer(std::move(impl));
  }
}

Status MpsGraphContext::copy_to_buffer(MpsGraphBuffer& buffer, const void* data,
                                       std::size_t byte_size) const {
  return copy_to_buffer_at(buffer, 0, data, byte_size);
}

Status MpsGraphContext::copy_to_buffer_at(MpsGraphBuffer& buffer,
                                          std::size_t byte_offset,
                                          const void* data,
                                          std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  if (!buffer.valid()) {
    return Status::invalid_argument("MPSGraph buffer is not initialized");
  }
  if (data == nullptr) {
    return Status::invalid_argument("MPSGraph copy source must not be null");
  }
  if (byte_offset > buffer.byte_size() || byte_size > buffer.byte_size() - byte_offset) {
    return Status::invalid_argument("MPSGraph copy size exceeds destination buffer size");
  }

  auto* destination = static_cast<std::byte*>([buffer.impl_->buffer contents]);
  std::memcpy(destination + byte_offset, data, byte_size);
  ++impl_->transfer_stats.host_to_device_calls;
  impl_->transfer_stats.host_to_device_bytes += byte_size;
  return Status::ok();
}

Status MpsGraphContext::copy_from_buffer(const MpsGraphBuffer& buffer, void* data,
                                         std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  if (!buffer.valid()) {
    return Status::invalid_argument("MPSGraph buffer is not initialized");
  }
  if (data == nullptr) {
    return Status::invalid_argument("MPSGraph copy destination must not be null");
  }
  if (byte_size > buffer.byte_size()) {
    return Status::invalid_argument("MPSGraph copy size exceeds source buffer size");
  }

  std::memcpy(data, [buffer.impl_->buffer contents], byte_size);
  ++impl_->transfer_stats.device_to_host_calls;
  impl_->transfer_stats.device_to_host_bytes += byte_size;
  return Status::ok();
}

Status MpsGraphContext::embedding_f32(const MpsGraphBuffer& weight,
                                      std::size_t vocab_size,
                                      std::size_t hidden_size,
                                      std::int64_t token,
                                      MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  if (token < 0 || static_cast<std::size_t>(token) >= vocab_size) {
    return Status::invalid_argument("MPSGraph embedding token is out of range");
  }
  auto vocab_status = validate_positive_dim(vocab_size, "embedding vocab_size");
  if (!vocab_status.is_ok()) {
    return vocab_status;
  }
  auto hidden_status = validate_positive_dim(hidden_size, "embedding hidden_size");
  if (!hidden_status.is_ok()) {
    return hidden_status;
  }
  std::size_t weight_values = 0;
  if (!checked_mul(vocab_size, hidden_size, weight_values)) {
    return Status::invalid_argument("MPSGraph embedding weight element count overflow");
  }
  auto weight_status = validate_f32_buffer(weight, weight_values, "embedding weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }
  auto output_status = validate_f32_buffer(output, hidden_size, "embedding output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* weight_shape = make_shape({vocab_size, hidden_size});
    MPSShape* token_shape = make_shape({1});
    MPSShape* output_shape = make_shape({hidden_size});
    MPSGraphTensor* weight_tensor =
      [graph placeholderWithShape:weight_shape dataType:MPSDataTypeFloat32 name:nil];
    const std::int32_t token_index = static_cast<std::int32_t>(token);
    NSData* token_data = [NSData dataWithBytes:&token_index length:sizeof(token_index)];
    MPSGraphTensor* indices_tensor =
      [graph constantWithData:token_data shape:token_shape dataType:MPSDataTypeInt32];
    MPSGraphTensor* gathered = [graph gatherWithUpdatesTensor:weight_tensor
                                                indicesTensor:indices_tensor
                                                         axis:0
                                              batchDimensions:0
                                                         name:nil];
    MPSGraphTensor* result = [graph reshapeTensor:gathered withShape:output_shape name:nil];

    MPSGraphTensorData* weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:weight.impl_->buffer
                                             shape:weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:output_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      weight_tensor : weight_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [weight_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::embedding_from_token_f32(const MpsGraphBuffer& weight,
                                                std::size_t vocab_size,
                                                std::size_t hidden_size,
                                                const MpsGraphBuffer& token,
                                                MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto vocab_status = validate_positive_dim(vocab_size, "embedding vocab_size");
  if (!vocab_status.is_ok()) {
    return vocab_status;
  }
  auto hidden_status = validate_positive_dim(hidden_size, "embedding hidden_size");
  if (!hidden_status.is_ok()) {
    return hidden_status;
  }
  std::size_t weight_values = 0;
  if (!checked_mul(vocab_size, hidden_size, weight_values)) {
    return Status::invalid_argument("MPSGraph embedding weight element count overflow");
  }
  auto weight_status = validate_f32_buffer(weight, weight_values, "embedding weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }
  if (!token.valid()) {
    return Status::invalid_argument("MPSGraph embedding token buffer is not initialized");
  }
  if (token.byte_size() < sizeof(std::int32_t)) {
    return Status::invalid_argument("MPSGraph embedding token buffer is too small");
  }
  auto output_status = validate_f32_buffer(output, hidden_size, "embedding output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* weight_shape = make_shape({vocab_size, hidden_size});
    MPSShape* token_shape = make_shape({1});
    MPSShape* output_shape = make_shape({hidden_size});
    MPSGraphTensor* weight_tensor =
      [graph placeholderWithShape:weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* token_tensor =
      [graph placeholderWithShape:token_shape dataType:MPSDataTypeInt32 name:nil];
    MPSGraphTensor* gathered = [graph gatherWithUpdatesTensor:weight_tensor
                                                indicesTensor:token_tensor
                                                         axis:0
                                              batchDimensions:0
                                                         name:nil];
    MPSGraphTensor* result = [graph reshapeTensor:gathered withShape:output_shape name:nil];

    MPSGraphTensorData* weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:weight.impl_->buffer
                                             shape:weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* token_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:token.impl_->buffer
                                             shape:token_shape
                                          dataType:MPSDataTypeInt32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:output_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      weight_tensor : weight_data,
      token_tensor : token_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [weight_data release];
    [token_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::rms_norm_f32(const MpsGraphBuffer& input,
                                     const MpsGraphBuffer& weight,
                                     std::size_t size, float eps,
                                     MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto size_status = validate_positive_dim(size, "rms norm size");
  if (!size_status.is_ok()) {
    return size_status;
  }
  auto input_status = validate_f32_buffer(input, size, "rms norm input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto weight_status = validate_f32_buffer(weight, size, "rms norm weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }
  auto output_status = validate_f32_buffer(output, size, "rms norm output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* vector_shape = make_shape({size});
    MPSGraphTensor* input_tensor =
      [graph placeholderWithShape:vector_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* weight_tensor =
      [graph placeholderWithShape:vector_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* result = build_rms_norm(graph, input_tensor, weight_tensor, size, eps);

    MPSGraphTensorData* input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:input.impl_->buffer
                                             shape:vector_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:weight.impl_->buffer
                                             shape:vector_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:vector_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      input_tensor : input_data,
      weight_tensor : weight_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [input_data release];
    [weight_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::qk_norm_f32(const MpsGraphBuffer& input,
                                    const MpsGraphBuffer& weight,
                                    std::size_t heads, std::size_t head_dim,
                                    float eps,
                                    MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto heads_status = validate_positive_dim(heads, "qk norm heads");
  if (!heads_status.is_ok()) {
    return heads_status;
  }
  auto head_dim_status = validate_positive_dim(head_dim, "qk norm head_dim");
  if (!head_dim_status.is_ok()) {
    return head_dim_status;
  }
  if (!std::isfinite(eps)) {
    return Status::invalid_argument("MPSGraph qk norm eps must be finite");
  }
  std::size_t values = 0;
  if (!checked_mul(heads, head_dim, values)) {
    return Status::invalid_argument("MPSGraph qk norm element count overflow");
  }
  auto input_status = validate_f32_buffer(input, values, "qk norm input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto weight_status = validate_f32_buffer(weight, head_dim, "qk norm weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }
  auto output_status = validate_f32_buffer(output, values, "qk norm output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* matrix_shape = make_shape({heads, head_dim});
    MPSShape* vector_shape = make_shape({head_dim});
    MPSGraphTensor* input_tensor =
      [graph placeholderWithShape:matrix_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* weight_tensor =
      [graph placeholderWithShape:vector_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* result =
      build_qk_norm(graph, input_tensor, weight_tensor, heads, head_dim, eps);

    MPSGraphTensorData* input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:input.impl_->buffer
                                             shape:matrix_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:weight.impl_->buffer
                                             shape:vector_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:matrix_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      input_tensor : input_data,
      weight_tensor : weight_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [input_data release];
    [weight_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::rope_f32(const MpsGraphBuffer& input,
                                 std::size_t heads, std::size_t head_dim,
                                 std::size_t position, float theta,
                                 MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto heads_status = validate_positive_dim(heads, "RoPE heads");
  if (!heads_status.is_ok()) {
    return heads_status;
  }
  auto head_dim_status = validate_positive_dim(head_dim, "RoPE head_dim");
  if (!head_dim_status.is_ok()) {
    return head_dim_status;
  }
  if (head_dim % 2U != 0) {
    return Status::invalid_argument("MPSGraph RoPE head_dim must be even");
  }
  if (!std::isfinite(theta) || theta <= 0.0F) {
    return Status::invalid_argument("MPSGraph RoPE theta must be positive and finite");
  }
  std::size_t values = 0;
  if (!checked_mul(heads, head_dim, values)) {
    return Status::invalid_argument("MPSGraph RoPE element count overflow");
  }
  auto input_status = validate_f32_buffer(input, values, "RoPE input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, values, "RoPE output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  const auto half_dim = head_dim / 2U;
  auto trig_bytes = f32_bytes(half_dim, "RoPE trig constants");
  if (!trig_bytes.is_ok()) {
    return trig_bytes.status();
  }

  std::vector<float> cos_values(half_dim);
  std::vector<float> sin_values(half_dim);
  for (std::size_t dim = 0; dim < half_dim; ++dim) {
    const auto exponent =
      static_cast<double>(2U * dim) / static_cast<double>(head_dim);
    const auto frequency = 1.0 / std::pow(static_cast<double>(theta), exponent);
    const auto angle = static_cast<double>(position) * frequency;
    cos_values[dim] = static_cast<float>(std::cos(angle));
    sin_values[dim] = static_cast<float>(std::sin(angle));
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* matrix_shape = make_shape({heads, head_dim});
    MPSGraphTensor* input_tensor =
      [graph placeholderWithShape:matrix_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* result =
      build_rope(graph, input_tensor, heads, head_dim, cos_values, sin_values,
                 trig_bytes.value());

    MPSGraphTensorData* input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:input.impl_->buffer
                                             shape:matrix_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:matrix_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      input_tensor : input_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [input_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::qk_norm_rope_f32(const MpsGraphBuffer& q_input,
                                         const MpsGraphBuffer& k_input,
                                         const MpsGraphBuffer& q_weight,
                                         const MpsGraphBuffer& k_weight,
                                         std::size_t q_heads,
                                         std::size_t kv_heads,
                                         std::size_t head_dim,
                                         std::size_t position,
                                         float eps, float theta,
                                         MpsGraphBuffer& q_output,
                                         MpsGraphBuffer& k_output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto q_heads_status = validate_positive_dim(q_heads, "qk norm rope q_heads");
  if (!q_heads_status.is_ok()) {
    return q_heads_status;
  }
  auto kv_heads_status = validate_positive_dim(kv_heads, "qk norm rope kv_heads");
  if (!kv_heads_status.is_ok()) {
    return kv_heads_status;
  }
  auto head_dim_status = validate_positive_dim(head_dim, "qk norm rope head_dim");
  if (!head_dim_status.is_ok()) {
    return head_dim_status;
  }
  if (head_dim % 2U != 0) {
    return Status::invalid_argument("MPSGraph qk norm rope head_dim must be even");
  }
  if (!std::isfinite(eps)) {
    return Status::invalid_argument("MPSGraph qk norm rope eps must be finite");
  }
  if (!std::isfinite(theta) || theta <= 0.0F) {
    return Status::invalid_argument("MPSGraph qk norm rope theta must be positive and finite");
  }
  std::size_t q_values = 0;
  if (!checked_mul(q_heads, head_dim, q_values)) {
    return Status::invalid_argument("MPSGraph qk norm rope q element count overflow");
  }
  std::size_t k_values = 0;
  if (!checked_mul(kv_heads, head_dim, k_values)) {
    return Status::invalid_argument("MPSGraph qk norm rope k element count overflow");
  }
  auto q_input_status = validate_f32_buffer(q_input, q_values, "qk norm rope q input");
  if (!q_input_status.is_ok()) {
    return q_input_status;
  }
  auto k_input_status = validate_f32_buffer(k_input, k_values, "qk norm rope k input");
  if (!k_input_status.is_ok()) {
    return k_input_status;
  }
  auto q_weight_status = validate_f32_buffer(q_weight, head_dim,
                                             "qk norm rope q weight");
  if (!q_weight_status.is_ok()) {
    return q_weight_status;
  }
  auto k_weight_status = validate_f32_buffer(k_weight, head_dim,
                                             "qk norm rope k weight");
  if (!k_weight_status.is_ok()) {
    return k_weight_status;
  }
  auto q_output_status = validate_f32_buffer(q_output, q_values,
                                             "qk norm rope q output");
  if (!q_output_status.is_ok()) {
    return q_output_status;
  }
  auto k_output_status = validate_f32_buffer(k_output, k_values,
                                             "qk norm rope k output");
  if (!k_output_status.is_ok()) {
    return k_output_status;
  }

  const auto half_dim = head_dim / 2U;
  auto trig_bytes = f32_bytes(half_dim, "qk norm rope trig constants");
  if (!trig_bytes.is_ok()) {
    return trig_bytes.status();
  }
  std::vector<float> cos_values(half_dim);
  std::vector<float> sin_values(half_dim);
  for (std::size_t dim = 0; dim < half_dim; ++dim) {
    const auto exponent =
      static_cast<double>(2U * dim) / static_cast<double>(head_dim);
    const auto frequency = 1.0 / std::pow(static_cast<double>(theta), exponent);
    const auto angle = static_cast<double>(position) * frequency;
    cos_values[dim] = static_cast<float>(std::cos(angle));
    sin_values[dim] = static_cast<float>(std::sin(angle));
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* q_shape = make_shape({q_heads, head_dim});
    MPSShape* k_shape = make_shape({kv_heads, head_dim});
    MPSShape* weight_shape = make_shape({head_dim});
    MPSGraphTensor* q_input_tensor =
      [graph placeholderWithShape:q_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* k_input_tensor =
      [graph placeholderWithShape:k_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* q_weight_tensor =
      [graph placeholderWithShape:weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* k_weight_tensor =
      [graph placeholderWithShape:weight_shape dataType:MPSDataTypeFloat32 name:nil];

    MPSGraphTensor* q_normed =
      build_qk_norm(graph, q_input_tensor, q_weight_tensor, q_heads, head_dim, eps);
    MPSGraphTensor* k_normed =
      build_qk_norm(graph, k_input_tensor, k_weight_tensor, kv_heads, head_dim, eps);
    MPSGraphTensor* q_result =
      build_rope(graph, q_normed, q_heads, head_dim, cos_values, sin_values,
                 trig_bytes.value());
    MPSGraphTensor* k_result =
      build_rope(graph, k_normed, kv_heads, head_dim, cos_values, sin_values,
                 trig_bytes.value());

    MPSGraphTensorData* q_input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:q_input.impl_->buffer
                                             shape:q_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* k_input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:k_input.impl_->buffer
                                             shape:k_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* q_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:q_weight.impl_->buffer
                                             shape:weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* k_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:k_weight.impl_->buffer
                                             shape:weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* q_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:q_output.impl_->buffer
                                             shape:q_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* k_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:k_output.impl_->buffer
                                             shape:k_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      q_input_tensor : q_input_data,
      k_input_tensor : k_input_data,
      q_weight_tensor : q_weight_data,
      k_weight_tensor : k_weight_data,
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      q_result : q_output_data,
      k_result : k_output_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, results, &impl_->graph_stats);
    [q_input_data release];
    [k_input_data release];
    [q_weight_data release];
    [k_weight_data release];
    [q_output_data release];
    [k_output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::matvec_f32(const MpsGraphBuffer& weight,
                                   std::size_t rows, std::size_t cols,
                                   const MpsGraphBuffer& input,
                                   MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto rows_status = validate_positive_dim(rows, "matvec rows");
  if (!rows_status.is_ok()) {
    return rows_status;
  }
  auto cols_status = validate_positive_dim(cols, "matvec cols");
  if (!cols_status.is_ok()) {
    return cols_status;
  }
  std::size_t weight_values = 0;
  if (!checked_mul(rows, cols, weight_values)) {
    return Status::invalid_argument("MPSGraph matvec weight element count overflow");
  }
  auto weight_status = validate_f32_buffer(weight, weight_values, "matvec weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }
  auto input_status = validate_f32_buffer(input, cols, "matvec input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, rows, "matvec output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* weight_shape = make_shape({rows, cols});
    MPSShape* input_shape = make_shape({cols, 1});
    MPSShape* output_shape = make_shape({rows});
    MPSGraphTensor* weight_tensor =
      [graph placeholderWithShape:weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* input_tensor =
      [graph placeholderWithShape:input_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* product =
      [graph matrixMultiplicationWithPrimaryTensor:weight_tensor
                                   secondaryTensor:input_tensor
                                              name:nil];
    MPSGraphTensor* result = [graph reshapeTensor:product withShape:output_shape name:nil];

    MPSGraphTensorData* weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:weight.impl_->buffer
                                             shape:weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:input.impl_->buffer
                                             shape:input_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:output_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      weight_tensor : weight_data,
      input_tensor : input_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [weight_data release];
    [input_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::qkv_matvec_f32(const MpsGraphBuffer& q_weight,
                                       const MpsGraphBuffer& k_weight,
                                       const MpsGraphBuffer& v_weight,
                                       std::size_t q_rows, std::size_t kv_rows,
                                       std::size_t cols,
                                       const MpsGraphBuffer& input,
                                       MpsGraphBuffer& q_output,
                                       MpsGraphBuffer& k_output,
                                       MpsGraphBuffer& v_output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto q_rows_status = validate_positive_dim(q_rows, "qkv matvec q_rows");
  if (!q_rows_status.is_ok()) {
    return q_rows_status;
  }
  auto kv_rows_status = validate_positive_dim(kv_rows, "qkv matvec kv_rows");
  if (!kv_rows_status.is_ok()) {
    return kv_rows_status;
  }
  auto cols_status = validate_positive_dim(cols, "qkv matvec cols");
  if (!cols_status.is_ok()) {
    return cols_status;
  }
  std::size_t q_weight_values = 0;
  if (!checked_mul(q_rows, cols, q_weight_values)) {
    return Status::invalid_argument("MPSGraph qkv matvec q weight element count overflow");
  }
  std::size_t kv_weight_values = 0;
  if (!checked_mul(kv_rows, cols, kv_weight_values)) {
    return Status::invalid_argument("MPSGraph qkv matvec KV weight element count overflow");
  }
  auto q_weight_status = validate_f32_buffer(q_weight, q_weight_values,
                                             "qkv matvec q weight");
  if (!q_weight_status.is_ok()) {
    return q_weight_status;
  }
  auto k_weight_status = validate_f32_buffer(k_weight, kv_weight_values,
                                             "qkv matvec k weight");
  if (!k_weight_status.is_ok()) {
    return k_weight_status;
  }
  auto v_weight_status = validate_f32_buffer(v_weight, kv_weight_values,
                                             "qkv matvec v weight");
  if (!v_weight_status.is_ok()) {
    return v_weight_status;
  }
  auto input_status = validate_f32_buffer(input, cols, "qkv matvec input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto q_output_status = validate_f32_buffer(q_output, q_rows, "qkv matvec q output");
  if (!q_output_status.is_ok()) {
    return q_output_status;
  }
  auto k_output_status = validate_f32_buffer(k_output, kv_rows, "qkv matvec k output");
  if (!k_output_status.is_ok()) {
    return k_output_status;
  }
  auto v_output_status = validate_f32_buffer(v_output, kv_rows, "qkv matvec v output");
  if (!v_output_status.is_ok()) {
    return v_output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* q_weight_shape = make_shape({q_rows, cols});
    MPSShape* kv_weight_shape = make_shape({kv_rows, cols});
    MPSShape* input_shape = make_shape({cols, 1});
    MPSShape* q_output_shape = make_shape({q_rows});
    MPSShape* kv_output_shape = make_shape({kv_rows});
    MPSGraphTensor* q_weight_tensor =
      [graph placeholderWithShape:q_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* k_weight_tensor =
      [graph placeholderWithShape:kv_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* v_weight_tensor =
      [graph placeholderWithShape:kv_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* input_tensor =
      [graph placeholderWithShape:input_shape dataType:MPSDataTypeFloat32 name:nil];

    MPSGraphTensor* q_product =
      [graph matrixMultiplicationWithPrimaryTensor:q_weight_tensor
                                   secondaryTensor:input_tensor
                                              name:nil];
    MPSGraphTensor* k_product =
      [graph matrixMultiplicationWithPrimaryTensor:k_weight_tensor
                                   secondaryTensor:input_tensor
                                              name:nil];
    MPSGraphTensor* v_product =
      [graph matrixMultiplicationWithPrimaryTensor:v_weight_tensor
                                   secondaryTensor:input_tensor
                                              name:nil];
    MPSGraphTensor* q_result =
      [graph reshapeTensor:q_product withShape:q_output_shape name:nil];
    MPSGraphTensor* k_result =
      [graph reshapeTensor:k_product withShape:kv_output_shape name:nil];
    MPSGraphTensor* v_result =
      [graph reshapeTensor:v_product withShape:kv_output_shape name:nil];

    MPSGraphTensorData* q_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:q_weight.impl_->buffer
                                             shape:q_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* k_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:k_weight.impl_->buffer
                                             shape:kv_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* v_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:v_weight.impl_->buffer
                                             shape:kv_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:input.impl_->buffer
                                             shape:input_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* q_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:q_output.impl_->buffer
                                             shape:q_output_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* k_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:k_output.impl_->buffer
                                             shape:kv_output_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* v_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:v_output.impl_->buffer
                                             shape:kv_output_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      q_weight_tensor : q_weight_data,
      k_weight_tensor : k_weight_data,
      v_weight_tensor : v_weight_data,
      input_tensor : input_data,
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      q_result : q_output_data,
      k_result : k_output_data,
      v_result : v_output_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, results, &impl_->graph_stats);
    [q_weight_data release];
    [k_weight_data release];
    [v_weight_data release];
    [input_data release];
    [q_output_data release];
    [k_output_data release];
    [v_output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::input_norm_qkv_qk_rope_f32(
  const MpsGraphBuffer& hidden, const MpsGraphBuffer& input_norm_weight,
  const MpsGraphBuffer& q_weight, const MpsGraphBuffer& k_weight,
  const MpsGraphBuffer& v_weight, const MpsGraphBuffer& q_norm_weight,
  const MpsGraphBuffer& k_norm_weight, std::size_t hidden_size,
  std::size_t q_heads, std::size_t kv_heads, std::size_t head_dim,
  std::size_t position, float eps, float theta, MpsGraphBuffer& normed_output,
  MpsGraphBuffer& q_output, MpsGraphBuffer& k_output, MpsGraphBuffer& v_output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto hidden_status = validate_positive_dim(hidden_size, "input/qkv hidden_size");
  if (!hidden_status.is_ok()) {
    return hidden_status;
  }
  auto q_heads_status = validate_positive_dim(q_heads, "input/qkv q_heads");
  if (!q_heads_status.is_ok()) {
    return q_heads_status;
  }
  auto kv_heads_status = validate_positive_dim(kv_heads, "input/qkv kv_heads");
  if (!kv_heads_status.is_ok()) {
    return kv_heads_status;
  }
  auto head_dim_status = validate_positive_dim(head_dim, "input/qkv head_dim");
  if (!head_dim_status.is_ok()) {
    return head_dim_status;
  }
  if (head_dim % 2U != 0) {
    return Status::invalid_argument("MPSGraph input/qkv head_dim must be even");
  }
  if (!std::isfinite(eps)) {
    return Status::invalid_argument("MPSGraph input/qkv eps must be finite");
  }
  if (!std::isfinite(theta) || theta <= 0.0F) {
    return Status::invalid_argument("MPSGraph input/qkv theta must be positive and finite");
  }
  std::size_t q_dim = 0;
  if (!checked_mul(q_heads, head_dim, q_dim)) {
    return Status::invalid_argument("MPSGraph input/qkv q dimension overflow");
  }
  std::size_t kv_dim = 0;
  if (!checked_mul(kv_heads, head_dim, kv_dim)) {
    return Status::invalid_argument("MPSGraph input/qkv KV dimension overflow");
  }
  std::size_t q_weight_values = 0;
  if (!checked_mul(q_dim, hidden_size, q_weight_values)) {
    return Status::invalid_argument("MPSGraph input/qkv q weight count overflow");
  }
  std::size_t kv_weight_values = 0;
  if (!checked_mul(kv_dim, hidden_size, kv_weight_values)) {
    return Status::invalid_argument("MPSGraph input/qkv KV weight count overflow");
  }
  auto hidden_buffer_status = validate_f32_buffer(hidden, hidden_size, "input/qkv hidden");
  if (!hidden_buffer_status.is_ok()) {
    return hidden_buffer_status;
  }
  auto input_norm_status = validate_f32_buffer(input_norm_weight, hidden_size,
                                               "input/qkv input norm weight");
  if (!input_norm_status.is_ok()) {
    return input_norm_status;
  }
  auto q_weight_status = validate_f32_buffer(q_weight, q_weight_values,
                                             "input/qkv q weight");
  if (!q_weight_status.is_ok()) {
    return q_weight_status;
  }
  auto k_weight_status = validate_f32_buffer(k_weight, kv_weight_values,
                                             "input/qkv k weight");
  if (!k_weight_status.is_ok()) {
    return k_weight_status;
  }
  auto v_weight_status = validate_f32_buffer(v_weight, kv_weight_values,
                                             "input/qkv v weight");
  if (!v_weight_status.is_ok()) {
    return v_weight_status;
  }
  auto q_norm_status = validate_f32_buffer(q_norm_weight, head_dim,
                                           "input/qkv q norm weight");
  if (!q_norm_status.is_ok()) {
    return q_norm_status;
  }
  auto k_norm_status = validate_f32_buffer(k_norm_weight, head_dim,
                                           "input/qkv k norm weight");
  if (!k_norm_status.is_ok()) {
    return k_norm_status;
  }
  auto normed_output_status = validate_f32_buffer(normed_output, hidden_size,
                                                  "input/qkv normed output");
  if (!normed_output_status.is_ok()) {
    return normed_output_status;
  }
  auto q_output_status = validate_f32_buffer(q_output, q_dim, "input/qkv q output");
  if (!q_output_status.is_ok()) {
    return q_output_status;
  }
  auto k_output_status = validate_f32_buffer(k_output, kv_dim, "input/qkv k output");
  if (!k_output_status.is_ok()) {
    return k_output_status;
  }
  auto v_output_status = validate_f32_buffer(v_output, kv_dim, "input/qkv v output");
  if (!v_output_status.is_ok()) {
    return v_output_status;
  }

  const auto half_dim = head_dim / 2U;
  auto trig_bytes = f32_bytes(half_dim, "input/qkv trig constants");
  if (!trig_bytes.is_ok()) {
    return trig_bytes.status();
  }
  std::vector<float> cos_values(half_dim);
  std::vector<float> sin_values(half_dim);
  for (std::size_t dim = 0; dim < half_dim; ++dim) {
    const auto exponent =
      static_cast<double>(2U * dim) / static_cast<double>(head_dim);
    const auto frequency = 1.0 / std::pow(static_cast<double>(theta), exponent);
    const auto angle = static_cast<double>(position) * frequency;
    cos_values[dim] = static_cast<float>(std::cos(angle));
    sin_values[dim] = static_cast<float>(std::sin(angle));
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* hidden_shape = make_shape({hidden_size});
    MPSShape* hidden_column_shape = make_shape({hidden_size, 1});
    MPSShape* q_weight_shape = make_shape({q_dim, hidden_size});
    MPSShape* kv_weight_shape = make_shape({kv_dim, hidden_size});
    MPSShape* q_shape = make_shape({q_heads, head_dim});
    MPSShape* q_vector_shape = make_shape({q_dim});
    MPSShape* k_shape = make_shape({kv_heads, head_dim});
    MPSShape* kv_vector_shape = make_shape({kv_dim});
    MPSShape* norm_weight_shape = make_shape({head_dim});

    MPSGraphTensor* hidden_tensor =
      [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* input_norm_tensor =
      [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* q_weight_tensor =
      [graph placeholderWithShape:q_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* k_weight_tensor =
      [graph placeholderWithShape:kv_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* v_weight_tensor =
      [graph placeholderWithShape:kv_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* q_norm_tensor =
      [graph placeholderWithShape:norm_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* k_norm_tensor =
      [graph placeholderWithShape:norm_weight_shape dataType:MPSDataTypeFloat32 name:nil];

    MPSGraphTensor* normed =
      build_rms_norm(graph, hidden_tensor, input_norm_tensor, hidden_size, eps);
    MPSGraphTensor* normed_column =
      [graph reshapeTensor:normed withShape:hidden_column_shape name:nil];
    MPSGraphTensor* q_projected =
      [graph matrixMultiplicationWithPrimaryTensor:q_weight_tensor
                                   secondaryTensor:normed_column
                                              name:nil];
    MPSGraphTensor* k_projected =
      [graph matrixMultiplicationWithPrimaryTensor:k_weight_tensor
                                   secondaryTensor:normed_column
                                              name:nil];
    MPSGraphTensor* v_projected =
      [graph matrixMultiplicationWithPrimaryTensor:v_weight_tensor
                                   secondaryTensor:normed_column
                                              name:nil];
    MPSGraphTensor* q_matrix =
      [graph reshapeTensor:q_projected withShape:q_shape name:nil];
    MPSGraphTensor* k_matrix =
      [graph reshapeTensor:k_projected withShape:k_shape name:nil];
    MPSGraphTensor* v_result =
      [graph reshapeTensor:v_projected withShape:kv_vector_shape name:nil];
    MPSGraphTensor* q_normed =
      build_qk_norm(graph, q_matrix, q_norm_tensor, q_heads, head_dim, eps);
    MPSGraphTensor* k_normed =
      build_qk_norm(graph, k_matrix, k_norm_tensor, kv_heads, head_dim, eps);
    MPSGraphTensor* q_rope =
      build_rope(graph, q_normed, q_heads, head_dim, cos_values, sin_values,
                 trig_bytes.value());
    MPSGraphTensor* k_rope =
      build_rope(graph, k_normed, kv_heads, head_dim, cos_values, sin_values,
                 trig_bytes.value());
    MPSGraphTensor* q_result =
      [graph reshapeTensor:q_rope withShape:q_vector_shape name:nil];
    MPSGraphTensor* k_result =
      [graph reshapeTensor:k_rope withShape:kv_vector_shape name:nil];

    MPSGraphTensorData* hidden_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:hidden.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* input_norm_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:input_norm_weight.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* q_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:q_weight.impl_->buffer
                                             shape:q_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* k_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:k_weight.impl_->buffer
                                             shape:kv_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* v_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:v_weight.impl_->buffer
                                             shape:kv_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* q_norm_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:q_norm_weight.impl_->buffer
                                             shape:norm_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* k_norm_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:k_norm_weight.impl_->buffer
                                             shape:norm_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* normed_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:normed_output.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* q_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:q_output.impl_->buffer
                                             shape:q_vector_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* k_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:k_output.impl_->buffer
                                             shape:kv_vector_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* v_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:v_output.impl_->buffer
                                             shape:kv_vector_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      hidden_tensor : hidden_data,
      input_norm_tensor : input_norm_data,
      q_weight_tensor : q_weight_data,
      k_weight_tensor : k_weight_data,
      v_weight_tensor : v_weight_data,
      q_norm_tensor : q_norm_data,
      k_norm_tensor : k_norm_data,
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      normed : normed_output_data,
      q_result : q_output_data,
      k_result : k_output_data,
      v_result : v_output_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, results, &impl_->graph_stats);
    [hidden_data release];
    [input_norm_data release];
    [q_weight_data release];
    [k_weight_data release];
    [v_weight_data release];
    [q_norm_data release];
    [k_norm_data release];
    [normed_output_data release];
    [q_output_data release];
    [k_output_data release];
    [v_output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::gate_up_matvec_f32(const MpsGraphBuffer& gate_weight,
                                           const MpsGraphBuffer& up_weight,
                                           std::size_t rows, std::size_t cols,
                                           const MpsGraphBuffer& input,
                                           MpsGraphBuffer& gate_output,
                                           MpsGraphBuffer& up_output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto rows_status = validate_positive_dim(rows, "gate/up matvec rows");
  if (!rows_status.is_ok()) {
    return rows_status;
  }
  auto cols_status = validate_positive_dim(cols, "gate/up matvec cols");
  if (!cols_status.is_ok()) {
    return cols_status;
  }
  std::size_t weight_values = 0;
  if (!checked_mul(rows, cols, weight_values)) {
    return Status::invalid_argument("MPSGraph gate/up matvec weight element count overflow");
  }
  auto gate_weight_status = validate_f32_buffer(gate_weight, weight_values,
                                                "gate/up matvec gate weight");
  if (!gate_weight_status.is_ok()) {
    return gate_weight_status;
  }
  auto up_weight_status = validate_f32_buffer(up_weight, weight_values,
                                              "gate/up matvec up weight");
  if (!up_weight_status.is_ok()) {
    return up_weight_status;
  }
  auto input_status = validate_f32_buffer(input, cols, "gate/up matvec input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto gate_output_status = validate_f32_buffer(gate_output, rows,
                                                "gate/up matvec gate output");
  if (!gate_output_status.is_ok()) {
    return gate_output_status;
  }
  auto up_output_status = validate_f32_buffer(up_output, rows,
                                              "gate/up matvec up output");
  if (!up_output_status.is_ok()) {
    return up_output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* weight_shape = make_shape({rows, cols});
    MPSShape* input_shape = make_shape({cols, 1});
    MPSShape* output_shape = make_shape({rows});
    MPSGraphTensor* gate_weight_tensor =
      [graph placeholderWithShape:weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* up_weight_tensor =
      [graph placeholderWithShape:weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* input_tensor =
      [graph placeholderWithShape:input_shape dataType:MPSDataTypeFloat32 name:nil];

    MPSGraphTensor* gate_product =
      [graph matrixMultiplicationWithPrimaryTensor:gate_weight_tensor
                                   secondaryTensor:input_tensor
                                              name:nil];
    MPSGraphTensor* up_product =
      [graph matrixMultiplicationWithPrimaryTensor:up_weight_tensor
                                   secondaryTensor:input_tensor
                                              name:nil];
    MPSGraphTensor* gate_result =
      [graph reshapeTensor:gate_product withShape:output_shape name:nil];
    MPSGraphTensor* up_result =
      [graph reshapeTensor:up_product withShape:output_shape name:nil];

    MPSGraphTensorData* gate_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:gate_weight.impl_->buffer
                                             shape:weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* up_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:up_weight.impl_->buffer
                                             shape:weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:input.impl_->buffer
                                             shape:input_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* gate_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:gate_output.impl_->buffer
                                             shape:output_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* up_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:up_output.impl_->buffer
                                             shape:output_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      gate_weight_tensor : gate_weight_data,
      up_weight_tensor : up_weight_data,
      input_tensor : input_data,
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      gate_result : gate_output_data,
      up_result : up_output_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, results, &impl_->graph_stats);
    [gate_weight_data release];
    [up_weight_data release];
    [input_data release];
    [gate_output_data release];
    [up_output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::attn_project_residual_norm_f32(
  const MpsGraphBuffer& o_weight, const MpsGraphBuffer& attn_output,
  const MpsGraphBuffer& residual, const MpsGraphBuffer& norm_weight,
  std::size_t hidden_size, std::size_t attn_dim, float eps,
  MpsGraphBuffer& residual_output, MpsGraphBuffer& norm_output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto hidden_status =
    validate_positive_dim(hidden_size, "attention projection hidden_size");
  if (!hidden_status.is_ok()) {
    return hidden_status;
  }
  auto attn_status = validate_positive_dim(attn_dim, "attention projection attn_dim");
  if (!attn_status.is_ok()) {
    return attn_status;
  }
  if (!std::isfinite(eps)) {
    return Status::invalid_argument("MPSGraph attention projection norm eps must be finite");
  }
  std::size_t weight_values = 0;
  if (!checked_mul(hidden_size, attn_dim, weight_values)) {
    return Status::invalid_argument("MPSGraph attention projection weight count overflow");
  }
  auto weight_status = validate_f32_buffer(o_weight, weight_values,
                                           "attention projection weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }
  auto attn_output_status = validate_f32_buffer(attn_output, attn_dim,
                                                "attention projection input");
  if (!attn_output_status.is_ok()) {
    return attn_output_status;
  }
  auto residual_status = validate_f32_buffer(residual, hidden_size,
                                             "attention projection residual");
  if (!residual_status.is_ok()) {
    return residual_status;
  }
  auto norm_weight_status = validate_f32_buffer(norm_weight, hidden_size,
                                                "attention projection norm weight");
  if (!norm_weight_status.is_ok()) {
    return norm_weight_status;
  }
  auto residual_output_status = validate_f32_buffer(
    residual_output, hidden_size, "attention projection residual output");
  if (!residual_output_status.is_ok()) {
    return residual_output_status;
  }
  auto norm_output_status = validate_f32_buffer(
    norm_output, hidden_size, "attention projection norm output");
  if (!norm_output_status.is_ok()) {
    return norm_output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* weight_shape = make_shape({hidden_size, attn_dim});
    MPSShape* attn_shape = make_shape({attn_dim});
    MPSShape* attn_column_shape = make_shape({attn_dim, 1});
    MPSShape* hidden_shape = make_shape({hidden_size});
    MPSGraphTensor* weight_tensor =
      [graph placeholderWithShape:weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* attn_tensor =
      [graph placeholderWithShape:attn_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* residual_tensor =
      [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* norm_weight_tensor =
      [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];

    MPSGraphTensor* attn_column =
      [graph reshapeTensor:attn_tensor withShape:attn_column_shape name:nil];
    MPSGraphTensor* projected =
      [graph matrixMultiplicationWithPrimaryTensor:weight_tensor
                                   secondaryTensor:attn_column
                                              name:nil];
    MPSGraphTensor* projected_vector =
      [graph reshapeTensor:projected withShape:hidden_shape name:nil];
    MPSGraphTensor* residual_result =
      [graph additionWithPrimaryTensor:residual_tensor
                        secondaryTensor:projected_vector
                                   name:nil];

    MPSGraphTensor* squared = [graph squareWithTensor:residual_result name:nil];
    MPSGraphTensor* sum = [graph reductionSumWithTensor:squared axis:0 name:nil];
    MPSGraphTensor* denom =
      [graph constantWithScalar:static_cast<double>(hidden_size)
                          shape:@[ @1 ]
                       dataType:MPSDataTypeFloat32];
    MPSGraphTensor* mean =
      [graph divisionWithPrimaryTensor:sum secondaryTensor:denom name:nil];
    MPSGraphTensor* eps_tensor =
      [graph constantWithScalar:static_cast<double>(eps)
                          shape:@[ @1 ]
                       dataType:MPSDataTypeFloat32];
    MPSGraphTensor* variance =
      [graph additionWithPrimaryTensor:mean secondaryTensor:eps_tensor name:nil];
    MPSGraphTensor* scale = [graph reciprocalSquareRootWithTensor:variance name:nil];
    MPSGraphTensor* normalized =
      [graph multiplicationWithPrimaryTensor:residual_result secondaryTensor:scale name:nil];
    MPSGraphTensor* norm_result =
      [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:norm_weight_tensor
                                        name:nil];

    MPSGraphTensorData* weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:o_weight.impl_->buffer
                                             shape:weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* attn_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:attn_output.impl_->buffer
                                             shape:attn_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* residual_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:residual.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* norm_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:norm_weight.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* residual_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:residual_output.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* norm_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:norm_output.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      weight_tensor : weight_data,
      attn_tensor : attn_data,
      residual_tensor : residual_data,
      norm_weight_tensor : norm_weight_data,
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      residual_result : residual_output_data,
      norm_result : norm_output_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, results, &impl_->graph_stats);
    [weight_data release];
    [attn_data release];
    [residual_data release];
    [norm_weight_data release];
    [residual_output_data release];
    [norm_output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::silu_mul_f32(const MpsGraphBuffer& gate,
                                     const MpsGraphBuffer& up,
                                     std::size_t size,
                                     MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto size_status = validate_positive_dim(size, "silu size");
  if (!size_status.is_ok()) {
    return size_status;
  }
  auto gate_status = validate_f32_buffer(gate, size, "silu gate");
  if (!gate_status.is_ok()) {
    return gate_status;
  }
  auto up_status = validate_f32_buffer(up, size, "silu up");
  if (!up_status.is_ok()) {
    return up_status;
  }
  auto output_status = validate_f32_buffer(output, size, "silu output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* shape = make_shape({size});
    MPSGraphTensor* gate_tensor =
      [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* up_tensor =
      [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* sigmoid = [graph sigmoidWithTensor:gate_tensor name:nil];
    MPSGraphTensor* silu =
      [graph multiplicationWithPrimaryTensor:gate_tensor secondaryTensor:sigmoid name:nil];
    MPSGraphTensor* result =
      [graph multiplicationWithPrimaryTensor:silu secondaryTensor:up_tensor name:nil];

    MPSGraphTensorData* gate_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:gate.impl_->buffer
                                             shape:shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* up_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:up.impl_->buffer
                                             shape:shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      gate_tensor : gate_data,
      up_tensor : up_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [gate_data release];
    [up_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::swiglu_down_residual_f32(const MpsGraphBuffer& gate,
                                                 const MpsGraphBuffer& up,
                                                 const MpsGraphBuffer& down_weight,
                                                 const MpsGraphBuffer& residual,
                                                 std::size_t hidden_size,
                                                 std::size_t intermediate_size,
                                                 MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto hidden_status = validate_positive_dim(hidden_size, "SwiGLU hidden_size");
  if (!hidden_status.is_ok()) {
    return hidden_status;
  }
  auto intermediate_status =
    validate_positive_dim(intermediate_size, "SwiGLU intermediate_size");
  if (!intermediate_status.is_ok()) {
    return intermediate_status;
  }
  std::size_t weight_values = 0;
  if (!checked_mul(hidden_size, intermediate_size, weight_values)) {
    return Status::invalid_argument("MPSGraph SwiGLU down weight element count overflow");
  }
  auto gate_status = validate_f32_buffer(gate, intermediate_size, "SwiGLU gate");
  if (!gate_status.is_ok()) {
    return gate_status;
  }
  auto up_status = validate_f32_buffer(up, intermediate_size, "SwiGLU up");
  if (!up_status.is_ok()) {
    return up_status;
  }
  auto weight_status = validate_f32_buffer(down_weight, weight_values,
                                           "SwiGLU down weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }
  auto residual_status = validate_f32_buffer(residual, hidden_size,
                                             "SwiGLU residual");
  if (!residual_status.is_ok()) {
    return residual_status;
  }
  auto output_status = validate_f32_buffer(output, hidden_size, "SwiGLU output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* intermediate_shape = make_shape({intermediate_size});
    MPSShape* intermediate_column_shape = make_shape({intermediate_size, 1});
    MPSShape* weight_shape = make_shape({hidden_size, intermediate_size});
    MPSShape* hidden_shape = make_shape({hidden_size});
    MPSGraphTensor* gate_tensor =
      [graph placeholderWithShape:intermediate_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* up_tensor =
      [graph placeholderWithShape:intermediate_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* weight_tensor =
      [graph placeholderWithShape:weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* residual_tensor =
      [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];

    MPSGraphTensor* sigmoid = [graph sigmoidWithTensor:gate_tensor name:nil];
    MPSGraphTensor* silu =
      [graph multiplicationWithPrimaryTensor:gate_tensor secondaryTensor:sigmoid name:nil];
    MPSGraphTensor* gated =
      [graph multiplicationWithPrimaryTensor:silu secondaryTensor:up_tensor name:nil];
    MPSGraphTensor* gated_column =
      [graph reshapeTensor:gated withShape:intermediate_column_shape name:nil];
    MPSGraphTensor* projected =
      [graph matrixMultiplicationWithPrimaryTensor:weight_tensor
                                   secondaryTensor:gated_column
                                              name:nil];
    MPSGraphTensor* projected_vector =
      [graph reshapeTensor:projected withShape:hidden_shape name:nil];
    MPSGraphTensor* result =
      [graph additionWithPrimaryTensor:residual_tensor
                        secondaryTensor:projected_vector
                                   name:nil];

    MPSGraphTensorData* gate_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:gate.impl_->buffer
                                             shape:intermediate_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* up_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:up.impl_->buffer
                                             shape:intermediate_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:down_weight.impl_->buffer
                                             shape:weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* residual_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:residual.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      gate_tensor : gate_data,
      up_tensor : up_data,
      weight_tensor : weight_data,
      residual_tensor : residual_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [gate_data release];
    [up_data release];
    [weight_data release];
    [residual_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::transformer_layer_f32(
  const MpsGraphBuffer& input_layernorm_weight,
  const MpsGraphBuffer& q_weight,
  const MpsGraphBuffer& k_weight,
  const MpsGraphBuffer& v_weight,
  const MpsGraphBuffer& o_weight,
  const MpsGraphBuffer& q_norm_weight,
  const MpsGraphBuffer& k_norm_weight,
  const MpsGraphBuffer& post_attention_layernorm_weight,
  const MpsGraphBuffer& gate_weight,
  const MpsGraphBuffer& up_weight,
  const MpsGraphBuffer& down_weight,
  std::size_t layer,
  std::size_t layers,
  std::size_t position,
  std::size_t capacity_tokens,
  std::size_t hidden_size,
  std::size_t intermediate_size,
  std::size_t heads,
  std::size_t kv_heads,
  std::size_t head_dim,
  float eps,
  float theta,
  MpsGraphBuffer& hidden,
  MpsGraphBuffer& key_cache,
  MpsGraphBuffer& value_cache) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto layers_status = validate_positive_dim(layers, "transformer layer layer count");
  if (!layers_status.is_ok()) {
    return layers_status;
  }
  auto capacity_status = validate_positive_dim(capacity_tokens, "transformer layer capacity");
  if (!capacity_status.is_ok()) {
    return capacity_status;
  }
  auto hidden_status = validate_positive_dim(hidden_size, "transformer layer hidden_size");
  if (!hidden_status.is_ok()) {
    return hidden_status;
  }
  auto intermediate_status =
    validate_positive_dim(intermediate_size, "transformer layer intermediate_size");
  if (!intermediate_status.is_ok()) {
    return intermediate_status;
  }
  auto heads_status = validate_positive_dim(heads, "transformer layer heads");
  if (!heads_status.is_ok()) {
    return heads_status;
  }
  auto kv_heads_status = validate_positive_dim(kv_heads, "transformer layer kv_heads");
  if (!kv_heads_status.is_ok()) {
    return kv_heads_status;
  }
  auto head_dim_status = validate_positive_dim(head_dim, "transformer layer head_dim");
  if (!head_dim_status.is_ok()) {
    return head_dim_status;
  }
  if (head_dim % 2U != 0) {
    return Status::invalid_argument("MPSGraph transformer layer head_dim must be even");
  }
  if (heads % kv_heads != 0) {
    return Status::invalid_argument(
      "MPSGraph transformer layer heads must be divisible by kv_heads");
  }
  if (layer >= layers) {
    return Status::invalid_argument("MPSGraph transformer layer index exceeds layer count");
  }
  if (position >= capacity_tokens) {
    return Status::invalid_argument(
      "MPSGraph transformer layer position exceeds KV cache capacity");
  }
  if (!std::isfinite(eps)) {
    return Status::invalid_argument("MPSGraph transformer layer eps must be finite");
  }
  if (!std::isfinite(theta) || theta <= 0.0F) {
    return Status::invalid_argument(
      "MPSGraph transformer layer theta must be positive and finite");
  }

  std::size_t attn_dim = 0;
  if (!checked_mul(heads, head_dim, attn_dim)) {
    return Status::invalid_argument("MPSGraph transformer layer attention dim overflow");
  }
  std::size_t kv_dim = 0;
  if (!checked_mul(kv_heads, head_dim, kv_dim)) {
    return Status::invalid_argument("MPSGraph transformer layer KV dim overflow");
  }
  std::size_t q_weight_values = 0;
  if (!checked_mul(attn_dim, hidden_size, q_weight_values)) {
    return Status::invalid_argument("MPSGraph transformer layer q weight count overflow");
  }
  std::size_t kv_weight_values = 0;
  if (!checked_mul(kv_dim, hidden_size, kv_weight_values)) {
    return Status::invalid_argument("MPSGraph transformer layer KV weight count overflow");
  }
  std::size_t o_weight_values = 0;
  if (!checked_mul(hidden_size, attn_dim, o_weight_values)) {
    return Status::invalid_argument("MPSGraph transformer layer o weight count overflow");
  }
  std::size_t mlp_weight_values = 0;
  if (!checked_mul(intermediate_size, hidden_size, mlp_weight_values)) {
    return Status::invalid_argument("MPSGraph transformer layer MLP weight count overflow");
  }
  std::size_t down_weight_values = 0;
  if (!checked_mul(hidden_size, intermediate_size, down_weight_values)) {
    return Status::invalid_argument("MPSGraph transformer layer down weight count overflow");
  }
  std::size_t cache_token_values = 0;
  if (!checked_mul(capacity_tokens, kv_dim, cache_token_values)) {
    return Status::invalid_argument("MPSGraph transformer layer cache token count overflow");
  }
  std::size_t cache_values = 0;
  if (!checked_mul(layers, cache_token_values, cache_values)) {
    return Status::invalid_argument("MPSGraph transformer layer cache count overflow");
  }

  auto hidden_buffer_status =
    validate_f32_buffer(hidden, hidden_size, "transformer layer hidden");
  if (!hidden_buffer_status.is_ok()) {
    return hidden_buffer_status;
  }
  auto input_norm_status = validate_f32_buffer(
    input_layernorm_weight, hidden_size, "transformer layer input norm weight");
  if (!input_norm_status.is_ok()) {
    return input_norm_status;
  }
  auto q_weight_status =
    validate_f32_buffer(q_weight, q_weight_values, "transformer layer q weight");
  if (!q_weight_status.is_ok()) {
    return q_weight_status;
  }
  auto k_weight_status =
    validate_f32_buffer(k_weight, kv_weight_values, "transformer layer k weight");
  if (!k_weight_status.is_ok()) {
    return k_weight_status;
  }
  auto v_weight_status =
    validate_f32_buffer(v_weight, kv_weight_values, "transformer layer v weight");
  if (!v_weight_status.is_ok()) {
    return v_weight_status;
  }
  auto o_weight_status =
    validate_f32_buffer(o_weight, o_weight_values, "transformer layer o weight");
  if (!o_weight_status.is_ok()) {
    return o_weight_status;
  }
  auto q_norm_status =
    validate_f32_buffer(q_norm_weight, head_dim, "transformer layer q norm weight");
  if (!q_norm_status.is_ok()) {
    return q_norm_status;
  }
  auto k_norm_status =
    validate_f32_buffer(k_norm_weight, head_dim, "transformer layer k norm weight");
  if (!k_norm_status.is_ok()) {
    return k_norm_status;
  }
  auto post_norm_status = validate_f32_buffer(
    post_attention_layernorm_weight, hidden_size,
    "transformer layer post attention norm weight");
  if (!post_norm_status.is_ok()) {
    return post_norm_status;
  }
  auto gate_weight_status =
    validate_f32_buffer(gate_weight, mlp_weight_values, "transformer layer gate weight");
  if (!gate_weight_status.is_ok()) {
    return gate_weight_status;
  }
  auto up_weight_status =
    validate_f32_buffer(up_weight, mlp_weight_values, "transformer layer up weight");
  if (!up_weight_status.is_ok()) {
    return up_weight_status;
  }
  auto down_weight_status =
    validate_f32_buffer(down_weight, down_weight_values, "transformer layer down weight");
  if (!down_weight_status.is_ok()) {
    return down_weight_status;
  }
  auto key_cache_status =
    validate_f32_buffer(key_cache, cache_values, "transformer layer key cache");
  if (!key_cache_status.is_ok()) {
    return key_cache_status;
  }
  auto value_cache_status =
    validate_f32_buffer(value_cache, cache_values, "transformer layer value cache");
  if (!value_cache_status.is_ok()) {
    return value_cache_status;
  }

  const auto half_dim = head_dim / 2U;
  auto trig_bytes = f32_bytes(half_dim, "transformer layer trig constants");
  if (!trig_bytes.is_ok()) {
    return trig_bytes.status();
  }
  std::vector<float> cos_values(half_dim);
  std::vector<float> sin_values(half_dim);
  for (std::size_t dim = 0; dim < half_dim; ++dim) {
    const auto exponent =
      static_cast<double>(2U * dim) / static_cast<double>(head_dim);
    const auto frequency = 1.0 / std::pow(static_cast<double>(theta), exponent);
    const auto angle = static_cast<double>(position) * frequency;
    cos_values[dim] = static_cast<float>(std::cos(angle));
    sin_values[dim] = static_cast<float>(std::sin(angle));
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* hidden_shape = make_shape({hidden_size});
    MPSShape* q_weight_shape = make_shape({attn_dim, hidden_size});
    MPSShape* kv_weight_shape = make_shape({kv_dim, hidden_size});
    MPSShape* o_weight_shape = make_shape({hidden_size, attn_dim});
    MPSShape* mlp_weight_shape = make_shape({intermediate_size, hidden_size});
    MPSShape* down_weight_shape = make_shape({hidden_size, intermediate_size});
    MPSShape* cache_shape = make_shape({layers, capacity_tokens, kv_heads, head_dim});
    MPSShape* q_norm_shape = make_shape({head_dim});

    MPSGraphTensor* hidden_tensor =
      [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* input_norm_tensor =
      [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* q_weight_tensor =
      [graph placeholderWithShape:q_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* k_weight_tensor =
      [graph placeholderWithShape:kv_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* v_weight_tensor =
      [graph placeholderWithShape:kv_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* o_weight_tensor =
      [graph placeholderWithShape:o_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* q_norm_tensor =
      [graph placeholderWithShape:q_norm_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* k_norm_tensor =
      [graph placeholderWithShape:q_norm_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* post_norm_tensor =
      [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* gate_weight_tensor =
      [graph placeholderWithShape:mlp_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* up_weight_tensor =
      [graph placeholderWithShape:mlp_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* down_weight_tensor =
      [graph placeholderWithShape:down_weight_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* key_cache_tensor =
      [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* value_cache_tensor =
      [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];

    const auto layer_outputs = build_transformer_layer_block(
      graph, hidden_tensor, input_norm_tensor, q_weight_tensor, k_weight_tensor,
      v_weight_tensor, o_weight_tensor, q_norm_tensor, k_norm_tensor, post_norm_tensor,
      gate_weight_tensor, up_weight_tensor, down_weight_tensor, key_cache_tensor,
      value_cache_tensor, layer, position, capacity_tokens, hidden_size,
      intermediate_size, heads, kv_heads, head_dim, eps, cos_values, sin_values,
      trig_bytes.value(), !impl_->sdpa_attention_disabled);

    MPSGraphTensorData* hidden_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:hidden.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* input_norm_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:input_layernorm_weight.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* q_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:q_weight.impl_->buffer
                                             shape:q_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* k_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:k_weight.impl_->buffer
                                             shape:kv_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* v_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:v_weight.impl_->buffer
                                             shape:kv_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* o_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:o_weight.impl_->buffer
                                             shape:o_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* q_norm_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:q_norm_weight.impl_->buffer
                                             shape:q_norm_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* k_norm_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:k_norm_weight.impl_->buffer
                                             shape:q_norm_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* post_norm_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:post_attention_layernorm_weight.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* gate_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:gate_weight.impl_->buffer
                                             shape:mlp_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* up_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:up_weight.impl_->buffer
                                             shape:mlp_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* down_weight_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:down_weight.impl_->buffer
                                             shape:down_weight_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* key_cache_input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:key_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* value_cache_input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:value_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* hidden_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:hidden.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* key_cache_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:key_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* value_cache_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:value_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      hidden_tensor : hidden_data,
      input_norm_tensor : input_norm_data,
      q_weight_tensor : q_weight_data,
      k_weight_tensor : k_weight_data,
      v_weight_tensor : v_weight_data,
      o_weight_tensor : o_weight_data,
      q_norm_tensor : q_norm_data,
      k_norm_tensor : k_norm_data,
      post_norm_tensor : post_norm_data,
      gate_weight_tensor : gate_weight_data,
      up_weight_tensor : up_weight_data,
      down_weight_tensor : down_weight_data,
      key_cache_tensor : key_cache_input_data,
      value_cache_tensor : value_cache_input_data,
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      layer_outputs.hidden : hidden_output_data,
      layer_outputs.key_cache : key_cache_output_data,
      layer_outputs.value_cache : value_cache_output_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, results, &impl_->graph_stats);

    [hidden_data release];
    [input_norm_data release];
    [q_weight_data release];
    [k_weight_data release];
    [v_weight_data release];
    [o_weight_data release];
    [q_norm_data release];
    [k_norm_data release];
    [post_norm_data release];
    [gate_weight_data release];
    [up_weight_data release];
    [down_weight_data release];
    [key_cache_input_data release];
    [value_cache_input_data release];
    [hidden_output_data release];
    [key_cache_output_data release];
    [value_cache_output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::transformer_stack_f32(
  const std::vector<MpsGraphTransformerLayerBuffers>& layer_buffers,
  std::size_t layer_offset,
  std::size_t total_layers,
  std::size_t position,
  std::size_t capacity_tokens,
  std::size_t hidden_size,
  std::size_t intermediate_size,
  std::size_t heads,
  std::size_t kv_heads,
  std::size_t head_dim,
  float eps,
  float theta,
  MpsGraphBuffer& hidden,
  MpsGraphBuffer& key_cache,
  MpsGraphBuffer& value_cache) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  if (layer_buffers.empty()) {
    return Status::invalid_argument("MPSGraph transformer stack layer count must be positive");
  }
  if (layer_offset > total_layers || layer_buffers.size() > total_layers - layer_offset) {
    return Status::invalid_argument("MPSGraph transformer stack layer range is invalid");
  }
  auto total_layers_status =
    validate_positive_dim(total_layers, "transformer stack total layer count");
  if (!total_layers_status.is_ok()) {
    return total_layers_status;
  }
  auto capacity_status = validate_positive_dim(capacity_tokens, "transformer stack capacity");
  if (!capacity_status.is_ok()) {
    return capacity_status;
  }
  auto hidden_status = validate_positive_dim(hidden_size, "transformer stack hidden_size");
  if (!hidden_status.is_ok()) {
    return hidden_status;
  }
  auto intermediate_status =
    validate_positive_dim(intermediate_size, "transformer stack intermediate_size");
  if (!intermediate_status.is_ok()) {
    return intermediate_status;
  }
  auto heads_status = validate_positive_dim(heads, "transformer stack heads");
  if (!heads_status.is_ok()) {
    return heads_status;
  }
  auto kv_heads_status = validate_positive_dim(kv_heads, "transformer stack kv_heads");
  if (!kv_heads_status.is_ok()) {
    return kv_heads_status;
  }
  auto head_dim_status = validate_positive_dim(head_dim, "transformer stack head_dim");
  if (!head_dim_status.is_ok()) {
    return head_dim_status;
  }
  if (head_dim % 2U != 0) {
    return Status::invalid_argument("MPSGraph transformer stack head_dim must be even");
  }
  if (heads % kv_heads != 0) {
    return Status::invalid_argument(
      "MPSGraph transformer stack heads must be divisible by kv_heads");
  }
  if (position >= capacity_tokens) {
    return Status::invalid_argument(
      "MPSGraph transformer stack position exceeds KV cache capacity");
  }
  if (!std::isfinite(eps)) {
    return Status::invalid_argument("MPSGraph transformer stack eps must be finite");
  }
  if (!std::isfinite(theta) || theta <= 0.0F) {
    return Status::invalid_argument(
      "MPSGraph transformer stack theta must be positive and finite");
  }

  std::size_t attn_dim = 0;
  if (!checked_mul(heads, head_dim, attn_dim)) {
    return Status::invalid_argument("MPSGraph transformer stack attention dim overflow");
  }
  std::size_t kv_dim = 0;
  if (!checked_mul(kv_heads, head_dim, kv_dim)) {
    return Status::invalid_argument("MPSGraph transformer stack KV dim overflow");
  }
  std::size_t q_weight_values = 0;
  if (!checked_mul(attn_dim, hidden_size, q_weight_values)) {
    return Status::invalid_argument("MPSGraph transformer stack q weight count overflow");
  }
  std::size_t kv_weight_values = 0;
  if (!checked_mul(kv_dim, hidden_size, kv_weight_values)) {
    return Status::invalid_argument("MPSGraph transformer stack KV weight count overflow");
  }
  std::size_t o_weight_values = 0;
  if (!checked_mul(hidden_size, attn_dim, o_weight_values)) {
    return Status::invalid_argument("MPSGraph transformer stack o weight count overflow");
  }
  std::size_t mlp_weight_values = 0;
  if (!checked_mul(intermediate_size, hidden_size, mlp_weight_values)) {
    return Status::invalid_argument("MPSGraph transformer stack MLP weight count overflow");
  }
  std::size_t down_weight_values = 0;
  if (!checked_mul(hidden_size, intermediate_size, down_weight_values)) {
    return Status::invalid_argument("MPSGraph transformer stack down weight count overflow");
  }
  std::size_t cache_token_values = 0;
  if (!checked_mul(capacity_tokens, kv_dim, cache_token_values)) {
    return Status::invalid_argument("MPSGraph transformer stack cache token count overflow");
  }
  std::size_t cache_values = 0;
  if (!checked_mul(total_layers, cache_token_values, cache_values)) {
    return Status::invalid_argument("MPSGraph transformer stack cache count overflow");
  }

  auto hidden_buffer_status =
    validate_f32_buffer(hidden, hidden_size, "transformer stack hidden");
  if (!hidden_buffer_status.is_ok()) {
    return hidden_buffer_status;
  }
  auto key_cache_status =
    validate_f32_buffer(key_cache, cache_values, "transformer stack key cache");
  if (!key_cache_status.is_ok()) {
    return key_cache_status;
  }
  auto value_cache_status =
    validate_f32_buffer(value_cache, cache_values, "transformer stack value cache");
  if (!value_cache_status.is_ok()) {
    return value_cache_status;
  }

  auto validate_layer_buffer =
    [](const MpsGraphBuffer* buffer, std::size_t values,
       const char* name) -> Status {
      if (buffer == nullptr) {
        return Status::invalid_argument(std::string{"MPSGraph "} + name +
                                        " buffer pointer is null");
      }
      return validate_f32_buffer(*buffer, values, name);
    };
  for (std::size_t layer = 0; layer < layer_buffers.size(); ++layer) {
    const auto& buffers = layer_buffers[layer];
    const auto prefix = "transformer stack layer " + std::to_string(layer) + " ";
    auto status = validate_layer_buffer(buffers.input_layernorm_weight, hidden_size,
                                        (prefix + "input norm weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
    status = validate_layer_buffer(buffers.q_weight, q_weight_values,
                                   (prefix + "q weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
    status = validate_layer_buffer(buffers.k_weight, kv_weight_values,
                                   (prefix + "k weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
    status = validate_layer_buffer(buffers.v_weight, kv_weight_values,
                                   (prefix + "v weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
    status = validate_layer_buffer(buffers.o_weight, o_weight_values,
                                   (prefix + "o weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
    status = validate_layer_buffer(buffers.q_norm_weight, head_dim,
                                   (prefix + "q norm weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
    status = validate_layer_buffer(buffers.k_norm_weight, head_dim,
                                   (prefix + "k norm weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
    status = validate_layer_buffer(buffers.post_attention_layernorm_weight, hidden_size,
                                   (prefix + "post attention norm weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
    status = validate_layer_buffer(buffers.gate_weight, mlp_weight_values,
                                   (prefix + "gate weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
    status = validate_layer_buffer(buffers.up_weight, mlp_weight_values,
                                   (prefix + "up weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
    status = validate_layer_buffer(buffers.down_weight, down_weight_values,
                                   (prefix + "down weight").c_str());
    if (!status.is_ok()) {
      return status;
    }
  }

  const auto half_dim = head_dim / 2U;
  auto trig_bytes = f32_bytes(half_dim, "transformer stack trig constants");
  if (!trig_bytes.is_ok()) {
    return trig_bytes.status();
  }
  std::vector<float> cos_values(half_dim);
  std::vector<float> sin_values(half_dim);
  for (std::size_t dim = 0; dim < half_dim; ++dim) {
    const auto exponent =
      static_cast<double>(2U * dim) / static_cast<double>(head_dim);
    const auto frequency = 1.0 / std::pow(static_cast<double>(theta), exponent);
    const auto angle = static_cast<double>(position) * frequency;
    cos_values[dim] = static_cast<float>(std::cos(angle));
    sin_values[dim] = static_cast<float>(std::sin(angle));
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* hidden_shape = make_shape({hidden_size});
    MPSShape* q_weight_shape = make_shape({attn_dim, hidden_size});
    MPSShape* kv_weight_shape = make_shape({kv_dim, hidden_size});
    MPSShape* o_weight_shape = make_shape({hidden_size, attn_dim});
    MPSShape* mlp_weight_shape = make_shape({intermediate_size, hidden_size});
    MPSShape* down_weight_shape = make_shape({hidden_size, intermediate_size});
    MPSShape* cache_shape =
      make_shape({total_layers, capacity_tokens, kv_heads, head_dim});
    MPSShape* q_norm_shape = make_shape({head_dim});

    MPSGraphTensor* hidden_tensor =
      [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* key_cache_tensor =
      [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* value_cache_tensor =
      [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
      [NSMutableDictionary dictionaryWithCapacity:
        static_cast<NSUInteger>(3U + layer_buffers.size() * 11U)];
    std::vector<MPSGraphTensorData*> allocated_data;
    allocated_data.reserve(6U + layer_buffers.size() * 11U);
    auto add_feed = [&](MPSGraphTensor* tensor, const MpsGraphBuffer& buffer,
                        MPSShape* shape) {
      MPSGraphTensorData* data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer.impl_->buffer
                                               shape:shape
                                            dataType:MPSDataTypeFloat32];
      [feeds setObject:data forKey:tensor];
      allocated_data.push_back(data);
    };

    add_feed(hidden_tensor, hidden, hidden_shape);
    add_feed(key_cache_tensor, key_cache, cache_shape);
    add_feed(value_cache_tensor, value_cache, cache_shape);

    MPSGraphTensor* current_hidden = hidden_tensor;
    MPSGraphTensor* current_key_cache = key_cache_tensor;
    MPSGraphTensor* current_value_cache = value_cache_tensor;
    for (std::size_t layer = 0; layer < layer_buffers.size(); ++layer) {
      const auto& buffers = layer_buffers[layer];
      MPSGraphTensor* input_norm_tensor =
        [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor* q_weight_tensor =
        [graph placeholderWithShape:q_weight_shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor* k_weight_tensor =
        [graph placeholderWithShape:kv_weight_shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor* v_weight_tensor =
        [graph placeholderWithShape:kv_weight_shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor* o_weight_tensor =
        [graph placeholderWithShape:o_weight_shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor* q_norm_tensor =
        [graph placeholderWithShape:q_norm_shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor* k_norm_tensor =
        [graph placeholderWithShape:q_norm_shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor* post_norm_tensor =
        [graph placeholderWithShape:hidden_shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor* gate_weight_tensor =
        [graph placeholderWithShape:mlp_weight_shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor* up_weight_tensor =
        [graph placeholderWithShape:mlp_weight_shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor* down_weight_tensor =
        [graph placeholderWithShape:down_weight_shape dataType:MPSDataTypeFloat32 name:nil];

      add_feed(input_norm_tensor, *buffers.input_layernorm_weight, hidden_shape);
      add_feed(q_weight_tensor, *buffers.q_weight, q_weight_shape);
      add_feed(k_weight_tensor, *buffers.k_weight, kv_weight_shape);
      add_feed(v_weight_tensor, *buffers.v_weight, kv_weight_shape);
      add_feed(o_weight_tensor, *buffers.o_weight, o_weight_shape);
      add_feed(q_norm_tensor, *buffers.q_norm_weight, q_norm_shape);
      add_feed(k_norm_tensor, *buffers.k_norm_weight, q_norm_shape);
      add_feed(post_norm_tensor, *buffers.post_attention_layernorm_weight, hidden_shape);
      add_feed(gate_weight_tensor, *buffers.gate_weight, mlp_weight_shape);
      add_feed(up_weight_tensor, *buffers.up_weight, mlp_weight_shape);
      add_feed(down_weight_tensor, *buffers.down_weight, down_weight_shape);

      const auto outputs = build_transformer_layer_block(
        graph, current_hidden, input_norm_tensor, q_weight_tensor, k_weight_tensor,
        v_weight_tensor, o_weight_tensor, q_norm_tensor, k_norm_tensor,
        post_norm_tensor, gate_weight_tensor, up_weight_tensor, down_weight_tensor,
        current_key_cache, current_value_cache, layer_offset + layer, position,
        capacity_tokens, hidden_size, intermediate_size, heads, kv_heads, head_dim,
        eps, cos_values, sin_values, trig_bytes.value(), !impl_->sdpa_attention_disabled);
      current_hidden = outputs.hidden;
      current_key_cache = outputs.key_cache;
      current_value_cache = outputs.value_cache;
    }

    MPSGraphTensorData* hidden_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:hidden.impl_->buffer
                                             shape:hidden_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* key_cache_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:key_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* value_cache_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:value_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      current_hidden : hidden_output_data,
      current_key_cache : key_cache_output_data,
      current_value_cache : value_cache_output_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, results, &impl_->graph_stats);

    for (auto* data : allocated_data) {
      [data release];
    }
    [hidden_output_data release];
    [key_cache_output_data release];
    [value_cache_output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::add_f32(const MpsGraphBuffer& lhs,
                                const MpsGraphBuffer& rhs,
                                std::size_t size,
                                MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto size_status = validate_positive_dim(size, "add size");
  if (!size_status.is_ok()) {
    return size_status;
  }
  auto lhs_status = validate_f32_buffer(lhs, size, "add lhs");
  if (!lhs_status.is_ok()) {
    return lhs_status;
  }
  auto rhs_status = validate_f32_buffer(rhs, size, "add rhs");
  if (!rhs_status.is_ok()) {
    return rhs_status;
  }
  auto output_status = validate_f32_buffer(output, size, "add output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* shape = make_shape({size});
    MPSGraphTensor* lhs_tensor =
      [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* rhs_tensor =
      [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* result =
      [graph additionWithPrimaryTensor:lhs_tensor secondaryTensor:rhs_tensor name:nil];

    MPSGraphTensorData* lhs_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:lhs.impl_->buffer
                                             shape:shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* rhs_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:rhs.impl_->buffer
                                             shape:shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      lhs_tensor : lhs_data,
      rhs_tensor : rhs_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [lhs_data release];
    [rhs_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::argmax_i32(const MpsGraphBuffer& input,
                                   std::size_t size,
                                   MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto size_status = validate_positive_dim(size, "argmax size");
  if (!size_status.is_ok()) {
    return size_status;
  }
  auto input_status = validate_f32_buffer(input, size, "argmax input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  if (!output.valid()) {
    return Status::invalid_argument("MPSGraph argmax output buffer is not initialized");
  }
  if (output.byte_size() < sizeof(std::int32_t)) {
    return Status::invalid_argument("MPSGraph argmax output buffer is too small");
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* input_shape = make_shape({size});
    MPSShape* output_shape = make_shape({1});
    MPSGraphTensor* input_tensor =
      [graph placeholderWithShape:input_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* result =
      [graph reductionArgMaximumWithTensor:input_tensor axis:0 name:nil];
    result = [graph reshapeTensor:result withShape:output_shape name:nil];

    MPSGraphTensorData* input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:input.impl_->buffer
                                             shape:input_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:output_shape
                                          dataType:MPSDataTypeInt32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      input_tensor : input_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [input_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::write_i32_token(const MpsGraphBuffer& token,
                                        MpsGraphBuffer& output,
                                        std::size_t index,
                                        std::size_t capacity) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto capacity_status = validate_positive_dim(capacity, "generated token capacity");
  if (!capacity_status.is_ok()) {
    return capacity_status;
  }
  if (index >= capacity) {
    return Status::invalid_argument("MPSGraph generated token index exceeds capacity");
  }
  if (!token.valid()) {
    return Status::invalid_argument("MPSGraph generated token source is not initialized");
  }
  if (token.byte_size() < sizeof(std::int32_t)) {
    return Status::invalid_argument("MPSGraph generated token source is too small");
  }
  std::size_t output_bytes = 0;
  if (!checked_mul(capacity, sizeof(std::int32_t), output_bytes)) {
    return Status::invalid_argument("MPSGraph generated token byte count overflow");
  }
  if (!output.valid()) {
    return Status::invalid_argument("MPSGraph generated token output is not initialized");
  }
  if (output.byte_size() < output_bytes) {
    return Status::invalid_argument("MPSGraph generated token output is too small");
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* token_shape = make_shape({1});
    MPSShape* output_shape = make_shape({capacity});
    MPSGraphTensor* output_tensor =
      [graph placeholderWithShape:output_shape dataType:MPSDataTypeInt32 name:nil];
    MPSGraphTensor* token_tensor =
      [graph placeholderWithShape:token_shape dataType:MPSDataTypeInt32 name:nil];
    MPSGraphTensor* result =
      [graph sliceUpdateDataTensor:output_tensor
                       updateTensor:token_tensor
                             starts:@[ @(static_cast<NSInteger>(index)) ]
                               ends:@[ @(static_cast<NSInteger>(index + 1U)) ]
                            strides:@[ @1 ]
                          startMask:0
                            endMask:0
                        squeezeMask:0
                               name:nil];

    MPSGraphTensorData* output_input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:output_shape
                                          dataType:MPSDataTypeInt32];
    MPSGraphTensorData* token_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:token.impl_->buffer
                                             shape:token_shape
                                          dataType:MPSDataTypeInt32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:output_shape
                                          dataType:MPSDataTypeInt32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      output_tensor : output_input_data,
      token_tensor : token_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [output_input_data release];
    [token_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::reset_generation_status_i32(MpsGraphBuffer& status) const {
  constexpr std::int32_t kGeneratedCount = 0;
  constexpr std::int32_t kLengthReason = 2;
  constexpr std::int32_t kNotFinished = 0;
  const std::int32_t initial_status[] = {kGeneratedCount, kLengthReason, kNotFinished};
  return copy_to_buffer(status, initial_status, sizeof(initial_status));
}

Status MpsGraphContext::update_generation_status_i32(
  const MpsGraphBuffer& token, const std::int64_t* eos_tokens,
  std::size_t eos_token_count, std::size_t step, bool final_step,
  MpsGraphBuffer& status) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto token_status = validate_i32_buffer(token, 1, "generation status token");
  if (!token_status.is_ok()) {
    return token_status;
  }
  auto status_status = validate_i32_buffer(status, 3, "generation status");
  if (!status_status.is_ok()) {
    return status_status;
  }
  if (eos_token_count == 0) {
    return Status::invalid_argument("MPSGraph generation status EOS tokens must not be empty");
  }
  if (eos_tokens == nullptr) {
    return Status::invalid_argument("MPSGraph generation status EOS tokens must not be null");
  }
  if (step > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() - 1)) {
    return Status::invalid_argument("MPSGraph generation status step exceeds int32 range");
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* scalar_shape = make_shape({1});
    MPSShape* status_shape = make_shape({3});
    MPSGraphTensor* token_tensor =
      [graph placeholderWithShape:scalar_shape dataType:MPSDataTypeInt32 name:nil];
    MPSGraphTensor* status_tensor =
      [graph placeholderWithShape:status_shape dataType:MPSDataTypeInt32 name:nil];

    MPSGraphTensor* previous_count =
      [graph sliceTensor:status_tensor dimension:0 start:0 length:1 name:nil];
    MPSGraphTensor* previous_reason =
      [graph sliceTensor:status_tensor dimension:0 start:1 length:1 name:nil];
    MPSGraphTensor* previous_finished_i32 =
      [graph sliceTensor:status_tensor dimension:0 start:2 length:1 name:nil];
    MPSGraphTensor* zero_i32 =
      [graph constantWithScalar:0.0 shape:scalar_shape dataType:MPSDataTypeInt32];
    MPSGraphTensor* previous_finished =
      [graph notEqualWithPrimaryTensor:previous_finished_i32
                       secondaryTensor:zero_i32
                                  name:nil];

    MPSGraphTensor* is_eos = nil;
    for (std::size_t index = 0; index < eos_token_count; ++index) {
      const auto eos = static_cast<std::int32_t>(eos_tokens[index]);
      MPSGraphTensor* eos_tensor =
        [graph constantWithScalar:static_cast<double>(eos)
                            shape:scalar_shape
                         dataType:MPSDataTypeInt32];
      MPSGraphTensor* matches =
        [graph equalWithPrimaryTensor:token_tensor secondaryTensor:eos_tensor name:nil];
      is_eos = is_eos == nil
                 ? matches
                 : [graph logicalORWithPrimaryTensor:is_eos
                                      secondaryTensor:matches
                                                 name:nil];
    }
    MPSGraphTensor* should_stop =
      [graph logicalORWithPrimaryTensor:previous_finished
                         secondaryTensor:is_eos
                                    name:nil];

    MPSGraphTensor* current_count =
      [graph constantWithScalar:static_cast<double>(step + 1U)
                          shape:scalar_shape
                       dataType:MPSDataTypeInt32];
    MPSGraphTensor* step_count =
      [graph selectWithPredicateTensor:is_eos
                   truePredicateTensor:previous_count
                  falsePredicateTensor:current_count
                                  name:nil];
    MPSGraphTensor* next_count =
      [graph selectWithPredicateTensor:should_stop
                   truePredicateTensor:previous_count
                  falsePredicateTensor:step_count
                                  name:nil];
    MPSGraphTensor* stop_reason =
      [graph constantWithScalar:1.0 shape:scalar_shape dataType:MPSDataTypeInt32];
    MPSGraphTensor* length_reason =
      [graph constantWithScalar:2.0 shape:scalar_shape dataType:MPSDataTypeInt32];
    MPSGraphTensor* step_reason =
      [graph selectWithPredicateTensor:is_eos
                   truePredicateTensor:stop_reason
                  falsePredicateTensor:(final_step ? length_reason : previous_reason)
                                  name:nil];
    MPSGraphTensor* next_reason =
      [graph selectWithPredicateTensor:previous_finished
                   truePredicateTensor:previous_reason
                  falsePredicateTensor:step_reason
                                  name:nil];
    MPSGraphTensor* one_i32 =
      [graph constantWithScalar:1.0 shape:scalar_shape dataType:MPSDataTypeInt32];
    MPSGraphTensor* next_finished =
      [graph selectWithPredicateTensor:should_stop
                   truePredicateTensor:one_i32
                  falsePredicateTensor:zero_i32
                                  name:nil];
    MPSGraphTensor* count_reason =
      [graph concatTensor:next_count withTensor:next_reason dimension:0 name:nil];
    MPSGraphTensor* result =
      [graph concatTensor:count_reason withTensor:next_finished dimension:0 name:nil];

    MPSGraphTensorData* token_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:token.impl_->buffer
                                             shape:scalar_shape
                                          dataType:MPSDataTypeInt32];
    MPSGraphTensorData* status_input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:status.impl_->buffer
                                             shape:status_shape
                                          dataType:MPSDataTypeInt32];
    MPSGraphTensorData* status_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:status.impl_->buffer
                                             shape:status_shape
                                          dataType:MPSDataTypeInt32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      token_tensor : token_data,
      status_tensor : status_input_data,
    };
    const auto result_status =
      run_graph_with_results(graph, impl_->queue, feeds, result, status_output_data, &impl_->graph_stats);
    [token_data release];
    [status_input_data release];
    [status_output_data release];
    [graph release];
    return result_status;
  }
}

Status MpsGraphContext::write_kv_cache_f32(const MpsGraphBuffer& source,
                                           MpsGraphBuffer& cache,
                                           std::size_t layer,
                                           std::size_t position,
                                           std::size_t layers,
                                           std::size_t capacity_tokens,
                                           std::size_t kv_heads,
                                           std::size_t head_dim) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto layers_status = validate_positive_dim(layers, "KV cache layers");
  if (!layers_status.is_ok()) {
    return layers_status;
  }
  auto capacity_status = validate_positive_dim(capacity_tokens, "KV cache capacity");
  if (!capacity_status.is_ok()) {
    return capacity_status;
  }
  auto kv_heads_status = validate_positive_dim(kv_heads, "KV cache kv_heads");
  if (!kv_heads_status.is_ok()) {
    return kv_heads_status;
  }
  auto head_dim_status = validate_positive_dim(head_dim, "KV cache head_dim");
  if (!head_dim_status.is_ok()) {
    return head_dim_status;
  }
  if (layer >= layers) {
    return Status::invalid_argument("MPSGraph KV cache write layer exceeds capacity");
  }
  if (position >= capacity_tokens) {
    return Status::invalid_argument("MPSGraph KV cache write position exceeds capacity");
  }

  std::size_t kv_values = 0;
  if (!checked_mul(kv_heads, head_dim, kv_values)) {
    return Status::invalid_argument("MPSGraph KV cache write source size overflow");
  }
  std::size_t token_values = 0;
  if (!checked_mul(capacity_tokens, kv_values, token_values)) {
    return Status::invalid_argument("MPSGraph KV cache write layer size overflow");
  }
  std::size_t cache_values = 0;
  if (!checked_mul(layers, token_values, cache_values)) {
    return Status::invalid_argument("MPSGraph KV cache write total size overflow");
  }

  auto source_status = validate_f32_buffer(source, kv_values, "KV cache write source");
  if (!source_status.is_ok()) {
    return source_status;
  }
  auto cache_status = validate_f32_buffer(cache, cache_values, "KV cache write destination");
  if (!cache_status.is_ok()) {
    return cache_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* source_shape = make_shape({kv_heads, head_dim});
    MPSShape* update_shape = make_shape({1, 1, kv_heads, head_dim});
    MPSShape* cache_shape = make_shape({layers, capacity_tokens, kv_heads, head_dim});
    MPSGraphTensor* cache_tensor =
      [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* source_tensor =
      [graph placeholderWithShape:source_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* update_tensor =
      [graph reshapeTensor:source_tensor withShape:update_shape name:nil];
    MPSGraphTensor* result =
      [graph sliceUpdateDataTensor:cache_tensor
                       updateTensor:update_tensor
                             starts:@[
                               @(static_cast<NSInteger>(layer)),
                               @(static_cast<NSInteger>(position)),
                               @0,
                               @0,
                             ]
                               ends:@[
                                 @(static_cast<NSInteger>(layer + 1U)),
                                 @(static_cast<NSInteger>(position + 1U)),
                                 @(static_cast<NSInteger>(kv_heads)),
                                 @(static_cast<NSInteger>(head_dim)),
                               ]
                            strides:@[ @1, @1, @1, @1 ]
                          startMask:0
                            endMask:0
                        squeezeMask:0
                               name:nil];

    MPSGraphTensorData* cache_input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* source_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:source.impl_->buffer
                                             shape:source_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* cache_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      cache_tensor : cache_input_data,
      source_tensor : source_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, cache_output_data, &impl_->graph_stats);
    [cache_input_data release];
    [source_data release];
    [cache_output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::write_kv_cache_pair_f32(const MpsGraphBuffer& key_source,
                                                const MpsGraphBuffer& value_source,
                                                MpsGraphBuffer& key_cache,
                                                MpsGraphBuffer& value_cache,
                                                std::size_t layer,
                                                std::size_t position,
                                                std::size_t layers,
                                                std::size_t capacity_tokens,
                                                std::size_t kv_heads,
                                                std::size_t head_dim) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto layers_status = validate_positive_dim(layers, "KV cache pair layers");
  if (!layers_status.is_ok()) {
    return layers_status;
  }
  auto capacity_status = validate_positive_dim(capacity_tokens, "KV cache pair capacity");
  if (!capacity_status.is_ok()) {
    return capacity_status;
  }
  auto kv_heads_status = validate_positive_dim(kv_heads, "KV cache pair kv_heads");
  if (!kv_heads_status.is_ok()) {
    return kv_heads_status;
  }
  auto head_dim_status = validate_positive_dim(head_dim, "KV cache pair head_dim");
  if (!head_dim_status.is_ok()) {
    return head_dim_status;
  }
  if (layer >= layers) {
    return Status::invalid_argument("MPSGraph KV cache pair write layer exceeds capacity");
  }
  if (position >= capacity_tokens) {
    return Status::invalid_argument("MPSGraph KV cache pair write position exceeds capacity");
  }

  std::size_t kv_values = 0;
  if (!checked_mul(kv_heads, head_dim, kv_values)) {
    return Status::invalid_argument("MPSGraph KV cache pair source size overflow");
  }
  std::size_t token_values = 0;
  if (!checked_mul(capacity_tokens, kv_values, token_values)) {
    return Status::invalid_argument("MPSGraph KV cache pair layer size overflow");
  }
  std::size_t cache_values = 0;
  if (!checked_mul(layers, token_values, cache_values)) {
    return Status::invalid_argument("MPSGraph KV cache pair total size overflow");
  }

  auto key_source_status = validate_f32_buffer(key_source, kv_values,
                                               "KV cache pair key source");
  if (!key_source_status.is_ok()) {
    return key_source_status;
  }
  auto value_source_status = validate_f32_buffer(value_source, kv_values,
                                                 "KV cache pair value source");
  if (!value_source_status.is_ok()) {
    return value_source_status;
  }
  auto key_cache_status = validate_f32_buffer(key_cache, cache_values,
                                              "KV cache pair key destination");
  if (!key_cache_status.is_ok()) {
    return key_cache_status;
  }
  auto value_cache_status = validate_f32_buffer(value_cache, cache_values,
                                                "KV cache pair value destination");
  if (!value_cache_status.is_ok()) {
    return value_cache_status;
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* source_shape = make_shape({kv_heads, head_dim});
    MPSShape* update_shape = make_shape({1, 1, kv_heads, head_dim});
    MPSShape* cache_shape = make_shape({layers, capacity_tokens, kv_heads, head_dim});
    MPSGraphTensor* key_cache_tensor =
      [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* value_cache_tensor =
      [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* key_source_tensor =
      [graph placeholderWithShape:source_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* value_source_tensor =
      [graph placeholderWithShape:source_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* key_update_tensor =
      [graph reshapeTensor:key_source_tensor withShape:update_shape name:nil];
    MPSGraphTensor* value_update_tensor =
      [graph reshapeTensor:value_source_tensor withShape:update_shape name:nil];
    NSArray<NSNumber*>* starts = @[
      @(static_cast<NSInteger>(layer)),
      @(static_cast<NSInteger>(position)),
      @0,
      @0,
    ];
    NSArray<NSNumber*>* ends = @[
      @(static_cast<NSInteger>(layer + 1U)),
      @(static_cast<NSInteger>(position + 1U)),
      @(static_cast<NSInteger>(kv_heads)),
      @(static_cast<NSInteger>(head_dim)),
    ];
    NSArray<NSNumber*>* strides = @[ @1, @1, @1, @1 ];
    MPSGraphTensor* key_result =
      [graph sliceUpdateDataTensor:key_cache_tensor
                       updateTensor:key_update_tensor
                             starts:starts
                               ends:ends
                            strides:strides
                          startMask:0
                            endMask:0
                        squeezeMask:0
                               name:nil];
    MPSGraphTensor* value_result =
      [graph sliceUpdateDataTensor:value_cache_tensor
                       updateTensor:value_update_tensor
                             starts:starts
                               ends:ends
                            strides:strides
                          startMask:0
                            endMask:0
                        squeezeMask:0
                               name:nil];

    MPSGraphTensorData* key_cache_input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:key_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* value_cache_input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:value_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* key_source_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:key_source.impl_->buffer
                                             shape:source_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* value_source_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:value_source.impl_->buffer
                                             shape:source_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* key_cache_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:key_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* value_cache_output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:value_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      key_cache_tensor : key_cache_input_data,
      value_cache_tensor : value_cache_input_data,
      key_source_tensor : key_source_data,
      value_source_tensor : value_source_data,
    };
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [@{
      key_result : key_cache_output_data,
      value_result : value_cache_output_data,
    } mutableCopy];
    Status status = Status::ok();
    @try {
      const auto started = SteadyClock::now();
      [graph runWithMTLCommandQueue:impl_->queue
                              feeds:feeds
                   targetOperations:nil
                  resultsDictionary:results];
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        SteadyClock::now() - started);
      ++impl_->graph_stats.graph_execute_calls;
      impl_->graph_stats.graph_execute_ns += static_cast<std::uint64_t>(elapsed.count());
      ++impl_->graph_stats.executable_cache_misses;
    } @catch (NSException* exception) {
      status = Status::internal_error(exception_to_string(exception));
    }

    [results release];
    [key_cache_input_data release];
    [value_cache_input_data release];
    [key_source_data release];
    [value_source_data release];
    [key_cache_output_data release];
    [value_cache_output_data release];
    [graph release];
    return status;
  }
}

Status MpsGraphContext::attention_f32(const MpsGraphBuffer& query,
                                      const MpsGraphBuffer& key_cache,
                                      const MpsGraphBuffer& value_cache,
                                      std::size_t layer, std::size_t position,
                                      std::size_t capacity_tokens,
                                      std::size_t heads, std::size_t kv_heads,
                                      std::size_t head_dim,
                                      MpsGraphBuffer& output) const {
  if (!valid()) {
    return Status::unavailable(kNotReady);
  }
  auto capacity_status = validate_positive_dim(capacity_tokens, "attention capacity");
  if (!capacity_status.is_ok()) {
    return capacity_status;
  }
  auto heads_status = validate_positive_dim(heads, "attention heads");
  if (!heads_status.is_ok()) {
    return heads_status;
  }
  auto kv_heads_status = validate_positive_dim(kv_heads, "attention kv_heads");
  if (!kv_heads_status.is_ok()) {
    return kv_heads_status;
  }
  auto head_dim_status = validate_positive_dim(head_dim, "attention head_dim");
  if (!head_dim_status.is_ok()) {
    return head_dim_status;
  }
  if (heads % kv_heads != 0) {
    return Status::invalid_argument("MPSGraph attention heads must be divisible by kv_heads");
  }
  if (position >= capacity_tokens) {
    return Status::invalid_argument("MPSGraph attention position exceeds KV cache capacity");
  }
  if (layer == std::numeric_limits<std::size_t>::max()) {
    return Status::invalid_argument("MPSGraph attention layer count overflow");
  }
  const auto layer_count = layer + 1U;
  auto layer_status = validate_positive_dim(layer_count, "attention layer count");
  if (!layer_status.is_ok()) {
    return layer_status;
  }

  std::size_t query_values = 0;
  if (!checked_mul(heads, head_dim, query_values)) {
    return Status::invalid_argument("MPSGraph attention query element count overflow");
  }
  std::size_t kv_dim = 0;
  if (!checked_mul(kv_heads, head_dim, kv_dim)) {
    return Status::invalid_argument("MPSGraph attention KV dimension overflow");
  }
  std::size_t cache_token_values = 0;
  if (!checked_mul(capacity_tokens, kv_dim, cache_token_values)) {
    return Status::invalid_argument("MPSGraph attention KV token count overflow");
  }
  std::size_t cache_values = 0;
  if (!checked_mul(layer_count, cache_token_values, cache_values)) {
    return Status::invalid_argument("MPSGraph attention KV cache element count overflow");
  }

  auto query_status = validate_f32_buffer(query, query_values, "attention query");
  if (!query_status.is_ok()) {
    return query_status;
  }
  auto key_status = validate_f32_buffer(key_cache, cache_values, "attention key cache");
  if (!key_status.is_ok()) {
    return key_status;
  }
  auto value_status = validate_f32_buffer(value_cache, cache_values, "attention value cache");
  if (!value_status.is_ok()) {
    return value_status;
  }
  auto output_status = validate_f32_buffer(output, query_values, "attention output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  const auto seq_len = position + 1U;
  const auto kv_group = heads / kv_heads;

  if (!impl_->sdpa_attention_disabled) {
    if (@available(macOS 15.0, *)) {
      @autoreleasepool {
        MPSGraph* graph = make_graph(&impl_->graph_stats);
        if (graph == nil) {
          return Status::unavailable("failed to create MPSGraph");
        }

        MPSShape* query_shape = make_shape({heads, head_dim});
        MPSShape* cache_shape = make_shape({layer_count, capacity_tokens, kv_heads, head_dim});
        MPSGraphTensorData* query_data = nil;
        MPSGraphTensorData* key_data = nil;
        MPSGraphTensorData* value_data = nil;
        MPSGraphTensorData* output_data = nil;
        Status sdpa_status = Status::ok();

        @try {
          MPSGraphTensor* query_tensor =
            [graph placeholderWithShape:query_shape dataType:MPSDataTypeFloat32 name:nil];
          MPSGraphTensor* key_tensor =
            [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];
          MPSGraphTensor* value_tensor =
            [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];

          MPSGraphTensor* layer_keys =
            [graph sliceTensor:key_tensor dimension:0
                         start:static_cast<NSInteger>(layer) length:1 name:nil];
          MPSGraphTensor* layer_values =
            [graph sliceTensor:value_tensor dimension:0
                         start:static_cast<NSInteger>(layer) length:1 name:nil];
          layer_keys =
            [graph reshapeTensor:layer_keys
                        withShape:make_shape({capacity_tokens, kv_heads, head_dim})
                             name:nil];
          layer_values =
            [graph reshapeTensor:layer_values
                        withShape:make_shape({capacity_tokens, kv_heads, head_dim})
                             name:nil];
          MPSGraphTensor* visible_keys =
            [graph sliceTensor:layer_keys dimension:0 start:0
                        length:static_cast<NSInteger>(seq_len) name:nil];
          MPSGraphTensor* visible_values =
            [graph sliceTensor:layer_values dimension:0 start:0
                        length:static_cast<NSInteger>(seq_len) name:nil];

          MPSGraphTensor* sdpa_query =
            [graph reshapeTensor:query_tensor
                        withShape:make_shape({1, heads, 1, head_dim})
                             name:nil];
          MPSGraphTensor* sdpa_keys =
            [graph transposeTensor:visible_keys permutation:@[ @1, @0, @2 ] name:nil];
          sdpa_keys = [graph reshapeTensor:sdpa_keys
                                  withShape:make_shape({kv_heads, 1, seq_len, head_dim})
                                       name:nil];
          sdpa_keys = [graph tileTensor:sdpa_keys
                          withMultiplier:make_shape({1, kv_group, 1, 1})
                                    name:nil];
          sdpa_keys = [graph reshapeTensor:sdpa_keys
                                  withShape:make_shape({1, heads, seq_len, head_dim})
                                       name:nil];

          MPSGraphTensor* sdpa_values =
            [graph transposeTensor:visible_values permutation:@[ @1, @0, @2 ] name:nil];
          sdpa_values = [graph reshapeTensor:sdpa_values
                                    withShape:make_shape({kv_heads, 1, seq_len, head_dim})
                                         name:nil];
          sdpa_values = [graph tileTensor:sdpa_values
                            withMultiplier:make_shape({1, kv_group, 1, 1})
                                      name:nil];
          sdpa_values = [graph reshapeTensor:sdpa_values
                                    withShape:make_shape({1, heads, seq_len, head_dim})
                                         name:nil];

          const auto attention_scale =
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
          MPSGraphTensor* sdpa_output =
            [graph scaledDotProductAttentionWithQueryTensor:sdpa_query
                                                  keyTensor:sdpa_keys
                                                valueTensor:sdpa_values
                                                      scale:attention_scale
                                                       name:nil];
          MPSGraphTensor* result =
            [graph reshapeTensor:sdpa_output withShape:query_shape name:nil];

          query_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:query.impl_->buffer
                                                    shape:query_shape
                                                 dataType:MPSDataTypeFloat32];
          key_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:key_cache.impl_->buffer
                                                    shape:cache_shape
                                                 dataType:MPSDataTypeFloat32];
          value_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:value_cache.impl_->buffer
                                                    shape:cache_shape
                                                 dataType:MPSDataTypeFloat32];
          output_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                                    shape:query_shape
                                                 dataType:MPSDataTypeFloat32];
          NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            query_tensor : query_data,
            key_tensor : key_data,
            value_tensor : value_data,
          };
          sdpa_status = run_graph_with_results(graph, impl_->queue, feeds, result,
                                               output_data, &impl_->graph_stats);
        } @catch (NSException* exception) {
          sdpa_status = Status::internal_error(exception_to_string(exception));
        }

        [query_data release];
        [key_data release];
        [value_data release];
        [output_data release];
        [graph release];

        if (sdpa_status.is_ok()) {
          return sdpa_status;
        }
        impl_->sdpa_attention_disabled = true;
      }
    }
  }

  @autoreleasepool {
    MPSGraph* graph = make_graph(&impl_->graph_stats);
    if (graph == nil) {
      return Status::unavailable("failed to create MPSGraph");
    }

    MPSShape* query_shape = make_shape({heads, head_dim});
    MPSShape* cache_shape = make_shape({layer_count, capacity_tokens, kv_heads, head_dim});
    MPSGraphTensor* query_tensor =
      [graph placeholderWithShape:query_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* key_tensor =
      [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* value_tensor =
      [graph placeholderWithShape:cache_shape dataType:MPSDataTypeFloat32 name:nil];

    MPSGraphTensor* result =
      build_grouped_attention_from_cache(graph, query_tensor, key_tensor, value_tensor,
                                         layer, position, capacity_tokens, heads,
                                         kv_heads, head_dim);

    MPSGraphTensorData* query_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:query.impl_->buffer
                                             shape:query_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* key_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:key_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* value_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:value_cache.impl_->buffer
                                             shape:cache_shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output.impl_->buffer
                                             shape:query_shape
                                          dataType:MPSDataTypeFloat32];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      query_tensor : query_data,
      key_tensor : key_data,
      value_tensor : value_data,
    };
    const auto status =
      run_graph_with_results(graph, impl_->queue, feeds, result, output_data, &impl_->graph_stats);
    [query_data release];
    [key_data release];
    [value_data release];
    [output_data release];
    [graph release];
    return status;
  }
}

BackendInfo query_backend() { return build_backend_info(); }

std::string format_backend_info(const BackendInfo& info) {
  std::ostringstream output;
  output << "MPSGraph compiled: " << (info.compiled ? "yes" : "no") << '\n';
  output << "MPSGraph available: " << (info.available ? "yes" : "no") << '\n';
  output << "MPSGraph ready: " << (info.graph_ready ? "yes" : "no") << '\n';
  if (!info.device_name.empty()) {
    output << "Device: " << info.device_name << '\n';
  }
  if (info.recommended_max_working_set_size != 0) {
    output << "Recommended max working set size: "
           << info.recommended_max_working_set_size << '\n';
  }
  output << "Low power: " << (info.low_power ? "yes" : "no") << '\n';
  output << "Headless: " << (info.headless ? "yes" : "no") << '\n';
  output << "Removable: " << (info.removable ? "yes" : "no") << '\n';
  if (!info.failure_reason.empty()) {
    output << "Reason: " << info.failure_reason << '\n';
  }
  return output.str();
}

Status run_operator_smoke_test() {
  const auto info = build_backend_info();
  if (!info.available || !info.graph_ready) {
    const auto message = info.failure_reason.empty()
                           ? std::string{"MPSGraph backend is not ready"}
                           : info.failure_reason;
    return Status::unavailable(message);
  }

  auto context_result = MpsGraphContext::create();
  if (!context_result.is_ok()) {
    return context_result.status();
  }
  auto context = std::move(context_result.value());

  auto embedding_weight = make_f32_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F});
  if (!embedding_weight.is_ok()) {
    return embedding_weight.status();
  }
  auto embedding_output = context.make_buffer(sizeof(float) * 2U);
  if (!embedding_output.is_ok()) {
    return embedding_output.status();
  }
  auto embedding_buffer = std::move(embedding_output.value());
  auto status =
    context.embedding_f32(embedding_weight.value(), 2, 2, 1, embedding_buffer);
  if (!status.is_ok()) {
    return status;
  }
  auto output = read_f32_buffer(context, embedding_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 3.0F, "embedding[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], 4.0F, "embedding[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto norm_input = make_f32_buffer(context, {1.0F, 1.0F});
  auto norm_weight = make_f32_buffer(context, {2.0F, 3.0F});
  if (!norm_input.is_ok()) {
    return norm_input.status();
  }
  if (!norm_weight.is_ok()) {
    return norm_weight.status();
  }
  auto norm_output = context.make_buffer(sizeof(float) * 2U);
  if (!norm_output.is_ok()) {
    return norm_output.status();
  }
  auto norm_buffer = std::move(norm_output.value());
  status = context.rms_norm_f32(norm_input.value(), norm_weight.value(), 2, 0.0F,
                                norm_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, norm_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 2.0F, "rms_norm[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], 3.0F, "rms_norm[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto qk_input = make_f32_buffer(context, {3.0F, 4.0F, 0.0F, 2.0F});
  auto qk_weight = make_f32_buffer(context, {1.0F, 2.0F});
  if (!qk_input.is_ok()) {
    return qk_input.status();
  }
  if (!qk_weight.is_ok()) {
    return qk_weight.status();
  }
  auto qk_output = context.make_buffer(sizeof(float) * 4U);
  if (!qk_output.is_ok()) {
    return qk_output.status();
  }
  auto qk_buffer = std::move(qk_output.value());
  status = context.qk_norm_f32(qk_input.value(), qk_weight.value(), 2, 2, 0.0F,
                               qk_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, qk_buffer, 4);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 3.0F / std::sqrt(12.5F), "qk_norm[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], 8.0F / std::sqrt(12.5F), "qk_norm[1]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[2], 0.0F, "qk_norm[2]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[3], 4.0F / std::sqrt(2.0F), "qk_norm[3]");
  if (!status.is_ok()) {
    return status;
  }

  auto rope_input = make_f32_buffer(context, {1.0F, 2.0F});
  if (!rope_input.is_ok()) {
    return rope_input.status();
  }
  auto rope_output = context.make_buffer(sizeof(float) * 2U);
  if (!rope_output.is_ok()) {
    return rope_output.status();
  }
  auto rope_buffer = std::move(rope_output.value());
  status = context.rope_f32(rope_input.value(), 1, 2, 0, 10000.0F, rope_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, rope_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 1.0F, "rope[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], 2.0F, "rope[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto q_norm_rope_input = make_f32_buffer(context, {3.0F, 4.0F, 0.0F, 2.0F});
  auto k_norm_rope_input = make_f32_buffer(context, {1.0F, 2.0F});
  auto q_norm_rope_weight = make_f32_buffer(context, {1.0F, 2.0F});
  auto k_norm_rope_weight = make_f32_buffer(context, {3.0F, 4.0F});
  if (!q_norm_rope_input.is_ok()) {
    return q_norm_rope_input.status();
  }
  if (!k_norm_rope_input.is_ok()) {
    return k_norm_rope_input.status();
  }
  if (!q_norm_rope_weight.is_ok()) {
    return q_norm_rope_weight.status();
  }
  if (!k_norm_rope_weight.is_ok()) {
    return k_norm_rope_weight.status();
  }
  auto q_norm_rope_output = context.make_buffer(sizeof(float) * 4U);
  auto k_norm_rope_output = context.make_buffer(sizeof(float) * 2U);
  if (!q_norm_rope_output.is_ok()) {
    return q_norm_rope_output.status();
  }
  if (!k_norm_rope_output.is_ok()) {
    return k_norm_rope_output.status();
  }
  auto q_norm_rope_buffer = std::move(q_norm_rope_output.value());
  auto k_norm_rope_buffer = std::move(k_norm_rope_output.value());
  status = context.qk_norm_rope_f32(q_norm_rope_input.value(),
                                    k_norm_rope_input.value(),
                                    q_norm_rope_weight.value(),
                                    k_norm_rope_weight.value(), 2, 1, 2, 1,
                                    0.0F, 10000.0F, q_norm_rope_buffer,
                                    k_norm_rope_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, q_norm_rope_buffer, 4);
  if (!output.is_ok()) {
    return output.status();
  }
  const auto q0_first = 3.0F / std::sqrt(12.5F);
  const auto q0_second = 8.0F / std::sqrt(12.5F);
  const auto q1_first = 0.0F;
  const auto q1_second = 4.0F / std::sqrt(2.0F);
  status = check_close(output.value()[0],
                       q0_first * std::cos(1.0F) - q0_second * std::sin(1.0F),
                       "qk_norm_rope.q[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1],
                       q0_second * std::cos(1.0F) + q0_first * std::sin(1.0F),
                       "qk_norm_rope.q[1]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[2],
                       q1_first * std::cos(1.0F) - q1_second * std::sin(1.0F),
                       "qk_norm_rope.q[2]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[3],
                       q1_second * std::cos(1.0F) + q1_first * std::sin(1.0F),
                       "qk_norm_rope.q[3]");
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, k_norm_rope_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  const auto k_first = 3.0F / std::sqrt(2.5F);
  const auto k_second = 8.0F / std::sqrt(2.5F);
  status = check_close(output.value()[0],
                       k_first * std::cos(1.0F) - k_second * std::sin(1.0F),
                       "qk_norm_rope.k[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1],
                       k_second * std::cos(1.0F) + k_first * std::sin(1.0F),
                       "qk_norm_rope.k[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto matvec_weight = make_f32_buffer(context, {1.0F, 2.0F, 3.0F, 1.0F});
  auto matvec_input = make_f32_buffer(context, {3.0F, 4.0F});
  if (!matvec_weight.is_ok()) {
    return matvec_weight.status();
  }
  if (!matvec_input.is_ok()) {
    return matvec_input.status();
  }
  auto matvec_output = context.make_buffer(sizeof(float) * 2U);
  if (!matvec_output.is_ok()) {
    return matvec_output.status();
  }
  auto matvec_buffer = std::move(matvec_output.value());
  status = context.matvec_f32(matvec_weight.value(), 2, 2, matvec_input.value(),
                              matvec_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, matvec_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 11.0F, "matvec[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], 13.0F, "matvec[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto q_weight = make_f32_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F});
  auto k_weight = make_f32_buffer(context, {5.0F, 6.0F});
  auto v_weight = make_f32_buffer(context, {7.0F, 8.0F});
  auto qkv_input = make_f32_buffer(context, {2.0F, 3.0F});
  if (!q_weight.is_ok()) {
    return q_weight.status();
  }
  if (!k_weight.is_ok()) {
    return k_weight.status();
  }
  if (!v_weight.is_ok()) {
    return v_weight.status();
  }
  if (!qkv_input.is_ok()) {
    return qkv_input.status();
  }
  auto q_output = context.make_buffer(sizeof(float) * 2U);
  auto k_output = context.make_buffer(sizeof(float));
  auto v_output = context.make_buffer(sizeof(float));
  if (!q_output.is_ok()) {
    return q_output.status();
  }
  if (!k_output.is_ok()) {
    return k_output.status();
  }
  if (!v_output.is_ok()) {
    return v_output.status();
  }
  auto q_buffer = std::move(q_output.value());
  auto k_buffer = std::move(k_output.value());
  auto v_buffer = std::move(v_output.value());
  status = context.qkv_matvec_f32(q_weight.value(), k_weight.value(), v_weight.value(),
                                  2, 1, 2, qkv_input.value(), q_buffer, k_buffer,
                                  v_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, q_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 8.0F, "qkv_matvec.q[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], 18.0F, "qkv_matvec.q[1]");
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, k_buffer, 1);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 28.0F, "qkv_matvec.k[0]");
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, v_buffer, 1);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 38.0F, "qkv_matvec.v[0]");
  if (!status.is_ok()) {
    return status;
  }

  auto gate_weight = make_f32_buffer(context, {1.0F, 0.0F, 0.0F, 1.0F});
  auto up_weight = make_f32_buffer(context, {2.0F, 0.0F, 0.0F, 3.0F});
  auto gate_up_input = make_f32_buffer(context, {4.0F, 5.0F});
  if (!gate_weight.is_ok()) {
    return gate_weight.status();
  }
  if (!up_weight.is_ok()) {
    return up_weight.status();
  }
  if (!gate_up_input.is_ok()) {
    return gate_up_input.status();
  }
  auto gate_output = context.make_buffer(sizeof(float) * 2U);
  auto up_output = context.make_buffer(sizeof(float) * 2U);
  if (!gate_output.is_ok()) {
    return gate_output.status();
  }
  if (!up_output.is_ok()) {
    return up_output.status();
  }
  auto gate_buffer = std::move(gate_output.value());
  auto up_buffer = std::move(up_output.value());
  status = context.gate_up_matvec_f32(gate_weight.value(), up_weight.value(), 2, 2,
                                      gate_up_input.value(), gate_buffer, up_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, gate_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 4.0F, "gate_up_matvec.gate[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], 5.0F, "gate_up_matvec.gate[1]");
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, up_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 8.0F, "gate_up_matvec.up[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], 15.0F, "gate_up_matvec.up[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto attn_o_weight =
    make_f32_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F});
  auto attn_project_input = make_f32_buffer(context, {2.0F, 3.0F});
  auto attn_residual = make_f32_buffer(context, {10.0F, 20.0F});
  auto attn_norm_weight = make_f32_buffer(context, {2.0F, 3.0F});
  if (!attn_o_weight.is_ok()) {
    return attn_o_weight.status();
  }
  if (!attn_project_input.is_ok()) {
    return attn_project_input.status();
  }
  if (!attn_residual.is_ok()) {
    return attn_residual.status();
  }
  if (!attn_norm_weight.is_ok()) {
    return attn_norm_weight.status();
  }
  auto attn_residual_output = context.make_buffer(sizeof(float) * 2U);
  auto attn_norm_output = context.make_buffer(sizeof(float) * 2U);
  if (!attn_residual_output.is_ok()) {
    return attn_residual_output.status();
  }
  if (!attn_norm_output.is_ok()) {
    return attn_norm_output.status();
  }
  auto attn_residual_buffer = std::move(attn_residual_output.value());
  auto attn_norm_buffer = std::move(attn_norm_output.value());
  status = context.attn_project_residual_norm_f32(
    attn_o_weight.value(), attn_project_input.value(), attn_residual.value(),
    attn_norm_weight.value(), 2, 2, 0.0F, attn_residual_buffer,
    attn_norm_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, attn_residual_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  const auto attn_res0 = 18.0F;
  const auto attn_res1 = 38.0F;
  status = check_close(output.value()[0], attn_res0,
                       "attn_project_residual_norm.residual[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], attn_res1,
                       "attn_project_residual_norm.residual[1]");
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, attn_norm_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  const auto attn_norm_scale =
    1.0F / std::sqrt((attn_res0 * attn_res0 + attn_res1 * attn_res1) / 2.0F);
  status = check_close(output.value()[0], attn_res0 * attn_norm_scale * 2.0F,
                       "attn_project_residual_norm.norm[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], attn_res1 * attn_norm_scale * 3.0F,
                       "attn_project_residual_norm.norm[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto gate = make_f32_buffer(context, {0.0F, 1.0F});
  auto up = make_f32_buffer(context, {5.0F, 6.0F});
  if (!gate.is_ok()) {
    return gate.status();
  }
  if (!up.is_ok()) {
    return up.status();
  }
  auto silu_output = context.make_buffer(sizeof(float) * 2U);
  if (!silu_output.is_ok()) {
    return silu_output.status();
  }
  auto silu_buffer = std::move(silu_output.value());
  status = context.silu_mul_f32(gate.value(), up.value(), 2, silu_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, silu_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 0.0F, "silu[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], 6.0F / (1.0F + std::exp(-1.0F)), "silu[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto swiglu_gate = make_f32_buffer(context, {0.0F, 1.0F});
  auto swiglu_up = make_f32_buffer(context, {5.0F, 6.0F});
  auto swiglu_down_weight =
    make_f32_buffer(context, {1.0F, 2.0F, 3.0F, 4.0F});
  auto swiglu_residual = make_f32_buffer(context, {10.0F, 20.0F});
  if (!swiglu_gate.is_ok()) {
    return swiglu_gate.status();
  }
  if (!swiglu_up.is_ok()) {
    return swiglu_up.status();
  }
  if (!swiglu_down_weight.is_ok()) {
    return swiglu_down_weight.status();
  }
  if (!swiglu_residual.is_ok()) {
    return swiglu_residual.status();
  }
  auto swiglu_output = context.make_buffer(sizeof(float) * 2U);
  if (!swiglu_output.is_ok()) {
    return swiglu_output.status();
  }
  auto swiglu_buffer = std::move(swiglu_output.value());
  status = context.swiglu_down_residual_f32(
    swiglu_gate.value(), swiglu_up.value(), swiglu_down_weight.value(),
    swiglu_residual.value(), 2, 2, swiglu_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, swiglu_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  const auto swiglu_0 = 0.0F;
  const auto swiglu_1 = 6.0F / (1.0F + std::exp(-1.0F));
  status = check_close(output.value()[0], 10.0F + swiglu_0 + 2.0F * swiglu_1,
                       "swiglu_down_residual[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1],
                       20.0F + 3.0F * swiglu_0 + 4.0F * swiglu_1,
                       "swiglu_down_residual[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto lhs = make_f32_buffer(context, {1.0F, 2.0F});
  auto rhs = make_f32_buffer(context, {3.0F, 4.0F});
  if (!lhs.is_ok()) {
    return lhs.status();
  }
  if (!rhs.is_ok()) {
    return rhs.status();
  }
  auto add_output = context.make_buffer(sizeof(float) * 2U);
  if (!add_output.is_ok()) {
    return add_output.status();
  }
  auto add_buffer = std::move(add_output.value());
  status = context.add_f32(lhs.value(), rhs.value(), 2, add_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, add_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], 4.0F, "add[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], 6.0F, "add[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto query = make_f32_buffer(context, {1.0F, 0.0F});
  auto key_cache = make_f32_buffer(context, {1.0F, 0.0F, 0.0F, 1.0F});
  auto value_cache = make_f32_buffer(context, {5.0F, 6.0F, 7.0F, 8.0F});
  if (!query.is_ok()) {
    return query.status();
  }
  if (!key_cache.is_ok()) {
    return key_cache.status();
  }
  if (!value_cache.is_ok()) {
    return value_cache.status();
  }
  auto attention_output = context.make_buffer(sizeof(float) * 2U);
  if (!attention_output.is_ok()) {
    return attention_output.status();
  }
  auto attention_buffer = std::move(attention_output.value());
  status = context.attention_f32(query.value(), key_cache.value(), value_cache.value(),
                                 0, 1, 2, 1, 1, 2, attention_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, attention_buffer, 2);
  if (!output.is_ok()) {
    return output.status();
  }
  const auto score0 = std::exp(1.0F / std::sqrt(2.0F));
  const auto weight0 = score0 / (score0 + 1.0F);
  const auto weight1 = 1.0F - weight0;
  status = check_close(output.value()[0], weight0 * 5.0F + weight1 * 7.0F,
                       "attention[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], weight0 * 6.0F + weight1 * 8.0F,
                       "attention[1]");
  if (!status.is_ok()) {
    return status;
  }

  auto gqa_query = make_f32_buffer(context, {1.0F, 0.0F, 0.0F, 1.0F});
  if (!gqa_query.is_ok()) {
    return gqa_query.status();
  }
  auto gqa_output = context.make_buffer(sizeof(float) * 4U);
  if (!gqa_output.is_ok()) {
    return gqa_output.status();
  }
  auto gqa_buffer = std::move(gqa_output.value());
  status = context.attention_f32(gqa_query.value(), key_cache.value(), value_cache.value(),
                                 0, 1, 2, 2, 1, 2, gqa_buffer);
  if (!status.is_ok()) {
    return status;
  }
  output = read_f32_buffer(context, gqa_buffer, 4);
  if (!output.is_ok()) {
    return output.status();
  }
  status = check_close(output.value()[0], weight0 * 5.0F + weight1 * 7.0F,
                       "gqa_attention[0]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[1], weight0 * 6.0F + weight1 * 8.0F,
                       "gqa_attention[1]");
  if (!status.is_ok()) {
    return status;
  }
  status = check_close(output.value()[2], weight1 * 5.0F + weight0 * 7.0F,
                       "gqa_attention[2]");
  if (!status.is_ok()) {
    return status;
  }
  return check_close(output.value()[3], weight1 * 6.0F + weight0 * 8.0F,
                     "gqa_attention[3]");
}

}  // namespace toyllm::mpsgraph
