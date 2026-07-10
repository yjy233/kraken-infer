#include "toyllm/backends/mps/mps_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace toyllm::mps {

namespace {

constexpr const char* kKernelSourcePart0 = R"metal(
#include <metal_stdlib>
using namespace metal;

static inline float bf16_to_float(ushort value) {
  return as_type<float>(static_cast<uint>(value) << 16);
}

static inline float f16_to_float(const device uchar* data) {
  const ushort value = static_cast<ushort>(data[0]) |
                       (static_cast<ushort>(data[1]) << 8);
  const uint sign = (static_cast<uint>(value) >> 15U) & 1U;
  const uint exponent = (static_cast<uint>(value) >> 10U) & 31U;
  const uint mantissa = static_cast<uint>(value) & 1023U;
  float result = 0.0f;
  if (exponent == 0U) {
    result = static_cast<float>(mantissa) * 5.960464477539063e-8f;
  } else if (exponent == 31U) {
    result = mantissa == 0U ? INFINITY : NAN;
  } else {
    result = (1.0f + static_cast<float>(mantissa) / 1024.0f) *
             exp2(static_cast<float>(exponent) - 15.0f);
  }
  return sign == 0U ? result : -result;
}

struct SizeParams {
  uint size;
  uint threads;
};

struct RowwiseParams {
  uint row_size;
  uint total;
};

struct Q4KMatVecParams {
  uint cols;
  uint blocks_per_row;
  uint threads_per_row;
};

struct QuantGetRowParams {
  uint row;
  uint cols;
  uint blocks_per_row;
};

struct QuantGetRowsParams {
  uint rows;
  uint tokens;
  uint cols;
  uint blocks_per_row;
};

struct QuantMatMulParams {
  uint cols;
  uint blocks_per_row;
  uint threads_per_row;
  uint rows;
  uint tokens;
};

struct EmbeddingParams {
  uint token;
  uint hidden_size;
};

struct RmsNormParams {
  uint size;
  float eps;
  uint threads;
};

struct BatchedRmsNormParams {
  uint size;
  uint tokens;
  float eps;
  uint threads;
};

struct QkNormParams {
  uint heads;
  uint head_dim;
  float eps;
  uint threads;
};

struct BatchedQkNormParams {
  uint tokens;
  uint heads;
  uint head_dim;
  float eps;
  uint threads;
};

struct L2NormParams {
  uint rows;
  uint row_size;
  float eps;
  uint threads;
};

struct SplitQkvL2NormParams {
  uint tokens;
  uint key_heads;
  uint value_heads;
  uint head_dim;
  uint conv_channels;
  float eps;
  uint threads;
};

struct GatedDeltaNetParams {
  uint key_heads;
  uint value_heads;
  uint head_dim;
  uint threads;
  float scale;
};

struct BatchedGatedDeltaNetParams {
  uint tokens;
  uint key_heads;
  uint value_heads;
  uint head_dim;
  uint threads;
  float scale;
};

struct SsmConvParams {
  uint conv_kernel;
  uint input_span;
  uint channels;
  uint tokens;
  uint sequences;
  uint total;
};

struct SsmConv1StatefulParams {
  uint conv_kernel;
  uint channels;
};

struct BuildSsmConvStateParams {
  uint conv_kernel;
  uint channels;
  uint tokens;
  uint input_span;
};

struct RopeParams {
  uint heads;
  uint head_dim;
  uint position;
  float theta;
};

struct MropeParams {
  uint tokens;
  uint heads;
  uint head_dim;
  uint n_dims;
  uint section_0;
  uint section_1;
  uint section_2;
  uint section_3;
  float theta;
};

struct CopyRegionParams {
  uint source_offset;
  uint destination_offset;
  uint size;
};

struct CopyRowsParams {
  uint source_offset;
  uint destination_offset;
  uint source_stride;
  uint destination_stride;
  uint row_size;
  uint total;
};

struct AttentionParams {
  uint layer;
  uint position;
  uint capacity_tokens;
  uint heads;
  uint kv_heads;
  uint head_dim;
  uint group;
  float scale;
};

struct BatchedAttentionParams {
  uint layer;
  uint start_position;
  uint tokens;
  uint capacity_tokens;
  uint heads;
  uint kv_heads;
  uint head_dim;
  uint group;
  uint threads;
  float scale;
};

struct FlashAttention256Params {
  uint layer;
  uint start_position;
  uint tokens;
  uint capacity_tokens;
  uint heads;
  uint kv_heads;
  uint group;
  float scale;
};

kernel void bf16_matvec(const device ushort* weight [[buffer(0)]],
                        const device float* input [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant uint& cols [[buffer(3)]],
                        constant uint& threads_per_row [[buffer(4)]],
                        uint3 group_id [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint row = group_id.x;
  float sum = 0.0f;
  const uint row_offset = row * cols;
  for (uint col = lane; col < cols; col += threads_per_row) {
    sum += bf16_to_float(weight[row_offset + col]) * input[col];
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads_per_row >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    output[row] = partials[0];
  }
}

static inline uint q4_k_scale_min(uint j, const device uchar* scales) {
  uint d = 0;
  uint m = 0;
  if (j < 4U) {
    d = scales[j] & 63U;
    m = scales[j + 4U] & 63U;
  } else {
    d = (scales[j + 4U] & 0xFU) | ((scales[j - 4U] >> 6U) << 4U);
    m = (scales[j + 4U] >> 4U) | ((scales[j] >> 6U) << 4U);
  }
  return d | (m << 8U);
}

kernel void q4_k_matvec(const device uchar* weight [[buffer(0)]],
                        const device float* input [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant Q4KMatVecParams& params [[buffer(3)]],
                        uint3 group_id [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint row = group_id.x;
  float sum = 0.0f;
  const uint block_size = 144U;
  const uint row_block = row * params.blocks_per_row;
  for (uint block = lane; block < params.blocks_per_row; block += params.threads_per_row) {
    const device uchar* q4 = weight + (row_block + block) * block_size;
    const float d = f16_to_float(q4);
    const float dmin = f16_to_float(q4 + 2U);
    const device uchar* scales = q4 + 4U;
    const device uchar* qs = q4 + 16U;
    const uint input_base = block * 256U;

    for (uint segment = 0; segment < 4U; ++segment) {
      const uint packed1 = q4_k_scale_min(segment * 2U, scales);
      const uint packed2 = q4_k_scale_min(segment * 2U + 1U, scales);
      const float d1 = d * static_cast<float>(packed1 & 0xFFU);
      const float m1 = dmin * static_cast<float>((packed1 >> 8U) & 0xFFU);
      const float d2 = d * static_cast<float>(packed2 & 0xFFU);
      const float m2 = dmin * static_cast<float>((packed2 >> 8U) & 0xFFU);
      const device uchar* q = qs + segment * 32U;
      const uint col_base = input_base + segment * 64U;
      for (uint i = 0; i < 32U; ++i) {
        const uint packed = q[i];
        sum += (d1 * static_cast<float>(packed & 0xFU) - m1) * input[col_base + i];
        sum += (d2 * static_cast<float>(packed >> 4U) - m2) *
               input[col_base + 32U + i];
      }
    }
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads_per_row >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    output[row] = partials[0];
  }
}

kernel void q5_k_matvec(const device uchar* weight [[buffer(0)]],
                        const device float* input [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant Q4KMatVecParams& params [[buffer(3)]],
                        uint3 group_id [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint row = group_id.x;
  float sum = 0.0f;
  const uint block_size = 176U;
  const uint row_block = row * params.blocks_per_row;
  for (uint block = lane; block < params.blocks_per_row; block += params.threads_per_row) {
    const device uchar* q5 = weight + (row_block + block) * block_size;
    const float d = f16_to_float(q5);
    const float dmin = f16_to_float(q5 + 2U);
    const device uchar* scales = q5 + 4U;
    const device uchar* qh = q5 + 16U;
    const device uchar* qs = q5 + 48U;
    const uint input_base = block * 256U;
    uint u1 = 1U;
    uint u2 = 2U;

    for (uint segment = 0; segment < 4U; ++segment) {
      const uint packed1 = q4_k_scale_min(segment * 2U, scales);
      const uint packed2 = q4_k_scale_min(segment * 2U + 1U, scales);
      const float d1 = d * static_cast<float>(packed1 & 0xFFU);
      const float m1 = dmin * static_cast<float>((packed1 >> 8U) & 0xFFU);
      const float d2 = d * static_cast<float>(packed2 & 0xFFU);
      const float m2 = dmin * static_cast<float>((packed2 >> 8U) & 0xFFU);
      const device uchar* q = qs + segment * 32U;
      const uint col_base = input_base + segment * 64U;
      for (uint i = 0; i < 32U; ++i) {
        const uint packed = q[i];
        const uint high1 = (qh[i] & u1) != 0U ? 16U : 0U;
        const uint high2 = (qh[i] & u2) != 0U ? 16U : 0U;
        sum += (d1 * static_cast<float>((packed & 0xFU) + high1) - m1) *
               input[col_base + i];
        sum += (d2 * static_cast<float>((packed >> 4U) + high2) - m2) *
               input[col_base + 32U + i];
      }
      u1 <<= 2U;
      u2 <<= 2U;
    }
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads_per_row >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    output[row] = partials[0];
  }
}

kernel void q6_k_matvec(const device uchar* weight [[buffer(0)]],
                        const device float* input [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant Q4KMatVecParams& params [[buffer(3)]],
                        uint3 group_id [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint row = group_id.x;
  float sum = 0.0f;
  const uint block_size = 210U;
  const uint row_block = row * params.blocks_per_row;
  for (uint block = lane; block < params.blocks_per_row; block += params.threads_per_row) {
    const device uchar* q6 = weight + (row_block + block) * block_size;
    const device uchar* ql_base = q6;
    const device uchar* qh_base = q6 + 128U;
    const device char* scales_base = reinterpret_cast<const device char*>(q6 + 192U);
    const float d = f16_to_float(q6 + 208U);
    const uint input_base = block * 256U;

    for (uint chunk = 0; chunk < 2U; ++chunk) {
      const device uchar* ql = ql_base + chunk * 64U;
      const device uchar* qh = qh_base + chunk * 32U;
      const device char* scales = scales_base + chunk * 8U;
      const uint col_base = input_base + chunk * 128U;
      for (uint i = 0; i < 32U; ++i) {
        const uint scale_index = i / 16U;
        const int q1 =
          static_cast<int>((ql[i] & 0xFU) | (((qh[i] >> 0U) & 3U) << 4U)) - 32;
        const int q2 =
          static_cast<int>((ql[i + 32U] & 0xFU) | (((qh[i] >> 2U) & 3U) << 4U)) - 32;
        const int q3 =
          static_cast<int>((ql[i] >> 4U) | (((qh[i] >> 4U) & 3U) << 4U)) - 32;
        const int q4 =
          static_cast<int>((ql[i + 32U] >> 4U) | (((qh[i] >> 6U) & 3U) << 4U)) - 32;
        sum += d * static_cast<float>(scales[scale_index + 0U]) *
               static_cast<float>(q1) * input[col_base + i];
        sum += d * static_cast<float>(scales[scale_index + 2U]) *
               static_cast<float>(q2) * input[col_base + i + 32U];
        sum += d * static_cast<float>(scales[scale_index + 4U]) *
               static_cast<float>(q3) * input[col_base + i + 64U];
        sum += d * static_cast<float>(scales[scale_index + 6U]) *
               static_cast<float>(q4) * input[col_base + i + 96U];
      }
    }
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads_per_row >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    output[row] = partials[0];
  }
}

struct BlockQ4K {
  half d;
  half dmin;
  uchar scales[12];
  uchar qs[128];
};

struct BlockQ5K {
  half d;
  half dmin;
  uchar scales[12];
  uchar qh[32];
  uchar qs[128];
};

struct BlockQ6K {
  uchar ql[128];
  uchar qh[64];
  char scales[16];
  half d;
};

static inline uchar2 k_quant_q4_scale_min(short j, short k,
                                          const device uchar* q) {
  if (j < 4) {
    return uchar2{uchar(q[j + k] & 63), uchar(q[j + 4 + k] & 63)};
  }
  return uchar2{uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xC0) >> 2)),
                uchar((q[j + 4 + k] >> 4) | ((q[j + k] & 0xC0) >> 2))};
}

static inline void dequantize_q4_k_mm(const device BlockQ4K* xb, short il,
                                      thread half4x4& reg) {
  const device uchar* q = xb->qs;
  const short is = (il / 4) * 2;
  q = q + (il / 4) * 32 + 16 * (il & 1);
  il = il & 3;
  const uchar2 sc = k_quant_q4_scale_min(is, il / 2, xb->scales);
  const float d = il < 2 ? static_cast<float>(xb->d)
                         : static_cast<float>(xb->d) / 16.0f;
  const float min_value = static_cast<float>(xb->dmin);
  const float dl = d * static_cast<float>(sc[0]);
  const float ml = min_value * static_cast<float>(sc[1]);
  const ushort mask = il < 2 ? 0x0F : 0xF0;

  for (int i = 0; i < 16; ++i) {
    reg[i / 4][i % 4] = static_cast<half>(dl * static_cast<float>(q[i] & mask) - ml);
  }
}

static inline void dequantize_q5_k_mm(const device BlockQ5K* xb, short il,
                                      thread half4x4& reg) {
  const device uchar* q = xb->qs;
  const device uchar* qh = xb->qh;
  const short is = (il / 4) * 2;
  q = q + 32 * (il / 4) + 16 * (il & 1);
  qh = qh + 16 * (il & 1);
  const uchar high_mask = static_cast<uchar>(1U << (il / 2));
  il = il & 3;
  const uchar2 sc = k_quant_q4_scale_min(is, il / 2, xb->scales);
  const float d = il < 2 ? static_cast<float>(xb->d)
                         : static_cast<float>(xb->d) / 16.0f;
  const float min_value = static_cast<float>(xb->dmin);
  const float dl = d * static_cast<float>(sc[0]);
  const float ml = min_value * static_cast<float>(sc[1]);
  const ushort mask = il < 2 ? 0x0F : 0xF0;
  const float qh_value = il < 2 ? 16.0f : 256.0f;

  for (int i = 0; i < 16; ++i) {
    const float high = (qh[i] & high_mask) != 0 ? qh_value : 0.0f;
    reg[i / 4][i % 4] =
      static_cast<half>(dl * (static_cast<float>(q[i] & mask) + high) - ml);
  }
}

static inline void dequantize_q6_k_mm(const device BlockQ6K* xb, short il,
                                      thread half4x4& reg) {
  const float d_all = static_cast<float>(xb->d);
  const device ushort* ql = reinterpret_cast<const device ushort*>(xb->ql);
  const device ushort* qh = reinterpret_cast<const device ushort*>(xb->qh);
  const device char* scales = xb->scales;

  ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
  qh = qh + 16 * (il / 8) + 8 * (il & 1);
  const float sc = static_cast<float>(scales[(il % 2) + 2 * (il / 2)]);
  il = (il / 2) & 3;

  const uint kmask1 = il > 1 ? (il > 2 ? 0xC0C0C0C0 : 0x30303030)
                             : (il > 0 ? 0x0C0C0C0C : 0x03030303);
  const uint kmask2 = il > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
  const float ml = d_all * sc * 32.0f;
  const float dl0 = d_all * sc;
  const float dl1 = dl0 / 256.0f;
  const float dl2 = dl0 / (256.0f * 256.0f);
  const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
  const uchar shr_h = il > 2 ? 2 : 0;
  const uchar shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);
  const uchar shr_l = il > 1 ? 4 : 0;

  for (int i = 0; i < 4; ++i) {
    const uint low = (static_cast<uint>(ql[2 * i]) |
                      (static_cast<uint>(ql[2 * i + 1]) << 16)) &
                     kmask2;
    const uint high = (static_cast<uint>(qh[2 * i]) |
                       (static_cast<uint>(qh[2 * i + 1]) << 16)) &
                      kmask1;
    const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
    reg[i][0] = static_cast<half>(dl0 * static_cast<float>(q & 0xFF) - ml);
    reg[i][1] = static_cast<half>(dl1 * static_cast<float>(q & 0xFF00) - ml);
    reg[i][2] = static_cast<half>(dl2 * static_cast<float>(q & 0xFF0000) - ml);
    reg[i][3] =
      static_cast<half>(dl3 * static_cast<float>(q & 0xFF000000) - ml);
  }
}

static inline void dequantize_q4_k_mv_ext(const device BlockQ4K* xb, short il,
                                          thread float4x4& reg) {
  half4x4 half_reg;
  dequantize_q4_k_mm(xb, il, half_reg);
  reg = static_cast<float4x4>(half_reg);
}

static inline void dequantize_q5_k_mv_ext(const device BlockQ5K* xb, short il,
                                          thread float4x4& reg) {
  half4x4 half_reg;
  dequantize_q5_k_mm(xb, il, half_reg);
  reg = static_cast<float4x4>(half_reg);
}

static inline void dequantize_q6_k_mv_ext(const device BlockQ6K* xb, short il,
                                          thread float4x4& reg) {
  half4x4 half_reg;
  dequantize_q6_k_mm(xb, il, half_reg);
  reg = static_cast<float4x4>(half_reg);
}

template <short r1ptg, typename BlockType,
          void (*dequantize_func)(const device BlockType*, short, thread float4x4&)>
static inline void k_quant_mul_mv_ext_body(const device uchar* weight,
                                           const device float* input,
                                           device float* output,
                                           constant QuantMatMulParams& params,
                                           uint3 group_id,
                                           ushort thread_index_in_simdgroup,
                                           ushort simd_group) {
  constexpr short nsg = 2;
  constexpr short nxpsg = 8;
  constexpr short nypsg = 32 / nxpsg;
  constexpr short chunks_per_block = 16;

  const short tx = thread_index_in_simdgroup % nxpsg;
  const short ty = thread_index_in_simdgroup / nxpsg;
  const uint row = group_id.x * (nypsg * nsg) + nypsg * simd_group + ty;
  const uint token_base = group_id.y * r1ptg;

  device const BlockType* xq =
    row < params.rows
      ? reinterpret_cast<const device BlockType*>(
          weight + static_cast<ulong>(row) *
                     static_cast<ulong>(params.blocks_per_row) *
                     sizeof(BlockType)) +
          tx / chunks_per_block
      : reinterpret_cast<const device BlockType*>(weight);

  device const float4x4* y4x4[r1ptg];
  for (short token = 0; token < r1ptg; ++token) {
    y4x4[token] =
      token_base + token < params.tokens
        ? reinterpret_cast<const device float4x4*>(
            input + static_cast<ulong>(token_base + token) *
                      static_cast<ulong>(params.cols)) +
          tx
        : reinterpret_cast<const device float4x4*>(input);
  }

  float sums[r1ptg] = {0.0f};
  short chunk = tx % chunks_per_block;
  for (uint input_chunk = tx; 16U * input_chunk < params.cols;
       input_chunk += nxpsg) {
    float4x4 weights;
    dequantize_func(xq, chunk, weights);

    for (short token = 0; token < r1ptg; ++token) {
      sums[token] += dot(weights[0], y4x4[token][0][0]) +
                     dot(weights[1], y4x4[token][0][1]) +
                     dot(weights[2], y4x4[token][0][2]) +
                     dot(weights[3], y4x4[token][0][3]);
      y4x4[token] += nxpsg;
    }

    chunk += nxpsg;
    if (chunk >= chunks_per_block) {
      xq += chunk / chunks_per_block;
      chunk %= chunks_per_block;
    }
  }

  for (short token = 0; token < r1ptg; ++token) {
    sums[token] += simd_shuffle_down(sums[token], 4);
    sums[token] += simd_shuffle_down(sums[token], 2);
    sums[token] += simd_shuffle_down(sums[token], 1);
  }

  if (tx == 0 && row < params.rows) {
    for (short token = 0; token < r1ptg && token_base + token < params.tokens;
         ++token) {
      output[(token_base + token) * params.rows + row] = sums[token];
    }
  }
}

kernel void q4_k_mul_mv_ext_r1_3(const device uchar* weight [[buffer(0)]],
                                 const device float* input [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant QuantMatMulParams& params [[buffer(3)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 ushort thread_index_in_simdgroup
                                   [[thread_index_in_simdgroup]],
                                 ushort simd_group
                                   [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mv_ext_body<3, BlockQ4K, dequantize_q4_k_mv_ext>(
    weight, input, output, params, group_id, thread_index_in_simdgroup,
    simd_group);
}

kernel void q4_k_mul_mv_ext_r1_4(const device uchar* weight [[buffer(0)]],
                                 const device float* input [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant QuantMatMulParams& params [[buffer(3)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 ushort thread_index_in_simdgroup
                                   [[thread_index_in_simdgroup]],
                                 ushort simd_group
                                   [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mv_ext_body<4, BlockQ4K, dequantize_q4_k_mv_ext>(
    weight, input, output, params, group_id, thread_index_in_simdgroup,
    simd_group);
}

kernel void q4_k_mul_mv_ext_r1_5(const device uchar* weight [[buffer(0)]],
                                 const device float* input [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant QuantMatMulParams& params [[buffer(3)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 ushort thread_index_in_simdgroup
                                   [[thread_index_in_simdgroup]],
                                 ushort simd_group
                                   [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mv_ext_body<5, BlockQ4K, dequantize_q4_k_mv_ext>(
    weight, input, output, params, group_id, thread_index_in_simdgroup,
    simd_group);
}

kernel void q5_k_mul_mv_ext_r1_3(const device uchar* weight [[buffer(0)]],
                                 const device float* input [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant QuantMatMulParams& params [[buffer(3)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 ushort thread_index_in_simdgroup
                                   [[thread_index_in_simdgroup]],
                                 ushort simd_group
                                   [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mv_ext_body<3, BlockQ5K, dequantize_q5_k_mv_ext>(
    weight, input, output, params, group_id, thread_index_in_simdgroup,
    simd_group);
}

kernel void q5_k_mul_mv_ext_r1_4(const device uchar* weight [[buffer(0)]],
                                 const device float* input [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant QuantMatMulParams& params [[buffer(3)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 ushort thread_index_in_simdgroup
                                   [[thread_index_in_simdgroup]],
                                 ushort simd_group
                                   [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mv_ext_body<4, BlockQ5K, dequantize_q5_k_mv_ext>(
    weight, input, output, params, group_id, thread_index_in_simdgroup,
    simd_group);
}

kernel void q5_k_mul_mv_ext_r1_5(const device uchar* weight [[buffer(0)]],
                                 const device float* input [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant QuantMatMulParams& params [[buffer(3)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 ushort thread_index_in_simdgroup
                                   [[thread_index_in_simdgroup]],
                                 ushort simd_group
                                   [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mv_ext_body<5, BlockQ5K, dequantize_q5_k_mv_ext>(
    weight, input, output, params, group_id, thread_index_in_simdgroup,
    simd_group);
}

kernel void q6_k_mul_mv_ext_r1_3(const device uchar* weight [[buffer(0)]],
                                 const device float* input [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant QuantMatMulParams& params [[buffer(3)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 ushort thread_index_in_simdgroup
                                   [[thread_index_in_simdgroup]],
                                 ushort simd_group
                                   [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mv_ext_body<3, BlockQ6K, dequantize_q6_k_mv_ext>(
    weight, input, output, params, group_id, thread_index_in_simdgroup,
    simd_group);
}

kernel void q6_k_mul_mv_ext_r1_4(const device uchar* weight [[buffer(0)]],
                                 const device float* input [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant QuantMatMulParams& params [[buffer(3)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 ushort thread_index_in_simdgroup
                                   [[thread_index_in_simdgroup]],
                                 ushort simd_group
                                   [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mv_ext_body<4, BlockQ6K, dequantize_q6_k_mv_ext>(
    weight, input, output, params, group_id, thread_index_in_simdgroup,
    simd_group);
}

kernel void q6_k_mul_mv_ext_r1_5(const device uchar* weight [[buffer(0)]],
                                 const device float* input [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant QuantMatMulParams& params [[buffer(3)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 ushort thread_index_in_simdgroup
                                   [[thread_index_in_simdgroup]],
                                 ushort simd_group
                                   [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mv_ext_body<5, BlockQ6K, dequantize_q6_k_mv_ext>(
    weight, input, output, params, group_id, thread_index_in_simdgroup,
    simd_group);
}

template <typename BlockType,
          void (*dequantize_func)(const device BlockType*, short, thread half4x4&)>
static inline void k_quant_mul_mm_simd_body(const device uchar* weight,
                                           const device float* input,
                                           device float* output,
                                           constant QuantMatMulParams& params,
                                           threadgroup char* shmem,
                                           uint3 group_id,
                                           ushort thread_index,
                                           ushort simd_group) {
  threadgroup half* weight_tile = reinterpret_cast<threadgroup half*>(shmem);
  threadgroup half* input_tile = reinterpret_cast<threadgroup half*>(shmem + 4096);

  constexpr short output_rows_per_group = 64;
  constexpr short output_tokens_per_group = 32;
  constexpr short k_tile = 32;
  constexpr short weight_loads_per_thread = k_tile / 16;
  constexpr short input_loads_per_thread = k_tile / 8;

  const uint row_base = group_id.y * output_rows_per_group;
  const uint token_base = group_id.x * output_tokens_per_group;
  const short active_rows =
    static_cast<short>(min(static_cast<uint>(output_rows_per_group),
                           params.rows - row_base));
  const short active_tokens =
    static_cast<short>(min(static_cast<uint>(output_tokens_per_group),
                           params.tokens - token_base));
  const short load_row = min(static_cast<short>(thread_index / weight_loads_per_thread),
                             static_cast<short>(active_rows - 1));
  const short load_token =
    min(static_cast<short>(thread_index / input_loads_per_thread),
        static_cast<short>(active_tokens - 1));
  const short il0 = thread_index % weight_loads_per_thread;
  short il = il0;

  device const BlockType* x =
    reinterpret_cast<device const BlockType*>(
      weight + static_cast<ulong>(row_base + load_row) *
                 static_cast<ulong>(params.blocks_per_row) * sizeof(BlockType));
  const short input_k_offset = 8 * (thread_index % input_loads_per_thread);
  device const float* y = input + static_cast<ulong>(token_base + load_token) *
                                    static_cast<ulong>(params.cols) +
                          static_cast<ulong>(input_k_offset);

  simdgroup_half8x8 weight_matrices[4];
  simdgroup_half8x8 input_matrices[2];
  simdgroup_float8x8 accumulators[8];

  for (short i = 0; i < 8; ++i) {
    accumulators[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
  }

  for (uint loop_k = 0; loop_k < params.cols; loop_k += k_tile) {
    half4x4 temp_weight;
    dequantize_func(x, il, temp_weight);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short i = 0; i < 16; ++i) {
      const short sx = 2 * il0 + i / 8;
      const short sy = (thread_index / weight_loads_per_thread) / 8;
      const short lx = (thread_index / weight_loads_per_thread) % 8;
      const short ly = i % 8;
      const short ib = 8 * sx + sy;
      *(weight_tile + 64 * ib + 8 * ly + lx) = temp_weight[i / 4][i % 4];
    }

    const short sx = thread_index % input_loads_per_thread;
    const short sy = (thread_index / input_loads_per_thread) / 8;
    const short ly = (thread_index / input_loads_per_thread) % 8;
    const short ib = 4 * sx + sy;
    *(threadgroup half2x4*)(input_tile + 64 * ib + 8 * ly) =
      static_cast<half2x4>(*reinterpret_cast<const device float2x4*>(y));

    il = (il + 2 < 16) ? il + 2 : il % 2;
    x = (il < 2) ? x + 1 : x;
    y += k_tile;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const half* weight_matrix =
      weight_tile + 4 * 64 * (simd_group % 2);
    threadgroup const half* input_matrix =
      input_tile + 2 * 64 * (simd_group / 2);

    for (short ik = 0; ik < k_tile / 8; ++ik) {
      simdgroup_barrier(mem_flags::mem_none);
      for (short i = 0; i < 4; ++i) {
        simdgroup_load(weight_matrices[i], weight_matrix + 64 * i, 8, 0, false);
      }
      simdgroup_barrier(mem_flags::mem_none);
      for (short i = 0; i < 2; ++i) {
        simdgroup_load(input_matrices[i], input_matrix + 64 * i, 8, 0, false);
      }
      simdgroup_barrier(mem_flags::mem_none);
      for (short i = 0; i < 8; ++i) {
        simdgroup_multiply_accumulate(accumulators[i], input_matrices[i / 4],
                                      weight_matrices[i % 4],
                                      accumulators[i]);
      }

      weight_matrix += 8 * 64;
      input_matrix += 4 * 64;
    }
  }

  if (row_base + output_rows_per_group <= params.rows &&
      token_base + output_tokens_per_group <= params.tokens) {
    device float* tile_output =
      output + (row_base + 32 * (simd_group & 1)) +
      (token_base + 16 * (simd_group >> 1)) * params.rows;
    for (short i = 0; i < 8; ++i) {
      simdgroup_store(accumulators[i],
                      tile_output + 8 * (i % 4) + 8 * params.rows * (i / 4),
                      params.rows, 0, false);
    }
  } else {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* temp_output =
      reinterpret_cast<threadgroup float*>(shmem) + 32 * (simd_group & 1) +
      16 * (simd_group >> 1) * output_rows_per_group;
    for (short i = 0; i < 8; ++i) {
      simdgroup_store(accumulators[i],
                      temp_output + 8 * (i % 4) +
                        8 * output_rows_per_group * (i / 4),
                      output_rows_per_group, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
      for (short token = thread_index; token < active_tokens;
           token += output_tokens_per_group) {
        device float* dst =
          output + row_base + (token_base + token) * params.rows;
        threadgroup float* src = reinterpret_cast<threadgroup float*>(shmem) +
                                 token * output_rows_per_group;
        for (short row = 0; row < active_rows; ++row) {
          dst[row] = src[row];
        }
      }
    }
  }
}

kernel void q4_k_mul_mm_simd(const device uchar* weight [[buffer(0)]],
                             const device float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant QuantMatMulParams& params [[buffer(3)]],
                             threadgroup char* shmem [[threadgroup(0)]],
                             uint3 group_id [[threadgroup_position_in_grid]],
                             ushort thread_index [[thread_index_in_threadgroup]],
                             ushort simd_group [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mm_simd_body<BlockQ4K, dequantize_q4_k_mm>(
    weight, input, output, params, shmem, group_id, thread_index, simd_group);
}

kernel void q5_k_mul_mm_simd(const device uchar* weight [[buffer(0)]],
                             const device float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant QuantMatMulParams& params [[buffer(3)]],
                             threadgroup char* shmem [[threadgroup(0)]],
                             uint3 group_id [[threadgroup_position_in_grid]],
                             ushort thread_index [[thread_index_in_threadgroup]],
                             ushort simd_group [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mm_simd_body<BlockQ5K, dequantize_q5_k_mm>(
    weight, input, output, params, shmem, group_id, thread_index, simd_group);
}

kernel void q6_k_mul_mm_simd(const device uchar* weight [[buffer(0)]],
                             const device float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant QuantMatMulParams& params [[buffer(3)]],
                             threadgroup char* shmem [[threadgroup(0)]],
                             uint3 group_id [[threadgroup_position_in_grid]],
                             ushort thread_index [[thread_index_in_threadgroup]],
                             ushort simd_group [[simdgroup_index_in_threadgroup]]) {
  k_quant_mul_mm_simd_body<BlockQ6K, dequantize_q6_k_mm>(
    weight, input, output, params, shmem, group_id, thread_index, simd_group);
}

kernel void q4_k_matmul(const device uchar* weight [[buffer(0)]],
                        const device float* input [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant QuantMatMulParams& params [[buffer(3)]],
                        uint3 group_id [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint row = group_id.x;
  const uint token = group_id.y;
  float sum = 0.0f;
  const uint block_size = 144U;
  const uint row_block = row * params.blocks_per_row;
  const uint token_input_base = token * params.cols;
  for (uint block = lane; block < params.blocks_per_row; block += params.threads_per_row) {
    const device uchar* q4 = weight + (row_block + block) * block_size;
    const float d = f16_to_float(q4);
    const float dmin = f16_to_float(q4 + 2U);
    const device uchar* scales = q4 + 4U;
    const device uchar* qs = q4 + 16U;
    const uint input_base = token_input_base + block * 256U;

    for (uint segment = 0; segment < 4U; ++segment) {
      const uint packed1 = q4_k_scale_min(segment * 2U, scales);
      const uint packed2 = q4_k_scale_min(segment * 2U + 1U, scales);
      const float d1 = d * static_cast<float>(packed1 & 0xFFU);
      const float m1 = dmin * static_cast<float>((packed1 >> 8U) & 0xFFU);
      const float d2 = d * static_cast<float>(packed2 & 0xFFU);
      const float m2 = dmin * static_cast<float>((packed2 >> 8U) & 0xFFU);
      const device uchar* q = qs + segment * 32U;
      const uint col_base = input_base + segment * 64U;
      for (uint i = 0; i < 32U; ++i) {
        const uint packed = q[i];
        sum += (d1 * static_cast<float>(packed & 0xFU) - m1) * input[col_base + i];
        sum += (d2 * static_cast<float>(packed >> 4U) - m2) *
               input[col_base + 32U + i];
      }
    }
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads_per_row >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    output[token * params.rows + row] = partials[0];
  }
}

kernel void q5_k_matmul(const device uchar* weight [[buffer(0)]],
                        const device float* input [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant QuantMatMulParams& params [[buffer(3)]],
                        uint3 group_id [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint row = group_id.x;
  const uint token = group_id.y;
  float sum = 0.0f;
  const uint block_size = 176U;
  const uint row_block = row * params.blocks_per_row;
  const uint token_input_base = token * params.cols;
  for (uint block = lane; block < params.blocks_per_row; block += params.threads_per_row) {
    const device uchar* q5 = weight + (row_block + block) * block_size;
    const float d = f16_to_float(q5);
    const float dmin = f16_to_float(q5 + 2U);
    const device uchar* scales = q5 + 4U;
    const device uchar* qh = q5 + 16U;
    const device uchar* qs = q5 + 48U;
    const uint input_base = token_input_base + block * 256U;
    uint u1 = 1U;
    uint u2 = 2U;

    for (uint segment = 0; segment < 4U; ++segment) {
      const uint packed1 = q4_k_scale_min(segment * 2U, scales);
      const uint packed2 = q4_k_scale_min(segment * 2U + 1U, scales);
      const float d1 = d * static_cast<float>(packed1 & 0xFFU);
      const float m1 = dmin * static_cast<float>((packed1 >> 8U) & 0xFFU);
      const float d2 = d * static_cast<float>(packed2 & 0xFFU);
      const float m2 = dmin * static_cast<float>((packed2 >> 8U) & 0xFFU);
      const device uchar* q = qs + segment * 32U;
      const uint col_base = input_base + segment * 64U;
      for (uint i = 0; i < 32U; ++i) {
        const uint packed = q[i];
        const uint high1 = (qh[i] & u1) != 0U ? 16U : 0U;
        const uint high2 = (qh[i] & u2) != 0U ? 16U : 0U;
        sum += (d1 * static_cast<float>((packed & 0xFU) + high1) - m1) *
               input[col_base + i];
        sum += (d2 * static_cast<float>((packed >> 4U) + high2) - m2) *
               input[col_base + 32U + i];
      }
      u1 <<= 2U;
      u2 <<= 2U;
    }
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads_per_row >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    output[token * params.rows + row] = partials[0];
  }
}

kernel void q6_k_matmul(const device uchar* weight [[buffer(0)]],
                        const device float* input [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant QuantMatMulParams& params [[buffer(3)]],
                        uint3 group_id [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint row = group_id.x;
  const uint token = group_id.y;
  float sum = 0.0f;
  const uint block_size = 210U;
  const uint row_block = row * params.blocks_per_row;
  const uint token_input_base = token * params.cols;
  for (uint block = lane; block < params.blocks_per_row; block += params.threads_per_row) {
    const device uchar* q6 = weight + (row_block + block) * block_size;
    const device uchar* ql_base = q6;
    const device uchar* qh_base = q6 + 128U;
    const device char* scales_base = reinterpret_cast<const device char*>(q6 + 192U);
    const float d = f16_to_float(q6 + 208U);
    const uint input_base = token_input_base + block * 256U;

    for (uint chunk = 0; chunk < 2U; ++chunk) {
      const device uchar* ql = ql_base + chunk * 64U;
      const device uchar* qh = qh_base + chunk * 32U;
      const device char* scales = scales_base + chunk * 8U;
      const uint col_base = input_base + chunk * 128U;
      for (uint i = 0; i < 32U; ++i) {
        const uint scale_index = i / 16U;
        const int q1 =
          static_cast<int>((ql[i] & 0xFU) | (((qh[i] >> 0U) & 3U) << 4U)) - 32;
        const int q2 =
          static_cast<int>((ql[i + 32U] & 0xFU) | (((qh[i] >> 2U) & 3U) << 4U)) - 32;
        const int q3 =
          static_cast<int>((ql[i] >> 4U) | (((qh[i] >> 4U) & 3U) << 4U)) - 32;
        const int q4 =
          static_cast<int>((ql[i + 32U] >> 4U) | (((qh[i] >> 6U) & 3U) << 4U)) - 32;
        const float w1 = d * static_cast<float>(scales[scale_index + 0U]) *
                         static_cast<float>(q1);
        const float w2 = d * static_cast<float>(scales[scale_index + 2U]) *
                         static_cast<float>(q2);
        const float w3 = d * static_cast<float>(scales[scale_index + 4U]) *
                         static_cast<float>(q3);
        const float w4 = d * static_cast<float>(scales[scale_index + 6U]) *
                         static_cast<float>(q4);
        sum += w1 * input[col_base + i];
        sum += w2 * input[col_base + i + 32U];
        sum += w3 * input[col_base + i + 64U];
        sum += w4 * input[col_base + i + 96U];
      }
    }
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads_per_row >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    output[token * params.rows + row] = partials[0];
  }
}

kernel void q4_k_get_row(const device uchar* weight [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant QuantGetRowParams& params [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
  if (index >= params.cols) {
    return;
  }
  const uint block_size = 144U;
  const uint block = index / 256U;
  const uint within = index % 256U;
  const uint segment = within / 64U;
  const uint in_segment = within % 64U;
  const uint scale_index = segment * 2U + (in_segment >= 32U ? 1U : 0U);
  const device uchar* q4 =
    weight + (params.row * params.blocks_per_row + block) * block_size;
  const float d = f16_to_float(q4);
  const float dmin = f16_to_float(q4 + 2U);
  const device uchar* scales = q4 + 4U;
  const device uchar* qs = q4 + 16U + segment * 32U;
  const uint packed_scale = q4_k_scale_min(scale_index, scales);
  const float ds = d * static_cast<float>(packed_scale & 0xFFU);
  const float ms = dmin * static_cast<float>((packed_scale >> 8U) & 0xFFU);
  const uint packed = qs[in_segment % 32U];
  const uint q = in_segment < 32U ? (packed & 0xFU) : (packed >> 4U);
  output[index] = ds * static_cast<float>(q) - ms;
}

kernel void q5_k_get_row(const device uchar* weight [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant QuantGetRowParams& params [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
  if (index >= params.cols) {
    return;
  }
  const uint block_size = 176U;
  const uint block = index / 256U;
  const uint within = index % 256U;
  const uint segment = within / 64U;
  const uint in_segment = within % 64U;
  const uint scale_index = segment * 2U + (in_segment >= 32U ? 1U : 0U);
  const device uchar* q5 =
    weight + (params.row * params.blocks_per_row + block) * block_size;
  const float d = f16_to_float(q5);
  const float dmin = f16_to_float(q5 + 2U);
  const device uchar* scales = q5 + 4U;
  const device uchar* qh = q5 + 16U;
  const device uchar* qs = q5 + 48U + segment * 32U;
  const uint packed_scale = q4_k_scale_min(scale_index, scales);
  const float ds = d * static_cast<float>(packed_scale & 0xFFU);
  const float ms = dmin * static_cast<float>((packed_scale >> 8U) & 0xFFU);
  const uint offset = in_segment % 32U;
  const uint packed = qs[offset];
  const uint high_mask = in_segment < 32U ? (1U << (2U * segment)) :
                                           (2U << (2U * segment));
  const uint high = (qh[offset] & high_mask) != 0U ? 16U : 0U;
  const uint q = (in_segment < 32U ? (packed & 0xFU) : (packed >> 4U)) + high;
  output[index] = ds * static_cast<float>(q) - ms;
}

kernel void q6_k_get_row(const device uchar* weight [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant QuantGetRowParams& params [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
  if (index >= params.cols) {
    return;
  }
  const uint block_size = 210U;
  const uint block = index / 256U;
  const uint within = index % 256U;
  const uint chunk = within / 128U;
  const uint in_chunk = within % 128U;
  const device uchar* q6 =
    weight + (params.row * params.blocks_per_row + block) * block_size;
  const device uchar* ql = q6 + chunk * 64U;
  const device uchar* qh = q6 + 128U + chunk * 32U;
  const device char* scales = reinterpret_cast<const device char*>(q6 + 192U + chunk * 8U);
  const float d = f16_to_float(q6 + 208U);

  uint offset = 0U;
  uint scale_index = 0U;
  uint q = 0U;
  if (in_chunk < 32U) {
    offset = in_chunk;
    scale_index = offset / 16U;
    q = (ql[offset] & 0xFU) | (((qh[offset] >> 0U) & 3U) << 4U);
  } else if (in_chunk < 64U) {
    offset = in_chunk - 32U;
    scale_index = offset / 16U + 2U;
    q = (ql[offset + 32U] & 0xFU) | (((qh[offset] >> 2U) & 3U) << 4U);
  } else if (in_chunk < 96U) {
    offset = in_chunk - 64U;
    scale_index = offset / 16U + 4U;
    q = (ql[offset] >> 4U) | (((qh[offset] >> 4U) & 3U) << 4U);
  } else {
    offset = in_chunk - 96U;
    scale_index = offset / 16U + 6U;
    q = (ql[offset + 32U] >> 4U) | (((qh[offset] >> 6U) & 3U) << 4U);
  }

  output[index] = d * static_cast<float>(scales[scale_index]) *
                  static_cast<float>(static_cast<int>(q) - 32);
}

)metal";

constexpr const char* kKernelSourcePart1 = R"metal(
kernel void q4_k_get_rows(const device uchar* weight [[buffer(0)]],
                          const device int* row_ids [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant QuantGetRowsParams& params [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
  const uint total = params.tokens * params.cols;
  if (index >= total) {
    return;
  }
  const uint token = index / params.cols;
  const uint dim = index % params.cols;
  const int row_i32 = row_ids[token];
  if (row_i32 < 0 || static_cast<uint>(row_i32) >= params.rows) {
    output[index] = 0.0f;
    return;
  }

  const uint block_size = 144U;
  const uint block = dim / 256U;
  const uint within = dim % 256U;
  const uint segment = within / 64U;
  const uint in_segment = within % 64U;
  const uint scale_index = segment * 2U + (in_segment >= 32U ? 1U : 0U);
  const uint row = static_cast<uint>(row_i32);
  const device uchar* q4 = weight + (row * params.blocks_per_row + block) * block_size;
  const float d = f16_to_float(q4);
  const float dmin = f16_to_float(q4 + 2U);
  const device uchar* scales = q4 + 4U;
  const device uchar* qs = q4 + 16U + segment * 32U;
  const uint packed_scale = q4_k_scale_min(scale_index, scales);
  const float ds = d * static_cast<float>(packed_scale & 0xFFU);
  const float ms = dmin * static_cast<float>((packed_scale >> 8U) & 0xFFU);
  const uint packed = qs[in_segment % 32U];
  const uint q = in_segment < 32U ? (packed & 0xFU) : (packed >> 4U);
  output[index] = ds * static_cast<float>(q) - ms;
}

kernel void q5_k_get_rows(const device uchar* weight [[buffer(0)]],
                          const device int* row_ids [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant QuantGetRowsParams& params [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
  const uint total = params.tokens * params.cols;
  if (index >= total) {
    return;
  }
  const uint token = index / params.cols;
  const uint dim = index % params.cols;
  const int row_i32 = row_ids[token];
  if (row_i32 < 0 || static_cast<uint>(row_i32) >= params.rows) {
    output[index] = 0.0f;
    return;
  }

  const uint block_size = 176U;
  const uint block = dim / 256U;
  const uint within = dim % 256U;
  const uint segment = within / 64U;
  const uint in_segment = within % 64U;
  const uint scale_index = segment * 2U + (in_segment >= 32U ? 1U : 0U);
  const uint row = static_cast<uint>(row_i32);
  const device uchar* q5 = weight + (row * params.blocks_per_row + block) * block_size;
  const float d = f16_to_float(q5);
  const float dmin = f16_to_float(q5 + 2U);
  const device uchar* scales = q5 + 4U;
  const device uchar* qh = q5 + 16U;
  const device uchar* qs = q5 + 48U + segment * 32U;
  const uint packed_scale = q4_k_scale_min(scale_index, scales);
  const float ds = d * static_cast<float>(packed_scale & 0xFFU);
  const float ms = dmin * static_cast<float>((packed_scale >> 8U) & 0xFFU);
  const uint offset = in_segment % 32U;
  const uint packed = qs[offset];
  const uint high_mask = in_segment < 32U ? (1U << (2U * segment)) :
                                           (2U << (2U * segment));
  const uint high = (qh[offset] & high_mask) != 0U ? 16U : 0U;
  const uint q = (in_segment < 32U ? (packed & 0xFU) : (packed >> 4U)) + high;
  output[index] = ds * static_cast<float>(q) - ms;
}

kernel void q6_k_get_rows(const device uchar* weight [[buffer(0)]],
                          const device int* row_ids [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant QuantGetRowsParams& params [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
  const uint total = params.tokens * params.cols;
  if (index >= total) {
    return;
  }
  const uint token = index / params.cols;
  const uint dim = index % params.cols;
  const int row_i32 = row_ids[token];
  if (row_i32 < 0 || static_cast<uint>(row_i32) >= params.rows) {
    output[index] = 0.0f;
    return;
  }

  const uint block_size = 210U;
  const uint block = dim / 256U;
  const uint within = dim % 256U;
  const uint chunk = within / 128U;
  const uint in_chunk = within % 128U;
  const uint row = static_cast<uint>(row_i32);
  const device uchar* q6 = weight + (row * params.blocks_per_row + block) * block_size;
  const device uchar* ql = q6 + chunk * 64U;
  const device uchar* qh = q6 + 128U + chunk * 32U;
  const device char* scales = reinterpret_cast<const device char*>(q6 + 192U + chunk * 8U);
  const float d = f16_to_float(q6 + 208U);

  uint offset = 0U;
  uint scale_index = 0U;
  uint q = 0U;
  if (in_chunk < 32U) {
    offset = in_chunk;
    scale_index = offset / 16U;
    q = (ql[offset] & 0xFU) | (((qh[offset] >> 0U) & 3U) << 4U);
  } else if (in_chunk < 64U) {
    offset = in_chunk - 32U;
    scale_index = offset / 16U + 2U;
    q = (ql[offset + 32U] & 0xFU) | (((qh[offset] >> 2U) & 3U) << 4U);
  } else if (in_chunk < 96U) {
    offset = in_chunk - 64U;
    scale_index = offset / 16U + 4U;
    q = (ql[offset] >> 4U) | (((qh[offset] >> 4U) & 3U) << 4U);
  } else {
    offset = in_chunk - 96U;
    scale_index = offset / 16U + 6U;
    q = (ql[offset + 32U] >> 4U) | (((qh[offset] >> 6U) & 3U) << 4U);
  }

  output[index] = d * static_cast<float>(scales[scale_index]) *
                  static_cast<float>(static_cast<int>(q) - 32);
}

kernel void embedding_bf16_f32(const device ushort* weight [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant EmbeddingParams& params [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  if (index >= params.hidden_size) {
    return;
  }
  output[index] = bf16_to_float(weight[params.token * params.hidden_size + index]);
}

kernel void rms_norm_f32_bf16(const device float* input [[buffer(0)]],
                              const device ushort* weight [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant RmsNormParams& params [[buffer(3)]],
                              uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  float sum = 0.0f;
  for (uint index = lane; index < params.size; index += params.threads) {
    const float value = input[index];
    sum += value * value;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale = rsqrt(partials[0] / static_cast<float>(params.size) + params.eps);
  for (uint index = lane; index < params.size; index += params.threads) {
    output[index] = input[index] * scale * bf16_to_float(weight[index]);
  }
}

kernel void rms_norm_f32_f32(const device float* input [[buffer(0)]],
                             const device float* weight [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant RmsNormParams& params [[buffer(3)]],
                             uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  float sum = 0.0f;
  for (uint index = lane; index < params.size; index += params.threads) {
    const float value = input[index];
    sum += value * value;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale = rsqrt(partials[0] / static_cast<float>(params.size) + params.eps);
  for (uint index = lane; index < params.size; index += params.threads) {
    output[index] = input[index] * scale * weight[index];
  }
}

kernel void rms_norm_f32_f32_batched(const device float* input [[buffer(0)]],
                                     const device float* weight [[buffer(1)]],
                                     device float* output [[buffer(2)]],
                                     constant BatchedRmsNormParams& params [[buffer(3)]],
                                     uint token [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  if (token >= params.tokens) {
    return;
  }

  const uint base = token * params.size;
  float sum = 0.0f;
  for (uint index = lane; index < params.size; index += params.threads) {
    const float value = input[base + index];
    sum += value * value;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale = rsqrt(partials[0] / static_cast<float>(params.size) + params.eps);
  for (uint index = lane; index < params.size; index += params.threads) {
    output[base + index] = input[base + index] * scale * weight[index];
  }
}

kernel void qk_norm_f32_bf16(device float* values [[buffer(0)]],
                             const device ushort* weight [[buffer(1)]],
                             constant QkNormParams& params [[buffer(2)]],
                             uint3 group_id [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint head = group_id.x;
  if (head >= params.heads) {
    return;
  }

  const uint base = head * params.head_dim;
  float sum = 0.0f;
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    const float value = values[base + dim];
    sum += value * value;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale =
    rsqrt(partials[0] / static_cast<float>(params.head_dim) + params.eps);
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    values[base + dim] *= scale * bf16_to_float(weight[dim]);
  }
}

kernel void qk_norm_f32_f32(device float* values [[buffer(0)]],
                            const device float* weight [[buffer(1)]],
                            constant QkNormParams& params [[buffer(2)]],
                            uint3 group_id [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint head = group_id.x;
  if (head >= params.heads) {
    return;
  }

  const uint base = head * params.head_dim;
  float sum = 0.0f;
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    const float value = values[base + dim];
    sum += value * value;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale =
    rsqrt(partials[0] / static_cast<float>(params.head_dim) + params.eps);
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    values[base + dim] *= scale * weight[dim];
  }
}

kernel void qk_norm_f32_f32_batched(device float* values [[buffer(0)]],
                                    const device float* weight [[buffer(1)]],
                                    constant BatchedQkNormParams& params [[buffer(2)]],
                                    uint3 group_id [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint head = group_id.x;
  const uint token = group_id.y;
  if (head >= params.heads || token >= params.tokens) {
    return;
  }

  const uint base = (token * params.heads + head) * params.head_dim;
  float sum = 0.0f;
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    const float value = values[base + dim];
    sum += value * value;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale =
    rsqrt(partials[0] / static_cast<float>(params.head_dim) + params.eps);
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    values[base + dim] *= scale * weight[dim];
  }
}

kernel void qwen35_norm_gated_f32_in_place(device float* values [[buffer(0)]],
                                           const device float* weight [[buffer(1)]],
                                           const device float* gate [[buffer(2)]],
                                           constant BatchedQkNormParams& params [[buffer(3)]],
                                           uint3 group_id [[threadgroup_position_in_grid]],
                                           uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint head = group_id.x;
  const uint token = group_id.y;
  if (head >= params.heads || token >= params.tokens) {
    return;
  }

  const uint base = (token * params.heads + head) * params.head_dim;
  float sum = 0.0f;
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    const float value = values[base + dim];
    sum += value * value;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale =
    rsqrt(partials[0] / static_cast<float>(params.head_dim) + params.eps);
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    const uint index = base + dim;
    const float gate_value = gate[index];
    const float gated_silu = gate_value / (1.0f + exp(-gate_value));
    values[index] = values[index] * scale * weight[dim] * gated_silu;
  }
}

kernel void l2_norm_f32_in_place(device float* values [[buffer(0)]],
                                 constant L2NormParams& params [[buffer(1)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint row = group_id.x;
  if (row >= params.rows) {
    return;
  }

  const uint base = row * params.row_size;
  float sum = 0.0f;
  for (uint dim = lane; dim < params.row_size; dim += params.threads) {
    const float value = values[base + dim];
    sum += value * value;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale = 1.0f / max(sqrt(partials[0]), params.eps);
  for (uint dim = lane; dim < params.row_size; dim += params.threads) {
    values[base + dim] *= scale;
  }
}

kernel void split_qkv_l2_norm_f32_qwen35(const device float* source [[buffer(0)]],
                                         device float* query [[buffer(1)]],
                                         device float* key [[buffer(2)]],
                                         device float* value [[buffer(3)]],
                                         constant SplitQkvL2NormParams& params [[buffer(4)]],
                                         uint3 group_id [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partials[256];
  const uint head = group_id.x;
  const uint token = group_id.y;
  const uint part = group_id.z;
  if (token >= params.tokens) {
    return;
  }

  const uint key_dim = params.key_heads * params.head_dim;
  if (part == 2U) {
    if (head >= params.value_heads) {
      return;
    }
    const uint source_base =
      token * params.conv_channels + 2U * key_dim + head * params.head_dim;
    const uint destination_base =
      (token * params.value_heads + head) * params.head_dim;
    for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
      value[destination_base + dim] = source[source_base + dim];
    }
    return;
  }

  if (head >= params.key_heads) {
    return;
  }
  const uint source_base =
    token * params.conv_channels + part * key_dim + head * params.head_dim;
  float sum = 0.0f;
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    const float element = source[source_base + dim];
    sum += element * element;
  }
  partials[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partials[lane] += partials[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float scale = 1.0f / max(sqrt(partials[0]), params.eps);
  device float* destination = part == 0U ? query : key;
  const uint destination_base = (token * params.key_heads + head) * params.head_dim;
  for (uint dim = lane; dim < params.head_dim; dim += params.threads) {
    destination[destination_base + dim] = source[source_base + dim] * scale;
  }
}

kernel void gated_delta_net_f32_in_place(const device float* query [[buffer(0)]],
                                         const device float* key [[buffer(1)]],
                                         const device float* value [[buffer(2)]],
                                         const device float* gate [[buffer(3)]],
                                         const device float* beta [[buffer(4)]],
                                         device float* state [[buffer(5)]],
                                         device float* output [[buffer(6)]],
                                         constant GatedDeltaNetParams& params [[buffer(7)]],
                                         uint3 group_id [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partial_sk[256];
  threadgroup float partial_y[256];
  const uint row = group_id.x;
  const uint head = group_id.y;
  if (row >= params.head_dim || head >= params.value_heads) {
    return;
  }

  const uint key_head = head % params.key_heads;
  const uint state_base = head * params.head_dim * params.head_dim +
                          row * params.head_dim;
  const uint qk_base = key_head * params.head_dim;
  const float gate_exp = exp(gate[head]);

  float k_value = 0.0f;
  float q_value = 0.0f;
  float sk = 0.0f;
  if (lane < params.head_dim) {
    const uint dim = lane;
    const uint state_index = state_base + dim;
    const float state_value = state[state_index] * gate_exp;
    state[state_index] = state_value;
    k_value = key[qk_base + dim];
    q_value = query[qk_base + dim];
    sk = state_value * k_value;
  }
  partial_sk[lane] = sk;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partial_sk[lane] += partial_sk[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float delta =
    (value[head * params.head_dim + row] - partial_sk[0]) * beta[head];
  float y = 0.0f;
  if (lane < params.head_dim) {
    const uint state_index = state_base + lane;
    const float updated = state[state_index] + k_value * delta;
    state[state_index] = updated;
    y = updated * q_value;
  }
  partial_y[lane] = y;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      partial_y[lane] += partial_y[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    output[head * params.head_dim + row] = partial_y[0] * params.scale;
  }
}

kernel void gated_delta_net_f32_batched_in_place(
    const device float* query [[buffer(0)]],
    const device float* key [[buffer(1)]],
    const device float* value [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    const device float* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant BatchedGatedDeltaNetParams& params [[buffer(7)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partial_sk[256];
  threadgroup float partial_y[256];
  const uint row = group_id.x;
  const uint head = group_id.y;
  if (row >= params.head_dim || head >= params.value_heads) {
    return;
  }

  const uint key_head = head % params.key_heads;
  const uint state_base = head * params.head_dim * params.head_dim +
                          row * params.head_dim;
  float state_value = 0.0f;
  if (lane < params.head_dim) {
    state_value = state[state_base + lane];
  }

  for (uint token = 0U; token < params.tokens; ++token) {
    const uint qk_base =
      token * params.key_heads * params.head_dim + key_head * params.head_dim;
    const uint value_base =
      token * params.value_heads * params.head_dim + head * params.head_dim;
    const uint gate_base = token * params.value_heads + head;
    const float gate_exp = exp(gate[gate_base]);

    float k_value = 0.0f;
    float q_value = 0.0f;
    float sk = 0.0f;
    if (lane < params.head_dim) {
      state_value *= gate_exp;
      k_value = key[qk_base + lane];
      q_value = query[qk_base + lane];
      sk = state_value * k_value;
    }
    partial_sk[lane] = sk;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
      if (lane < stride) {
        partial_sk[lane] += partial_sk[lane + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float delta =
      (value[value_base + row] - partial_sk[0]) * beta[gate_base];
    float y = 0.0f;
    if (lane < params.head_dim) {
      state_value += k_value * delta;
      y = state_value * q_value;
    }
    partial_y[lane] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
      if (lane < stride) {
        partial_y[lane] += partial_y[lane + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0U) {
      output[value_base + row] = partial_y[0] * params.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane < params.head_dim) {
    state[state_base + lane] = state_value;
  }
}

kernel void gated_delta_net_f32_batched_qwen35(
    const device float* query [[buffer(0)]],
    const device float* key [[buffer(1)]],
    const device float* value [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    const device float* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant BatchedGatedDeltaNetParams& params [[buffer(7)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_pos [[thread_position_in_threadgroup]]) {
  constexpr uint nsg = 4U;
  const uint lane = thread_pos.x;
  const uint row = group_id.x * nsg + thread_pos.y;
  const uint head = group_id.y;
  if (lane >= 32U || thread_pos.y >= nsg ||
      row >= params.head_dim || head >= params.value_heads) {
    return;
  }

  const uint key_head = head % params.key_heads;
  const uint state_base = head * params.head_dim * params.head_dim +
                          row * params.head_dim;
  const uint state_col = lane * nsg;
  float ls0 = state[state_base + state_col + 0U];
  float ls1 = state[state_base + state_col + 1U];
  float ls2 = state[state_base + state_col + 2U];
  float ls3 = state[state_base + state_col + 3U];

  for (uint token = 0U; token < params.tokens; ++token) {
    const uint qk_base =
      token * params.key_heads * params.head_dim + key_head * params.head_dim;
    const uint value_base =
      token * params.value_heads * params.head_dim + head * params.head_dim;
    const uint gate_base = token * params.value_heads + head;
    const float gate_exp = exp(gate[gate_base]);
    ls0 *= gate_exp;
    ls1 *= gate_exp;
    ls2 *= gate_exp;
    ls3 *= gate_exp;

    const float k0 = key[qk_base + state_col + 0U];
    const float k1 = key[qk_base + state_col + 1U];
    const float k2 = key[qk_base + state_col + 2U];
    const float k3 = key[qk_base + state_col + 3U];
    const float sk = simd_sum(ls0 * k0 + ls1 * k1 + ls2 * k2 + ls3 * k3);
    const float delta = (value[value_base + row] - sk) * beta[gate_base];

    ls0 += k0 * delta;
    ls1 += k1 * delta;
    ls2 += k2 * delta;
    ls3 += k3 * delta;

    const float q0 = query[qk_base + state_col + 0U];
    const float q1 = query[qk_base + state_col + 1U];
    const float q2 = query[qk_base + state_col + 2U];
    const float q3 = query[qk_base + state_col + 3U];
    const float y = simd_sum(ls0 * q0 + ls1 * q1 + ls2 * q2 + ls3 * q3);
    if (lane == 0U) {
      output[value_base + row] = y * params.scale;
    }
  }

  state[state_base + state_col + 0U] = ls0;
  state[state_base + state_col + 1U] = ls1;
  state[state_base + state_col + 2U] = ls2;
  state[state_base + state_col + 3U] = ls3;
}

kernel void ssm_conv_f32(const device float* input [[buffer(0)]],
                         const device float* weights [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         constant SsmConvParams& params [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
  if (index >= params.total) {
    return;
  }
  const uint channel = index % params.channels;
  const uint token = (index / params.channels) % params.tokens;
  const uint sequence = index / (params.channels * params.tokens);
  if (sequence >= params.sequences) {
    return;
  }

  const uint input_sequence_offset = sequence * params.channels * params.input_span;
  const uint input_channel_offset = input_sequence_offset + channel * params.input_span;
  const uint kernel_channel_offset = channel * params.conv_kernel;
  float sum = 0.0f;
  for (uint conv = 0; conv < params.conv_kernel; ++conv) {
    sum += input[input_channel_offset + token + conv] *
           weights[kernel_channel_offset + conv];
  }
  output[index] = sum;
}

kernel void ssm_conv1_f32_stateful(device float* state [[buffer(0)]],
                                   const device float* input [[buffer(1)]],
                                   const device float* weights [[buffer(2)]],
                                   device float* output [[buffer(3)]],
                                   constant SsmConv1StatefulParams& params [[buffer(4)]],
                                   uint channel [[thread_position_in_grid]]) {
  if (channel >= params.channels || params.conv_kernel < 2U) {
    return;
  }
  const uint state_width = params.conv_kernel - 1U;
  const uint state_base = channel * state_width;
  const uint weight_base = channel * params.conv_kernel;
  const float current = input[channel];
  float sum = 0.0f;
  for (uint index = 0; index < state_width; ++index) {
    sum += state[state_base + index] * weights[weight_base + index];
  }
  sum += current * weights[weight_base + state_width];

  for (uint index = 0; index + 1U < state_width; ++index) {
    state[state_base + index] = state[state_base + index + 1U];
  }
  state[state_base + state_width - 1U] = current;
  output[channel] = sum;
}

kernel void build_ssm_conv_state_f32(device float* state [[buffer(0)]],
                                     const device float* input [[buffer(1)]],
                                     device float* conv_input [[buffer(2)]],
                                     constant BuildSsmConvStateParams& params [[buffer(3)]],
                                     uint channel [[thread_position_in_grid]]) {
  if (channel >= params.channels || params.conv_kernel < 2U) {
    return;
  }

  const uint state_width = params.conv_kernel - 1U;
  const uint state_base = channel * state_width;
  const uint conv_base = channel * params.input_span;
  for (uint position = 0U; position < state_width; ++position) {
    conv_input[conv_base + position] = state[state_base + position];
  }
  for (uint token = 0U; token < params.tokens; ++token) {
    conv_input[conv_base + state_width + token] = input[token * params.channels + channel];
  }
  for (uint position = 0U; position < state_width; ++position) {
    const uint source_position = params.tokens + position;
    float updated = 0.0f;
    if (source_position < state_width) {
      updated = state[state_base + source_position];
    } else {
      const uint token = source_position - state_width;
      updated = input[token * params.channels + channel];
    }
    state[state_base + position] = updated;
  }
}

kernel void rope_f32(device float* values [[buffer(0)]],
                     constant RopeParams& params [[buffer(1)]],
                     uint index [[thread_position_in_grid]]) {
  const uint half_dim = params.head_dim >> 1U;
  const uint total = params.heads * half_dim;
  if (index >= total) {
    return;
  }

  const uint head = index / half_dim;
  const uint dim = index % half_dim;
  const uint base = head * params.head_dim;
  const float exponent = static_cast<float>(2U * dim) / static_cast<float>(params.head_dim);
  const float freq = 1.0f / pow(params.theta, exponent);
  const float angle = static_cast<float>(params.position) * freq;
  const float cos_value = cos(angle);
  const float sin_value = sin(angle);
  const float x0 = values[base + dim];
  const float x1 = values[base + half_dim + dim];
  values[base + dim] = x0 * cos_value - x1 * sin_value;
  values[base + half_dim + dim] = x1 * cos_value + x0 * sin_value;
}

kernel void mrope_f32_in_place(device float* values [[buffer(0)]],
                               const device int* positions [[buffer(1)]],
                               constant MropeParams& params [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  const uint half_dims = params.n_dims / 2U;
  const uint total = params.tokens * params.heads * half_dims;
  if (index >= total) {
    return;
  }

  const uint pair_index = index % half_dims;
  const uint head = (index / half_dims) % params.heads;
  const uint token = index / (half_dims * params.heads);
  const uint section_dims =
    params.section_0 + params.section_1 + params.section_2 + params.section_3;
  if (section_dims == 0U) {
    return;
  }

  const uint sector = pair_index % section_dims;
  const uint section_01 = params.section_0 + params.section_1;
  const uint section_012 = section_01 + params.section_2;
  uint position_index = 0U;
  if (sector < params.section_0) {
    position_index = 0U;
  } else if (sector < section_01) {
    position_index = 1U;
  } else if (sector < section_012) {
    position_index = 2U;
  } else {
    position_index = 3U;
  }

  const float position = static_cast<float>(positions[token + params.tokens * position_index]);
  const float exponent =
    static_cast<float>(2U * pair_index) / static_cast<float>(params.n_dims);
  const float freq = 1.0f / pow(params.theta, exponent);
  const float angle = position * freq;
  const float cos_value = cos(angle);
  const float sin_value = sin(angle);
  const uint base = token * params.heads * params.head_dim + head * params.head_dim;
  const float x0 = values[base + pair_index];
  const float x1 = values[base + half_dims + pair_index];
  values[base + pair_index] = x0 * cos_value - x1 * sin_value;
  values[base + half_dims + pair_index] = x1 * cos_value + x0 * sin_value;
}

kernel void add_f32_in_place(device float* target [[buffer(0)]],
                             const device float* delta [[buffer(1)]],
                             constant SizeParams& params [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
  if (index >= params.size) {
    return;
  }
  target[index] += delta[index];
}

kernel void mul_f32_in_place(device float* target [[buffer(0)]],
                             const device float* rhs [[buffer(1)]],
                             constant SizeParams& params [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
  if (index >= params.size) {
    return;
  }
  target[index] *= rhs[index];
}

kernel void add_f32_row_in_place(device float* target [[buffer(0)]],
                                 const device float* row [[buffer(1)]],
                                 constant RowwiseParams& params [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= params.total) {
    return;
  }
  target[index] += row[index % params.row_size];
}

kernel void mul_f32_row_in_place(device float* target [[buffer(0)]],
                                 const device float* row [[buffer(1)]],
                                 constant RowwiseParams& params [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= params.total) {
    return;
  }
  target[index] *= row[index % params.row_size];
}

kernel void sigmoid_f32_in_place(device float* values [[buffer(0)]],
                                 constant SizeParams& params [[buffer(1)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= params.size) {
    return;
  }
  values[index] = 1.0f / (1.0f + exp(-values[index]));
}

kernel void softplus_f32_in_place(device float* values [[buffer(0)]],
                                  constant SizeParams& params [[buffer(1)]],
                                  uint index [[thread_position_in_grid]]) {
  if (index >= params.size) {
    return;
  }
  const float value = values[index];
  values[index] = value > 20.0f ? value : log(1.0f + exp(value));
}

kernel void prepare_qwen35_gdn_gate_beta_f32(
    device float* gate [[buffer(0)]],
    device float* beta [[buffer(1)]],
    const device float* gate_bias [[buffer(2)]],
    const device float* gate_scale [[buffer(3)]],
    constant RowwiseParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.total) {
    return;
  }
  const uint column = index % params.row_size;
  const float gate_value = gate[index] + gate_bias[column];
  gate[index] =
    (gate_value > 20.0f ? gate_value : log(1.0f + exp(gate_value))) *
    gate_scale[column];
  const float beta_value = beta[index];
  beta[index] = 1.0f / (1.0f + exp(-beta_value));
}

)metal";

constexpr const char* kKernelSourcePart2 = R"metal(
kernel void silu_f32_in_place(device float* values [[buffer(0)]],
                              constant SizeParams& params [[buffer(1)]],
                              uint index [[thread_position_in_grid]]) {
  if (index >= params.size) {
    return;
  }
  const float value = values[index];
  values[index] = value / (1.0f + exp(-value));
}

kernel void silu_mul_f32_in_place(device float* gate [[buffer(0)]],
                                  const device float* up [[buffer(1)]],
                                  constant SizeParams& params [[buffer(2)]],
                                  uint index [[thread_position_in_grid]]) {
  if (index >= params.size) {
    return;
  }
  const float value = gate[index];
  gate[index] = (value / (1.0f + exp(-value))) * up[index];
}

kernel void copy_f32_region(const device float* source [[buffer(0)]],
                            device float* destination [[buffer(1)]],
                            constant CopyRegionParams& params [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  if (index >= params.size) {
    return;
  }
  destination[params.destination_offset + index] =
    source[params.source_offset + index];
}

kernel void copy_f32_rows(const device float* source [[buffer(0)]],
                          device float* destination [[buffer(1)]],
                          constant CopyRowsParams& params [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
  if (index >= params.total) {
    return;
  }
  const uint row = index / params.row_size;
  const uint column = index % params.row_size;
  destination[params.destination_offset + row * params.destination_stride + column] =
    source[params.source_offset + row * params.source_stride + column];
}

kernel void copy_f32_rows_to_f16(const device float* source [[buffer(0)]],
                                 device half* destination [[buffer(1)]],
                                 constant CopyRowsParams& params [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= params.total) {
    return;
  }
  const uint row = index / params.row_size;
  const uint column = index % params.row_size;
  destination[params.destination_offset + row * params.destination_stride + column] =
    half(source[params.source_offset + row * params.source_stride + column]);
}

kernel void copy_f16_rows(const device half* source [[buffer(0)]],
                          device half* destination [[buffer(1)]],
                          constant CopyRowsParams& params [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
  if (index >= params.total) {
    return;
  }
  const uint row = index / params.row_size;
  const uint column = index % params.row_size;
  destination[params.destination_offset + row * params.destination_stride + column] =
    source[params.source_offset + row * params.source_stride + column];
}

kernel void argmax_f32_i32(const device float* values [[buffer(0)]],
                           device int* output [[buffer(1)]],
                           constant SizeParams& params [[buffer(2)]],
                           uint lane [[thread_index_in_threadgroup]]) {
  if (params.size == 0U || lane >= params.threads) {
    return;
  }
  threadgroup float best_values[256];
  threadgroup uint best_indices[256];

  uint best_index = params.size;
  float best_value = -INFINITY;
  for (uint i = lane; i < params.size; i += params.threads) {
    const float value = values[i];
    if (value > best_value || (value == best_value && i < best_index)) {
      best_value = value;
      best_index = i;
    }
  }
  best_values[lane] = best_value;
  best_indices[lane] = best_index;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      const float rhs_value = best_values[lane + stride];
      const uint rhs_index = best_indices[lane + stride];
      if (rhs_value > best_values[lane] ||
          (rhs_value == best_values[lane] && rhs_index < best_indices[lane])) {
        best_values[lane] = rhs_value;
        best_indices[lane] = rhs_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    output[0] = static_cast<int>(best_indices[0]);
  }
}

kernel void argmax_prob_f32_i32(const device float* values [[buffer(0)]],
                                device int* token_output [[buffer(1)]],
                                device float* probability_output [[buffer(2)]],
                                constant SizeParams& params [[buffer(3)]],
                                uint lane [[thread_index_in_threadgroup]]) {
  if (params.size == 0U || lane >= params.threads) {
    return;
  }
  threadgroup float best_values[256];
  threadgroup uint best_indices[256];
  threadgroup float sums[256];

  uint best_index = params.size;
  float best_value = -INFINITY;
  for (uint i = lane; i < params.size; i += params.threads) {
    const float value = values[i];
    if (value > best_value || (value == best_value && i < best_index)) {
      best_value = value;
      best_index = i;
    }
  }
  best_values[lane] = best_value;
  best_indices[lane] = best_index;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      const float rhs_value = best_values[lane + stride];
      const uint rhs_index = best_indices[lane + stride];
      if (rhs_value > best_values[lane] ||
          (rhs_value == best_values[lane] && rhs_index < best_indices[lane])) {
        best_values[lane] = rhs_value;
        best_indices[lane] = rhs_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float max_value = best_values[0];
  float sum = 0.0f;
  for (uint i = lane; i < params.size; i += params.threads) {
    sum += exp(values[i] - max_value);
  }
  sums[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.threads >> 1U; stride > 0U; stride >>= 1U) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0U) {
    token_output[0] = static_cast<int>(best_indices[0]);
    probability_output[0] = sums[0] > 0.0f ? 1.0f / sums[0] : 0.0f;
  }
}

kernel void attention_f32(const device float* query [[buffer(0)]],
                          const device float* key_cache [[buffer(1)]],
                          const device float* value_cache [[buffer(2)]],
                          device float* output [[buffer(3)]],
                          constant AttentionParams& params [[buffer(4)]],
                          uint index [[thread_position_in_grid]]) {
  const uint head_dim = params.head_dim;
  const uint total = params.heads * head_dim;
  if (index >= total) {
    return;
  }

  const uint head = index / head_dim;
  const uint dim = index % head_dim;
  const uint kv_head = head / params.group;
  const uint kv_dim = params.kv_heads * head_dim;
  float max_score = -INFINITY;

  for (uint token = 0; token <= params.position; ++token) {
    float score = 0.0f;
    const uint query_base = head * head_dim;
    const uint cache_base =
      ((params.layer * params.capacity_tokens + token) * kv_dim) +
      kv_head * head_dim;
    for (uint value_dim = 0; value_dim < head_dim; ++value_dim) {
      score += query[query_base + value_dim] * key_cache[cache_base + value_dim];
    }
    score *= params.scale;
    max_score = max(max_score, score);
  }

  float denom = 0.0f;
  float value = 0.0f;
  for (uint token = 0; token <= params.position; ++token) {
    float score = 0.0f;
    const uint query_base = head * head_dim;
    const uint cache_base =
      ((params.layer * params.capacity_tokens + token) * kv_dim) +
      kv_head * head_dim;
    for (uint value_dim = 0; value_dim < head_dim; ++value_dim) {
      score += query[query_base + value_dim] * key_cache[cache_base + value_dim];
    }
    const float weight = exp(score * params.scale - max_score);
    denom += weight;
    value += weight * value_cache[cache_base + dim];
  }
  output[index] = value / denom;
}

kernel void attention_f32_f16_kv(const device float* query [[buffer(0)]],
                                 const device half* key_cache [[buffer(1)]],
                                 const device half* value_cache [[buffer(2)]],
                                 device float* output [[buffer(3)]],
                                 constant AttentionParams& params [[buffer(4)]],
                                 uint index [[thread_position_in_grid]]) {
  const uint head = index / params.head_dim;
  const uint dim = index % params.head_dim;
  if (head >= params.heads) {
    return;
  }

  const uint head_dim = params.head_dim;
  const uint kv_head = head / params.group;
  const uint kv_dim = params.kv_heads * head_dim;
  float max_score = -INFINITY;
  for (uint token = 0; token <= params.position; ++token) {
    float score = 0.0f;
    const uint query_base = head * head_dim;
    const uint cache_base =
      ((params.layer * params.capacity_tokens + token) * kv_dim) +
      kv_head * head_dim;
    for (uint value_dim = 0; value_dim < head_dim; ++value_dim) {
      score += query[query_base + value_dim] *
               float(key_cache[cache_base + value_dim]);
    }
    score *= params.scale;
    max_score = max(max_score, score);
  }

  float denom = 0.0f;
  float value = 0.0f;
  for (uint token = 0; token <= params.position; ++token) {
    float score = 0.0f;
    const uint query_base = head * head_dim;
    const uint cache_base =
      ((params.layer * params.capacity_tokens + token) * kv_dim) +
      kv_head * head_dim;
    for (uint value_dim = 0; value_dim < head_dim; ++value_dim) {
      score += query[query_base + value_dim] *
               float(key_cache[cache_base + value_dim]);
    }
    const float weight = exp(score * params.scale - max_score);
    denom += weight;
    value += weight * float(value_cache[cache_base + dim]);
  }
  output[index] = value / denom;
}

kernel void attention_f32_batched(const device float* query [[buffer(0)]],
                                  const device float* key_cache [[buffer(1)]],
                                  const device float* value_cache [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant BatchedAttentionParams& params [[buffer(4)]],
                                  uint3 group_id [[threadgroup_position_in_grid]],
                                  ushort lane [[thread_index_in_simdgroup]],
                                  ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint query_tile = 8U;
  constexpr uint simd_width = 32U;
  constexpr uint max_values_per_lane = 8U;
  const uint token = group_id.x * query_tile + static_cast<uint>(simdgroup);
  const uint head = group_id.y;
  if (token >= params.tokens || head >= params.heads ||
      static_cast<uint>(simdgroup) >= query_tile) {
    return;
  }

  const uint head_dim = params.head_dim;
  const uint kv_head = head / params.group;
  const uint kv_dim = params.kv_heads * head_dim;
  const uint query_base = token * params.heads * head_dim + head * head_dim;
  const uint query_position = params.start_position + token;
  float max_score = -INFINITY;
  float denom = 0.0f;
  float value[max_values_per_lane];

  for (uint i = 0U; i < max_values_per_lane; ++i) {
    value[i] = 0.0f;
  }

  for (uint cache_token = 0U; cache_token <= query_position; ++cache_token) {
    const uint cache_base =
      ((params.layer * params.capacity_tokens + cache_token) * kv_dim) +
      kv_head * head_dim;
    float score_partial = 0.0f;
    for (uint dim = static_cast<uint>(lane); dim < head_dim; dim += simd_width) {
      score_partial += query[query_base + dim] * key_cache[cache_base + dim];
    }
    const float scaled_score = simd_sum(score_partial) * params.scale;
    const float new_max = max(max_score, scaled_score);
    const float previous_scale = exp(max_score - new_max);
    const float weight = exp(scaled_score - new_max);
    denom = denom * previous_scale + weight;

    for (uint i = 0U; i < max_values_per_lane; ++i) {
      const uint dim = static_cast<uint>(lane) + i * simd_width;
      if (dim < head_dim) {
        value[i] =
          value[i] * previous_scale + weight * value_cache[cache_base + dim];
      }
    }
    max_score = new_max;
  }

  const float denom_scale = denom == 0.0f ? 0.0f : 1.0f / denom;
  for (uint i = 0U; i < max_values_per_lane; ++i) {
    const uint dim = static_cast<uint>(lane) + i * simd_width;
    if (dim < head_dim) {
      output[query_base + dim] = value[i] * denom_scale;
    }
  }
}

kernel void attention_f32_batched_f16_kv(
                                  const device float* query [[buffer(0)]],
                                  const device half* key_cache [[buffer(1)]],
                                  const device half* value_cache [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant BatchedAttentionParams& params [[buffer(4)]],
                                  uint3 group_id [[threadgroup_position_in_grid]],
                                  ushort lane [[thread_index_in_simdgroup]],
                                  ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint query_tile = 8U;
  constexpr uint simd_width = 32U;
  constexpr uint max_values_per_lane = 8U;
  const uint token = group_id.x * query_tile + static_cast<uint>(simdgroup);
  const uint head = group_id.y;
  if (token >= params.tokens || head >= params.heads ||
      static_cast<uint>(simdgroup) >= query_tile) {
    return;
  }

  const uint head_dim = params.head_dim;
  const uint kv_head = head / params.group;
  const uint kv_dim = params.kv_heads * head_dim;
  const uint query_base = token * params.heads * head_dim + head * head_dim;
  const uint query_position = params.start_position + token;
  float max_score = -INFINITY;
  float denom = 0.0f;
  float value[max_values_per_lane];

  for (uint i = 0U; i < max_values_per_lane; ++i) {
    value[i] = 0.0f;
  }

  for (uint cache_token = 0U; cache_token <= query_position; ++cache_token) {
    const uint cache_base =
      ((params.layer * params.capacity_tokens + cache_token) * kv_dim) +
      kv_head * head_dim;
    float score_partial = 0.0f;
    for (uint dim = static_cast<uint>(lane); dim < head_dim; dim += simd_width) {
      score_partial += query[query_base + dim] *
                       float(key_cache[cache_base + dim]);
    }
    const float scaled_score = simd_sum(score_partial) * params.scale;
    const float new_max = max(max_score, scaled_score);
    const float previous_scale = exp(max_score - new_max);
    const float weight = exp(scaled_score - new_max);
    denom = denom * previous_scale + weight;

    for (uint i = 0U; i < max_values_per_lane; ++i) {
      const uint dim = static_cast<uint>(lane) + i * simd_width;
      if (dim < head_dim) {
        value[i] =
          value[i] * previous_scale +
          weight * float(value_cache[cache_base + dim]);
      }
    }
    max_score = new_max;
  }

  const float denom_scale = denom == 0.0f ? 0.0f : 1.0f / denom;
  for (uint i = 0U; i < max_values_per_lane; ++i) {
    const uint dim = static_cast<uint>(lane) + i * simd_width;
    if (dim < head_dim) {
      output[query_base + dim] = value[i] * denom_scale;
    }
  }
}

template <uint cache_tile>
kernel void attention_f32_batched_tiled_impl(
                                        const device float* query [[buffer(0)]],
                                        const device float* key_cache [[buffer(1)]],
                                        const device float* value_cache [[buffer(2)]],
                                        device float* output [[buffer(3)]],
                                        constant BatchedAttentionParams& params [[buffer(4)]],
                                        uint3 group_id [[threadgroup_position_in_grid]],
                                        ushort thread_index [[thread_index_in_threadgroup]],
                                        ushort lane [[thread_index_in_simdgroup]],
                                        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint query_tile = 8U;
  constexpr uint simd_width = 32U;
  constexpr uint max_head_dim = 256U;
  constexpr uint max_values_per_lane = max_head_dim / simd_width;
  threadgroup float key_tile[cache_tile * max_head_dim];
  threadgroup float value_tile[cache_tile * max_head_dim];

  const uint group_token_start = group_id.x * query_tile;
  const uint head = group_id.y;
  if (group_token_start >= params.tokens || head >= params.heads ||
      params.head_dim > max_head_dim) {
    return;
  }

  const uint head_dim = params.head_dim;
  const uint active_queries = min(query_tile, params.tokens - group_token_start);
  const uint max_query_position =
    params.start_position + group_token_start + active_queries - 1U;
  const uint token = group_token_start + static_cast<uint>(simdgroup);
  const bool active_query = static_cast<uint>(simdgroup) < active_queries;
  const uint kv_head = head / params.group;
  const uint kv_dim = params.kv_heads * head_dim;
  const uint query_base = token * params.heads * head_dim + head * head_dim;
  const uint query_position = params.start_position + token;

  float query_values[max_values_per_lane];
  float value[max_values_per_lane];
  for (uint i = 0U; i < max_values_per_lane; ++i) {
    const uint dim = static_cast<uint>(lane) + i * simd_width;
    query_values[i] = active_query && dim < head_dim ? query[query_base + dim] : 0.0f;
    value[i] = 0.0f;
  }

  float max_score = -INFINITY;
  float denom = 0.0f;

  for (uint cache_block = 0U; cache_block <= max_query_position;
       cache_block += cache_tile) {
    const uint tile_values = cache_tile * head_dim;
    for (uint index = static_cast<uint>(thread_index); index < tile_values;
         index += params.threads) {
      const uint local_token = index / head_dim;
      const uint dim = index - local_token * head_dim;
      const uint cache_token = cache_block + local_token;
      const uint tile_index = local_token * max_head_dim + dim;
      if (cache_token <= max_query_position) {
        const uint cache_base =
          ((params.layer * params.capacity_tokens + cache_token) * kv_dim) +
          kv_head * head_dim;
        key_tile[tile_index] = key_cache[cache_base + dim];
        value_tile[tile_index] = value_cache[cache_base + dim];
      } else {
        key_tile[tile_index] = 0.0f;
        value_tile[tile_index] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active_query) {
      for (uint local_token = 0U; local_token < cache_tile; ++local_token) {
        const uint cache_token = cache_block + local_token;
        if (cache_token > query_position) {
          break;
        }

        float score_partial = 0.0f;
        for (uint i = 0U; i < max_values_per_lane; ++i) {
          const uint dim = static_cast<uint>(lane) + i * simd_width;
          if (dim < head_dim) {
            score_partial +=
              query_values[i] * key_tile[local_token * max_head_dim + dim];
          }
        }
        const float scaled_score = simd_sum(score_partial) * params.scale;
        const float new_max = max(max_score, scaled_score);
        const float previous_scale = exp(max_score - new_max);
        const float weight = exp(scaled_score - new_max);
        denom = denom * previous_scale + weight;

        for (uint i = 0U; i < max_values_per_lane; ++i) {
          const uint dim = static_cast<uint>(lane) + i * simd_width;
          if (dim < head_dim) {
            value[i] = value[i] * previous_scale +
                       weight * value_tile[local_token * max_head_dim + dim];
          }
        }
        max_score = new_max;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (active_query) {
    const float denom_scale = denom == 0.0f ? 0.0f : 1.0f / denom;
    for (uint i = 0U; i < max_values_per_lane; ++i) {
      const uint dim = static_cast<uint>(lane) + i * simd_width;
      if (dim < head_dim) {
        output[query_base + dim] = value[i] * denom_scale;
      }
    }
  }
}

template [[host_name("attention_f32_batched_tiled_16")]]
kernel void attention_f32_batched_tiled_impl<16U>(
                                        const device float* query,
                                        const device float* key_cache,
                                        const device float* value_cache,
                                        device float* output,
                                        constant BatchedAttentionParams& params,
                                        uint3 group_id,
                                        ushort thread_index,
                                        ushort lane,
                                        ushort simdgroup);

template [[host_name("attention_f32_batched_tiled_32")]]
kernel void attention_f32_batched_tiled_impl<32U>(
                                        const device float* query,
                                        const device float* key_cache,
                                        const device float* value_cache,
                                        device float* output,
                                        constant BatchedAttentionParams& params,
                                        uint3 group_id,
                                        ushort thread_index,
                                        ushort lane,
                                        ushort simdgroup);

kernel void attention_f32_batched_flash256(
                                        const device float* query [[buffer(0)]],
                                        const device float* key_cache [[buffer(1)]],
                                        const device float* value_cache [[buffer(2)]],
                                        device float* output [[buffer(3)]],
                                        constant FlashAttention256Params& params [[buffer(4)]],
                                        const device float* key_tail [[buffer(5)]],
                                        const device float* value_tail [[buffer(6)]],
                                        threadgroup half* shmem_f16 [[threadgroup(0)]],
                                        uint3 group_id [[threadgroup_position_in_grid]],
                                        ushort lane [[thread_index_in_simdgroup]],
                                        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr short query_tile = 8;
  constexpr short cache_tile = 64;
  constexpr uint cache_tile_u = 64U;
  constexpr short head_dim = 256;
  constexpr short simd_width = 32;
  constexpr short simdgroups = 4;
  constexpr short queries_per_simdgroup = query_tile / simdgroups;
  constexpr short score_stride = 2 * cache_tile;
  constexpr short padded_value_dim = 256;
  constexpr short query_storage = head_dim + 2 * padded_value_dim;
  constexpr short q4_count = head_dim / 4;
  constexpr short q8_count = head_dim / 8;
  constexpr short value4_count = head_dim / 4;
  constexpr short padded_value4_count = padded_value_dim / 4;
  constexpr short padded_value8_count = padded_value_dim / 8;
  constexpr short output_matrices_per_simdgroup = padded_value8_count / simdgroups;

  const uint query_start = group_id.x * static_cast<uint>(query_tile);
  const uint head = group_id.y;
  if (query_start >= params.tokens || head >= params.heads) {
    return;
  }

  const uint kv_head = head / params.group;
  const uint kv_dim = params.kv_heads * static_cast<uint>(head_dim);
  const uint key_count = params.start_position + params.tokens;
  const uint padded_key_count =
    ((key_count + cache_tile_u - 1U) / cache_tile_u) * cache_tile_u;
  const uint query_head_stride = params.heads * static_cast<uint>(head_dim);
  const uint cache_layer_base = params.layer * params.capacity_tokens * kv_dim;
  const uint kv_head_base = kv_head * static_cast<uint>(head_dim);

  threadgroup half* shared_query = shmem_f16;
  threadgroup half4* shared_query4 = reinterpret_cast<threadgroup half4*>(shared_query);
  threadgroup float* shared_output =
    reinterpret_cast<threadgroup float*>(shmem_f16 + query_tile * head_dim);
  threadgroup float4* shared_output4 =
    reinterpret_cast<threadgroup float4*>(shared_output);
  threadgroup float* shared_scores =
    reinterpret_cast<threadgroup float*>(shmem_f16 + query_tile * query_storage);
  threadgroup float2* shared_scores2 =
    reinterpret_cast<threadgroup float2*>(shared_scores);

  for (short query_offset = 0; query_offset < query_tile; ++query_offset) {
    const uint token = query_start + static_cast<uint>(query_offset);
    const bool active_query = token < params.tokens;
    const device float4* query4 = reinterpret_cast<const device float4*>(
      query + token * query_head_stride + head * static_cast<uint>(head_dim));
    for (short index = static_cast<short>(lane); index < q4_count;
         index += simd_width) {
      shared_query4[query_offset * q4_count + index] =
        active_query ? half4(query4[index]) : half4(0.0h);
    }
    for (short index = static_cast<short>(lane); index < padded_value4_count;
         index += simd_width) {
      shared_output4[query_offset * padded_value4_count + index] = float4(0.0f);
    }
    for (short index = static_cast<short>(lane); index < score_stride;
         index += simd_width) {
      shared_scores[query_offset * score_stride + index] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float normalizer[queries_per_simdgroup] = {0.0f, 0.0f};
  float max_score[queries_per_simdgroup] = {-FLT_MAX / 2.0f, -FLT_MAX / 2.0f};

  for (uint cache_block = 0; cache_block < padded_key_count;
       cache_block += cache_tile_u) {
    const bool tail_block = cache_block + cache_tile_u > key_count;
    device const float* key_block = tail_block
      ? key_tail + kv_head_base
      : key_cache + cache_layer_base + cache_block * kv_dim + kv_head_base;

    for (short column_block = 0; column_block < (cache_tile / 8) / simdgroups;
         ++column_block) {
      simdgroup_float8x8 qk =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
      device const float* key_ptr =
        key_block +
        (static_cast<uint>(simdgroup) * 8U +
         static_cast<uint>(column_block) * 8U * static_cast<uint>(simdgroups)) *
          kv_dim;

      simdgroup_half8x8 query_matrix[2];
      simdgroup_float8x8 key_matrix[2];
      for (short index = 0; index < q8_count / 2; ++index) {
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_load(query_matrix[0], shared_query + 0 * 8 + 16 * index,
                       head_dim);
        simdgroup_load(query_matrix[1], shared_query + 1 * 8 + 16 * index,
                       head_dim);
        simdgroup_load(key_matrix[0], key_ptr + 0 * 8 + 16 * index, kv_dim,
                       0, true);
        simdgroup_load(key_matrix[1], key_ptr + 1 * 8 + 16 * index, kv_dim,
                       0, true);
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_multiply_accumulate(qk, query_matrix[0], key_matrix[0], qk);
        simdgroup_multiply_accumulate(qk, query_matrix[1], key_matrix[1], qk);
      }

      threadgroup float* score_ptr =
        shared_scores +
        static_cast<uint>(simdgroup) * 8U +
        static_cast<uint>(column_block) * 8U * static_cast<uint>(simdgroups);
      simdgroup_store(qk, score_ptr, score_stride, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short local_query = 0; local_query < queries_per_simdgroup;
         ++local_query) {
      const short query_offset =
        local_query * simdgroups + static_cast<short>(simdgroup);
      const uint token = query_start + static_cast<uint>(query_offset);
      const uint query_position = params.start_position + token;
      float2 scores =
        shared_scores2[query_offset * score_stride / 2 + lane] * params.scale;

      const uint cache_token0 = cache_block + static_cast<uint>(lane) * 2U;
      const uint cache_token1 = cache_token0 + 1U;
      if (token >= params.tokens || cache_token0 >= key_count ||
          cache_token0 > query_position) {
        scores[0] = -INFINITY;
      }
      if (token >= params.tokens || cache_token1 >= key_count ||
          cache_token1 > query_position) {
        scores[1] = -INFINITY;
      }

      const float previous_max = max_score[local_query];
      max_score[local_query] =
        simd_max(max(max_score[local_query], max(scores[0], scores[1])));

      const float previous_scale = exp(previous_max - max_score[local_query]);
      const float2 weights = exp(scores - max_score[local_query]);
      normalizer[local_query] =
        normalizer[local_query] * previous_scale +
        simd_sum(weights[0] + weights[1]);
      shared_scores2[query_offset * score_stride / 2 + lane] = weights;

      for (short index = static_cast<short>(lane); index < value4_count;
           index += simd_width) {
        shared_output4[query_offset * padded_value4_count + index] *=
          previous_scale;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float* value_block = tail_block
      ? value_tail + kv_head_base
      : value_cache + cache_layer_base + cache_block * kv_dim + kv_head_base;
    for (short column_pair = 0; column_pair < cache_tile / 16; ++column_pair) {
      simdgroup_float8x8 weights0;
      simdgroup_float8x8 weights1;
      simdgroup_load(weights0, shared_scores + 16 * column_pair + 0,
                     score_stride, 0, false);
      simdgroup_load(weights1, shared_scores + 16 * column_pair + 8,
                     score_stride, 0, false);

      for (short output_index = 0; output_index < output_matrices_per_simdgroup / 2;
           ++output_index) {
        simdgroup_float8x8 value_matrix[4];
        simdgroup_float8x8 output_matrix[2];
        const device float* value_ptr =
          value_block + static_cast<uint>(column_pair) * 16U * kv_dim +
          8U * static_cast<uint>(simdgroup) +
          static_cast<uint>(output_index) * 16U *
            static_cast<uint>(simdgroups);
        simdgroup_load(value_matrix[0],
                       value_ptr + 0U * static_cast<uint>(simdgroups), kv_dim,
                       0, false);
        simdgroup_load(value_matrix[1],
                       value_ptr + 8U * static_cast<uint>(simdgroups), kv_dim,
                       0, false);
        simdgroup_load(value_matrix[2],
                       value_ptr + 0U * static_cast<uint>(simdgroups) +
                         8U * kv_dim,
                       kv_dim, 0, false);
        simdgroup_load(value_matrix[3],
                       value_ptr + 8U * static_cast<uint>(simdgroups) +
                         8U * kv_dim,
                       kv_dim, 0, false);

        threadgroup float* output_ptr =
          shared_output +
          8U * static_cast<uint>(simdgroup) +
          static_cast<uint>(output_index) * 16U *
            static_cast<uint>(simdgroups);
        simdgroup_load(output_matrix[0], output_ptr, padded_value_dim, 0,
                       false);
        simdgroup_load(output_matrix[1], output_ptr + 8U * simdgroups,
                       padded_value_dim, 0, false);

        simdgroup_multiply_accumulate(output_matrix[0], weights0,
                                      value_matrix[0], output_matrix[0]);
        simdgroup_multiply_accumulate(output_matrix[1], weights0,
                                      value_matrix[1], output_matrix[1]);
        simdgroup_multiply_accumulate(output_matrix[0], weights1,
                                      value_matrix[2], output_matrix[0]);
        simdgroup_multiply_accumulate(output_matrix[1], weights1,
                                      value_matrix[3], output_matrix[1]);

        simdgroup_store(output_matrix[0], output_ptr, padded_value_dim, 0,
                        false);
        simdgroup_store(output_matrix[1], output_ptr + 8U * simdgroups,
                        padded_value_dim, 0, false);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (short local_query = 0; local_query < queries_per_simdgroup;
       ++local_query) {
    const short query_offset =
      local_query * simdgroups + static_cast<short>(simdgroup);
    const uint token = query_start + static_cast<uint>(query_offset);
    if (token >= params.tokens) {
      continue;
    }
    const float output_scale =
      normalizer[local_query] == 0.0f ? 0.0f : 1.0f / normalizer[local_query];
    device float4* output4 = reinterpret_cast<device float4*>(
      output + token * query_head_stride + head * static_cast<uint>(head_dim));
    for (short index = static_cast<short>(lane); index < value4_count;
         index += simd_width) {
      output4[index] =
        shared_output4[query_offset * padded_value4_count + index] *
      output_scale;
    }
  }
}

)metal";

constexpr const char* kKernelSourcePart3 = R"metal(
kernel void attention_f32_batched_flash256_f16_kv(
                                        const device float* query [[buffer(0)]],
                                        const device half* key_cache [[buffer(1)]],
                                        const device half* value_cache [[buffer(2)]],
                                        device float* output [[buffer(3)]],
                                        constant FlashAttention256Params& params [[buffer(4)]],
                                        const device half* key_tail [[buffer(5)]],
                                        const device half* value_tail [[buffer(6)]],
                                        threadgroup half* shmem_f16 [[threadgroup(0)]],
                                        uint3 group_id [[threadgroup_position_in_grid]],
                                        ushort lane [[thread_index_in_simdgroup]],
                                        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr short query_tile = 8;
  constexpr short cache_tile = 64;
  constexpr uint cache_tile_u = 64U;
  constexpr short head_dim = 256;
  constexpr short simd_width = 32;
  constexpr short simdgroups = 4;
  constexpr short queries_per_simdgroup = query_tile / simdgroups;
  constexpr short score_stride = 2 * cache_tile;
  constexpr short padded_value_dim = 256;
  constexpr short query_storage = head_dim + 2 * padded_value_dim;
  constexpr short q4_count = head_dim / 4;
  constexpr short q8_count = head_dim / 8;
  constexpr short value4_count = head_dim / 4;
  constexpr short padded_value4_count = padded_value_dim / 4;
  constexpr short padded_value8_count = padded_value_dim / 8;
  constexpr short output_matrices_per_simdgroup = padded_value8_count / simdgroups;

  const uint query_start = group_id.x * static_cast<uint>(query_tile);
  const uint head = group_id.y;
  if (query_start >= params.tokens || head >= params.heads) {
    return;
  }

  const uint kv_head = head / params.group;
  const uint kv_dim = params.kv_heads * static_cast<uint>(head_dim);
  const uint key_count = params.start_position + params.tokens;
  const uint padded_key_count =
    ((key_count + cache_tile_u - 1U) / cache_tile_u) * cache_tile_u;
  const uint query_head_stride = params.heads * static_cast<uint>(head_dim);
  const uint cache_layer_base = params.layer * params.capacity_tokens * kv_dim;
  const uint kv_head_base = kv_head * static_cast<uint>(head_dim);

  threadgroup half* shared_query = shmem_f16;
  threadgroup half4* shared_query4 = reinterpret_cast<threadgroup half4*>(shared_query);
  threadgroup float* shared_output =
    reinterpret_cast<threadgroup float*>(shmem_f16 + query_tile * head_dim);
  threadgroup float4* shared_output4 =
    reinterpret_cast<threadgroup float4*>(shared_output);
  threadgroup float* shared_scores =
    reinterpret_cast<threadgroup float*>(shmem_f16 + query_tile * query_storage);
  threadgroup float2* shared_scores2 =
    reinterpret_cast<threadgroup float2*>(shared_scores);

  for (short query_offset = 0; query_offset < query_tile; ++query_offset) {
    const uint token = query_start + static_cast<uint>(query_offset);
    const bool active_query = token < params.tokens;
    const device float4* query4 = reinterpret_cast<const device float4*>(
      query + token * query_head_stride + head * static_cast<uint>(head_dim));
    for (short index = static_cast<short>(lane); index < q4_count;
         index += simd_width) {
      shared_query4[query_offset * q4_count + index] =
        active_query ? half4(query4[index]) : half4(0.0h);
    }
    for (short index = static_cast<short>(lane); index < padded_value4_count;
         index += simd_width) {
      shared_output4[query_offset * padded_value4_count + index] = float4(0.0f);
    }
    for (short index = static_cast<short>(lane); index < score_stride;
         index += simd_width) {
      shared_scores[query_offset * score_stride + index] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float normalizer[queries_per_simdgroup] = {0.0f, 0.0f};
  float max_score[queries_per_simdgroup] = {-FLT_MAX / 2.0f, -FLT_MAX / 2.0f};

  for (uint cache_block = 0; cache_block < padded_key_count;
       cache_block += cache_tile_u) {
    const bool tail_block = cache_block + cache_tile_u > key_count;
    device const half* key_block = tail_block
      ? key_tail + kv_head_base
      : key_cache + cache_layer_base + cache_block * kv_dim + kv_head_base;

    for (short column_block = 0; column_block < (cache_tile / 8) / simdgroups;
         ++column_block) {
      simdgroup_float8x8 qk =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
      device const half* key_ptr =
        key_block +
        (static_cast<uint>(simdgroup) * 8U +
         static_cast<uint>(column_block) * 8U * static_cast<uint>(simdgroups)) *
          kv_dim;

      simdgroup_half8x8 query_matrix[2];
      simdgroup_half8x8 key_matrix[2];
      for (short index = 0; index < q8_count / 2; ++index) {
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_load(query_matrix[0], shared_query + 0 * 8 + 16 * index,
                       head_dim);
        simdgroup_load(query_matrix[1], shared_query + 1 * 8 + 16 * index,
                       head_dim);
        simdgroup_load(key_matrix[0], key_ptr + 0 * 8 + 16 * index, kv_dim,
                       0, true);
        simdgroup_load(key_matrix[1], key_ptr + 1 * 8 + 16 * index, kv_dim,
                       0, true);
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_multiply_accumulate(qk, query_matrix[0], key_matrix[0], qk);
        simdgroup_multiply_accumulate(qk, query_matrix[1], key_matrix[1], qk);
      }

      threadgroup float* score_ptr =
        shared_scores +
        static_cast<uint>(simdgroup) * 8U +
        static_cast<uint>(column_block) * 8U * static_cast<uint>(simdgroups);
      simdgroup_store(qk, score_ptr, score_stride, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short local_query = 0; local_query < queries_per_simdgroup;
         ++local_query) {
      const short query_offset =
        local_query * simdgroups + static_cast<short>(simdgroup);
      const uint token = query_start + static_cast<uint>(query_offset);
      const uint query_position = params.start_position + token;
      float2 scores =
        shared_scores2[query_offset * score_stride / 2 + lane] * params.scale;

      const uint cache_token0 = cache_block + static_cast<uint>(lane) * 2U;
      const uint cache_token1 = cache_token0 + 1U;
      if (token >= params.tokens || cache_token0 >= key_count ||
          cache_token0 > query_position) {
        scores[0] = -INFINITY;
      }
      if (token >= params.tokens || cache_token1 >= key_count ||
          cache_token1 > query_position) {
        scores[1] = -INFINITY;
      }

      const float previous_max = max_score[local_query];
      max_score[local_query] =
        simd_max(max(max_score[local_query], max(scores[0], scores[1])));

      const float previous_scale = exp(previous_max - max_score[local_query]);
      const float2 weights = exp(scores - max_score[local_query]);
      normalizer[local_query] =
        normalizer[local_query] * previous_scale +
        simd_sum(weights[0] + weights[1]);
      shared_scores2[query_offset * score_stride / 2 + lane] = weights;

      for (short index = static_cast<short>(lane); index < value4_count;
           index += simd_width) {
        shared_output4[query_offset * padded_value4_count + index] *=
          previous_scale;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const half* value_block = tail_block
      ? value_tail + kv_head_base
      : value_cache + cache_layer_base + cache_block * kv_dim + kv_head_base;
    for (short column_pair = 0; column_pair < cache_tile / 16; ++column_pair) {
      simdgroup_float8x8 weights0;
      simdgroup_float8x8 weights1;
      simdgroup_load(weights0, shared_scores + 16 * column_pair + 0,
                     score_stride, 0, false);
      simdgroup_load(weights1, shared_scores + 16 * column_pair + 8,
                     score_stride, 0, false);

      for (short output_index = 0; output_index < output_matrices_per_simdgroup / 2;
           ++output_index) {
        simdgroup_half8x8 value_matrix[4];
        simdgroup_float8x8 output_matrix[2];
        const device half* value_ptr =
          value_block + static_cast<uint>(column_pair) * 16U * kv_dim +
          8U * static_cast<uint>(simdgroup) +
          static_cast<uint>(output_index) * 16U *
            static_cast<uint>(simdgroups);
        simdgroup_load(value_matrix[0],
                       value_ptr + 0U * static_cast<uint>(simdgroups), kv_dim,
                       0, false);
        simdgroup_load(value_matrix[1],
                       value_ptr + 8U * static_cast<uint>(simdgroups), kv_dim,
                       0, false);
        simdgroup_load(value_matrix[2],
                       value_ptr + 0U * static_cast<uint>(simdgroups) +
                         8U * kv_dim,
                       kv_dim, 0, false);
        simdgroup_load(value_matrix[3],
                       value_ptr + 8U * static_cast<uint>(simdgroups) +
                         8U * kv_dim,
                       kv_dim, 0, false);

        threadgroup float* output_ptr =
          shared_output +
          8U * static_cast<uint>(simdgroup) +
          static_cast<uint>(output_index) * 16U *
            static_cast<uint>(simdgroups);
        simdgroup_load(output_matrix[0], output_ptr, padded_value_dim, 0,
                       false);
        simdgroup_load(output_matrix[1], output_ptr + 8U * simdgroups,
                       padded_value_dim, 0, false);

        simdgroup_multiply_accumulate(output_matrix[0], weights0,
                                      value_matrix[0], output_matrix[0]);
        simdgroup_multiply_accumulate(output_matrix[1], weights0,
                                      value_matrix[1], output_matrix[1]);
        simdgroup_multiply_accumulate(output_matrix[0], weights1,
                                      value_matrix[2], output_matrix[0]);
        simdgroup_multiply_accumulate(output_matrix[1], weights1,
                                      value_matrix[3], output_matrix[1]);

        simdgroup_store(output_matrix[0], output_ptr, padded_value_dim, 0,
                        false);
        simdgroup_store(output_matrix[1], output_ptr + 8U * simdgroups,
                        padded_value_dim, 0, false);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (short local_query = 0; local_query < queries_per_simdgroup;
       ++local_query) {
    const short query_offset =
      local_query * simdgroups + static_cast<short>(simdgroup);
    const uint token = query_start + static_cast<uint>(query_offset);
    if (token >= params.tokens) {
      continue;
    }
    const float output_scale =
      normalizer[local_query] == 0.0f ? 0.0f : 1.0f / normalizer[local_query];
    device float4* output4 = reinterpret_cast<device float4*>(
      output + token * query_head_stride + head * static_cast<uint>(head_dim));
    for (short index = static_cast<short>(lane); index < value4_count;
         index += simd_width) {
      output4[index] =
        shared_output4[query_offset * padded_value4_count + index] *
        output_scale;
    }
  }
}
)metal";

std::string make_kernel_source() {
  std::string source;
  source.reserve(std::strlen(kKernelSourcePart0) +
                 std::strlen(kKernelSourcePart1) +
                 std::strlen(kKernelSourcePart2) +
                 std::strlen(kKernelSourcePart3));
  source += kKernelSourcePart0;
  source += kKernelSourcePart1;
  source += kKernelSourcePart2;
  source += kKernelSourcePart3;
  return source;
}

std::string yes_no(bool value) {
  return value ? "yes" : "no";
}

enum class EnvFlag {
  unset,
  enabled,
  disabled,
};

EnvFlag tiled_attention_env_flag() {
  const char* value = std::getenv("KRAKEN_QWEN35_TILED_ATTENTION");
  if (value == nullptr) {
    return EnvFlag::unset;
  }
  const std::string_view setting{value};
  if (setting == "1" || setting == "true" || setting == "TRUE" ||
      setting == "on" || setting == "ON" || setting == "yes" ||
      setting == "YES") {
    return EnvFlag::enabled;
  }
  if (setting == "0" || setting == "false" || setting == "FALSE" ||
      setting == "off" || setting == "OFF" || setting == "no" ||
      setting == "NO") {
    return EnvFlag::disabled;
  }
  return EnvFlag::unset;
}

bool should_use_tiled_attention(std::size_t tokens, std::size_t head_dim) {
  if (head_dim > 256U) {
    return false;
  }
  const auto env_flag = tiled_attention_env_flag();
  if (env_flag == EnvFlag::disabled) {
    return false;
  }
  if (env_flag == EnvFlag::enabled) {
    return tokens >= 16U;
  }
  return tokens >= 1024U;
}

std::size_t requested_tiled_attention_cache_tile() {
  const char* value = std::getenv("KRAKEN_QWEN35_TILED_ATTENTION_TILE");
  if (value == nullptr) {
    return 16U;
  }
  const std::string_view setting{value};
  if (setting == "32") {
    return 32U;
  }
  return 16U;
}

EnvFlag flash_attention_env_flag() {
  const char* value = std::getenv("KRAKEN_QWEN35_FLASH_ATTENTION");
  if (value == nullptr) {
    return EnvFlag::unset;
  }
  const std::string_view setting{value};
  if (setting == "1" || setting == "true" || setting == "TRUE" ||
      setting == "on" || setting == "ON" || setting == "yes" ||
      setting == "YES") {
    return EnvFlag::enabled;
  }
  if (setting == "0" || setting == "false" || setting == "FALSE" ||
      setting == "off" || setting == "OFF" || setting == "no" ||
      setting == "NO") {
    return EnvFlag::disabled;
  }
  return EnvFlag::unset;
}

std::string nsstring_to_string(NSString* value) {
  if (value == nil) {
    return {};
  }
  const char* utf8 = [value UTF8String];
  return utf8 == nullptr ? std::string{} : std::string{utf8};
}

std::string nserror_to_string(NSError* error) {
  if (error == nil) {
    return "unknown Metal error";
  }
  return nsstring_to_string([error localizedDescription]);
}

bool fits_nsuinteger(std::size_t value) {
  return value <= static_cast<std::size_t>(std::numeric_limits<NSUInteger>::max());
}

bool fits_uint32(std::size_t value) {
  return value <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t& result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16U);
}

NSUInteger choose_threadgroup_width(NSUInteger max_threads) {
  const NSUInteger limit = static_cast<NSUInteger>(256);
  const NSUInteger capped = std::min(max_threads, limit);
  NSUInteger width = static_cast<NSUInteger>(1);
  while (width < capped && width <= capped / static_cast<NSUInteger>(2)) {
    width *= static_cast<NSUInteger>(2);
  }
  return width;
}

struct MatVecLayout {
  std::size_t elements{0};
  std::size_t weight_bytes{0};
  std::size_t input_bytes{0};
  std::size_t output_bytes{0};
};

struct QuantMatVecLayout {
  std::size_t blocks_per_row{0};
  std::size_t weight_bytes{0};
  std::size_t input_bytes{0};
  std::size_t output_bytes{0};
};

struct SizeParams {
  std::uint32_t size{0};
  std::uint32_t threads{0};
};

struct RowwiseParams {
  std::uint32_t row_size{0};
  std::uint32_t total{0};
};

struct Q4KMatVecParams {
  std::uint32_t cols{0};
  std::uint32_t blocks_per_row{0};
  std::uint32_t threads_per_row{0};
};

struct QuantGetRowParams {
  std::uint32_t row{0};
  std::uint32_t cols{0};
  std::uint32_t blocks_per_row{0};
};

struct QuantGetRowsParams {
  std::uint32_t rows{0};
  std::uint32_t tokens{0};
  std::uint32_t cols{0};
  std::uint32_t blocks_per_row{0};
};

struct QuantMatMulParams {
  std::uint32_t cols{0};
  std::uint32_t blocks_per_row{0};
  std::uint32_t threads_per_row{0};
  std::uint32_t rows{0};
  std::uint32_t tokens{0};
};

struct EmbeddingParams {
  std::uint32_t token{0};
  std::uint32_t hidden_size{0};
};

struct RmsNormParams {
  std::uint32_t size{0};
  float eps{0.0F};
  std::uint32_t threads{0};
};

struct BatchedRmsNormParams {
  std::uint32_t size{0};
  std::uint32_t tokens{0};
  float eps{0.0F};
  std::uint32_t threads{0};
};

struct QkNormParams {
  std::uint32_t heads{0};
  std::uint32_t head_dim{0};
  float eps{0.0F};
  std::uint32_t threads{0};
};

struct BatchedQkNormParams {
  std::uint32_t tokens{0};
  std::uint32_t heads{0};
  std::uint32_t head_dim{0};
  float eps{0.0F};
  std::uint32_t threads{0};
};

struct L2NormParams {
  std::uint32_t rows{0};
  std::uint32_t row_size{0};
  float eps{0.0F};
  std::uint32_t threads{0};
};

struct SplitQkvL2NormParams {
  std::uint32_t tokens{0};
  std::uint32_t key_heads{0};
  std::uint32_t value_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t conv_channels{0};
  float eps{0.0F};
  std::uint32_t threads{0};
};

struct GatedDeltaNetParams {
  std::uint32_t key_heads{0};
  std::uint32_t value_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t threads{0};
  float scale{0.0F};
};

struct BatchedGatedDeltaNetParams {
  std::uint32_t tokens{0};
  std::uint32_t key_heads{0};
  std::uint32_t value_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t threads{0};
  float scale{0.0F};
};

struct SsmConvParams {
  std::uint32_t conv_kernel{0};
  std::uint32_t input_span{0};
  std::uint32_t channels{0};
  std::uint32_t tokens{0};
  std::uint32_t sequences{0};
  std::uint32_t total{0};
};

struct SsmConv1StatefulParams {
  std::uint32_t conv_kernel{0};
  std::uint32_t channels{0};
};

struct BuildSsmConvStateParams {
  std::uint32_t conv_kernel{0};
  std::uint32_t channels{0};
  std::uint32_t tokens{0};
  std::uint32_t input_span{0};
};

struct RopeParams {
  std::uint32_t heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t position{0};
  float theta{0.0F};
};

struct MropeParams {
  std::uint32_t tokens{0};
  std::uint32_t heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t n_dims{0};
  std::uint32_t section_0{0};
  std::uint32_t section_1{0};
  std::uint32_t section_2{0};
  std::uint32_t section_3{0};
  float theta{0.0F};
};

struct CopyRegionParams {
  std::uint32_t source_offset{0};
  std::uint32_t destination_offset{0};
  std::uint32_t size{0};
};

struct CopyRowsParams {
  std::uint32_t source_offset{0};
  std::uint32_t destination_offset{0};
  std::uint32_t source_stride{0};
  std::uint32_t destination_stride{0};
  std::uint32_t row_size{0};
  std::uint32_t total{0};
};

struct AttentionParams {
  std::uint32_t layer{0};
  std::uint32_t position{0};
  std::uint32_t capacity_tokens{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t group{0};
  float scale{0.0F};
};

struct BatchedAttentionParams {
  std::uint32_t layer{0};
  std::uint32_t start_position{0};
  std::uint32_t tokens{0};
  std::uint32_t capacity_tokens{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t group{0};
  std::uint32_t threads{0};
  float scale{0.0F};
};

struct FlashAttention256Params {
  std::uint32_t layer{0};
  std::uint32_t start_position{0};
  std::uint32_t tokens{0};
  std::uint32_t capacity_tokens{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t group{0};
  float scale{0.0F};
};

Result<MatVecLayout> make_matvec_layout(std::size_t rows, std::size_t cols) {
  if (rows == 0 || cols == 0) {
    return Status::invalid_argument("MPS matvec rows and cols must be greater than zero");
  }
  if (!fits_uint32(cols)) {
    return Status::invalid_argument("MPS matvec cols exceeds uint32 range");
  }
  if (!fits_nsuinteger(rows)) {
    return Status::invalid_argument("MPS matvec rows exceeds NSUInteger range");
  }

  std::size_t elements = 0;
  if (!checked_mul(rows, cols, elements)) {
    return Status::invalid_argument("MPS matvec weight element count overflow");
  }
  if (!fits_uint32(elements)) {
    return Status::invalid_argument("MPS matvec weight element count exceeds kernel range");
  }

  std::size_t weight_bytes = 0;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  if (!checked_mul(elements, sizeof(std::uint16_t), weight_bytes)) {
    return Status::invalid_argument("MPS matvec weight byte count overflow");
  }
  if (!checked_mul(cols, sizeof(float), input_bytes) ||
      !checked_mul(rows, sizeof(float), output_bytes)) {
    return Status::invalid_argument("MPS matvec buffer byte count overflow");
  }
  return MatVecLayout{elements, weight_bytes, input_bytes, output_bytes};
}

Result<QuantMatVecLayout> make_q4_k_matvec_layout(std::size_t rows, std::size_t cols) {
  if (rows == 0 || cols == 0) {
    return Status::invalid_argument("MPS Q4_K matvec rows and cols must be greater than zero");
  }
  if (cols % 256U != 0) {
    return Status::invalid_argument("MPS Q4_K matvec cols must be divisible by 256");
  }
  if (!fits_uint32(cols)) {
    return Status::invalid_argument("MPS Q4_K matvec cols exceeds uint32 range");
  }
  if (!fits_nsuinteger(rows)) {
    return Status::invalid_argument("MPS Q4_K matvec rows exceeds NSUInteger range");
  }
  const std::size_t blocks_per_row = cols / 256U;
  if (!fits_uint32(blocks_per_row)) {
    return Status::invalid_argument("MPS Q4_K matvec block count exceeds uint32 range");
  }

  std::size_t total_blocks = 0;
  if (!checked_mul(rows, blocks_per_row, total_blocks)) {
    return Status::invalid_argument("MPS Q4_K matvec block count overflow");
  }
  std::size_t weight_bytes = 0;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  if (!checked_mul(total_blocks, static_cast<std::size_t>(144), weight_bytes)) {
    return Status::invalid_argument("MPS Q4_K matvec weight byte count overflow");
  }
  if (!checked_mul(cols, sizeof(float), input_bytes) ||
      !checked_mul(rows, sizeof(float), output_bytes)) {
    return Status::invalid_argument("MPS Q4_K matvec buffer byte count overflow");
  }
  return QuantMatVecLayout{blocks_per_row, weight_bytes, input_bytes, output_bytes};
}

Result<QuantMatVecLayout> make_q5_k_matvec_layout(std::size_t rows, std::size_t cols) {
  if (rows == 0 || cols == 0) {
    return Status::invalid_argument("MPS Q5_K matvec rows and cols must be greater than zero");
  }
  if (cols % 256U != 0) {
    return Status::invalid_argument("MPS Q5_K matvec cols must be divisible by 256");
  }
  if (!fits_uint32(cols)) {
    return Status::invalid_argument("MPS Q5_K matvec cols exceeds uint32 range");
  }
  if (!fits_nsuinteger(rows)) {
    return Status::invalid_argument("MPS Q5_K matvec rows exceeds NSUInteger range");
  }
  const std::size_t blocks_per_row = cols / 256U;
  if (!fits_uint32(blocks_per_row)) {
    return Status::invalid_argument("MPS Q5_K matvec block count exceeds uint32 range");
  }

  std::size_t total_blocks = 0;
  if (!checked_mul(rows, blocks_per_row, total_blocks)) {
    return Status::invalid_argument("MPS Q5_K matvec block count overflow");
  }
  std::size_t weight_bytes = 0;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  if (!checked_mul(total_blocks, static_cast<std::size_t>(176), weight_bytes)) {
    return Status::invalid_argument("MPS Q5_K matvec weight byte count overflow");
  }
  if (!checked_mul(cols, sizeof(float), input_bytes) ||
      !checked_mul(rows, sizeof(float), output_bytes)) {
    return Status::invalid_argument("MPS Q5_K matvec buffer byte count overflow");
  }
  return QuantMatVecLayout{blocks_per_row, weight_bytes, input_bytes, output_bytes};
}

Result<QuantMatVecLayout> make_q6_k_matvec_layout(std::size_t rows, std::size_t cols) {
  if (rows == 0 || cols == 0) {
    return Status::invalid_argument("MPS Q6_K matvec rows and cols must be greater than zero");
  }
  if (cols % 256U != 0) {
    return Status::invalid_argument("MPS Q6_K matvec cols must be divisible by 256");
  }
  if (!fits_uint32(cols)) {
    return Status::invalid_argument("MPS Q6_K matvec cols exceeds uint32 range");
  }
  if (!fits_nsuinteger(rows)) {
    return Status::invalid_argument("MPS Q6_K matvec rows exceeds NSUInteger range");
  }
  const std::size_t blocks_per_row = cols / 256U;
  if (!fits_uint32(blocks_per_row)) {
    return Status::invalid_argument("MPS Q6_K matvec block count exceeds uint32 range");
  }

  std::size_t total_blocks = 0;
  if (!checked_mul(rows, blocks_per_row, total_blocks)) {
    return Status::invalid_argument("MPS Q6_K matvec block count overflow");
  }
  std::size_t weight_bytes = 0;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  if (!checked_mul(total_blocks, static_cast<std::size_t>(210), weight_bytes)) {
    return Status::invalid_argument("MPS Q6_K matvec weight byte count overflow");
  }
  if (!checked_mul(cols, sizeof(float), input_bytes) ||
      !checked_mul(rows, sizeof(float), output_bytes)) {
    return Status::invalid_argument("MPS Q6_K matvec buffer byte count overflow");
  }
  return QuantMatVecLayout{blocks_per_row, weight_bytes, input_bytes, output_bytes};
}

Result<QuantMatVecLayout> make_k_quant_matmul_layout(std::size_t rows,
                                                     std::size_t cols,
                                                     std::size_t tokens,
                                                     std::size_t block_bytes,
                                                     const char* name) {
  if (rows == 0 || cols == 0 || tokens == 0) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " matmul rows, cols and tokens must be greater than zero");
  }
  if (cols % 256U != 0) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " matmul cols must be divisible by 256");
  }
  if (!fits_uint32(cols) || !fits_uint32(rows)) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " matmul dimensions exceed uint32 range");
  }
  if (!fits_nsuinteger(rows) || !fits_nsuinteger(tokens)) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " matmul dimensions exceed NSUInteger range");
  }
  const std::size_t blocks_per_row = cols / 256U;
  if (!fits_uint32(blocks_per_row)) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " matmul block count exceeds uint32 range");
  }

  std::size_t total_blocks = 0;
  if (!checked_mul(rows, blocks_per_row, total_blocks)) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " matmul block count overflow");
  }
  std::size_t weight_bytes = 0;
  if (!checked_mul(total_blocks, block_bytes, weight_bytes)) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " matmul weight byte count overflow");
  }

  std::size_t input_values = 0;
  std::size_t output_values = 0;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  if (!checked_mul(tokens, cols, input_values) ||
      !checked_mul(tokens, rows, output_values) ||
      !checked_mul(input_values, sizeof(float), input_bytes) ||
      !checked_mul(output_values, sizeof(float), output_bytes)) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " matmul buffer byte count overflow");
  }

  return QuantMatVecLayout{blocks_per_row, weight_bytes, input_bytes, output_bytes};
}

Result<std::uint32_t> checked_u32(std::size_t value, const char* name) {
  if (!fits_uint32(value)) {
    return Status::invalid_argument(std::string{"MPS "} + name + " exceeds uint32 range");
  }
  return static_cast<std::uint32_t>(value);
}

Result<std::size_t> strided_f32_region_end(std::size_t offset,
                                           std::size_t rows,
                                           std::size_t stride,
                                           std::size_t row_size,
                                           const char* name) {
  if (rows == 0 || row_size == 0) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " rows and row size must be positive");
  }
  if (stride < row_size) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " stride must be at least row size");
  }
  std::size_t last_row = 0;
  if (!checked_mul(rows - 1U, stride, last_row)) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " row offset overflow");
  }
  if (offset > std::numeric_limits<std::size_t>::max() - last_row) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " offset overflow");
  }
  const std::size_t last_row_offset = offset + last_row;
  if (row_size > std::numeric_limits<std::size_t>::max() - last_row_offset) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " end offset overflow");
  }
  return last_row_offset + row_size;
}

Status validate_f32_buffer(const MpsBuffer& buffer, std::size_t values,
                           const char* name) {
  std::size_t bytes = 0;
  if (!checked_mul(values, sizeof(float), bytes)) {
    return Status::invalid_argument(std::string{"MPS "} + name + " byte count overflow");
  }
  if (!buffer.valid()) {
    return Status::invalid_argument(std::string{"MPS "} + name + " buffer is not initialized");
  }
  if (buffer.byte_size() < bytes) {
    return Status::invalid_argument(std::string{"MPS "} + name + " buffer is too small");
  }
  return Status::ok();
}

Status validate_f16_buffer(const MpsBuffer& buffer, std::size_t values,
                           const char* name) {
  std::size_t bytes = 0;
  if (!checked_mul(values, sizeof(std::uint16_t), bytes)) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " byte count overflow");
  }
  if (!buffer.valid()) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " buffer is not initialized");
  }
  if (buffer.byte_size() < bytes) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " buffer is too small");
  }
  return Status::ok();
}

Status validate_i32_buffer(const MpsBuffer& buffer, std::size_t values,
                           const char* name) {
  std::size_t bytes = 0;
  if (!checked_mul(values, sizeof(std::int32_t), bytes)) {
    return Status::invalid_argument(std::string{"MPS "} + name + " byte count overflow");
  }
  if (!buffer.valid()) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " buffer is not initialized");
  }
  if (buffer.byte_size() < bytes) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " buffer is too small");
  }
  return Status::ok();
}

Result<std::size_t> validate_f32_buffer_region(const MpsBuffer& buffer,
                                               std::size_t offset,
                                               std::size_t values,
                                               const char* name) {
  if (!buffer.valid()) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " buffer is not initialized");
  }
  std::size_t offset_bytes = 0;
  std::size_t bytes = 0;
  if (!checked_mul(offset, sizeof(float), offset_bytes) ||
      !checked_mul(values, sizeof(float), bytes)) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " byte count overflow");
  }
  if (offset_bytes > buffer.byte_size() ||
      bytes > buffer.byte_size() - offset_bytes) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " buffer region is too small");
  }
  if (!fits_nsuinteger(offset_bytes)) {
    return Status::invalid_argument(std::string{"MPS "} + name +
                                    " offset exceeds NSUInteger range");
  }
  return offset_bytes;
}

}  // namespace

struct MpsBuffer::Impl {
  id<MTLBuffer> buffer{nil};
  std::size_t byte_size{0};

  ~Impl() {
    if (buffer != nil) {
      [buffer release];
    }
  }
};

struct MpsContext::Impl {
  id<MTLDevice> device{nil};
  id<MTLCommandQueue> queue{nil};
  id<MTLCommandBuffer> active_command_buffer{nil};
  bool graph_active{false};
  id<MTLComputePipelineState> matvec_pipeline{nil};
  id<MTLComputePipelineState> q4_k_matvec_pipeline{nil};
  id<MTLComputePipelineState> q5_k_matvec_pipeline{nil};
  id<MTLComputePipelineState> q6_k_matvec_pipeline{nil};
  id<MTLComputePipelineState> q4_k_matmul_pipeline{nil};
  id<MTLComputePipelineState> q5_k_matmul_pipeline{nil};
  id<MTLComputePipelineState> q6_k_matmul_pipeline{nil};
  id<MTLComputePipelineState> q4_k_mul_mv_ext_r1_3_pipeline{nil};
  id<MTLComputePipelineState> q4_k_mul_mv_ext_r1_4_pipeline{nil};
  id<MTLComputePipelineState> q4_k_mul_mv_ext_r1_5_pipeline{nil};
  id<MTLComputePipelineState> q5_k_mul_mv_ext_r1_3_pipeline{nil};
  id<MTLComputePipelineState> q5_k_mul_mv_ext_r1_4_pipeline{nil};
  id<MTLComputePipelineState> q5_k_mul_mv_ext_r1_5_pipeline{nil};
  id<MTLComputePipelineState> q6_k_mul_mv_ext_r1_3_pipeline{nil};
  id<MTLComputePipelineState> q6_k_mul_mv_ext_r1_4_pipeline{nil};
  id<MTLComputePipelineState> q6_k_mul_mv_ext_r1_5_pipeline{nil};
  id<MTLComputePipelineState> q4_k_mul_mm_simd_pipeline{nil};
  id<MTLComputePipelineState> q5_k_mul_mm_simd_pipeline{nil};
  id<MTLComputePipelineState> q6_k_mul_mm_simd_pipeline{nil};
  id<MTLComputePipelineState> q4_k_get_row_pipeline{nil};
  id<MTLComputePipelineState> q5_k_get_row_pipeline{nil};
  id<MTLComputePipelineState> q6_k_get_row_pipeline{nil};
  id<MTLComputePipelineState> q4_k_get_rows_pipeline{nil};
  id<MTLComputePipelineState> q5_k_get_rows_pipeline{nil};
  id<MTLComputePipelineState> q6_k_get_rows_pipeline{nil};
  id<MTLComputePipelineState> embedding_pipeline{nil};
  id<MTLComputePipelineState> rms_norm_pipeline{nil};
  id<MTLComputePipelineState> rms_norm_f32_pipeline{nil};
  id<MTLComputePipelineState> rms_norm_f32_batched_pipeline{nil};
  id<MTLComputePipelineState> qk_norm_pipeline{nil};
  id<MTLComputePipelineState> qk_norm_f32_pipeline{nil};
  id<MTLComputePipelineState> qk_norm_f32_batched_pipeline{nil};
  id<MTLComputePipelineState> qwen35_norm_gated_pipeline{nil};
  id<MTLComputePipelineState> l2_norm_pipeline{nil};
  id<MTLComputePipelineState> split_qkv_l2_norm_qwen35_pipeline{nil};
  id<MTLComputePipelineState> gated_delta_net_pipeline{nil};
  id<MTLComputePipelineState> gated_delta_net_batched_pipeline{nil};
  id<MTLComputePipelineState> gated_delta_net_batched_qwen35_pipeline{nil};
  id<MTLComputePipelineState> ssm_conv_pipeline{nil};
  id<MTLComputePipelineState> build_ssm_conv_state_pipeline{nil};
  id<MTLComputePipelineState> ssm_conv1_stateful_pipeline{nil};
  id<MTLComputePipelineState> rope_pipeline{nil};
  id<MTLComputePipelineState> mrope_pipeline{nil};
  id<MTLComputePipelineState> add_pipeline{nil};
  id<MTLComputePipelineState> mul_pipeline{nil};
  id<MTLComputePipelineState> add_row_pipeline{nil};
  id<MTLComputePipelineState> mul_row_pipeline{nil};
  id<MTLComputePipelineState> sigmoid_pipeline{nil};
  id<MTLComputePipelineState> softplus_pipeline{nil};
  id<MTLComputePipelineState> prepare_qwen35_gdn_gate_beta_pipeline{nil};
  id<MTLComputePipelineState> silu_pipeline{nil};
  id<MTLComputePipelineState> silu_mul_pipeline{nil};
  id<MTLComputePipelineState> copy_region_pipeline{nil};
  id<MTLComputePipelineState> copy_rows_pipeline{nil};
  id<MTLComputePipelineState> copy_rows_to_f16_pipeline{nil};
  id<MTLComputePipelineState> copy_f16_rows_pipeline{nil};
  id<MTLComputePipelineState> argmax_pipeline{nil};
  id<MTLComputePipelineState> argmax_prob_pipeline{nil};
  id<MTLComputePipelineState> attention_pipeline{nil};
  id<MTLComputePipelineState> attention_f16_kv_pipeline{nil};
  id<MTLComputePipelineState> attention_batched_pipeline{nil};
  id<MTLComputePipelineState> attention_batched_f16_kv_pipeline{nil};
  id<MTLComputePipelineState> attention_batched_tiled_pipeline{nil};
  id<MTLComputePipelineState> attention_batched_tiled32_pipeline{nil};
  id<MTLComputePipelineState> attention_batched_flash256_pipeline{nil};
  id<MTLComputePipelineState> attention_batched_flash256_f16_kv_pipeline{nil};
  // Reuse flash-tail padding storage instead of allocating it on every
  // non-64-aligned attention call, matching llama.cpp's extra workspace model.
  MpsBuffer flash_f32_key_tail_scratch;
  MpsBuffer flash_f32_value_tail_scratch;
  MpsBuffer flash_f16_key_tail_scratch;
  MpsBuffer flash_f16_value_tail_scratch;

  [[nodiscard]] id<MTLCommandBuffer> command_buffer() {
    if (graph_active) {
      return active_command_buffer;
    }
    return [queue commandBuffer];
  }

  [[nodiscard]] Status finish_command_buffer(id<MTLCommandBuffer> command_buffer,
                                             const std::string& error_prefix) {
    if (graph_active && command_buffer == active_command_buffer) {
      return Status::ok();
    }
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
      return Status::internal_error(error_prefix +
                                    nserror_to_string([command_buffer error]));
    }
    return Status::ok();
  }

  [[nodiscard]] Status begin_graph() {
    if (graph_active) {
      return Status::invalid_argument("MPS graph execution is already active");
    }
    active_command_buffer = [[queue commandBuffer] retain];
    if (active_command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    graph_active = true;
    return Status::ok();
  }

  [[nodiscard]] Status commit_graph() {
    if (!graph_active) {
      return Status::ok();
    }
    id<MTLCommandBuffer> command_buffer = active_command_buffer;
    active_command_buffer = nil;
    graph_active = false;

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    Status status = Status::ok();
    if ([command_buffer status] == MTLCommandBufferStatusError) {
      status = Status::internal_error("MPS graph command failed: " +
                                      nserror_to_string([command_buffer error]));
    }
    [command_buffer release];
    return status;
  }

  void abort_graph() {
    if (active_command_buffer != nil) {
      [active_command_buffer release];
      active_command_buffer = nil;
    }
    graph_active = false;
  }

  ~Impl() {
    abort_graph();
    if (attention_pipeline != nil) {
      [attention_pipeline release];
    }
    if (attention_f16_kv_pipeline != nil) {
      [attention_f16_kv_pipeline release];
    }
    if (attention_batched_pipeline != nil) {
      [attention_batched_pipeline release];
    }
    if (attention_batched_f16_kv_pipeline != nil) {
      [attention_batched_f16_kv_pipeline release];
    }
    if (attention_batched_tiled_pipeline != nil) {
      [attention_batched_tiled_pipeline release];
    }
    if (attention_batched_tiled32_pipeline != nil) {
      [attention_batched_tiled32_pipeline release];
    }
    if (attention_batched_flash256_pipeline != nil) {
      [attention_batched_flash256_pipeline release];
    }
    if (attention_batched_flash256_f16_kv_pipeline != nil) {
      [attention_batched_flash256_f16_kv_pipeline release];
    }
    if (argmax_pipeline != nil) {
      [argmax_pipeline release];
    }
    if (argmax_prob_pipeline != nil) {
      [argmax_prob_pipeline release];
    }
    if (copy_region_pipeline != nil) {
      [copy_region_pipeline release];
    }
    if (copy_rows_pipeline != nil) {
      [copy_rows_pipeline release];
    }
    if (copy_rows_to_f16_pipeline != nil) {
      [copy_rows_to_f16_pipeline release];
    }
    if (copy_f16_rows_pipeline != nil) {
      [copy_f16_rows_pipeline release];
    }
    if (silu_mul_pipeline != nil) {
      [silu_mul_pipeline release];
    }
    if (add_pipeline != nil) {
      [add_pipeline release];
    }
    if (mul_pipeline != nil) {
      [mul_pipeline release];
    }
    if (add_row_pipeline != nil) {
      [add_row_pipeline release];
    }
    if (mul_row_pipeline != nil) {
      [mul_row_pipeline release];
    }
    if (sigmoid_pipeline != nil) {
      [sigmoid_pipeline release];
    }
    if (softplus_pipeline != nil) {
      [softplus_pipeline release];
    }
    if (prepare_qwen35_gdn_gate_beta_pipeline != nil) {
      [prepare_qwen35_gdn_gate_beta_pipeline release];
    }
    if (silu_pipeline != nil) {
      [silu_pipeline release];
    }
    if (rope_pipeline != nil) {
      [rope_pipeline release];
    }
    if (mrope_pipeline != nil) {
      [mrope_pipeline release];
    }
    if (qk_norm_pipeline != nil) {
      [qk_norm_pipeline release];
    }
    if (qk_norm_f32_batched_pipeline != nil) {
      [qk_norm_f32_batched_pipeline release];
    }
    if (qwen35_norm_gated_pipeline != nil) {
      [qwen35_norm_gated_pipeline release];
    }
    if (rms_norm_pipeline != nil) {
      [rms_norm_pipeline release];
    }
    if (rms_norm_f32_pipeline != nil) {
      [rms_norm_f32_pipeline release];
    }
    if (rms_norm_f32_batched_pipeline != nil) {
      [rms_norm_f32_batched_pipeline release];
    }
    if (embedding_pipeline != nil) {
      [embedding_pipeline release];
    }
    if (qk_norm_f32_pipeline != nil) {
      [qk_norm_f32_pipeline release];
    }
    if (l2_norm_pipeline != nil) {
      [l2_norm_pipeline release];
    }
    if (split_qkv_l2_norm_qwen35_pipeline != nil) {
      [split_qkv_l2_norm_qwen35_pipeline release];
    }
    if (gated_delta_net_pipeline != nil) {
      [gated_delta_net_pipeline release];
    }
    if (gated_delta_net_batched_pipeline != nil) {
      [gated_delta_net_batched_pipeline release];
    }
    if (gated_delta_net_batched_qwen35_pipeline != nil) {
      [gated_delta_net_batched_qwen35_pipeline release];
    }
    if (ssm_conv_pipeline != nil) {
      [ssm_conv_pipeline release];
    }
    if (ssm_conv1_stateful_pipeline != nil) {
      [ssm_conv1_stateful_pipeline release];
    }
    if (build_ssm_conv_state_pipeline != nil) {
      [build_ssm_conv_state_pipeline release];
    }
    if (q4_k_matvec_pipeline != nil) {
      [q4_k_matvec_pipeline release];
    }
    if (q5_k_matvec_pipeline != nil) {
      [q5_k_matvec_pipeline release];
    }
    if (q6_k_matvec_pipeline != nil) {
      [q6_k_matvec_pipeline release];
    }
    if (q4_k_matmul_pipeline != nil) {
      [q4_k_matmul_pipeline release];
    }
    if (q5_k_matmul_pipeline != nil) {
      [q5_k_matmul_pipeline release];
    }
    if (q6_k_matmul_pipeline != nil) {
      [q6_k_matmul_pipeline release];
    }
    if (q4_k_mul_mv_ext_r1_3_pipeline != nil) {
      [q4_k_mul_mv_ext_r1_3_pipeline release];
    }
    if (q4_k_mul_mv_ext_r1_4_pipeline != nil) {
      [q4_k_mul_mv_ext_r1_4_pipeline release];
    }
    if (q4_k_mul_mv_ext_r1_5_pipeline != nil) {
      [q4_k_mul_mv_ext_r1_5_pipeline release];
    }
    if (q5_k_mul_mv_ext_r1_3_pipeline != nil) {
      [q5_k_mul_mv_ext_r1_3_pipeline release];
    }
    if (q5_k_mul_mv_ext_r1_4_pipeline != nil) {
      [q5_k_mul_mv_ext_r1_4_pipeline release];
    }
    if (q5_k_mul_mv_ext_r1_5_pipeline != nil) {
      [q5_k_mul_mv_ext_r1_5_pipeline release];
    }
    if (q6_k_mul_mv_ext_r1_3_pipeline != nil) {
      [q6_k_mul_mv_ext_r1_3_pipeline release];
    }
    if (q6_k_mul_mv_ext_r1_4_pipeline != nil) {
      [q6_k_mul_mv_ext_r1_4_pipeline release];
    }
    if (q6_k_mul_mv_ext_r1_5_pipeline != nil) {
      [q6_k_mul_mv_ext_r1_5_pipeline release];
    }
    if (q4_k_mul_mm_simd_pipeline != nil) {
      [q4_k_mul_mm_simd_pipeline release];
    }
    if (q5_k_mul_mm_simd_pipeline != nil) {
      [q5_k_mul_mm_simd_pipeline release];
    }
    if (q6_k_mul_mm_simd_pipeline != nil) {
      [q6_k_mul_mm_simd_pipeline release];
    }
    if (q4_k_get_row_pipeline != nil) {
      [q4_k_get_row_pipeline release];
    }
    if (q5_k_get_row_pipeline != nil) {
      [q5_k_get_row_pipeline release];
    }
    if (q6_k_get_row_pipeline != nil) {
      [q6_k_get_row_pipeline release];
    }
    if (q4_k_get_rows_pipeline != nil) {
      [q4_k_get_rows_pipeline release];
    }
    if (q5_k_get_rows_pipeline != nil) {
      [q5_k_get_rows_pipeline release];
    }
    if (q6_k_get_rows_pipeline != nil) {
      [q6_k_get_rows_pipeline release];
    }
    if (matvec_pipeline != nil) {
      [matvec_pipeline release];
    }
    if (queue != nil) {
      [queue release];
    }
    if (device != nil) {
      [device release];
    }
  }
};

Status encode_k_quant_mul_mv_ext(id<MTLCommandBuffer> command_buffer,
                                 id<MTLComputePipelineState> pipeline,
                                 id<MTLBuffer> weight,
                                 id<MTLBuffer> input,
                                 id<MTLBuffer> output,
                                 const QuantMatVecLayout& layout,
                                 std::size_t rows,
                                 std::size_t cols,
                                 std::size_t tokens,
                                 std::size_t r1ptg,
                                 const char* name) {
  constexpr NSUInteger kThreadsPerSimdgroup = 32;
  constexpr NSUInteger kSimdgroupsPerThreadgroup = 2;
  constexpr NSUInteger kRowsPerThreadgroup = 8;
  const NSUInteger required_threads =
    kThreadsPerSimdgroup * kSimdgroupsPerThreadgroup;
  if ([pipeline maxTotalThreadsPerThreadgroup] < required_threads) {
    return Status::internal_error(std::string{"MPS "} + name +
                                  " mul_mv_ext pipeline cannot run 64 threads");
  }

  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  if (encoder == nil) {
    return Status::unavailable("failed to create Metal compute encoder");
  }

  QuantMatMulParams params{
    static_cast<std::uint32_t>(cols),
    static_cast<std::uint32_t>(layout.blocks_per_row),
    0U,
    static_cast<std::uint32_t>(rows),
    static_cast<std::uint32_t>(tokens),
  };
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:weight offset:0 atIndex:0];
  [encoder setBuffer:input offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];

  const NSUInteger row_groups =
    (static_cast<NSUInteger>(rows) + kRowsPerThreadgroup - 1U) /
    kRowsPerThreadgroup;
  const NSUInteger token_groups =
    (static_cast<NSUInteger>(tokens) + static_cast<NSUInteger>(r1ptg) - 1U) /
    static_cast<NSUInteger>(r1ptg);
  const MTLSize group_count = MTLSizeMake(row_groups, token_groups, 1);
  const MTLSize threadgroup_size =
    MTLSizeMake(kThreadsPerSimdgroup, kSimdgroupsPerThreadgroup, 1);
  [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
  [encoder endEncoding];
  return Status::ok();
}

std::size_t k_quant_mul_mv_ext_r1ptg(std::size_t tokens) {
  switch (tokens) {
    case 5:
      return 5;
    case 6:
      return 3;
    default:
      return 4;
  }
}

Status encode_k_quant_mul_mm_simd(id<MTLCommandBuffer> command_buffer,
                                  id<MTLComputePipelineState> pipeline,
                                  id<MTLBuffer> weight,
                                  id<MTLBuffer> input,
                                  id<MTLBuffer> output,
                                  const QuantMatVecLayout& layout,
                                  std::size_t rows,
                                  std::size_t cols,
                                  std::size_t tokens,
                                  const char* name) {
  constexpr NSUInteger kThreadsPerSimdgroup = 32;
  constexpr NSUInteger kSimdgroupsPerThreadgroup = 4;
  constexpr NSUInteger kRowsPerThreadgroup = 64;
  constexpr NSUInteger kTokensPerThreadgroup = 32;
  constexpr NSUInteger kThreadgroupMemoryBytes = 8192;
  const NSUInteger required_threads =
    kThreadsPerSimdgroup * kSimdgroupsPerThreadgroup;
  if ([pipeline maxTotalThreadsPerThreadgroup] < required_threads) {
    return Status::internal_error(std::string{"MPS "} + name +
                                  " mul_mm simd pipeline cannot run 128 threads");
  }

  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  if (encoder == nil) {
    return Status::unavailable("failed to create Metal compute encoder");
  }

  QuantMatMulParams params{
    static_cast<std::uint32_t>(cols),
    static_cast<std::uint32_t>(layout.blocks_per_row),
    0U,
    static_cast<std::uint32_t>(rows),
    static_cast<std::uint32_t>(tokens),
  };
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:weight offset:0 atIndex:0];
  [encoder setBuffer:input offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  [encoder setThreadgroupMemoryLength:kThreadgroupMemoryBytes atIndex:0];

  const NSUInteger token_groups =
    (static_cast<NSUInteger>(tokens) + kTokensPerThreadgroup - 1U) /
    kTokensPerThreadgroup;
  const NSUInteger row_groups =
    (static_cast<NSUInteger>(rows) + kRowsPerThreadgroup - 1U) /
    kRowsPerThreadgroup;
  const MTLSize group_count = MTLSizeMake(token_groups, row_groups, 1);
  const MTLSize threadgroup_size =
    MTLSizeMake(kThreadsPerSimdgroup, kSimdgroupsPerThreadgroup, 1);
  [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
  [encoder endEncoding];
  return Status::ok();
}

MpsBuffer::MpsBuffer() = default;
MpsBuffer::~MpsBuffer() = default;
MpsBuffer::MpsBuffer(MpsBuffer&& other) noexcept = default;
MpsBuffer& MpsBuffer::operator=(MpsBuffer&& other) noexcept = default;
MpsBuffer::MpsBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool MpsBuffer::valid() const {
  return impl_ != nullptr && impl_->buffer != nil;
}

std::size_t MpsBuffer::byte_size() const {
  return impl_ == nullptr ? 0 : impl_->byte_size;
}

MpsMatVecWorkspace::MpsMatVecWorkspace() = default;
MpsMatVecWorkspace::~MpsMatVecWorkspace() = default;
MpsMatVecWorkspace::MpsMatVecWorkspace(MpsMatVecWorkspace&& other) noexcept = default;
MpsMatVecWorkspace& MpsMatVecWorkspace::operator=(MpsMatVecWorkspace&& other) noexcept =
  default;
MpsMatVecWorkspace::MpsMatVecWorkspace(std::size_t rows, std::size_t cols,
                                       MpsBuffer input, MpsBuffer output)
    : rows_(rows), cols_(cols), input_(std::move(input)), output_(std::move(output)) {}

bool MpsMatVecWorkspace::valid() const {
  return rows_ > 0 && cols_ > 0 && input_.valid() && output_.valid();
}

std::size_t MpsMatVecWorkspace::rows() const {
  return rows_;
}

std::size_t MpsMatVecWorkspace::cols() const {
  return cols_;
}

MpsContext::MpsContext() = default;
MpsContext::~MpsContext() = default;
MpsContext::MpsContext(MpsContext&& other) noexcept = default;
MpsContext& MpsContext::operator=(MpsContext&& other) noexcept = default;
MpsContext::MpsContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Result<MpsContext> MpsContext::create() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return Status::unavailable("Metal returned no default device");
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      [device release];
      return Status::unavailable("failed to create Metal command queue");
    }

    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->queue = queue;

    NSError* error = nil;
    const auto kernel_source = make_kernel_source();
    NSString* source = [NSString stringWithUTF8String:kernel_source.c_str()];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
      return Status::internal_error("failed to compile Metal kernels: " +
                                    nserror_to_string(error));
    }

    auto make_pipeline = [&](const char* name) -> Result<id<MTLComputePipelineState>> {
      id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:name]];
      if (function == nil) {
        return Status::internal_error("failed to find Metal kernel " + std::string{name});
      }

      error = nil;
      id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
      [function release];
      if (pipeline == nil) {
        return Status::internal_error("failed to create " + std::string{name} +
                                      " pipeline: " + nserror_to_string(error));
      }
      return pipeline;
    };

    auto matvec_pipeline = make_pipeline("bf16_matvec");
    if (!matvec_pipeline.is_ok()) {
      [library release];
      return matvec_pipeline.status();
    }
    impl->matvec_pipeline = matvec_pipeline.value();

    auto q4_k_matvec_pipeline = make_pipeline("q4_k_matvec");
    if (!q4_k_matvec_pipeline.is_ok()) {
      [library release];
      return q4_k_matvec_pipeline.status();
    }
    impl->q4_k_matvec_pipeline = q4_k_matvec_pipeline.value();

    auto q5_k_matvec_pipeline = make_pipeline("q5_k_matvec");
    if (!q5_k_matvec_pipeline.is_ok()) {
      [library release];
      return q5_k_matvec_pipeline.status();
    }
    impl->q5_k_matvec_pipeline = q5_k_matvec_pipeline.value();

    auto q6_k_matvec_pipeline = make_pipeline("q6_k_matvec");
    if (!q6_k_matvec_pipeline.is_ok()) {
      [library release];
      return q6_k_matvec_pipeline.status();
    }
    impl->q6_k_matvec_pipeline = q6_k_matvec_pipeline.value();

    auto q4_k_matmul_pipeline = make_pipeline("q4_k_matmul");
    if (!q4_k_matmul_pipeline.is_ok()) {
      [library release];
      return q4_k_matmul_pipeline.status();
    }
    impl->q4_k_matmul_pipeline = q4_k_matmul_pipeline.value();

    auto q5_k_matmul_pipeline = make_pipeline("q5_k_matmul");
    if (!q5_k_matmul_pipeline.is_ok()) {
      [library release];
      return q5_k_matmul_pipeline.status();
    }
    impl->q5_k_matmul_pipeline = q5_k_matmul_pipeline.value();

    auto q6_k_matmul_pipeline = make_pipeline("q6_k_matmul");
    if (!q6_k_matmul_pipeline.is_ok()) {
      [library release];
      return q6_k_matmul_pipeline.status();
    }
    impl->q6_k_matmul_pipeline = q6_k_matmul_pipeline.value();

    auto assign_pipeline = [&](id<MTLComputePipelineState>& target,
                               const char* name) -> Status {
      auto pipeline = make_pipeline(name);
      if (!pipeline.is_ok()) {
        return pipeline.status();
      }
      target = pipeline.value();
      return Status::ok();
    };

    auto pipeline_status =
      assign_pipeline(impl->q4_k_mul_mv_ext_r1_3_pipeline,
                      "q4_k_mul_mv_ext_r1_3");
    if (!pipeline_status.is_ok()) {
      [library release];
      return pipeline_status;
    }
    pipeline_status =
      assign_pipeline(impl->q4_k_mul_mv_ext_r1_4_pipeline,
                      "q4_k_mul_mv_ext_r1_4");
    if (!pipeline_status.is_ok()) {
      [library release];
      return pipeline_status;
    }
    pipeline_status =
      assign_pipeline(impl->q4_k_mul_mv_ext_r1_5_pipeline,
                      "q4_k_mul_mv_ext_r1_5");
    if (!pipeline_status.is_ok()) {
      [library release];
      return pipeline_status;
    }
    pipeline_status =
      assign_pipeline(impl->q5_k_mul_mv_ext_r1_3_pipeline,
                      "q5_k_mul_mv_ext_r1_3");
    if (!pipeline_status.is_ok()) {
      [library release];
      return pipeline_status;
    }
    pipeline_status =
      assign_pipeline(impl->q5_k_mul_mv_ext_r1_4_pipeline,
                      "q5_k_mul_mv_ext_r1_4");
    if (!pipeline_status.is_ok()) {
      [library release];
      return pipeline_status;
    }
    pipeline_status =
      assign_pipeline(impl->q5_k_mul_mv_ext_r1_5_pipeline,
                      "q5_k_mul_mv_ext_r1_5");
    if (!pipeline_status.is_ok()) {
      [library release];
      return pipeline_status;
    }
    pipeline_status =
      assign_pipeline(impl->q6_k_mul_mv_ext_r1_3_pipeline,
                      "q6_k_mul_mv_ext_r1_3");
    if (!pipeline_status.is_ok()) {
      [library release];
      return pipeline_status;
    }
    pipeline_status =
      assign_pipeline(impl->q6_k_mul_mv_ext_r1_4_pipeline,
                      "q6_k_mul_mv_ext_r1_4");
    if (!pipeline_status.is_ok()) {
      [library release];
      return pipeline_status;
    }
    pipeline_status =
      assign_pipeline(impl->q6_k_mul_mv_ext_r1_5_pipeline,
                      "q6_k_mul_mv_ext_r1_5");
    if (!pipeline_status.is_ok()) {
      [library release];
      return pipeline_status;
    }

    auto q4_k_mul_mm_simd_pipeline = make_pipeline("q4_k_mul_mm_simd");
    if (!q4_k_mul_mm_simd_pipeline.is_ok()) {
      [library release];
      return q4_k_mul_mm_simd_pipeline.status();
    }
    impl->q4_k_mul_mm_simd_pipeline = q4_k_mul_mm_simd_pipeline.value();

    auto q5_k_mul_mm_simd_pipeline = make_pipeline("q5_k_mul_mm_simd");
    if (!q5_k_mul_mm_simd_pipeline.is_ok()) {
      [library release];
      return q5_k_mul_mm_simd_pipeline.status();
    }
    impl->q5_k_mul_mm_simd_pipeline = q5_k_mul_mm_simd_pipeline.value();

    auto q6_k_mul_mm_simd_pipeline = make_pipeline("q6_k_mul_mm_simd");
    if (!q6_k_mul_mm_simd_pipeline.is_ok()) {
      [library release];
      return q6_k_mul_mm_simd_pipeline.status();
    }
    impl->q6_k_mul_mm_simd_pipeline = q6_k_mul_mm_simd_pipeline.value();

    auto q4_k_get_row_pipeline = make_pipeline("q4_k_get_row");
    if (!q4_k_get_row_pipeline.is_ok()) {
      [library release];
      return q4_k_get_row_pipeline.status();
    }
    impl->q4_k_get_row_pipeline = q4_k_get_row_pipeline.value();

    auto q5_k_get_row_pipeline = make_pipeline("q5_k_get_row");
    if (!q5_k_get_row_pipeline.is_ok()) {
      [library release];
      return q5_k_get_row_pipeline.status();
    }
    impl->q5_k_get_row_pipeline = q5_k_get_row_pipeline.value();

    auto q6_k_get_row_pipeline = make_pipeline("q6_k_get_row");
    if (!q6_k_get_row_pipeline.is_ok()) {
      [library release];
      return q6_k_get_row_pipeline.status();
    }
    impl->q6_k_get_row_pipeline = q6_k_get_row_pipeline.value();

    auto q4_k_get_rows_pipeline = make_pipeline("q4_k_get_rows");
    if (!q4_k_get_rows_pipeline.is_ok()) {
      [library release];
      return q4_k_get_rows_pipeline.status();
    }
    impl->q4_k_get_rows_pipeline = q4_k_get_rows_pipeline.value();

    auto q5_k_get_rows_pipeline = make_pipeline("q5_k_get_rows");
    if (!q5_k_get_rows_pipeline.is_ok()) {
      [library release];
      return q5_k_get_rows_pipeline.status();
    }
    impl->q5_k_get_rows_pipeline = q5_k_get_rows_pipeline.value();

    auto q6_k_get_rows_pipeline = make_pipeline("q6_k_get_rows");
    if (!q6_k_get_rows_pipeline.is_ok()) {
      [library release];
      return q6_k_get_rows_pipeline.status();
    }
    impl->q6_k_get_rows_pipeline = q6_k_get_rows_pipeline.value();

    auto embedding_pipeline = make_pipeline("embedding_bf16_f32");
    if (!embedding_pipeline.is_ok()) {
      [library release];
      return embedding_pipeline.status();
    }
    impl->embedding_pipeline = embedding_pipeline.value();

    auto rms_norm_pipeline = make_pipeline("rms_norm_f32_bf16");
    if (!rms_norm_pipeline.is_ok()) {
      [library release];
      return rms_norm_pipeline.status();
    }
    impl->rms_norm_pipeline = rms_norm_pipeline.value();

    auto rms_norm_f32_pipeline = make_pipeline("rms_norm_f32_f32");
    if (!rms_norm_f32_pipeline.is_ok()) {
      [library release];
      return rms_norm_f32_pipeline.status();
    }
    impl->rms_norm_f32_pipeline = rms_norm_f32_pipeline.value();

    auto rms_norm_f32_batched_pipeline = make_pipeline("rms_norm_f32_f32_batched");
    if (!rms_norm_f32_batched_pipeline.is_ok()) {
      [library release];
      return rms_norm_f32_batched_pipeline.status();
    }
    impl->rms_norm_f32_batched_pipeline = rms_norm_f32_batched_pipeline.value();

    auto qk_norm_pipeline = make_pipeline("qk_norm_f32_bf16");
    if (!qk_norm_pipeline.is_ok()) {
      [library release];
      return qk_norm_pipeline.status();
    }
    impl->qk_norm_pipeline = qk_norm_pipeline.value();

    auto qk_norm_f32_pipeline = make_pipeline("qk_norm_f32_f32");
    if (!qk_norm_f32_pipeline.is_ok()) {
      [library release];
      return qk_norm_f32_pipeline.status();
    }
    impl->qk_norm_f32_pipeline = qk_norm_f32_pipeline.value();

    auto qk_norm_f32_batched_pipeline = make_pipeline("qk_norm_f32_f32_batched");
    if (!qk_norm_f32_batched_pipeline.is_ok()) {
      [library release];
      return qk_norm_f32_batched_pipeline.status();
    }
    impl->qk_norm_f32_batched_pipeline = qk_norm_f32_batched_pipeline.value();

    auto qwen35_norm_gated_pipeline =
      make_pipeline("qwen35_norm_gated_f32_in_place");
    if (!qwen35_norm_gated_pipeline.is_ok()) {
      [library release];
      return qwen35_norm_gated_pipeline.status();
    }
    impl->qwen35_norm_gated_pipeline = qwen35_norm_gated_pipeline.value();

    auto l2_norm_pipeline = make_pipeline("l2_norm_f32_in_place");
    if (!l2_norm_pipeline.is_ok()) {
      [library release];
      return l2_norm_pipeline.status();
    }
    impl->l2_norm_pipeline = l2_norm_pipeline.value();

    auto split_qkv_l2_norm_qwen35_pipeline =
      make_pipeline("split_qkv_l2_norm_f32_qwen35");
    if (!split_qkv_l2_norm_qwen35_pipeline.is_ok()) {
      [library release];
      return split_qkv_l2_norm_qwen35_pipeline.status();
    }
    impl->split_qkv_l2_norm_qwen35_pipeline =
      split_qkv_l2_norm_qwen35_pipeline.value();

    auto gated_delta_net_pipeline = make_pipeline("gated_delta_net_f32_in_place");
    if (!gated_delta_net_pipeline.is_ok()) {
      [library release];
      return gated_delta_net_pipeline.status();
    }
    impl->gated_delta_net_pipeline = gated_delta_net_pipeline.value();

    auto gated_delta_net_batched_pipeline =
      make_pipeline("gated_delta_net_f32_batched_in_place");
    if (!gated_delta_net_batched_pipeline.is_ok()) {
      [library release];
      return gated_delta_net_batched_pipeline.status();
    }
    impl->gated_delta_net_batched_pipeline =
      gated_delta_net_batched_pipeline.value();

    auto gated_delta_net_batched_qwen35_pipeline =
      make_pipeline("gated_delta_net_f32_batched_qwen35");
    if (!gated_delta_net_batched_qwen35_pipeline.is_ok()) {
      [library release];
      return gated_delta_net_batched_qwen35_pipeline.status();
    }
    impl->gated_delta_net_batched_qwen35_pipeline =
      gated_delta_net_batched_qwen35_pipeline.value();

    auto ssm_conv_pipeline = make_pipeline("ssm_conv_f32");
    if (!ssm_conv_pipeline.is_ok()) {
      [library release];
      return ssm_conv_pipeline.status();
    }
    impl->ssm_conv_pipeline = ssm_conv_pipeline.value();

    auto build_ssm_conv_state_pipeline = make_pipeline("build_ssm_conv_state_f32");
    if (!build_ssm_conv_state_pipeline.is_ok()) {
      [library release];
      return build_ssm_conv_state_pipeline.status();
    }
    impl->build_ssm_conv_state_pipeline = build_ssm_conv_state_pipeline.value();

    auto ssm_conv1_stateful_pipeline = make_pipeline("ssm_conv1_f32_stateful");
    if (!ssm_conv1_stateful_pipeline.is_ok()) {
      [library release];
      return ssm_conv1_stateful_pipeline.status();
    }
    impl->ssm_conv1_stateful_pipeline = ssm_conv1_stateful_pipeline.value();

    auto rope_pipeline = make_pipeline("rope_f32");
    if (!rope_pipeline.is_ok()) {
      [library release];
      return rope_pipeline.status();
    }
    impl->rope_pipeline = rope_pipeline.value();

    auto mrope_pipeline = make_pipeline("mrope_f32_in_place");
    if (!mrope_pipeline.is_ok()) {
      [library release];
      return mrope_pipeline.status();
    }
    impl->mrope_pipeline = mrope_pipeline.value();

    auto add_pipeline = make_pipeline("add_f32_in_place");
    if (!add_pipeline.is_ok()) {
      [library release];
      return add_pipeline.status();
    }
    impl->add_pipeline = add_pipeline.value();

    auto mul_pipeline = make_pipeline("mul_f32_in_place");
    if (!mul_pipeline.is_ok()) {
      [library release];
      return mul_pipeline.status();
    }
    impl->mul_pipeline = mul_pipeline.value();

    auto add_row_pipeline = make_pipeline("add_f32_row_in_place");
    if (!add_row_pipeline.is_ok()) {
      [library release];
      return add_row_pipeline.status();
    }
    impl->add_row_pipeline = add_row_pipeline.value();

    auto mul_row_pipeline = make_pipeline("mul_f32_row_in_place");
    if (!mul_row_pipeline.is_ok()) {
      [library release];
      return mul_row_pipeline.status();
    }
    impl->mul_row_pipeline = mul_row_pipeline.value();

    auto sigmoid_pipeline = make_pipeline("sigmoid_f32_in_place");
    if (!sigmoid_pipeline.is_ok()) {
      [library release];
      return sigmoid_pipeline.status();
    }
    impl->sigmoid_pipeline = sigmoid_pipeline.value();

    auto softplus_pipeline = make_pipeline("softplus_f32_in_place");
    if (!softplus_pipeline.is_ok()) {
      [library release];
      return softplus_pipeline.status();
    }
    impl->softplus_pipeline = softplus_pipeline.value();

    auto prepare_qwen35_gdn_gate_beta_pipeline =
      make_pipeline("prepare_qwen35_gdn_gate_beta_f32");
    if (!prepare_qwen35_gdn_gate_beta_pipeline.is_ok()) {
      [library release];
      return prepare_qwen35_gdn_gate_beta_pipeline.status();
    }
    impl->prepare_qwen35_gdn_gate_beta_pipeline =
      prepare_qwen35_gdn_gate_beta_pipeline.value();

    auto silu_pipeline = make_pipeline("silu_f32_in_place");
    if (!silu_pipeline.is_ok()) {
      [library release];
      return silu_pipeline.status();
    }
    impl->silu_pipeline = silu_pipeline.value();

    auto silu_mul_pipeline = make_pipeline("silu_mul_f32_in_place");
    if (!silu_mul_pipeline.is_ok()) {
      [library release];
      return silu_mul_pipeline.status();
    }
    impl->silu_mul_pipeline = silu_mul_pipeline.value();

    auto copy_region_pipeline = make_pipeline("copy_f32_region");
    if (!copy_region_pipeline.is_ok()) {
      [library release];
      return copy_region_pipeline.status();
    }
    impl->copy_region_pipeline = copy_region_pipeline.value();

    auto copy_rows_pipeline = make_pipeline("copy_f32_rows");
    if (!copy_rows_pipeline.is_ok()) {
      [library release];
      return copy_rows_pipeline.status();
    }
    impl->copy_rows_pipeline = copy_rows_pipeline.value();

    auto copy_rows_to_f16_pipeline = make_pipeline("copy_f32_rows_to_f16");
    if (!copy_rows_to_f16_pipeline.is_ok()) {
      [library release];
      return copy_rows_to_f16_pipeline.status();
    }
    impl->copy_rows_to_f16_pipeline = copy_rows_to_f16_pipeline.value();

    auto copy_f16_rows_pipeline = make_pipeline("copy_f16_rows");
    if (!copy_f16_rows_pipeline.is_ok()) {
      [library release];
      return copy_f16_rows_pipeline.status();
    }
    impl->copy_f16_rows_pipeline = copy_f16_rows_pipeline.value();

    auto argmax_pipeline = make_pipeline("argmax_f32_i32");
    if (!argmax_pipeline.is_ok()) {
      [library release];
      return argmax_pipeline.status();
    }
    impl->argmax_pipeline = argmax_pipeline.value();

    auto argmax_prob_pipeline = make_pipeline("argmax_prob_f32_i32");
    if (!argmax_prob_pipeline.is_ok()) {
      [library release];
      return argmax_prob_pipeline.status();
    }
    impl->argmax_prob_pipeline = argmax_prob_pipeline.value();

    auto attention_pipeline = make_pipeline("attention_f32");
    if (!attention_pipeline.is_ok()) {
      [library release];
      return attention_pipeline.status();
    }
    impl->attention_pipeline = attention_pipeline.value();

    auto attention_f16_kv_pipeline = make_pipeline("attention_f32_f16_kv");
    if (!attention_f16_kv_pipeline.is_ok()) {
      [library release];
      return attention_f16_kv_pipeline.status();
    }
    impl->attention_f16_kv_pipeline = attention_f16_kv_pipeline.value();

    auto attention_batched_pipeline = make_pipeline("attention_f32_batched");
    if (!attention_batched_pipeline.is_ok()) {
      [library release];
      return attention_batched_pipeline.status();
    }
    impl->attention_batched_pipeline = attention_batched_pipeline.value();

    auto attention_batched_f16_kv_pipeline =
      make_pipeline("attention_f32_batched_f16_kv");
    if (!attention_batched_f16_kv_pipeline.is_ok()) {
      [library release];
      return attention_batched_f16_kv_pipeline.status();
    }
    impl->attention_batched_f16_kv_pipeline =
      attention_batched_f16_kv_pipeline.value();

    auto attention_batched_tiled_pipeline =
      make_pipeline("attention_f32_batched_tiled_16");
    if (!attention_batched_tiled_pipeline.is_ok()) {
      [library release];
      return attention_batched_tiled_pipeline.status();
    }
    impl->attention_batched_tiled_pipeline =
      attention_batched_tiled_pipeline.value();

    constexpr NSUInteger kAttentionTiled32ThreadgroupBytes =
      32U * 256U * sizeof(float) * 2U;
    if ([device maxThreadgroupMemoryLength] >= kAttentionTiled32ThreadgroupBytes) {
      auto attention_batched_tiled32_pipeline =
        make_pipeline("attention_f32_batched_tiled_32");
      if (attention_batched_tiled32_pipeline.is_ok()) {
        impl->attention_batched_tiled32_pipeline =
          attention_batched_tiled32_pipeline.value();
      }
    }

    auto attention_batched_flash256_pipeline =
      make_pipeline("attention_f32_batched_flash256");
    if (attention_batched_flash256_pipeline.is_ok()) {
      impl->attention_batched_flash256_pipeline =
        attention_batched_flash256_pipeline.value();
    }
    auto attention_batched_flash256_f16_kv_pipeline =
      make_pipeline("attention_f32_batched_flash256_f16_kv");
    if (attention_batched_flash256_f16_kv_pipeline.is_ok()) {
      impl->attention_batched_flash256_f16_kv_pipeline =
        attention_batched_flash256_f16_kv_pipeline.value();
    }
    [library release];

    return MpsContext(std::move(impl));
  }
}

bool MpsContext::valid() const {
  return impl_ != nullptr && impl_->device != nil && impl_->queue != nil &&
         impl_->matvec_pipeline != nil && impl_->q4_k_matvec_pipeline != nil &&
         impl_->q5_k_matvec_pipeline != nil && impl_->q6_k_matvec_pipeline != nil &&
         impl_->q4_k_matmul_pipeline != nil && impl_->q5_k_matmul_pipeline != nil &&
         impl_->q6_k_matmul_pipeline != nil &&
         impl_->q4_k_mul_mv_ext_r1_3_pipeline != nil &&
         impl_->q4_k_mul_mv_ext_r1_4_pipeline != nil &&
         impl_->q4_k_mul_mv_ext_r1_5_pipeline != nil &&
         impl_->q5_k_mul_mv_ext_r1_3_pipeline != nil &&
         impl_->q5_k_mul_mv_ext_r1_4_pipeline != nil &&
         impl_->q5_k_mul_mv_ext_r1_5_pipeline != nil &&
         impl_->q6_k_mul_mv_ext_r1_3_pipeline != nil &&
         impl_->q6_k_mul_mv_ext_r1_4_pipeline != nil &&
         impl_->q6_k_mul_mv_ext_r1_5_pipeline != nil &&
         impl_->q4_k_mul_mm_simd_pipeline != nil &&
         impl_->q5_k_mul_mm_simd_pipeline != nil &&
         impl_->q6_k_mul_mm_simd_pipeline != nil &&
         impl_->q4_k_get_row_pipeline != nil && impl_->q5_k_get_row_pipeline != nil &&
         impl_->q6_k_get_row_pipeline != nil &&
         impl_->q4_k_get_rows_pipeline != nil && impl_->q5_k_get_rows_pipeline != nil &&
         impl_->q6_k_get_rows_pipeline != nil &&
         impl_->embedding_pipeline != nil && impl_->rms_norm_pipeline != nil &&
         impl_->rms_norm_f32_pipeline != nil &&
         impl_->rms_norm_f32_batched_pipeline != nil &&
         impl_->qk_norm_pipeline != nil &&
         impl_->qk_norm_f32_pipeline != nil &&
         impl_->qk_norm_f32_batched_pipeline != nil &&
         impl_->qwen35_norm_gated_pipeline != nil &&
         impl_->l2_norm_pipeline != nil &&
         impl_->split_qkv_l2_norm_qwen35_pipeline != nil &&
         impl_->gated_delta_net_pipeline != nil &&
         impl_->gated_delta_net_batched_pipeline != nil &&
         impl_->gated_delta_net_batched_qwen35_pipeline != nil &&
         impl_->ssm_conv_pipeline != nil &&
         impl_->build_ssm_conv_state_pipeline != nil &&
         impl_->ssm_conv1_stateful_pipeline != nil && impl_->rope_pipeline != nil &&
         impl_->mrope_pipeline != nil &&
         impl_->add_pipeline != nil && impl_->mul_pipeline != nil &&
         impl_->add_row_pipeline != nil && impl_->mul_row_pipeline != nil &&
         impl_->sigmoid_pipeline != nil && impl_->softplus_pipeline != nil &&
         impl_->prepare_qwen35_gdn_gate_beta_pipeline != nil &&
         impl_->silu_pipeline != nil && impl_->silu_mul_pipeline != nil &&
         impl_->copy_region_pipeline != nil && impl_->copy_rows_pipeline != nil &&
         impl_->copy_rows_to_f16_pipeline != nil &&
         impl_->copy_f16_rows_pipeline != nil &&
         impl_->argmax_pipeline != nil &&
         impl_->attention_pipeline != nil &&
         impl_->attention_f16_kv_pipeline != nil &&
         impl_->attention_batched_pipeline != nil &&
         impl_->attention_batched_f16_kv_pipeline != nil &&
         impl_->attention_batched_tiled_pipeline != nil;
}

Status MpsContext::begin_graph() const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  return impl_->begin_graph();
}

Status MpsContext::commit_graph() const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  return impl_->commit_graph();
}

void MpsContext::abort_graph() const {
  if (impl_ != nullptr) {
    impl_->abort_graph();
  }
}

Result<MpsBuffer> MpsContext::make_buffer(std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (byte_size == 0) {
    return Status::invalid_argument("MPS buffer size must be greater than zero");
  }
  if (!fits_nsuinteger(byte_size)) {
    return Status::invalid_argument("MPS buffer size exceeds NSUInteger range");
  }

  @autoreleasepool {
    id<MTLBuffer> buffer =
      [impl_->device newBufferWithLength:static_cast<NSUInteger>(byte_size)
                                 options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      return Status::unavailable("failed to allocate Metal buffer");
    }

    auto impl = std::make_unique<MpsBuffer::Impl>();
    impl->buffer = buffer;
    impl->byte_size = byte_size;
    return MpsBuffer(std::move(impl));
  }
}

namespace {

Status ensure_mps_buffer_size(const MpsContext& context, MpsBuffer& buffer,
                              std::size_t byte_size, const char* name) {
  if (buffer.valid() && buffer.byte_size() >= byte_size) {
    return Status::ok();
  }
  auto replacement = context.make_buffer(byte_size);
  if (!replacement.is_ok()) {
    return Status(replacement.status().code(),
                  std::string{name} + ": " + replacement.status().message());
  }
  buffer = std::move(replacement.value());
  return Status::ok();
}

}  // namespace

Status MpsContext::copy_to_buffer(MpsBuffer& buffer, const void* data,
                                  std::size_t byte_size) const {
  return copy_to_buffer_at(buffer, 0, data, byte_size);
}

Status MpsContext::copy_to_buffer_at(MpsBuffer& buffer,
                                     std::size_t destination_offset,
                                     const void* data,
                                     std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!buffer.valid()) {
    return Status::invalid_argument("MPS buffer is not initialized");
  }
  if (data == nullptr) {
    return Status::invalid_argument("MPS copy source must not be null");
  }
  if (destination_offset > buffer.byte_size() ||
      byte_size > buffer.byte_size() - destination_offset) {
    return Status::invalid_argument("MPS copy size exceeds destination buffer size");
  }

  auto status = impl_->commit_graph();
  if (!status.is_ok()) {
    return status;
  }

  auto* destination = static_cast<std::uint8_t*>([buffer.impl_->buffer contents]);
  std::memcpy(destination + destination_offset, data, byte_size);
  return Status::ok();
}

Status MpsContext::copy_from_buffer(const MpsBuffer& buffer, void* data,
                                    std::size_t byte_size) const {
  return copy_from_buffer_at(buffer, 0, data, byte_size);
}

Status MpsContext::copy_from_buffer_at(const MpsBuffer& buffer,
                                       std::size_t source_offset,
                                       void* data,
                                       std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!buffer.valid()) {
    return Status::invalid_argument("MPS buffer is not initialized");
  }
  if (data == nullptr) {
    return Status::invalid_argument("MPS copy destination must not be null");
  }
  if (source_offset > buffer.byte_size() ||
      byte_size > buffer.byte_size() - source_offset) {
    return Status::invalid_argument("MPS copy size exceeds source buffer size");
  }
  auto status = impl_->commit_graph();
  if (!status.is_ok()) {
    return status;
  }

  const auto* source = static_cast<const std::uint8_t*>([buffer.impl_->buffer contents]);
  std::memcpy(data, source + source_offset, byte_size);
  return Status::ok();
}

Status MpsContext::copy_buffer_region(const MpsBuffer& source,
                                      MpsBuffer& destination,
                                      std::size_t source_offset,
                                      std::size_t destination_offset,
                                      std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!source.valid() || !destination.valid()) {
    return Status::invalid_argument("MPS buffer is not initialized");
  }
  if (source_offset > source.byte_size() ||
      byte_size > source.byte_size() - source_offset) {
    return Status::invalid_argument("MPS copy size exceeds source buffer size");
  }
  if (destination_offset > destination.byte_size() ||
      byte_size > destination.byte_size() - destination_offset) {
    return Status::invalid_argument("MPS copy size exceeds destination buffer size");
  }
  auto status = impl_->commit_graph();
  if (!status.is_ok()) {
    return status;
  }

  const auto* source_bytes =
    static_cast<const std::uint8_t*>([source.impl_->buffer contents]);
  auto* destination_bytes =
    static_cast<std::uint8_t*>([destination.impl_->buffer contents]);
  std::memmove(destination_bytes + destination_offset,
               source_bytes + source_offset, byte_size);
  return Status::ok();
}

Status MpsContext::zero_buffer(MpsBuffer& buffer, std::size_t byte_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!buffer.valid()) {
    return Status::invalid_argument("MPS buffer is not initialized");
  }
  if (byte_size > buffer.byte_size()) {
    return Status::invalid_argument("MPS zero size exceeds buffer size");
  }

  std::memset([buffer.impl_->buffer contents], 0, byte_size);
  return Status::ok();
}

Result<MpsMatVecWorkspace> MpsContext::make_matvec_workspace(std::size_t rows,
                                                             std::size_t cols) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto layout_result = make_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();

  auto input_buffer_result = make_buffer(layout.input_bytes);
  if (!input_buffer_result.is_ok()) {
    return input_buffer_result.status();
  }
  auto input_buffer = std::move(input_buffer_result.value());

  auto output_buffer_result = make_buffer(layout.output_bytes);
  if (!output_buffer_result.is_ok()) {
    return output_buffer_result.status();
  }
  auto output_buffer = std::move(output_buffer_result.value());

  return MpsMatVecWorkspace(rows, cols, std::move(input_buffer), std::move(output_buffer));
}

Result<std::vector<float>> MpsContext::matvec_bf16_f32(const MpsBuffer& weight,
                                                       std::size_t rows,
                                                       std::size_t cols,
                                                       const std::vector<float>& input) const {
  auto workspace_result = make_matvec_workspace(rows, cols);
  if (!workspace_result.is_ok()) {
    return workspace_result.status();
  }
  auto workspace = std::move(workspace_result.value());
  return matvec_bf16_f32(weight, workspace, input);
}

Status MpsContext::matvec_bf16_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS weight buffer is not initialized");
  }

  auto layout_result = make_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  if (weight.byte_size() < layout.weight_bytes) {
    return Status::invalid_argument("MPS matvec weight buffer is smaller than rows * cols");
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
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    [encoder setComputePipelineState:impl_->matvec_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    const auto cols_u32 = static_cast<std::uint32_t>(cols);
    [encoder setBytes:&cols_u32 length:sizeof(cols_u32) atIndex:3];
    const NSUInteger max_threads = [impl_->matvec_pipeline maxTotalThreadsPerThreadgroup];
    if (max_threads == 0) {
      return Status::internal_error("MPS matvec pipeline reported zero max threads");
    }
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    const auto threads_per_group_u32 = static_cast<std::uint32_t>(threads_per_group);
    [encoder setBytes:&threads_per_group_u32
               length:sizeof(threads_per_group_u32)
              atIndex:4];

    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer, "MPS matvec command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::matmul_f32_f32_device(const MpsBuffer& lhs,
                                         const MpsBuffer& rhs,
                                         std::size_t lhs_rows,
                                         std::size_t inner_cols,
                                         std::size_t rhs_cols,
                                         MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!lhs.valid() || !rhs.valid() || !output.valid()) {
    return Status::invalid_argument("MPS F32 matmul buffers must be initialized");
  }
  if (lhs_rows == 0 || inner_cols == 0 || rhs_cols == 0) {
    return Status::invalid_argument("MPS F32 matmul dimensions must be positive");
  }
  if (!fits_nsuinteger(lhs_rows) || !fits_nsuinteger(inner_cols) ||
      !fits_nsuinteger(rhs_cols)) {
    return Status::invalid_argument("MPS F32 matmul dimensions exceed NSUInteger range");
  }

  std::size_t lhs_values = 0;
  std::size_t rhs_values = 0;
  std::size_t output_values = 0;
  if (!checked_mul(lhs_rows, inner_cols, lhs_values) ||
      !checked_mul(inner_cols, rhs_cols, rhs_values) ||
      !checked_mul(lhs_rows, rhs_cols, output_values)) {
    return Status::invalid_argument("MPS F32 matmul value count overflow");
  }
  auto lhs_status = validate_f32_buffer(lhs, lhs_values, "F32 matmul lhs");
  if (!lhs_status.is_ok()) {
    return lhs_status;
  }
  auto rhs_status = validate_f32_buffer(rhs, rhs_values, "F32 matmul rhs");
  if (!rhs_status.is_ok()) {
    return rhs_status;
  }
  auto output_status = validate_f32_buffer(output, output_values,
                                           "F32 matmul output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }

    const auto lhs_row_bytes =
      static_cast<NSUInteger>(inner_cols * sizeof(float));
    const auto rhs_row_bytes =
      static_cast<NSUInteger>(rhs_cols * sizeof(float));
    const auto output_row_bytes =
      static_cast<NSUInteger>(rhs_cols * sizeof(float));
    MPSMatrixDescriptor* lhs_descriptor =
      [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(lhs_rows)
                                            columns:static_cast<NSUInteger>(inner_cols)
                                           rowBytes:lhs_row_bytes
                                           dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* rhs_descriptor =
      [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(inner_cols)
                                            columns:static_cast<NSUInteger>(rhs_cols)
                                           rowBytes:rhs_row_bytes
                                           dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* output_descriptor =
      [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(lhs_rows)
                                            columns:static_cast<NSUInteger>(rhs_cols)
                                           rowBytes:output_row_bytes
                                           dataType:MPSDataTypeFloat32];
    MPSMatrix* lhs_matrix =
      [[MPSMatrix alloc] initWithBuffer:lhs.impl_->buffer descriptor:lhs_descriptor];
    MPSMatrix* rhs_matrix =
      [[MPSMatrix alloc] initWithBuffer:rhs.impl_->buffer descriptor:rhs_descriptor];
    MPSMatrix* output_matrix =
      [[MPSMatrix alloc] initWithBuffer:output.impl_->buffer
                             descriptor:output_descriptor];
    MPSMatrixMultiplication* multiplication =
      [[MPSMatrixMultiplication alloc]
        initWithDevice:impl_->device
         transposeLeft:false
        transposeRight:false
            resultRows:static_cast<NSUInteger>(lhs_rows)
         resultColumns:static_cast<NSUInteger>(rhs_cols)
       interiorColumns:static_cast<NSUInteger>(inner_cols)
                 alpha:1.0
                  beta:0.0];
    [multiplication encodeToCommandBuffer:command_buffer
                               leftMatrix:lhs_matrix
                              rightMatrix:rhs_matrix
                             resultMatrix:output_matrix];
    [multiplication release];
    [output_matrix release];
    [rhs_matrix release];
    [lhs_matrix release];

    auto finish_status =
      impl_->finish_command_buffer(command_buffer, "MPS F32 matmul command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Result<std::vector<float>> MpsContext::matvec_bf16_f32(
  const MpsBuffer& weight, MpsMatVecWorkspace& workspace,
  const std::vector<float>& input) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS weight buffer is not initialized");
  }
  if (!workspace.valid()) {
    return Status::invalid_argument("MPS matvec workspace is not initialized");
  }
  const auto rows = workspace.rows();
  const auto cols = workspace.cols();
  if (input.size() != cols) {
    return Status::invalid_argument("MPS matvec input size does not match cols");
  }

  auto layout_result = make_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  if (weight.byte_size() < layout.weight_bytes) {
    return Status::invalid_argument("MPS matvec weight buffer is smaller than rows * cols");
  }
  if (workspace.input_.byte_size() < layout.input_bytes ||
      workspace.output_.byte_size() < layout.output_bytes) {
    return Status::invalid_argument("MPS matvec workspace buffers are too small");
  }
  const auto copy_status = copy_to_buffer(workspace.input_, input.data(), layout.input_bytes);
  if (!copy_status.is_ok()) {
    return copy_status;
  }

  const auto matvec_status =
    matvec_bf16_f32_device(weight, rows, cols, workspace.input_, workspace.output_);
  if (!matvec_status.is_ok()) {
    return matvec_status;
  }

  std::vector<float> output(rows);
  const auto read_status = copy_from_buffer(workspace.output_, output.data(), layout.output_bytes);
  if (!read_status.is_ok()) {
    return read_status;
  }
  return output;
}

Status MpsContext::matvec_q4_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q4_K weight buffer is not initialized");
  }
  auto layout_result = make_q4_k_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  if (weight.byte_size() < layout.weight_bytes) {
    return Status::invalid_argument("MPS Q4_K weight buffer is too small");
  }
  auto input_status = validate_f32_buffer(input, cols, "Q4_K matvec input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, rows, "Q4_K matvec output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->q4_k_matvec_pipeline maxTotalThreadsPerThreadgroup];
    if (max_threads == 0) {
      return Status::internal_error("MPS Q4_K matvec pipeline reported zero max threads");
    }
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    Q4KMatVecParams params{
      static_cast<std::uint32_t>(cols),
      static_cast<std::uint32_t>(layout.blocks_per_row),
      static_cast<std::uint32_t>(threads_per_group),
    };
    [encoder setComputePipelineState:impl_->q4_k_matvec_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS Q4_K matvec command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::matmul_q4_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          std::size_t tokens,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q4_K weight buffer is not initialized");
  }
  auto layout_result =
    make_k_quant_matmul_layout(rows, cols, tokens, 144U, "Q4_K");
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  if (weight.byte_size() < layout.weight_bytes) {
    return Status::invalid_argument("MPS Q4_K matmul weight buffer is too small");
  }
  std::size_t input_values = 0;
  std::size_t output_values = 0;
  if (!checked_mul(cols, tokens, input_values) ||
      !checked_mul(rows, tokens, output_values)) {
    return Status::invalid_argument("MPS Q4_K matmul value count overflow");
  }
  auto input_status = validate_f32_buffer(input, input_values, "Q4_K matmul input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, output_values, "Q4_K matmul output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  if (tokens >= 4U && tokens <= 8U) {
    @autoreleasepool {
      const auto r1ptg = k_quant_mul_mv_ext_r1ptg(tokens);
      id<MTLComputePipelineState> pipeline = r1ptg == 3U
        ? impl_->q4_k_mul_mv_ext_r1_3_pipeline
        : (r1ptg == 5U ? impl_->q4_k_mul_mv_ext_r1_5_pipeline
                       : impl_->q4_k_mul_mv_ext_r1_4_pipeline);
      id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
      if (command_buffer == nil) {
        return Status::unavailable("failed to create Metal command buffer");
      }
      auto encode_status = encode_k_quant_mul_mv_ext(
        command_buffer, pipeline, weight.impl_->buffer, input.impl_->buffer,
        output.impl_->buffer, layout, rows, cols, tokens, r1ptg, "Q4_K");
      if (!encode_status.is_ok()) {
        return encode_status;
      }
      return impl_->finish_command_buffer(
        command_buffer, "MPS Q4_K mul_mv_ext command failed: ");
    }
  }

  if (tokens > 8U) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
      if (command_buffer == nil) {
        return Status::unavailable("failed to create Metal command buffer");
      }
      auto encode_status = encode_k_quant_mul_mm_simd(
        command_buffer, impl_->q4_k_mul_mm_simd_pipeline, weight.impl_->buffer,
        input.impl_->buffer, output.impl_->buffer, layout, rows, cols, tokens, "Q4_K");
      if (!encode_status.is_ok()) {
        return encode_status;
      }
      return impl_->finish_command_buffer(
        command_buffer, "MPS Q4_K mul_mm simd command failed: ");
    }
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->q4_k_matmul_pipeline maxTotalThreadsPerThreadgroup];
    if (max_threads == 0) {
      return Status::internal_error("MPS Q4_K matmul pipeline reported zero max threads");
    }
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    QuantMatMulParams params{
      static_cast<std::uint32_t>(cols),
      static_cast<std::uint32_t>(layout.blocks_per_row),
      static_cast<std::uint32_t>(threads_per_group),
      static_cast<std::uint32_t>(rows),
      static_cast<std::uint32_t>(tokens),
    };
    [encoder setComputePipelineState:impl_->q4_k_matmul_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    const NSUInteger token_tile = static_cast<NSUInteger>(1);
    const NSUInteger token_groups =
      (static_cast<NSUInteger>(tokens) + token_tile - static_cast<NSUInteger>(1)) /
      token_tile;
    const MTLSize group_count =
      MTLSizeMake(static_cast<NSUInteger>(rows), token_groups, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer, "MPS Q4_K matmul command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Result<std::vector<float>> MpsContext::matvec_q4_k_f32(
  const MpsBuffer& weight, std::size_t rows, std::size_t cols,
  const std::vector<float>& input) const {
  if (input.size() != cols) {
    return Status::invalid_argument("MPS Q4_K matvec input size does not match cols");
  }
  auto layout_result = make_q4_k_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  auto input_buffer_result = make_buffer(layout.input_bytes);
  if (!input_buffer_result.is_ok()) {
    return input_buffer_result.status();
  }
  auto input_buffer = std::move(input_buffer_result.value());
  auto output_buffer_result = make_buffer(layout.output_bytes);
  if (!output_buffer_result.is_ok()) {
    return output_buffer_result.status();
  }
  auto output_buffer = std::move(output_buffer_result.value());
  const auto copy_status = copy_to_buffer(input_buffer, input.data(), layout.input_bytes);
  if (!copy_status.is_ok()) {
    return copy_status;
  }
  const auto matvec_status =
    matvec_q4_k_f32_device(weight, rows, cols, input_buffer, output_buffer);
  if (!matvec_status.is_ok()) {
    return matvec_status;
  }
  std::vector<float> output(rows);
  const auto read_status =
    copy_from_buffer(output_buffer, output.data(), layout.output_bytes);
  if (!read_status.is_ok()) {
    return read_status;
  }
  return output;
}

Status MpsContext::matvec_q5_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q5_K weight buffer is not initialized");
  }
  auto layout_result = make_q5_k_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  if (weight.byte_size() < layout.weight_bytes) {
    return Status::invalid_argument("MPS Q5_K weight buffer is too small");
  }
  auto input_status = validate_f32_buffer(input, cols, "Q5_K matvec input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, rows, "Q5_K matvec output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->q5_k_matvec_pipeline maxTotalThreadsPerThreadgroup];
    if (max_threads == 0) {
      return Status::internal_error("MPS Q5_K matvec pipeline reported zero max threads");
    }
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    Q4KMatVecParams params{
      static_cast<std::uint32_t>(cols),
      static_cast<std::uint32_t>(layout.blocks_per_row),
      static_cast<std::uint32_t>(threads_per_group),
    };
    [encoder setComputePipelineState:impl_->q5_k_matvec_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS Q5_K matvec command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::matmul_q5_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          std::size_t tokens,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q5_K weight buffer is not initialized");
  }
  auto layout_result =
    make_k_quant_matmul_layout(rows, cols, tokens, 176U, "Q5_K");
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  if (weight.byte_size() < layout.weight_bytes) {
    return Status::invalid_argument("MPS Q5_K matmul weight buffer is too small");
  }
  std::size_t input_values = 0;
  std::size_t output_values = 0;
  if (!checked_mul(cols, tokens, input_values) ||
      !checked_mul(rows, tokens, output_values)) {
    return Status::invalid_argument("MPS Q5_K matmul value count overflow");
  }
  auto input_status = validate_f32_buffer(input, input_values, "Q5_K matmul input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, output_values, "Q5_K matmul output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  if (tokens >= 4U && tokens <= 8U) {
    @autoreleasepool {
      const auto r1ptg = k_quant_mul_mv_ext_r1ptg(tokens);
      id<MTLComputePipelineState> pipeline = r1ptg == 3U
        ? impl_->q5_k_mul_mv_ext_r1_3_pipeline
        : (r1ptg == 5U ? impl_->q5_k_mul_mv_ext_r1_5_pipeline
                       : impl_->q5_k_mul_mv_ext_r1_4_pipeline);
      id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
      if (command_buffer == nil) {
        return Status::unavailable("failed to create Metal command buffer");
      }
      auto encode_status = encode_k_quant_mul_mv_ext(
        command_buffer, pipeline, weight.impl_->buffer, input.impl_->buffer,
        output.impl_->buffer, layout, rows, cols, tokens, r1ptg, "Q5_K");
      if (!encode_status.is_ok()) {
        return encode_status;
      }
      return impl_->finish_command_buffer(
        command_buffer, "MPS Q5_K mul_mv_ext command failed: ");
    }
  }

  if (tokens > 8U) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
      if (command_buffer == nil) {
        return Status::unavailable("failed to create Metal command buffer");
      }
      auto encode_status = encode_k_quant_mul_mm_simd(
        command_buffer, impl_->q5_k_mul_mm_simd_pipeline, weight.impl_->buffer,
        input.impl_->buffer, output.impl_->buffer, layout, rows, cols, tokens, "Q5_K");
      if (!encode_status.is_ok()) {
        return encode_status;
      }
      return impl_->finish_command_buffer(
        command_buffer, "MPS Q5_K mul_mm simd command failed: ");
    }
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->q5_k_matmul_pipeline maxTotalThreadsPerThreadgroup];
    if (max_threads == 0) {
      return Status::internal_error("MPS Q5_K matmul pipeline reported zero max threads");
    }
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    QuantMatMulParams params{
      static_cast<std::uint32_t>(cols),
      static_cast<std::uint32_t>(layout.blocks_per_row),
      static_cast<std::uint32_t>(threads_per_group),
      static_cast<std::uint32_t>(rows),
      static_cast<std::uint32_t>(tokens),
    };
    [encoder setComputePipelineState:impl_->q5_k_matmul_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    const NSUInteger token_tile = static_cast<NSUInteger>(1);
    const NSUInteger token_groups =
      (static_cast<NSUInteger>(tokens) + token_tile - static_cast<NSUInteger>(1)) /
      token_tile;
    const MTLSize group_count =
      MTLSizeMake(static_cast<NSUInteger>(rows), token_groups, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer, "MPS Q5_K matmul command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Result<std::vector<float>> MpsContext::matvec_q5_k_f32(
  const MpsBuffer& weight, std::size_t rows, std::size_t cols,
  const std::vector<float>& input) const {
  if (input.size() != cols) {
    return Status::invalid_argument("MPS Q5_K matvec input size does not match cols");
  }
  auto layout_result = make_q5_k_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  auto input_buffer_result = make_buffer(layout.input_bytes);
  if (!input_buffer_result.is_ok()) {
    return input_buffer_result.status();
  }
  auto input_buffer = std::move(input_buffer_result.value());
  auto output_buffer_result = make_buffer(layout.output_bytes);
  if (!output_buffer_result.is_ok()) {
    return output_buffer_result.status();
  }
  auto output_buffer = std::move(output_buffer_result.value());
  const auto copy_status = copy_to_buffer(input_buffer, input.data(), layout.input_bytes);
  if (!copy_status.is_ok()) {
    return copy_status;
  }
  const auto matvec_status =
    matvec_q5_k_f32_device(weight, rows, cols, input_buffer, output_buffer);
  if (!matvec_status.is_ok()) {
    return matvec_status;
  }
  std::vector<float> output(rows);
  const auto read_status =
    copy_from_buffer(output_buffer, output.data(), layout.output_bytes);
  if (!read_status.is_ok()) {
    return read_status;
  }
  return output;
}

Status MpsContext::matvec_q6_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q6_K weight buffer is not initialized");
  }
  auto layout_result = make_q6_k_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  if (weight.byte_size() < layout.weight_bytes) {
    return Status::invalid_argument("MPS Q6_K weight buffer is too small");
  }
  auto input_status = validate_f32_buffer(input, cols, "Q6_K matvec input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, rows, "Q6_K matvec output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->q6_k_matvec_pipeline maxTotalThreadsPerThreadgroup];
    if (max_threads == 0) {
      return Status::internal_error("MPS Q6_K matvec pipeline reported zero max threads");
    }
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    Q4KMatVecParams params{
      static_cast<std::uint32_t>(cols),
      static_cast<std::uint32_t>(layout.blocks_per_row),
      static_cast<std::uint32_t>(threads_per_group),
    };
    [encoder setComputePipelineState:impl_->q6_k_matvec_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS Q6_K matvec command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::matmul_q6_k_f32_device(const MpsBuffer& weight,
                                          std::size_t rows, std::size_t cols,
                                          std::size_t tokens,
                                          const MpsBuffer& input,
                                          MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q6_K weight buffer is not initialized");
  }
  auto layout_result =
    make_k_quant_matmul_layout(rows, cols, tokens, 210U, "Q6_K");
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  if (weight.byte_size() < layout.weight_bytes) {
    return Status::invalid_argument("MPS Q6_K matmul weight buffer is too small");
  }
  std::size_t input_values = 0;
  std::size_t output_values = 0;
  if (!checked_mul(cols, tokens, input_values) ||
      !checked_mul(rows, tokens, output_values)) {
    return Status::invalid_argument("MPS Q6_K matmul value count overflow");
  }
  auto input_status = validate_f32_buffer(input, input_values, "Q6_K matmul input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, output_values, "Q6_K matmul output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  if (tokens >= 4U && tokens <= 8U) {
    @autoreleasepool {
      const auto r1ptg = k_quant_mul_mv_ext_r1ptg(tokens);
      id<MTLComputePipelineState> pipeline = r1ptg == 3U
        ? impl_->q6_k_mul_mv_ext_r1_3_pipeline
        : (r1ptg == 5U ? impl_->q6_k_mul_mv_ext_r1_5_pipeline
                       : impl_->q6_k_mul_mv_ext_r1_4_pipeline);
      id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
      if (command_buffer == nil) {
        return Status::unavailable("failed to create Metal command buffer");
      }
      auto encode_status = encode_k_quant_mul_mv_ext(
        command_buffer, pipeline, weight.impl_->buffer, input.impl_->buffer,
        output.impl_->buffer, layout, rows, cols, tokens, r1ptg, "Q6_K");
      if (!encode_status.is_ok()) {
        return encode_status;
      }
      return impl_->finish_command_buffer(
        command_buffer, "MPS Q6_K mul_mv_ext command failed: ");
    }
  }

  if (tokens > 8U) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
      if (command_buffer == nil) {
        return Status::unavailable("failed to create Metal command buffer");
      }
      auto encode_status = encode_k_quant_mul_mm_simd(
        command_buffer, impl_->q6_k_mul_mm_simd_pipeline, weight.impl_->buffer,
        input.impl_->buffer, output.impl_->buffer, layout, rows, cols, tokens, "Q6_K");
      if (!encode_status.is_ok()) {
        return encode_status;
      }
      return impl_->finish_command_buffer(
        command_buffer, "MPS Q6_K mul_mm simd command failed: ");
    }
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->q6_k_matmul_pipeline maxTotalThreadsPerThreadgroup];
    if (max_threads == 0) {
      return Status::internal_error("MPS Q6_K matmul pipeline reported zero max threads");
    }
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    QuantMatMulParams params{
      static_cast<std::uint32_t>(cols),
      static_cast<std::uint32_t>(layout.blocks_per_row),
      static_cast<std::uint32_t>(threads_per_group),
      static_cast<std::uint32_t>(rows),
      static_cast<std::uint32_t>(tokens),
    };
    [encoder setComputePipelineState:impl_->q6_k_matmul_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    const NSUInteger token_tile = static_cast<NSUInteger>(1);
    const NSUInteger token_groups =
      (static_cast<NSUInteger>(tokens) + token_tile - static_cast<NSUInteger>(1)) /
      token_tile;
    const MTLSize group_count =
      MTLSizeMake(static_cast<NSUInteger>(rows), token_groups, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer, "MPS Q6_K matmul command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Result<std::vector<float>> MpsContext::matvec_q6_k_f32(
  const MpsBuffer& weight, std::size_t rows, std::size_t cols,
  const std::vector<float>& input) const {
  if (input.size() != cols) {
    return Status::invalid_argument("MPS Q6_K matvec input size does not match cols");
  }
  auto layout_result = make_q6_k_matvec_layout(rows, cols);
  if (!layout_result.is_ok()) {
    return layout_result.status();
  }
  const auto& layout = layout_result.value();
  auto input_buffer_result = make_buffer(layout.input_bytes);
  if (!input_buffer_result.is_ok()) {
    return input_buffer_result.status();
  }
  auto input_buffer = std::move(input_buffer_result.value());
  auto output_buffer_result = make_buffer(layout.output_bytes);
  if (!output_buffer_result.is_ok()) {
    return output_buffer_result.status();
  }
  auto output_buffer = std::move(output_buffer_result.value());
  const auto copy_status = copy_to_buffer(input_buffer, input.data(), layout.input_bytes);
  if (!copy_status.is_ok()) {
    return copy_status;
  }
  const auto matvec_status =
    matvec_q6_k_f32_device(weight, rows, cols, input_buffer, output_buffer);
  if (!matvec_status.is_ok()) {
    return matvec_status;
  }
  std::vector<float> output(rows);
  const auto read_status =
    copy_from_buffer(output_buffer, output.data(), layout.output_bytes);
  if (!read_status.is_ok()) {
    return read_status;
  }
  return output;
}

Status MpsContext::dequantize_row_q4_k_f32(const MpsBuffer& weight, std::size_t row,
                                           std::size_t cols, MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q4_K get-row weight buffer is not initialized");
  }
  if (cols == 0 || cols % 256U != 0) {
    return Status::invalid_argument("MPS Q4_K get-row cols must be positive and divisible by 256");
  }
  const auto blocks_per_row = cols / 256U;
  std::size_t row_bytes = 0;
  std::size_t required_bytes = 0;
  if (row == std::numeric_limits<std::size_t>::max() ||
      !checked_mul(blocks_per_row, static_cast<std::size_t>(144), row_bytes) ||
      !checked_mul(row + 1U, row_bytes, required_bytes)) {
    return Status::invalid_argument("MPS Q4_K get-row weight byte count overflow");
  }
  if (weight.byte_size() < required_bytes) {
    return Status::invalid_argument("MPS Q4_K get-row weight buffer is too small");
  }
  auto output_status = validate_f32_buffer(output, cols, "Q4_K get-row output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  auto row_u32 = checked_u32(row, "Q4_K get-row row");
  if (!row_u32.is_ok()) {
    return row_u32.status();
  }
  auto cols_u32 = checked_u32(cols, "Q4_K get-row cols");
  if (!cols_u32.is_ok()) {
    return cols_u32.status();
  }
  auto blocks_u32 = checked_u32(blocks_per_row, "Q4_K get-row blocks");
  if (!blocks_u32.is_ok()) {
    return blocks_u32.status();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }
    QuantGetRowParams params{row_u32.value(), cols_u32.value(), blocks_u32.value()};
    [encoder setComputePipelineState:impl_->q4_k_get_row_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads =
      [impl_->q4_k_get_row_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(cols), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS Q4_K get-row command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::dequantize_rows_q4_k_f32(const MpsBuffer& weight,
                                            std::size_t rows,
                                            const MpsBuffer& row_ids,
                                            std::size_t tokens,
                                            std::size_t cols,
                                            MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q4_K get-rows weight buffer is not initialized");
  }
  if (rows == 0 || tokens == 0 || cols == 0 || cols % 256U != 0) {
    return Status::invalid_argument("MPS Q4_K get-rows rows, tokens and cols must be positive and cols divisible by 256");
  }
  const auto blocks_per_row = cols / 256U;
  std::size_t row_bytes = 0;
  std::size_t required_bytes = 0;
  if (!checked_mul(blocks_per_row, static_cast<std::size_t>(144), row_bytes) ||
      !checked_mul(rows, row_bytes, required_bytes)) {
    return Status::invalid_argument("MPS Q4_K get-rows weight byte count overflow");
  }
  if (weight.byte_size() < required_bytes) {
    return Status::invalid_argument("MPS Q4_K get-rows weight buffer is too small");
  }
  auto ids_status = validate_i32_buffer(row_ids, tokens, "Q4_K get-rows row ids");
  if (!ids_status.is_ok()) {
    return ids_status;
  }
  std::size_t total_values = 0;
  if (!checked_mul(cols, tokens, total_values)) {
    return Status::invalid_argument("MPS Q4_K get-rows value count overflow");
  }
  auto output_status = validate_f32_buffer(output, total_values, "Q4_K get-rows output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  auto rows_u32 = checked_u32(rows, "Q4_K get-rows rows");
  if (!rows_u32.is_ok()) {
    return rows_u32.status();
  }
  auto tokens_u32 = checked_u32(tokens, "Q4_K get-rows tokens");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  auto cols_u32 = checked_u32(cols, "Q4_K get-rows cols");
  if (!cols_u32.is_ok()) {
    return cols_u32.status();
  }
  auto blocks_u32 = checked_u32(blocks_per_row, "Q4_K get-rows blocks");
  if (!blocks_u32.is_ok()) {
    return blocks_u32.status();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }
    QuantGetRowsParams params{
      rows_u32.value(),
      tokens_u32.value(),
      cols_u32.value(),
      blocks_u32.value(),
    };
    [encoder setComputePipelineState:impl_->q4_k_get_rows_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:row_ids.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const NSUInteger max_threads =
      [impl_->q4_k_get_rows_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size =
      MTLSizeMake(static_cast<NSUInteger>(total_values), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer, "MPS Q4_K get-rows command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::dequantize_row_q5_k_f32(const MpsBuffer& weight, std::size_t row,
                                           std::size_t cols, MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q5_K get-row weight buffer is not initialized");
  }
  if (cols == 0 || cols % 256U != 0) {
    return Status::invalid_argument("MPS Q5_K get-row cols must be positive and divisible by 256");
  }
  const auto blocks_per_row = cols / 256U;
  std::size_t row_bytes = 0;
  std::size_t required_bytes = 0;
  if (row == std::numeric_limits<std::size_t>::max() ||
      !checked_mul(blocks_per_row, static_cast<std::size_t>(176), row_bytes) ||
      !checked_mul(row + 1U, row_bytes, required_bytes)) {
    return Status::invalid_argument("MPS Q5_K get-row weight byte count overflow");
  }
  if (weight.byte_size() < required_bytes) {
    return Status::invalid_argument("MPS Q5_K get-row weight buffer is too small");
  }
  auto output_status = validate_f32_buffer(output, cols, "Q5_K get-row output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  auto row_u32 = checked_u32(row, "Q5_K get-row row");
  if (!row_u32.is_ok()) {
    return row_u32.status();
  }
  auto cols_u32 = checked_u32(cols, "Q5_K get-row cols");
  if (!cols_u32.is_ok()) {
    return cols_u32.status();
  }
  auto blocks_u32 = checked_u32(blocks_per_row, "Q5_K get-row blocks");
  if (!blocks_u32.is_ok()) {
    return blocks_u32.status();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }
    QuantGetRowParams params{row_u32.value(), cols_u32.value(), blocks_u32.value()};
    [encoder setComputePipelineState:impl_->q5_k_get_row_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads =
      [impl_->q5_k_get_row_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(cols), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS Q5_K get-row command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::dequantize_rows_q5_k_f32(const MpsBuffer& weight,
                                            std::size_t rows,
                                            const MpsBuffer& row_ids,
                                            std::size_t tokens,
                                            std::size_t cols,
                                            MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q5_K get-rows weight buffer is not initialized");
  }
  if (rows == 0 || tokens == 0 || cols == 0 || cols % 256U != 0) {
    return Status::invalid_argument("MPS Q5_K get-rows rows, tokens and cols must be positive and cols divisible by 256");
  }
  const auto blocks_per_row = cols / 256U;
  std::size_t row_bytes = 0;
  std::size_t required_bytes = 0;
  if (!checked_mul(blocks_per_row, static_cast<std::size_t>(176), row_bytes) ||
      !checked_mul(rows, row_bytes, required_bytes)) {
    return Status::invalid_argument("MPS Q5_K get-rows weight byte count overflow");
  }
  if (weight.byte_size() < required_bytes) {
    return Status::invalid_argument("MPS Q5_K get-rows weight buffer is too small");
  }
  auto ids_status = validate_i32_buffer(row_ids, tokens, "Q5_K get-rows row ids");
  if (!ids_status.is_ok()) {
    return ids_status;
  }
  std::size_t total_values = 0;
  if (!checked_mul(cols, tokens, total_values)) {
    return Status::invalid_argument("MPS Q5_K get-rows value count overflow");
  }
  auto output_status = validate_f32_buffer(output, total_values, "Q5_K get-rows output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  auto rows_u32 = checked_u32(rows, "Q5_K get-rows rows");
  if (!rows_u32.is_ok()) {
    return rows_u32.status();
  }
  auto tokens_u32 = checked_u32(tokens, "Q5_K get-rows tokens");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  auto cols_u32 = checked_u32(cols, "Q5_K get-rows cols");
  if (!cols_u32.is_ok()) {
    return cols_u32.status();
  }
  auto blocks_u32 = checked_u32(blocks_per_row, "Q5_K get-rows blocks");
  if (!blocks_u32.is_ok()) {
    return blocks_u32.status();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }
    QuantGetRowsParams params{
      rows_u32.value(),
      tokens_u32.value(),
      cols_u32.value(),
      blocks_u32.value(),
    };
    [encoder setComputePipelineState:impl_->q5_k_get_rows_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:row_ids.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const NSUInteger max_threads =
      [impl_->q5_k_get_rows_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size =
      MTLSizeMake(static_cast<NSUInteger>(total_values), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer, "MPS Q5_K get-rows command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::dequantize_row_q6_k_f32(const MpsBuffer& weight, std::size_t row,
                                           std::size_t cols, MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q6_K get-row weight buffer is not initialized");
  }
  if (cols == 0 || cols % 256U != 0) {
    return Status::invalid_argument("MPS Q6_K get-row cols must be positive and divisible by 256");
  }
  const auto blocks_per_row = cols / 256U;
  std::size_t row_bytes = 0;
  std::size_t required_bytes = 0;
  if (row == std::numeric_limits<std::size_t>::max() ||
      !checked_mul(blocks_per_row, static_cast<std::size_t>(210), row_bytes) ||
      !checked_mul(row + 1U, row_bytes, required_bytes)) {
    return Status::invalid_argument("MPS Q6_K get-row weight byte count overflow");
  }
  if (weight.byte_size() < required_bytes) {
    return Status::invalid_argument("MPS Q6_K get-row weight buffer is too small");
  }
  auto output_status = validate_f32_buffer(output, cols, "Q6_K get-row output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  auto row_u32 = checked_u32(row, "Q6_K get-row row");
  if (!row_u32.is_ok()) {
    return row_u32.status();
  }
  auto cols_u32 = checked_u32(cols, "Q6_K get-row cols");
  if (!cols_u32.is_ok()) {
    return cols_u32.status();
  }
  auto blocks_u32 = checked_u32(blocks_per_row, "Q6_K get-row blocks");
  if (!blocks_u32.is_ok()) {
    return blocks_u32.status();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }
    QuantGetRowParams params{row_u32.value(), cols_u32.value(), blocks_u32.value()};
    [encoder setComputePipelineState:impl_->q6_k_get_row_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads =
      [impl_->q6_k_get_row_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(cols), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS Q6_K get-row command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::dequantize_rows_q6_k_f32(const MpsBuffer& weight,
                                            std::size_t rows,
                                            const MpsBuffer& row_ids,
                                            std::size_t tokens,
                                            std::size_t cols,
                                            MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS Q6_K get-rows weight buffer is not initialized");
  }
  if (rows == 0 || tokens == 0 || cols == 0 || cols % 256U != 0) {
    return Status::invalid_argument("MPS Q6_K get-rows rows, tokens and cols must be positive and cols divisible by 256");
  }
  const auto blocks_per_row = cols / 256U;
  std::size_t row_bytes = 0;
  std::size_t required_bytes = 0;
  if (!checked_mul(blocks_per_row, static_cast<std::size_t>(210), row_bytes) ||
      !checked_mul(rows, row_bytes, required_bytes)) {
    return Status::invalid_argument("MPS Q6_K get-rows weight byte count overflow");
  }
  if (weight.byte_size() < required_bytes) {
    return Status::invalid_argument("MPS Q6_K get-rows weight buffer is too small");
  }
  auto ids_status = validate_i32_buffer(row_ids, tokens, "Q6_K get-rows row ids");
  if (!ids_status.is_ok()) {
    return ids_status;
  }
  std::size_t total_values = 0;
  if (!checked_mul(cols, tokens, total_values)) {
    return Status::invalid_argument("MPS Q6_K get-rows value count overflow");
  }
  auto output_status = validate_f32_buffer(output, total_values, "Q6_K get-rows output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  auto rows_u32 = checked_u32(rows, "Q6_K get-rows rows");
  if (!rows_u32.is_ok()) {
    return rows_u32.status();
  }
  auto tokens_u32 = checked_u32(tokens, "Q6_K get-rows tokens");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  auto cols_u32 = checked_u32(cols, "Q6_K get-rows cols");
  if (!cols_u32.is_ok()) {
    return cols_u32.status();
  }
  auto blocks_u32 = checked_u32(blocks_per_row, "Q6_K get-rows blocks");
  if (!blocks_u32.is_ok()) {
    return blocks_u32.status();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }
    QuantGetRowsParams params{
      rows_u32.value(),
      tokens_u32.value(),
      cols_u32.value(),
      blocks_u32.value(),
    };
    [encoder setComputePipelineState:impl_->q6_k_get_rows_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:row_ids.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const NSUInteger max_threads =
      [impl_->q6_k_get_rows_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size =
      MTLSizeMake(static_cast<NSUInteger>(total_values), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer, "MPS Q6_K get-rows command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::embedding_bf16_f32(const MpsBuffer& weight, std::int64_t token,
                                      std::size_t hidden_size,
                                      MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (!weight.valid()) {
    return Status::invalid_argument("MPS embedding weight buffer is not initialized");
  }
  if (token < 0) {
    return Status::invalid_argument("MPS embedding token must be non-negative");
  }
  auto hidden_u32 = checked_u32(hidden_size, "embedding hidden size");
  if (!hidden_u32.is_ok()) {
    return hidden_u32.status();
  }
  auto token_u32 = checked_u32(static_cast<std::size_t>(token), "embedding token");
  if (!token_u32.is_ok()) {
    return token_u32.status();
  }
  std::size_t token_end = 0;
  if (!checked_mul(static_cast<std::size_t>(token) + 1U, hidden_size, token_end) ||
      token_end > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
    return Status::invalid_argument("MPS embedding weight byte count overflow");
  }
  if (weight.byte_size() < token_end * sizeof(std::uint16_t)) {
    return Status::invalid_argument("MPS embedding weight buffer is too small");
  }
  auto output_status = validate_f32_buffer(output, hidden_size, "embedding output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    EmbeddingParams params{token_u32.value(), hidden_u32.value()};
    [encoder setComputePipelineState:impl_->embedding_pipeline];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->embedding_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(hidden_size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS embedding command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::rms_norm_f32_bf16(const MpsBuffer& input, const MpsBuffer& weight,
                                     std::size_t size, float eps,
                                     MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "rms norm size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto input_status = validate_f32_buffer(input, size, "rms norm input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, size, "rms norm output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  if (!weight.valid() || weight.byte_size() < size * sizeof(std::uint16_t)) {
    return Status::invalid_argument("MPS rms norm weight buffer is too small");
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->rms_norm_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    RmsNormParams params{size_u32.value(), eps,
                         static_cast<std::uint32_t>(threads_per_group)};
    [encoder setComputePipelineState:impl_->rms_norm_pipeline];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const MTLSize group_count = MTLSizeMake(1, 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS rms norm command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::rms_norm_f32_f32(const MpsBuffer& input, const MpsBuffer& weight,
                                    std::size_t size, float eps,
                                    MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "rms norm size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto input_status = validate_f32_buffer(input, size, "rms norm input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, size, "rms norm output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  auto weight_status = validate_f32_buffer(weight, size, "rms norm weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->rms_norm_f32_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    RmsNormParams params{size_u32.value(), eps,
                         static_cast<std::uint32_t>(threads_per_group)};
    [encoder setComputePipelineState:impl_->rms_norm_f32_pipeline];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const MTLSize group_count = MTLSizeMake(1, 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS rms norm f32 command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::rms_norm_f32_f32_batched(const MpsBuffer& input,
                                            const MpsBuffer& weight,
                                            std::size_t tokens,
                                            std::size_t size, float eps,
                                            MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (tokens == 0 || size == 0) {
    return Status::invalid_argument("MPS batched rms norm dimensions must be positive");
  }
  auto tokens_u32 = checked_u32(tokens, "batched rms norm tokens");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  auto size_u32 = checked_u32(size, "batched rms norm size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  std::size_t values = 0;
  if (!checked_mul(tokens, size, values)) {
    return Status::invalid_argument("MPS batched rms norm value count overflow");
  }
  auto input_status = validate_f32_buffer(input, values, "batched rms norm input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto output_status = validate_f32_buffer(output, values, "batched rms norm output");
  if (!output_status.is_ok()) {
    return output_status;
  }
  auto weight_status = validate_f32_buffer(weight, size, "batched rms norm weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->rms_norm_f32_batched_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    BatchedRmsNormParams params{
      size_u32.value(),
      tokens_u32.value(),
      eps,
      static_cast<std::uint32_t>(threads_per_group),
    };
    [encoder setComputePipelineState:impl_->rms_norm_f32_batched_pipeline];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(tokens), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer, "MPS batched rms norm command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::qk_norm_f32_bf16(MpsBuffer& values, const MpsBuffer& weight,
                                    std::size_t heads, std::size_t head_dim,
                                    float eps) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (heads == 0 || head_dim == 0) {
    return Status::invalid_argument("MPS qk norm dimensions must be positive");
  }
  auto heads_u32 = checked_u32(heads, "qk norm heads");
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  auto head_dim_u32 = checked_u32(head_dim, "qk norm head_dim");
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  auto values_status = validate_f32_buffer(values, heads * head_dim, "qk norm values");
  if (!values_status.is_ok()) {
    return values_status;
  }
  if (!weight.valid() || weight.byte_size() < head_dim * sizeof(std::uint16_t)) {
    return Status::invalid_argument("MPS qk norm weight buffer is too small");
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->qk_norm_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    QkNormParams params{heads_u32.value(), head_dim_u32.value(), eps,
                        static_cast<std::uint32_t>(threads_per_group)};
    [encoder setComputePipelineState:impl_->qk_norm_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(heads), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS qk norm command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::qk_norm_f32_f32(MpsBuffer& values, const MpsBuffer& weight,
                                   std::size_t heads, std::size_t head_dim,
                                   float eps) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (heads == 0 || head_dim == 0) {
    return Status::invalid_argument("MPS qk norm dimensions must be positive");
  }
  auto heads_u32 = checked_u32(heads, "qk norm heads");
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  auto head_dim_u32 = checked_u32(head_dim, "qk norm head_dim");
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  auto values_status = validate_f32_buffer(values, heads * head_dim, "qk norm values");
  if (!values_status.is_ok()) {
    return values_status;
  }
  auto weight_status = validate_f32_buffer(weight, head_dim, "qk norm weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->qk_norm_f32_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    QkNormParams params{heads_u32.value(), head_dim_u32.value(), eps,
                        static_cast<std::uint32_t>(threads_per_group)};
    [encoder setComputePipelineState:impl_->qk_norm_f32_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(heads), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS qk norm f32 command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::qk_norm_f32_f32_batched(MpsBuffer& values,
                                           const MpsBuffer& weight,
                                           std::size_t tokens,
                                           std::size_t heads,
                                           std::size_t head_dim,
                                           float eps) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (tokens == 0 || heads == 0 || head_dim == 0) {
    return Status::invalid_argument("MPS batched qk norm dimensions must be positive");
  }
  auto tokens_u32 = checked_u32(tokens, "batched qk norm tokens");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  auto heads_u32 = checked_u32(heads, "batched qk norm heads");
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  auto head_dim_u32 = checked_u32(head_dim, "batched qk norm head_dim");
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  std::size_t values_count = 0;
  std::size_t head_values = 0;
  if (!checked_mul(heads, head_dim, head_values) ||
      !checked_mul(tokens, head_values, values_count)) {
    return Status::invalid_argument("MPS batched qk norm value count overflow");
  }
  auto values_status = validate_f32_buffer(values, values_count, "batched qk norm values");
  if (!values_status.is_ok()) {
    return values_status;
  }
  auto weight_status = validate_f32_buffer(weight, head_dim, "batched qk norm weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->qk_norm_f32_batched_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    BatchedQkNormParams params{
      tokens_u32.value(),
      heads_u32.value(),
      head_dim_u32.value(),
      eps,
      static_cast<std::uint32_t>(threads_per_group),
    };
    [encoder setComputePipelineState:impl_->qk_norm_f32_batched_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const MTLSize group_count =
      MTLSizeMake(static_cast<NSUInteger>(heads), static_cast<NSUInteger>(tokens), 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer, "MPS batched qk norm command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::qwen35_norm_gated_f32_in_place(
    MpsBuffer& values, const MpsBuffer& weight, const MpsBuffer& gate,
    std::size_t tokens, std::size_t heads, std::size_t head_dim,
    float eps) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (tokens == 0 || heads == 0 || head_dim == 0) {
    return Status::invalid_argument("MPS Qwen3.5 norm-gated dimensions must be positive");
  }
  auto tokens_u32 = checked_u32(tokens, "Qwen3.5 norm-gated tokens");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  auto heads_u32 = checked_u32(heads, "Qwen3.5 norm-gated heads");
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  auto head_dim_u32 = checked_u32(head_dim, "Qwen3.5 norm-gated head_dim");
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  std::size_t values_count = 0;
  std::size_t head_values = 0;
  if (!checked_mul(heads, head_dim, head_values) ||
      !checked_mul(tokens, head_values, values_count)) {
    return Status::invalid_argument("MPS Qwen3.5 norm-gated value count overflow");
  }
  auto values_status =
    validate_f32_buffer(values, values_count, "Qwen3.5 norm-gated values");
  if (!values_status.is_ok()) {
    return values_status;
  }
  auto gate_status =
    validate_f32_buffer(gate, values_count, "Qwen3.5 norm-gated gate");
  if (!gate_status.is_ok()) {
    return gate_status;
  }
  auto weight_status =
    validate_f32_buffer(weight, head_dim, "Qwen3.5 norm-gated weight");
  if (!weight_status.is_ok()) {
    return weight_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->qwen35_norm_gated_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    BatchedQkNormParams params{
      tokens_u32.value(),
      heads_u32.value(),
      head_dim_u32.value(),
      eps,
      static_cast<std::uint32_t>(threads_per_group),
    };
    [encoder setComputePipelineState:impl_->qwen35_norm_gated_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:weight.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:gate.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const MTLSize group_count =
      MTLSizeMake(static_cast<NSUInteger>(heads), static_cast<NSUInteger>(tokens), 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS Qwen3.5 norm-gated command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::l2_norm_f32_in_place(MpsBuffer& values, std::size_t rows,
                                        std::size_t row_size, float eps) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (rows == 0 || row_size == 0) {
    return Status::invalid_argument("MPS l2 norm dimensions must be positive");
  }
  auto rows_u32 = checked_u32(rows, "l2 norm rows");
  if (!rows_u32.is_ok()) {
    return rows_u32.status();
  }
  auto row_size_u32 = checked_u32(row_size, "l2 norm row size");
  if (!row_size_u32.is_ok()) {
    return row_size_u32.status();
  }
  auto values_status = validate_f32_buffer(values, rows * row_size, "l2 norm values");
  if (!values_status.is_ok()) {
    return values_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads = [impl_->l2_norm_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    L2NormParams params{rows_u32.value(), row_size_u32.value(), eps,
                        static_cast<std::uint32_t>(threads_per_group)};
    [encoder setComputePipelineState:impl_->l2_norm_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS l2 norm command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::split_qkv_l2_norm_f32_qwen35(
    const MpsBuffer& source, MpsBuffer& query, MpsBuffer& key,
    MpsBuffer& value, std::size_t tokens, std::size_t key_heads,
    std::size_t value_heads, std::size_t head_dim, float eps) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (tokens == 0 || key_heads == 0 || value_heads == 0 || head_dim == 0) {
    return Status::invalid_argument("MPS Qwen3.5 split qkv dimensions must be positive");
  }

  std::size_t key_dim = 0;
  std::size_t value_dim = 0;
  std::size_t doubled_key_dim = 0;
  if (!checked_mul(key_heads, head_dim, key_dim) ||
      !checked_mul(value_heads, head_dim, value_dim) ||
      !checked_mul(static_cast<std::size_t>(2), key_dim, doubled_key_dim) ||
      doubled_key_dim > std::numeric_limits<std::size_t>::max() - value_dim) {
    return Status::invalid_argument("MPS Qwen3.5 split qkv channel count overflow");
  }
  const std::size_t conv_channels = doubled_key_dim + value_dim;

  std::size_t source_values = 0;
  std::size_t key_values = 0;
  std::size_t value_values = 0;
  if (!checked_mul(tokens, conv_channels, source_values) ||
      !checked_mul(tokens, key_dim, key_values) ||
      !checked_mul(tokens, value_dim, value_values)) {
    return Status::invalid_argument("MPS Qwen3.5 split qkv value count overflow");
  }

  auto tokens_u32 = checked_u32(tokens, "Qwen3.5 split qkv tokens");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  auto key_heads_u32 = checked_u32(key_heads, "Qwen3.5 split qkv key heads");
  if (!key_heads_u32.is_ok()) {
    return key_heads_u32.status();
  }
  auto value_heads_u32 =
    checked_u32(value_heads, "Qwen3.5 split qkv value heads");
  if (!value_heads_u32.is_ok()) {
    return value_heads_u32.status();
  }
  auto head_dim_u32 = checked_u32(head_dim, "Qwen3.5 split qkv head dim");
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  auto conv_channels_u32 =
    checked_u32(conv_channels, "Qwen3.5 split qkv conv channels");
  if (!conv_channels_u32.is_ok()) {
    return conv_channels_u32.status();
  }

  auto source_status = validate_f32_buffer(source, source_values,
                                           "Qwen3.5 split qkv source");
  if (!source_status.is_ok()) {
    return source_status;
  }
  auto query_status = validate_f32_buffer(query, key_values,
                                          "Qwen3.5 split qkv query");
  if (!query_status.is_ok()) {
    return query_status;
  }
  auto key_status = validate_f32_buffer(key, key_values, "Qwen3.5 split qkv key");
  if (!key_status.is_ok()) {
    return key_status;
  }
  auto value_status = validate_f32_buffer(value, value_values,
                                          "Qwen3.5 split qkv value");
  if (!value_status.is_ok()) {
    return value_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->split_qkv_l2_norm_qwen35_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    SplitQkvL2NormParams params{
      tokens_u32.value(),
      key_heads_u32.value(),
      value_heads_u32.value(),
      head_dim_u32.value(),
      conv_channels_u32.value(),
      eps,
      static_cast<std::uint32_t>(threads_per_group),
    };
    [encoder setComputePipelineState:impl_->split_qkv_l2_norm_qwen35_pipeline];
    [encoder setBuffer:source.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:query.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:key.impl_->buffer offset:0 atIndex:2];
    [encoder setBuffer:value.impl_->buffer offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    const MTLSize group_count =
      MTLSizeMake(static_cast<NSUInteger>(std::max(key_heads, value_heads)),
                  static_cast<NSUInteger>(tokens), static_cast<NSUInteger>(3));
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS Qwen3.5 split qkv command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::gated_delta_net_f32_in_place(
  const MpsBuffer& query, const MpsBuffer& key, const MpsBuffer& value,
  const MpsBuffer& gate, const MpsBuffer& beta, MpsBuffer& state,
  std::size_t key_heads, std::size_t value_heads, std::size_t head_dim,
  MpsBuffer& output) const {
  return gated_delta_net_f32_in_place_at(query, key, value, gate, beta, state, 0,
                                         key_heads, value_heads, head_dim, output);
}

Status MpsContext::gated_delta_net_f32_in_place_at(
  const MpsBuffer& query, const MpsBuffer& key, const MpsBuffer& value,
  const MpsBuffer& gate, const MpsBuffer& beta, MpsBuffer& state,
  std::size_t state_offset, std::size_t key_heads, std::size_t value_heads,
  std::size_t head_dim, MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (key_heads == 0 || value_heads == 0 || head_dim == 0) {
    return Status::invalid_argument("MPS gated delta net dimensions must be positive");
  }
  if (value_heads % key_heads != 0) {
    return Status::invalid_argument("MPS gated delta net value heads must repeat key heads");
  }
  if (head_dim > 256U) {
    return Status::invalid_argument("MPS gated delta net head_dim exceeds 256");
  }

  std::size_t key_values = 0;
  std::size_t value_values = 0;
  std::size_t state_values = 0;
  if (!checked_mul(key_heads, head_dim, key_values) ||
      !checked_mul(value_heads, head_dim, value_values) ||
      !checked_mul(value_values, head_dim, state_values)) {
    return Status::invalid_argument("MPS gated delta net value count overflow");
  }

  auto query_status = validate_f32_buffer(query, key_values, "gated delta net query");
  if (!query_status.is_ok()) {
    return query_status;
  }
  auto key_status = validate_f32_buffer(key, key_values, "gated delta net key");
  if (!key_status.is_ok()) {
    return key_status;
  }
  auto value_status = validate_f32_buffer(value, value_values, "gated delta net value");
  if (!value_status.is_ok()) {
    return value_status;
  }
  auto gate_status = validate_f32_buffer(gate, value_heads, "gated delta net gate");
  if (!gate_status.is_ok()) {
    return gate_status;
  }
  auto beta_status = validate_f32_buffer(beta, value_heads, "gated delta net beta");
  if (!beta_status.is_ok()) {
    return beta_status;
  }
  auto state_offset_bytes =
    validate_f32_buffer_region(state, state_offset, state_values, "gated delta net state");
  if (!state_offset_bytes.is_ok()) {
    return state_offset_bytes.status();
  }
  auto output_status = validate_f32_buffer(output, value_values,
                                           "gated delta net output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  auto key_heads_u32 = checked_u32(key_heads, "gated delta net key_heads");
  if (!key_heads_u32.is_ok()) {
    return key_heads_u32.status();
  }
  auto value_heads_u32 = checked_u32(value_heads, "gated delta net value_heads");
  if (!value_heads_u32.is_ok()) {
    return value_heads_u32.status();
  }
  auto head_dim_u32 = checked_u32(head_dim, "gated delta net head_dim");
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->gated_delta_net_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    if (threads_per_group < static_cast<NSUInteger>(head_dim)) {
      return Status::unavailable("MPS gated delta net pipeline cannot cover head_dim");
    }
    GatedDeltaNetParams params{
      key_heads_u32.value(),
      value_heads_u32.value(),
      head_dim_u32.value(),
      static_cast<std::uint32_t>(threads_per_group),
      1.0F / std::sqrt(static_cast<float>(head_dim)),
    };
    [encoder setComputePipelineState:impl_->gated_delta_net_pipeline];
    [encoder setBuffer:query.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:key.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:value.impl_->buffer offset:0 atIndex:2];
    [encoder setBuffer:gate.impl_->buffer offset:0 atIndex:3];
    [encoder setBuffer:beta.impl_->buffer offset:0 atIndex:4];
    [encoder setBuffer:state.impl_->buffer
                offset:static_cast<NSUInteger>(state_offset_bytes.value())
               atIndex:5];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:6];
    [encoder setBytes:&params length:sizeof(params) atIndex:7];
    const MTLSize group_count =
      MTLSizeMake(static_cast<NSUInteger>(head_dim),
                  static_cast<NSUInteger>(value_heads), 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS gated delta net command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::gated_delta_net_f32_batched_in_place(
  const MpsBuffer& query, const MpsBuffer& key, const MpsBuffer& value,
  const MpsBuffer& gate, const MpsBuffer& beta, MpsBuffer& state,
  std::size_t tokens, std::size_t key_heads, std::size_t value_heads,
  std::size_t head_dim, MpsBuffer& output) const {
  return gated_delta_net_f32_batched_in_place_at(
    query, key, value, gate, beta, state, 0, tokens, key_heads, value_heads,
    head_dim, output);
}

Status MpsContext::gated_delta_net_f32_batched_in_place_at(
  const MpsBuffer& query, const MpsBuffer& key, const MpsBuffer& value,
  const MpsBuffer& gate, const MpsBuffer& beta, MpsBuffer& state,
  std::size_t state_offset, std::size_t tokens, std::size_t key_heads,
  std::size_t value_heads, std::size_t head_dim, MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (tokens == 0 || key_heads == 0 || value_heads == 0 || head_dim == 0) {
    return Status::invalid_argument("MPS batched gated delta net dimensions must be positive");
  }
  if (value_heads % key_heads != 0) {
    return Status::invalid_argument("MPS batched gated delta net value heads must repeat key heads");
  }
  if (head_dim > 256U) {
    return Status::invalid_argument("MPS batched gated delta net head_dim exceeds 256");
  }

  std::size_t key_values_per_token = 0;
  std::size_t value_values_per_token = 0;
  std::size_t key_values = 0;
  std::size_t value_values = 0;
  std::size_t gate_values = 0;
  std::size_t state_values = 0;
  if (!checked_mul(key_heads, head_dim, key_values_per_token) ||
      !checked_mul(value_heads, head_dim, value_values_per_token) ||
      !checked_mul(tokens, key_values_per_token, key_values) ||
      !checked_mul(tokens, value_values_per_token, value_values) ||
      !checked_mul(tokens, value_heads, gate_values) ||
      !checked_mul(value_values_per_token, head_dim, state_values)) {
    return Status::invalid_argument("MPS batched gated delta net value count overflow");
  }

  auto query_status = validate_f32_buffer(query, key_values,
                                          "batched gated delta net query");
  if (!query_status.is_ok()) {
    return query_status;
  }
  auto key_status = validate_f32_buffer(key, key_values,
                                        "batched gated delta net key");
  if (!key_status.is_ok()) {
    return key_status;
  }
  auto value_status = validate_f32_buffer(value, value_values,
                                          "batched gated delta net value");
  if (!value_status.is_ok()) {
    return value_status;
  }
  auto gate_status = validate_f32_buffer(gate, gate_values,
                                         "batched gated delta net gate");
  if (!gate_status.is_ok()) {
    return gate_status;
  }
  auto beta_status = validate_f32_buffer(beta, gate_values,
                                         "batched gated delta net beta");
  if (!beta_status.is_ok()) {
    return beta_status;
  }
  auto state_offset_bytes =
    validate_f32_buffer_region(state, state_offset, state_values,
                               "batched gated delta net state");
  if (!state_offset_bytes.is_ok()) {
    return state_offset_bytes.status();
  }
  auto output_status = validate_f32_buffer(output, value_values,
                                           "batched gated delta net output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  auto tokens_u32 = checked_u32(tokens, "batched gated delta net tokens");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  auto key_heads_u32 = checked_u32(key_heads, "batched gated delta net key_heads");
  if (!key_heads_u32.is_ok()) {
    return key_heads_u32.status();
  }
  auto value_heads_u32 =
    checked_u32(value_heads, "batched gated delta net value_heads");
  if (!value_heads_u32.is_ok()) {
    return value_heads_u32.status();
  }
  auto head_dim_u32 = checked_u32(head_dim, "batched gated delta net head_dim");
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }

  if (head_dim == 128U && key_heads == value_heads) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
      if (command_buffer == nil) {
        return Status::unavailable("failed to create Metal command buffer");
      }
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (encoder == nil) {
        return Status::unavailable("failed to create Metal compute encoder");
      }

      BatchedGatedDeltaNetParams params{
        tokens_u32.value(),
        key_heads_u32.value(),
        value_heads_u32.value(),
        head_dim_u32.value(),
        128U,
        1.0F / std::sqrt(static_cast<float>(head_dim)),
      };
      [encoder setComputePipelineState:impl_->gated_delta_net_batched_qwen35_pipeline];
      [encoder setBuffer:query.impl_->buffer offset:0 atIndex:0];
      [encoder setBuffer:key.impl_->buffer offset:0 atIndex:1];
      [encoder setBuffer:value.impl_->buffer offset:0 atIndex:2];
      [encoder setBuffer:gate.impl_->buffer offset:0 atIndex:3];
      [encoder setBuffer:beta.impl_->buffer offset:0 atIndex:4];
      [encoder setBuffer:state.impl_->buffer
                  offset:static_cast<NSUInteger>(state_offset_bytes.value())
                 atIndex:5];
      [encoder setBuffer:output.impl_->buffer offset:0 atIndex:6];
      [encoder setBytes:&params length:sizeof(params) atIndex:7];
      const MTLSize group_count =
        MTLSizeMake(static_cast<NSUInteger>(head_dim / 4U),
                    static_cast<NSUInteger>(value_heads), 1);
      const MTLSize threadgroup_size = MTLSizeMake(32, 4, 1);
      [encoder dispatchThreadgroups:group_count
               threadsPerThreadgroup:threadgroup_size];
      [encoder endEncoding];
      auto finish_status =
        impl_->finish_command_buffer(
          command_buffer, "MPS Qwen3.5 batched gated delta net command failed: ");
      if (!finish_status.is_ok()) {
        return finish_status;
      }
    }
    return Status::ok();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->gated_delta_net_batched_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    if (threads_per_group < static_cast<NSUInteger>(head_dim)) {
      return Status::unavailable("MPS batched gated delta net pipeline cannot cover head_dim");
    }
    BatchedGatedDeltaNetParams params{
      tokens_u32.value(),
      key_heads_u32.value(),
      value_heads_u32.value(),
      head_dim_u32.value(),
      static_cast<std::uint32_t>(threads_per_group),
      1.0F / std::sqrt(static_cast<float>(head_dim)),
    };
    [encoder setComputePipelineState:impl_->gated_delta_net_batched_pipeline];
    [encoder setBuffer:query.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:key.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:value.impl_->buffer offset:0 atIndex:2];
    [encoder setBuffer:gate.impl_->buffer offset:0 atIndex:3];
    [encoder setBuffer:beta.impl_->buffer offset:0 atIndex:4];
    [encoder setBuffer:state.impl_->buffer
                offset:static_cast<NSUInteger>(state_offset_bytes.value())
               atIndex:5];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:6];
    [encoder setBytes:&params length:sizeof(params) atIndex:7];
    const MTLSize group_count =
      MTLSizeMake(static_cast<NSUInteger>(head_dim),
                  static_cast<NSUInteger>(value_heads), 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS batched gated delta net command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::ssm_conv_f32(const MpsBuffer& input, const MpsBuffer& kernel,
                                std::size_t conv_kernel, std::size_t channels,
                                std::size_t tokens, std::size_t sequences,
                                MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (conv_kernel == 0 || channels == 0 || tokens == 0 || sequences == 0) {
    return Status::invalid_argument("MPS SSM conv dimensions must be positive");
  }
  std::size_t input_span = 0;
  if (conv_kernel > std::numeric_limits<std::size_t>::max() - tokens) {
    return Status::invalid_argument("MPS SSM conv input span overflow");
  }
  input_span = conv_kernel - 1U + tokens;

  std::size_t input_values = 0;
  std::size_t kernel_values = 0;
  std::size_t output_values = 0;
  if (!checked_mul(input_span, channels, input_values) ||
      !checked_mul(input_values, sequences, input_values) ||
      !checked_mul(conv_kernel, channels, kernel_values) ||
      !checked_mul(channels, tokens, output_values) ||
      !checked_mul(output_values, sequences, output_values)) {
    return Status::invalid_argument("MPS SSM conv value count overflow");
  }

  auto input_status = validate_f32_buffer(input, input_values, "SSM conv input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto kernel_status = validate_f32_buffer(kernel, kernel_values, "SSM conv kernel");
  if (!kernel_status.is_ok()) {
    return kernel_status;
  }
  auto output_status = validate_f32_buffer(output, output_values, "SSM conv output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  auto conv_kernel_u32 = checked_u32(conv_kernel, "SSM conv kernel");
  if (!conv_kernel_u32.is_ok()) {
    return conv_kernel_u32.status();
  }
  auto input_span_u32 = checked_u32(input_span, "SSM conv input span");
  if (!input_span_u32.is_ok()) {
    return input_span_u32.status();
  }
  auto channels_u32 = checked_u32(channels, "SSM conv channels");
  if (!channels_u32.is_ok()) {
    return channels_u32.status();
  }
  auto tokens_u32 = checked_u32(tokens, "SSM conv tokens");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  auto sequences_u32 = checked_u32(sequences, "SSM conv sequences");
  if (!sequences_u32.is_ok()) {
    return sequences_u32.status();
  }
  auto total_u32 = checked_u32(output_values, "SSM conv output values");
  if (!total_u32.is_ok()) {
    return total_u32.status();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    SsmConvParams params{conv_kernel_u32.value(), input_span_u32.value(),
                         channels_u32.value(), tokens_u32.value(),
                         sequences_u32.value(), total_u32.value()};
    [encoder setComputePipelineState:impl_->ssm_conv_pipeline];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:kernel.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const NSUInteger max_threads = [impl_->ssm_conv_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(output_values), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS SSM conv command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::build_ssm_conv_state_f32(MpsBuffer& state,
                                            std::size_t state_offset,
                                            const MpsBuffer& input,
                                            std::size_t conv_kernel,
                                            std::size_t channels,
                                            std::size_t tokens,
                                            MpsBuffer& conv_input) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (conv_kernel < 2 || channels == 0 || tokens == 0) {
    return Status::invalid_argument("MPS SSM conv-state dimensions are invalid");
  }
  if (conv_kernel - 1U > std::numeric_limits<std::size_t>::max() - tokens) {
    return Status::invalid_argument("MPS SSM conv-state input span overflow");
  }
  const std::size_t state_width = conv_kernel - 1U;
  const std::size_t input_span = state_width + tokens;

  std::size_t state_values = 0;
  std::size_t input_values = 0;
  std::size_t conv_input_values = 0;
  if (!checked_mul(state_width, channels, state_values) ||
      !checked_mul(tokens, channels, input_values) ||
      !checked_mul(input_span, channels, conv_input_values)) {
    return Status::invalid_argument("MPS SSM conv-state value count overflow");
  }

  auto state_offset_bytes =
    validate_f32_buffer_region(state, state_offset, state_values,
                               "SSM conv-state state");
  if (!state_offset_bytes.is_ok()) {
    return state_offset_bytes.status();
  }
  auto input_status = validate_f32_buffer(input, input_values,
                                          "SSM conv-state input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto conv_input_status = validate_f32_buffer(conv_input, conv_input_values,
                                               "SSM conv-state conv input");
  if (!conv_input_status.is_ok()) {
    return conv_input_status;
  }

  auto conv_kernel_u32 = checked_u32(conv_kernel, "SSM conv-state kernel");
  if (!conv_kernel_u32.is_ok()) {
    return conv_kernel_u32.status();
  }
  auto channels_u32 = checked_u32(channels, "SSM conv-state channels");
  if (!channels_u32.is_ok()) {
    return channels_u32.status();
  }
  auto tokens_u32 = checked_u32(tokens, "SSM conv-state tokens");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  auto input_span_u32 = checked_u32(input_span, "SSM conv-state input span");
  if (!input_span_u32.is_ok()) {
    return input_span_u32.status();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    BuildSsmConvStateParams params{
      conv_kernel_u32.value(),
      channels_u32.value(),
      tokens_u32.value(),
      input_span_u32.value(),
    };
    [encoder setComputePipelineState:impl_->build_ssm_conv_state_pipeline];
    [encoder setBuffer:state.impl_->buffer
                offset:static_cast<NSUInteger>(state_offset_bytes.value())
               atIndex:0];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:conv_input.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const NSUInteger max_threads =
      [impl_->build_ssm_conv_state_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(channels), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS SSM conv-state command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::ssm_conv1_f32_stateful(MpsBuffer& state,
                                          const MpsBuffer& input,
                                          const MpsBuffer& kernel,
                                          std::size_t conv_kernel,
                                          std::size_t channels,
                                          MpsBuffer& output) const {
  return ssm_conv1_f32_stateful_at(state, 0, input, kernel, conv_kernel,
                                   channels, output);
}

Status MpsContext::ssm_conv1_f32_stateful_at(MpsBuffer& state,
                                             std::size_t state_offset,
                                             const MpsBuffer& input,
                                             const MpsBuffer& kernel,
                                             std::size_t conv_kernel,
                                             std::size_t channels,
                                             MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (conv_kernel < 2 || channels == 0) {
    return Status::invalid_argument("MPS stateful SSM conv dimensions are invalid");
  }

  std::size_t state_values = 0;
  std::size_t kernel_values = 0;
  if (!checked_mul(conv_kernel - 1U, channels, state_values) ||
      !checked_mul(conv_kernel, channels, kernel_values)) {
    return Status::invalid_argument("MPS stateful SSM conv value count overflow");
  }

  auto state_offset_bytes =
    validate_f32_buffer_region(state, state_offset, state_values,
                               "stateful SSM conv state");
  if (!state_offset_bytes.is_ok()) {
    return state_offset_bytes.status();
  }
  auto input_status = validate_f32_buffer(input, channels, "stateful SSM conv input");
  if (!input_status.is_ok()) {
    return input_status;
  }
  auto kernel_status = validate_f32_buffer(kernel, kernel_values,
                                           "stateful SSM conv kernel");
  if (!kernel_status.is_ok()) {
    return kernel_status;
  }
  auto output_status = validate_f32_buffer(output, channels,
                                           "stateful SSM conv output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  auto conv_kernel_u32 = checked_u32(conv_kernel, "stateful SSM conv kernel");
  if (!conv_kernel_u32.is_ok()) {
    return conv_kernel_u32.status();
  }
  auto channels_u32 = checked_u32(channels, "stateful SSM conv channels");
  if (!channels_u32.is_ok()) {
    return channels_u32.status();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    SsmConv1StatefulParams params{conv_kernel_u32.value(), channels_u32.value()};
    [encoder setComputePipelineState:impl_->ssm_conv1_stateful_pipeline];
    [encoder setBuffer:state.impl_->buffer
                offset:static_cast<NSUInteger>(state_offset_bytes.value())
               atIndex:0];
    [encoder setBuffer:input.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:kernel.impl_->buffer offset:0 atIndex:2];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    const NSUInteger max_threads =
      [impl_->ssm_conv1_stateful_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(channels), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS stateful SSM conv command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::rope_f32(MpsBuffer& values, std::size_t heads,
                            std::size_t head_dim, std::size_t position,
                            float theta) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (heads == 0 || head_dim == 0 || head_dim % 2U != 0) {
    return Status::invalid_argument("MPS RoPE dimensions are invalid");
  }
  auto heads_u32 = checked_u32(heads, "RoPE heads");
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  auto head_dim_u32 = checked_u32(head_dim, "RoPE head_dim");
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  auto position_u32 = checked_u32(position, "RoPE position");
  if (!position_u32.is_ok()) {
    return position_u32.status();
  }
  auto values_status = validate_f32_buffer(values, heads * head_dim, "RoPE values");
  if (!values_status.is_ok()) {
    return values_status;
  }
  const auto total = heads * (head_dim / 2U);

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    RopeParams params{heads_u32.value(), head_dim_u32.value(), position_u32.value(),
                      theta};
    [encoder setComputePipelineState:impl_->rope_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    const NSUInteger max_threads = [impl_->rope_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(total), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS RoPE command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::mrope_f32_in_place(MpsBuffer& values, const MpsBuffer& positions,
                                      std::size_t tokens, std::size_t heads,
                                      std::size_t head_dim, std::size_t n_dims,
                                      std::size_t section_0, std::size_t section_1,
                                      std::size_t section_2, std::size_t section_3,
                                      float theta) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (tokens == 0 || heads == 0 || head_dim == 0 || n_dims == 0 ||
      n_dims > head_dim || n_dims % 2U != 0) {
    return Status::invalid_argument("MPS MRoPE dimensions are invalid");
  }
  if (section_0 + section_1 + section_2 + section_3 == 0) {
    return Status::invalid_argument("MPS MRoPE sections must not all be zero");
  }

  std::size_t value_count = 0;
  if (!checked_mul(tokens, heads, value_count) ||
      !checked_mul(value_count, head_dim, value_count)) {
    return Status::invalid_argument("MPS MRoPE value count overflow");
  }
  auto values_status = validate_f32_buffer(values, value_count, "MRoPE values");
  if (!values_status.is_ok()) {
    return values_status;
  }
  std::size_t position_count = 0;
  if (!checked_mul(tokens, static_cast<std::size_t>(4), position_count) ||
      position_count > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
    return Status::invalid_argument("MPS MRoPE position count overflow");
  }
  if (!positions.valid() || positions.byte_size() < position_count * sizeof(std::int32_t)) {
    return Status::invalid_argument("MPS MRoPE position buffer is too small");
  }

  auto tokens_u32 = checked_u32(tokens, "MRoPE tokens");
  auto heads_u32 = checked_u32(heads, "MRoPE heads");
  auto head_dim_u32 = checked_u32(head_dim, "MRoPE head dim");
  auto n_dims_u32 = checked_u32(n_dims, "MRoPE n_dims");
  auto section_0_u32 = checked_u32(section_0, "MRoPE section 0");
  auto section_1_u32 = checked_u32(section_1, "MRoPE section 1");
  auto section_2_u32 = checked_u32(section_2, "MRoPE section 2");
  auto section_3_u32 = checked_u32(section_3, "MRoPE section 3");
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  if (!n_dims_u32.is_ok()) {
    return n_dims_u32.status();
  }
  if (!section_0_u32.is_ok()) {
    return section_0_u32.status();
  }
  if (!section_1_u32.is_ok()) {
    return section_1_u32.status();
  }
  if (!section_2_u32.is_ok()) {
    return section_2_u32.status();
  }
  if (!section_3_u32.is_ok()) {
    return section_3_u32.status();
  }
  std::size_t pair_count = 0;
  if (!checked_mul(tokens, heads, pair_count) ||
      !checked_mul(pair_count, n_dims / 2U, pair_count)) {
    return Status::invalid_argument("MPS MRoPE pair count overflow");
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    MropeParams params{tokens_u32.value(),    heads_u32.value(),
                       head_dim_u32.value(),  n_dims_u32.value(),
                       section_0_u32.value(), section_1_u32.value(),
                       section_2_u32.value(), section_3_u32.value(),
                       theta};
    [encoder setComputePipelineState:impl_->mrope_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:positions.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->mrope_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(pair_count), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS MRoPE command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::add_f32_in_place(MpsBuffer& target, const MpsBuffer& delta,
                                    std::size_t size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "add size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto target_status = validate_f32_buffer(target, size, "add target");
  if (!target_status.is_ok()) {
    return target_status;
  }
  auto delta_status = validate_f32_buffer(delta, size, "add delta");
  if (!delta_status.is_ok()) {
    return delta_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    SizeParams params{size_u32.value()};
    [encoder setComputePipelineState:impl_->add_pipeline];
    [encoder setBuffer:target.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:delta.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->add_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS add command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::mul_f32_in_place(MpsBuffer& target, const MpsBuffer& rhs,
                                    std::size_t size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "mul size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto target_status = validate_f32_buffer(target, size, "mul target");
  if (!target_status.is_ok()) {
    return target_status;
  }
  auto rhs_status = validate_f32_buffer(rhs, size, "mul rhs");
  if (!rhs_status.is_ok()) {
    return rhs_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    SizeParams params{size_u32.value()};
    [encoder setComputePipelineState:impl_->mul_pipeline];
    [encoder setBuffer:target.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:rhs.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->mul_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS mul command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::add_f32_row_in_place(MpsBuffer& target, const MpsBuffer& row,
                                        std::size_t rows,
                                        std::size_t row_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (rows == 0 || row_size == 0) {
    return Status::invalid_argument("MPS rowwise add dimensions must be positive");
  }
  std::size_t total = 0;
  if (!checked_mul(rows, row_size, total)) {
    return Status::invalid_argument("MPS rowwise add value count overflow");
  }
  auto row_size_u32 = checked_u32(row_size, "rowwise add row size");
  if (!row_size_u32.is_ok()) {
    return row_size_u32.status();
  }
  auto total_u32 = checked_u32(total, "rowwise add total");
  if (!total_u32.is_ok()) {
    return total_u32.status();
  }
  auto target_status = validate_f32_buffer(target, total, "rowwise add target");
  if (!target_status.is_ok()) {
    return target_status;
  }
  auto row_status = validate_f32_buffer(row, row_size, "rowwise add row");
  if (!row_status.is_ok()) {
    return row_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    RowwiseParams params{row_size_u32.value(), total_u32.value()};
    [encoder setComputePipelineState:impl_->add_row_pipeline];
    [encoder setBuffer:target.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:row.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->add_row_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(total), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS rowwise add command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::mul_f32_row_in_place(MpsBuffer& target, const MpsBuffer& row,
                                        std::size_t rows,
                                        std::size_t row_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (rows == 0 || row_size == 0) {
    return Status::invalid_argument("MPS rowwise mul dimensions must be positive");
  }
  std::size_t total = 0;
  if (!checked_mul(rows, row_size, total)) {
    return Status::invalid_argument("MPS rowwise mul value count overflow");
  }
  auto row_size_u32 = checked_u32(row_size, "rowwise mul row size");
  if (!row_size_u32.is_ok()) {
    return row_size_u32.status();
  }
  auto total_u32 = checked_u32(total, "rowwise mul total");
  if (!total_u32.is_ok()) {
    return total_u32.status();
  }
  auto target_status = validate_f32_buffer(target, total, "rowwise mul target");
  if (!target_status.is_ok()) {
    return target_status;
  }
  auto row_status = validate_f32_buffer(row, row_size, "rowwise mul row");
  if (!row_status.is_ok()) {
    return row_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    RowwiseParams params{row_size_u32.value(), total_u32.value()};
    [encoder setComputePipelineState:impl_->mul_row_pipeline];
    [encoder setBuffer:target.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:row.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->mul_row_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(total), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS rowwise mul command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::sigmoid_f32_in_place(MpsBuffer& values, std::size_t size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "sigmoid size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto values_status = validate_f32_buffer(values, size, "sigmoid values");
  if (!values_status.is_ok()) {
    return values_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    SizeParams params{size_u32.value()};
    [encoder setComputePipelineState:impl_->sigmoid_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    const NSUInteger max_threads = [impl_->sigmoid_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS sigmoid command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::softplus_f32_in_place(MpsBuffer& values, std::size_t size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "softplus size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto values_status = validate_f32_buffer(values, size, "softplus values");
  if (!values_status.is_ok()) {
    return values_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    SizeParams params{size_u32.value()};
    [encoder setComputePipelineState:impl_->softplus_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    const NSUInteger max_threads = [impl_->softplus_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS softplus command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::prepare_qwen35_gdn_gate_beta_f32(
    MpsBuffer& gate, MpsBuffer& beta, const MpsBuffer& gate_bias,
    const MpsBuffer& gate_scale, std::size_t rows,
    std::size_t row_size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (rows == 0 || row_size == 0) {
    return Status::invalid_argument("MPS Qwen3.5 GDN gate dimensions must be positive");
  }
  std::size_t total = 0;
  if (!checked_mul(rows, row_size, total)) {
    return Status::invalid_argument("MPS Qwen3.5 GDN gate value count overflow");
  }
  auto row_size_u32 = checked_u32(row_size, "Qwen3.5 GDN gate row size");
  if (!row_size_u32.is_ok()) {
    return row_size_u32.status();
  }
  auto total_u32 = checked_u32(total, "Qwen3.5 GDN gate total");
  if (!total_u32.is_ok()) {
    return total_u32.status();
  }
  auto gate_status = validate_f32_buffer(gate, total, "Qwen3.5 GDN gate");
  if (!gate_status.is_ok()) {
    return gate_status;
  }
  auto beta_status = validate_f32_buffer(beta, total, "Qwen3.5 GDN beta");
  if (!beta_status.is_ok()) {
    return beta_status;
  }
  auto bias_status =
    validate_f32_buffer(gate_bias, row_size, "Qwen3.5 GDN gate bias");
  if (!bias_status.is_ok()) {
    return bias_status;
  }
  auto scale_status =
    validate_f32_buffer(gate_scale, row_size, "Qwen3.5 GDN gate scale");
  if (!scale_status.is_ok()) {
    return scale_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    RowwiseParams params{row_size_u32.value(), total_u32.value()};
    [encoder setComputePipelineState:impl_->prepare_qwen35_gdn_gate_beta_pipeline];
    [encoder setBuffer:gate.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:beta.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:gate_bias.impl_->buffer offset:0 atIndex:2];
    [encoder setBuffer:gate_scale.impl_->buffer offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    const NSUInteger max_threads =
      [impl_->prepare_qwen35_gdn_gate_beta_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(total), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS Qwen3.5 GDN gate command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::silu_f32_in_place(MpsBuffer& values, std::size_t size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "silu size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto values_status = validate_f32_buffer(values, size, "silu values");
  if (!values_status.is_ok()) {
    return values_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    SizeParams params{size_u32.value()};
    [encoder setComputePipelineState:impl_->silu_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    const NSUInteger max_threads = [impl_->silu_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS silu command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::silu_mul_f32_in_place(MpsBuffer& gate, const MpsBuffer& up,
                                         std::size_t size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto size_u32 = checked_u32(size, "silu mul size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto gate_status = validate_f32_buffer(gate, size, "silu gate");
  if (!gate_status.is_ok()) {
    return gate_status;
  }
  auto up_status = validate_f32_buffer(up, size, "silu up");
  if (!up_status.is_ok()) {
    return up_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    SizeParams params{size_u32.value()};
    [encoder setComputePipelineState:impl_->silu_mul_pipeline];
    [encoder setBuffer:gate.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:up.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->silu_mul_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS silu mul command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::copy_f32_region(const MpsBuffer& source, MpsBuffer& destination,
                                   std::size_t source_offset,
                                   std::size_t destination_offset,
                                   std::size_t size) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto source_offset_u32 = checked_u32(source_offset, "copy source offset");
  if (!source_offset_u32.is_ok()) {
    return source_offset_u32.status();
  }
  auto destination_offset_u32 = checked_u32(destination_offset, "copy destination offset");
  if (!destination_offset_u32.is_ok()) {
    return destination_offset_u32.status();
  }
  auto size_u32 = checked_u32(size, "copy size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto source_status = validate_f32_buffer(source, source_offset + size, "copy source");
  if (!source_status.is_ok()) {
    return source_status;
  }
  auto destination_status =
    validate_f32_buffer(destination, destination_offset + size, "copy destination");
  if (!destination_status.is_ok()) {
    return destination_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    CopyRegionParams params{source_offset_u32.value(), destination_offset_u32.value(),
                            size_u32.value()};
    [encoder setComputePipelineState:impl_->copy_region_pipeline];
    [encoder setBuffer:source.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:destination.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->copy_region_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS copy region command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::copy_f32_rows(const MpsBuffer& source, MpsBuffer& destination,
                                 std::size_t rows, std::size_t row_size,
                                 std::size_t source_stride,
                                 std::size_t source_offset,
                                 std::size_t destination_stride,
                                 std::size_t destination_offset) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto source_end = strided_f32_region_end(source_offset, rows, source_stride,
                                           row_size, "copy rows source");
  if (!source_end.is_ok()) {
    return source_end.status();
  }
  auto destination_end =
    strided_f32_region_end(destination_offset, rows, destination_stride,
                           row_size, "copy rows destination");
  if (!destination_end.is_ok()) {
    return destination_end.status();
  }
  std::size_t total = 0;
  if (!checked_mul(rows, row_size, total)) {
    return Status::invalid_argument("MPS copy rows value count overflow");
  }

  auto source_offset_u32 = checked_u32(source_offset, "copy rows source offset");
  if (!source_offset_u32.is_ok()) {
    return source_offset_u32.status();
  }
  auto destination_offset_u32 =
    checked_u32(destination_offset, "copy rows destination offset");
  if (!destination_offset_u32.is_ok()) {
    return destination_offset_u32.status();
  }
  auto source_stride_u32 = checked_u32(source_stride, "copy rows source stride");
  if (!source_stride_u32.is_ok()) {
    return source_stride_u32.status();
  }
  auto destination_stride_u32 =
    checked_u32(destination_stride, "copy rows destination stride");
  if (!destination_stride_u32.is_ok()) {
    return destination_stride_u32.status();
  }
  auto row_size_u32 = checked_u32(row_size, "copy rows row size");
  if (!row_size_u32.is_ok()) {
    return row_size_u32.status();
  }
  auto total_u32 = checked_u32(total, "copy rows total");
  if (!total_u32.is_ok()) {
    return total_u32.status();
  }

  auto source_status = validate_f32_buffer(source, source_end.value(),
                                           "copy rows source");
  if (!source_status.is_ok()) {
    return source_status;
  }
  auto destination_status =
    validate_f32_buffer(destination, destination_end.value(),
                        "copy rows destination");
  if (!destination_status.is_ok()) {
    return destination_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    CopyRowsParams params{
      source_offset_u32.value(),
      destination_offset_u32.value(),
      source_stride_u32.value(),
      destination_stride_u32.value(),
      row_size_u32.value(),
      total_u32.value(),
    };
    [encoder setComputePipelineState:impl_->copy_rows_pipeline];
    [encoder setBuffer:source.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:destination.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads = [impl_->copy_rows_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(total), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS copy rows command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::copy_f32_rows_to_f16(
  const MpsBuffer& source, MpsBuffer& destination, std::size_t rows,
  std::size_t row_size, std::size_t source_stride,
  std::size_t source_offset, std::size_t destination_stride,
  std::size_t destination_offset) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto source_end = strided_f32_region_end(source_offset, rows, source_stride,
                                           row_size, "copy rows f32->f16 source");
  if (!source_end.is_ok()) {
    return source_end.status();
  }
  auto destination_end =
    strided_f32_region_end(destination_offset, rows, destination_stride,
                           row_size, "copy rows f32->f16 destination");
  if (!destination_end.is_ok()) {
    return destination_end.status();
  }
  std::size_t total = 0;
  if (!checked_mul(rows, row_size, total)) {
    return Status::invalid_argument("MPS copy rows f32->f16 value count overflow");
  }

  auto source_offset_u32 = checked_u32(source_offset, "copy rows f32->f16 source offset");
  if (!source_offset_u32.is_ok()) {
    return source_offset_u32.status();
  }
  auto destination_offset_u32 =
    checked_u32(destination_offset, "copy rows f32->f16 destination offset");
  if (!destination_offset_u32.is_ok()) {
    return destination_offset_u32.status();
  }
  auto source_stride_u32 = checked_u32(source_stride, "copy rows f32->f16 source stride");
  if (!source_stride_u32.is_ok()) {
    return source_stride_u32.status();
  }
  auto destination_stride_u32 =
    checked_u32(destination_stride, "copy rows f32->f16 destination stride");
  if (!destination_stride_u32.is_ok()) {
    return destination_stride_u32.status();
  }
  auto row_size_u32 = checked_u32(row_size, "copy rows f32->f16 row size");
  if (!row_size_u32.is_ok()) {
    return row_size_u32.status();
  }
  auto total_u32 = checked_u32(total, "copy rows f32->f16 total");
  if (!total_u32.is_ok()) {
    return total_u32.status();
  }

  auto source_status = validate_f32_buffer(source, source_end.value(),
                                           "copy rows f32->f16 source");
  if (!source_status.is_ok()) {
    return source_status;
  }
  auto destination_status =
    validate_f16_buffer(destination, destination_end.value(),
                        "copy rows f32->f16 destination");
  if (!destination_status.is_ok()) {
    return destination_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    CopyRowsParams params{
      source_offset_u32.value(),
      destination_offset_u32.value(),
      source_stride_u32.value(),
      destination_stride_u32.value(),
      row_size_u32.value(),
      total_u32.value(),
    };
    [encoder setComputePipelineState:impl_->copy_rows_to_f16_pipeline];
    [encoder setBuffer:source.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:destination.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads =
      [impl_->copy_rows_to_f16_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(total), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS copy rows f32->f16 command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::copy_f16_rows(const MpsBuffer& source, MpsBuffer& destination,
                                 std::size_t rows, std::size_t row_size,
                                 std::size_t source_stride,
                                 std::size_t source_offset,
                                 std::size_t destination_stride,
                                 std::size_t destination_offset) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  auto source_end = strided_f32_region_end(source_offset, rows, source_stride,
                                           row_size, "copy f16 rows source");
  if (!source_end.is_ok()) {
    return source_end.status();
  }
  auto destination_end =
    strided_f32_region_end(destination_offset, rows, destination_stride,
                           row_size, "copy f16 rows destination");
  if (!destination_end.is_ok()) {
    return destination_end.status();
  }
  std::size_t total = 0;
  if (!checked_mul(rows, row_size, total)) {
    return Status::invalid_argument("MPS copy f16 rows value count overflow");
  }

  auto source_offset_u32 = checked_u32(source_offset, "copy f16 rows source offset");
  if (!source_offset_u32.is_ok()) {
    return source_offset_u32.status();
  }
  auto destination_offset_u32 =
    checked_u32(destination_offset, "copy f16 rows destination offset");
  if (!destination_offset_u32.is_ok()) {
    return destination_offset_u32.status();
  }
  auto source_stride_u32 = checked_u32(source_stride, "copy f16 rows source stride");
  if (!source_stride_u32.is_ok()) {
    return source_stride_u32.status();
  }
  auto destination_stride_u32 =
    checked_u32(destination_stride, "copy f16 rows destination stride");
  if (!destination_stride_u32.is_ok()) {
    return destination_stride_u32.status();
  }
  auto row_size_u32 = checked_u32(row_size, "copy f16 rows row size");
  if (!row_size_u32.is_ok()) {
    return row_size_u32.status();
  }
  auto total_u32 = checked_u32(total, "copy f16 rows total");
  if (!total_u32.is_ok()) {
    return total_u32.status();
  }

  auto source_status = validate_f16_buffer(source, source_end.value(),
                                           "copy f16 rows source");
  if (!source_status.is_ok()) {
    return source_status;
  }
  auto destination_status =
    validate_f16_buffer(destination, destination_end.value(),
                        "copy f16 rows destination");
  if (!destination_status.is_ok()) {
    return destination_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    CopyRowsParams params{
      source_offset_u32.value(),
      destination_offset_u32.value(),
      source_stride_u32.value(),
      destination_stride_u32.value(),
      row_size_u32.value(),
      total_u32.value(),
    };
    [encoder setComputePipelineState:impl_->copy_f16_rows_pipeline];
    [encoder setBuffer:source.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:destination.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const NSUInteger max_threads =
      [impl_->copy_f16_rows_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group =
      std::min(max_threads, static_cast<NSUInteger>(256));
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(total), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS copy f16 rows command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::argmax_f32_i32(const MpsBuffer& values, std::size_t size,
                                  MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (size == 0) {
    return Status::invalid_argument("MPS argmax size must be positive");
  }
  auto size_u32 = checked_u32(size, "argmax size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto values_status = validate_f32_buffer(values, size, "argmax values");
  if (!values_status.is_ok()) {
    return values_status;
  }
  if (!output.valid()) {
    return Status::invalid_argument("MPS argmax output buffer is not initialized");
  }
  if (output.byte_size() < sizeof(std::int32_t)) {
    return Status::invalid_argument("MPS argmax output buffer is too small");
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->argmax_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    SizeParams params{size_u32.value(),
                      static_cast<std::uint32_t>(threads_per_group)};
    [encoder setComputePipelineState:impl_->argmax_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(1), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS argmax command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::argmax_prob_f32_i32(const MpsBuffer& values, std::size_t size,
                                       MpsBuffer& token_output,
                                       MpsBuffer& probability_output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (size == 0) {
    return Status::invalid_argument("MPS argmax probability size must be positive");
  }
  auto size_u32 = checked_u32(size, "argmax probability size");
  if (!size_u32.is_ok()) {
    return size_u32.status();
  }
  auto values_status = validate_f32_buffer(values, size, "argmax probability values");
  if (!values_status.is_ok()) {
    return values_status;
  }
  if (!token_output.valid()) {
    return Status::invalid_argument(
      "MPS argmax probability token output buffer is not initialized");
  }
  if (token_output.byte_size() < sizeof(std::int32_t)) {
    return Status::invalid_argument("MPS argmax probability token output buffer is too small");
  }
  auto probability_status =
    validate_f32_buffer(probability_output, 1, "argmax probability output");
  if (!probability_status.is_ok()) {
    return probability_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->argmax_prob_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    SizeParams params{size_u32.value(),
                      static_cast<std::uint32_t>(threads_per_group)};
    [encoder setComputePipelineState:impl_->argmax_prob_pipeline];
    [encoder setBuffer:values.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:token_output.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:probability_output.impl_->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const MTLSize group_count = MTLSizeMake(static_cast<NSUInteger>(1), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS argmax probability command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::attention_f32(const MpsBuffer& query, const MpsBuffer& key_cache,
                                 const MpsBuffer& value_cache, std::size_t layer,
                                 std::size_t position, std::size_t capacity_tokens,
                                 std::size_t heads, std::size_t kv_heads,
                                 std::size_t head_dim, MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (heads == 0 || kv_heads == 0 || head_dim == 0 || capacity_tokens == 0) {
    return Status::invalid_argument("MPS attention dimensions must be positive");
  }
  if (heads % kv_heads != 0) {
    return Status::invalid_argument("MPS attention heads must be divisible by kv_heads");
  }
  if (head_dim > 256U) {
    return Status::invalid_argument("MPS attention head_dim exceeds 256");
  }
  if (position >= capacity_tokens) {
    return Status::invalid_argument("MPS attention position exceeds KV cache capacity");
  }
  auto layer_u32 = checked_u32(layer, "attention layer");
  auto position_u32 = checked_u32(position, "attention position");
  auto capacity_u32 = checked_u32(capacity_tokens, "attention capacity");
  auto heads_u32 = checked_u32(heads, "attention heads");
  auto kv_heads_u32 = checked_u32(kv_heads, "attention kv_heads");
  auto head_dim_u32 = checked_u32(head_dim, "attention head_dim");
  auto group_u32 = checked_u32(heads / kv_heads, "attention group");
  if (!layer_u32.is_ok()) {
    return layer_u32.status();
  }
  if (!position_u32.is_ok()) {
    return position_u32.status();
  }
  if (!capacity_u32.is_ok()) {
    return capacity_u32.status();
  }
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  if (!kv_heads_u32.is_ok()) {
    return kv_heads_u32.status();
  }
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  if (!group_u32.is_ok()) {
    return group_u32.status();
  }
  const auto attn_dim = heads * head_dim;
  const auto kv_dim = kv_heads * head_dim;
  const auto cache_values = (layer + 1U) * capacity_tokens * kv_dim;
  auto query_status = validate_f32_buffer(query, attn_dim, "attention query");
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
  auto output_status = validate_f32_buffer(output, attn_dim, "attention output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->attention_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    if (threads_per_group < static_cast<NSUInteger>(256)) {
      return Status::unavailable("MPS attention pipeline requires 256 threads");
    }
    if (threads_per_group < static_cast<NSUInteger>(head_dim)) {
      return Status::unavailable("MPS attention pipeline cannot cover head_dim");
    }
    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    AttentionParams params{
      layer_u32.value(),
      position_u32.value(),
      capacity_u32.value(),
      heads_u32.value(),
      kv_heads_u32.value(),
      head_dim_u32.value(),
      group_u32.value(),
      scale,
    };
    [encoder setComputePipelineState:impl_->attention_pipeline];
    [encoder setBuffer:query.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:key_cache.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:value_cache.impl_->buffer offset:0 atIndex:2];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    const MTLSize group_count = MTLSizeMake(1, static_cast<NSUInteger>(heads), 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status = impl_->finish_command_buffer(command_buffer, "MPS attention command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::attention_f32_f16_kv(
  const MpsBuffer& query, const MpsBuffer& key_cache,
  const MpsBuffer& value_cache, std::size_t layer, std::size_t position,
  std::size_t capacity_tokens, std::size_t heads, std::size_t kv_heads,
  std::size_t head_dim, MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (heads == 0 || kv_heads == 0 || head_dim == 0 || capacity_tokens == 0) {
    return Status::invalid_argument("MPS F16 KV attention dimensions must be positive");
  }
  if (heads % kv_heads != 0) {
    return Status::invalid_argument("MPS F16 KV attention heads must be divisible by kv_heads");
  }
  if (head_dim > 256U) {
    return Status::invalid_argument("MPS F16 KV attention head_dim exceeds 256");
  }
  if (position >= capacity_tokens) {
    return Status::invalid_argument("MPS F16 KV attention position exceeds KV cache capacity");
  }
  auto layer_u32 = checked_u32(layer, "F16 KV attention layer");
  auto position_u32 = checked_u32(position, "F16 KV attention position");
  auto capacity_u32 = checked_u32(capacity_tokens, "F16 KV attention capacity");
  auto heads_u32 = checked_u32(heads, "F16 KV attention heads");
  auto kv_heads_u32 = checked_u32(kv_heads, "F16 KV attention kv_heads");
  auto head_dim_u32 = checked_u32(head_dim, "F16 KV attention head_dim");
  auto group_u32 = checked_u32(heads / kv_heads, "F16 KV attention group");
  if (!layer_u32.is_ok()) {
    return layer_u32.status();
  }
  if (!position_u32.is_ok()) {
    return position_u32.status();
  }
  if (!capacity_u32.is_ok()) {
    return capacity_u32.status();
  }
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  if (!kv_heads_u32.is_ok()) {
    return kv_heads_u32.status();
  }
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  if (!group_u32.is_ok()) {
    return group_u32.status();
  }
  const auto attn_dim = heads * head_dim;
  const auto kv_dim = kv_heads * head_dim;
  const auto cache_values = (layer + 1U) * capacity_tokens * kv_dim;
  auto query_status = validate_f32_buffer(query, attn_dim,
                                          "F16 KV attention query");
  if (!query_status.is_ok()) {
    return query_status;
  }
  auto key_status = validate_f16_buffer(key_cache, cache_values,
                                        "F16 KV attention key cache");
  if (!key_status.is_ok()) {
    return key_status;
  }
  auto value_status = validate_f16_buffer(value_cache, cache_values,
                                          "F16 KV attention value cache");
  if (!value_status.is_ok()) {
    return value_status;
  }
  auto output_status = validate_f32_buffer(output, attn_dim,
                                           "F16 KV attention output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    const NSUInteger max_threads =
      [impl_->attention_f16_kv_pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = choose_threadgroup_width(max_threads);
    if (threads_per_group < static_cast<NSUInteger>(256)) {
      return Status::unavailable("MPS F16 KV attention pipeline requires 256 threads");
    }
    if (threads_per_group < static_cast<NSUInteger>(head_dim)) {
      return Status::unavailable("MPS F16 KV attention pipeline cannot cover head_dim");
    }
    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    AttentionParams params{
      layer_u32.value(),
      position_u32.value(),
      capacity_u32.value(),
      heads_u32.value(),
      kv_heads_u32.value(),
      head_dim_u32.value(),
      group_u32.value(),
      scale,
    };
    [encoder setComputePipelineState:impl_->attention_f16_kv_pipeline];
    [encoder setBuffer:query.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:key_cache.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:value_cache.impl_->buffer offset:0 atIndex:2];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    const MTLSize group_count = MTLSizeMake(1, static_cast<NSUInteger>(heads), 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS F16 KV attention command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::attention_f32_batched(const MpsBuffer& query,
                                         const MpsBuffer& key_cache,
                                         const MpsBuffer& value_cache,
                                         std::size_t layer,
                                         std::size_t start_position,
                                         std::size_t tokens,
                                         std::size_t capacity_tokens,
                                         std::size_t heads,
                                         std::size_t kv_heads,
                                         std::size_t head_dim,
                                         MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (tokens == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 ||
      capacity_tokens == 0) {
    return Status::invalid_argument("MPS batched attention dimensions must be positive");
  }
  if (heads % kv_heads != 0) {
    return Status::invalid_argument("MPS batched attention heads must be divisible by kv_heads");
  }
  if (head_dim > 256U) {
    return Status::invalid_argument("MPS batched attention head_dim exceeds 256");
  }
  if (start_position > std::numeric_limits<std::size_t>::max() - tokens ||
      start_position + tokens > capacity_tokens) {
    return Status::invalid_argument("MPS batched attention chunk exceeds KV cache capacity");
  }

  std::size_t attn_dim = 0;
  std::size_t kv_dim = 0;
  std::size_t query_values = 0;
  std::size_t cache_layers = 0;
  std::size_t cache_values = 0;
  if (!checked_mul(heads, head_dim, attn_dim) ||
      !checked_mul(kv_heads, head_dim, kv_dim) ||
      !checked_mul(tokens, attn_dim, query_values) ||
      !checked_mul(layer + 1U, capacity_tokens, cache_layers) ||
      !checked_mul(cache_layers, kv_dim, cache_values)) {
    return Status::invalid_argument("MPS batched attention value count overflow");
  }

  auto query_status = validate_f32_buffer(query, query_values,
                                          "batched attention query");
  if (!query_status.is_ok()) {
    return query_status;
  }
  auto key_status = validate_f32_buffer(key_cache, cache_values,
                                        "batched attention key cache");
  if (!key_status.is_ok()) {
    return key_status;
  }
  auto value_status = validate_f32_buffer(value_cache, cache_values,
                                          "batched attention value cache");
  if (!value_status.is_ok()) {
    return value_status;
  }
  auto output_status = validate_f32_buffer(output, query_values,
                                           "batched attention output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  auto layer_u32 = checked_u32(layer, "batched attention layer");
  auto start_position_u32 =
    checked_u32(start_position, "batched attention start position");
  auto tokens_u32 = checked_u32(tokens, "batched attention tokens");
  auto capacity_u32 = checked_u32(capacity_tokens, "batched attention capacity");
  auto heads_u32 = checked_u32(heads, "batched attention heads");
  auto kv_heads_u32 = checked_u32(kv_heads, "batched attention kv_heads");
  auto head_dim_u32 = checked_u32(head_dim, "batched attention head_dim");
  auto group_u32 = checked_u32(heads / kv_heads, "batched attention group");
  if (!layer_u32.is_ok()) {
    return layer_u32.status();
  }
  if (!start_position_u32.is_ok()) {
    return start_position_u32.status();
  }
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  if (!capacity_u32.is_ok()) {
    return capacity_u32.status();
  }
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  if (!kv_heads_u32.is_ok()) {
    return kv_heads_u32.status();
  }
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  if (!group_u32.is_ok()) {
    return group_u32.status();
  }

  const std::size_t key_count = start_position + tokens;
  const auto flash_env_flag = flash_attention_env_flag();
  const bool flash_compatible =
    impl_->attention_batched_flash256_pipeline != nil && head_dim == 256U;
  const bool use_flash_attention =
    flash_compatible && flash_env_flag != EnvFlag::disabled &&
    (flash_env_flag == EnvFlag::enabled || tokens >= 64U);
  const bool use_tiled_attention =
    !use_flash_attention && should_use_tiled_attention(tokens, head_dim);

  const bool use_flash_tail =
    use_flash_attention && key_count % 64U != 0U;
  if (use_flash_tail) {
    std::size_t flash_tail_values = 0;
    std::size_t flash_tail_bytes = 0;
    if (!checked_mul(static_cast<std::size_t>(64U), kv_dim,
                     flash_tail_values) ||
        !checked_mul(flash_tail_values, sizeof(float), flash_tail_bytes)) {
      return Status::invalid_argument("MPS flash attention tail size overflow");
    }

    auto status = ensure_mps_buffer_size(
      *this, impl_->flash_f32_key_tail_scratch, flash_tail_bytes,
      "MPS flash attention key tail scratch");
    if (!status.is_ok()) {
      return status;
    }
    status = ensure_mps_buffer_size(
      *this, impl_->flash_f32_value_tail_scratch, flash_tail_bytes,
      "MPS flash attention value tail scratch");
    if (!status.is_ok()) {
      return status;
    }
    status = zero_buffer(impl_->flash_f32_key_tail_scratch, flash_tail_bytes);
    if (!status.is_ok()) {
      return status;
    }
    status = zero_buffer(impl_->flash_f32_value_tail_scratch, flash_tail_bytes);
    if (!status.is_ok()) {
      return status;
    }

    const std::size_t tail_tokens = key_count % 64U;
    const std::size_t tail_start = key_count - tail_tokens;
    std::size_t tail_layer_offset = 0;
    std::size_t tail_source_offset = 0;
    if (!checked_mul(layer, capacity_tokens, tail_layer_offset) ||
        tail_layer_offset >
          std::numeric_limits<std::size_t>::max() - tail_start ||
        !checked_mul(tail_layer_offset + tail_start, kv_dim,
                     tail_source_offset)) {
      return Status::invalid_argument("MPS flash attention tail offset overflow");
    }
    status = copy_f32_rows(key_cache, impl_->flash_f32_key_tail_scratch,
                           tail_tokens, kv_dim, kv_dim, tail_source_offset,
                           kv_dim, 0);
    if (!status.is_ok()) {
      return status;
    }
    status = copy_f32_rows(value_cache, impl_->flash_f32_value_tail_scratch,
                           tail_tokens, kv_dim, kv_dim, tail_source_offset,
                           kv_dim, 0);
    if (!status.is_ok()) {
      return status;
    }
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    id<MTLComputePipelineState> pipeline = impl_->attention_batched_pipeline;
    if (use_flash_attention) {
      pipeline = impl_->attention_batched_flash256_pipeline;
    } else if (use_tiled_attention) {
      pipeline = requested_tiled_attention_cache_tile() == 32U &&
                 impl_->attention_batched_tiled32_pipeline != nil
        ? impl_->attention_batched_tiled32_pipeline
        : impl_->attention_batched_tiled_pipeline;
    }
    const NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = use_flash_attention
      ? static_cast<NSUInteger>(128)
      : choose_threadgroup_width(max_threads);
    if (threads_per_group < static_cast<NSUInteger>(256)) {
      if (!use_flash_attention) {
        return Status::unavailable("MPS batched attention pipeline requires 256 threads");
      }
    }
    if (threads_per_group < static_cast<NSUInteger>(head_dim)) {
      if (!use_flash_attention) {
        return Status::unavailable("MPS batched attention pipeline cannot cover head_dim");
      }
    }
    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:query.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:key_cache.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:value_cache.impl_->buffer offset:0 atIndex:2];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:3];
    if (use_flash_attention) {
      FlashAttention256Params params{
        layer_u32.value(),
        start_position_u32.value(),
        tokens_u32.value(),
        capacity_u32.value(),
        heads_u32.value(),
        kv_heads_u32.value(),
        group_u32.value(),
        scale,
      };
      [encoder setBytes:&params length:sizeof(params) atIndex:4];
      const MpsBuffer& key_tail =
        use_flash_tail ? impl_->flash_f32_key_tail_scratch : key_cache;
      const MpsBuffer& value_tail =
        use_flash_tail ? impl_->flash_f32_value_tail_scratch : value_cache;
      [encoder setBuffer:key_tail.impl_->buffer offset:0 atIndex:5];
      [encoder setBuffer:value_tail.impl_->buffer offset:0 atIndex:6];
      [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(16384)
                                  atIndex:0];
    } else {
      BatchedAttentionParams params{
        layer_u32.value(),
        start_position_u32.value(),
        tokens_u32.value(),
        capacity_u32.value(),
        heads_u32.value(),
        kv_heads_u32.value(),
        head_dim_u32.value(),
        group_u32.value(),
        static_cast<std::uint32_t>(threads_per_group),
        scale,
      };
      [encoder setBytes:&params length:sizeof(params) atIndex:4];
    }
    const NSUInteger query_tile = static_cast<NSUInteger>(8);
    const NSUInteger token_groups =
      (static_cast<NSUInteger>(tokens) + query_tile - static_cast<NSUInteger>(1)) /
      query_tile;
    const MTLSize group_count =
      MTLSizeMake(token_groups, static_cast<NSUInteger>(heads), 1);
    const MTLSize threadgroup_size = use_flash_attention
      ? MTLSizeMake(static_cast<NSUInteger>(32), static_cast<NSUInteger>(4), 1)
      : MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS batched attention command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

Status MpsContext::attention_f32_batched_f16_kv(
  const MpsBuffer& query, const MpsBuffer& key_cache,
  const MpsBuffer& value_cache, std::size_t layer,
  std::size_t start_position, std::size_t tokens,
  std::size_t capacity_tokens, std::size_t heads, std::size_t kv_heads,
  std::size_t head_dim, MpsBuffer& output) const {
  if (!valid()) {
    return Status::unavailable("MPS context is not initialized");
  }
  if (tokens == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 ||
      capacity_tokens == 0) {
    return Status::invalid_argument("MPS F16 KV batched attention dimensions must be positive");
  }
  if (heads % kv_heads != 0) {
    return Status::invalid_argument(
      "MPS F16 KV batched attention heads must be divisible by kv_heads");
  }
  if (head_dim > 256U) {
    return Status::invalid_argument("MPS F16 KV batched attention head_dim exceeds 256");
  }
  if (start_position > std::numeric_limits<std::size_t>::max() - tokens ||
      start_position + tokens > capacity_tokens) {
    return Status::invalid_argument(
      "MPS F16 KV batched attention chunk exceeds KV cache capacity");
  }

  std::size_t attn_dim = 0;
  std::size_t kv_dim = 0;
  std::size_t query_values = 0;
  std::size_t cache_layers = 0;
  std::size_t cache_values = 0;
  if (!checked_mul(heads, head_dim, attn_dim) ||
      !checked_mul(kv_heads, head_dim, kv_dim) ||
      !checked_mul(tokens, attn_dim, query_values) ||
      !checked_mul(layer + 1U, capacity_tokens, cache_layers) ||
      !checked_mul(cache_layers, kv_dim, cache_values)) {
    return Status::invalid_argument("MPS F16 KV batched attention value count overflow");
  }

  auto query_status = validate_f32_buffer(query, query_values,
                                          "F16 KV batched attention query");
  if (!query_status.is_ok()) {
    return query_status;
  }
  auto key_status = validate_f16_buffer(key_cache, cache_values,
                                        "F16 KV batched attention key cache");
  if (!key_status.is_ok()) {
    return key_status;
  }
  auto value_status = validate_f16_buffer(value_cache, cache_values,
                                          "F16 KV batched attention value cache");
  if (!value_status.is_ok()) {
    return value_status;
  }
  auto output_status = validate_f32_buffer(output, query_values,
                                           "F16 KV batched attention output");
  if (!output_status.is_ok()) {
    return output_status;
  }

  auto layer_u32 = checked_u32(layer, "F16 KV batched attention layer");
  auto start_position_u32 =
    checked_u32(start_position, "F16 KV batched attention start position");
  auto tokens_u32 = checked_u32(tokens, "F16 KV batched attention tokens");
  auto capacity_u32 =
    checked_u32(capacity_tokens, "F16 KV batched attention capacity");
  auto heads_u32 = checked_u32(heads, "F16 KV batched attention heads");
  auto kv_heads_u32 = checked_u32(kv_heads, "F16 KV batched attention kv_heads");
  auto head_dim_u32 = checked_u32(head_dim, "F16 KV batched attention head_dim");
  auto group_u32 =
    checked_u32(heads / kv_heads, "F16 KV batched attention group");
  if (!layer_u32.is_ok()) {
    return layer_u32.status();
  }
  if (!start_position_u32.is_ok()) {
    return start_position_u32.status();
  }
  if (!tokens_u32.is_ok()) {
    return tokens_u32.status();
  }
  if (!capacity_u32.is_ok()) {
    return capacity_u32.status();
  }
  if (!heads_u32.is_ok()) {
    return heads_u32.status();
  }
  if (!kv_heads_u32.is_ok()) {
    return kv_heads_u32.status();
  }
  if (!head_dim_u32.is_ok()) {
    return head_dim_u32.status();
  }
  if (!group_u32.is_ok()) {
    return group_u32.status();
  }

  const std::size_t key_count = start_position + tokens;
  const auto flash_env_flag = flash_attention_env_flag();
  const bool flash_compatible =
    impl_->attention_batched_flash256_f16_kv_pipeline != nil &&
    head_dim == 256U;
  const bool use_flash_attention =
    flash_compatible && flash_env_flag != EnvFlag::disabled &&
    (flash_env_flag == EnvFlag::enabled || tokens >= 64U);

  const bool use_flash_tail =
    use_flash_attention && key_count % 64U != 0U;
  if (use_flash_tail) {
    std::size_t flash_tail_values = 0;
    std::size_t flash_tail_bytes = 0;
    if (!checked_mul(static_cast<std::size_t>(64U), kv_dim,
                     flash_tail_values) ||
        !checked_mul(flash_tail_values, sizeof(std::uint16_t),
                     flash_tail_bytes)) {
      return Status::invalid_argument("MPS F16 KV flash attention tail size overflow");
    }

    auto status = ensure_mps_buffer_size(
      *this, impl_->flash_f16_key_tail_scratch, flash_tail_bytes,
      "MPS F16 KV flash attention key tail scratch");
    if (!status.is_ok()) {
      return status;
    }
    status = ensure_mps_buffer_size(
      *this, impl_->flash_f16_value_tail_scratch, flash_tail_bytes,
      "MPS F16 KV flash attention value tail scratch");
    if (!status.is_ok()) {
      return status;
    }

    const std::size_t tail_tokens = key_count % 64U;
    const std::size_t tail_start = key_count - tail_tokens;
    std::size_t tail_layer_offset = 0;
    std::size_t tail_source_offset = 0;
    if (!checked_mul(layer, capacity_tokens, tail_layer_offset) ||
        tail_layer_offset >
          std::numeric_limits<std::size_t>::max() - tail_start ||
        !checked_mul(tail_layer_offset + tail_start, kv_dim,
                     tail_source_offset)) {
      return Status::invalid_argument("MPS F16 KV flash attention tail offset overflow");
    }
    status = zero_buffer(impl_->flash_f16_key_tail_scratch, flash_tail_bytes);
    if (!status.is_ok()) {
      return status;
    }
    status = zero_buffer(impl_->flash_f16_value_tail_scratch, flash_tail_bytes);
    if (!status.is_ok()) {
      return status;
    }
    status = copy_f16_rows(key_cache, impl_->flash_f16_key_tail_scratch,
                           tail_tokens, kv_dim, kv_dim, tail_source_offset,
                           kv_dim, 0);
    if (!status.is_ok()) {
      return status;
    }
    status = copy_f16_rows(value_cache, impl_->flash_f16_value_tail_scratch,
                           tail_tokens, kv_dim, kv_dim, tail_source_offset,
                           kv_dim, 0);
    if (!status.is_ok()) {
      return status;
    }
  }

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    if (command_buffer == nil) {
      return Status::unavailable("failed to create Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return Status::unavailable("failed to create Metal compute encoder");
    }

    id<MTLComputePipelineState> pipeline =
      use_flash_attention ? impl_->attention_batched_flash256_f16_kv_pipeline
                          : impl_->attention_batched_f16_kv_pipeline;
    const NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger threads_per_group = use_flash_attention
      ? static_cast<NSUInteger>(128)
      : choose_threadgroup_width(max_threads);
    if (!use_flash_attention &&
        threads_per_group < static_cast<NSUInteger>(256)) {
      return Status::unavailable(
        "MPS F16 KV batched attention pipeline requires 256 threads");
    }
    if (!use_flash_attention &&
        threads_per_group < static_cast<NSUInteger>(head_dim)) {
      return Status::unavailable(
        "MPS F16 KV batched attention pipeline cannot cover head_dim");
    }
    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:query.impl_->buffer offset:0 atIndex:0];
    [encoder setBuffer:key_cache.impl_->buffer offset:0 atIndex:1];
    [encoder setBuffer:value_cache.impl_->buffer offset:0 atIndex:2];
    [encoder setBuffer:output.impl_->buffer offset:0 atIndex:3];
    if (use_flash_attention) {
      FlashAttention256Params params{
        layer_u32.value(),
        start_position_u32.value(),
        tokens_u32.value(),
        capacity_u32.value(),
        heads_u32.value(),
        kv_heads_u32.value(),
        group_u32.value(),
        scale,
      };
      [encoder setBytes:&params length:sizeof(params) atIndex:4];
      const MpsBuffer& key_tail =
        use_flash_tail ? impl_->flash_f16_key_tail_scratch : key_cache;
      const MpsBuffer& value_tail =
        use_flash_tail ? impl_->flash_f16_value_tail_scratch : value_cache;
      [encoder setBuffer:key_tail.impl_->buffer offset:0 atIndex:5];
      [encoder setBuffer:value_tail.impl_->buffer offset:0 atIndex:6];
      [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(16384)
                                  atIndex:0];
    } else {
      BatchedAttentionParams params{
        layer_u32.value(),
        start_position_u32.value(),
        tokens_u32.value(),
        capacity_u32.value(),
        heads_u32.value(),
        kv_heads_u32.value(),
        head_dim_u32.value(),
        group_u32.value(),
        static_cast<std::uint32_t>(threads_per_group),
        scale,
      };
      [encoder setBytes:&params length:sizeof(params) atIndex:4];
    }
    const NSUInteger query_tile = static_cast<NSUInteger>(8);
    const NSUInteger token_groups =
      (static_cast<NSUInteger>(tokens) + query_tile - static_cast<NSUInteger>(1)) /
      query_tile;
    const MTLSize group_count =
      MTLSizeMake(token_groups, static_cast<NSUInteger>(heads), 1);
    const MTLSize threadgroup_size = use_flash_attention
      ? MTLSizeMake(static_cast<NSUInteger>(32), static_cast<NSUInteger>(4), 1)
      : MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    auto finish_status =
      impl_->finish_command_buffer(command_buffer,
                                   "MPS F16 KV batched attention command failed: ");
    if (!finish_status.is_ok()) {
      return finish_status;
    }
  }
  return Status::ok();
}

BackendInfo query_backend() {
  BackendInfo info{};
  info.compiled = true;

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      info.failure_reason = "Metal returned no default device";
      return info;
    }

    info.available = true;
    info.compute_ready = true;
    info.forward_ready = true;

    NSString* name = [device name];
    if (name != nil) {
      info.device_name = [name UTF8String];
    }

    info.recommended_max_working_set_size =
      static_cast<std::uint64_t>([device recommendedMaxWorkingSetSize]);
    info.low_power = [device isLowPower];
    info.headless = [device isHeadless];
    info.removable = [device isRemovable];
  }

  return info;
}

std::string format_backend_info(const BackendInfo& info) {
  std::ostringstream output;
  output << "MPS compiled: " << yes_no(info.compiled) << '\n';
  output << "MPS available: " << yes_no(info.available) << '\n';
  output << "MPS compute ready: " << yes_no(info.compute_ready) << '\n';
  output << "MPS full forward ready: " << yes_no(info.forward_ready) << '\n';
  if (!info.device_name.empty()) {
    output << "Metal device: " << info.device_name << '\n';
  }
  if (info.recommended_max_working_set_size > 0) {
    output << "Recommended max working set: " << info.recommended_max_working_set_size
           << " bytes\n";
  }
  output << "Low power: " << yes_no(info.low_power) << '\n';
  output << "Headless: " << yes_no(info.headless) << '\n';
  output << "Removable: " << yes_no(info.removable) << '\n';
  if (!info.failure_reason.empty()) {
    output << "Reason: " << info.failure_reason << '\n';
  }
  return output.str();
}

Status run_operator_smoke_test() {
  auto context_result = MpsContext::create();
  if (!context_result.is_ok()) {
    return context_result.status();
  }
  auto context = std::move(context_result.value());

  const std::uint16_t weight[] = {
    float_to_bf16(1.0F),
    float_to_bf16(2.0F),
    float_to_bf16(3.0F),
    float_to_bf16(1.0F),
  };
  auto weight_buffer_result = context.make_buffer(sizeof(weight));
  if (!weight_buffer_result.is_ok()) {
    return weight_buffer_result.status();
  }
  auto weight_buffer = std::move(weight_buffer_result.value());
  const auto copy_status = context.copy_to_buffer(weight_buffer, weight, sizeof(weight));
  if (!copy_status.is_ok()) {
    return copy_status;
  }

  const std::vector<float> input{3.0F, 4.0F};
  auto output_result = context.matvec_bf16_f32(weight_buffer, 2, 2, input);
  if (!output_result.is_ok()) {
    return output_result.status();
  }
  const auto& output = output_result.value();
  if (output.size() != 2 || std::abs(output[0] - 11.0F) > 1e-5F ||
      std::abs(output[1] - 13.0F) > 1e-5F) {
    std::ostringstream message;
    message << "MPS operator smoke mismatch: got [";
    if (!output.empty()) {
      message << output[0];
    }
    if (output.size() > 1) {
      message << ", " << output[1];
    }
    message << ']';
    return Status::internal_error(message.str());
  }
  return Status::ok();
}

}  // namespace toyllm::mps
