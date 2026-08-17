// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 dequant primitives for RDNA2 (gfx1030):
//   - E2M1 nibble -> FP16 (inline bit-trick, no LUT, no constant memory)
//   - UE8M0 -> FP16 conversion (power-of-two scale)
//   - V_DOT2_F32_F16 wrapper (4 calls covering 8 elements)
//   - 64-bit CAS atomic-add for output reduction
//
// Used by:
//   * csrc/rocm/mxfp4_dot2_dense.cu
//   * csrc/rocm/mxfp4_dot2_moe.cu
//
// gfx1030 (RDNA2) does NOT support dynamic initialization for
// __device__ __constant__ variables ("dynamic initialization is not
// supported for __device__, __constant__, __shared__, and __managed__
// variables"). The previous design used a __constant__ LUT which
// failed to compile under hipcc for gfx1030. This rewrite uses inline
// bit-trick conversion (no LUT, no host init, no cross-TU issue),
// matching the pattern already used by W8A16-FP8/W8A8-FP8 in
// qdq_fp8_rdna2.cuh:fp8_e4m3_to_fp16_bits.
//
// Activation dtype is fp16; scale dtype is uint8 (UE8M0 power-of-two).
// E2M1 has no zero point -- the high nibble encodes sign.

#ifndef _MXFP4_DOT2_RDNA2_CUH
#define _MXFP4_DOT2_RDNA2_CUH

#include <cstdint>

#include <hip/hip_fp16.h>

namespace vllm {
namespace mxfp4_dot2 {

// ---------------------------------------------------------------------------
// E2M1 nibble -> FP16 bits (inline bit-trick).
// ---------------------------------------------------------------------------
// E2M1 element layout (4 bits, LSB first):
//   s e e m
//   0 0 0 0 = +0      -> 0x0000
//   0 0 0 1 = +0.5    -> 0x3800   (subnormal)
//   0 0 1 0 = +1.0    -> 0x3C00
//   0 0 1 1 = +1.5    -> 0x3E00
//   0 1 0 0 = +2.0    -> 0x4000
//   0 1 0 1 = +3.0    -> 0x4200
//   0 1 1 0 = +4.0    -> 0x4400
//   0 1 1 1 = +6.0    -> 0x4600
//   1 x x x = negate the above (high bit set).
//
// E2M1 exp bias = 1 (2-bit exponent). FP16 exp bias = 15.
// Normal: fp16_exp = exp2 + 14, fp16_mant = mant1 << 9.
__forceinline__ __device__ uint16_t e2m1_to_fp16_bits(uint8_t nibble) {
  uint8_t sign = (nibble >> 3) & 0x1u;
  uint8_t exp2 = (nibble >> 1) & 0x3u;
  uint8_t mant1 = nibble & 0x1u;
  uint16_t sign_bit = (uint16_t)(sign << 15);
  uint16_t bits;
  if (exp2 == 0) {
    bits = (mant1 == 0) ? sign_bit : (uint16_t)(sign_bit | 0x3800u);
  } else {
    uint16_t exp_fp16 = (uint16_t)(exp2 + 14);
    uint16_t mant_fp16 = (uint16_t)(mant1 << 9);
    bits = (uint16_t)(sign_bit | (exp_fp16 << 10) | mant_fp16);
  }
  return bits;
}

// ---------------------------------------------------------------------------
// UE8M0 -> FP16 conversion
// ---------------------------------------------------------------------------
// UE8M0 is 8-bit unsigned exponent (power-of-two scale): scale = 2^(k-127).
// FP16 representation of 2^(scale-127):
//   sign=0, exp=(scale-127+15)=(scale-112), mantissa=0
// So the FP16 bits are: (scale - 112) << 10
//
// Cycle cost: 1 cycle (constant-time shift, no memory access).
// Range: FP16 normal range is [2^-14, 2^15] = [6.1e-5, 32768].
// Subnormal UE8M0 (scale < 113) -> FP16 subnormal (rare, may underflow).
// Overflow UE8M0 (scale > 142) -> FP16 infinity (very rare in practice).
__forceinline__ __device__ half ue8m0_to_fp16(uint8_t scale_byte) {
  unsigned short bits = (unsigned short)((scale_byte - 112) << 10);
  return __ushort_as_half(bits);
}

// ---------------------------------------------------------------------------
// dot22_8_f -- 4 x V_DOT2_F32_F16 calls covering 8 consecutive K positions
// ---------------------------------------------------------------------------
__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  float result = 0.0f;
  const half2* a2_ptr = reinterpret_cast<const half2*>(a_ptr);
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result = __builtin_amdgcn_fdot2(dq[i], *a2_ptr++, result, /*clamp=*/false);
  }
  return result;
}

// ---------------------------------------------------------------------------
// 64-bit CAS atomic-add for 4 fp16 output columns
// ---------------------------------------------------------------------------
// Packed atomic-add via CAS-loop on a 64-bit word (4 fp16 lanes per CAS).
// RDNA2 (gfx1030) does NOT have native v_global_atomic_pk_add_f16 (that
// landed on gfx940), so this lowers to global_atomic_cmpswap_b64 plus retry.
__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// ---------------------------------------------------------------------------
// Dequantize 8 E2M1 nibbles (one uint32) into 4 half2 pairs (8 fp16 values)
// ---------------------------------------------------------------------------
// qa packs 8 nibbles LSB-first (matching OCP spec). sh carries the
// same-row scale (broadcast half2). dq[4] is 4 half2 pairs = 8 fp16
// values suitable for dot22_8_f.
//
// Per-iter: 8 inline bit-trick conversions + 4 hfma2 (~8 + 4 cycles).
__forceinline__ __device__ void dequant_e2m1_8_fp16(
    uint32_t qa, half2 scale2, half2 (&dq)[4]) {
  dq[0] = __halves2half2(
      __ushort_as_half(e2m1_to_fp16_bits((uint8_t)(qa & 0xFu))),
      __ushort_as_half(e2m1_to_fp16_bits((uint8_t)((qa >> 4) & 0xFu)))) *
         scale2;
  dq[1] = __halves2half2(
      __ushort_as_half(e2m1_to_fp16_bits((uint8_t)((qa >> 8) & 0xFu))),
      __ushort_as_half(e2m1_to_fp16_bits((uint8_t)((qa >> 12) & 0xFu)))) *
         scale2;
  dq[2] = __halves2half2(
      __ushort_as_half(e2m1_to_fp16_bits((uint8_t)((qa >> 16) & 0xFu))),
      __ushort_as_half(e2m1_to_fp16_bits((uint8_t)((qa >> 20) & 0xFu)))) *
         scale2;
  dq[3] = __halves2half2(
      __ushort_as_half(e2m1_to_fp16_bits((uint8_t)((qa >> 24) & 0xFu))),
      __ushort_as_half(e2m1_to_fp16_bits((uint8_t)((qa >> 28) & 0xFu)))) *
         scale2;
}

// ---------------------------------------------------------------------------
// Zero helper for FP16
// ---------------------------------------------------------------------------
template <typename T>
__forceinline__ __device__ T tzero();

template <>
__forceinline__ __device__ half tzero<half>() {
  return __float2half_rn(0.0f);
}

}  // namespace mxfp4_dot2
}  // namespace vllm

#endif  // _MXFP4_DOT2_RDNA2_CUH
