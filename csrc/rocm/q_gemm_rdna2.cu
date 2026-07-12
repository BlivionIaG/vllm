// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 GPTQ kernel for RDNA2 (gfx1030 class), fp16. Adapted from
// W4A16 GPTQ kernel for RDNA3 (csrc/rocm/q_gemm_rdna3.cu):
//
//   1. M_COUNT ∈ {1, 2, 4, 8} covers M ∈ [1, 15] in tiles; larger M uses
//      M_COUNT=8 and is correct but leaves throughput on the table.
//      gfx1030 has no WMMA ISA (that landed on RDNA3, gfx1100+).
//
//   2. Direct write to f16 output via packed CAS-loop on a 64-bit
//      word (atomic_add_pk4_f16). gfx1030 has no native
//      v_global_atomic_pk_add_f16, so the kernel emulates one with
//      global_atomic_cmpswap_b64. Caller passes a zero-initialised f16
//      output tensor and every block atomically adds its partial sum.
//
//   3. Wave32 geometry: THREADS_X=256 (8 waves per block),
//      BLOCK_KN_SIZE=256, each thread computes 4 N output columns.

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "q_gemm_rdna2_common.cuh"
#include "qdq_4_rdna2.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace gptq_rdna2 {

// BLOCK_KN_SIZE = 256 (was 128 in exllama). Each block covers 256 K
// elements and THREADS_X*4 = 1024 N columns. For Qwen-class K=4096 this
// halves gridDim.z (32 → 16) and therefore halves the atomic count per
// output position vs the exllama default. THREADS_X=256 = 8 waves on RDNA2
// wave32; with ~32 wave slots per CU we still fit 4 blocks per CU at peak.
#define BLOCK_KN_SIZE 256
#define THREADS_X 256

// Device code below is RDNA2-only; non-RDNA2 device passes fall through to
// the empty __global__ stub at the #else below for symbol parity.
#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// Shared per-dtype helpers are in q_gemm_rdna2_common.cuh (tzero, dot22_8_f,
// atomic_add_pk4_f16, load4_zeros, load4_scales, refresh_group, epilogue).

// ---------------------------------------------------------------------------
// Main kernel.
// ---------------------------------------------------------------------------

template <typename T, int M_COUNT>
__global__ void gemm_q4_kernel_rdna2(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int offset_m = blockIdx.y * M_COUNT;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // LDS layout: [M_COUNT][BLOCK_KN_SIZE + LDS_PAD]. PAD=8 avoids bank
  // conflicts when different M rows read the same K offset. Cost: 16B LDS
  // per block, negligible.
  constexpr int LDS_PAD = 8;
  __shared__ T block_a[M_COUNT][BLOCK_KN_SIZE + LDS_PAD];

  // Stage A into LDS: each thread loads one K element per M row. Invalid
  // rows past size_m are zero-padded. LDS staging is required even at M=1
  // because the inner loop indexes block_a[m][a_off] unconditionally.
  static_assert(BLOCK_KN_SIZE == THREADS_X,
                "BLOCK_KN_SIZE must equal THREADS_X (1 K element per thread)");
  if (offset_k + t < end_k) {
#pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
      T av;
      if (offset_m + m < size_m) {
        const T* a_row = a + (offset_m + m) * size_k;
        if (b_q_perm)
          av = a_row[b_q_perm[offset_k + t]];
        else
          av = a_row[offset_k + t];
      } else {
        av = tzero<T>();  // zero-pad invalid M rows
      }
      block_a[m][t] = av;
    }
  }

  // Threads beyond the right edge of N have nothing to do. Note: we must NOT
  // return before __syncthreads() if any thread in the block participates in
  // the LDS load above — but here all THREADS_X (=256) threads always do,
  // regardless of whether their `n` is in bounds.
  __syncthreads();

  if (n >= size_n) return;

  // Group bookkeeping. We require size_k % groups == 0 (groupsize divides K).
  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // qweight stride: weights are [K/8, N] uint32 with K packed at dim 0.
  int qk = offset_k / 8;
  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Per-column dequant constants. We hold one set of (z, y) pairs per column.
  // fp16 uses the exllama (z1z16, y1y16) double-pair to enable the upper-
  // nibble-*16 trick.
  half2 z1z16_h[4][2], y1y16_h[4][2];
  refresh_group<4>(group, n, b_qzeros, b_scales, size_n, zero_offset,
                   z1z16_h, y1y16_h);

  float block_c[M_COUNT][4];
  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
  #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  // Group transitions are checked at K-block granularity; we require
  // groupsize >= 32 (mirrors exllama). Prefetch 4 weight words ahead of
  // dequant/FMA to hide global load latency.
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group<4>(group, n, b_qzeros, b_scales, size_n, zero_offset,
                       z1z16_h, y1y16_h);
    }

    // Prefetch all four j-iterations' weight words. The compiler emits 4
    // global_load_b128 instructions back-to-back; the dependent dequant +
    // FMA work below hides their latency.
    int4 b_w[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      b_w[j] = *(const int4*)(b_ptr + j * size_n);
    }
    b_ptr += 4 * size_n;

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;

      half2 dq[4][4];
      dequant_4bit_8_fp16((uint32_t)b_w[j].x, dq[0], z1z16_h[0], y1y16_h[0]);
      dequant_4bit_8_fp16((uint32_t)b_w[j].y, dq[1], z1z16_h[1], y1y16_h[1]);
      dequant_4bit_8_fp16((uint32_t)b_w[j].z, dq[2], z1z16_h[2], y1y16_h[2]);
      dequant_4bit_8_fp16((uint32_t)b_w[j].w, dq[3], z1z16_h[3], y1y16_h[3]);

  #pragma unroll
      for (int m = 0; m < M_COUNT; ++m) {
        const half* a_ptr = reinterpret_cast<const half*>(&block_a[m][a_off]);
        block_c[m][0] += dot22_8_f(dq[0], a_ptr);
        block_c[m][1] += dot22_8_f(dq[1], a_ptr);
        block_c[m][2] += dot22_8_f(dq[2], a_ptr);
        block_c[m][3] += dot22_8_f(dq[3], a_ptr);
      }
    }
    k += 32;  // 4 weight words * 8 nibbles = 32 K elements
  }

  // Pack partial sums into two half2 pairs and atomically add to the
  // zero-initialized fp16 output.
  epilogue<M_COUNT>(block_c, offset_m, size_m, size_n, n, c);
}

#else  // non-RDNA2 device pass: empty __global__ for symbol parity.

template <typename T, int M_COUNT>
__global__ void gemm_q4_kernel_rdna2(const T*, const uint32_t*, const uint32_t*,
    const T*, T*, const int, const int,
    const int, const int, const int,
    const int*) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher.
// ---------------------------------------------------------------------------

template <typename T, int M_COUNT>
void launch_gemm_q4_for_mcount(const T* a, const uint32_t* b_q_weight,
                               const uint32_t* b_qzeros, const T* b_scales,
                               const int* b_q_perm, T* c, int size_m,
                               int size_n, int size_k, int groups,
                               int zero_offset, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  gemm_q4_kernel_rdna2<T, M_COUNT><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
}
// Dispatch to the largest M_COUNT template that tiles size_m without wasting
// more than half the last tile. M_COUNT is capped at 8; the kernel still
// produces correct output for M >= 16 but is decode-optimized, not a prefill
// GEMM.
template <typename T>
void launch_gemm_q4(const T* a, const uint32_t* b_q_weight,
                    const uint32_t* b_qzeros, const T* b_scales,
                    const int* b_q_perm, T* c, int size_m, int size_n,
                    int size_k, int groups, bool use_v2_format,
                    cudaStream_t stream) {
  const int zero_offset = use_v2_format ? 0 : 1;

  if (size_m == 1) {
    launch_gemm_q4_for_mcount<T, 1>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  } else if (size_m <= 3) {
    launch_gemm_q4_for_mcount<T, 2>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  } else if (size_m <= 7) {
    launch_gemm_q4_for_mcount<T, 4>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  } else {
    // M_COUNT=8 covers M in [8, 15] and all larger M. The kernel is correct
    // but decode-optimized; throughput at large M is intentionally left to
    // a separate prefill GEMM when available.
    launch_gemm_q4_for_mcount<T, 8>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                    c, size_m, size_n, size_k, groups,
                                    zero_offset, stream);
  }
}

}  // namespace gptq_rdna2
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
//
// Inputs:
//   a         [M, K]            half
//   b_q_weight[K/8, N]          uint32 (already shuffled via gptq_shuffle)
//   b_qzeros  [groups, N/8]     uint32 (packed 4-bit zeros)
//   b_scales  [groups, N]       half
//   b_g_idx   [K] or empty      int32 (act-order permutation; empty=identity)
//   use_v2_format                bool   (true = GPTQv2, no +1 zero offset)
//
// Output:
//   c         [M, N]            half

torch::Tensor gptq_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_qzeros.is_cuda(), "b_qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "gptq_gemm_rdna2 only supports fp16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(1);
  int groups = (int)b_qzeros.size(0);

  TORCH_CHECK(b_q_weight.size(0) * 8 == size_k,
              "b_q_weight first dim must be K/8");
  TORCH_CHECK(b_scales.size(0) == groups,
              "b_scales must have same group count as qzeros");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales last dim must be N");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8 (64-bit atomic CAS)");

  auto opts = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  at::Tensor c = torch::zeros({size_m, size_n}, opts);

  const int* g_idx_ptr = nullptr;
  if (!b_g_idx.device().is_meta() && b_g_idx.numel() > 0) {
    TORCH_CHECK(b_g_idx.scalar_type() == torch::kInt32,
                "b_g_idx must be int32");
    g_idx_ptr = (const int*)b_g_idx.data_ptr();
  }

  vllm::gptq_rdna2::launch_gemm_q4<half>(
      (const half*)a.data_ptr(), (const uint32_t*)b_q_weight.data_ptr(),
      (const uint32_t*)b_qzeros.data_ptr(), (const half*)b_scales.data_ptr(),
      g_idx_ptr, (half*)c.data_ptr(), size_m, size_n, size_k, groups,
      use_v2_format, stream);

  return c;
}
