"""FA RDNA2 attention backend for gfx1030.

This module loads the FA RDNA2 kernels (paged decode + paged prefill)
via torch.utils.cpp_extension.load_inline and exposes Python-callable
functions. It is opt-in via VLLM_USE_RDNA2_FA=1.

The kernel is loaded lazily on first call to avoid compilation
cost when the backend is not selected.

Usage:
    from vllm.attention.ops.fa_rdna2_backend import fa_rdna2_decode_paged
    out = fa_rdna2_decode_paged(Q, key_cache, value_cache, block_table,
                                seq_lens, block_size, kv_splits=8)
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 256
_DYNAMO_CONFIG_APPLIED = True


_KERNEL_SRC = None
_EXT = None


def _load_kernel():
    """Compile and load the FA RDNA2 kernel via load_inline."""
    global _KERNEL_SRC, _EXT
    if _EXT is not None:
        return _EXT

    import pathlib
    kernel_path = (
        pathlib.Path(__file__).resolve().parents[4]
        / "csrc" / "rocm" / "fa_rdna2.cu"
    )
    if not kernel_path.is_file():
        raise FileNotFoundError(
            f"fa_rdna2.cu not found at {kernel_path}. "
            "Ensure csrc/rocm/fa_rdna2.cu is in the vLLM source tree.")

    with open(kernel_path) as f:
        _KERNEL_SRC = f.read()

    # load_inline does not inherit the ROCm SDK include/library paths that
    # the main cmake build gets from ROCM_HOME, so torch's header-only
    # include chain (thrust/complex.h etc.) and the hip runtime link
    # (-lamdhip64) are not found by default. Probe candidates in order.
    _extra_rocm_includes = []
    _extra_rocm_libdirs = []
    for _cand in ("/opt/rocm-7.2.0/core-7.14", "/opt/rocm", "/opt/rocm-7.2.0"):
        if pathlib.Path(_cand, "include", "thrust", "complex.h").is_file():
            _extra_rocm_includes = ["-isystem" + _cand + "/include"]
        if pathlib.Path(_cand, "lib", "libamdhip64.so").is_file():
            _extra_rocm_libdirs = ["-L" + _cand + "/lib"]

    cpp_src = """
torch::Tensor fa_rdna2_decode_paged(torch::Tensor Q,
                                    torch::Tensor key_cache,
                                    torch::Tensor value_cache,
                                    torch::Tensor block_table,
                                    torch::Tensor seq_lens,
                                    int64_t block_size, int64_t kv_splits,
                                    int64_t sliding_window);
torch::Tensor fa_rdna2_decode_paged_fp8(torch::Tensor Q,
                                        torch::Tensor key_cache,
                                        torch::Tensor value_cache,
                                        torch::Tensor block_table,
                                        torch::Tensor seq_lens,
                                        int64_t block_size, int64_t kv_splits,
                                        int64_t sliding_window,
                                        double k_scale, double v_scale);
torch::Tensor fa_rdna2_prefill_paged_varlen(torch::Tensor Q,
                                            torch::Tensor key_cache,
                                            torch::Tensor value_cache,
                                            torch::Tensor block_table,
                                            torch::Tensor cu_query_lens,
                                            torch::Tensor seq_lens,
                                            int64_t block_size,
                                            int64_t causal,
                                            int64_t sliding_window);
torch::Tensor fa_rdna2_prefill_paged_varlen_fp8(torch::Tensor Q,
                                                torch::Tensor key_cache,
                                                torch::Tensor value_cache,
                                                torch::Tensor block_table,
                                                torch::Tensor cu_query_lens,
                                                torch::Tensor seq_lens,
                                                int64_t block_size,
                                                int64_t causal,
                                                int64_t sliding_window,
                                                double k_scale,
                                                double v_scale);
torch::Tensor fa_rdna2_decode_paged_int8(torch::Tensor Q,
                                         torch::Tensor key_cache,
                                         torch::Tensor value_cache,
                                         torch::Tensor block_table,
                                         torch::Tensor seq_lens,
                                         int64_t block_size, int64_t kv_splits,
                                         int64_t sliding_window,
                                         torch::Tensor k_scale,
                                         torch::Tensor v_scale);
void reshape_and_cache_int8_rdna2(torch::Tensor key, torch::Tensor value,
                                  torch::Tensor kv_cache,
                                  torch::Tensor slot_mapping);
torch::Tensor fa_rdna2_prefill_paged_varlen_short(torch::Tensor Q,
                                                  torch::Tensor key_cache,
                                                  torch::Tensor value_cache,
                                                  torch::Tensor block_table,
                                                  torch::Tensor cu_query_lens,
                                                  torch::Tensor seq_lens,
                                                  int64_t block_size,
                                                  int64_t causal,
                                                  int64_t sliding_window);
torch::Tensor fa_rdna2_prefill_paged_varlen_splitk(torch::Tensor Q,
                                                  torch::Tensor key_cache,
                                                  torch::Tensor value_cache,
                                                  torch::Tensor block_table,
                                                  torch::Tensor cu_query_lens,
                                                  torch::Tensor seq_lens,
                                                  int64_t block_size,
                                                  int64_t causal,
                                                  int64_t kv_splits,
                                                  int64_t sliding_window);
torch::Tensor fa_rdna2_prefill_paged_varlen_int8(torch::Tensor Q,
                                                torch::Tensor key_cache,
                                                torch::Tensor value_cache,
                                                torch::Tensor block_table,
                                                torch::Tensor cu_query_lens,
                                                torch::Tensor seq_lens,
                                                int64_t block_size,
                                                int64_t causal,
                                                int64_t sliding_window,
                                                int64_t kv_splits,
                                                torch::Tensor k_scale,
                                                torch::Tensor v_scale);
"""

    _EXT = load_inline(
        name="fa_rdna2_backend",
        cpp_sources=[cpp_src],
        cuda_sources=[_KERNEL_SRC],
        functions=["fa_rdna2_decode_paged",
                   "fa_rdna2_decode_paged_fp8",
                   "fa_rdna2_decode_paged_int8",
                   "fa_rdna2_prefill_paged_varlen",
                   "fa_rdna2_prefill_paged_varlen_fp8",
                   "fa_rdna2_prefill_paged_varlen_short",
                   "fa_rdna2_prefill_paged_varlen_splitk",
                   "fa_rdna2_prefill_paged_varlen_int8",
                   "reshape_and_cache_int8_rdna2"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx1030"]
                          + _extra_rocm_includes,
        extra_ldflags=_extra_rocm_libdirs,
        verbose=False,
    )
    return _EXT


def is_available() -> bool:
    """Check if FA RDNA2 backend is available.

    Returns True only when:
    - VLLM_USE_RDNA2_FA=1 is set
    - torch CUDA is available (HIP)
    - The GPU is gfx1030 (RDNA2)
    """
    if os.environ.get("VLLM_USE_RDNA2_FA") != "1":
        return False
    if not torch.cuda.is_available():
        return False
    # Cover gfx1030 family (gfx1030/1031/1032/1035). The kernel itself is
    # compiled for gfx1030 only via load_inline; the family match is
    # intentional since other gfx103x variants share the same ISA.
    try:
        props = torch.cuda.get_device_properties(0)
        if "gfx103" not in props.gcnArchName:
            return False
    except Exception:
        return False
    return True




def fa_rdna2_decode_paged(Q: torch.Tensor,
                          key_cache: torch.Tensor,
                          value_cache: torch.Tensor,
                          block_table: torch.Tensor,
                          seq_lens: torch.Tensor,
                          block_size: int = 16,
                          kv_splits: int = 8,
                          sliding_window: int = 0) -> torch.Tensor:
    """FA2 decode kernel for gfx1030 reading from paged KV cache.

    Args:
        Q: [num_tokens, H_q, D] fp16 query tensor
        key_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged K cache
        value_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged V cache
        block_table: [num_tokens, max_blocks] int32 per-query block indices
        seq_lens: [num_tokens] int32 per-query KV length
        block_size: physical block size (16, 32, etc.)
        kv_splits: number of CTAs per head (1..16)
        sliding_window: sliding window size (0 = no window)

    Returns:
        O: [num_tokens, H_q, D] fp16 attention output
    """
    ext = _load_kernel()
    return ext.fa_rdna2_decode_paged(
        Q, key_cache, value_cache, block_table, seq_lens,
        block_size, kv_splits, sliding_window)


def fa_rdna2_decode_paged_fp8(Q: torch.Tensor,
                              key_cache: torch.Tensor,
                              value_cache: torch.Tensor,
                              block_table: torch.Tensor,
                              seq_lens: torch.Tensor,
                              block_size: int = 16,
                              kv_splits: int = 8,
                              sliding_window: int = 0,
                              k_scale: float = 1.0,
                              v_scale: float = 1.0) -> torch.Tensor:
    """FA2 decode kernel for gfx1030 with fp8 (e4m3) KV cache.

    Same kernel as fa_rdna2_decode_paged but K/V are read from an fp8
    cache (uint8 storage) and dequantized fp8->fp16 inline at the KV
    load point inside the HIP kernel, applying the per-tensor scales.

    Args:
        Q: [num_tokens, H_q, D] fp16 query tensor
        key_cache: [num_blocks, H_kv, D/x, block_size, x] uint8 fp8 K cache
        value_cache: [num_blocks, H_kv, D/x, block_size, x] uint8 fp8 V cache
        block_table: [num_tokens, max_blocks] int32 per-query block indices
        seq_lens: [num_tokens] int32 per-query KV length
        block_size: physical block size (16, 32, etc.)
        kv_splits: number of CTAs per head (1..16)
        sliding_window: sliding window size (0 = no window)
        k_scale: per-tensor fp8 scale for K
        v_scale: per-tensor fp8 scale for V

    Returns:
        O: [num_tokens, H_q, D] fp16 attention output
    """
    ext = _load_kernel()
    if key_cache.dtype != torch.uint8:
        key_cache = key_cache.view(torch.uint8)
    if value_cache.dtype != torch.uint8:
        value_cache = value_cache.view(torch.uint8)
    # DEBUG: log tensor shapes for diagnosis
    import sys
    return ext.fa_rdna2_decode_paged_fp8(
        Q, key_cache, value_cache, block_table, seq_lens,
        block_size, kv_splits, sliding_window, float(k_scale), float(v_scale))






def fa_rdna2_prefill_paged_varlen(Q: torch.Tensor,
                                   key_cache: torch.Tensor,
                                   value_cache: torch.Tensor,
                                   block_table: torch.Tensor,
                                   cu_query_lens: torch.Tensor,
                                   seq_lens: torch.Tensor,
                                   block_size: int = 16,
                                   causal: bool = True,
                                   sliding_window: int = 0) -> torch.Tensor:
    """FA2 paged prefill kernel for gfx1030 with varlen (multiple sequences).

    Gap 2: supports chunked prefill with multiple sequences per launch.
    Each query block of BR_PREFILL tokens may span a different sequence.
    The kernel uses cu_query_lens to determine which sequence each block
    belongs to, then reads that sequence's seq_lens and block_table slice.

    Supports HEAD_DIM=128 and HEAD_DIM=256.

    Args:
        Q: [num_tokens, H_q, D] fp16 query tensor
        key_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged K cache
        value_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged V cache
        block_table: [num_seqs, max_blocks] int32 per-sequence block indices
        cu_query_lens: [num_seqs + 1] int32 cumulative query counts
        seq_lens: [num_seqs] int32 per-sequence KV length
        block_size: physical block size (16, 32, 784, etc.)
        causal: whether to apply causal masking (per-sequence)

    Returns:
        O: [num_tokens, H_q, D] fp16 attention output
    """
    ext = _load_kernel()
    return ext.fa_rdna2_prefill_paged_varlen(
        Q, key_cache, value_cache, block_table, cu_query_lens, seq_lens,
        block_size, int(causal), sliding_window)


def fa_rdna2_prefill_paged_varlen_fp8(Q: torch.Tensor,
                                      key_cache: torch.Tensor,
                                      value_cache: torch.Tensor,
                                      block_table: torch.Tensor,
                                      cu_query_lens: torch.Tensor,
                                      seq_lens: torch.Tensor,
                                      block_size: int = 16,
                                      causal: bool = True,
                                      sliding_window: int = 0,
                                      k_scale: float = 1.0,
                                      v_scale: float = 1.0) -> torch.Tensor:
    """FA2 paged prefill kernel for gfx1030 with fp8 (e4m3) KV cache.

    Same kernel as fa_rdna2_prefill_paged_varlen but K/V are read from an
    fp8 cache (uint8 storage) and dequantized inline at the KV load point.

    Supports HEAD_DIM=128 and HEAD_DIM=256.

    Args:
        Q: [num_tokens, H_q, D] fp16 query tensor
        key_cache: [num_blocks, H_kv, D/x, block_size, x] uint8 fp8 K cache
        value_cache: [num_blocks, H_kv, D/x, block_size, x] uint8 fp8 V cache
        block_table: [num_seqs, max_blocks] int32 per-sequence block indices
        cu_query_lens: [num_seqs + 1] int32 cumulative query counts
        seq_lens: [num_seqs] int32 per-sequence KV length
        block_size: physical block size (16, 32, 784, etc.)
        causal: whether to apply causal masking (per-sequence)
        sliding_window: sliding window size (0 = no window)
        k_scale: per-tensor fp8 scale for K
        v_scale: per-tensor fp8 scale for V

    Returns:
        O: [num_tokens, H_q, D] fp16 attention output
    """
    ext = _load_kernel()
    if key_cache.dtype != torch.uint8:
        key_cache = key_cache.view(torch.uint8)
    if value_cache.dtype != torch.uint8:
        value_cache = value_cache.view(torch.uint8)
    return ext.fa_rdna2_prefill_paged_varlen_fp8(
        Q, key_cache, value_cache, block_table, cu_query_lens, seq_lens,
        block_size, int(causal), sliding_window, float(k_scale),
        float(v_scale))


def fa_rdna2_prefill_paged_varlen_short(Q: torch.Tensor,
                                        key_cache: torch.Tensor,
                                        value_cache: torch.Tensor,
                                        block_table: torch.Tensor,
                                        cu_query_lens: torch.Tensor,
                                        seq_lens: torch.Tensor,
                                        block_size: int = 16,
                                        causal: bool = True,
                                        sliding_window: int = 0) -> torch.Tensor:
    """Sub-4096 optimized FA2 paged prefill kernel for gfx1030 (HEAD_DIM=128).

    Uses BR_PREFILL=32, THREADS_PREFILL=256 for better grid utilization
    at short sequence lengths (< 4096 tokens). Larger BR processes more
    query tokens per CTA, more threads improve warp utilization.

    Only valid for HEAD_DIM=128. For HEAD_DIM=256, use
    fa_rdna2_prefill_paged_varlen (which is gated to max_seq_len >= 4096).

    Args:
        Q: [num_tokens, H_q, 128] fp16 query tensor
        key_cache: [num_blocks, H_kv, 128/x, block_size, x] fp16 paged K cache
        value_cache: [num_blocks, H_kv, 128/x, block_size, x] fp16 paged V cache
        block_table: [num_seqs, max_blocks] int32 per-sequence block indices
        cu_query_lens: [num_seqs + 1] int32 cumulative query counts
        seq_lens: [num_seqs] int32 per-sequence KV length
        block_size: physical block size (16, 32, 784, etc.)
        causal: whether to apply causal masking (per-sequence)
        sliding_window: sliding window size (0 = no window)

    Returns:
        O: [num_tokens, H_q, 128] fp16 attention output
    """
    ext = _load_kernel()
    return ext.fa_rdna2_prefill_paged_varlen_short(
        Q, key_cache, value_cache, block_table, cu_query_lens, seq_lens,
        block_size, int(causal), sliding_window)


def fa_rdna2_prefill_paged_varlen_splitk(Q: torch.Tensor,
                                         key_cache: torch.Tensor,
                                         value_cache: torch.Tensor,
                                         block_table: torch.Tensor,
                                         cu_query_lens: torch.Tensor,
                                         seq_lens: torch.Tensor,
                                         block_size: int = 16,
                                         causal: bool = True,
                                         sliding_window: int = 0,
                                         kv_splits: int = 4) -> torch.Tensor:
    """FA2 paged prefill with split-K varlen for better grid utilization.

    Partitions the KV sequence across kv_splits CTAs per (q_block, h_q).
    Each split produces partial O/M/L; a reduction kernel merges them.
    Use for large seq_len where H_q CTAs per q_block underutilize the GPU.

    Args:
        Q: [num_tokens, H_q, D] fp16 query tensor
        key_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged K cache
        value_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged V cache
        block_table: [num_seqs, max_blocks] int32 per-sequence block indices
        cu_query_lens: [num_seqs + 1] int32 cumulative query counts
        seq_lens: [num_seqs] int32 per-sequence KV length
        block_size: physical block size (16, 32, 784, etc.)
        causal: whether to apply causal masking (per-sequence)
        kv_splits: number of KV splits (1..16). 1 = no split.
        sliding_window: sliding window size (0 = no window)

    Returns:
        O: [num_tokens, H_q, D] fp16 attention output
    """
    ext = _load_kernel()
    return ext.fa_rdna2_prefill_paged_varlen_splitk(
        Q, key_cache, value_cache, block_table, cu_query_lens, seq_lens,
        block_size, int(causal), int(kv_splits), sliding_window)


def fa_rdna2_prefill_paged_varlen_int8(Q: torch.Tensor,
                                       key_cache: torch.Tensor,
                                       value_cache: torch.Tensor,
                                       block_table: torch.Tensor,
                                       cu_query_lens: torch.Tensor,
                                       seq_lens: torch.Tensor,
                                       block_size: int = 16,
                                       causal: bool = True,
                                       sliding_window: int = 0,
                                       kv_splits: int = 4,
                                       k_scale: torch.Tensor | None = None,
                                       v_scale: torch.Tensor | None = None) -> torch.Tensor:
    """FA2 paged prefill with split-K varlen for int8 per-token-head KV.

    Replaces the Python-side int8 dequant path that used to live in
    rdna_attn.py. Reads K/V from an int8 per-token-head cache and
    dequantizes inline using per-(token, head) fp32 scale tables, using
    the v3 wiki "live contract" ISA pattern (packed int reads from sK,
    fused i8->fp32->scale->fp16->V_DOT2_F32_F16 against fp16 Q).

    Args:
        Q: [num_tokens, H_q, D] fp16 query tensor
        key_cache: [num_blocks, H_kv, D, block_size, 1] int8 paged K cache
            (x_dim=1 for int8 cache; int8 has no fp16-style packing)
        value_cache: [num_blocks, H_kv, D, block_size, 1] int8 paged V cache
        block_table: [num_seqs, max_blocks] int32 per-sequence block indices
        cu_query_lens: [num_seqs + 1] int32 cumulative query counts
        seq_lens: [num_seqs] int32 per-sequence KV length
        block_size: physical block size (must be 16 for the int8 path)
        causal: whether to apply causal masking (per-sequence)
        sliding_window: sliding window size (0 = no window)
        kv_splits: number of KV splits (1..16). 1 = no split.
        k_scale: [num_tokens, H_kv] fp32 per-(token, head) K scales
        v_scale: [num_tokens, H_kv] fp32 per-(token, head) V scales

    Returns:
        O: [num_tokens, H_q, D] fp16 attention output
    """
    ext = _load_kernel()
    if k_scale is None or v_scale is None:
        raise ValueError(
            "fa_rdna2_prefill_paged_varlen_int8 requires k_scale/v_scale tensors")
    if k_scale.dtype != torch.float32:
        k_scale = k_scale.float()
    if v_scale.dtype != torch.float32:
        v_scale = v_scale.float()
    return ext.fa_rdna2_prefill_paged_varlen_int8(
        Q, key_cache, value_cache, block_table, cu_query_lens, seq_lens,
        block_size, int(causal), int(sliding_window), int(kv_splits),
        k_scale, v_scale)


fa_rdna2_decode_paged = torch.compiler.allow_in_graph(fa_rdna2_decode_paged)
fa_rdna2_decode_paged_fp8 = torch.compiler.allow_in_graph(fa_rdna2_decode_paged_fp8)
fa_rdna2_prefill_paged_varlen = torch.compiler.allow_in_graph(fa_rdna2_prefill_paged_varlen)
fa_rdna2_prefill_paged_varlen_fp8 = torch.compiler.allow_in_graph(fa_rdna2_prefill_paged_varlen_fp8)
fa_rdna2_prefill_paged_varlen_short = torch.compiler.allow_in_graph(fa_rdna2_prefill_paged_varlen_short)
fa_rdna2_prefill_paged_varlen_splitk = torch.compiler.allow_in_graph(fa_rdna2_prefill_paged_varlen_splitk)
fa_rdna2_prefill_paged_varlen_int8 = torch.compiler.allow_in_graph(fa_rdna2_prefill_paged_varlen_int8)


def fa_rdna2_decode_paged_int8(Q, key_cache, value_cache, block_table, seq_lens,
                               block_size=16, kv_splits=8, sliding_window=0,
                               k_scale=None, v_scale=None):
    ext = _load_kernel()
    if k_scale is None or v_scale is None:
        raise ValueError("fa_rdna2_decode_paged_int8 requires k_scale/v_scale tensors")
    if k_scale.dtype != torch.float32:
        k_scale = k_scale.float()
    if v_scale.dtype != torch.float32:
        v_scale = v_scale.float()
    return ext.fa_rdna2_decode_paged_int8(
        Q, key_cache, value_cache, block_table, seq_lens,
        block_size, kv_splits, sliding_window, k_scale, v_scale)


fa_rdna2_decode_paged_int8 = torch.compiler.allow_in_graph(fa_rdna2_decode_paged_int8)


def reshape_and_cache_int8_rdna2(key, value, kv_cache, slot_mapping):
    """INT8 per-(token, head) KV-cache writer for RDNA2 (gfx1030).

    Quantizes fp16 K/V to int8 with per-(token, head) scales (one
    absmax / 127 per (token, head) pair) and writes them into the
    interleaved cache layout consumed by fa_rdna2_decode_paged_int8:

        kv_cache: [2, num_blocks, H_kv, D + 4, block_size] int8
        - kv_cache[0, b, h, 0..D, s]      : K int8 bytes
        - kv_cache[0, b, h, D..D + 4, s]  : K scale (raw fp32 LE)
        - kv_cache[1, ...]                : V same

    Per the kv-int8.md wiki contract. Slot mapping values of -1 are
    treated as no-op (matching the vLLM standard semantics).

    Returns:
        None — writes into kv_cache in place.
    """
    ext = _load_kernel()
    return ext.reshape_and_cache_int8_rdna2(
        key, value, kv_cache, slot_mapping)


reshape_and_cache_int8_rdna2 = torch.compiler.allow_in_graph(
    reshape_and_cache_int8_rdna2)

_RDNA2_GFX10X = False
try:
    if torch.cuda.is_available():
        _props = torch.cuda.get_device_properties(0)
        _arch = getattr(_props, "gcnArchName", "")
        if _arch.startswith("gfx103") or _arch.startswith("gfx10"):
            _RDNA2_GFX10X = True
except Exception:
    _RDNA2_GFX10X = False

if _RDNA2_GFX10X:
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator as _PC
        if not getattr(_PC, "_rdna2_patched", False):
            _PC.disabled = property(lambda self: True)
            _PC._rdna2_patched = True
    except Exception as _e:
        import warnings as _w
        _w.warn(f"fa_rdna2 PyNcclCommunicator disabled-patch failed: {_e}", RuntimeWarning)



