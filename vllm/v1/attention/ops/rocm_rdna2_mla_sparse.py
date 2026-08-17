"""Sparse MLA attention kernels for gfx1030 (RDNA2).

This module mirrors the Triton kernels in
`vllm.v1.attention.ops.rocm_aiter_mla_sparse` but is isolated to the
gfx1030 family. RDNA2 has no native bf16 dot2 (`v_dot2_f32_bf16` is
RDNA3+), so these kernels run the dot operands in fp16
(`v_dot2_f32_f16`). The CDNA path keeps its bf16 kernels untouched in
the aiter module.

Activation:
  - VLLM_USE_RDNA2_MLA=1  → call the HIP kernel when it is registered
  - otherwise            → fp16 Triton kernels in this module

Architecture gate:
  - Only active on gfx1030/gfx1031/gfx1032/gfx1035. The dispatcher
    in deepseek_v4/amd/rocm.py only routes here when on_gfx10x() is True.
"""
import os
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _as_int32_contiguous_1d,
    _expand_2d_block_scales,
    _validate_dsv4_sparse_dims,
    build_ragged_indices_from_dense,
)

logger = init_logger(__name__)


_KERNEL_AVAILABLE: bool | None = None


def _kernel_available() -> bool:
    global _KERNEL_AVAILABLE
    if _KERNEL_AVAILABLE is None:
        try:
            from vllm import _rocm_C  # noqa: F401
            _KERNEL_AVAILABLE = hasattr(
                torch.ops._rocm_C, "sparse_mla_decode_rdna2")
        except Exception:
            _KERNEL_AVAILABLE = False
    return _KERNEL_AVAILABLE


def is_available() -> bool:
    """Return True if the HIP sparse MLA decode kernel can be used."""
    if os.environ.get("VLLM_USE_RDNA2_MLA") != "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        from vllm.platforms.rocm import on_gfx10x
        if not on_gfx10x():
            return False
    except Exception:
        return False
    return _kernel_available()


@triton.jit
def _rdna2_sparse_attn_prefill_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    slot = tl.load(
        kv_indices_ptr + kv_start + k_offsets, mask=k_offsets < kv_len, other=-1
    )
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        valid = in_range & (slot >= 0) & (slot < num_kv)
        safe_slot = tl.where(valid, slot, 0)

        kv = tl.load(
            kv_ptr
            + safe_slot[:, None] * kv_stride_n
            + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )

        next_k_pos = k_start + BLOCK_K + k_offsets
        slot = tl.load(
            kv_indices_ptr + kv_start + next_k_pos, mask=next_k_pos < kv_len, other=-1
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _rdna2_sparse_attn_decode_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    # SWA K-cache (main): C++ encoder writes FNUZ on gfx942, OCP on gfx950.
    # Compressed K-cache (extra): Triton encoder writes OCP everywhere.
    IS_FNUZ_MAIN: tl.constexpr,
    IS_FNUZ_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start

    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.float16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.float16)

    for k_start in tl.range(0, main_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_len
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ_MAIN:
            x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
        else:
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = x_fp8.to(tl.float16) * scales.to(tl.float16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float16)
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start

        for k_start in tl.range(0, extra_len, BLOCK_K):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_len
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            if IS_FNUZ_EXTRA:
                x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
            else:
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.float16) * scales.to(tl.float16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float16)
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope,
                tl.trans(k_rope),
            )
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out_nope = tl.where(
            l_final[:, None] > 0.0,
            (acc_nope * alpha[:, None]) / denom[:, None],
            0.0,
        )
        out_rope = tl.where(
            l_final[:, None] > 0.0,
            (acc_rope * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out_nope = tl.where(l_i[:, None] > 0.0, acc_nope / denom[:, None], 0.0)
        out_rope = tl.where(l_i[:, None] > 0.0, acc_rope / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + nope_offsets[None, :],
        out_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets[None, :],
        out_rope,
        mask=head_mask[:, None],
    )


def _rdna2_sparse_attn_decode_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {main_cache.shape}"
    )
    assert main_indices.ndim == 1, (
        f"expected main_indices=[nnz], got {main_indices.shape}"
    )
    assert main_indptr.ndim == 1, f"expected main_indptr=[b+1], got {main_indptr.shape}"
    assert (
        not q.is_cpu
        and not main_cache.is_cpu
        and not main_indices.is_cpu
        and not main_indptr.is_cpu
    )

    main_indices = _as_int32_contiguous_1d(main_indices)
    main_indptr = _as_int32_contiguous_1d(main_indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert main_indptr.numel() == num_queries + 1, (
        f"expected main_indptr shape [{num_queries + 1}], got {main_indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rdna2_sparse_attn_decode_triton",
    )

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_indptr is not None
        assert extra_indices.ndim == 1, (
            f"expected extra_indices=[nnz], got {extra_indices.shape}"
        )
        assert extra_indptr.ndim == 1, (
            f"expected extra_indptr=[b+1], got {extra_indptr.shape}"
        )
        extra_indices = _as_int32_contiguous_1d(extra_indices)
        extra_indptr = _as_int32_contiguous_1d(extra_indptr)
        assert extra_indptr.numel() == num_queries + 1, (
            f"expected extra_indptr shape [{num_queries + 1}], got {extra_indptr.shape}"
        )
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    block_h = 16
    out = torch.empty_like(q)
    heads_blocks = triton.cdiv(num_heads, block_h)
    nope_block = triton.next_power_of_2(nope_head_dim)
    is_fnuz = current_platform.is_fp8_fnuz()

    # gfx1030 Wave32 benefits from larger BLOCK_K. 32k-context
    # decode iterations drop from 1k to 0.5k for BLOCK_K=64.
    block_k = 64 if head_dim < 256 else 32
    _rdna2_sparse_attn_decode_kernel[(num_queries, heads_blocks)](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        scale,
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=nope_block,
        ROPE_DIM=rope_head_dim,
        IS_FNUZ_MAIN=is_fnuz,
        IS_FNUZ_EXTRA=False,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


def _rdna2_sparse_attn_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 1, f"expected indices=[nnz], got {indices.shape}"
    assert indptr.ndim == 1, f"expected indptr=[sq+1], got {indptr.shape}"
    assert not q.is_cpu and not kv.is_cpu and not indices.is_cpu and not indptr.is_cpu

    if kv.dtype != q.dtype:
        kv = kv.to(q.dtype)

    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rdna2_sparse_attn_prefill_triton",
    )

    block_h = 16
    block_d = triton.next_power_of_2(head_dim)
    block_k = 16 if head_dim >= 256 else 32
    num_warps = 4
    out = torch.empty_like(q)
    _rdna2_sparse_attn_prefill_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out


def _hip_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    swa_ragged_indices: torch.Tensor,
    swa_ragged_indptr: torch.Tensor,
    attn_sink: torch.Tensor | None,
    scale: float,
    output: torch.Tensor,
) -> None:
    logger.info_once(
        "RDNA2 sparse MLA decode using HIP kernel sparse_mla_decode_rdna2")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"RDNA2 HIP sparse MLA decode expects fp16 or bf16 q, got {q.dtype}")
    B, H, D = q.shape
    main_block_size = swa_k_cache.size(1)
    main_num_rows = swa_k_cache.size(0) * main_block_size

    if swa_only:
        extra_cache = torch.empty(0, device=q.device, dtype=torch.uint8)
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(B + 1, device=q.device, dtype=torch.int32)
        extra_block_size = 0
        extra_num_rows = 0
    else:
        assert kv_cache is not None
        assert topk_ragged_indices is not None and topk_ragged_indptr is not None
        extra_cache = kv_cache
        extra_indices = topk_ragged_indices
        extra_indptr = topk_ragged_indptr
        extra_block_size = kv_cache.size(1)
        extra_num_rows = kv_cache.size(0) * extra_block_size

    if attn_sink is None:
        sink = torch.empty(0, device=q.device, dtype=torch.float32)
    else:
        sink = attn_sink[:H].to(torch.float32).contiguous()

    out = torch.empty(B, H, D, device=q.device, dtype=q.dtype)
    torch.ops._rocm_C.sparse_mla_decode_rdna2(
        q,
        swa_k_cache,
        swa_ragged_indices.to(torch.int32).contiguous(),
        swa_ragged_indptr.to(torch.int32).contiguous(),
        extra_cache,
        extra_indices.to(torch.int32).contiguous(),
        extra_indptr.to(torch.int32).contiguous(),
        main_block_size,
        main_num_rows,
        extra_block_size,
        extra_num_rows,
        float(scale),
        sink,
        out,
    )
    output.copy_(out)


def rocm_rdna2_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    """Sparse MLA decode for gfx1030: HIP kernel when available, else fp16 Triton."""
    if (
        is_available()
        and swa_ragged_indices is not None
        and swa_ragged_indptr is not None
    ):
        _hip_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_only=swa_only,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            swa_ragged_indices=swa_ragged_indices,
            swa_ragged_indptr=swa_ragged_indptr,
            attn_sink=attn_sink,
            scale=scale,
            output=output,
        )
        return

    assert swa_k_cache.dtype == torch.uint8, (
        "RDNA2 Triton sparse decode expects uint8 fp8_ds_mla SWA cache, "
        f"got {swa_k_cache.dtype}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_rdna2_sparse_attn_decode",
    )
    if q.dtype != torch.float16:
        q = q.to(torch.float16)

    if swa_ragged_indices is None or swa_ragged_indptr is None:
        main_indices_2d = swa_indices.reshape(swa_indices.shape[0], -1)
        main_ragged_indices, main_ragged_indptr = build_ragged_indices_from_dense(
            main_indices_2d,
            swa_lens if swa_lens is not None else (main_indices_2d >= 0).sum(
                dim=-1, dtype=torch.int32),
            num_rows=swa_k_cache.shape[0] * swa_k_cache.shape[1],
        )
    else:
        main_ragged_indices = swa_ragged_indices
        main_ragged_indptr = swa_ragged_indptr

    extra_cache = None
    extra_indices = None
    extra_indptr = None
    if not swa_only:
        assert kv_cache is not None
        assert topk_indices is not None or (
            topk_ragged_indices is not None and topk_ragged_indptr is not None
        )
        assert kv_cache.dtype == torch.uint8, (
            "RDNA2 Triton sparse decode expects uint8 fp8_ds_mla extra cache, "
            f"got {kv_cache.dtype}"
        )
        extra_cache = kv_cache
        if topk_ragged_indices is not None and topk_ragged_indptr is not None:
            extra_indices = topk_ragged_indices
            extra_indptr = topk_ragged_indptr
        else:
            assert topk_indices is not None
            extra_indices_2d = topk_indices.reshape(topk_indices.shape[0], -1)
            extra_indices, extra_indptr = build_ragged_indices_from_dense(
                extra_indices_2d,
                topk_lens
                if topk_lens is not None
                else (extra_indices_2d >= 0).sum(dim=-1, dtype=torch.int32),
                num_rows=kv_cache.shape[0] * kv_cache.shape[1],
            )

    attn_out = _rdna2_sparse_attn_decode_triton(
        q=q,
        main_cache=swa_k_cache,
        main_indices=main_ragged_indices,
        main_indptr=main_ragged_indptr,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
    )
    output.copy_(attn_out.to(output.dtype))


def _hip_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    logger.info_once(
        "RDNA2 sparse MLA prefill using HIP kernel sparse_mla_prefill_rdna2")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"RDNA2 HIP sparse MLA prefill expects fp16 or bf16 q, got {q.dtype}")
    B, H, D = q.shape
    num_kv = kv.shape[0]
    if attn_sink is None:
        sink = torch.empty(0, device=q.device, dtype=torch.float32)
    else:
        sink = attn_sink[:H].to(torch.float32).contiguous()
    out = torch.empty(B, H, D, device=q.device, dtype=q.dtype)
    torch.ops._rocm_C.sparse_mla_prefill_rdna2(
        q,
        kv,
        indices.to(torch.int32).contiguous(),
        indptr.to(torch.int32).contiguous(),
        num_kv,
        float(scale),
        sink,
        out,
    )
    output.copy_(out)


def rocm_rdna2_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
) -> None:
    assert kv.ndim == 3 and kv.shape[1] == 1, (
        f"RDNA2 Triton sparse prefill expects kv=[skv,1,d], got {kv.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_rdna2_sparse_attn_prefill",
    )
    # HIP path: plain fp16/bf16 kv rows + ragged indices, no fp8 slots.
    if is_available():
        if ragged_indices is not None and ragged_indptr is not None:
            _hip_sparse_attn_prefill(
                q=q,
                kv=kv.squeeze(1),
                indices=ragged_indices,
                indptr=ragged_indptr,
                scale=scale,
                attn_sink=attn_sink,
                output=output,
            )
        else:
            indices_2d = indices.reshape(indices.shape[0], -1)
            rag_indices, rag_indptr = build_ragged_indices_from_dense(
                indices_2d,
                topk_length
                if topk_length is not None
                else (indices_2d >= 0).sum(dim=-1, dtype=torch.int32),
                num_rows=kv.shape[0],
            )
            _hip_sparse_attn_prefill(
                q=q,
                kv=kv.squeeze(1),
                indices=rag_indices,
                indptr=rag_indptr,
                scale=scale,
                attn_sink=attn_sink,
                output=output,
            )
        return

    if q.dtype != torch.float16:
        q = q.to(torch.float16)
    if kv.dtype != torch.float16:
        kv = kv.to(torch.float16)
    if ragged_indices is not None and ragged_indptr is not None:
        output_chunk = _rdna2_sparse_attn_prefill_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=ragged_indices,
            indptr=ragged_indptr,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
        )
    else:
        indices_2d = indices.reshape(indices.shape[0], -1)
        ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
            indices_2d,
            topk_length
            if topk_length is not None
            else (indices_2d >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=kv.shape[0],
        )
        output_chunk = _rdna2_sparse_attn_prefill_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=ragged_indices,
            indptr=ragged_indptr,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
        )
    output.copy_(output_chunk.to(output.dtype))


@triton.jit
def _rdna2_inverse_rope_gptj_kernel(
    o_ptr,  # [T, H, D] input
    out_ptr,  # [T, H, D] fp16 output
    pos_ptr,  # [T] positions
    cos_sin_ptr,  # [P, rope_dim] fp32 (cos[:half] | sin[half:])
    s_t,
    s_h,  # input row strides (last dim contiguous)
    os_t,
    os_h,  # output row strides
    cs_stride,  # cos_sin_cache row stride
    NOPE: tl.constexpr,  # non-rope head dims (passed through)
    HALF: tl.constexpr,  # rope_dim // 2
    BLOCK_NOPE: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    """Fused inverse GPT-J RoPE on the trailing rope_dim of each (token, head).

    Mirrors ``DeepseekV4ScalingRotaryEmbedding.forward_native(inverse=True)``
    for the GPT-J (non-neox) layout, writing fp16 directly. Replaces the
    clone + index_select + repeat_interleave + neg + stack + cat + cast chain
    (~10 small kernels) with a single launch.
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    in_base = t * s_t + h * s_h
    out_base = t * os_t + h * os_h

    # NoPE lanes pass through unchanged (only cast to fp16).
    n = tl.arange(0, BLOCK_NOPE)
    nmask = n < NOPE
    vals = tl.load(o_ptr + in_base + n, mask=nmask)
    tl.store(out_ptr + out_base + n, vals.to(tl.float16), mask=nmask)

    # RoPE lanes: out_even = a*cos + b*sin, out_odd = b*cos - a*sin
    # (a = even lane, b = odd lane; sin negated for the inverse rotation).
    pos = tl.load(pos_ptr + t).to(tl.int64)
    k = tl.arange(0, BLOCK_HALF)
    kmask = k < HALF
    a = tl.load(o_ptr + in_base + NOPE + 2 * k, mask=kmask).to(tl.float32)
    b = tl.load(o_ptr + in_base + NOPE + 2 * k + 1, mask=kmask).to(tl.float32)
    cos = tl.load(cos_sin_ptr + pos * cs_stride + k, mask=kmask)
    sin = tl.load(cos_sin_ptr + pos * cs_stride + HALF + k, mask=kmask)
    out_even = a * cos + b * sin
    out_odd = b * cos - a * sin
    tl.store(out_ptr + out_base + NOPE + 2 * k, out_even.to(tl.float16), mask=kmask)
    tl.store(
        out_ptr + out_base + NOPE + 2 * k + 1, out_odd.to(tl.float16), mask=kmask
    )


def _rdna2_fused_inverse_rope_gptj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    """fp16 inverse GPT-J RoPE via a single fused Triton kernel."""
    assert o.dim() == 3 and o.stride(-1) == 1, (
        "_rdna2_fused_inverse_rope_gptj expects a [T, H, D] input with a "
        "contiguous last dim"
    )
    assert rope_head_dim > 0 and rope_head_dim % 2 == 0, (
        f"_rdna2_fused_inverse_rope_gptj expects an even rope_head_dim, "
        f"got {rope_head_dim}"
    )
    assert cos_sin_cache.shape[-1] == rope_head_dim, (
        "_rdna2_fused_inverse_rope_gptj expects cos_sin_cache laid out as "
        f"[P, {rope_head_dim}] = cos | sin, got {tuple(cos_sin_cache.shape)}"
    )
    num_tokens, num_heads, head_dim = o.shape
    out = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=torch.float16, device=o.device
    )
    if num_tokens == 0:
        return out
    _rdna2_inverse_rope_gptj_kernel[(num_tokens, num_heads)](
        o,
        out,
        positions,
        cos_sin_cache,
        o.stride(0),
        o.stride(1),
        out.stride(0),
        out.stride(1),
        cos_sin_cache.stride(0),
        NOPE=head_dim - rope_head_dim,
        HALF=rope_head_dim // 2,
        BLOCK_NOPE=triton.next_power_of_2(head_dim - rope_head_dim),
        BLOCK_HALF=triton.next_power_of_2(rope_head_dim // 2),
    )
    return out


def _get_cached_wo_a_fp16(
    wo_a: torch.nn.Module,
    n_local_groups: int,
    o_lora_rank: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Dequantize wo_a to fp16 once and cache it on the module.

    wo_a weights are static, so the fp8 -> fp32 -> (* block scale) -> fp16
    dequant only needs to run once. Recomputing it every decode step shows up
    in the profile as the largest copy/mul kernels. SGLang / ATOM keep wo_a in
    bf16 and feed a plain bf16 GEMM; on gfx10x we mirror that with fp16.
    """
    cached = getattr(wo_a, "_dsv4_wo_a_fp16", None)
    if cached is not None:
        return cached
    if hasattr(wo_a, "weight_scale_inv"):
        wo_a_weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.float32
        )
        wo_a_scale = _expand_2d_block_scales(
            wo_a.weight_scale_inv.view(
                n_local_groups, -1, wo_a.weight_scale_inv.shape[-1]
            ),
            o_lora_rank,
            hidden_dim,
        )
        cached = (wo_a_weight * wo_a_scale).to(torch.float16)
    else:
        cached = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.float16
        )
    wo_a._dsv4_wo_a_fp16 = cached
    return cached


def rocm_rdna2_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """fp16 inverse-RoPE + WO_A bmm path used on gfx10x.

    Fuses the inverse GPT-J RoPE into one Triton kernel and caches the fp16
    wo_a weight so the per-step dequant disappears.
    """
    o_ref = _rdna2_fused_inverse_rope_gptj(
        o, positions, rotary_emb.cos_sin_cache, rope_head_dim
    )
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1)

    wo_a_weight = _get_cached_wo_a_fp16(
        wo_a, n_local_groups, o_lora_rank, o_ref.shape[-1]
    )

    return torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)
