#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA2 fused MoE W8A16 (INT8) HIP kernel.

Tests ``moe_w8a16_gemm_rdna2`` against the per-expert dense reference
(create INT8 weights, fp16 scale/zero, run fused MoE kernel, compare).

bf16 is not supported on gfx1030 (lacks v_dot2_f32_bf16 intrinsic — RDNA3+
feature). bf16 variants are skipped via marker.

Run `pytest tests/kernels/quantization/test_rdna2_w8a16_moe.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 MoE W8A16 kernel is ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.platforms.rocm import on_gfx10x  # noqa: E402

device = "cuda"

gfx1030_only = pytest.mark.skipif(
    not (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_w8a16_gemm_rdna2")
    ),
    reason="Requires gfx1030 with moe_w8a16_gemm_rdna2 op",
)

# Real K/N/top_k/group_size dims, E capped at 16 for memory.
MODEL_CONFIGS = [
    pytest.param(16, 2048, 768, 8, 32, id="Qwen3-30B-A3B"),
    pytest.param(16, 2048, 512, 8, 32, id="Qwen3.6-35B-A3B"),
]

NUM_TOKENS = [1, 4, 16, 64, 256]


def _make_int8_weights(E, K, N):
    """Create random int8 weights [E, K, N]."""
    return torch.randint(
        -128, 127, (E, K, N), dtype=torch.int8, device=device
    )


def _make_scales(E, groups, N, dtype):
    return torch.rand(E, groups, N, dtype=dtype, device=device) * 0.05 + 0.001


def _make_zeros(E, groups, N, dtype):
    return torch.zeros(E, groups, N, dtype=dtype, device=device)


def _symmetric_int8_ref(x_fp16, w_int8, scale, zero, group_size):
    """Reference: dequantize w_int8 -> fp16, then fp16 @ fp16 GEMM.

    Returns a tensor of shape ``[M, E, N]`` where each slice ``[m, e, :]``
    is ``x[m] @ w_dequant[e].T``. Caller slices along ``e`` to extract
    a single expert's contribution.
    """
    E, K, N = w_int8.shape
    K_groups = K // group_size
    w_fp16 = w_int8.to(torch.float32)
    s = scale.to(torch.float32).view(E, K_groups, 1, N)
    z = zero.to(torch.float32).view(E, K_groups, 1, N)
    w_dequant = ((w_fp16.view(E, K_groups, group_size, N) - z) * s).view(E, K, N)
    return torch.einsum('mk,ekn->men', x_fp16.float(), w_dequant)


@gfx1030_only
@pytest.mark.parametrize("E, K, N_inter, top_k, group_size", MODEL_CONFIGS)
@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("block_size_m", [1, 4])
def test_fused_moe_w8a16_w1_matches_dense(
    E, K, N_inter, top_k, group_size, M, block_size_m
):
    """w1 GEMM via fused kernel matches per-expert dense reference."""
    N_gate_up = N_inter * 2
    groups = K // group_size

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w13 = _make_int8_weights(E, K, N_gate_up)
    w13_s = _make_scales(E, groups, N_gate_up, torch.float16)
    w13_z = _make_zeros(E, groups, N_gate_up, torch.float16)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, E)

    fused_out = torch.zeros(M * top_k, N_gate_up, dtype=torch.float16, device=device)
    ops.moe_w8a16_gemm_rdna2(
        x,
        fused_out,
        w13,
        w13_s,
        w13_z,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        block_size_m,
        False,
        0,
    )

    ref = _symmetric_int8_ref(x, w13, w13_s, w13_z, group_size)
    # ref: [M, K] @ w [E, K, N] -> [M, E, (top_k, N)] then routed by topk_ids
    # Per-expert reference: for each (m, k), gather expert e = topk_ids[m, k]
    # and take ref_out[m, k, :] = w_fp16 matmul at position e
    # For sanity, just compare per-expert dense result vs fused.
    torch.testing.assert_close(
        fused_out.float().cpu().numpy(),
        ref.transpose(0, 1).contiguous().view(-1, N_gate_up).float().cpu().numpy(),
        atol=2.0,
        rtol=0.05,
    )


@gfx1030_only
@pytest.mark.parametrize("E, K, N_inter, top_k, group_size", MODEL_CONFIGS[:1])
@pytest.mark.parametrize("block_size_m", [1, 4])
def test_fused_moe_w8a16_per_expert_exact(
    E, K, N_inter, top_k, group_size, block_size_m
):
    """Each expert's output matches the dense dequant @ x reference."""
    N_gate_up = N_inter * 2
    groups = K // group_size
    M = 4

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w13 = _make_int8_weights(E, K, N_gate_up)
    w13_s = _make_scales(E, groups, N_gate_up, torch.float16)
    w13_z = _make_zeros(E, groups, N_gate_up, torch.float16)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, E)

    fused_out = torch.zeros(M * top_k, N_gate_up, dtype=torch.float16, device=device)
    ops.moe_w8a16_gemm_rdna2(
        x,
        fused_out,
        w13,
        w13_s,
        w13_z,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        block_size_m,
        False,
        0,
    )

    # Flat-order: token m, expert k -> index m*top_k + k
    for m in range(M):
        for k in range(top_k):
            e = topk_ids[m, k].item()
            flat = m * top_k + k
            # Reference: dequantize expert e's weights, GEMM with x[m]
            w_fp = _symmetric_int8_ref(
                x[m : m + 1], w13[e : e + 1], w13_s[e : e + 1],
                w13_z[e : e + 1], group_size,
            ).squeeze()  # [1, 1, N] -> [N]
            torch.testing.assert_close(
                fused_out[flat].float().cpu(),
                w_fp.float().cpu(),
                atol=2.0,
                rtol=0.05,
            )


@gfx1030_only
def test_w8a16_bf16_rejected():
    """Passing bf16 activations must fail at the kernel."""
    if not on_gfx10x():
        pytest.skip("gfx1030 only")
    N = 8
    K = 8
    E = 1
    G = 1
    M = 1

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.zeros(E, K, N, dtype=torch.int8, device=device)
    s = torch.ones(E, G, N, dtype=torch.bfloat16, device=device)
    z = torch.zeros(E, G, N, dtype=torch.bfloat16, device=device)
    out = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    si, ei, ntp = moe_align_block_size(
        torch.zeros(M, 1, dtype=torch.int32, device=device), 1, E)

    with pytest.raises(RuntimeError, match="fp16"):
        ops.moe_w8a16_gemm_rdna2(x, out, w, s, z, torch.empty(0, device=device),
                                si, ei, ntp, 1, 1, False, 0)
