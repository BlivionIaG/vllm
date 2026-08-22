# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the new native int8 FA-RDNA2 decode kernel.

Compares the v3 wiki "live contract" int8 decode kernel
``fa_rdna2_decode_paged_int8`` against the fp16 reference
``fa_rdna2_decode_paged`` on the same Q/K/V inputs (the fp16 reference
uses the same K_fp16/V_fp16 that were quantized to int8 for the int8
kernel). Asserts max abs error within atol=1e-2.

Test matrix:
    HEAD_DIM in {128, 256}
    seq_len in {512, 4096}
    H_q=4, H_kv=1 (GQA ratio 4:1)
    BLOCK_M=1 (decode-style: one query token per sequence)
    block_size=16 (matches int8 cache layout)
    kv_splits=1 (fast path)
"""
import os

import torch

os.environ.setdefault("VLLM_USE_RDNA2_FA", "1")


def _make_int8_cache(K_fp16, V_fp16, num_blocks, H_kv, head_size, block_size,
                     num_tokens):
    """Quantize fp16 K/V to int8 with per-(token, head) scales.

    Mirrors the layout vLLM produces for INT8_PER_TOKEN_HEAD:
        key_cache:   [num_blocks, H_kv, head_size, block_size, 1] int8
        k_scale:     [num_tokens, H_kv] fp32 (per-token-head scale)
    The scale for token t, head h is absmax(K[t, h, :]) / 127.
    """
    K_t = K_fp16.view(num_blocks, block_size, H_kv, head_size)
    V_t = V_fp16.view(num_blocks, block_size, H_kv, head_size)

    K_amax = K_t.abs().amax(dim=-1)  # [num_blocks, block_size, H_kv]
    V_amax = V_t.abs().amax(dim=-1)
    k_scale = (K_amax / 127.0).clamp(min=1e-12).to(torch.float32)
    v_scale = (V_amax / 127.0).clamp(min=1e-12).to(torch.float32)
    k_scale = k_scale.view(num_tokens, H_kv).contiguous()
    v_scale = v_scale.view(num_tokens, H_kv).contiguous()

    k_scale_b = k_scale.view(num_blocks, block_size, H_kv, 1)
    v_scale_b = v_scale.view(num_blocks, block_size, H_kv, 1)
    K_q = (K_t.float() / k_scale_b).round().clamp(-127, 127).to(torch.int8)
    V_q = (V_t.float() / v_scale_b).round().clamp(-127, 127).to(torch.int8)

    K_paged = K_q.permute(0, 2, 3, 1).unsqueeze(-1).contiguous()
    V_paged = V_q.permute(0, 2, 3, 1).unsqueeze(-1).contiguous()
    return K_paged, V_paged, k_scale, v_scale


def _build_paged_cache(K_fp16, V_fp16, num_blocks, H_kv, head_size,
                       block_size):
    """Build a 5D fp16 paged cache from contiguous K/V.

    Returns (K_paged, V_paged, block_table, seq_lens).
    block_table is [num_tokens, max_blocks=1] where each row maps token
    index -> physical block index (here, token t -> block t/block_size).
    """
    K_t = K_fp16.view(num_blocks, block_size, H_kv, head_size)
    V_t = V_fp16.view(num_blocks, block_size, H_kv, head_size)
    K_paged = K_t.permute(0, 2, 3, 1).unsqueeze(-1).contiguous()
    V_paged = V_t.permute(0, 2, 3, 1).unsqueeze(-1).contiguous()

    num_tokens = num_blocks * block_size
    # max_blocks must be large enough to cover any token's KV range.
    # With seq_lens[i] = num_tokens for all i, max_blocks = num_blocks.
    max_blocks = num_blocks
    # block_table[t, k] = k for all (t, k) — each token uses the same
    # physical blocks [0..num_blocks) in order (identity 1:1 mapping).
    block_table = (torch.arange(num_tokens * max_blocks, device=K_fp16.device)
                   .view(num_tokens, max_blocks)
                   % max_blocks).to(torch.int32).contiguous()
    seq_lens = torch.full((num_tokens,), num_tokens, dtype=torch.int32,
                          device=K_fp16.device)
    return K_paged, V_paged, block_table, seq_lens


def _run_int8_decode(Q, K_int8, V_int8, k_scale, v_scale, block_table,
                     seq_lens, block_size, kv_splits, sliding_window):
    from vllm.v1.attention.ops import fa_rdna2_backend as fa
    ext = fa._load_kernel()
    return ext.fa_rdna2_decode_paged_int8(
        Q, K_int8, V_int8, block_table, seq_lens,
        block_size, kv_splits, sliding_window, k_scale, v_scale)


def _run_fp16_decode(Q, K_fp16, V_fp16, block_table, seq_lens, block_size,
                     kv_splits, sliding_window):
    from vllm.v1.attention.ops import fa_rdna2_backend as fa
    ext = fa._load_kernel()
    return ext.fa_rdna2_decode_paged(
        Q, K_fp16, V_fp16, block_table, seq_lens,
        block_size, kv_splits, sliding_window)


def _run_one_case(head_size, seq_len, H_q, H_kv, block_size, kv_splits,
                  device="cuda", seed=42):
    torch.manual_seed(seed)
    num_tokens = seq_len
    num_blocks = (seq_len + block_size - 1) // block_size

    Q = torch.randn(num_tokens, H_q, head_size, dtype=torch.float16,
                    device=device)
    K_fp16 = torch.randn(num_tokens, H_kv, head_size, dtype=torch.float16,
                         device=device)
    V_fp16 = torch.randn(num_tokens, H_kv, head_size, dtype=torch.float16,
                         device=device)

    K_int8, V_int8, k_scale, v_scale = _make_int8_cache(
        K_fp16, V_fp16, num_blocks, H_kv, head_size, block_size, num_tokens)
    K_p16, V_p16, block_table, seq_lens = _build_paged_cache(
        K_fp16, V_fp16, num_blocks, H_kv, head_size, block_size)

    out_int8 = _run_int8_decode(
        Q, K_int8, V_int8, k_scale, v_scale, block_table, seq_lens,
        block_size, kv_splits, 0)
    out_fp16 = _run_fp16_decode(
        Q, K_p16, V_p16, block_table, seq_lens, block_size, kv_splits, 0)

    diff = (out_int8.float() - out_fp16.float()).abs()
    max_abs = diff.max().item()
    denom = out_fp16.float().abs().clamp(min=1e-6)
    max_rel = (diff / denom).max().item()
    return max_abs, max_rel, out_int8.shape, out_fp16.shape


def test_int8_decode_correctness():
    """All 8 cases (2 head_dims x 4 seq_lens) must pass at atol=1e-2."""
    if not torch.cuda.is_available():
        print("SKIP: torch.cuda.is_available() is False")
        return
    props = torch.cuda.get_device_properties(0)
    if "gfx103" not in getattr(props, "gcnArchName", ""):
        print(f"SKIP: device is {props.gcnArchName}, not gfx1030")
        return

    H_q = 4
    H_kv = 1
    block_size = 16
    kv_splits = 1
    for head_size in (128, 256):
        for seq_len in (512, 4096):
            max_abs, max_rel, shape_i, shape_f = _run_one_case(
                head_size=head_size, seq_len=seq_len,
                H_q=H_q, H_kv=H_kv, block_size=block_size,
                kv_splits=kv_splits)
            print(
                f"D={head_size} seq_len={seq_len}: "
                f"max_abs_err={max_abs:.4f} max_rel_err={max_rel:.4f} "
                f"int8.shape={tuple(shape_i)} fp16.shape={tuple(shape_f)}"
            )
            assert max_abs <= 1e-2, (
                f"D={head_size} seq_len={seq_len}: "
                f"max abs err {max_abs:.4f} > 1e-2"
            )


if __name__ == "__main__":
    test_int8_decode_correctness()
    print("test_int8_decode_correctness: PASS")
