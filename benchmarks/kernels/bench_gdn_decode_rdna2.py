#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbench for the RDNA2 GDN packed decode kernel (gfx1030).

1) Chained-recurrence correctness: 32 sequential decode steps with fresh
   random inputs each step, comparing HIP output and fp32 state against
   the Triton reference after every step. Recurrence errors compound, so
   this is a stronger check than the single-step pytest parity test.
2) Timing: HIP vs Triton packed decode at decode-realistic batch sizes.

Run on .176: python benchmarks/kernels/bench_gdn_decode_rdna2.py
"""

import time

import torch

import vllm._rocm_C  # noqa: F401  (registers torch.ops._rocm_C)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

device = "cuda"
K = 128
V = 128
H = 4
HV = 12
SCALE = K**-0.5
CHAIN_STEPS = 32


def _inputs(B, num_blocks, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn(B, qkv_dim, device=device, dtype=torch.float16,
                            generator=gen)
    a = torch.randn(B, HV, device=device, dtype=torch.float16, generator=gen)
    b = torch.randn(B, HV, device=device, dtype=torch.float16, generator=gen)
    A_log = torch.randn(HV, device=device, dtype=torch.float32,
                        generator=gen)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float16,
                          generator=gen)
    state0 = torch.randn(num_blocks, HV, V, K, device=device,
                         dtype=torch.float32, generator=gen)
    idx = torch.randperm(num_blocks, device=device, generator=gen)
    idx = idx[:B].to(torch.int32)
    return mixed_qkv, a, b, A_log, dt_bias, state0, idx


def _ref(mixed_qkv, a, b, A_log, dt_bias, state, out, idx):
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        scale=SCALE, initial_state=state, out=out, ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True)


def _hip(mixed_qkv, a, b, A_log, dt_bias, state, out, idx):
    torch.ops._rocm_C.gdn_decode_rdna2(mixed_qkv, a, b, A_log, dt_bias, out,
                                       state, idx, SCALE, True)


def chain_correctness():
    print("== chained recurrence correctness "
          f"({CHAIN_STEPS} sequential decode steps) ==")
    ok = True
    for B in (1, 8, 32):
        _, _, _, A_log, dt_bias, state0, idx = _inputs(B, B, seed=7)
        state_ref = state0.clone()
        state_hip = state0.clone()
        out_ref = torch.zeros(B, 1, HV, V, device=device,
                              dtype=torch.float16)
        out_hip = torch.zeros_like(out_ref)
        max_out_err = 0.0
        for step in range(CHAIN_STEPS):
            mixed_qkv, a, b, _, _, _, _ = _inputs(B, B, seed=1000 + step)
            _ref(mixed_qkv, a, b, A_log, dt_bias, state_ref, out_ref, idx)
            _hip(mixed_qkv, a, b, A_log, dt_bias, state_hip, out_hip, idx)
            err = (out_hip.float() - out_ref.float()).abs().max().item()
            max_out_err = max(max_out_err, err)
        state_err = (state_hip - state_ref).abs().max().item()
        state_rel = state_err / max(state_ref.abs().max().item(), 1e-6)
        status = "OK" if state_rel < 1e-2 else "FAIL"
        ok = ok and status == "OK"
        print(f"  B={B:3d}: max out err over {CHAIN_STEPS} steps = "
              f"{max_out_err:.3e}, final state abs err = {state_err:.3e} "
              f"(rel {state_rel:.2e}) [{status}]")
    return ok


def _time(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def timing():
    print("== timing (us/call, 300 iters after 30 warmup) ==")
    for B in (1, 8, 32):
        mixed_qkv, a, b, A_log, dt_bias, state0, idx = _inputs(B, B, seed=3)
        out = torch.zeros(B, 1, HV, V, device=device, dtype=torch.float16)
        t_tri = _time(lambda: _ref(mixed_qkv, a, b, A_log, dt_bias,
                                   state0, out, idx))
        t_hip = _time(lambda: _hip(mixed_qkv, a, b, A_log, dt_bias,
                                   state0, out, idx))
        print(f"  B={B:3d}: HIP {t_hip:7.1f} us | Triton {t_tri:7.1f} us | "
              f"HIP/Triton {t_hip / t_tri:.2f}x")


if __name__ == "__main__":
    if not chain_correctness():
        raise SystemExit("chained-recurrence correctness FAILED")
    timing()
