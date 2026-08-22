# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for dense AWQ dispatch through RDNA2W4A16LinearKernel.

Exercises the full Python wiring — not just the raw HIP op — so the test
catches regressions in any of the three changes that landed alongside AWQ
support:

  1. SUPPORTED_QUANT_TYPES includes scalar_types.uint4
     (rdna2_w4a16.py type gate)
  2. transform_w_zp transposes the AWQ repack output [N//8, G] packed_dim=0
     into the kernel's expected [G, N//8] packed_dim=1 (qzeros indexing is
     b_qzeros + g * (size_n/8); see q_gemm_rdna2_common.cuh:106)
  3. use_v2_format=True routes zero_offset=0 in q_gemm_rdna2.cu:219 for
     AWQ literal zeros (vs zero_offset=1 for GPTQv1 stored-zero-1)

Test pattern is mirrored from tests/kernels/quantization/test_rdna2_w4a16.py
(the GPTQ test for the same kernel). That test uses raw nibbles in
[K, N] for the reference (no need to know about the post-shuffle layout)
and `rel_l2 < 5e-2` tolerance to accommodate the gptq_shuffle
row-permutation within each K-block of 8. We do the same here.

Test matrix:
    M in {1, 4, 8, 16, 64}
    K = N = 4096, group_size = 128
"""
import pytest
import torch

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import (
    RDNA2W4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

try:
    from vllm.platforms.rocm import on_gfx10x
except ImportError:  # pragma: no cover — defensive
    on_gfx10x = lambda: False  # noqa: E731

device = "cuda"

WEIGHT_TYPE = scalar_types.uint4  # AWQ: no zero-bias, kernel uses use_v2_format=True
PACK_FACTOR = 8  # 8 x 4-bit nibbles per int32

# Skip everything unless we are on the only architecture the kernel is built for.
pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_rocm()
        and on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna2")
    ),
    reason="requires gfx1030 with the _rocm_C.gptq_gemm_rdna2 op built in",
)


# ---------------------------------------------------------------------------
# Reference (raw nibbles; shuffle is hidden from this layer)
# ---------------------------------------------------------------------------


def _reference(
    x_mk: torch.Tensor,
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_ng: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """fp32 reference for the RDNA2 W4A16 op (AWQ flavor).

    q_int4_kn:  [K, N] int32 raw stored nibbles in [0, 15]
                (the ORIGINAL pre-pack, pre-shuffle values).
    scales_gn:  [K//G, N] per-group scales (act dtype).
    zeros_ng:   [N, K//G] int32 raw stored AWQ zero points in [0, 15].
                AWQ stores zeros in output-major order so this is
                the *pre-AWQ-repack* layout (zeros_gn[n, g] = zero for
                output col n, group g). After the AWQ repack + kernel
                transform_w_zp, the kernel sees this as [G, N//8].
    group_size: G.

    AWQ semantics: literal zero, so W_dequant = (q - z) * s, NO +1.
    """
    K, N = q_int4_kn.shape
    s_full = scales_gn.repeat_interleave(group_size, dim=0).to(torch.float32)
    # AWQ stores zeros in output-major order; expand to [K, N] by repeating
    # along the K dim of the (N, K//G) layout.
    z_full = zeros_ng.repeat_interleave(group_size, dim=1).to(torch.float32).T
    w_fp = (q_int4_kn.to(torch.float32) - z_full) * s_full
    out = x_mk.to(torch.float32) @ w_fp
    return out.to(x_mk.dtype)


# ---------------------------------------------------------------------------
# Layer construction (post-AWQ-repack layout for qzeros)
# ---------------------------------------------------------------------------


def _build_layer(
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_ng: torch.Tensor,
) -> torch.nn.Module:
    """Build a layer carrying AWQ-repacked parameters, as the loader would.

    The qweight is the standard (K//8, N) packed_dim=0 layout (matches the
    AWQ repack output). The qzeros is the AWQ repack's [N//8, G]
    packed_dim=0 layout — i.e. *transposed* from the kernel's expected
    [G, N//8] packed_dim=1. RDNA2W4A16LinearKernel.transform_w_zp is
    responsible for transposing it back.
    """
    no_loader = lambda *args, **kwargs: None  # noqa: E731

    # qweight: pack (K, N) raw nibbles into (K//8, N) packed_dim=0.
    qweight = pack_quantized_values_into_int32(q_int4_kn, WEIGHT_TYPE, packed_dim=0)

    # qzeros: pack the AWQ-format (N, G) raw nibbles into (N//8, G)
    # packed_dim=0 (i.e. we pack along the N dim, then the G axis is the
    # one NOT packed — this is the AWQ repack output shape).
    qzeros = pack_quantized_values_into_int32(zeros_ng, WEIGHT_TYPE, packed_dim=0)

    class _Layer(torch.nn.Module):
        pass

    layer = _Layer()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(
            data=qweight,
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=PACK_FACTOR,
        ),
    )
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(
            data=scales_gn.to(torch.float16),
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
        ),
    )
    layer.register_parameter(
        "qzeros",
        PackedvLLMParameter(
            data=qzeros,
            weight_loader=no_loader,
            output_dim=0,    # AWQ repack: G axis is the "input" of the [N//8, G] layout
            input_dim=1,     # so input_dim=1 (the G axis), output_dim=0 (the N//8 axis)
            packed_dim=0,
            packed_factor=PACK_FACTOR,
        ),
    )
    return layer


def _run_kernel(
    x_mk: torch.Tensor,
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_ng: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    K, N = q_int4_kn.shape
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=True,
        has_g_idx=False,
    )
    ok, reason = RDNA2W4A16LinearKernel.can_implement(config)
    assert ok, f"can_implement rejected uint4: {reason}"

    layer = _build_layer(q_int4_kn, scales_gn, zeros_ng)
    kernel = RDNA2W4A16LinearKernel(
        config,
        w_q_param_name="qweight",
        w_s_param_name="scales",
        w_zp_param_name="qzeros",
    )
    kernel.process_weights_after_loading(layer)
    return kernel.apply_weights(layer, x_mk)


# Same tolerance the existing GPTQ test uses for the same kernel.
# gptq_shuffle permutes rows within each K-block of 8; the rel_l2 metric
# averages over the whole matmul output so the noise is diluted.
_REL_L2_TOL = 5e-2


def _assert_close(out: torch.Tensor, ref: torch.Tensor):
    rel_l2 = (out.to(torch.float32) - ref.to(torch.float32)).norm() / ref.to(
        torch.float32
    ).norm()
    assert rel_l2 < _REL_L2_TOL, (
        f"relative L2 error {rel_l2:.4f} exceeds {_REL_L2_TOL}"
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M", [1, 4, 8, 16, 64])
def test_awq_w4a16_dense_correctness(M: int, dist_init):
    """Dense AWQ through RDNA2 dispatch matches the dequant+matmul ref."""
    set_random_seed(0)
    K, N, group_size = 4096, 4096, 128
    G = K // group_size  # 32

    # Random inputs: q_int4_kn in [0, 15] (raw nibbles), scales ~ U(0.01, 0.06),
    # zeros_ng in [0, 15] (AWQ format = output-major, so shape [N, G]).
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((G, N), device=device, dtype=torch.float32) + 0.01
    ).to(torch.float16)
    zeros_ng = torch.randint(0, 16, (N, G), device=device, dtype=torch.int32)

    out = _run_kernel(x_mk, q_int4_kn, scales_gn, zeros_ng, group_size)
    ref = _reference(x_mk, q_int4_kn, scales_gn, zeros_ng, group_size)

    assert out.shape == ref.shape, (
        f"shape mismatch: kernel={tuple(out.shape)} ref={tuple(ref.shape)}"
    )
    _assert_close(out, ref)
