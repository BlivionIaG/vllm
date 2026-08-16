# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8-FP8 dense linear kernel for AMD RDNA2 (gfx1030).

DeepSeek V4 Flash attention / shared experts: FP8 (E4M3) weights + FP8
(E4M3) activations. Per-tile FP8->fp16 conversion via inline bit-trick
(no LUT, no constant memory), then v_dot2_f32_f16 for the inner dot.
fp32 accumulator, packed CAS-64 atomic-add epilogue.

Activation dequant happens at LDS staging (once per element), making
the inner loop identical to the W8A16-FP8 shape. See
``gemm_w8a8_fp8_dense_rdna2.cu`` for the kernel.

gfx1030 lacks v_dot2_f32_bf16 (RDNA3+ feature), so this kernel rejects
bf16 activations at can_implement time. bf16 checkpoints must be cast
to fp16 before quantization.

The kernel is registered in _POSSIBLE_FP8_KERNELS[ROCM] -- but only
fires if the C++ op is registered (rebuild required). Without the
rebuild, falls through to the next kernel in the registry.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


class RDNA2W8A8FP8LinearKernel(FP8ScaledMMLinearKernel):
    SUPPORTED_WEIGHT_DTYPES = [scalar_types.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        # ROCm gates via on_gfx10x() in can_implement.
        return 60

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA2 W8A8 FP8 kernel is ROCm-only"

        from vllm.platforms.rocm import on_gfx10x

        if not on_gfx10x():
            return False, "RDNA2 W8A8 FP8 kernel requires gfx1030"

        if not (
            hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "gemm_w8a8_fp8_dense")
        ):
            return (
                False,
                "torch.ops._rocm_C.gemm_w8a8_fp8_dense missing -- rebuild "
                "C++ extension (gemm_w8a8_fp8_dense_rdna2.cu)",
            )

        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        supported, reason = cls.is_supported()
        if not supported:
            return False, reason

        # Act type: fp16 or bf16. gfx1030 lacks v_dot2_f32_bf16, so bf16
        # activations are cast to fp16 at LDS staging (As.to(fp16) in
        # apply_scaled_mm). The kernel itself only consumes fp16.
        if c.input_dtype not in (torch.float16, torch.bfloat16):
            return (
                False,
                f"RDNA2 W8A8 FP8 kernel supports fp16/bf16 activations on "
                f"gfx1030 (got {c.input_dtype})",
            )

        # Out dtype: fp16 or bf16. Kernel writes fp16 via atomic_add_pk4_f16
        # (packed CAS on fp16); apply_scaled_mm casts the output back to
        # out_dtype if needed.
        if c.out_dtype not in (torch.float16, torch.bfloat16):
            return (
                False,
                f"RDNA2 W8A8 FP8 kernel supports fp16/bf16 output on gfx1030 "
                f"(got {c.out_dtype})",
            )

        # Block-strategy weight quant: row = N-block, col = K-block.
        # We need a positive K-axis block (col > 0) since the kernel
        # uses col as group_size.
        ws = c.weight_quant_key.scale.group_shape
        if not (ws.row > 0 and ws.col > 0):
            return (
                False,
                "RDNA2 W8A8 FP8 kernel requires block-quant weight scales "
                f"with positive N-block and K-block (got group_shape={ws})",
            )

        # Activation quant: per-tensor OR per-token × per-block-K.
        as_gs = c.activation_quant_key.scale.group_shape
        if as_gs.is_per_tensor():
            pass
        elif as_gs.row == 1 and as_gs.col > 0:
            ws_gs = c.weight_quant_key.scale.group_shape
            if ws_gs.col != as_gs.col:
                return (
                    False,
                    f"RDNA2 W8A8 FP8 kernel requires activation K-block "
                    f"size == weight K-block size (got act_col={as_gs.col}, "
                    f"weight_col={ws_gs.col})",
                )
        else:
            return (
                False,
                f"RDNA2 W8A8 FP8 kernel only supports per-tensor or "
                f"per-block-K activation scales "
                f"(GroupShape(row={as_gs.row}, col={as_gs.col}))",
            )

        # Symmetric quant only (kernel assumes zero=null).
        if not c.weight_quant_key.symmetric:
            return (
                False,
                "RDNA2 W8A8 FP8 kernel only supports symmetric weight quant "
                f"(asymmetric={not c.weight_quant_key.symmetric})",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Canonicalize weight to (K, N) layout (kernel reads [K, N] uint8).
        if getattr(layer, "_rdna2_w8a8_fp8_prepared", False):
            return
        w_name, w_s_name = self.layer_param_names[0], self.layer_param_names[1]
        N_out, K = self.config.weight_shape

        w = getattr(layer, w_name)
        if w.shape != (K, N_out):
            replace_parameter(layer, w_name, w.t().contiguous())
            prepared = getattr(layer, w_name)
            prepared.input_dim = 0
            prepared.output_dim = 1

        if not hasattr(layer, w_s_name):
            # Fused-QKV (MergedColumnParallelLinear): no per-projection w_scale;
            # per-channel scale is passed via apply_scaled_mm.
            layer._rdna2_w8a8_fp8_per_channel = True
            layer._rdna2_w8a8_fp8_prepared = True
            return

        ws = getattr(layer, w_s_name)
        layer._rdna2_w8a8_fp8_per_channel = False
        if ws.dim() == 2:
            grp = self.config.weight_quant_key.scale.group_shape
            n_groups = N_out // grp.row
            k_groups = K // grp.col
            if ws.shape == (n_groups, k_groups):
                ws_prepared = ws.t().contiguous()
                ws_prepared = ws_prepared.repeat_interleave(grp.row, dim=1)
                replace_parameter(
                    layer, w_s_name, ws_prepared.to(torch.float16)
                )
        layer._rdna2_w8a8_fp8_prepared = True

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # W8A8 path: A is FP8 (E4M3), B is FP8 (E4M3), As is per-row fp16
        # or per-tensor fp16, Bs is fp16 per-group. Kernel does inline
        # FP8->FP16 dequant at LDS staging (once per element).
        assert A.dtype == torch.float8_e4m3fn, (
            f"RDNA2 W8A8 FP8 kernel: A must be fp8 e4m3fn (got {A.dtype})"
        )
        assert B.dtype == torch.float8_e4m3fn, (
            f"RDNA2 W8A8 FP8 kernel: B must be fp8 e4m3fn (got {B.dtype})"
        )

        # Kernel layout: a_q [M, K] uint8, a_scale [M, 1] or [1] fp16.
        # vLLM passes A as the dynamic-quantized activation; As is the
        # per-row (or per-tensor) fp16 scale.
        a_q = A.view(torch.uint8)
        if As.dtype != torch.float16:
            As = As.to(torch.float16)

        M, K = A.shape
        B_in, N = B.shape
        if B_in != K:
            B = B.t().contiguous()
            if Bs.dim() == 2 and Bs.shape[0] != K // 128:
                Bs = Bs.t().contiguous()
        K_b, N = B.shape
        assert K == K_b, f"K mismatch: A={K}, B={K_b}"

        b_bytes = B.view(torch.uint8)

        # Canonicalize b_scales. Two cases:
        #   per-channel (fused-QKV): Bs is 1D [N], group_size=0 sentinel.
        #   per-group:              Bs is 2D [K_groups, N], group_size=K/K_groups.
        per_channel = (Bs.dim() == 1)
        if per_channel:
            assert Bs.shape[0] == N, (
                f"per-channel Bs must be [N] (got shape={tuple(Bs.shape)})"
            )
            group_size = 0
        else:
            if Bs.dim() == 2 and Bs.shape[1] != N:
                block_n = N // Bs.shape[1]
                Bs = Bs.repeat_interleave(block_n, dim=1).contiguous()
            K_groups = Bs.shape[0]
            assert K % K_groups == 0, (
                f"K={K} not divisible by K_groups={K_groups}"
            )
            group_size = K // K_groups

        output_shape = [*output_shape[:-1], N]
        output = torch.zeros((M, N), dtype=torch.float16, device=A.device)

        ops.gemm_w8a8_fp8_dense(a_q, As, b_bytes, Bs, output, group_size)

        if bias is not None:
            output.add_(bias)

        if out_dtype != torch.float16:
            output = output.to(out_dtype)

        return output.view(*output_shape)
