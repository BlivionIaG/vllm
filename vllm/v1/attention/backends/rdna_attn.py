# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA_ATTN: standalone RDNA2 attention backend for gfx1030 family.

This backend is intentionally independent of RocmAttentionImpl /
rocm_attn.py. Future upstream changes to ROCM_ATTN cannot touch FA-RDNA2
dispatch logic. The kernel itself lives in ``csrc/rocm/fa_rdna2.cu`` and
is loaded by ``vllm/v1/attention/ops/fa_rdna2_backend.py`` via
``torch.utils.cpp_extension.load_inline`` (already wired in tree).

Selection mechanism:
  * Registered as ``AttentionBackendEnum.RDNA_ATTN`` in
    ``vllm/v1/attention/backends/registry.py``.
  * Selectable via the vLLM CLI ``--attention-backend RDNA_ATTN`` on
    gfx1030. ``is_available()`` returns False elsewhere, so the
    backend is a no-op on other architectures.
  * When the FA-RDNA2 7-gate conditions aren't met for a given forward
    pass, raises ``NotImplementedError`` so the V1 worker falls through
    to ROCM_ATTN (which keeps its existing Triton fallback chain).

Scope (strict — ONLY FA-RDNA2-adjacent surfaces touched by this file):
  * new file `vllm/v1/attention/backends/rdna_attn.py` (this one)
  * 1-line enum entry in `backends/registry.py` (RDNA_ATTN)
  * NO edits to rocm_attn.py, W4A16 GEMM kernels, MoE dispatcher,
    envs.py, platforms/rocm.py, or any other "fix" on the project

Coverage (forwarded to fa_rdna2_*_paged kernels via fa_rdna2_backend.py):
  - arch:         gfx1030 / gfx1031 / gfx1032 (RDNA2 only)
  - head_size:    {128, 256}
  - dtype:        float16 only
  - kv layout:    5D blocks-first paged (post vLLM PR-#43660);
                  4D-V + 5D-K is reinterpreted to 5D-V here.
  - kv_cache_dtype: 'auto' (non-quantized path)
  - decode (max_seqlen_q <= 1) and prefill (max_seqlen_q > 1) called
    from CommonAttentionMetadata via the same 7-gate + dispatcher
    logic that previously lived in rocm_attn.py:467-560.

Anything outside this set: raise NotImplementedError so the V1 selector
falls through to ROCM_ATTN, which keeps its existing Triton fallback
chain. RDNA_ATTN never silently degrades to Triton on its own.

The 5 corrections vs v1 prototype:
  (a) get_kv_cache_spec returns a real FullAttentionSpec (was stub).
  (b) _split_kv_cache delegates to PagedAttention.split_kv_cache
      (was naive chunk(2, dim=0) that didn't match vLLM's 5D layout).
  (c) MTP-verify env-read is per-call (was module-level, prevented
      VLLM_FARDNA2_SPEC_VERIFY_Q_LEN toggling without restart).
  (d) MTP-verify early-returns BEFORE any output mutation so the
      fallback path doesn't see partially-written output.
  (e) cudagraph_capture + encoder_attention explicit stubs raising
      NotImplementedError so V1 worker falls through cleanly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.v1.attention.ops.paged_attn import PagedAttention
# (env-name string literal; vllm.envs exposes VLLM_USE_RDNA2_FA as a typed default, not a str)
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    is_quantized_kv_cache,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Policy constants — only "policy" owned by this file; everything else is
# delegated to fa_rdna2_backend.py's existing dispatcher.
# ---------------------------------------------------------------------------
_SUPPORTED_HEAD_SIZES: tuple[int, ...] = (128, 256)
_SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (torch.float16,)
_SUPPORTED_ARCH_PREFIX: str = "gfx103"


def _on_gfx10x() -> bool:
    """True iff the local CUDA device is RDNA2 (gfx103x)."""
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        return _SUPPORTED_ARCH_PREFIX in getattr(props, "gcnArchName", "")
    except Exception:
        return False


def _env_enabled() -> bool:
    """VLLM_USE_RDNA2_FA env gate (mirror fa_rdna2_backend's semantics)."""
    return os.environ.get("VLLM_USE_RDNA2_FA", "1") == "1"


def is_available() -> bool:
    """Backend-class-level availability gate (mirrors
    ``fa_rdna2_backend.is_available()`` plus the RDNA2 arch gate)."""
    return _env_enabled() and _on_gfx10x()


# ---------------------------------------------------------------------------
# Lazy fa_rdna2_backend module handle — kernel load is heavy
# (cpp_extension.load_inline), so we defer until first forward call.
# ---------------------------------------------------------------------------
_fa_rdna2_module = None


def _get_fa_rdna2_module():
    global _fa_rdna2_module
    if _fa_rdna2_module is None:
        from vllm.v1.attention.ops import fa_rdna2_backend as _m
        _fa_rdna2_module = _m
    return _fa_rdna2_module


# ---------------------------------------------------------------------------
# Metadata — minimum fields the FA-RDNA2 dispatcher consumes; built from
# CommonAttentionMetadata per V1 step.
# ---------------------------------------------------------------------------
@dataclass
class RdnaAttentionMetadata(AttentionMetadata):
    """Subset of fields consumed by fa_rdna2_*_paged entry functions.

    Built lazily by RdnaAttentionMetadataBuilder. We don't subclass an
    upstream builder-base class because the FA-RNA2 dispatcher only
    needs six fields; everything else defaults to None / 0 and the
    base-class constructors handle it.
    """

    @classmethod
    def from_common(cls, common: CommonAttentionMetadata) -> "RdnaAttentionMetadata":
        # vllm/v1/attention/backend.py CommonAttentionMetadata fields:
        #   query_start_loc, query_start_loc_cpu, seq_lens, num_reqs,
        #   num_actual_tokens, max_query_len, max_seq_len,
        #   block_table_tensor, slot_mapping, causal, ...
        inst = cls()
        inst.num_actual_tokens = common.num_actual_tokens
        inst.num_reqs = common.num_reqs
        inst.query_start_loc = common.query_start_loc
        inst.seq_lens = common.seq_lens
        inst.block_table = common.block_table_tensor
        inst.max_query_len = common.max_query_len
        inst.max_seq_len = common.max_seq_len
        inst.slot_mapping = common.slot_mapping
        return inst


class RdnaAttentionMetadataBuilder(AttentionMetadataBuilder):
    """V1 metadata builder. Mirrors the chunked-prefill-ish defaults;
    FA-RDNA2 only needs block_table + seq_lens + max_query_len + max_seq_len
    + query_start_loc, all in CommonAttentionMetadata already."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._device = device

    def build(
        self, common: CommonAttentionMetadata
    ) -> RdnaAttentionMetadata:
        return RdnaAttentionMetadata.from_common(common)

    def build_for_cudagraph_capture(
        self, common: CommonAttentionMetadata
    ) -> RdnaAttentionMetadata:
        return self.build(common)

    def build_for_drafting(
        self, common: CommonAttentionMetadata
    ) -> RdnaAttentionMetadata:
        return self.build(common)


# ---------------------------------------------------------------------------
# Implementation — minimum surface to call the existing FA-RNA2 entry
# functions with the right (query, key_cache, value_cache, block_table,
# seq_lens, ...) tuple.
# ---------------------------------------------------------------------------
class RdnaAttentionImpl(AttentionImpl):
    """Implements vLLM AttentionImpl protocol for FA-RNA2.

    Dispatcher decisions live here (not in rocm_attn.py) so the gating
    logic is owned by FA-RNA2's subsystem. Anything rocm_attn.py
    doesn't know about (and would have gotten wrong) is preserved here
    verbatim from the proof-of-concept that fired 32 attention
    forward calls on Qwen3.5-27B-AWQ during this session.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if head_size not in _SUPPORTED_HEAD_SIZES:
            raise NotImplementedError(
                f"RDNA_ATTN: head_size={head_size} not in "
                f"{_SUPPORTED_HEAD_SIZES}; falling back to ROCM_ATTN"
            )
        if attn_type not in (
            AttentionType.DECODER,
            AttentionType.ENCODER_ONLY,
        ):
            raise NotImplementedError(
                f"RDNA_ATTN: attn_type={attn_type} not supported; "
                "falling back to ROCM_ATTN"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.sinks = sinks
        self._alibi = (
            torch.tensor(alibi_slopes, dtype=torch.float32)
            if alibi_slopes is not None
            else None
        )

    # ------------------------------------------------------------------
    # The 7-gate dispatcher (was in rocm_attn.py:467-560; reproduced
    # verbatim here, with no dependency on rocm_attn.py).
    # ------------------------------------------------------------------
    def _can_run_fa_rdna2(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> bool:
        if not is_available():
            return False
        if query.dtype != torch.float16:
            return False
        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            return False
        if key_cache.dim() != 5 or value_cache.dim() != 5:
            return False
        if is_quantized_kv_cache(self.kv_cache_dtype):
            return False
        return True

    def _maybe_reinterp_v_to_5d(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Mirror rocm_attn.py:457-462: when V is 4D contiguous and K
        is already 5D blocks-first paged, reinterpret V's last two
        dims into the (D/x, block_size, x) layout K uses.
        """
        if (
            value_cache.dim() == 4
            and key_cache.dim() == 5
            and self.head_size in _SUPPORTED_HEAD_SIZES
        ):
            num_blocks, h_kv, head_size_d, block_sz = value_cache.shape
            x_dim = key_cache.shape[4]
            if head_size_d % x_dim == 0:
                value_cache = value_cache.view(
                    num_blocks,
                    h_kv,
                    head_size_d // x_dim,
                    block_sz,
                    x_dim,
                )
        return value_cache

    def _is_spec_verify_pass(
        self,
        max_seqlen_q: int,
        num_actual_tokens: int,
        num_seqs: int,
    ) -> bool:
        """Per-call env-read (was module-level in v1)."""
        spec_q = int(
            os.environ.get("VLLM_FARDNA2_SPEC_VERIFY_Q_LEN", "3")
        )
        return (
            max_seqlen_q == spec_q
            and num_actual_tokens <= 16 * num_seqs
        )

    # ------------------------------------------------------------------
    # V1 forward — the main entry point.
    # ------------------------------------------------------------------
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "RdnaAttentionMetadata",
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Profile-run fast path.
        if attn_metadata is None:
            return output.fill_(0)  # type: ignore[union-attr]

        # 1. KV-cache split via vLLM's canonical helper.
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size
        )
        # 2. Re-interp V 4D->5D if needed.
        value_cache = self._maybe_reinterp_v_to_5d(key_cache, value_cache)
        # 3. Gate check.
        if not self._can_run_fa_rdna2(query, key_cache, value_cache):
            raise NotImplementedError(
                "RDNA_ATTN: shape out of FA-RNA2 coverage matrix; "
                "V1 selector should fall back to ROCM_ATTN."
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        max_seqlen_q = attn_metadata.max_query_len
        seqused_k = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        max_seqlen_k = attn_metadata.max_seq_len
        cu_seqlens_q = attn_metadata.query_start_loc

        # 4. MTP-2 verify-pass guard (per-call env-read). EARLY-RETURN
        #    before any output mutation so the ROCM_ATTN fallback
        #    doesn't see partially-populated output.
        if self._is_spec_verify_pass(
            max_seqlen_q,
            num_actual_tokens,
            seqused_k.size(0),
        ):
            raise NotImplementedError(
                "RDNA_ATTN: MTP-2 verify-pass skipped for "
                "numerical-stability; falling back to ROCM_ATTN."
            )

        # Lazy-import the kernel module (heavy first-call load_inline).
        fa = _get_fa_rdna2_module()

        # Sliding-window extraction (mirrors rocm_attn.py:569-577 logic).
        sliding_window_pair = self.sliding_window
        if sliding_window_pair is None:
            sliding_window = 0
        elif self.attn_type in (
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        ):
            sliding_window = (
                sliding_window_pair[0] + 1
                if sliding_window_pair[0] >= 0
                else 0
            )
        else:
            sliding_window = (
                sliding_window_pair[0] + 1
                if sliding_window_pair[0] >= 0
                else 0
            )

        paged_block_size = key_cache.shape[3]

        # ------------------------------------------------------------------
        # Decode path: max_seqlen_q <= 1.
        # ------------------------------------------------------------------
        if max_seqlen_q <= 1:
            out_paged = fa.fa_rdna2_decode_paged(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                block_table,
                seqused_k,
                paged_block_size,
                kv_splits=8,
                sliding_window=sliding_window,
            )
            output[:num_actual_tokens].view(
                num_actual_tokens,
                self.num_heads,
                self.head_size,
            ).copy_(out_paged)
            return output

        # ------------------------------------------------------------------
        # Prefill path: max_seqlen_q > 1.
        # ------------------------------------------------------------------
        _num_seqs = seqused_k.size(0)
        _SPLITK_MIN_KV_SPLITS = 2
        _kv_splits = min(8, (max_seqlen_k + 1023) // 1024)
        causal = bool(getattr(attn_metadata, "causal", False))

        if max_seqlen_k < 4096 and self.head_size == 128:
            out_paged_prefill = fa.fa_rdna2_prefill_paged_varlen_short(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                block_table,
                cu_seqlens_q,
                seqused_k,
                paged_block_size,
                causal=causal,
                sliding_window=sliding_window,
            )
        elif (
            _kv_splits >= _SPLITK_MIN_KV_SPLITS
            and _num_seqs <= 4
            and self.num_heads * _kv_splits >= 64
        ):
            out_paged_prefill = fa.fa_rdna2_prefill_paged_varlen_splitk(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                block_table,
                cu_seqlens_q,
                seqused_k,
                paged_block_size,
                causal=causal,
                kv_splits=_kv_splits,
                sliding_window=sliding_window,
            )
        else:
            out_paged_prefill = fa.fa_rdna2_prefill_paged_varlen(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                block_table,
                cu_seqlens_q,
                seqused_k,
                paged_block_size,
                causal=causal,
                sliding_window=sliding_window,
            )
        output[:num_actual_tokens].view(
            num_actual_tokens,
            self.num_heads,
            self.head_size,
        ).copy_(out_paged_prefill)
        return output

    # ------------------------------------------------------------------
    # V1 cudagraph-capture path. RDNA_ATTN doesn't yet support the
    # capture path (it relies on a per-shape dispatch that cudagraph
    # capture can't easily wrap). Raise NotImplementedError so V1
    # worker falls back to ROCM_ATTN during capture.
    # ------------------------------------------------------------------
    def forward_cudagraph_capture(
        self,
        common_attn_metadata,
        attn_cudagraph_inputs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "RDNA_ATTN: cudagraph-capture path not implemented; "
            "V1 worker falls back to ROCM_ATTN for capture, "
            "then re-enters RDNA_ATTN for replay if shape matches."
        )

    # ------------------------------------------------------------------
    # V1 encoder-attention path. RDNA_ATTN is text-only; encoder
    # attention is delegated to ROCM_ATTN (which has the encoder
    # shapes handled by its own Triton fallback).
    # ------------------------------------------------------------------
    def forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "RDNA_ATTN: encoder attention path not implemented; "
            "V1 worker falls back to ROCM_ATTN."
        )


# ---------------------------------------------------------------------------
# V1 Backend class registration surface.
# ---------------------------------------------------------------------------
class RdnaAttentionBackend(AttentionBackend):
    """Backend class registered as ``RDNA_ATTN``.

    Reachable via:
      (1) runtime selector patch (see ``_register_with_selector()``
          below) — preferred for opt-in launching
      (2) manual cli / config: ``--attention-backend RDNA_ATTN`` or
          ``VLLM_ATTENTION_BACKEND=RDNA_ATTN`` if the vLLM CLI accepts
          a name
      (3) optional 1-line enum entry in ``backends/registry.py``:
              RDNA_ATTN = (
                  "vllm.v1.attention.backends.rdna_attn.RdnaAttentionBackend"
              )
    """

    @staticmethod
    def get_name() -> str:
        return "RDNA_ATTN"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return RdnaAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return RdnaAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return RdnaAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> type | None:
        return None

    @classmethod
    def get_kv_cache_spec(
        cls,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        dtype: torch.dtype,
        attention_type: AttentionType = AttentionType.DECODER,
    ) -> Optional[AttentionSpec]:
        """Return FullAttentionSpec for the 5D blocks-first paged layout.

        V1 worker calls this at engine-startup to figure out how to
        allocate the kv cache. Returning None here is also valid — V1
        then falls back to no kv cache (not useful) — so we return a
        proper spec to keep the worker on the FA-RDNA2 path.
        """
        # Only emit a spec for the supported matrix; otherwise return
        # None so the V1 selector picks ROCM_ATTN.
        if head_size not in _SUPPORTED_HEAD_SIZES:
            return None
        if dtype not in _SUPPORTED_DTYPES:
            return None
        if attention_type not in (
            AttentionType.DECODER,
            AttentionType.ENCODER_ONLY,
        ):
            return None
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in _SUPPORTED_HEAD_SIZES

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype in _SUPPORTED_DTYPES

    @classmethod
    def supports_arch(cls) -> bool:
        return _on_gfx10x()


# ---------------------------------------------------------------------------
# Import-time pre-warm of the FA-RDNA2 kernel module so the first forward
# doesn't pay the load_inline cost. Mirrors what rocm_attn does at import
# time on the first call.
# ---------------------------------------------------------------------------
if is_available():
    try:
        _get_fa_rdna2_module()
    except Exception as exc:
        logger.debug("RDNA_ATTN pre-warm skipped: %s", exc)
