# Handoff: Qwen3.8-27B-AWQ fa_rdna2 Debugging — RESOLVED

**Date**: 2026-08-29 (resolved same day, second session)
**Author**: Sisyphus
**Branch**: `rdna2_extras` (local: `vllm_humanwork_next`, remote: `/home/chenco_adm/vllm_humanwork`)
**Goal**: Fully working Qwen3.8-27B-AWQ + fa_rdna2 hip + prefix caching + chunked prefill

---

## 0. Resolution Summary (2026-08-29)

T3 (fa_rdna2) now produces **coherent output end-to-end** on
Qwen3.8-27B-AWQ-INT4: short prompts ("Paris"), 1k and 5k needle probes
("ZEPHYR"/"QUASAR"), deterministic (temp=0 identical), dispatch proven via
`[RDNASHAPES]` log. T2 (fa_rdna2 off) regression: unaffected.

The deterministic garbage was **four stacked bugs**, not one:

| # | Bug | Where | Fix commit |
|---|-----|-------|-----------|
| A | Kernels read V with K's **packed** strides, but `reshape_and_cache` writes V **unpacked** (slot-innermost) → every V read permuted | `fa_rdna2.cu` (9 kernels used `stride_kc*` for V) | `e538b4f9b7` |
| B | Splitk kernels wrote partials for padding rows (`br ≥ br_size`) and reduce kernels processed padding tokens → OOB when `N % BR_PREFILL != 0` (the 5000-token crash; 5000%16=8) | `fa_rdna2.cu` splitk×4 + reduce×2 | `e538b4f9b7` |
| C | `RdnaAttentionMetadata.from_common` dropped `causal` → `getattr(..., False)` → **all prefills ran non-causal** | `rdna_attn.py` | `322c2acffd` |
| D | `do_kv_cache_update` always used the **dense** native writer; hybrid caches are stride-padded/interleaved `[nb, 2, ...]` (K block-stride 1605632 ≠ dense 802816) → K/V landed at wrong physical blocks | `rdna_attn.py` (missing `has_native_kv_cache_layout` branch) | `322c2acffd` |

**Why the existing tests missed it**: `test_fa_rdna2_shape_sweep.py` fills V
directly in packed layout (never goes through the production writer) and uses
a contiguous `[2, nb, ...]` cache where dense == view strides. The new
`tests/kernels/attention/test_fa_rdna2_writer_layout.py` goes through the
real writer and covers both layouts — 19/19 pass on .176.

**V-stride detail**: the correct 5D V view of the unpacked `[nb,h,D,bs]`
buffer is `view(nb, h, D/x, x, bs).permute(0, 1, 2, 4, 3)` → strides
`(…, x*bs, 1, bs)`. Kernel gets separate `stride_vc0..4` from
`value_cache.stride()`. K strides unchanged (packed).

**Key debug artifact**: `/tmp/rdna_attn_dump.pt` on .176 (first prefill
call's full tensors) — `VLLM_DBG_RDNA_DUMP=1` reproduces it.

### Debug env vars (committed, env-gated)
- `VLLM_DBG_RDNA_SHAPES=1` — one-time per-layer log of K/V/Q shapes, strides, causal, seq_lens
- `VLLM_DBG_RDNA_DUMP=1` — one-time per-layer tensor dump to `/tmp/rdna_attn_dump.pt` (107MB per layer, D2H stall — debug only)

### Remaining known issues (not blocking T3)
1. ~~Chunked prefill / prefix-cache q_local offset~~ — FIXED in
   `2bb0b7d0b5` (causal mask now uses the absolute query position
   `(seq_len - seq_query_len) + q_local` in all 7 prefill kernels).
   Validated: kernel-level chunked-offset cases + server probe with
   `--enable-prefix-caching --max-num-batched-tokens 2048` (5k prompt in
   3 chunks, prefix hit pass 2, needles found, T2 control matches).
2. **int8 KV cache + hybrid layout**: `do_kv_cache_update`'s int8 branch
   (`reshape_and_cache_int8_rdna2`) has no stride-aware variant. fp16 path
   unaffected.
3. **fa_rdna2 performance re-bench**: the kernels were never benchmarked in a
   working state. Re-run the W4A16×FA matrix from
   `.omo/plans/rdna2-full-matrix-bench-2026-07-19.md`.
4. `pkill -f "entrypoints.cli.main"` self-matches the ssh command line —
   use `pkill -f "entrypoints[.]cli"`.
5. Prefix caching on hybrid models runs in Mamba cache 'align' mode
   (experimental upstream warning) — works in the probe, watch for drift.

---

## 1. Current State — What's Working vs What's Broken (SUPERSEDED)

| Config | Status | Evidence |
|--------|--------|----------|
| **T1**: Triton W4A16 + Triton FA + GDN Triton + eager | ✅ **WORKS** | Coherent output, deterministic (6/7 anchors, 340c identical) |
| **T2**: + RDNA2W4A16LinearKernel (V_DOT2 AWQ) | ✅ **WORKS** | Same as T1, kernel selected correctly |
| **T3**: + fa_rdna2 (RDNA_ATTN, `VLLM_USE_RDNA2_FA=1`) | ❌ **BROKEN** | Deterministic garbage: "There is no meaningfully relevant content in the query" |
| **T3 + splitk fix** (07ebb0d6b) | ❌ **STILL BROKEN** | Crash at 5000t fixed, but short-prompt garbage remains |

**Bottom line**: The ONLY difference between working (T2) and broken (T3) is `VLLM_USE_RDNA2_FA=0` vs `=1`. The fa_rdna2 attention kernels produce wrong output for this model.

---

## 2. What Was Fixed This Session

### 2.1 fa_rdna2 splitk partial-buffer indexing overflow (commit 07ebb0d6b)

**Root cause**: The splitk prefill kernels (128/256 + int8) and their reduce kernels wrote partial O/M/L buffers with the legacy `[N, H_q, BR_PREFILL, kv_splits, D]` indexing (per-head multiplied by BR_PREFILL), but the host allocates `O_partial` as `[N, H_q, kv_splits, D]` (per-token rows). With BR_PREFILL=16 the mismatch silently corrupted output at short context and produced an out-of-bounds page fault (`fa_prefill_paged_varlen_splitk_kernel_256`) at long context (=5000 tok on Qwen3.8-27B-AWQ hybrid).

**Fix**: Changed all splitk kernels + both reduce kernels to index partial buffers per query token:
```
slot = (q_start_global + br) * H_q * kv_splits + h_q * kv_splits + split
```
matching the host `[N, H_q, kv_splits, D]` layout.

**Result**: Crash at 5000t fixed. Server survives long-prompt. But short-prompt garbage remains — different bug.

### 2.2 exl3 work (all committed in rdna2_extras)

| Commit | Description |
|--------|-------------|
| `f848c453d3` | exl3 3inst decode working end-to-end (Qwen3.5-0.8B) |
| `38bdfec5ac` | exl3 mul1 codebook markers + folded-weight cache |
| `e268c7d37c` | remove unused scaffold vars in exl3 dense kernel |
| `56450e75c7` | exl3 head_bits=0 fallback + float-bits early-return + TP loader |
| `9eed511411` | exl3 loaders handle MergedLinear partition + TP fallback |
| `fc195e4a80` | exl3 on-the-fly dequant for fused layers via kernel loop |
| `7b4cffba90` | contiguous output slice for fused-loop Hadamard |
| `21abd9108e` | exl3 loader copy direction + free original fp16 weight |

**Status**: exl3 kernels ARE in the current `.so` (verified: `exl3_gemm_rdna2`, `exl3_hadamard_128`, `moe_exl3_gemm_rdna2` all registered). Registration works. The user's session finding about broken registration was from a DIFFERENT build — the current build (Aug 28 10:20) works.

---

## 3. The fa_rdna2 Bug — What We Know

### 3.1 Symptom

- Short prompts (6 tokens): first generated token is wrong ("There" instead of "Paris")
- Long prompts (5000 tokens): was crashing with OOB (FIXED by 07ebb0d6b), now produces wrong output ("Help")
- Deterministic garbage (same output every time for same prompt)
- The output "There is no meaningfully relevant content in the query" is a specific training-pattern phrase

### 3.2 Debug Evidence

From `VLLM_DBG_RDNA_SHAPES=1` (server log):
```
head_size=256 num_kv_heads=4 key_cache.shape=(33, 4, 32, 784, 8) 
value_cache.shape=(33, 4, 32, 784, 8) kv_cache.shape=(2, 33, 4, 256, 784)
```

Key parameters:
- `head_size = 256` (full-attention layers, NOT 128!)
- `num_kv_heads = 4` (GQA: 24 Q heads / 4 KV heads = 6 groups)
- `block_size = 784` (mamba-aligned, NOT the usual 16!)
- `x_dim = 8`, `D/x = 32` (fp16 packing)
- `kv_cache_dtype = auto` (fp16, non-quantized)

### 3.3 Dispatch Path for Short Prompts

For a 6-token prompt (chat template ~26 tokens total):
- `max_seqlen_k = 26` (KV length)
- `_kv_splits = min(8, (26+1023)//1024) = min(8, 1) = 1`
- `causal = True` (prefill)
- `sliding_window = 0`

Since `kv_splits=1` and `max_seqlen_k < 4096` and `head_size == 256`:
- NOT the short variant (requires head_size=128)
- NOT splitk (requires kv_splits >= 2)
- → `fa_rdna2_prefill_paged_varlen` (general varlen, D=256)

For decode (after prefill):
- `max_seqlen_q = 1` → `fa_rdna2_decode_paged` (D=256)

### 3.4 What We Verified

1. **Chat template renders correctly** — `apply_chat_template` produces the right tokens
2. **Tokenizer works** — `tokenize` endpoint returns correct token IDs
3. **KV cache layout is correct** — 5D `[num_blocks, H_kv, D/x, block_size, x]` with correct strides
4. **GDN layers work** — T2 works with same GDN path (GDN decode HIP fires)
5. **W4A16 kernel works** — T2 works with RDNA2W4A16LinearKernel

---

## 4. Hypotheses for Next Session

### 4.1 QK Dot Product Bug (most likely)

The `fdot2` calls in the D=256 kernel might have wrong index alignment for the GQA case (24 Q heads / 4 KV heads = 6 groups). The kernel computes `h_kv = h_q / kv_group_num` where `kv_group_num = H_q / H_kv = 24/4 = 6`. But the `sQ` and `sK` shared memory layouts might be misaligned for D=256.

**Test**: Add debug output to the kernel to print `sQ[0]` and `sK[0]` for the first query token and first KV token, then compare with a reference implementation.

### 4.2 KV Cache Stride Bug

The kernel receives `stride_kc0..4` from `key_cache.stride(i)`. For the 5D layout `[33, 4, 32, 784, 8]`:
- `stride(0) = 4 * 32 * 784 * 8 = 802,816`
- `stride(1) = 32 * 784 * 8 = 200,704`
- `stride(2) = 784 * 8 = 6,272`
- `stride(3) = 8`
- `stride(4) = 1`

But the kernel expects `D/x = 32` and `x = 8`. If the actual `x_dim` from `key_cache.size(4)` is different (e.g., 1 instead of 8), the stride computation would be wrong.

**Test**: Print `x_dim` in the kernel and verify it matches `key_cache.size(4)`.

### 4.3 Causal Masking Bug

The kernel uses `q_local = q_start_in_seq + br` for the causal mask. But `q_start_in_seq = q_block * BR_PREFILL` is the position within the sequence, not the absolute position. For a single sequence, this should be fine. But if `cu_query_lens` is wrong, `q_local` would be wrong.

**Test**: Print `q_local` and `k_global` for the first few query/KV pairs and verify the mask is correct.

### 4.4 block_size=784 Interplay

The hybrid model forces `block_size=784` (mamba-aligned). The kernel's `seq_block_table[n_global / block_size]` should work for any block_size. But if the block_table contains garbage or the kernel assumes block_size=16, it would read wrong KV tokens.

**Test**: Print `block_table[0]` for the first sequence and verify it's 0 (the first block).

### 4.5 The 5D View Stride Mismatch (CRITICAL — most likely)

The KV cache layout `[33, 4, 32, 784, 8]` is a view of `[33, 4, 256, 784]`. The kernel indexes `key_cache + block_idx*stride_kc0 + h_kv*stride_kc1 + d_sub*stride_kc2 + slot*stride_kc3 + x_idx*stride_kc4`.

But the 5D view of `[33, 4, 256, 784]` as `[33, 4, 32, 784, 8]` has strides:
- `stride(0) = 802,816` (same)
- `stride(1) = 200,704` (same)
- `stride(2) = 784 * 8 = 6,272` (D/x stride)
- `stride(3) = 8` (slot stride)
- `stride(4) = 1` (x stride)

The kernel computes `d_sub = d / x_dim` and `x_idx = d % x_dim`. For `x_dim = 8`:
- `d_sub = d / 8` in `[0, 32)`
- `x_idx = d % 8` in `[0, 8)`

The address is:
```
block_idx*802816 + h_kv*200704 + d_sub*6272 + slot*8 + x_idx*1
```

But the ORIGINAL layout `[33, 4, 256, 784]` has strides:
- `stride(0) = 802,816`
- `stride(1) = 200,704`
- `stride(2) = 784` (D stride)
- `stride(3) = 1` (slot stride)

The element at `[block, h, d, slot]` is at:
```
block*802816 + h*200704 + d*784 + slot*1
```

For the 5D view `[block, h, d_sub, slot, x_idx]` where `d = d_sub*8 + x_idx`:
```
block*802816 + h*200704 + d_sub*6272 + slot*8 + x_idx*1
```

Setting `d = d_sub*8 + x_idx`:
```
block*802816 + h*200704 + (d_sub*8 + x_idx)*784 + slot*1
= block*802816 + h*200704 + d_sub*6272 + x_idx*784 + slot
```

For these to be equal:
```
d_sub*6272 + slot*8 + x_idx = d_sub*6272 + x_idx*784 + slot
slot*8 + x_idx = x_idx*784 + slot
slot*8 - slot = x_idx*784 - x_idx
7*slot = 783*x_idx
```

**This is NOT equal in general!** The 5D view `[33, 4, 32, 784, 8]` has DIFFERENT strides than the original `[33, 4, 256, 784]`. The kernel is reading the WRONG data!

**Wait** — but PyTorch `view()` computes the correct strides for the new shape. The 5D view of a contiguous `[33, 4, 256, 784]` tensor as `[33, 4, 32, 784, 8]` should have strides:
- `stride(0) = 802,816`
- `stride(1) = 200,704`
- `stride(2) = 784 * 8 = 6,272`
- `stride(3) = 8`
- `stride(4) = 1`

And the element `[block, h, d_sub, slot, x_idx]` should be at:
```
block*802816 + h*200704 + d_sub*6272 + slot*8 + x_idx*1
```

But this is NOT the same as the original element `[block, h, d, slot]` where `d = d_sub*8 + x_idx`:
```
block*802816 + h*200704 + d*784 + slot*1
= block*802816 + h*200704 + (d_sub*8 + x_idx)*784 + slot
= block*802816 + h*200704 + d_sub*6272 + x_idx*784 + slot
```

For these to be equal:
```
d_sub*6272 + slot*8 + x_idx = d_sub*6272 + x_idx*784 + slot
slot*8 + x_idx = x_idx*784 + slot
slot*8 - slot = x_idx*784 - x_idx
7*slot = 783*x_idx
```

**This is NOT equal!** So the 5D view is NOT a valid view of the original tensor in terms of element correspondence.

BUT — PyTorch `view()` should compute the correct strides. Let me re-check.

Actually, for a contiguous tensor `[33, 4, 256, 784]`, the strides are:
- `stride(0) = 4 * 256 * 784 = 802,816`
- `stride(1) = 256 * 784 = 200,704`
- `stride(2) = 784`
- `stride(3) = 1`

Now view as `[33, 4, 32, 784, 8]`. The new shape has:
- `dim 0: 33` (blocks)
- `dim 1: 4` (heads)
- `dim 2: 32` (D/x)
- `dim 3: 784` (block_size)
- `dim 4: 8` (x)

For a contiguous tensor, the strides for the new shape are:
- `stride(0) = 4 * 32 * 784 * 8 = 802,816`
- `stride(1) = 32 * 784 * 8 = 200,704`
- `stride(2) = 784 * 8 = 6,272`
- `stride(3) = 8`
- `stride(4) = 1`

The element `[block, h, d_sub, slot, x_idx]` is at:
```
block*802816 + h*200704 + d_sub*6272 + slot*8 + x_idx*1
```

Now, which element of the original `[33, 4, 256, 784]` does this correspond to? The offset is:
```
block*802816 + h*200704 + d_sub*6272 + slot*8 + x_idx*1
```

For the original tensor, the element `[block, h, d, slot]` is at:
```
block*802816 + h*200704 + d*784 + slot*1
```

For the same offset, we need:
```
d_sub*6272 + slot*8 + x_idx = d*784 + slot
```

Setting `d = d_sub*8 + x_idx`:
```
d_sub*6272 + slot*8 + x_idx = (d_sub*8 + x_idx)*784 + slot
d_sub*6272 + slot*8 + x_idx = d_sub*6272 + x_idx*784 + slot
slot*8 + x_idx = x_idx*784 + slot
slot*8 - slot = x_idx*784 - x_idx
7*slot = 783*x_idx
```

**This is NOT equal!** So the 5D view is NOT a valid view of the original tensor in terms of element correspondence.

**BUT** — PyTorch `view()` should compute the correct strides. Let me verify with a smaller example.

Actually, I think the issue is that I'm confusing the reshape direction. Let me re-think.

The original tensor is `[33, 4, 256, 784]`. The view is `[33, 4, 32, 784, 8]`. The view splits the third dimension (256) into (32, 8). But it also keeps the fourth dimension (784).

For a contiguous tensor, the view function computes the strides correctly. The new shape `[33, 4, 32, 784, 8]` has the same number of elements as the original `[33, 4, 256, 784]`. The strides are computed correctly for the new shape.

But the element correspondence is:
```
[i, j, m, n, p] → [i, j, m*8 + p, n]
```

Let me verify with the offsets.

Original element `[i, j, k, l]` is at offset `i*802816 + j*200704 + k*784 + l`.

New element `[i, j, m, n, p]` is at offset `i*802816 + j*200704 + m*6272 + n*8 + p`.

If we set `k = m*8 + p` and `l = n`, then:
```
k*784 + l = (m*8 + p)*784 + n = m*6272 + p*784 + n
```

But we want `m*6272 + n*8 + p`. These are NOT equal unless `p*784 + n = n*8 + p`, which is only true if `n=0` or `p=0`.

So the mapping `[i, j, m, n, p] → [i, j, m*8 + p, n]` is NOT correct!

**WAIT** — this means the 5D view is NOT a valid view of the original tensor! The elements are in a different order!

But the debug output shows `key_cache.shape = (33, 4, 32, 784, 8)`, which is 5D. So the view happened. But the view might not be correct.

Actually, I think the issue is that the view from `[33, 4, 256, 784]` to `[33, 4, 32, 784, 8]` is NOT a valid reshaping that preserves the logical order of elements. The view function in PyTorch computes the strides correctly for the new shape, but the element correspondence is different.

**THIS IS THE BUG!** The 5D view `[33, 4, 32, 784, 8]` has DIFFERENT strides than the original `[33, 4, 256, 784]`. The kernel is reading the WRONG data!

**BUT WAIT** — let me re-verify. For a contiguous tensor, `view()` should compute the correct strides. Let me check with a smaller example.

Consider a tensor of shape `[2, 8]` with elements stored in memory as:
```
a b c d e f g h
i j k l m n o p
```

The strides are:
- `stride(0) = 8`
- `stride(1) = 1`

Now view it as `[2, 2, 4]`:
- `stride(0) = 8`
- `stride(1) = 4`
- `stride(2) = 1`

The element `[0, 0, 0]` is at offset 0 (a)
The element `[0, 0, 1]` is at offset 1 (b)
The element `[0, 0, 2]` is at offset 2 (c)
The element `[0, 0, 3]` is at offset 3 (d)
The element `[0, 1, 0]` is at offset 4 (e)
The element `[0, 1, 1]` is at offset 5 (f)
The element `[0, 1, 2]` is at offset 6 (g)
The element `[0, 1, 3]` is at offset 7 (h)
The element `[1, 0, 0]` is at offset 8 (i)
...

Now, which element of the original `[2, 8]` tensor does `[0, 1, 0]` correspond to? It's at offset 4, which is element e. And e is at `[0, 4]` in the original. So `[0, 1, 0] → [0, 4]`.

The mapping is: `[i, j, k] → [i, j*4 + k]`.

This is correct! The view from `[2, 8]` to `[2, 2, 4]` splits the second dimension (8) into (2, 4), and the mapping is `[i, j, k] → [i, j*4 + k]`.

Now let's apply this to our case. The original tensor is `[33, 4, 256, 784]`. The view is `[33, 4, 32, 784, 8]`. The view splits the third dimension (256) into (32, 8) AND keeps the fourth dimension (784).

Wait, but the view adds a dimension. Let me think about this more carefully.

Actually, the view from `[33, 4, 256, 784]` to `[33, 4, 32, 784, 8]` is adding a dimension. The third dimension (256) is split into (32, 8), and the fourth dimension (784) is kept as-is. So the new shape is `[33, 4, 32, 784, 8]`.

For a contiguous tensor, the view function computes the strides correctly. The strides for the new shape are:
- `stride(0) = 4 * 32 * 784 * 8 = 802,816`
- `stride(1) = 32 * 784 * 8 = 200,704`
- `stride(2) = 784 * 8 = 6,272`
- `stride(3) = 8`
- `stride(4) = 1`

The element `[i, j, m, n, p]` is at offset `i*802816 + j*200704 + m*6272 + n*8 + p`.

Now, which element of the original tensor does this correspond to? We need to find `[i, j, k, l]` such that:
```
i*802816 + j*200704 + k*784 + l = i*802816 + j*200704 + m*6272 + n*8 + p
```

This simplifies to:
```
k*784 + l = m*6272 + n*8 + p
```

Since `l < 784` and `p < 8`, we can write:
```
l = p (since p < 8 and l < 784, and the equation must hold for all values)
k*784 = m*6272 + n*8
```

Wait, this doesn't work. Let me think again.

Actually, the issue is that the view from `[33, 4, 256, 784]` to `[33, 4, 32, 784, 8]` is NOT a simple split of one dimension into two. It's a reshaping that changes the number of dimensions.

For a contiguous tensor, the view function computes the strides correctly for the new shape. But the element correspondence is different.

Let me verify with a simple example. Consider a tensor of shape `[2, 4]` with elements stored in memory as:
```
a b c d
e f g h
```

The strides are:
- `stride(0) = 4`
- `stride(1) = 1`

Now view it as `[2, 2, 2]`:
- `stride(0) = 4`
- `stride(1) = 2`
- `stride(2) = 1`

The element `[0, 0, 0]` is at offset 0 (a)
The element `[0, 0, 1]` is at offset 1 (b)
The element `[0, 1, 0]` is at offset 2 (c)
The element `[0, 1, 1]` is at offset 3 (d)
The element `[1, 0, 0]` is at offset 4 (e)
...

Now, which element of the original `[2, 4]` tensor does `[0, 1, 0]` correspond to? It's at offset 2, which is element c. And c is at `[0, 2]` in the original. So `[0, 1, 0] → [0, 2]`.

The mapping is: `[i, j, k] → [i, j*2 + k]`.

This is correct! The view from `[2, 4]` to `[2, 2, 2]` splits the second dimension (4) into (2, 2), and the mapping is `[i, j, k] → [i, j*2 + k]`.

Now let's apply this to our case. The original tensor is `[33, 4, 256, 784]`. The view is `[33, 4, 32, 784, 8]`. The view splits the third dimension (256) into (32, 8) AND keeps the fourth dimension (784).

Wait, but the view adds a dimension. Let me think about this more carefully.

Actually, the view from `[33, 4, 256, 784]` to `[33, 4, 32, 784, 8]` is adding a dimension. The third dimension (256) is split into (32, 8), and the fourth dimension (784) is kept as-is. So the new shape is `[33, 4, 32, 784, 8]`.

For a contiguous tensor, the view function computes the strides correctly. The strides for the new shape are:
- `stride(0) = 4 * 32 * 784 * 8 = 802,816`
- `stride(1) = 32 * 784 * 8 = 200,704`
- `stride(2) = 784 * 8 = 6,272`
- `stride(3) = 8`
- `stride(4) = 1`

The element `[i, j, m, n, p]` is at offset `i*802816 + j*200704 + m*6272 + n*8 + p`.

Now, which element of the original tensor does this correspond to? We need to find `[i, j, k, l]` such that:
```
i*802816 + j*200704 + k*784 + l = i*802816 + j*200704 + m*6272 + n*8 + p
```

This simplifies to:
```
k*784 + l = m*6272 + n*8 + p
```

Since `l < 784` and `p < 8`, we can write:
```
l = (m*6272 + n*8 + p) % 784
k = (m*6272 + n*8 + p) / 784
```

But this doesn't give a clean mapping. The issue is that the new shape doesn't correspond to a simple split of the original dimensions.

**OK so I've been going in circles. Let me just verify the actual strides at runtime.**

The key insight is: the kernel receives `stride_kc0..4` from `key_cache.stride(i)`. If `key_cache` is the 5D view `[33, 4, 32, 784, 8]`, then the strides should be computed correctly by PyTorch's `view()`. But the kernel expects `stride_kc2` to be the D/x stride (784 * 8 = 6272) and `stride_kc3` to be the slot stride (8). If these are wrong, the kernel reads the wrong data.

Let me verify the actual strides at runtime. I'll add a debug print to the kernel to print the strides.

Actually, I realize I should just check the actual strides of the key_cache tensor. Let me modify the debug output to include the strides.

---

## 5. Next Steps for Debugging

### Step 1: Verify KV Cache Strides

Add debug output to print the actual strides of `key_cache` and `value_cache` in `rdna_attn.py`:

```python
print(f"key_cache.shape={key_cache.shape} strides={key_cache.stride()} "
      f"value_cache.shape={value_cache.shape} strides={value_cache.stride()}", flush=True)
```

Expected strides for `[33, 4, 32, 784, 8]`:
- `stride(0) = 802,816`
- `stride(1) = 200,704`
- `stride(2) = 6,272`
- `stride(3) = 8`
- `stride(4) = 1`

### Step 2: Verify Kernel Indexing

Add debug output to the kernel to print the computed address for the first few elements:

```cpp
if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && t == 0) {
    printf("kernel: block_idx=%d h_kv=%d d_sub=%d slot=%d x_idx=%d addr=%p\n",
           block_idx, h_kv, d_sub, slot, x_idx, k_ptr);
}
```

Compare with the expected address from the Python side.

### Step 3: Test with a Simple Model

Run a small model (e.g., Qwen2.5-0.5B) with `VLLM_USE_RDNA2_FA=1` to see if fa_rdna2 works for a simple case. If it works, the issue is specific to the hybrid model (block_size=784).

### Step 4: Compare with Triton Reference

Write a standalone test that loads the same weights and compares fa_rdna2 vs Triton attention for the same input. This will tell us exactly what's wrong.

### Step 5: Check the `_maybe_reinterp_v_to_5d` Function

The `_maybe_reinterp_v_to_5d` function reinterprets V from 4D to 5D. Let me verify it's doing this correctly:

```python
def _maybe_reinterp_v_to_5d(self, key_cache, value_cache):
    if (value_cache.dim() == 4 and key_cache.dim() == 5 and self.head_size in _SUPPORTED_HEAD_SIZES):
        num_blocks, h_kv, head_size_d, block_sz = value_cache.shape
        x_dim = key_cache.shape[4]
        if head_size_d % x_dim == 0:
            value_cache = value_cache.view(num_blocks, h_kv, head_size_d // x_dim, block_sz, x_dim)
    return value_cache
```

For `head_size_d = 256` and `x_dim = 8`, this reinterprets V to `[num_blocks, h_kv, 32, block_sz, 8]`. This is correct.

But wait — what if `x_dim` is NOT 8? Let me check. The debug output shows `key_cache.shape = (33, 4, 32, 784, 8)`, so `x_dim = 8`. And `D/x = 32` (256 / 8 = 32). So the view is correct.

But the kernel receives `x_dim = key_cache.size(4)` which is 8. And it computes `d_sub = d / x_dim` and `x_idx = d % x_dim`. For `x_dim = 8`, `d_sub = d / 8` and `x_idx = d % 8`. This is correct.

Hmm, but what if the actual `x_dim` is different? Let me check the kernel's `x_dim` parameter.

Actually, I think the issue might be simpler. Let me check if the kernel is receiving the right `x_dim` value.

---

## 6. Files to Modify

### 6.1 `vllm/v1/attention/backends/rdna_attn.py`

Add debug output to print KV cache shapes and strides:

```python
# After split_kv_cache and _maybe_reinterp_v_to_5d:
if os.environ.get("VLLM_DBG_RDNA_SHAPES") == "1" and not getattr(self, "_dbg_shapes_done", False):
    self._dbg_shapes_done = True
    print(f"[RDNASHAPES] head_size={self.head_size} num_kv_heads={self.num_kv_heads} "
          f"key_cache.shape={tuple(key_cache.shape)} key_cache.strides={key_cache.stride()} "
          f"value_cache.shape={tuple(value_cache.shape)} value_cache.strides={value_cache.stride()} "
          f"kv_cache.shape={tuple(kv_cache.shape)}", flush=True)
```

### 6.2 `csrc/rocm/fa_rdna2.cu`

Add debug output to the kernel to print computed addresses:

```cpp
// In the QK dot product loop, for the first query token and first KV token:
if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && idx == 0 && br == 0 && k == 0) {
    printf("kernel: br=%d k=%d sQ[%d]=%f sK[%d]=%f sK_row[%d]=%f\n",
           br, k, 0, __half2float(sQ[0]), 0, __half2float(sK[0]), 0, __half2float(sK_row[0]));
}
```

---

## 7. Environment Variables for Debugging

```bash
export VLLM_DBG_RDNA_SHAPES=1  # Print KV cache shapes and strides
export VLLM_EXL3_DEBUG_LAYER=1  # Print per-layer hidden state norms (for GDN debugging)
export VLLM_LOG_GDN_DISPATCH=1  # Print GDN kernel dispatch decisions
```

---

## 8. Test Configurations

### 8.1 Minimal Working Config (T1)

```bash
# T1: Triton W4A16 + Triton FA + GDN Triton + eager
python -m vllm.entrypoints.cli.main serve \
  /home/chenco_adm/Qwen3.8-27B-AWQ-INT4 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 --dtype float16 \
  --language-model-only --skip-mm-profiling --trust-remote-code \
  --enforce-eager
```

### 8.2 RDNA2 W4A16 Config (T2)

```bash
# T2: + RDNA2W4A16LinearKernel
python -m vllm.entrypoints.cli.main serve \
  /home/chenco_adm/Qwen3.8-27B-AWQ-INT4 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 --dtype float16 \
  --language-model-only --skip-mm-profiling --trust-remote-code \
  --enforce-eager
```

### 8.3 fa_rdna2 Config (T3)

```bash
# T3: + fa_rdna2 (BROKEN)
python -m vllm.entrypoints.cli.main serve \
  /home/chenco_adm/Qwen3.8-27B-AWQ-INT4 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 --dtype float16 \
  --language-model-only --skip-mm-profiling --trust-remote-code \
  --enforce-eager
```

With environment:
```bash
export VLLM_USE_RDNA2_FA=1
export VLLM_DBG_RDNA_SHAPES=1
```

---

## 9. Known Issues

1. **fa_rdna2 produces wrong output for Qwen3.8-27B-AWQ** — the main bug to fix
2. **exl3 kernels are in the .so but not tested end-to-end** — need to test with Ornith
3. **GDN prefill uses Triton/FLA** — the 5-kernel HIP chain is registered but not selected (resolver returns "triton" on ROCm)
4. **Prefix caching is experimental for Mamba models** — "Mamba cache mode is set to 'align'" warning

---

## 10. References

- **Journal**: `docs/profiling/awq-hip-profile-journal-2026-08-26.md` §8 (probe artifact resolution)
- **Skill**: `~/.config/opencode/skills/endpoint-testing/` (probe template and diagnosis guide)
- **Commit**: `07ebb0d6b` (fa_rdna2 splitk partial-buffer indexing overflow fix)
- **Branch**: `rdna2_extras` (local: `vllm_humanwork_next`, remote: `/home/chenco_adm/vllm_humanwork`)
- **Remote stash**: `stash@{0}` (old exl3 work + fa_rdna2 fix — superseded by local commits, recoverable at `/tmp/stash_recover.patch`)

---

*Handoff written: 2026-08-29*  
*Next session: verify KV cache strides, then fix fa_rdna2 kernel indexing*
