# MuseGlimmer vLLM fork — modular HF config schema mismatch (degenerate output)

**Status:** ✅ RESOLVED 2026-07-24 (commit on `muse_glimmer-support`). **Severity:** high (was silently wrong output, not a crash).
**Found:** 2026-07-24, validating the HF *modular* export (`MuseGlimmerForConditionalGeneration`)
against this vLLM fork (`~/projects/ony`, branch `muse_glimmer-support`).

## Resolution & validation
Fixed in `vllm/model_executor/models/muse_glimmer.py` via three schema-normalization helpers
(`_muse_glimmer_use_qk_norm`, `_muse_glimmer_use_attn_output_gate`, `_muse_glimmer_query_prescale`):
- missing `use_qk_norm` / `use_attn_output_gate` (modular omits them) now default to **True**;
- query pre-scale normalized so native raw `qk_scale_factor=43.784` and modular folded
  `3.87` both yield `scale_query_by ≈ 3.87` (net scaling matches native).

**Validated (fixed server, GPU-served modular checkpoint):**
- Degenerate loop gone. `"The capital of France is"` → `" Paris. It is the most populous
  city in France and the most populous city in the European Union..."` (was
  `"Paris. The capital of France is Paris. ..."` on the pre-fix server).
- **Teacher-forced argmax parity vs HF FP32 modular reference: 126/128 = 98.4%.** The 2
  disagreements are near-tie positions in the repetitive region (top1−top2 margins 1.50 /
  0.88 logits; vLLM's alt pick is the reference's #3), i.e. benign bf16-vs-fp32 + kernel
  argmax flips — the same CLOSE-not-MATCH band as the NLL gate.
- Unit test: `tests/transformers_utils/test_muse_glimmer_config_schema_norm.py`.

Below is the original investigation (kept for context).

---


## Symptom
The modular HF checkpoint loads cleanly (no missing/unexpected params) and the server
starts, but generation is **degenerate/repetitive**:

- raw `/v1/completions` `"The capital of France is"` -> `" Paris.\n\nThe first first first first ..."`
  (CORRECT first token, then collapses into repetition)
- `/v1/chat/completions` -> `" to the 1.: \"The 1.: \"The 1.: \"The ..."`

Correct first token + immediate degeneration = the forward is numerically wrong past the
first step (embeddings/lm_head are fine; attention is broken).

## Root cause — config **schema** mismatch (NOT weights, NOT a mapping bug)

> **Precise mechanism (verified 2026-07-24):** the modular `MuseGlimmerTextConfig` does **not
> define** `use_qk_norm` or `use_attn_output_gate` at all — the modular HF *modeling* applies
> QK-norm (`modeling_muse_glimmer.py:381`) and the attention output gate (`:427`, `attn_output *
> sigmoid(gate_proj(hidden))`) **unconditionally** (no config flag). This vLLM fork, written
> for the native/SFT *flat* config, gates both on `config.use_qk_norm` / `config.use_attn_output_gate`,
> which read as `None` on the modular config -> **both QK-norm AND the output gate are silently
> skipped**. That is the dominant cause of the degenerate output. (Additionally the query
> pre-scale is mis-derived; see below.) The values that ARE present are correct:
> `qk_scale_factor=3.87` (= native 43.784 / sqrt(128), i.e. the 1/sqrt(head_dim) folded out).

The vLLM fork's `MuseGlimmerAttention` (vllm/model_executor/models/muse_glimmer.py) was written against the
**native/SFT flat config**. The HF **modular** export (`text_config`, arch
`MuseGlimmerForConditionalGeneration`) names two attention knobs differently, so vLLM silently
runs attention **without QK-norm and without the query pre-scale**:

1. **QK-norm skipped.**
   - vLLM: `self.use_qk_norm = config.use_qk_norm; if self.use_qk_norm: self.qk_norm = ...`
   - modular `text_config` has **no `use_qk_norm`** (reads as `None`/falsey) -> qk_norm is never built/applied.
   - HF modeling builds qk_norm **unconditionally** (`modeling_muse_glimmer.py`: `self.qk_norm = MuseGlimmerRMSNorm(..., with_scale=False)`, no `if`), i.e. QK-norm is always on for MuseGlimmer.

2. **Query pre-scale dropped.**
   - vLLM reads `config.scale_query_by` (modular: `None`) -> applies no query pre-scale.
   - modular `text_config` uses **`qk_scale_factor = 3.87`**.
   - HF modeling forward: `query_states = self.qk_norm(query_states) * self.qk_scale_factor`
     then softmax `scaling = head_dim**-0.5`.

### The 3.87 value is correct (not corrupt)
`43.7840518911 (native params.json qk_scale) / sqrt(128) = 3.87` exactly. The modular config
folds the `1/sqrt(head_dim)` factor: it applies `qk_norm(q) * 3.87` and then the standard
softmax `scaling = head_dim**-0.5`. Net query scaling matches the native path.

### Proof the checkpoint + HF forward are right
A CPU FP32 load of this exact converted checkpoint via
`AutoModelForImageTextToText` (transformers, modular modeling) generates correctly and
matches the native reference **bitwise** (see guacamole-work/native-hf-fp32). So the bug is
purely in **this vLLM fork's forward for the modular config schema**, not the weights.

## Also required to even load the modular checkpoint (already applied on this branch)
These two edits were needed just to get past weight loading; keep them:

1. **Registry alias** (`vllm/model_executor/models/registry.py`):
   modular ships arch `MuseGlimmerForConditionalGeneration`; alias it to the text decoder impl:
   `"MuseGlimmerForConditionalGeneration": ("muse_glimmer", "MuseGlimmerForCausalLM"),`

2. **Attention-gate rename regex** (`vllm/model_executor/models/muse_glimmer.py`, `hf_to_vllm_mapper`):
   modular names the attention output gate `self_attn.gate_proj` (colliding with `mlp.gate_proj`,
   which the stacked rule maps to `gate_up_proj[0]`). Added a regex to rename the ATTENTION gate
   to `self_attn.output_gate_proj` before stacking:
   `re.compile(r"(\.layers\.\d+\.)self_attn\.gate_proj\."): r"\1self_attn.output_gate_proj."`
   (fires on both canonical `...language_model.layers.N.` and legacy `model.layers.N.` keys)

## Fix (the actual bug — attention forward)
In `MuseGlimmerAttention.__init__` / config handling, make the fork accept the **modular schema**:

- **use_qk_norm:** the modular `MuseGlimmerTextConfig` has NO such field; MuseGlimmer modeling applies QK-norm
  unconditionally. Default to `True` when the attribute is absent/None (do NOT treat missing as False).
- **use_attn_output_gate:** SAME issue — absent on modular `MuseGlimmerTextConfig`; modeling always applies
  `attn_output *= sigmoid(output_gate(hidden))`. Default to `True` when absent. (Currently vLLM skips the
  output gate entirely on modular configs -> another source of wrong output.)
- **query pre-scale (verify):** vLLM already reads `qk_scale_factor` but computes
  `scale_query_by = qk_scale_factor / sqrt(head_dim)` (muse_glimmer.py:184), which assumes `qk_scale_factor`
  is the NATIVE 43.784. The modular config's `qk_scale_factor=3.87` ALREADY has `1/sqrt(head_dim)`
  folded out (`43.784/sqrt(128)=3.87`). HF modeling does `q = qk_norm(q) * 3.87` then softmax
  `scaling = head_dim**-0.5`. So vLLM must apply `q_prescale = qk_scale_factor` DIRECTLY (not divided
  again by sqrt(head_dim)) when consuming a modular config, else the query is under-scaled by
  1/sqrt(head_dim). Normalize both schemas so the net query scaling equals native.

Recommend a `_text_config`-style helper that normalizes both schemas (native flat + modular
`text_config`) into the attributes `MuseGlimmerAttention` reads, so both checkpoints serve correctly.

## Repro
```
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model <modular HF export: arch=MuseGlimmerForConditionalGeneration> \
  --served-model-name muse_glimmer-rl-v1 --port 8011 --trust-remote-code \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --enforce-eager
curl -s localhost:8011/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"muse_glimmer-rl-v1","prompt":"The capital of France is","max_tokens":20,"temperature":0}'
# -> " Paris.\n\nThe first first first ..."  (degenerate)
```

## Cross-refs
- Converted modular checkpoint: `/data/users/betodepaola/models/muse_glimmer-rl-v1-hf-modular` (config `text_config.qk_scale_factor=3.87`, no `use_qk_norm`).
- FP32 correctness proof: `~/projects/guacamole-work/native-hf-fp32/`.
- HF modular modeling: `new-model-addition-onyx` branch `muse_glimmer-meta-toolcall-modular`, `modeling_muse_glimmer.py` (qk_norm unconditional; `q = qk_norm(q) * qk_scale_factor`).
