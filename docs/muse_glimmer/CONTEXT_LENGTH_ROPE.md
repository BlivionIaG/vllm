# MuseGlimmer context length / RoPE cache sizing (max_position_embeddings)

**Status:** resolved (checkpoint-config fix). **Severity:** high (CUDA crash past 16K).
**Found:** 2026-07-25, running long-context evals against the vLLM fork.

## Symptom
Any request whose position crossed **16384** crashed the vLLM EngineCore: a TP worker
died with a C++ frame trace ("cancelled" / `EngineDeadError`), not a clean Python
exception. The scheduler dump showed the request had `num_computed_tokens=[16384]` and
was scheduling the next chunk when the worker died.

## Root cause
vLLM sizes its RoPE cos/sin cache to the model config's `max_position_embeddings`.
The HF/modular MuseGlimmer checkpoint's `config.json` (from the converter's `build_config()`)
declared **`max_position_embeddings: 16384`**, so vLLM built rotary tables for only
16384 positions. Position 16385 indexed past the end of that table → CUDA illegal
memory access → worker crash. Classic out-of-bounds; exactly what the
`VLLM_ALLOW_LONG_MAX_MODEL_LEN` warning predicts.

This is a checkpoint-config defect, NOT a model-capability limit. MuseGlimmer natively
supports long context (`rope_type=default`, `rope_theta=500000`, no rope_scaling).
Production (`guac26b` IPNext tenant, the backend behind `muse_glimmer_v1_3`) serves this model
at **`max_seq_len: 131072`** (newer checkpoints / `guac26blong` at 262144) with the
identical rope config — confirming 128K is safe.

## Fix
Set `max_position_embeddings = 131072` in the checkpoint's `config.json`
(`text_config.max_position_embeddings` for the nested/modular schema). vLLM then sizes
the RoPE cache for 128K and long-context requests work. Verified: a 20,014-token
request that previously crashed the engine now returns coherent output; short-context
generations unchanged.

Serve with `--max-model-len 131072`. No `VLLM_ALLOW_LONG_MAX_MODEL_LEN` override is
needed once the config declares 131072.

## ⚠️ Action for the HF conversion script
The MuseGlimmer→HF converter's `build_config()` (in `new-model-addition-onyx`) hardcodes
`max_position_embeddings=16384`. It must emit **131072** so freshly-converted
checkpoints don't reintroduce this crash. Until then, patch the exported
`config.json` post-conversion.

## Verified production serving flags (guac26b) — for parity
`max_seq_len:131072`, `max_num_batched_tokens:8192`, `skip_special_tokens:false`,
`enable_auto_tool_choice:true`, `model_parallel_size:8` (TP=8). The fork matches all of
these except TP (we validate at TP=4, a capacity-only difference).
