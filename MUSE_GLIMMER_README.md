# MuseGlimmer support in vLLM (branch: `muse_glimmer-support`)

Native vLLM inference for MuseGlimmer (`MuseGlimmerForCausalLM`) — no `trust_remote_code`.
This branch ports the HuggingFace MuseGlimmer inference path into the fork so partners
can serve MuseGlimmer checkpoints directly, including text, tool calling, images, and
temporally patched videos.

## What's implemented

| Area | File | Status |
|---|---|---|
| Text model | `vllm/model_executor/models/muse_glimmer.py` | ✅ NLL-validated vs HF; loads canonical + legacy norm names |
| Config (native) | `vllm/transformers_utils/configs/muse_glimmer.py` | ✅ registered (muse_glimmer / muse_glimmer_text / muse_glimmer_vision); accepts flat + nested layouts |
| Model registry | `vllm/model_executor/models/registry.py` | ✅ `MuseGlimmerForCausalLM` |
| ATEM tool parser | `vllm/tool_parsers/muse_glimmer_tool_parser.py` | ✅ unit-tested + served E2E (`--tool-call-parser muse_glimmer`) |
| Reasoning parser | `vllm/reasoning/muse_glimmer_reasoning_parser.py` | ✅ unit-tested + served E2E (`--reasoning-parser muse_glimmer`) |
| Chat template | `examples/tool_chat_template_muse_glimmer.jinja` | ✅ renders + round-trips |
| Vision (multimodal) | `vllm/model_executor/models/muse_glimmer.py` | ✅ image + video preprocessing, native encoder, and weight loading |

## Checkpoint compatibility (two config layouts)

The fork serves **both** MuseGlimmer HF config layouts natively (no trust_remote_code):

* **Canonical / nested** — current HF converter (`convert_muse_glimmer_weights_to_hf.py`,
  transformers 5.15): `text_config` / `vision_config` sub-dicts, canonical field
  names. Also uses canonical per-layer norm names.
* **Flat / legacy** — older converter (e.g. `guac-pytorch/tree/rl_v1/hf`,
  transformers 5.9): all fields top-level (`hidden_act`, `output_soft_cap_temp`,
  `rope_theta`, `vision_*`, `patch_token_id`). Also uses legacy guac norm names
  (`post_attn_norm` / `post_ffn_norm`).

`MuseGlimmerConfig` normalizes flat → nested (hoist + rename) on load, and
`MuseGlimmerForCausalLM.load_weights` maps legacy → canonical norm names scoped to the
legacy `model.layers.*` prefix so canonical checkpoints pass through untouched.
Both paths are verified to load AND serve tool-calling end-to-end.

> ⚠️ **Legacy checkpoints often ship `<|eom|>` (200007) in `eos_token_id`**
> (e.g. `rl_v1/hf` has `[200001, 200007, 200008]`). This collapses single-turn
> parallel tool-calling. Serve with a corrected `generation_config.json`
> (`eos_token_id = [200001, 200008]`) — see below.

## Known model behavior: tool-call namespacing

MuseGlimmer is trained on `namespace.function` tool names. Given a **namespaced** name
(e.g. `weather.get`) it emits `<atem:invoke name="weather.get">` correctly.
Given a **bare** name (e.g. `get_weather`) it synthesizes a namespace and emits
`get_weather.get_weather`. This is model behavior, **not** a parser bug (verified
against raw output). Partners should pass namespaced tool names, or a future
template/parser option can normalize bare names.


## Text-path architecture (vs Gemma2)

MuseGlimmer's text decoder is a Gemma2 derivative with these deltas, all handled in
`muse_glimmer.py`: SiLU-gated MLP; scaleless-RMSNorm-normalized token embeddings;
sandwich RMSNorms with a baked `+1` weight offset (distinct pre/post eps);
weightless fp32 QK-norm applied **before** RoPE with a query pre-scale;
per-head sigmoid attention output gate; iRoPE layout (NoPE layers → full
attention, RoPE layers → sliding window) with **interleaved** RoPE
(`is_neox_style=False`); logits pre-scaled by `output_multiplier` then tanh
soft-capped; untied `lm_head`.

## Validation — NLL parity vs HF reference

Teacher-forced per-token NLL (via vLLM `prompt_logprobs`) against the HF
reference on two distributions:

| Dataset | Docs | MATCH | CLOSE | MISMATCH | max mean-NLL Δ |
|---|---|---|---|---|---|
| notes_v5 (prose) | 20 | 11 | 9 | 0 | 0.0042 |
| fbsource_python (code) | 20 | 16 | 4 | 0 | 0.0050 |

`MATCH` < 1e-3, `CLOSE` < 0.1 (expected under bf16 + different kernels).

## Serving (tool-calling)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <muse_glimmer-hf-checkpoint> \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --chat-template examples/tool_chat_template_muse_glimmer.jinja \
  --generation-config auto
```

## Stop tokens (important)

Correct `eos_token_id = [200001, 200008]` = `<|end_of_text|>` + `<|eot|>`.
**Do NOT include `<|eom|>` (200007)** — it is end-of-*message* (turn continues;
it separates the reasoning block and each non-final parallel tool call). If
`<|eom|>` is a stop token, single-turn parallel tool-calling collapses to ~0%.
Ship this in the checkpoint's `generation_config.json`.

## Checkpoint requirement for tool-calling

Full tool-calling requires a checkpoint converted with the current HF MuseGlimmer
converter (`convert_muse_glimmer_weights_to_hf.py`), whose tokenizer registers the ATEM
framing tokens (`<|eom|>`, `<|eot|>`, `<|start|>`, `<|message|>` at ids
7/8/22/23) as real special tokens. Legacy `guac` HF exports lack these as
single tokens; the text/NLL path still works there, but token-level stop tokens
and clean framing require a properly converted checkpoint.
