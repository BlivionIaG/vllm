#!/usr/bin/env python3
"""
REAP saliency capture for DeepSeek V4 Flash (custom inference/model.py).

Loads the 0xSero K160 REAP'd checkpoint (or any compatible V4-Flash MXFP4
checkpoint with inference/model.py), registers forward hooks on every
MoE expert to capture per-token output norms, and on every MoE gate
to capture the per-token top-k gate weights.

Accumulates per-expert saliency:
  S_j = sum over tokens t where expert j in top-k of [g_j(t) * ||f_j(t)||]
  C_j = count of active tokens

Saves to REAP_STATS_JSONL with one record per (layer, expert):
  {"layer": int, "expert": int, "saliency_sum": float, "count": int}

Run once on calibration data, then use reap_score.py to convert to
per-expert REAP scores, then structural_prune_k128.py to actually prune.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                   help="Path to model dir with config.json + inference/model.py")
    p.add_argument("--out", required=True,
                   help="Output JSONL path for per-expert saliency stats")
    p.add_argument("--num-samples", type=int, default=64,
                   help="Number of calibration samples (default 64)")
    p.add_argument("--max-tokens-per-sample", type=int, default=2048,
                   help="Max tokens per calibration sample (default 2048)")
    p.add_argument("--layers", default="all",
                   help="Comma-separated layer indices, or 'all' (default all)")
    return p.parse_args()


def build_calibration_inputs(tokenizer, num_samples, max_tokens, device):
    """Load a small slice of wikitext or similar for calibration.

    We use a few short public-domain passages to avoid HF dataset downloads.
    Replace with a richer corpus if available.
    """
    passages = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "In a hole in the ground there lived a hobbit. " * 20,
        "It was the best of times, it was the worst of times. " * 20,
        "To be, or not to be, that is the question. " * 20,
        "All happy families are alike; each unhappy family is unhappy in its own way. " * 20,
        "Call me Ishmael. Some years ago—never mind how long precisely—"
        "having little or no money in my purse, and nothing particular to interest me on shore, "
        "I thought I would sail about a little and see the watery part of the world. " * 5,
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
        "must be in want of a wife. " * 10,
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendia was to remember "
        "that distant afternoon when his father took him to discover ice. " * 10,
        "The Zen of Python, by Tim Peters: Beautiful is better than ugly. "
        "Explicit is better than implicit. Simple is better than complex. "
        "Complex is better than complicated. Flat is better than nested. " * 10,
        "In the beginning God created the heavens and the earth. " * 20,
    ]
    # Repeat to reach num_samples
    while len(passages) < num_samples:
        passages = passages + passages
    passages = passages[:num_samples]

    out = []
    for text in passages:
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
        ids = ids[:, :max_tokens]
        if ids.shape[1] < 16:
            continue
        out.append(ids.to(device))
    return out


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir} ...")
    t0 = time.time()

    # Load config to know dims
    import json as _json
    cfg = _json.load(open(model_dir / "config.json"))
    n_routed = cfg.get("n_routed_experts", 160)
    n_layers = cfg.get("num_hidden_layers", 43)
    n_hash = cfg.get("n_hash_layers", 3)  # from inference/config.json
    if "n_hash_layers" not in cfg:
        # Try inference/config.json
        inf_cfg = model_dir / "inference" / "config.json"
        if inf_cfg.exists():
            n_hash = _json.load(open(inf_cfg)).get("n_hash_layers", 0)
    print(f"  n_routed_experts={n_routed}, n_layers={n_layers}, n_hash_layers={n_hash}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    # Build calibration inputs
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Building calibration inputs ...")
    calib = build_calibration_inputs(tokenizer, args.num_samples,
                                    args.max_tokens_per_sample, device)
    print(f"  {len(calib)} samples, total tokens ~{sum(x.shape[1] for x in calib)}")

    # Allocate per-(layer, expert) accumulators on CPU
    # Keys: (layer_id, expert_id) -> [saliency_sum, count]
    accum = {}  # dict[(int, int)] = [sum: float, count: int]

    # Hooks ------------------------------------------------------------
    # We hook:
    #   - nn.Module named "layers.{L}.ffn" (the MoE block) -- to get
    #     weights, indices via inspecting the call args
    #   - each nn.Module in self.experts (Expert modules) -- to capture
    #     ||f_j(x)||_2 for tokens routed to that expert
    #
    # The 0xSero / V4-Flash custom MoE.forward signature is:
    #   def forward(self, x, input_ids): -> y
    #   and gate.forward(x, input_ids) -> (weights, indices)
    #
    # We use a pre-hook on each Expert module. The pre-hook receives
    # (module, args). For an Expert(x, weights=None), we want to know
    # which tokens are in this expert's batch and the gate weight. The
    # MoE.forward does:
    #   for i in range(self.experts_start_idx, self.experts_end_idx):
    #     if counts[i] == 0: continue
    #     idx, top = torch.where(indices == i)
    #     y[idx] += expert(x[idx], weights[idx, top, None])
    # So each expert receives (x[idx], weights[idx, top, None]) when
    # called. The pre-hook sees (module, args) where args = (x[idx], weights).

    layer_filter = None
    if args.layers != "all":
        layer_filter = set(int(x) for x in args.layers.split(","))

    # We need to track per-call: which expert index, and per-token gate weight.
    # The Expert pre-hook fires per Expert call within a single MoE.forward.
    # To associate with the expert index, we inspect module.__class__.__name__
    # and the module's position in the ModuleList. Simpler: walk the parent
    # MoE module to get self.experts.index(expert_module).
    #
    # To avoid fragile parent-walking, we add a custom attribute
    # `expert_global_idx` to each Expert module by patching model after load.

    # Defer model load until after parsing
    print("Loading model weights (this may take a few minutes) ...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # offload to CPU as needed
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Find MoE modules and tag their experts
    moe_modules = {}  # layer_id -> MoE module
    for name, module in model.named_modules():
        # The 0xSero custom MoE class is named "MoE" in model.py
        if module.__class__.__name__ == "MoE" and "shared_experts" in dir(module):
            # Extract layer id from name like "layers.3.ffn"
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "layers":
                try:
                    lid = int(parts[1])
                except ValueError:
                    continue
                if layer_filter is not None and lid not in layer_filter:
                    continue
                moe_modules[lid] = module

    print(f"  found {len(moe_modules)} MoE layers (out of {n_layers})")
    if not moe_modules:
        print("ERROR: no MoE modules found, aborting")
        sys.exit(1)

    # Tag each Expert with its global index in self.experts
    for lid, moe in moe_modules.items():
        for eidx, expert in enumerate(moe.experts):
            if expert is None:
                continue
            expert._reap_global_idx = eidx

    # Hook factory: pre-hook on Expert captures ||f_j(x)||_2 and the gate weight.
    # The MoE forward passes (x[idx], weights[idx, top, None]) to expert.
    # So the pre-hook args are (x, weights) for the routed tokens.
    def make_expert_pre_hook(layer_id):
        def hook(module, args):
            # Skip if not in the routed-experts range
            eidx = getattr(module, "_reap_global_idx", None)
            if eidx is None or eidx >= n_routed:
                return
            if len(args) < 2:
                return
            x, weights = args[0], args[1]
            # weights shape: [num_tokens_routed_to_this_expert, 1] (or [N, 1])
            if weights is None:
                return
            # Flatten in case of weird shapes
            w = weights.float().reshape(-1)
            # Compute ||f_j(x)|| -- hook fires BEFORE forward, so we don't
            # have f_j yet. Instead, we hook on forward POST (output) below.
            # The pre-hook just records the gate weight.
            key = (layer_id, eidx)
            if key not in accum:
                accum[key] = [0.0, 0]
            # Defer saliency accumulation to the post-hook
            module._reap_pending_weights = w.detach().cpu()
        return hook

    def make_expert_post_hook(layer_id):
        def hook(module, args, output):
            eidx = getattr(module, "_reap_global_idx", None)
            if eidx is None or eidx >= n_routed:
                return
            w = getattr(module, "_reap_pending_weights", None)
            if w is None:
                return
            # output shape: [num_tokens_routed, dim]
            norms = torch.linalg.norm(output.float(), dim=-1).reshape(-1).detach().cpu()
            if norms.numel() != w.numel():
                # shape mismatch -- skip
                module._reap_pending_weights = None
                return
            contrib = (norms * w).sum().item()
            cnt = int(norms.numel())
            key = (layer_id, eidx)
            if key not in accum:
                accum[key] = [0.0, 0]
            accum[key][0] += contrib
            accum[key][1] += cnt
            module._reap_pending_weights = None
        return hook

    handles = []
    for lid, moe in moe_modules.items():
        for eidx, expert in enumerate(moe.experts):
            if expert is None:
                continue
            h1 = expert.register_forward_pre_hook(make_expert_pre_hook(lid))
            h2 = expert.register_forward_hook(make_expert_post_hook(lid))
            handles.extend([h1, h2])

    print(f"  registered {len(handles)} forward hooks")

    # Run calibration --------------------------------------------------
    print(f"Running {len(calib)} calibration samples ...")
    t0 = time.time()
    with torch.no_grad():
        for si, input_ids in enumerate(calib):
            try:
                _ = model(input_ids=input_ids)
            except Exception as e:
                print(f"  sample {si} FAILED: {type(e).__name__}: {e}")
                continue
            if (si + 1) % 8 == 0 or si == len(calib) - 1:
                rate = (si + 1) / (time.time() - t0)
                eta = (len(calib) - si - 1) / max(rate, 1e-6)
                print(f"  [{si+1}/{len(calib)}] {rate:.2f} samples/s, "
                      f"ETA {eta/60:.1f} min, accum entries={len(accum)}")

    # Save
    print(f"Writing {len(accum)} per-expert records to {out_path} ...")
    with open(out_path, "w") as f:
        for (lid, eidx), (ssum, cnt) in sorted(accum.items()):
            f.write(json.dumps({
                "layer": int(lid),
                "expert": int(eidx),
                "saliency_sum": float(ssum),
                "count": int(cnt),
            }) + "\n")
    print(f"Done in {(time.time()-t0)/60:.1f} min. Output: {out_path}")


if __name__ == "__main__":
    main()
