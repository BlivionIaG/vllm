#!/usr/bin/env python3
"""
K128 REAP/prune: keep exactly the top-128 most-active experts per layer.

Selection criterion: experts with the highest num_captured tokens during
calibration (max across w1/w2/w3). High capture count = frequently routed
= most important to preserve. This is the standard REAP criterion.

Per layer:
- Selects top-128 experts by max(num_captured across mats)
- Builds remap old_idx -> new_idx in [0, 128)
- Compacts expert weight tensors (drops pruned, renames kept to contiguous)
- Masks pruned experts in ffn.gate.bias to -inf so top-k never selects them

Config keeps n_routed_experts at 256 (router shape compatibility); the
patched model.py reads _expert_masks and applies the mask at the gate.

Output: 43 layers x 128 experts x {w1,w2,w3} = 16,128 expert tensors
(vs 43 x 256 x 3 = 33,024 in the source = 51% reduction in expert storage).
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict

MODEL_DIR = Path("/home/chenco_adm/models/DeepSeek-V4-Flash-0731-Int4-FP8")
STATS_FILE = Path("/home/chenco_adm/v4_quant/expert_stats.jsonl")
OUTPUT_DIR = Path("/home/chenco_adm/models/DeepSeek-V4-Flash-0731-Int4-FP8-K128")
NUM_SHARDS = 46
KEEP_K = 128


def select_topk(stats_file: Path, n_total_experts: int, keep_k: int,
                moe_layers: set):
    """For each layer, returns (pruned_set, old_to_new_map, n_kept).

    Selection: top-keep_k experts by max(num_captured across w1/w2/w3).
    Experts with no stats entry are treated as 0 captures.
    Layers present in the model but missing from stats (calibration gap)
    keep all experts (no prune) so the model remains loadable.
    """
    captures = defaultdict(lambda: defaultdict(int))
    layers_seen = set()

    with open(stats_file) as f:
        for line in f:
            s = json.loads(line)
            layer = s["layer"]
            if isinstance(layer, str):
                continue
            expert = s["expert"]
            captured = s.get("num_captured", 0)
            layers_seen.add(layer)
            if captured > captures[layer][expert]:
                captures[layer][expert] = captured

    pruned = {}
    remap = {}
    kept_count = {}

    for layer in sorted(moe_layers):
        if layer not in layers_seen:
            # Calibration gap: keep all experts for this layer.
            pruned[layer] = set()
            remap[layer] = {e: e for e in range(n_total_experts)}
            kept_count[layer] = n_total_experts
            continue
        layer_caps = [(e, captures[layer].get(e, 0))
                      for e in range(n_total_experts)]
        layer_caps.sort(key=lambda x: (-x[1], x[0]))
        kept_ids = [e for e, _ in layer_caps[:keep_k]]
        pruned_ids = [e for e, _ in layer_caps[keep_k:]]

        pruned[layer] = set(pruned_ids)
        kept_sorted = sorted(kept_ids)
        remap[layer] = {old: new for new, old in enumerate(kept_sorted)}
        kept_count[layer] = keep_k

    return pruned, remap, kept_count


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    print(f"Reading {STATS_FILE}...")
    from safetensors import safe_open
    gate_dim = None
    for shard_idx in range(1, NUM_SHARDS + 1):
        src = MODEL_DIR / f"model-{shard_idx:05d}-of-00046.safetensors"
        if not src.exists():
            continue
        with safe_open(str(src), framework="pt") as f:
            for k in f.keys():
                if k.startswith("layers.") and ".ffn.gate.weight" in k:
                    gate_dim = f.get_tensor(k).shape[0]
                    break
        if gate_dim is not None:
            break
    if gate_dim is None:
        raise RuntimeError("Could not find ffn.gate.weight in any shard")

    # Discover all MoE layers (those with a ffn.gate.weight). Layers without
    # stats get a no-op (keep all experts) so the model stays loadable.
    moe_layers = set()
    for shard_idx in range(1, NUM_SHARDS + 1):
        src = MODEL_DIR / f"model-{shard_idx:05d}-of-00046.safetensors"
        if not src.exists():
            continue
        with safe_open(str(src), framework="pt") as f:
            for k in f.keys():
                if k.startswith("layers.") and ".ffn.gate.weight" in k:
                    moe_layers.add(int(k.split(".")[1]))
    print(f"Source model: {gate_dim} routed experts across {len(moe_layers)} "
          f"MoE layers, keeping top-{KEEP_K} per layer")

    pruned, remap, kept = select_topk(STATS_FILE, gate_dim, KEEP_K, moe_layers)

    fallback_layers = [L for L in sorted(moe_layers) if kept[L] != KEEP_K]
    if fallback_layers:
        print(f"Fallback (no stats, kept all {gate_dim}): {fallback_layers}")
    total_pruned = sum(len(v) for v in pruned.values())
    total_all = sum(kept.values()) + total_pruned
    print(f"Pruned experts: {total_pruned}/{total_all} "
          f"({total_pruned/total_all*100:.1f}%)")
    kept_counts = sorted(set(kept.values()))
    print(f"Kept per layer: {kept_counts} (per-layer counts: {sorted(kept.values())})")

    for f in MODEL_DIR.iterdir():
        if f.is_file() and not f.name.startswith("model-") and not f.name.startswith("."):
            shutil.copy2(f, OUTPUT_DIR / f.name)

    src_inference = MODEL_DIR / "inference"
    if src_inference.exists():
        dst_inference = OUTPUT_DIR / "inference"
        if dst_inference.exists():
            shutil.rmtree(dst_inference)
        shutil.copytree(src_inference, dst_inference)
        patched_model = Path("/home/chenco_adm/v4_quant/model_patched.py")
        if patched_model.exists():
            shutil.copy2(patched_model, dst_inference / "model.py")
            print(f"Patched model.py -> {dst_inference / 'model.py'}")

    from safetensors.torch import save_file

    for shard_idx in range(1, NUM_SHARDS + 1):
        src = MODEL_DIR / f"model-{shard_idx:05d}-of-00046.safetensors"
        dst = OUTPUT_DIR / f"model-{shard_idx:05d}-of-00046.safetensors"
        if not src.exists():
            continue

        new_tensors = {}
        with safe_open(str(src), framework="pt") as f:
            meta = dict(f.metadata() or {})
            for k in f.keys():
                t = f.get_tensor(k)
                parts = k.split(".")
                if (len(parts) >= 6 and parts[0] == "layers"
                        and parts[2] == "ffn" and parts[3] == "experts"):
                    layer_idx = int(parts[1])
                    expert_idx = int(parts[4])
                    if expert_idx not in remap.get(layer_idx, {}):
                        continue
                    new_expert_idx = remap[layer_idx][expert_idx]
                    new_key = (f"layers.{layer_idx}.ffn.experts."
                               f"{new_expert_idx}." + ".".join(parts[5:]))
                    new_tensors[new_key] = t
                elif (parts[-1] == "bias" and len(parts) >= 4
                      and parts[2] == "ffn" and parts[3] == "gate"):
                    layer_idx = int(parts[1])
                    pruned_set = pruned.get(layer_idx, set())
                    if pruned_set:
                        new_bias = t.clone()
                        for pruned_idx in pruned_set:
                            new_bias[pruned_idx] = -1e9
                        new_tensors[k] = new_bias
                    else:
                        new_tensors[k] = t
                else:
                    new_tensors[k] = t

        save_file(new_tensors, str(dst), metadata=meta)
        del new_tensors
        print(f"  shard {shard_idx}/{NUM_SHARDS} done", flush=True)

    expert_masks = {}
    for layer_idx in sorted(kept.keys()):
        pruned_set = pruned.get(layer_idx, set())
        mask = [False] * gate_dim
        for old_idx in range(gate_dim):
            if old_idx not in pruned_set:
                mask[old_idx] = True
        expert_masks[str(layer_idx)] = mask

    config_path = OUTPUT_DIR / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    old_n = config.get("n_routed_experts", gate_dim)
    config["n_routed_experts"] = old_n
    config["_original_n_routed_experts"] = old_n
    config["_prune_method"] = "top_k_by_max_captures"
    config["_prune_keep_k"] = KEEP_K
    config["_kept_per_layer"] = {str(k): v for k, v in kept.items()}
    config["_expert_masks"] = expert_masks

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Updated config.json with _expert_masks ({len(expert_masks)} layers)")
    print(f"Done. K128 pruned model at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
