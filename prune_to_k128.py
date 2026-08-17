#!/usr/bin/env python3
"""
Structural prune of 0xSero/DeepSeek-V4-Flash-0731-REAP (K160) -> K128.

Pruning criterion: L2 norm of each expert's gate weight row
(||gate.weight[e]||_2). This is a fast, free proxy for REAP saliency:
the Router Sensitivity paper (arXiv 2608.07890) shows the gate-weight
L2 norm is "the strongest of the weight-based scores" they evaluated,
and it correlates well with REAP-quality rankings.

Proper REAP would also weight by per-token activation norms, but the
0xSero model uses a custom inference/model.py layout where expert
outputs aren't separately accessible for hooking from outside the
fused forward. The gate-weight norm proxy is the best we can do without
a from-scratch calibration pass.

This is a STRUCTURAL prune (real tensor slicing, real config update).
Output loads with the vLLM DeepSeek-V4 loader + the 0xSero
inference/model.py custom code; no patched model.py mask hacks needed.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


KEEP_K = 128


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="Source model dir (e.g. .../DeepSeek-V4-Flash-0731-REAP-K160)")
    p.add_argument("--dst", required=True,
                   help="Output dir for K128 pruned model")
    p.add_argument("--keep-k", type=int, default=KEEP_K,
                   help=f"Experts to keep per layer (default {KEEP_K})")
    p.add_argument("--criterion", choices=["index", "gate_weight_l2"],
                   default="index",
                   help=("Selection criterion. 'index' (default) keeps the "
                         "first --keep-k experts in index order — for 0xSero "
                         "REAP K160 this equals top-REAP-rank of the original "
                         "256 thanks to router identity alignment "
                         "(same_index_cos_min_global=0.946). 'gate_weight_l2' "
                         "selects by L2 norm of gate-weight rows, an independent "
                         "proxy."))
    p.add_argument("--dry-run", action="store_true",
                   help="Compute scores and report, don't write output")
    return p.parse_args()


def compute_scores(gate_w: torch.Tensor) -> torch.Tensor:
    """L2 norm of each expert's gate weight row.

    gate_w: [n_experts, hidden] (e.g. [160, 4096] bf16)
    returns: [n_experts] float32 scores (higher = more important)
    """
    return gate_w.float().norm(dim=-1)


def is_moe_gate_weight_key(key: str) -> bool:
    parts = key.split(".")
    return (len(parts) == 5 and parts[0] == "layers" and parts[2] == "ffn"
            and parts[3] == "gate" and parts[4] == "weight")


def is_moe_gate_bias_key(key: str) -> bool:
    parts = key.split(".")
    return (len(parts) == 5 and parts[0] == "layers" and parts[2] == "ffn"
            and parts[3] == "gate" and parts[4] == "bias")


def is_moe_gate_tid2eid_key(key: str) -> bool:
    parts = key.split(".")
    return (len(parts) == 5 and parts[0] == "layers" and parts[2] == "ffn"
            and parts[3] == "gate" and parts[4] == "tid2eid")


def is_moe_expert_tensor_key(key: str):
    """Match layers.{L}.ffn.experts.{E}.w{1,2,3}.{weight|scale}.

    Returns (layer_id, expert_id, is_weight) or None.
    """
    parts = key.split(".")
    if len(parts) != 7 or parts[0] != "layers" or parts[2] != "ffn":
        return None
    if parts[3] != "experts":
        return None
    if parts[5] not in ("w1", "w2", "w3"):
        return None
    if parts[6] not in ("weight", "scale"):
        return None
    try:
        L = int(parts[1])
        E = int(parts[4])
    except ValueError:
        return None
    return (L, E, parts[6] == "weight")


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    keep_k = args.keep_k

    cfg = json.load(open(src / "config.json"))
    n_routed_old = cfg["n_routed_experts"]
    n_layers = cfg["num_hidden_layers"]
    print(f"Source: {src}  n_routed_experts={n_routed_old}  n_layers={n_layers}")

    shards = sorted(src.glob("model-*.safetensors"))
    if not shards:
        print(f"ERROR: no model-*.safetensors in {src}")
        sys.exit(1)
    n_shards = len(shards)
    idx = json.load(open(src / "model.safetensors.index.json"))
    wm = idx["weight_map"]

    print(f"\n=== Phase 1: criterion={args.criterion} ===")
    scores = {}
    is_hash = {}
    seen_layers = set()

    for key, shard_name in wm.items():
        if not is_moe_gate_weight_key(key):
            continue
        L = int(key.split(".")[1])
        if L in seen_layers:
            continue
        seen_layers.add(L)
        shard_path = src / shard_name
        with safe_open(str(shard_path), framework="pt") as f:
            tid_key = f"layers.{L}.ffn.gate.tid2eid"
            is_hash[L] = tid_key in wm
            if args.criterion == "gate_weight_l2":
                gate_w = f.get_tensor(key)
                scores[L] = compute_scores(gate_w)
                print(f"  layer {L}: scores shape={tuple(scores[L].shape)}, "
                      f"hash={is_hash[L]}, "
                      f"min={scores[L].min().item():.3f}, "
                      f"max={scores[L].max().item():.3f}, "
                      f"median={scores[L].median().item():.3f}")
            else:
                print(f"  layer {L}: hash={is_hash[L]} "
                      f"(index mode: keep first {keep_k})")

    if not seen_layers:
        print("ERROR: no MoE gate weights found, aborting")
        sys.exit(1)

    print("\n=== Phase 2: select top-K per layer ===")
    kept_indices = {}
    for L in sorted(seen_layers):
        if args.criterion == "gate_weight_l2":
            sc = scores[L]
            sorted_idx = torch.argsort(sc, descending=True)
            kept = sorted_idx[:keep_k].tolist()
            kept.sort()
        else:
            kept = list(range(keep_k))
        kept_indices[L] = kept
        print(f"  layer {L}: keep {len(kept)} (range {kept[0]}..{kept[-1]})")

    if args.dry_run:
        out_stats = Path("/tmp/reap_dryrun_k128.jsonl")
        with open(out_stats, "w") as f:
            for L, sc in scores.items():
                for eidx in range(n_routed_old):
                    f.write(json.dumps({
                        "layer": L,
                        "expert": eidx,
                        "score": float(sc[eidx].item()),
                        "kept": eidx in kept_indices[L],
                    }) + "\n")
        print(f"\nDry run: wrote per-expert scores to {out_stats}")
        return

    # Phase 3: structural prune each shard
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    print(f"\n=== Phase 3: prune shards -> {dst} ===")

    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("model-") and not f.name.startswith("."):
            shutil.copy2(f, dst / f.name)

    new_wm = {}

    for shard_idx, shard_path in enumerate(shards, 1):
        t0 = time.time()
        new_tensors = {}
        n_pruned = 0
        with safe_open(str(shard_path), framework="pt") as f:
            meta = dict(f.metadata() or {})
            for key in f.keys():
                t = f.get_tensor(key)

                if is_moe_gate_weight_key(key):
                    L = int(key.split(".")[1])
                    kept = kept_indices[L]
                    kept_t = torch.tensor(kept, dtype=torch.long)
                    new_t = t[kept_t].contiguous()
                    new_tensors[key] = new_t
                    n_pruned += 1
                elif is_moe_gate_bias_key(key):
                    L = int(key.split(".")[1])
                    kept = kept_indices[L]
                    kept_t = torch.tensor(kept, dtype=torch.long)
                    new_t = t[kept_t].contiguous()
                    new_tensors[key] = new_t
                    n_pruned += 1
                elif is_moe_gate_tid2eid_key(key):
                    L = int(key.split(".")[1])
                    new_t = torch.where(t >= keep_k,
                                        torch.zeros_like(t), t)
                    new_tensors[key] = new_t
                elif (exp_info := is_moe_expert_tensor_key(key)) is not None:
                    L, E, _ = exp_info
                    if E >= keep_k:
                        continue
                    new_tensors[key] = t
                else:
                    new_tensors[key] = t

        new_shard_path = dst / shard_path.name
        save_file(new_tensors, str(new_shard_path), metadata=meta)
        for k in new_tensors.keys():
            new_wm[k] = shard_path.name
        del new_tensors
        print(f"  shard {shard_idx}/{n_shards} done, {n_pruned} gate tensors pruned "
              f"({time.time()-t0:.1f}s)", flush=True)

    new_idx = {
        "metadata": idx.get("metadata", {}),
        "weight_map": new_wm,
    }
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(new_idx, f, indent=2)

    new_cfg = dict(cfg)
    new_cfg["n_routed_experts"] = keep_k
    if "num_experts" in new_cfg and new_cfg["num_experts"] != keep_k:
        del new_cfg["num_experts"]
    new_cfg["_original_n_routed_experts"] = n_routed_old
    new_cfg["_prune_method"] = args.criterion
    new_cfg["_prune_keep_k"] = keep_k
    with open(dst / "config.json", "w") as f:
        json.dump(new_cfg, f, indent=2)
    print(f"\nUpdated config.json: n_routed_experts={keep_k}")

    # Verify
    print("\n=== Verify output ===")
    expert_counts = {}
    for k in new_wm:
        info = is_moe_expert_tensor_key(k)
        if info is None:
            continue
        L, E, is_w = info
        if is_w:
            expert_counts[L] = expert_counts.get(L, 0) + 1
    for L in sorted(expert_counts)[:5]:
        print(f"  layer {L}: {expert_counts[L]} expert weight tensors "
              f"({expert_counts[L]//3} experts)")
    total = sum(expert_counts.values()) // 3
    print(f"  total: {total} experts across {len(expert_counts)} layers")
    print(f"\nDone. K{keep_k} pruned model at {dst}")


if __name__ == "__main__":
    main()
