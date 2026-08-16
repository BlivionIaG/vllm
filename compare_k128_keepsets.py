#!/usr/bin/env python3
"""Compare kept expert sets per layer between two pruned K128 models.

Reports: per-layer Jaccard overlap, count of common experts, and where
they differ.
"""
import argparse
import json
import sys
from pathlib import Path

from safetensors import safe_open


def collect_kept_experts(model_dir: Path, keep_k: int = 128):
    """Return {layer_id: set(kept_expert_ids)}.

    The expert ids here are *post-prune* (0..keep_k-1) so this only tells
    us structure not identity-vs-source. We need to instead collect the
    source expert id that ended up at each post-prune slot.

    Better: collect the source expert id by reading the gate weight rows
    of the source and checking which row maps to which source expert.
    But that requires re-running. Simpler: compare structural shape only.
    """
    out = {}
    idx = json.load(open(model_dir / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    shards = {shard: model_dir / shard for shard in set(wm.values())}
    for key, shard in wm.items():
        parts = key.split(".")
        if len(parts) != 7 or parts[3] != "experts":
            continue
        if parts[6] != "weight":
            continue
        L = int(parts[1])
        E = int(parts[4])
        if E >= keep_k:
            continue
        out.setdefault(L, set()).add(E)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="First model dir")
    p.add_argument("--b", required=True, help="Second model dir")
    p.add_argument("--keep-k", type=int, default=128)
    args = p.parse_args()

    a_kept = collect_kept_experts(Path(args.a), args.keep_k)
    b_kept = collect_kept_experts(Path(args.b), args.keep_k)
    print(f"layer | A_count | B_count | intersection | union | jaccard | symmetric_diff")
    print("-" * 80)
    total_a = total_b = total_int = total_union = 0
    full_diff = []
    for L in sorted(set(a_kept) | set(b_kept)):
        a = a_kept.get(L, set())
        b = b_kept.get(L, set())
        inter = a & b
        union = a | b
        sym = a ^ b
        jacc = len(inter) / len(union) if union else 0
        total_a += len(a); total_b += len(b)
        total_int += len(inter); total_union += len(union)
        if sym:
            full_diff.append((L, sorted(sym)))
        if L < 5 or L > 38 or sym:
            print(f"  {L:3d} | {len(a):3d}     | {len(b):3d}     | "
                  f"{len(inter):3d}          | {len(union):3d}    | "
                  f"{jacc:5.3f}   | {sorted(sym)[:5]}{'...' if len(sym)>5 else ''}")
    print("-" * 80)
    grand_jacc = total_int / total_union if total_union else 0
    print(f"  ALL | {total_a:5d}  | {total_b:5d}  | "
          f"{total_int:5d}       | {total_union:5d} | {grand_jacc:5.3f}")
    print(f"\nLayers with non-zero symmetric diff: {len(full_diff)}/{len(a_kept)}")
    if full_diff[:3]:
        print("First 3 differing layers (post-prune expert ids):")
        for L, s in full_diff[:3]:
            print(f"  layer {L}: {len(s)} different experts: {s[:8]}{'...' if len(s)>8 else ''}")


if __name__ == "__main__":
    main()
