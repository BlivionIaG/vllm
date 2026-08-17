#!/usr/bin/env python3
"""Compare which K160 source expert IDs were kept by each prune method.

The index-mode prune keeps K160 experts [0..127] in every layer.
The gate_weight_l2 prune keeps a per-layer argmax-of-L2 subset.

We read the gate_weight_l2 dry-run output (/tmp/reap_dryrun_k128.jsonl)
to recover its per-layer kept set, and compare to {0..127}.
"""
import argparse
import json
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--l2-dryrun", default="/tmp/reap_dryrun_k128.jsonl",
                   help="Path to gate_weight_l2 dry-run jsonl")
    args = p.parse_args()

    l2_kept = defaultdict(set)
    with open(args.l2_dryrun) as f:
        for line in f:
            rec = json.loads(line)
            if rec["kept"]:
                l2_kept[rec["layer"]].add(rec["expert"])

    print(f"layer | l2_kept (sample)                              | symdiff_vs_[0..127]")
    print("-" * 100)
    total_l2 = 0
    total_sym = 0
    n_diff_layers = 0
    for L in sorted(l2_kept):
        a = l2_kept[L]
        b = set(range(128))
        sym = a ^ b
        total_l2 += len(a)
        total_sym += len(sym)
        if sym:
            n_diff_layers += 1
        if L < 5 or L > 38 or sym:
            in_l2_only = sorted(sym - b)
            in_index_only = sorted((b - a) & sym)
            print(f"  {L:3d} | n={len(a):3d}  e.g. {sorted(a)[:5]}  | "
                  f"symdiff_n={len(sym):3d}  "
                  f"{'l2-only: ' + str(in_l2_only[:5]) if in_l2_only else ''}"
                  f"{' index-only: ' + str(in_index_only[:5]) if in_index_only else ''}")
    print("-" * 100)
    print(f"  ALL | l2 total kept: {total_l2}  symdiff vs [0..127]: {total_sym}")
    print(f"  {n_diff_layers}/{len(l2_kept)} layers differ in kept set")
    if total_sym:
        print(f"  Mean per-layer symdiff: {total_sym/len(l2_kept):.1f}/128 "
              f"({100*total_sym/(128*len(l2_kept)):.1f}% of all slots differ)")


if __name__ == "__main__":
    main()
