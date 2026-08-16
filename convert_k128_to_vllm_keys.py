#!/usr/bin/env python3
"""
Convert K128 model from 0xSero naming (w1/w2/w3) to vLLM DeepSeek-V4 naming
(gate_up_proj/down_proj).

0xSero raw keys (per layer):
  ffn.experts.{E}.w1.{weight,scale}     ->  ffn.experts.{E}.gate_up_proj.{weight,weight_scale}
  ffn.experts.{E}.w2.{weight,scale}     ->  ffn.experts.{E}.down_proj.{weight,weight_scale}
  ffn.experts.{E}.w3.{weight,scale}     ->  (concat into w1 -> gate_up_proj)
  ffn.shared_experts.w1.{weight,scale}  ->  ffn.shared_experts.gate_up_proj.{weight,weight_scale}
  ffn.shared_experts.w2.{weight,scale}  ->  ffn.shared_experts.down_proj.{weight,weight_scale}
  ffn.shared_experts.w3.{weight,scale}  ->  (concat into w1 -> gate_up_proj)
  ffn.gate.tid2eid (hash layers only)   ->  unchanged

Concat axis: w1 and w3 are [out_features, in_features]; concat along out_features.
Scale tensors: same shape as the per-output-block scale grid; concat along the
same axis.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


W_RE = re.compile(r"^layers\.(\d+)\.ffn\.(experts\.(\d+)|shared_experts)\.w([123])\.(weight|scale)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists():
        import subprocess
        subprocess.run(["rm", "-rf", str(dst)], check=True)
    dst.mkdir(parents=True)

    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("model-") and not f.name.startswith("."):
            shutil.copy2(f, dst / f.name)

    idx = json.load(open(src / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    shards = sorted({shard for shard in wm.values()})

    n_renamed = 0
    n_concat = 0
    n_unchanged = 0

    for shard_idx, shard in enumerate(shards, 1):
        t0 = time.time()
        shard_path = src / shard
        out_path = dst / shard
        new_tensors = {}
        concat_buffer = {}

        with safe_open(str(shard_path), framework="pt") as f:
            meta = dict(f.metadata() or {})
            keys = list(f.keys())
            for key in keys:
                m = W_RE.match(key)
                if not m:
                    new_tensors[key] = f.get_tensor(key)
                    n_unchanged += 1
                    continue

                layer_id = m.group(1)
                expert_part = m.group(2)
                w_idx = m.group(4)
                field = m.group(5)
                if expert_part == "shared_experts":
                    new_key = f"layers.{layer_id}.ffn.shared_experts"
                else:
                    new_key = f"layers.{layer_id}.ffn.{expert_part}"
                if w_idx in ("1", "3"):
                    new_key += ".gate_up_proj"
                else:
                    new_key += ".down_proj"
                new_key += f".{'weight' if field == 'weight' else 'weight_scale'}"

                tensor = f.get_tensor(key)
                if w_idx in ("1", "3"):
                    if expert_part == "shared_experts":
                        buf_key = f"layers.{layer_id}.ffn.shared_experts.gate_up_proj"
                    else:
                        buf_key = f"layers.{layer_id}.ffn.{expert_part}.gate_up_proj"
                    if field == "weight":
                        buf_key_full = buf_key + ".weight"
                    else:
                        buf_key_full = buf_key + ".weight_scale"
                    if buf_key_full not in concat_buffer:
                        concat_buffer[buf_key_full] = {1: None, 3: None}
                    concat_buffer[buf_key_full][int(w_idx)] = tensor
                else:
                    new_tensors[new_key] = tensor
                    n_renamed += 1

        for new_key, parts in concat_buffer.items():
            w1_t = parts[1]
            w3_t = parts[3]
            if w1_t is None or w3_t is None:
                print(f"WARN: missing w1/w3 for {new_key}", file=sys.stderr)
                continue
            new_tensors[new_key] = torch.cat([w1_t, w3_t], dim=0).contiguous()
            n_concat += 1

        save_file(new_tensors, str(out_path), metadata=meta)
        for k in new_tensors.keys():
            wm[k] = shard
        del new_tensors, concat_buffer
        print(f"  shard {shard_idx}/{len(shards)} done ({time.time()-t0:.1f}s) "
              f"[renamed={n_renamed} concat={n_concat} passthrough={n_unchanged}]",
              flush=True)

    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": idx.get("metadata", {}), "weight_map": wm}, f, indent=2)
    print(f"\nDone. Output at {dst}")


if __name__ == "__main__":
    main()
