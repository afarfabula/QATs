#!/usr/bin/env python3
"""Revive dead post-softmax LSQ scales in an evaluation checkpoint.

The native quantizer clamps non-positive alpha to 1e-5 at forward time, but
clip() blocks gradients there. This diagnostic replaces those dead entries with
a positive layer statistic so the accuracy effect of the dead region can be
measured before changing the training parameterization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stat", choices=["median", "mean", "q25"], default="median")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    changed_tensors = 0
    changed_values = 0
    for name, tensor in state.items():
        if not name.endswith("quan_a_softmax_fn.s") or not torch.is_tensor(tensor):
            continue
        positive = tensor.float() > 1e-5
        if not bool(positive.any()):
            raise RuntimeError(f"no positive scale remains in {name}")
        values = tensor.float()[positive]
        if args.stat == "mean":
            replacement = values.mean()
        elif args.stat == "q25":
            replacement = values.quantile(0.25)
        else:
            replacement = values.median()
        dead = ~positive.to(device=tensor.device)
        count = int(dead.sum().item())
        if count:
            tensor[dead] = replacement.to(device=tensor.device, dtype=tensor.dtype)
            changed_tensors += 1
            changed_values += count

    if changed_values == 0:
        print("no dead attention-softmax scales found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(
        f"revived {changed_values} scales in {changed_tensors} tensors "
        f"with layer {args.stat}: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
