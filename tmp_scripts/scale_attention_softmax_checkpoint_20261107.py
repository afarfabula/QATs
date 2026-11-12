#!/usr/bin/env python3
"""Scale learned attention-softmax LSQ steps in an existing QAT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Only scale positive stored steps; leave parameters already clamped at 1e-5 unchanged.",
    )
    args = parser.parse_args()

    if args.factor <= 0:
        raise ValueError("--factor must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    changed = 0
    for name, tensor in state.items():
        if not name.endswith("quan_a_softmax_fn.s") or not torch.is_tensor(tensor):
            continue
        if args.positive_only:
            positive = tensor > 1e-5
            tensor[positive] *= args.factor
        else:
            tensor.clamp_min_(1e-5).mul_(args.factor)
        changed += 1

    if changed == 0:
        raise RuntimeError("no attention softmax quantizer scales found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"scaled {changed} attention softmax quantizers by {args.factor}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
