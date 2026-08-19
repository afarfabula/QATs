#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["all", "nonquant"],
        default="all",
        help="all: average every floating tensor; nonquant: average normal weights but keep quantizer state from the last checkpoint",
    )
    parser.add_argument("checkpoints", nargs="+")
    return parser.parse_args()


def is_quant_state_key(key: str) -> bool:
    quant_markers = (
        "quant",
        "lsq",
        "clip",
        "bound",
        "signed",
        "calib",
        "observer",
        "scale",
        "zero_point",
    )
    lowered = key.lower()
    return any(marker in lowered for marker in quant_markers)


def main():
    args = parse_args()
    paths = [Path(p) for p in args.checkpoints]
    if len(paths) < 2:
        raise SystemExit("need at least two checkpoints")
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"checkpoint not found: {path}")

    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    state_dicts = [ckpt["state_dict"] for ckpt in checkpoints]
    keys = list(state_dicts[0].keys())
    if any(list(sd.keys()) != keys for sd in state_dicts[1:]):
        raise SystemExit("checkpoint state_dict keys do not match")

    avg_state = {}
    for key in keys:
        values = [sd[key] for sd in state_dicts]
        first = values[0]
        should_average = (
            torch.is_tensor(first)
            and first.is_floating_point()
            and not (args.mode == "nonquant" and is_quant_state_key(key))
        )
        if should_average:
            avg_state[key] = torch.stack([v.float() for v in values], dim=0).mean(dim=0).to(dtype=first.dtype)
        else:
            avg_state[key] = values[-1].clone() if torch.is_tensor(values[-1]) else values[-1]

    out_ckpt = dict(checkpoints[-1])
    out_ckpt["state_dict"] = avg_state
    out_ckpt["averaged_from"] = [str(path) for path in paths]
    out_ckpt["average_mode"] = args.mode
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out_path)
    print(f"wrote {out_path}")
    print("averaged_from:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
