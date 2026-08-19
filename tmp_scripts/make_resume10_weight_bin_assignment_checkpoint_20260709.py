#!/usr/bin/env python3
"""Create a single-checkpoint masked weight-bin assignment candidate."""

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_checkpoint(path: str) -> dict:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    return torch.load(path, map_location="cpu", weights_only=False)


def state_dict_of(checkpoint: dict) -> Dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state_dict")
    return state


def get_tensor(state: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = state.get(key)
    if value is None:
        value = state.get(f"module.{key}")
    if value is None or not torch.is_tensor(value):
        raise KeyError(key)
    return value


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["changed_bin", "unchanged_bin", "up_bin", "down_bin", "toward_zero", "away_from_zero", "central_from", "central_to"],
        default="changed_bin",
    )
    parser.add_argument("--include-move", default="1")
    parser.add_argument("--qmin", type=int, default=-8)
    parser.add_argument("--qmax", type=int, default=7)
    args = parser.parse_args()

    base = load_checkpoint(args.base)
    donor = load_checkpoint(args.donor)
    base_state = state_dict_of(base)
    donor_state = state_dict_of(donor)
    output = copy.deepcopy(base)
    output_state = state_dict_of(output)

    weight_key = f"{args.module}.weight"
    scale_key = f"{args.module}.lsqw_fn.s"
    base_weight = get_tensor(base_state, weight_key).float()
    donor_weight = get_tensor(donor_state, weight_key).float()
    scale = get_tensor(base_state, scale_key).float()
    if scale.ndim == 1 and base_weight.ndim >= 2:
        scale = scale.view(-1, *([1] * (base_weight.ndim - 1)))
    base_bin = torch.clamp(torch.round(base_weight / scale), args.qmin, args.qmax)
    donor_bin = torch.clamp(torch.round(donor_weight / scale), args.qmin, args.qmax)
    delta_bin = donor_bin - base_bin
    changed_mask = delta_bin.ne(0)
    if args.mode == "changed_bin":
        mask = changed_mask
    elif args.mode == "unchanged_bin":
        mask = ~changed_mask
    elif args.mode == "up_bin":
        mask = delta_bin.gt(0)
    elif args.mode == "down_bin":
        mask = delta_bin.lt(0)
    elif args.mode == "toward_zero":
        mask = changed_mask & donor_bin.abs().lt(base_bin.abs())
    elif args.mode == "away_from_zero":
        mask = changed_mask & donor_bin.abs().gt(base_bin.abs())
    elif args.mode == "central_from":
        mask = changed_mask & base_bin.abs().le(1)
    else:
        mask = changed_mask & donor_bin.abs().le(1)
    if int(mask.sum().item()) == 0:
        raise ValueError(f"empty assignment mask for mode={args.mode}")

    output_weight = get_tensor(output_state, weight_key).clone()
    output_weight[mask] = get_tensor(donor_state, weight_key).to(output_weight.dtype)[mask]
    actual_weight_key = weight_key if weight_key in output_state else f"module.{weight_key}"
    output_state[actual_weight_key] = output_weight

    copied = [actual_weight_key]
    if parse_bool(args.include_move):
        for suffix in ("move_b4.bias", "move_aft.bias"):
            key = f"{args.module}.{suffix}"
            donor_value = get_tensor(donor_state, key)
            actual_key = key if key in output_state else f"module.{key}"
            output_state[actual_key] = donor_value.detach().clone()
            copied.append(actual_key)

    output["state_dict"] = output_state
    meta = output.setdefault("weight_bin_assignment_20260709", {})
    meta.update(
        {
            "base": args.base,
            "donor": args.donor,
            "module": args.module,
            "mode": args.mode,
            "include_move": parse_bool(args.include_move),
            "changed_bin_elements": int(changed_mask.sum().item()),
            "up_bin_elements": int(delta_bin.gt(0).sum().item()),
            "down_bin_elements": int(delta_bin.lt(0).sum().item()),
            "toward_zero_elements": int((changed_mask & donor_bin.abs().lt(base_bin.abs())).sum().item()),
            "away_from_zero_elements": int((changed_mask & donor_bin.abs().gt(base_bin.abs())).sum().item()),
            "central_from_elements": int((changed_mask & base_bin.abs().le(1)).sum().item()),
            "central_to_elements": int((changed_mask & donor_bin.abs().le(1)).sum().item()),
            "assigned_weight_elements": int(mask.sum().item()),
            "total_weight_elements": int(mask.numel()),
            "assigned_fraction": float(mask.float().mean().item()),
            "copied_tensors": copied,
        }
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    torch.save(output, tmp_path)
    tmp_path.replace(out_path)
    print(json.dumps({"output": str(out_path), **meta}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
