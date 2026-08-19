#!/usr/bin/env python3
"""Compare Phase 2S -> source parameter deltas with and without feature output."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


DEFAULT_CHECKPOINTS = {
    "base": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar",
    "phase2w": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar",
    "nofeat": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_nofeatout_source_gate_20260708/checkpoint-2.pth.tar",
}

TOP1 = {
    "base": 80.5220,
    "phase2w": 80.5400,
    "nofeat": 80.4640,
}


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    result = {}
    for key, value in state.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        name = key[len("module.") :] if key.startswith("module.") else key
        result[name] = value.detach().cpu()
    return result


def param_kind(name: str) -> str:
    if "quan_a_softmax" in name:
        return "softmax_quant"
    if any(token in name for token in ("input_quant_fn", "quant_x_4_qkv", "quan_a_q", "quan_a_k", "quan_a_v", "quan_a_qkx")):
        return "act_quant"
    if any(token in name for token in ("lsqw_fn", "statsq_fn", "qk_quant", "v_quant")):
        return "weight_quant"
    if ".move_v" in name:
        return "move_v_shift"
    if ".attn.proj.move_" in name:
        return "proj_move_shift"
    if ".move_" in name or name.startswith("move_"):
        return "move_shift"
    if ".attn.proj." in name:
        return "attn_proj"
    if ".attn.q." in name or ".attn.k." in name or ".attn.v." in name or ".attn.qkv." in name:
        return "attn_qkv"
    if ".mlp." in name or ".fc1." in name or ".fc2." in name:
        return "mlp"
    if ".norm" in name or name.startswith("norm."):
        return "norm"
    if "relative_position_bias" in name:
        return "relpos"
    if name.startswith("head."):
        return "head"
    return "other"


def stage_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "features" and parts[1].isdigit() and parts[2].isdigit():
        return f"features.{parts[1]}.{parts[2]}"
    if len(parts) >= 2 and parts[0] == "features" and parts[1].isdigit():
        return f"features.{parts[1]}"
    return parts[0]


def module_name(name: str) -> str:
    parts = name.split(".")
    markers = (
        "input_quant_fn",
        "quant_x_4_qkv",
        "quan_a_q_fn",
        "quan_a_k_fn",
        "quan_a_v_fn",
        "quan_a_qkx_fn",
        "quan_a_softmax_fn",
        "lsqw_fn",
        "statsq_fn",
        "qk_quant",
        "v_quant",
    )
    for marker in markers:
        if marker in parts:
            return ".".join(parts[: parts.index(marker) + 1])
    if len(parts) > 1:
        return ".".join(parts[:-1])
    return name


def group_keys(name: str):
    return {
        "kind": (param_kind(name),),
        "stage_kind": (stage_name(name), param_kind(name)),
        "module_kind": (module_name(name), param_kind(name)),
        "param": (name,),
    }


def empty_group():
    return {
        "phase2w_sq": 0.0,
        "cmp_sq": 0.0,
        "dot": 0.0,
        "extra_sq": 0.0,
        "numel": 0,
        "params": 0,
    }


def tensor_stats(base: torch.Tensor, phase2w: torch.Tensor, cmp_tensor: torch.Tensor) -> Tuple[float, float, float, float]:
    base_f = base.float().reshape(-1)
    phase2w_delta = phase2w.float().reshape(-1) - base_f
    cmp_delta = cmp_tensor.float().reshape(-1) - base_f
    extra_delta = cmp_delta - phase2w_delta
    return (
        float(phase2w_delta.dot(phase2w_delta).item()),
        float(cmp_delta.dot(cmp_delta).item()),
        float(phase2w_delta.dot(cmp_delta).item()),
        float(extra_delta.dot(extra_delta).item()),
    )


def finalize_group(group_type: str, key: Iterable[str], rec: dict, cmp_label: str, top1: Dict[str, float]) -> dict:
    phase2w_norm = rec["phase2w_sq"] ** 0.5
    cmp_norm = rec["cmp_sq"] ** 0.5
    extra_norm = rec["extra_sq"] ** 0.5
    denom = max(phase2w_norm * cmp_norm, 1e-12)
    return {
        "group_type": group_type,
        "key": "|".join(key),
        "params": rec["params"],
        "numel": rec["numel"],
        "phase2w_delta_l2": phase2w_norm,
        "cmp_label": cmp_label,
        "cmp_delta_l2": cmp_norm,
        "cmp_over_phase2w": cmp_norm / max(phase2w_norm, 1e-12),
        "delta_cosine_to_phase2w": rec["dot"] / denom,
        "extra_l2_vs_phase2w": extra_norm,
        "extra_over_phase2w": extra_norm / max(phase2w_norm, 1e-12),
        "phase2w_top1": top1["phase2w"],
        "cmp_top1": top1[cmp_label],
        "delta_top1": top1[cmp_label] - top1["phase2w"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_source_delta_20260708")
    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument("--base-checkpoint", default=DEFAULT_CHECKPOINTS["base"])
    parser.add_argument("--phase2w-checkpoint", default=DEFAULT_CHECKPOINTS["phase2w"])
    parser.add_argument("--cmp-label", default="nofeat")
    parser.add_argument("--cmp-checkpoint", default=DEFAULT_CHECKPOINTS["nofeat"])
    parser.add_argument("--base-top1", type=float, default=TOP1["base"])
    parser.add_argument("--phase2w-top1", type=float, default=TOP1["phase2w"])
    parser.add_argument("--cmp-top1", type=float, default=TOP1["nofeat"])
    args = parser.parse_args()

    checkpoints = {
        "base": args.base_checkpoint,
        "phase2w": args.phase2w_checkpoint,
        args.cmp_label: args.cmp_checkpoint,
    }
    top1 = {
        "base": args.base_top1,
        "phase2w": args.phase2w_top1,
        args.cmp_label: args.cmp_top1,
    }
    states = {label: load_state_dict(path) for label, path in checkpoints.items()}
    base = states["base"]
    phase2w = states["phase2w"]
    cmp_state = states[args.cmp_label]
    groups = {group_type: defaultdict(empty_group) for group_type in ("kind", "stage_kind", "module_kind", "param")}

    for name, base_tensor in base.items():
        phase2w_tensor = phase2w.get(name)
        cmp_tensor = cmp_state.get(name)
        if phase2w_tensor is None or cmp_tensor is None:
            continue
        if phase2w_tensor.shape != base_tensor.shape or cmp_tensor.shape != base_tensor.shape:
            continue
        phase2w_sq, cmp_sq, dot, extra_sq = tensor_stats(base_tensor, phase2w_tensor, cmp_tensor)
        for group_type, key in group_keys(name).items():
            rec = groups[group_type][key]
            rec["numel"] += int(base_tensor.numel())
            rec["params"] += 1
            rec["phase2w_sq"] += phase2w_sq
            rec["cmp_sq"] += cmp_sq
            rec["dot"] += dot
            rec["extra_sq"] += extra_sq

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoints": checkpoints,
        "top1": top1,
        "cmp_label": args.cmp_label,
        "rows": {},
    }
    fields = [
        "group_type",
        "key",
        "params",
        "numel",
        "phase2w_delta_l2",
        "cmp_label",
        "cmp_delta_l2",
        "cmp_over_phase2w",
        "delta_cosine_to_phase2w",
        "extra_l2_vs_phase2w",
        "extra_over_phase2w",
        "phase2w_top1",
        "cmp_top1",
        "delta_top1",
    ]
    for group_type, group in groups.items():
        rows = [finalize_group(group_type, key, rec, args.cmp_label, top1) for key, rec in group.items()]
        rows.sort(key=lambda item: (-item["extra_over_phase2w"], item["delta_cosine_to_phase2w"]))
        result["rows"][group_type] = rows[: args.topn]
        path = out_dir / f"{group_type}_delta.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
