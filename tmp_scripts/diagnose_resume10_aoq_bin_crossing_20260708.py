#!/usr/bin/env python3
"""AOQ-style weight bin-crossing diagnostics for the resume10-to-81 branch."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


DEFAULT_CHECKPOINTS = {
    "ckpt10": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar",
    "phase2s": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon_vartrust_selective_gate_20260707/checkpoint-1.pth.tar",
    "phase2w": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar",
    "phase2z": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar",
    "phase2br_e3": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_5epoch_curve_from_phase2s_20260708/checkpoint-4.pth.tar",
}

TOP1 = {
    "ckpt10": 80.3640,
    "phase2s": 80.5220,
    "phase2w": 80.5400,
    "phase2z": 80.5540,
    "phase2br_e3": 80.5460,
}


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    result: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = key[len("module.") :] if key.startswith("module.") else key
        if value.is_floating_point():
            result[name] = value.detach().cpu()
    return result


def stage_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "features" and parts[1].isdigit() and parts[2].isdigit():
        return f"features.{parts[1]}.{parts[2]}"
    if len(parts) >= 2 and parts[0] == "features" and parts[1].isdigit():
        return f"features.{parts[1]}"
    return parts[0]


def weight_kind(name: str) -> str:
    if ".attn.q.weight" in name:
        return "attn_q"
    if ".attn.k.weight" in name:
        return "attn_k"
    if ".attn.v.weight" in name:
        return "attn_v"
    if ".attn.qkv.weight" in name:
        return "attn_qkv"
    if ".attn.proj.weight" in name:
        return "attn_proj"
    if ".mlp.fc1.weight" in name:
        return "mlp_fc1"
    if ".mlp.fc2.weight" in name:
        return "mlp_fc2"
    if name.endswith(".weight"):
        return "other_weight"
    return "other"


def module_from_weight_name(name: str) -> str:
    if name.endswith(".weight"):
        return name[: -len(".weight")]
    return name


def parent_module(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "features" and parts[1].isdigit() and parts[2].isdigit():
        if ".attn." in name:
            return f"features.{parts[1]}.{parts[2]}.attn"
        if ".mlp." in name:
            return f"features.{parts[1]}.{parts[2]}.mlp"
        return f"features.{parts[1]}.{parts[2]}"
    return module_from_weight_name(name).rsplit(".", 1)[0]


def statsq_scale(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 2:
        return 2.0 * weight.detach().abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    if weight.ndim == 3:
        return 2.0 * weight.detach().abs().mean(dim=-1, keepdim=True).mean(dim=0, keepdim=True).clamp_min(1e-12)
    flat = weight.detach().reshape(weight.shape[0], -1) if weight.ndim > 1 else weight.detach().reshape(1, -1)
    return 2.0 * flat.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)


def lsq_scale_from_state(state: Dict[str, torch.Tensor], weight_name: str) -> Optional[torch.Tensor]:
    module = module_from_weight_name(weight_name)
    key = f"{module}.lsqw_fn.s"
    scale = state.get(key)
    if scale is None:
        return None
    if scale.ndim == 1 and state[weight_name].ndim >= 2 and scale.numel() == state[weight_name].shape[0]:
        return scale.float().reshape(-1, *([1] * (state[weight_name].ndim - 1))).clamp_min(1e-12)
    return scale.float().clamp_min(1e-12)


def statsq_bins(weight: torch.Tensor, bit: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = statsq_scale(weight.float())
    n = float(2 ** (bit - 1))
    scaled = (weight.float() / scale).clamp(-1.0, 1.0 - 1e-6)
    pre_round = scaled * n - 0.5
    bins = torch.round(pre_round).clamp(-n, n - 1).to(torch.int16)
    center_dist = (pre_round - torch.round(pre_round)).abs().clamp(max=0.5)
    near_boundary = 0.5 - center_dist
    return bins, near_boundary, scale


def lsq_bins(weight: torch.Tensor, scale: torch.Tensor, bit: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    thd_neg = -2 ** (bit - 1)
    thd_pos = 2 ** (bit - 1) - 1
    normalized = weight.float() / scale.float().clamp_min(1e-12)
    clipped = normalized.clamp(float(thd_neg), float(thd_pos))
    bins = torch.round(clipped).clamp(thd_neg, thd_pos).to(torch.int16)
    center_dist = (clipped - torch.round(clipped)).abs().clamp(max=0.5)
    boundary_dist = 0.5 - center_dist
    return bins, boundary_dist, scale


def compute_qk_composite(state: Dict[str, torch.Tensor], module: str) -> Optional[torch.Tensor]:
    q = state.get(f"{module}.q.weight")
    k = state.get(f"{module}.k.weight")
    if q is None or k is None or q.ndim != 2 or k.ndim != 2 or q.shape != k.shape:
        return None
    dim = q.shape[0]
    if dim == 768:
        heads = 24
    elif dim == 384:
        heads = 12
    elif dim == 192:
        heads = 6
    elif dim == 96:
        heads = 3
    else:
        return None
    qh = q.float().reshape(heads, dim // heads, q.shape[1])
    kh = k.float().reshape(heads, dim // heads, k.shape[1])
    return (qh.transpose(-2, -1).contiguous() @ kh).reshape(heads * dim, dim)


def selected_weight_names(state: Dict[str, torch.Tensor], patterns: Sequence[str]) -> List[str]:
    names = []
    skip_tokens = (
        "norm",
        "head.",
        "relative_position_bias",
        "input_quant_fn",
        "quan_a_",
        "quant_x_4_qkv",
        "statsq_fn",
        "lsqw_fn",
        "qk_quant",
        "v_quant",
        "move_",
    )
    for name, tensor in state.items():
        if not name.endswith(".weight"):
            continue
        if tensor.ndim < 2:
            continue
        if any(token in name for token in skip_tokens):
            continue
        if patterns and not any(pattern in name for pattern in patterns):
            continue
        names.append(name)
    return sorted(names)


def tensor_metrics(
    before_bins: torch.Tensor,
    after_bins: torch.Tensor,
    before_boundary_dist: torch.Tensor,
    after_boundary_dist: torch.Tensor,
    before_scale: torch.Tensor,
    after_scale: torch.Tensor,
    near_margin: float,
) -> Dict[str, float]:
    if before_bins.shape != after_bins.shape:
        raise ValueError(f"bin shape mismatch: {tuple(before_bins.shape)} vs {tuple(after_bins.shape)}")
    numel = int(before_bins.numel())
    changed = before_bins != after_bins
    delta = (after_bins.int() - before_bins.int()).abs()
    before_near = before_boundary_dist.float() <= float(near_margin)
    after_near = after_boundary_dist.float() <= float(near_margin)
    either_near = before_near | after_near
    if before_scale.numel() == after_scale.numel():
        scale_ratio = float((after_scale.float().mean() / before_scale.float().mean().clamp_min(1e-12)).item())
    else:
        scale_ratio = float("nan")
    return {
        "numel": numel,
        "changed": int(changed.sum().item()),
        "changed_fraction": float(changed.float().mean().item()) if numel else 0.0,
        "mean_abs_bin_delta": float(delta.float().mean().item()) if numel else 0.0,
        "max_abs_bin_delta": int(delta.max().item()) if numel else 0,
        "before_near_fraction": float(before_near.float().mean().item()) if numel else 0.0,
        "after_near_fraction": float(after_near.float().mean().item()) if numel else 0.0,
        "changed_and_near_fraction": float((changed & either_near).float().mean().item()) if numel else 0.0,
        "changed_given_near": float((changed & either_near).sum().item() / max(1, either_near.sum().item())),
        "mean_before_boundary_dist": float(before_boundary_dist.float().mean().item()) if numel else 0.0,
        "mean_after_boundary_dist": float(after_boundary_dist.float().mean().item()) if numel else 0.0,
        "scale_ratio_after_over_before": scale_ratio,
    }


def add_aggregate(aggregates: Dict[Tuple[str, str, str], dict], key: Tuple[str, str, str], metrics: dict) -> None:
    rec = aggregates.setdefault(
        key,
        {
            "pairs": 0,
            "numel": 0,
            "changed": 0,
            "weighted_abs_bin_delta": 0.0,
            "weighted_before_near": 0.0,
            "weighted_after_near": 0.0,
            "weighted_changed_near": 0.0,
            "scale_ratios": [],
        },
    )
    numel = int(metrics["numel"])
    rec["pairs"] += 1
    rec["numel"] += numel
    rec["changed"] += int(metrics["changed"])
    rec["weighted_abs_bin_delta"] += float(metrics["mean_abs_bin_delta"]) * numel
    rec["weighted_before_near"] += float(metrics["before_near_fraction"]) * numel
    rec["weighted_after_near"] += float(metrics["after_near_fraction"]) * numel
    rec["weighted_changed_near"] += float(metrics["changed_and_near_fraction"]) * numel
    ratio = float(metrics["scale_ratio_after_over_before"])
    if math.isfinite(ratio):
        rec["scale_ratios"].append(ratio)


def finalize_aggregate(pair_label: str, key: Tuple[str, str, str], rec: dict, top1_delta: float) -> dict:
    numel = max(1, int(rec["numel"]))
    ratios = rec["scale_ratios"]
    return {
        "pair": pair_label,
        "group_type": key[0],
        "group": key[1],
        "quantizer": key[2],
        "pairs": rec["pairs"],
        "numel": rec["numel"],
        "changed": rec["changed"],
        "changed_fraction": rec["changed"] / numel,
        "mean_abs_bin_delta": rec["weighted_abs_bin_delta"] / numel,
        "before_near_fraction": rec["weighted_before_near"] / numel,
        "after_near_fraction": rec["weighted_after_near"] / numel,
        "changed_and_near_fraction": rec["weighted_changed_near"] / numel,
        "mean_scale_ratio": sum(ratios) / len(ratios) if ratios else float("nan"),
        "top1_delta": top1_delta,
    }


def analyze_pair(
    before_label: str,
    after_label: str,
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
    patterns: Sequence[str],
    near_margin: float,
) -> Tuple[List[dict], List[dict]]:
    pair_label = f"{before_label}->{after_label}"
    pair_rows: List[dict] = []
    aggregates: Dict[Tuple[str, str, str], dict] = {}
    top1_delta = TOP1.get(after_label, 0.0) - TOP1.get(before_label, 0.0)

    for name in selected_weight_names(before, patterns):
        if name not in after or before[name].shape != after[name].shape:
            continue
        module = module_from_weight_name(name)
        if f"{module}.lsqw_fn.s" in before and f"{module}.lsqw_fn.s" in after:
            before_scale = lsq_scale_from_state(before, name)
            after_scale = lsq_scale_from_state(after, name)
            if before_scale is None or after_scale is None:
                continue
            before_bins, before_boundary, before_scale_used = lsq_bins(before[name], before_scale)
            after_bins, after_boundary, after_scale_used = lsq_bins(after[name], after_scale)
            quantizer = "lsq_weight"
        else:
            before_bins, before_boundary, before_scale_used = statsq_bins(before[name])
            after_bins, after_boundary, after_scale_used = statsq_bins(after[name])
            quantizer = "statsq_like_weight"

        metrics = tensor_metrics(
            before_bins,
            after_bins,
            before_boundary,
            after_boundary,
            before_scale_used,
            after_scale_used,
            near_margin,
        )
        row = {
            "pair": pair_label,
            "before": before_label,
            "after": after_label,
            "top1_before": TOP1.get(before_label, 0.0),
            "top1_after": TOP1.get(after_label, 0.0),
            "top1_delta": top1_delta,
            "name": name,
            "module": module,
            "parent_module": parent_module(name),
            "stage": stage_name(name),
            "kind": weight_kind(name),
            "quantizer": quantizer,
            **metrics,
        }
        pair_rows.append(row)
        for group_type, group in (
            ("stage_kind", f"{stage_name(name)}|{weight_kind(name)}"),
            ("parent_kind", f"{parent_module(name)}|{weight_kind(name)}"),
            ("module", module),
            ("kind", weight_kind(name)),
        ):
            add_aggregate(aggregates, (group_type, group, quantizer), metrics)

    qk_modules = sorted(
        {
            name[: -len(".q.weight")]
            for name in before
            if name.endswith(".q.weight") and (not patterns or any(pattern in name for pattern in patterns))
        }
    )
    for module in qk_modules:
        before_qk = compute_qk_composite(before, module)
        after_qk = compute_qk_composite(after, module)
        if before_qk is None or after_qk is None or before_qk.shape != after_qk.shape:
            continue
        before_bins, before_boundary, before_scale_used = statsq_bins(before_qk)
        after_bins, after_boundary, after_scale_used = statsq_bins(after_qk)
        metrics = tensor_metrics(
            before_bins,
            after_bins,
            before_boundary,
            after_boundary,
            before_scale_used,
            after_scale_used,
            near_margin,
        )
        name = f"{module}.qk_composite"
        row = {
            "pair": pair_label,
            "before": before_label,
            "after": after_label,
            "top1_before": TOP1.get(before_label, 0.0),
            "top1_after": TOP1.get(after_label, 0.0),
            "top1_delta": top1_delta,
            "name": name,
            "module": module,
            "parent_module": module,
            "stage": stage_name(module),
            "kind": "attn_qk_composite",
            "quantizer": "qk_statsq_composite",
            **metrics,
        }
        pair_rows.append(row)
        for group_type, group in (
            ("stage_kind", f"{stage_name(module)}|attn_qk_composite"),
            ("parent_kind", f"{module}|attn_qk_composite"),
            ("module", module),
            ("kind", "attn_qk_composite"),
        ):
            add_aggregate(aggregates, (group_type, group, "qk_statsq_composite"), metrics)

    aggregate_rows = [finalize_aggregate(pair_label, key, rec, top1_delta) for key, rec in aggregates.items()]
    pair_rows.sort(key=lambda item: (-item["changed_fraction"], -item["changed_and_near_fraction"], item["name"]))
    aggregate_rows.sort(key=lambda item: (-item["changed_fraction"], -item["changed_and_near_fraction"], item["group_type"], item["group"]))
    return pair_rows, aggregate_rows


def write_tsv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_pairs(spec: str) -> List[Tuple[str, str]]:
    result = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "->" not in item:
            raise ValueError(f"pair must use before->after syntax: {item}")
        before, after = (part.strip() for part in item.split("->", 1))
        result.append((before, after))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_aoq_bin_crossing_20260708")
    parser.add_argument("--pairs", default="ckpt10->phase2s,phase2s->phase2w,phase2w->phase2z,phase2z->phase2br_e3")
    parser.add_argument("--module-patterns", default="features.5.5,features.7.1")
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument("--topn", type=int, default=200)
    for label, path in DEFAULT_CHECKPOINTS.items():
        parser.add_argument(f"--{label}-checkpoint", default=path)
        parser.add_argument(f"--{label}-top1", type=float, default=TOP1[label])
    args = parser.parse_args()

    checkpoint_paths = {label: getattr(args, f"{label}_checkpoint") for label in DEFAULT_CHECKPOINTS}
    for label in DEFAULT_CHECKPOINTS:
        TOP1[label] = float(getattr(args, f"{label}_top1"))
    states = {label: load_state_dict(path) for label, path in checkpoint_paths.items()}
    pairs = parse_pairs(args.pairs)
    patterns = tuple(part.strip() for part in args.module_patterns.split(",") if part.strip())

    all_pair_rows: List[dict] = []
    all_aggregate_rows: List[dict] = []
    result = {
        "checkpoints": checkpoint_paths,
        "top1": {label: TOP1[label] for label in checkpoint_paths},
        "pairs": [f"{before}->{after}" for before, after in pairs],
        "module_patterns": patterns,
        "near_margin": args.near_margin,
        "pair_rows_top": {},
        "aggregate_rows_top": {},
    }
    for before, after in pairs:
        if before not in states or after not in states:
            raise KeyError(f"unknown pair labels: {before}->{after}")
        pair_rows, aggregate_rows = analyze_pair(before, after, states[before], states[after], patterns, args.near_margin)
        all_pair_rows.extend(pair_rows)
        all_aggregate_rows.extend(aggregate_rows)
        label = f"{before}->{after}"
        result["pair_rows_top"][label] = pair_rows[: args.topn]
        result["aggregate_rows_top"][label] = aggregate_rows[: args.topn]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "pair_bin_crossing.tsv", all_pair_rows)
    write_tsv(out_dir / "aggregate_bin_crossing.tsv", all_aggregate_rows)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {out_dir}")
    for pair in result["pairs"]:
        rows = [row for row in all_aggregate_rows if row["pair"] == pair and row["group_type"] == "stage_kind"]
        print(pair)
        for row in rows[:8]:
            print(
                "  {group} {quantizer} changed={changed_fraction:.6f} "
                "near_after={after_near_fraction:.6f} top1_delta={top1_delta:.4f}".format(**row)
            )


if __name__ == "__main__":
    main()
