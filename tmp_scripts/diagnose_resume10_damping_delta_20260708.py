#!/usr/bin/env python3
"""Compare parameter delta direction for damping gates in resume10-to-81."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


DEFAULT_CHECKPOINTS = {
    "base": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar",
    "best250": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar",
    "highdrift_damp": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_attn55_71_250upd_highdrift_damp05_gate_20260708/checkpoint-3.pth.tar",
    "movev_damp": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_250upd_movev_damp05_gate_20260708/checkpoint-3.pth.tar",
}

TOP1 = {
    "base": 80.5400,
    "best250": 80.5540,
    "highdrift_damp": 80.5100,
    "movev_damp": 80.5100,
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
        "best_sq": 0.0,
        "cmp_sq": defaultdict(float),
        "dot": defaultdict(float),
        "extra_sq": defaultdict(float),
        "numel": 0,
        "params": 0,
    }


def tensor_stats(base: torch.Tensor, best: torch.Tensor, cmp: torch.Tensor) -> Tuple[float, float, float, float]:
    base_f = base.float().reshape(-1)
    best_delta = best.float().reshape(-1) - base_f
    cmp_delta = cmp.float().reshape(-1) - base_f
    extra_delta = cmp_delta - best_delta
    return (
        float(best_delta.dot(best_delta).item()),
        float(cmp_delta.dot(cmp_delta).item()),
        float(best_delta.dot(cmp_delta).item()),
        float(extra_delta.dot(extra_delta).item()),
    )


def finalize_group(group_type: str, key: Iterable[str], rec: dict, labels: Iterable[str]) -> list:
    rows = []
    best_norm = rec["best_sq"] ** 0.5
    for label in labels:
        cmp_norm = rec["cmp_sq"][label] ** 0.5
        extra_norm = rec["extra_sq"][label] ** 0.5
        denom = max(best_norm * cmp_norm, 1e-12)
        rows.append(
            {
                "group_type": group_type,
                "key": "|".join(key),
                "cmp": label,
                "params": rec["params"],
                "numel": rec["numel"],
                "best_delta_l2": best_norm,
                "cmp_delta_l2": cmp_norm,
                "cmp_over_best": cmp_norm / max(best_norm, 1e-12),
                "delta_cosine_to_best": rec["dot"][label] / denom,
                "extra_l2_vs_best": extra_norm,
                "extra_over_best": extra_norm / max(best_norm, 1e-12),
                "best_top1": TOP1["best250"],
                "cmp_top1": TOP1[label],
                "delta_top1_vs_best": TOP1[label] - TOP1["best250"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_damping_delta_20260708")
    parser.add_argument("--topn", type=int, default=160)
    args = parser.parse_args()

    states = {label: load_state_dict(path) for label, path in DEFAULT_CHECKPOINTS.items()}
    base = states["base"]
    best = states["best250"]
    cmp_labels = ["highdrift_damp", "movev_damp"]
    groups = {group_type: defaultdict(empty_group) for group_type in ("kind", "stage_kind", "module_kind", "param")}

    for name, base_tensor in base.items():
        best_tensor = best.get(name)
        if best_tensor is None or best_tensor.shape != base_tensor.shape:
            continue
        cmp_tensors = {
            label: states[label].get(name)
            for label in cmp_labels
            if states[label].get(name) is not None and states[label][name].shape == base_tensor.shape
        }
        if not cmp_tensors:
            continue
        keys = group_keys(name)
        for label, cmp_tensor in cmp_tensors.items():
            best_sq, cmp_sq, dot, extra_sq = tensor_stats(base_tensor, best_tensor, cmp_tensor)
            for group_type, key in keys.items():
                rec = groups[group_type][key]
                if rec["params"] == 0 or label == cmp_labels[0]:
                    rec["numel"] += int(base_tensor.numel()) if label == cmp_labels[0] else 0
                    rec["params"] += 1 if label == cmp_labels[0] else 0
                    rec["best_sq"] += best_sq if label == cmp_labels[0] else 0.0
                rec["cmp_sq"][label] += cmp_sq
                rec["dot"][label] += dot
                rec["extra_sq"][label] += extra_sq

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoints": DEFAULT_CHECKPOINTS,
        "top1": TOP1,
        "rows": {},
    }
    for group_type, group in groups.items():
        rows = []
        for key, rec in group.items():
            rows.extend(finalize_group(group_type, key, rec, cmp_labels))
        rows.sort(key=lambda item: (item["delta_top1_vs_best"], -item["extra_over_best"], item["delta_cosine_to_best"]))
        result["rows"][group_type] = rows[: args.topn]
        path = out_dir / f"{group_type}_delta.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "group_type",
                "key",
                "cmp",
                "params",
                "numel",
                "best_delta_l2",
                "cmp_delta_l2",
                "cmp_over_best",
                "delta_cosine_to_best",
                "extra_l2_vs_best",
                "extra_over_best",
                "best_top1",
                "cmp_top1",
                "delta_top1_vs_best",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
