#!/usr/bin/env python3
"""Compare short-update checkpoint drift for the resume10-to-81 branch."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


DEFAULT_CHECKPOINTS = {
    "base": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar",
    "u125": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_125upd_gate_20260708/checkpoint-3.pth.tar",
    "u250": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar",
    "u300": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_300upd_gate_20260708/checkpoint-3.pth.tar",
    "u500": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_500upd_gate_20260708/checkpoint-3.pth.tar",
}

TOP1 = {
    "base": 80.5400,
    "u125": 80.5220,
    "u250": 80.5540,
    "u300": 80.4860,
    "u500": 80.5300,
}


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    result = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = key[len("module.") :] if key.startswith("module.") else key
        if value.is_floating_point():
            result[name] = value.detach().cpu()
    return result


def param_kind(name: str) -> str:
    if any(token in name for token in ("quan_a_softmax", "softmax")):
        return "softmax_quant"
    if any(token in name for token in ("input_quant_fn", "quant_x_4_qkv", "quan_a_q", "quan_a_k", "quan_a_v", "quan_a_qkx")):
        return "act_quant"
    if any(token in name for token in ("lsqw_fn", "statsq_fn", "qk_quant", "v_quant")):
        return "weight_quant"
    if ".move_v" in name:
        return "move_v_shift"
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


def l2_delta(before: torch.Tensor, after: torch.Tensor) -> Tuple[float, float, float, float, int]:
    if before.shape != after.shape:
        return 0.0, 0.0, 0.0, 0.0, 0
    before_f = before.float()
    after_f = after.float()
    delta = after_f - before_f
    delta_l2 = float(delta.norm().item())
    base_l2 = float(before_f.norm().clamp_min(1e-12).item())
    rel_l2 = delta_l2 / base_l2
    max_abs = float(delta.abs().max().item()) if delta.numel() else 0.0
    return delta_l2, base_l2, rel_l2, max_abs, int(delta.numel())


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = list(xs)
    y = list(ys)
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((item - mx) ** 2 for item in x)
    vy = sum((item - my) ** 2 for item in y)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def add_group(groups, key, label, delta_l2, base_l2, rel_l2, max_abs, numel):
    rec = groups[key]
    point = rec["points"][label]
    point["delta_sq"] += delta_l2 * delta_l2
    point["base_sq"] += base_l2 * base_l2
    point["max_rel"] = max(point["max_rel"], rel_l2)
    point["max_abs"] = max(point["max_abs"], max_abs)
    point["numel"] += numel
    point["params"] += 1


def finalize(groups, labels, group_type):
    rows = []
    update_labels = [label for label in labels if label != "base"]
    update_top1 = [TOP1[label] for label in update_labels]
    for key, rec in groups.items():
        values = {}
        params = 0
        numel = 0
        for label in update_labels:
            point = rec["points"][label]
            delta = math.sqrt(point["delta_sq"])
            base = max(math.sqrt(point["base_sq"]), 1e-12)
            values[label] = delta / base
            params = max(params, point["params"])
            numel = max(numel, point["numel"])
        drift_curve = [values[label] for label in update_labels]
        row = {
            "group_type": group_type,
            "key": key if isinstance(key, str) else "|".join(key),
            "params": params,
            "numel": numel,
            "rel_125": values.get("u125", 0.0),
            "rel_250": values.get("u250", 0.0),
            "rel_300": values.get("u300", 0.0),
            "rel_500": values.get("u500", 0.0),
            "corr_top1": pearson(drift_curve, update_top1),
            "gain_250_minus_125": TOP1["u250"] - TOP1["u125"],
            "drop_300_minus_250": TOP1["u300"] - TOP1["u250"],
            "drop_500_minus_250": TOP1["u500"] - TOP1["u250"],
            "drift_300_minus_250": values.get("u300", 0.0) - values.get("u250", 0.0),
            "drift_500_minus_250": values.get("u500", 0.0) - values.get("u250", 0.0),
        }
        rows.append(row)
    rows.sort(key=lambda item: (item["corr_top1"], -item["drift_300_minus_250"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_short_update_drift_20260708.json")
    parser.add_argument("--out-tsv", default="/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_short_update_drift_20260708.tsv")
    parser.add_argument("--topn", type=int, default=120)
    args = parser.parse_args()

    states = {label: load_state_dict(path) for label, path in DEFAULT_CHECKPOINTS.items()}
    labels = list(DEFAULT_CHECKPOINTS)
    base = states["base"]
    group_defs = {
        "kind": lambda name: param_kind(name),
        "stage_kind": lambda name: (stage_name(name), param_kind(name)),
        "module_kind": lambda name: (module_name(name), param_kind(name)),
        "param": lambda name: name,
    }
    groups = {
        group_type: defaultdict(lambda: {"points": defaultdict(lambda: {"delta_sq": 0.0, "base_sq": 0.0, "max_rel": 0.0, "max_abs": 0.0, "numel": 0, "params": 0})})
        for group_type in group_defs
    }

    for name, before in base.items():
        for label in labels:
            if label == "base":
                continue
            after = states[label].get(name)
            if after is None:
                continue
            delta_l2, base_l2, rel_l2, max_abs, numel = l2_delta(before, after)
            if numel == 0:
                continue
            for group_type, key_fn in group_defs.items():
                add_group(groups[group_type], key_fn(name), label, delta_l2, base_l2, rel_l2, max_abs, numel)

    result = {
        "checkpoints": DEFAULT_CHECKPOINTS,
        "top1": TOP1,
        "rows": {},
    }
    all_rows = []
    for group_type, group in groups.items():
        rows = finalize(group, labels, group_type)
        selected = rows[: args.topn]
        result["rows"][group_type] = selected
        all_rows.extend(selected)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    out_tsv = Path(args.out_tsv)
    fields = [
        "group_type",
        "key",
        "params",
        "numel",
        "rel_125",
        "rel_250",
        "rel_300",
        "rel_500",
        "corr_top1",
        "drift_300_minus_250",
        "drift_500_minus_250",
    ]
    with out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", delimiter="\t")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"wrote {out_json}")
    print(f"wrote {out_tsv}")


if __name__ == "__main__":
    main()
