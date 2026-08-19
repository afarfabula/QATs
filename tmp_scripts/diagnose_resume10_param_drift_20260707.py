#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = key
        if name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value.detach().cpu()
    return cleaned


def param_kind(name: str) -> str:
    if any(token in name for token in ("input_quant_fn", "quant_x_4_qkv", "quan_a_q", "quan_a_k", "quan_a_v", "quan_a_qkx", "quan_a_softmax")):
        return "act_quant"
    if any(token in name for token in ("lsqw_fn", "statsq_fn", "qk_quant", "v_quant")):
        return "weight_quant"
    if ".move_" in name or name.startswith("move_"):
        return "move_shift"
    if ".attn.q." in name or ".attn.k." in name or ".attn.v." in name or ".attn.qkv." in name:
        return "attn_qkv_weight"
    if ".attn.proj." in name:
        return "attn_proj_weight"
    if ".mlp." in name or ".fc1." in name or ".fc2." in name:
        return "mlp_weight"
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
    for marker in ("input_quant_fn", "quant_x_4_qkv", "quan_a_q_fn", "quan_a_k_fn", "quan_a_v_fn", "quan_a_qkx_fn", "quan_a_softmax_fn", "lsqw_fn", "statsq_fn", "qk_quant", "v_quant"):
        if marker in parts:
            return ".".join(parts[: parts.index(marker) + 1])
    if len(parts) > 1:
        return ".".join(parts[:-1])
    return name


def tensor_delta(before: torch.Tensor, after: torch.Tensor) -> Tuple[float, float, float, int]:
    if before.shape != after.shape:
        return 0.0, 0.0, 0.0, 0
    before_f = before.float()
    after_f = after.float()
    delta = after_f - before_f
    delta_norm = float(delta.norm().item())
    base_norm = float(before_f.norm().clamp_min(1e-12).item())
    rel = delta_norm / base_norm
    max_abs = float(delta.abs().max().item()) if delta.numel() else 0.0
    return delta_norm, base_norm, rel, max_abs


def add_record(groups, key, delta_norm: float, base_norm: float, rel: float, max_abs: float, numel: int) -> None:
    rec = groups[key]
    rec["delta_sq"] += delta_norm * delta_norm
    rec["base_sq"] += base_norm * base_norm
    rec["max_rel"] = max(rec["max_rel"], rel)
    rec["max_abs"] = max(rec["max_abs"], max_abs)
    rec["numel"] += int(numel)
    rec["params"] += 1


def summarize_groups(groups):
    rows = []
    for key, rec in groups.items():
        delta = rec["delta_sq"] ** 0.5
        base = max(rec["base_sq"] ** 0.5, 1e-12)
        rows.append(
            {
                **{f"key{i}": value for i, value in enumerate(key if isinstance(key, tuple) else (key,))},
                "params": rec["params"],
                "numel": rec["numel"],
                "rel_l2": delta / base,
                "delta_l2": delta,
                "base_l2": base,
                "max_param_rel_l2": rec["max_rel"],
                "max_abs_delta": rec["max_abs"],
            }
        )
    rows.sort(key=lambda item: item["rel_l2"], reverse=True)
    return rows


def compare_states(label: str, before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor], topn: int):
    by_kind = defaultdict(lambda: {"delta_sq": 0.0, "base_sq": 0.0, "max_rel": 0.0, "max_abs": 0.0, "numel": 0, "params": 0})
    by_stage_kind = defaultdict(lambda: {"delta_sq": 0.0, "base_sq": 0.0, "max_rel": 0.0, "max_abs": 0.0, "numel": 0, "params": 0})
    by_module_kind = defaultdict(lambda: {"delta_sq": 0.0, "base_sq": 0.0, "max_rel": 0.0, "max_abs": 0.0, "numel": 0, "params": 0})
    param_rows = []
    for name, before_tensor in before.items():
        after_tensor = after.get(name)
        if after_tensor is None or before_tensor.shape != after_tensor.shape:
            continue
        delta_norm, base_norm, rel, max_abs = tensor_delta(before_tensor, after_tensor)
        kind = param_kind(name)
        stage = stage_name(name)
        module = module_name(name)
        numel = before_tensor.numel()
        add_record(by_kind, (label, kind), delta_norm, base_norm, rel, max_abs, numel)
        add_record(by_stage_kind, (label, stage, kind), delta_norm, base_norm, rel, max_abs, numel)
        add_record(by_module_kind, (label, module, kind), delta_norm, base_norm, rel, max_abs, numel)
        param_rows.append(
            {
                "case": label,
                "name": name,
                "kind": kind,
                "stage": stage,
                "module": module,
                "numel": numel,
                "rel_l2": rel,
                "delta_l2": delta_norm,
                "base_l2": base_norm,
                "max_abs_delta": max_abs,
            }
        )
    param_rows.sort(key=lambda item: item["rel_l2"], reverse=True)
    return {
        "by_kind": summarize_groups(by_kind),
        "by_stage_kind": summarize_groups(by_stage_kind),
        "by_module_kind": summarize_groups(by_module_kind)[:topn],
        "top_params": param_rows[:topn],
    }


def write_tsv(path: Path, sections) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("section\tcase\tkey0\tkey1\tkey2\tparams\tnumel\trel_l2\tdelta_l2\tbase_l2\tmax_param_rel_l2\tmax_abs_delta\tname\tkind\tstage\tmodule\n")
        for section, rows in sections:
            for row in rows:
                handle.write(
                    "\t".join(
                        str(row.get(col, ""))
                        for col in (
                            "section",
                            "case",
                            "key0",
                            "key1",
                            "key2",
                            "params",
                            "numel",
                            "rel_l2",
                            "delta_l2",
                            "base_l2",
                            "max_param_rel_l2",
                            "max_abs_delta",
                            "name",
                            "kind",
                            "stage",
                            "module",
                        )
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt10", required=True)
    parser.add_argument("--ckpt2", required=True)
    parser.add_argument("--ckpt3", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--topn", type=int, default=80)
    args = parser.parse_args()

    ckpt10 = load_state_dict(args.ckpt10)
    ckpt2 = load_state_dict(args.ckpt2)
    ckpt3 = load_state_dict(args.ckpt3)
    result = {
        "ckpt10_to_phase1f_ckpt2": compare_states("ckpt10_to_phase1f_ckpt2", ckpt10, ckpt2, args.topn),
        "phase1f_ckpt2_to_ckpt3": compare_states("phase1f_ckpt2_to_ckpt3", ckpt2, ckpt3, args.topn),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    sections = []
    for case_name, case_result in result.items():
        for section_name, rows in case_result.items():
            prepared = []
            for row in rows:
                prepared.append({"section": section_name, "case": case_name, **row})
            sections.append((section_name, prepared))
    write_tsv(Path(args.out_tsv), sections)


if __name__ == "__main__":
    main()
