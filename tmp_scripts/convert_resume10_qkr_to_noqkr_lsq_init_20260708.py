#!/usr/bin/env python3
"""Convert a QKR/StatsQ Swin-T checkpoint into a no-QKR/LSQ init checkpoint."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch


DEFAULT_QKR = "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar"
DEFAULT_TEMPLATE = "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_smoke2upd_20260708/checkpoint-11.pth.tar"
DEFAULT_OUTPUT = "/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar"
DEFAULT_REINIT_OUTPUT = "/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_weightreinit_20260708/checkpoint-init.pth.tar"


def load_checkpoint(path: str) -> dict:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    return torch.load(path, map_location="cpu", weights_only=False)


def strip_module(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key[len("module.") :] if key.startswith("module.") else key: value for key, value in state.items()}


def source_tensor(source: Dict[str, torch.Tensor], key: str):
    value = source.get(key)
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def copy_if_shape_matches(target: Dict[str, torch.Tensor], source: Dict[str, torch.Tensor], key: str) -> bool:
    src = source_tensor(source, key)
    dst = target.get(key)
    if torch.is_tensor(src) and torch.is_tensor(dst) and src.shape == dst.shape:
        target[key] = src.clone()
        return True
    return False


def convert_attention_qkv(target: Dict[str, torch.Tensor], source: Dict[str, torch.Tensor], qkv_key: str) -> bool:
    prefix = qkv_key[: -len(".qkv.weight")]
    q = source_tensor(source, f"{prefix}.q.weight")
    k = source_tensor(source, f"{prefix}.k.weight")
    v = source_tensor(source, f"{prefix}.v.weight")
    dst = target.get(qkv_key)
    if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v) and torch.is_tensor(dst)):
        return False
    merged = torch.cat([q, k, v], dim=0)
    if merged.shape != dst.shape:
        return False
    target[qkv_key] = merged.clone()
    return True


def convert_attention_qkv_bias(target: Dict[str, torch.Tensor], source: Dict[str, torch.Tensor], qkv_bias_key: str) -> bool:
    prefix = qkv_bias_key[: -len(".qkv.bias")]
    q_bias = source_tensor(source, f"{prefix}.q_bias")
    k_bias = source_tensor(source, f"{prefix}.k_bias")
    v_bias = source_tensor(source, f"{prefix}.v.bias")
    dst = target.get(qkv_bias_key)
    if not (torch.is_tensor(q_bias) and torch.is_tensor(k_bias) and torch.is_tensor(v_bias) and torch.is_tensor(dst)):
        return False
    merged = torch.cat([q_bias, k_bias, v_bias], dim=0)
    if merged.shape != dst.shape:
        return False
    target[qkv_bias_key] = merged.clone()
    return True


def convert_qkv_move_bias(target: Dict[str, torch.Tensor], source: Dict[str, torch.Tensor], key: str) -> bool:
    prefix = key.split(".qkv.", 1)[0]
    suffix = key.split(".qkv.", 1)[1]
    src_key = f"{prefix}.quant_x_4_qkv.{suffix}"
    return copy_if_shape_matches(target, source, src_key)


def lsq_weight_scale_from_weight(weight: torch.Tensor, thd_pos: int = 7) -> torch.Tensor:
    weight = weight.detach().float()
    if weight.ndim == 2:
        return 2.0 * weight.abs().mean(dim=-1) / (float(thd_pos) ** 0.5)
    if weight.ndim == 4:
        return 2.0 * weight.abs().mean(dim=(1, 2, 3)) / (float(thd_pos) ** 0.5)
    raise ValueError(f"unsupported LSQ weight shape {tuple(weight.shape)}")


def reinit_weight_lsq_from_weight(target: Dict[str, torch.Tensor]) -> Dict[str, object]:
    reinitialized = []
    skipped = []
    for scale_key in sorted(key for key in target if key.endswith(".lsqw_fn.s")):
        weight_key = scale_key[: -len(".lsqw_fn.s")] + ".weight"
        scale = target.get(scale_key)
        weight = target.get(weight_key)
        if not (torch.is_tensor(scale) and torch.is_tensor(weight)):
            skipped.append({"scale_key": scale_key, "reason": "missing tensor"})
            continue
        try:
            new_scale = lsq_weight_scale_from_weight(weight)
        except ValueError as exc:
            skipped.append({"scale_key": scale_key, "weight_key": weight_key, "reason": str(exc)})
            continue
        if new_scale.shape != scale.shape:
            skipped.append(
                {
                    "scale_key": scale_key,
                    "weight_key": weight_key,
                    "reason": f"shape mismatch new={tuple(new_scale.shape)} old={tuple(scale.shape)}",
                }
            )
            continue
        ratio = (scale.detach().float() / new_scale.clamp_min(1e-12)).reshape(-1)
        target[scale_key] = new_scale.to(dtype=scale.dtype).clone()
        reinitialized.append(
            {
                "scale_key": scale_key,
                "weight_key": weight_key,
                "old_abs_mean": float(scale.detach().float().abs().mean().item()),
                "new_abs_mean": float(new_scale.abs().mean().item()),
                "old_over_new_mean": float(ratio.mean().item()),
                "old_over_new_min": float(ratio.min().item()),
                "old_over_new_max": float(ratio.max().item()),
            }
        )
    return {
        "reinitialized": reinitialized,
        "skipped": skipped,
        "counts": {
            "weight_lsq_reinitialized": len(reinitialized),
            "weight_lsq_skipped": len(skipped),
        },
    }


def convert_state(source: Dict[str, torch.Tensor], template: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], dict]:
    target = {key: value.clone() if torch.is_tensor(value) else value for key, value in template.items()}
    stats = {
        "direct_copied": [],
        "qkv_weight_merged": [],
        "qkv_bias_merged": [],
        "qkv_aux_copied": [],
        "template_kept": [],
    }
    for key in list(target.keys()):
        if key.endswith(".qkv.weight"):
            if convert_attention_qkv(target, source, key):
                stats["qkv_weight_merged"].append(key)
            else:
                stats["template_kept"].append(key)
            continue
        if key.endswith(".qkv.bias"):
            if convert_attention_qkv_bias(target, source, key):
                stats["qkv_bias_merged"].append(key)
            else:
                stats["template_kept"].append(key)
            continue
        if ".qkv.input_quant_fn." in key or ".qkv.move_" in key:
            if convert_qkv_move_bias(target, source, key):
                stats["qkv_aux_copied"].append(key)
            else:
                stats["template_kept"].append(key)
            continue
        if "lsqw_fn." in key:
            stats["template_kept"].append(key)
            continue
        if copy_if_shape_matches(target, source, key):
            stats["direct_copied"].append(key)
        else:
            stats["template_kept"].append(key)
    stats["counts"] = {name: len(values) for name, values in stats.items() if isinstance(values, list)}
    return target, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkr-checkpoint", default=DEFAULT_QKR)
    parser.add_argument("--template-checkpoint", default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default="")
    parser.add_argument("--reinit-weight-lsq-from-weight", action="store_true")
    args = parser.parse_args()

    qkr_ckpt = load_checkpoint(args.qkr_checkpoint)
    template_ckpt = load_checkpoint(args.template_checkpoint)
    qkr_state = strip_module(qkr_ckpt.get("state_dict", qkr_ckpt))
    template_state = strip_module(template_ckpt.get("state_dict", template_ckpt))
    converted_state, stats = convert_state(qkr_state, template_state)
    if args.reinit_weight_lsq_from_weight:
        stats["weight_lsq_reinit"] = reinit_weight_lsq_from_weight(converted_state)
        stats["counts"].update(stats["weight_lsq_reinit"]["counts"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    new_ckpt = {
        "epoch": int(qkr_ckpt.get("epoch", template_ckpt.get("epoch", 0))),
        "state_dict": converted_state,
        "conversion_meta": {
            "source_qkr_checkpoint": args.qkr_checkpoint,
            "template_checkpoint": args.template_checkpoint,
            "method": (
                "merge q/k/v into qkv, copy compatible tensors, keep LSQ state from no-QKR template"
                + (
                    ", reinitialize weight LSQ scales from converted weights"
                    if args.reinit_weight_lsq_from_weight
                    else ""
                )
            ),
            "stats": stats,
        },
    }
    torch.save(new_ckpt, output)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(new_ckpt["conversion_meta"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")
    print(f"summary {summary_path}")
    print(json.dumps(stats["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
