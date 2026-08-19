#!/usr/bin/env python3
"""Diagnose qkv/LSQ scale consistency for converted no-QKR Swin checkpoints."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable

import torch


DEFAULT_SOURCE = "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar"
DEFAULT_TEMPLATE = "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_smoke2upd_20260708/checkpoint-11.pth.tar"
DEFAULT_CONVERTED = "/mlx_devbox/users/quyanyi/playground/qat_public_repro/converted_resume10_clean_lsq_noqkr_init_20260708/checkpoint-init.pth.tar"
DEFAULT_OUTPUT = "/mlx_devbox/users/quyanyi/playground/QATs/docs/resume10_noqkr_lsq_scale_diagnosis_20260708"


def load_state(path: str) -> Dict[str, torch.Tensor]:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return {key[len("module.") :] if key.startswith("module.") else key: value for key, value in state.items()}


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().reshape(-1)
    return {
        "mean": float(x.mean().item()),
        "abs_mean": float(x.abs().mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def ideal_lsq_scale(weight: torch.Tensor, thd_pos: int = 7) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"expected 2D weight, got {tuple(weight.shape)}")
    return 2.0 * weight.detach().float().abs().mean(dim=-1) / (float(thd_pos) ** 0.5)


def safe_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.detach().float() / b.detach().float().clamp_min(1e-12)


def attention_prefixes(state: Dict[str, torch.Tensor]) -> Iterable[str]:
    for key in sorted(state):
        if key.endswith(".qkv.weight"):
            yield key[: -len(".qkv.weight")]


def collect_module(prefix: str, source: Dict[str, torch.Tensor], template: Dict[str, torch.Tensor], converted: Dict[str, torch.Tensor]) -> Dict[str, object]:
    qkv_key = f"{prefix}.qkv.weight"
    scale_key = f"{prefix}.qkv.lsqw_fn.s"
    result: Dict[str, object] = {"module": prefix}
    qkv = converted.get(qkv_key)
    scale = converted.get(scale_key)
    if torch.is_tensor(qkv):
        ideal = ideal_lsq_scale(qkv)
        result["converted_qkv_weight"] = tensor_stats(qkv)
        result["ideal_lsq_s"] = tensor_stats(ideal)
    else:
        result["converted_qkv_weight_missing"] = True
        return result
    if torch.is_tensor(scale):
        ratio = safe_ratio(scale, ideal)
        result["converted_lsq_s"] = tensor_stats(scale)
        result["converted_s_over_ideal"] = tensor_stats(ratio)
        result["converted_s_over_ideal_p05_p50_p95"] = [
            float(v) for v in torch.quantile(ratio.reshape(-1), torch.tensor([0.05, 0.50, 0.95])).tolist()
        ]
    else:
        result["converted_lsq_s_missing"] = True
    tmpl_scale = template.get(scale_key)
    if torch.is_tensor(tmpl_scale) and torch.is_tensor(scale) and tmpl_scale.shape == scale.shape:
        result["template_s_equal_converted"] = bool(torch.equal(tmpl_scale, scale))
        result["template_lsq_s"] = tensor_stats(tmpl_scale)
    tmpl_weight = template.get(qkv_key)
    if torch.is_tensor(tmpl_weight) and tmpl_weight.shape == qkv.shape:
        weight_delta = (qkv.detach().float() - tmpl_weight.detach().float()).reshape(-1)
        result["converted_minus_template_qkv"] = tensor_stats(weight_delta)
    source_q = source.get(f"{prefix}.q.weight")
    source_k = source.get(f"{prefix}.k.weight")
    source_v = source.get(f"{prefix}.v.weight")
    if torch.is_tensor(source_q) and torch.is_tensor(source_k) and torch.is_tensor(source_v):
        merged = torch.cat([source_q, source_k, source_v], dim=0)
        result["source_qkv_merged_equal_converted"] = bool(torch.equal(merged, qkv))
        result["source_q_weight"] = tensor_stats(source_q)
        result["source_k_weight"] = tensor_stats(source_k)
        result["source_v_weight"] = tensor_stats(source_v)
    qk_scale = source.get(f"{prefix}.qk_quant.s")
    v_scale = source.get(f"{prefix}.v_quant.s")
    if torch.is_tensor(qk_scale):
        result["source_qk_quant_s"] = tensor_stats(qk_scale)
    if torch.is_tensor(v_scale):
        result["source_v_quant_s"] = tensor_stats(v_scale)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--converted", default=DEFAULT_CONVERTED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = load_state(args.source)
    template = load_state(args.template)
    converted = load_state(args.converted)
    rows = [collect_module(prefix, source, template, converted) for prefix in attention_prefixes(converted)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "noqkr_lsq_scale_diagnosis.json"
    tsv_path = out_dir / "noqkr_lsq_scale_diagnosis.tsv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write(
            "module\tconverted_abs_mean\tideal_s_abs_mean\tconverted_s_abs_mean\t"
            "s_over_ideal_mean\ts_over_ideal_p05\ts_over_ideal_p50\ts_over_ideal_p95\t"
            "template_s_equal_converted\tsource_merge_equal_converted\n"
        )
        for row in rows:
            ratio_q = row.get("converted_s_over_ideal_p05_p50_p95", ["", "", ""])
            f.write(
                f"{row['module']}\t"
                f"{row.get('converted_qkv_weight', {}).get('abs_mean', '')}\t"
                f"{row.get('ideal_lsq_s', {}).get('abs_mean', '')}\t"
                f"{row.get('converted_lsq_s', {}).get('abs_mean', '')}\t"
                f"{row.get('converted_s_over_ideal', {}).get('mean', '')}\t"
                f"{ratio_q[0]}\t{ratio_q[1]}\t{ratio_q[2]}\t"
                f"{row.get('template_s_equal_converted', '')}\t"
                f"{row.get('source_qkv_merged_equal_converted', '')}\n"
            )
    print(f"wrote {json_path}")
    print(f"wrote {tsv_path}")
    print(f"modules={len(rows)}")


if __name__ == "__main__":
    main()
