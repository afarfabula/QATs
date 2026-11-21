#!/usr/bin/env python3
"""Measure DeiT student pre/post-quant attention against the FP teacher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from timm.models import create_model, load_checkpoint


QATS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QATS))
sys.path.insert(0, str(QATS / "third_party" / "OFQ" / "tools"))
sys.path.insert(0, str(QATS / "third_party" / "OFQ"))

import qat_launch as ql  # noqa: E402
from offline_attention_oscillation import build_args, build_probe_loader  # noqa: E402


def metric_values(attn: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    p = attn.float().clamp_min(eps)
    row_sum = p.sum(dim=-1).clamp_min(eps)
    p = p / row_sum.unsqueeze(-1)
    entropy = -(p * p.log()).sum(dim=-1).mean()
    top1 = p.max(dim=-1).values.mean()
    top5 = p.topk(min(5, p.shape[-1]), dim=-1).values.sum(dim=-1).mean()
    return {
        "entropy": float(entropy),
        "top1": float(top1),
        "top5": float(top5),
    }


def attention_kl(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 1.0) -> float:
    s = ql.retemper_attention_prob(student, temperature).float().clamp_min(1e-8)
    t = ql.retemper_attention_prob(teacher, temperature).float().clamp_min(1e-8)
    return float(F.kl_div(s.log(), t, reduction="none").sum(dim=-1).mean())


def layer_index_from_quantizer_name(name: str) -> int:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    raise ValueError(f"cannot parse layer index from {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="/datadisk2/linyichen/OFQ/ImageNet-1K")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    run_args = build_args(args.args_yaml)
    run_args.batch_size = args.batch_size
    run_args.workers = 4
    run_args.prefetcher = False
    run_args.collect_attention = True

    student = create_model(
        run_args.model,
        num_classes=run_args.num_classes,
        drop_rate=run_args.drop,
        pretrained=False,
        qqkkvv=False,
    )
    if run_args.quantized:
        student = ql.get_ofq_qat_model(student, run_args)
    ql.enable_attention_collection(student)
    load_checkpoint(student, args.checkpoint, strict=False)
    student.to(device).eval()
    ql.set_attention_mode(student, collect_attention=True, qqkkvv=False)
    ql.set_attention_collection_target(student, "post_quant")

    teacher = ql.create_ofq_teacher_model(run_args)
    ql.set_attention_mode(teacher, collect_attention=True, qqkkvv=False)
    ql.set_attention_collection_target(teacher, "post_quant")
    teacher.to(device).eval()

    loader = build_probe_loader(run_args, args.data)
    captures, handles = ql.install_attention_quantization_hooks(student)
    rows: dict[int, dict[str, list[float]]] = {}

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                images = batch[0] if not isinstance(batch, torch.Tensor) else batch
                images = images.to(device, non_blocking=True)
                captures.clear()
                student_out = student(images)
                teacher_out = teacher(images)
                student_attn = student_out[1]
                teacher_attn = teacher_out[1]

                by_layer = {}
                for name, pre, post in captures:
                    by_layer[layer_index_from_quantizer_name(name)] = (pre.float(), post.float())

                for layer in sorted(by_layer):
                    pre, post = by_layer[layer]
                    teacher_layer = teacher_attn[layer]
                    if teacher_layer is None:
                        continue
                    teacher_layer = teacher_layer.float()
                    total = post.numel()
                    meaningful = pre > 1e-4
                    meaningful_count = int(meaningful.sum())
                    stat = rows.setdefault(
                        layer,
                        {
                            "total": [],
                            "post_zero": [],
                            "rows": [],
                            "rows_ge50_zero": [],
                            "rows_ge80_zero": [],
                            "post_mass": [],
                            "top5_retention": [],
                            "meaningful": [],
                            "meaningful_zero": [],
                            "pre_top1": [],
                            "post_top1": [],
                            "teacher_top1": [],
                            "pre_top5": [],
                            "post_top5": [],
                            "teacher_top5": [],
                            "pre_entropy": [],
                            "post_entropy": [],
                            "teacher_entropy": [],
                            "kl_post_teacher_t1": [],
                            "kl_post_teacher_t2": [],
                            "kl_pre_teacher_t2": [],
                        },
                    )
                    stat["total"].append(float(total))
                    stat["post_zero"].append(float((post == 0).sum()))
                    row_zero_rate = (post == 0).float().mean(dim=-1)
                    stat["rows"].append(float(row_zero_rate.numel()))
                    stat["rows_ge50_zero"].append(float((row_zero_rate >= 0.5).sum()))
                    stat["rows_ge80_zero"].append(float((row_zero_rate >= 0.8).sum()))
                    pre_sum = pre.sum(dim=-1)
                    post_sum = post.sum(dim=-1)
                    stat["post_mass"].append(float(post_sum.mean()))
                    top5_idx = pre.topk(min(5, pre.shape[-1]), dim=-1).indices
                    pre_top5_mass = pre.gather(-1, top5_idx).sum(dim=-1)
                    post_top5_mass = post.gather(-1, top5_idx).sum(dim=-1)
                    stat["top5_retention"].append(
                        float((post_top5_mass / pre_top5_mass.clamp_min(1e-8)).mean())
                    )
                    stat["meaningful"].append(float(meaningful_count))
                    stat["meaningful_zero"].append(float(((post == 0) & meaningful).sum()))
                    for prefix, tensor in (("pre", pre), ("post", post), ("teacher", teacher_layer)):
                        values = metric_values(tensor)
                        stat[f"{prefix}_top1"].append(values["top1"])
                        stat[f"{prefix}_top5"].append(values["top5"])
                        stat[f"{prefix}_entropy"].append(values["entropy"])
                    stat["kl_post_teacher_t1"].append(attention_kl(post, teacher_layer, 1.0))
                    stat["kl_post_teacher_t2"].append(attention_kl(post, teacher_layer, 2.0))
                    stat["kl_pre_teacher_t2"].append(attention_kl(pre, teacher_layer, 2.0))
    finally:
        for handle in handles:
            handle.remove()

    def mean(key: str, stat: dict[str, list[float]]) -> float:
        return sum(stat[key]) / max(len(stat[key]), 1)

    print(
        f"{'layer':>5} {'zero%':>7} {'row>=80%0':>10} {'meaningful0%':>12} {'mass%':>7} {'top5ret%':>9} "
        f"{'s-pre-top1':>11} {'s-q-top1':>9} {'t-top1':>8} "
        f"{'s-q-top5':>9} {'t-top5':>8} "
        f"{'s-pre-H':>8} {'s-q-H':>8} {'t-H':>8} "
        f"{'KLqT1':>9} {'KLqT2':>9} {'KLpreT2':>10}"
    )
    for layer in sorted(rows):
        stat = rows[layer]
        total = sum(stat["total"])
        meaningful = sum(stat["meaningful"])
        print(
            f"{layer:>5d} "
            f"{100.0 * sum(stat['post_zero']) / max(total, 1):>7.3f} "
            f"{100.0 * sum(stat['rows_ge80_zero']) / max(sum(stat['rows']), 1):>10.3f} "
            f"{100.0 * sum(stat['meaningful_zero']) / max(meaningful, 1):>12.3f} "
            f"{100.0 * mean('post_mass', stat):>7.3f} "
            f"{100.0 * mean('top5_retention', stat):>9.3f} "
            f"{mean('pre_top1', stat):>11.5f} {mean('post_top1', stat):>9.5f} {mean('teacher_top1', stat):>8.5f} "
            f"{mean('post_top5', stat):>9.5f} {mean('teacher_top5', stat):>8.5f} "
            f"{mean('pre_entropy', stat):>8.4f} {mean('post_entropy', stat):>8.4f} {mean('teacher_entropy', stat):>8.4f} "
            f"{mean('kl_post_teacher_t1', stat):>9.4f} {mean('kl_post_teacher_t2', stat):>9.4f} "
            f"{mean('kl_pre_teacher_t2', stat):>10.4f}"
        )

    total = sum(sum(stat["total"]) for stat in rows.values())
    post_zero = sum(sum(stat["post_zero"]) for stat in rows.values())
    meaningful = sum(sum(stat["meaningful"]) for stat in rows.values())
    meaningful_zero = sum(sum(stat["meaningful_zero"]) for stat in rows.values())
    row_count = sum(sum(stat["rows"]) for stat in rows.values())
    rows_ge50_zero = sum(sum(stat["rows_ge50_zero"]) for stat in rows.values())
    rows_ge80_zero = sum(sum(stat["rows_ge80_zero"]) for stat in rows.values())
    print(
        f"OVERALL post_quant_zero={100.0 * post_zero / max(total, 1):.3f}% "
        f"meaningful_zero={100.0 * meaningful_zero / max(meaningful, 1):.3f}% "
        f"rows_ge50_zero={100.0 * rows_ge50_zero / max(row_count, 1):.3f}% "
        f"rows_ge80_zero={100.0 * rows_ge80_zero / max(row_count, 1):.3f}% "
        f"elements={int(total)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
