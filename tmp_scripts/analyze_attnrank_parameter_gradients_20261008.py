#!/usr/bin/env python3
"""Per-tensor gradient breakdown for the prev-step attention-relation ranking loss.

Builds the real QAT Swin-T (W4A4 + qk-reparam) with attention collection limited to
global blocks 8/9/10/11, computes the soft-KD loss and the attention-relation ranking
loss on one batch, and reports ||g_rank|| vs ||g_kd|| per parameter / per attention block.

This is what showed that the *global* ratio (0.0039) is misleading: on the tensors that
actually control the attention ordering the rank gradient is 3-6x larger than KD's, so
any lambda that looks "calibrated" globally is 2-3 orders of magnitude too strong.

See docs/attn_relation_ranking_qat_prevstep_l8to11_failure_20261008.md section 5.

Usage:
    CUDA_VISIBLE_DEVICES=2 python analyze_attnrank_parameter_gradients_20261008.py \
        [--data /datadisk2/linyichen/OFQ/ImageNet-1K] [--batch-size 8]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch

QATS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QATS))
sys.path.insert(0, str(QATS / "third_party" / "OFQ"))

import qat_launch as Q  # noqa: E402
import src  # noqa: F401,E402  (registers OFQ swin_t / deit under timm)
from timm.models import create_model  # noqa: E402

DEFAULT_DATA = "/datadisk2/linyichen/OFQ/ImageNet-1K"
DEFAULT_TEACHER = "/home_ext/quyanyi/qat_weights/swin_t-704ceda3.pth"
DEFAULT_CONFIG = str(QATS / "third_party" / "OFQ" / "configs" / "swin_t_imagenet.attn_q.yml")
RANK_LAYERS = [(8, 12), (9, 12), (10, 24), (11, 24)]


def heads_str(spec):
    return "custom_subset:" + ",".join(f"{layer}:{h}" for layer, n in spec for h in range(n))


def build_runtime_args(data: str, teacher: str, config: str):
    flags = [
        "--method", "ofq", "--stage", "train", "--config", config,
        "--model", "swin_t", "--data", data, "--dataset-format", "folder",
        "--output", "/tmp/attnrank_gradcheck", "--experiment", "gradcheck",
        "--devices", "0", "--nproc-per-node", "1", "--model-type", "swin",
        "--teacher", "swin_t", "--teacher-type", "swin", "--teacher-pretrained",
        "--teacher-checkpoint", teacher,
        "--epochs", "100", "--scheduler-epochs", "100",
        "--batch-size", "32", "--workers", "2", "--lr", "2e-4", "--min-lr", "5e-6", "--weight-decay", "0.0",
        "--grad-accum-steps", "16", "--epoch-checkpoint-interval", "1", "--checkpoint-hist", "2",
        "--wbits", "4", "--abits", "4", "--wq-mode", "statsq", "--aq-mode", "lsq",
        "--wq-per-channel", "--aq-per-channel", "--aq-clip-learnable",
        "--pretrained", "--pretrained-initialized",
        "--use-kd", "--kd-hard-and-soft", "0", "--teacher-soft-temperature", "2.75",
        "--quantized", "--qk-reparam", "--qk-reparam-type", "0", "--amp", "--amp-dtype", "bf16",
        "--train-scheme", "ema_ref_attn_kl", "--ref-update", "prev_step", "--ref-update-interval", "50",
        "--ref-attn-kl-weight", "0.0", "--ref-logit-kl-weight", "0.0",
        "--ref-head-mode", heads_str(RANK_LAYERS),
        "--attn-rank-weight", "0.0", "--attn-rank-source", "ref", "--attn-rank-topk", "5",
    ]
    saved_argv = sys.argv
    try:
        sys.argv = ["qat_launch.py"] + flags
        args = Q.parse_args()
    finally:
        sys.argv = saved_argv
    runtime_args = Q.build_ofq_runtime_config(args)
    runtime_args.local_rank, runtime_args.world_size, runtime_args.distributed = 0, 1, False
    return runtime_args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=5)
    cli = parser.parse_args()

    runtime_args = build_runtime_args(cli.data, cli.teacher, cli.config)
    device = torch.device(cli.device)
    torch.manual_seed(42)

    model = create_model(
        runtime_args.model,
        drop_path=runtime_args.drop_path,
        num_classes=runtime_args.num_classes,
        pretrained=runtime_args.pretrained,
        qqkkvv=False,
    )
    model = Q.get_ofq_qat_model(model, runtime_args)
    Q.set_attention_mode(model, collect_attention=True)
    Q.set_selected_attention_heads(model, Q.ref_head_map(runtime_args.ref_head_mode))
    model = model.to(device)

    teacher = Q.create_ofq_teacher_model(runtime_args).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    ref = Q.clone_ref_model(model)  # prev-step ref == current weights at t0

    inputs = torch.randn(cli.batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        teacher_out = teacher(inputs)
        ref_out = ref(inputs)
    teacher_logit = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out
    ref_attn = ref_out[1]

    student_out = model(inputs)
    student_logit, student_attn = student_out[0], student_out[1]

    loss_fn = Q.load_ofq_training_module().attention_relation_ranking_loss
    heads = Q.parse_ref_head_mode(runtime_args.ref_head_mode)
    kd = Q.teacher_soft_kd_with_temperature(student_logit, teacher_logit, temperature=2.75)
    rank, pairs = loss_fn(student_attn, ref_attn, heads=heads, topk=cli.topk, min_attn=runtime_args.attn_rank_min_attn)
    print(f"heads={len(heads) if heads else 'all'}  KD={float(kd):.4f}  AttnRank={float(rank):.6f}  valid_pairs={pairs}")

    def grad_norms(loss, retain_graph=False):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=retain_graph)
        return {name: (param.grad.norm().item() if param.grad is not None else 0.0)
                for name, param in model.named_parameters()}

    g_rank = grad_norms(rank, retain_graph=True)
    g_kd = grad_norms(kd)

    attn_names = [name for name, module in model.named_modules() if Q.is_attention_module(module)]

    def block_of(name):
        for index, attn_name in enumerate(attn_names):
            if name.startswith(attn_name + "."):
                return index
        return -1

    agg: dict[int, list] = {}
    for name in g_rank:
        entry = agg.setdefault(block_of(name), [0.0, 0.0, 0])
        entry[0] += g_rank[name] ** 2
        entry[1] += g_kd[name] ** 2
        entry[2] += 1

    selected = {layer for layer, _ in heads} if heads else None
    print(f"\n{'attnblk':>7} {'params':>7} {'||g_rank||':>12} {'||g_kd||':>12} {'ratio':>9}  constrained")
    for block in sorted(agg):
        rank_sq, kd_sq, count = agg[block]
        mark = "" if block < 0 else ("YES" if (selected is None or block in selected) else "no")
        print(f"{block:>7} {count:>7} {rank_sq ** 0.5:>12.4e} {kd_sq ** 0.5:>12.4e} "
              f"{rank_sq ** 0.5 / max(kd_sq ** 0.5, 1e-30):>9.4f}  {mark}")

    total_rank = sum(v * v for v in g_rank.values()) ** 0.5
    total_kd = sum(v * v for v in g_kd.values()) ** 0.5
    print(f"\nTOTAL: ||g_rank||={total_rank:.4e}  ||g_kd||={total_kd:.4e}  ratio={total_rank / max(total_kd, 1e-30):.5f}")
    print("  ↑ 这个全局比例就是会骗人的那个数:被 500 多个 rank 根本碰不到的张量稀释了")

    print("\ntensors where the rank gradient DOMINATES the kd gradient (ratio = ||g_rank||/||g_kd||):")
    ranked = sorted(((g_rank[name] / g_kd[name], name, g_rank[name], g_kd[name])
                     for name in g_rank if g_kd[name] > 0), reverse=True)
    for ratio, name, rank_norm, kd_norm in ranked[:12]:
        print(f"  {ratio:8.3f}  {name:<62} rank={rank_norm:.3e} kd={kd_norm:.3e}")

    zero_rank = [name for name in g_rank if g_rank[name] == 0.0 and g_kd[name] > 0]
    print(f"\nparams with zero rank grad but nonzero kd grad: {len(zero_rank)}/{len(g_rank)}")
    print("  examples:", zero_rank[:6], "(结构上的下游:最后一块的 V/proj、norm、head —— 符合预期)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
