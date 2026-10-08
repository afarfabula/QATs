#!/usr/bin/env python3
"""Compare two step-100 checkpoints (same seed, only lambda differs) to locate the failure.

For each checkpoint it reports
  * held-out Top-1 on real val images,
  * the attention entropy of constrained blocks 8/10/11,
  * the relative_position_bias_table statistics,
  * ||param||_lambda_high / ||param||_lambda_low for the attention q/k/v/proj weights and the
    activation-quantizer LSQ scales.

The last table is what pinpointed the mechanism: everything is ~1.0 except
`attn.quan_a_softmax_fn.s`, which collapses to ~0.5-0.7 -- i.e. the ranking loss drives the
4-bit softmax quantizer step down so that the attention tail quantizes to exactly 0, and the
`clamp_min(eps)` in the loss then turns those pairs into free zero loss.

See docs/attn_relation_ranking_qat_prevstep_l8to11_failure_20261008.md section 5.5.

Usage:
    CUDA_VISIBLE_DEVICES=4 python analyze_attnrank_failure_checkpoints_20261008.py \
        --run-a /datadisk2/quyanyi/qat_runs/attnrank_ckpt_w0_20261008 \
        --run-b /datadisk2/quyanyi/qat_runs/attnrank_ckpt_w30_20261008
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

QATS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QATS / "third_party" / "OFQ" / "tools"))
sys.path.insert(0, str(QATS / "third_party" / "OFQ"))

from offline_attention_oscillation import build_args, build_model, build_probe_loader  # noqa: E402

CONSTRAINED_BLOCKS = {8: "5.4", 9: "5.5", 10: "7.0", 11: "7.1"}
TENSOR_SUFFIXES = (
    "attn.q.weight",
    "attn.k.weight",
    "attn.v.weight",
    "attn.proj.weight",
    "attn.quan_a_qkx_fn.s",
    "attn.quan_a_softmax_fn.s",
    "attn.quan_a_v_fn.s",
)


@torch.no_grad()
def evaluate(run_dir: str, ckpt: str, data: str, max_samples: int, device: torch.device):
    args = build_args(os.path.join(run_dir, "args.yaml"))
    args.batch_size = 32
    args.workers = 4
    args.prefetcher = False
    model = build_model(args, ckpt, device)
    loader = build_probe_loader(args, data)
    model.eval()

    correct = total = 0
    entropies: dict[int, list] = {block: [] for block in (8, 10, 11)}
    for batch in loader:
        images, target = (batch[0], batch[1]) if not isinstance(batch, torch.Tensor) else (batch, None)
        if target is None:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        out = model(images)
        logit = out[0] if isinstance(out, tuple) else out
        attn = out[1] if isinstance(out, tuple) else None
        correct += int((logit.argmax(1) == target).sum())
        total += images.size(0)
        if attn is not None:
            for block in entropies:
                tensor = attn[block]
                if tensor is None:
                    continue
                prob = tensor.float().clamp_min(1e-12)
                prob = prob / prob.sum(-1, keepdim=True)
                entropies[block].append(float((-(prob * prob.log()).sum(-1)).mean()))
        if total >= max_samples:
            break

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    return {
        "top1": correct / max(total, 1),
        "samples": total,
        "entropy": {k: sum(v) / max(len(v), 1) for k, v in entropies.items()},
        "state": state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", default="/datadisk2/quyanyi/qat_runs/attnrank_ckpt_w0_20261008")
    parser.add_argument("--run-b", default="/datadisk2/quyanyi/qat_runs/attnrank_ckpt_w30_20261008")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="只做汇总(不做 A/B 逐张量对比):给一串 run 目录")
    parser.add_argument("--step", default="step_0100.pth.tar")
    parser.add_argument("--data", default="/datadisk2/linyichen/OFQ/ImageNet-1K")
    parser.add_argument("--max-samples", type=int, default=6400)
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args()

    device = torch.device(cli.device)

    if cli.runs:
        print(f"{'run':<46} {'top-1':>8} {'H(b8)':>8} {'H(b10)':>8} {'H(b11)':>8}")
        for run_dir in cli.runs:
            ckpt = os.path.join(run_dir, "step_checkpoints", cli.step)
            r = evaluate(run_dir, ckpt, cli.data, cli.max_samples, device)
            print(f"{Path(run_dir).name:<46} {r['top1'] * 100:>7.2f}% "
                  f"{r['entropy'][8]:>8.4f} {r['entropy'][10]:>8.4f} {r['entropy'][11]:>8.4f}")
        return 0

    results = {}
    for label, run_dir in (("A", cli.run_a), ("B", cli.run_b)):
        ckpt = os.path.join(run_dir, "step_checkpoints", cli.step)
        results[label] = evaluate(run_dir, ckpt, cli.data, cli.max_samples, device)
        r = results[label]
        print(f"\n== {label}: {run_dir} ==  top-1 = {r['top1'] * 100:.2f}% ({r['samples']} imgs)")
        print("   attention entropy: " + "  ".join(f"block{b}={r['entropy'][b]:.4f}" for b in (8, 10, 11)))

    sa, sb = results["A"]["state"], results["B"]["state"]
    print("\nrelative_position_bias_table std (A -> B):")
    for key in sorted(k for k in sa if "relative_position_bias_table" in k):
        if key in sb:
            print(f"   {key:<60} {float(sa[key].float().std()):.5f} -> {float(sb[key].float().std()):.5f}")

    print(f"\n{'block':>6} {'tensor':<28} {'||.||_A':>12} {'||.||_B':>12} {'B/A':>8}")
    for block, prefix in CONSTRAINED_BLOCKS.items():
        for suffix in TENSOR_SUFFIXES:
            matches = [k for k in sa if k.startswith(f"features.{prefix}.") and k.endswith(suffix)]
            if not matches or matches[0] not in sb:
                continue
            key = matches[0]
            na = float(sa[key].float().norm())
            nb = float(sb[key].float().norm())
            print(f"{block:>6} {suffix:<28} {na:>12.5f} {nb:>12.5f} {nb / max(na, 1e-12):>8.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
