#!/usr/bin/env python3
"""Plot the Swin-T W4A4 100-epoch local logit-rank and attention-rank paths.

Output:
  docs/figures/swin_t_w4a4_100ep_new_baseline_vs_three_paths_top1.png

All curves are complete 100-epoch runs with 50k-image full validation every epoch.
The three archived playground paths are contextual comparisons from the older trainer.
The red curve is the 2026-09-24 local run with classification logits ranking
(lambda=0.04) -- it is NOT a pure lambda=0 baseline, despite the historical file name.
The purple curve is the 2026-10-14 local teacher attention-relation ranking run
(weight=0.3). A pure lambda=0 local 100-epoch baseline does not exist yet; the same-path
lambda=0 control was only run for 10 epochs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

QATS = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = QATS / "experiment_logs" / "fullval_ge10"
FIG_DIR = QATS / "docs" / "figures"

NEW_BASELINE_LOG = Path(
    "/datadisk2/quyanyi/qat_runs/ofq_swin_t_w4a4_100ep_logitrank_20260924.log"
)
LATEST_RANK_LOG = Path(
    "/datadisk2/quyanyi/qat_runs/swin100ep_20261014_teacherrank_w03_v2.log"
)

NEW_BASELINE_NAME = "本地 100ep：logit-rank λ=0.04"
LATEST_RANK_NAME = "本地 100ep：teacher attention-rank λ=0.3"
TRAJECTORIES: List[Tuple[str, Path]] = [
    (
        "不开 KL（旧 trainer 100ep）",
        ARCHIVE_DIR
        / "playground__train_ofq_100ep_fromscratch_original_ofq_public_control_20260714.log",
    ),
    (
        "自参考 KL（旧 trainer 100ep）",
        ARCHIVE_DIR
        / "playground__train_ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.log",
    ),
    (
        "FP teacher KL（旧 trainer 100ep）",
        ARCHIVE_DIR
        / "playground__train_ofq_100ep_fromscratch_teacher_sparse_attnkl_clipgrad_20260804.log",
    ),
    (NEW_BASELINE_NAME, NEW_BASELINE_LOG),
    (LATEST_RANK_NAME, LATEST_RANK_LOG),
]

ACC1_RE = re.compile(r"Acc@1:\s*([0-9.]+)")
SUMMARY_MARKER = "Test: [distributed-summary]"


def read_acc1_curve(path: Path) -> List[float]:
    values: List[float] = []
    with path.open() as handle:
        for line in handle:
            if SUMMARY_MARKER not in line:
                continue
            match = ACC1_RE.search(line)
            if match:
                values.append(float(match.group(1)))
    if not values:
        raise RuntimeError(f"no full-validation rows found in {path}")
    return values


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    curves: Dict[str, List[float]] = {
        name: read_acc1_curve(path) for name, path in TRAJECTORIES
    }
    lengths = {name: len(values) for name, values in curves.items()}
    print(f"curve lengths: {lengths}")
    if len(lengths) != 5 or any(length != 100 for length in lengths.values()):
        raise RuntimeError(f"expected five complete 100-epoch curves, got {lengths}")

    style = {
        "不开 KL（旧 trainer 100ep）": dict(
            color="black", linewidth=3.4, alpha=0.30
        ),
        "自参考 KL（旧 trainer 100ep）": dict(
            color="tab:orange", linewidth=1.5, alpha=0.78
        ),
        "FP teacher KL（旧 trainer 100ep）": dict(
            color="tab:green", linewidth=1.5, alpha=0.78
        ),
        NEW_BASELINE_NAME: dict(color="tab:red", linewidth=2.0),
        LATEST_RANK_NAME: dict(color="tab:purple", linewidth=3.0),
    }

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    for name, _ in TRAJECTORIES:
        values = curves[name]
        epochs = list(range(1, len(values) + 1))
        best_epoch = max(epochs, key=lambda e: values[e - 1])
        ax.plot(
            epochs,
            values,
            label=f"{name}：{values[best_epoch - 1]:.4f}% @ep{best_epoch}",
            **style[name],
        )
        ax.plot(
            [best_epoch],
            [values[best_epoch - 1]],
            marker="o",
            markersize=5.5,
            color=style[name]["color"],
            linestyle="none",
            zorder=5,
        )

    ax.set_xlim(-1, 112)
    ax.set_ylim(77.4, 81.36)
    ax.set_xlabel("训练轮次 epoch")
    ax.set_ylabel("ImageNet Top-1 (%)")
    ax.set_title(
        "Swin-T W4A4 100 epoch：两种 rank 损失与旧三条 KL 轨迹\n"
        "（每轮 50k 全量验证，effective batch 512）"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.6, framealpha=0.94)
    ax.text(
        0.012,
        0.028,
        "红/紫为本地 qat_launch 4×3090：红=logit-rank，紫=teacher attention-rank；"
        "两条都含辅助损失\n"
        "黑/橙/绿为旧 trainer 的 KL 抑制轨迹，作为同规模背景对比；"
        "纯 λ=0 同路径只跑到 10ep",
        transform=ax.transAxes,
        fontsize=7.9,
        color="0.30",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.9),
    )

    inset = ax.inset_axes([0.56, 0.33, 0.405, 0.32])
    for name, _ in TRAJECTORIES:
        values = curves[name]
        epochs = list(range(1, len(values) + 1))
        inset.plot(
            epochs,
            values,
            **{key: value for key, value in style[name].items() if key != "alpha"},
        )
    inset.set_xlim(60, 100)
    inset.set_ylim(80.30, 81.02)
    inset.grid(alpha=0.3)
    inset.tick_params(labelsize=7.5)
    inset.set_title("epoch 60-100 放大", fontsize=8.5)

    fig.tight_layout()
    out_path = FIG_DIR / "swin_t_w4a4_100ep_new_baseline_vs_three_paths_top1.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
