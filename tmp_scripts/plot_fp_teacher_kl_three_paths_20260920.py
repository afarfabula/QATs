#!/usr/bin/env python3
"""Plot the three 100epoch from-scratch trajectories used in the FP teacher KL note.

Reads `Test: [distributed-summary]` rows from archived training logs and writes:
  docs/figures/fp_teacher_kl_three_paths_top1.png
  docs/figures/fp_teacher_kl_three_paths_delta_vs_nokl.png
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
LOG_DIR = QATS / "experiment_logs" / "fullval_ge10"
FIG_DIR = QATS / "docs" / "figures"

TRAJECTORIES: List[Tuple[str, str, str]] = [
    (
        "不开 KL（基线）",
        "playground__train_ofq_100ep_fromscratch_original_ofq_public_control_20260714.log",
        "tab:blue",
    ),
    (
        "自参考 KL",
        "playground__train_ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713.log",
        "tab:orange",
    ),
    (
        "FP teacher KL",
        "playground__train_ofq_100ep_fromscratch_teacher_sparse_attnkl_clipgrad_20260804.log",
        "tab:green",
    ),
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
    curves: Dict[str, List[float]] = {}
    for name, filename, _ in TRAJECTORIES:
        curves[name] = read_acc1_curve(LOG_DIR / filename)

    lengths = {name: len(values) for name, values in curves.items()}
    print(f"curve lengths: {lengths}")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    # Draw the control first and thickest so the two KL trajectories that overlap
    # it (by construction, both share the control trajectory early on) stay visible.
    draw_order = [TRAJECTORIES[0], TRAJECTORIES[1], TRAJECTORIES[2]]
    style = {
        "不开 KL（基线）": dict(color="black", linewidth=3.4, alpha=0.30),
        "自参考 KL": dict(color="tab:orange", linewidth=1.5),
        "FP teacher KL": dict(color="tab:green", linewidth=1.5),
    }
    for name, _, color in draw_order:
        values = curves[name]
        epochs = list(range(len(values)))
        ax.plot(epochs, values, label=name, **style[name])
        best_epoch = max(range(len(values)), key=lambda i: values[i])
        ax.plot(
            [best_epoch],
            [values[best_epoch]],
            marker="o",
            markersize=5,
            color=color,
            linestyle="none",
        )
    text_xy = {
        "不开 KL（基线）": (50.0, 80.97),
        "自参考 KL": (16.0, 81.06),
        "FP teacher KL": (66.0, 81.15),
    }
    for name, _, color in TRAJECTORIES:
        values = curves[name]
        best_epoch = max(range(len(values)), key=lambda i: values[i])
        ax.annotate(
            f"{name}: {values[best_epoch]:.4f} @ ep{best_epoch}",
            xy=(best_epoch, values[best_epoch]),
            xytext=text_xy[name],
            textcoords="data",
            fontsize=8.5,
            color=style[name]["color"] if name == "不开 KL（基线）" else color,
            arrowprops=dict(arrowstyle="-", linewidth=0.7, alpha=0.45, color="0.35"),
        )
    ax.set_xlim(-2, 108)
    ax.set_ylim(77.4, 81.28)
    ax.set_xlabel("训练轮次 epoch")
    ax.set_ylabel("ImageNet Top-1 (%)")
    ax.set_title("Swin-T W4A4 三条 100 epoch 训练曲线（每轮全量验证集）")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    top1_path = FIG_DIR / "fp_teacher_kl_three_paths_top1.png"
    fig.savefig(top1_path, dpi=160)
    plt.close(fig)

    baseline = curves["不开 KL（基线）"]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    for name, _, color in TRAJECTORIES[1:]:
        values = curves[name]
        epochs = list(range(len(values)))
        deltas = [values[i] - baseline[i] for i in epochs]
        mean_delta = sum(deltas[5:]) / len(deltas[5:])
        ax.plot(
            epochs,
            deltas,
            color=color,
            linewidth=1.8,
            label=f"{name}（第 5-99 轮平均 {mean_delta:+.4f}）",
        )
    ax.set_xlabel("训练轮次 epoch")
    ax.set_ylabel("相对不开 KL 基线的 Top-1 差异 (pp)")
    ax.set_title("相对基线的逐轮差异：两条 KL 都没有稳定的正收益")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    delta_path = FIG_DIR / "fp_teacher_kl_three_paths_delta_vs_nokl.png"
    fig.savefig(delta_path, dpi=160)
    plt.close(fig)

    print(f"wrote {top1_path}")
    print(f"wrote {delta_path}")


if __name__ == "__main__":
    main()
