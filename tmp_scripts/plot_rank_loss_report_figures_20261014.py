#!/usr/bin/env python3
"""Generate every figure used by the consolidated rank-loss experiment report.

Output directory: docs/figures/

The numbers are copied from the raw logs / checkpoint evaluations recorded in
  * docs/deit_tiny_w4a4_smoke_and_speed_20260923.md  (logits ranking, 2026-09-23/24)
  * docs/attn_relation_ranking_qat_prevstep_l8to11_failure_20261008.md  (attention rank, wave 1)
  * docs/attn_relation_ranking_qat_anticheat_experiments_20261008.md  (anti-cheat + reference ablation)
  * /datadisk2/quyanyi/qat_runs/{swin3ep_20261009_*.log,ofq_swin_t_w4a4_lambda0_control_10ep_20260929.log}
and re-checked against the raw logs where the logs still exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

QATS = Path(__file__).resolve().parents[1]
FIG_DIR = QATS / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "tab:blue"
ORANGE = "tab:orange"
GREEN = "tab:green"
RED = "tab:red"
PURPLE = "tab:purple"
BROWN = "tab:brown"
GRAY = "0.45"
BLACK = "black"


# --------------------------------------------------------------------------------------
# 1. logits ranking: gradient probe + lambda sweep
# --------------------------------------------------------------------------------------
def fig_logitrank_lambda_calibration() -> None:
    rho = [0.03, 0.1, 0.3]
    lambda_for_rho = [0.0118, 0.0392, 0.1175]
    lam = [0.0, 0.04, 0.12, 0.4]
    base = [52.2579, 52.2566, 52.2536, 52.2446]
    rank = [None, 0.6328, 0.6188, 0.5712]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    ax = axes[0]
    ax.plot(rho, lambda_for_rho, marker="o", color=BLUE, linewidth=1.8, label="λ(ρ) = ρ · ||g_KD|| / ||g_rank||")
    ax.scatter([0.1], [0.0392], s=90, zorder=5, color=RED, label="文档取值：ρ=0.1 → λ≈0.04")
    ax.axhline(0.04, color=RED, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("目标梯度占比 ρ")
    ax.set_ylabel("推荐 ranking 权重 λ")
    ax.set_title("梯度范数探针定标\n||dKD/ds||=1.559e-2，||dRank/ds||=3.978e-2，ratio=0.392", fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")

    ax = axes[1]
    xs = np.arange(len(lam))
    ax.plot(xs, base, marker="o", color=BLUE, linewidth=1.8, label="BaseLoss（KD，左轴）")
    ax.set_ylabel("BaseLoss（≥700 步均值）", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(52.20, 52.27)
    ax2 = ax.twinx()
    ax2.plot(xs[1:], rank[1:], marker="s", color=ORANGE, linewidth=1.8, label="LogitRank（右轴）")
    ax2.set_ylabel("LogitRank", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.set_ylim(0.53, 0.67)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0\n(对照)", "0.04\n(选中)", "0.12", "0.4"])
    ax.set_xlabel("logits-ranking 权重 λ")
    ax.set_title("λ 扫描：主 loss 三档几乎完全重合\n（λ 放大 10 倍只有边际收益，因此取 λ=0.04）", fontsize=10.5)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="lower left")

    fig.tight_layout()
    out = FIG_DIR / "rank_loss_logitrank_lambda_calibration_20261014.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------------------
# 2. logits ranking: same-path 10-epoch control + trainer-path confound
# --------------------------------------------------------------------------------------
def fig_logitrank_same_path_control() -> None:
    epochs = list(range(1, 11))
    control = [78.4180, 78.9160, 79.0740, 79.3260, 79.2760, 79.3400, 79.5480, 79.5640, 79.4460, 79.6640]
    rankrun = [78.3120, 78.7800, 79.1800, 79.2960, 79.4260, 79.5080, 79.4780, 79.4920, 79.3720, 79.5620]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    ax = axes[0]
    ax.plot(epochs, control, marker="o", color=BLUE, linewidth=2.0, label="λ=0 对照（同路径 10ep）")
    ax.plot(epochs, rankrun, marker="s", color=RED, linewidth=2.0, label="logits-ranking λ=0.04（100ep 前 10 轮）")
    ax.annotate(f"ep10: {control[-1]:.3f}", xy=(10, control[-1]), xytext=(8.1, 79.78),
                fontsize=8.5, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate(f"ep10: {rankrun[-1]:.3f}", xy=(10, rankrun[-1]), xytext=(8.1, 79.26),
                fontsize=8.5, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.set_xlabel("epoch")
    ax.set_ylabel("ImageNet Top-1 (%)，每轮 50k 全量验证")
    ax.set_title("同一条 qat_launch 路径：λ=0 对照 vs logits-ranking\n"
                 "ep10 对照反而高 0.102 分，未见 ranking 收益", fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[1]
    labels = ["old train.py 路径\n(playground 同款入口)", "qat_launch 路径\n(λ=0 对照)", "qat_launch 路径\n(logits-ranking)"]
    values = [76.97, 78.418, 78.312]
    colors = [GRAY, BLUE, RED]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.3f}", ha="center", fontsize=9)
    ax.annotate("", xy=(1, 78.418), xytext=(0, 76.97),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(0.5, 77.6, "+1.45", ha="center", fontsize=10, color="black")
    ax.set_ylim(76.0, 79.1)
    ax.set_ylabel("epoch 0 Top-1 (%)")
    ax.set_title("100ep 对比的混杂因素：trainer 入口本身差 +1.45 分\n"
                 "（同数据、同 4 卡、同 effective batch 512、同 seed）", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "rank_loss_logitrank_same_path_control_20261014.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------------------
# 3. attention-relation ranking: prev-step lambda cliff
# --------------------------------------------------------------------------------------
def fig_attnrank_prevstep_lambda_cliff() -> None:
    lam = [0.0, 0.01, 0.03, 0.1, 0.3, 3.0, 10.0, 30.0]
    top1 = [69.3640, 69.4820, 69.6500, 69.4760, 68.7480, 34.4520, 10.8540, 6.4320]
    ce = [1.7161, 1.7055, 1.7364, 1.7532, 1.7530, 4.6223, 6.0520, 6.3022]
    base = [52.1206, 52.1530, 52.1525, 52.1510, 52.1489, 52.2000, 52.2219, 52.2337]
    pos = np.arange(len(lam))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    ax = axes[0]
    ax.plot(pos, top1, marker="o", color=RED, linewidth=2.0)
    for x, y in zip(pos, top1):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.6)
    ax.axvspan(-0.4, 4.4, color=GREEN, alpha=0.08)
    ax.text(2.0, 20, "可用区 λ≤0.3\n但仍无正收益", ha="center", fontsize=9, color=GREEN)
    ax.annotate("悬崖 0.3 → 3\n69.4% → 34.5%", xy=(5, 34.45), xytext=(3.4, 52),
                fontsize=9, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.set_xticks(pos)
    ax.set_xticklabels(["0", "0.01", "0.03", "0.1", "0.3", "3", "10", "30"])
    ax.set_ylim(0, 78)
    ax.set_xlabel("prev-step attention-rank 权重 λ")
    ax.set_ylabel("50k 全量验证 Top-1 (%)")
    ax.set_title("prev-step rank：λ 是一条悬崖\n100 个 optimizer step（epoch 0 的 2%）", fontsize=10.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(pos, ce, marker="o", color=ORANGE, linewidth=2.0, label="val CE（左轴）")
    ax.set_ylabel("val cross-entropy", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_ylim(1.4, 6.7)
    ax2 = ax.twinx()
    ax2.plot(pos, base, marker="s", color=BLUE, linewidth=2.0, label="train KD BaseLoss（右轴）")
    ax2.set_ylabel("train BaseLoss", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.set_ylim(52.10, 52.26)
    ax.set_xticks(pos)
    ax.set_xticklabels(["0", "0.01", "0.03", "0.1", "0.3", "3", "10", "30"])
    ax.set_xlabel("prev-step attention-rank 权重 λ")
    ax.set_title("训练日志完全不会报警\nBaseLoss 52.12→52.23（+0.2%），Top-1 69.4%→6.4%", fontsize=10.5)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="center left")

    fig.tight_layout()
    out = FIG_DIR / "rank_loss_attnrank_prevstep_lambda_cliff_20261014.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------------------
# 4. attention-relation ranking: anti-cheat interventions
# --------------------------------------------------------------------------------------
def fig_attnrank_anticheat() -> None:
    labels = [
        "λ=0\n参照",
        "λ=3\n原始 softplus",
        "λ=3\npre-quant",
        "λ=3\ndetached",
        "λ=3\n冻 s",
        "λ=3\n冻 move",
        "λ=3\n冻两者",
        "λ=3\nhinge m=0",
        "λ=10\n冻两者",
        "λ=30\nhinge m=0",
    ]
    top1 = [73.58, 38.48, 41.17, 38.41, 39.39, 39.34, 38.92, 58.39, 9.78, 9.72]
    colors = [GRAY, RED, BLUE, BLUE, BLUE, BLUE, BLUE, ORANGE, RED, RED]

    h10 = [2.6325, 1.9929, 2.0475, 2.0135, 2.0529, 2.0005, 2.0539, 3.5053, 1.7621, 3.6342]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9))
    ax = axes[0]
    bars = ax.bar(np.arange(len(labels)), top1, color=colors, width=0.62)
    for bar, value in zip(bars, top1):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylim(0, 82)
    ax.set_ylabel("6400 张 val Top-1 (%)")
    ax.set_title("堵住已知作弊旋钮救不回模型（λ=3：38.48 → 38.92）\n"
                 "hinge 能止住锐化，但 λ=3 仍只有 58.39%（-15 分）", fontsize=10.3)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(73.58, color=GRAY, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(9.35, 74.6, "λ=0 参照 73.58", ha="right", fontsize=8.5, color=GRAY)

    ax = axes[1]
    bars = ax.bar(np.arange(len(labels)), h10, color=colors, width=0.62)
    ax.axhline(2.6325, color=GRAY, linestyle="--", linewidth=1.2, label="λ=0 参照 H=2.63")
    ax.axhline(np.log(49), color=BLACK, linestyle=":", linewidth=1.2, label="49 key 最大熵 ln49=3.89")
    for bar, value in zip(bars, h10):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylim(1.5, 4.15)
    ax.set_ylabel("block10 注意力熵")
    ax.set_title("两种相反的退化方向都能把 ranking loss 做到 0\n"
                 "softplus → 锐化（熵↓）；hinge → 压平造并列（熵↑）", fontsize=10.3)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = FIG_DIR / "rank_loss_attnrank_anticheat_20261014.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------------------
# 5. attention-relation ranking: reference ablation at 2% epoch
# --------------------------------------------------------------------------------------
def fig_attnrank_reference_ablation() -> None:
    baseline = 73.58
    curves: List[Tuple[str, Sequence[float], Sequence[float], str, str]] = [
        ("prev-step 自己 + naive KL", [0.01, 0.03, 0.1, 0.3], [72.22, 59.47, 12.14, 1.47], ORANGE, "o"),
        ("FP teacher 固定锚 + KL", [0.003, 0.01, 0.03, 0.1], [74.30, 74.84, 74.59, 72.66], GREEN, "s"),
        ("FP teacher 固定锚 + rank", [0.3, 3.0, 10.0, 30.0], [75.03, 41.64, 4.52, 1.00], RED, "^"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.axhline(baseline, color=GRAY, linestyle="--", linewidth=1.3, label="λ=0 参照 73.58%")
    for name, xs, ys, color, marker in curves:
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2.0, markersize=7, label=name)
        best_i = int(np.argmax(ys))
        ax.scatter([xs[best_i]], [ys[best_i]], s=130, facecolors="none", edgecolors=color, linewidths=2.0, zorder=5)
        ax.annotate(f"{ys[best_i]:.2f}%", (xs[best_i], ys[best_i]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color=color)
    ax.set_xscale("log")
    ax.set_xlim(0.002, 60)
    ax.set_ylim(0, 80)
    ax.set_xlabel("约束权重 λ")
    ax.set_ylabel("6400 张 val Top-1 (%)")
    ax.set_title("参考系对照：110 个 optimizer step（epoch 0 的 2%）\n"
                 "固定 FP teacher 把可用权重区间放大 3~10 倍，但只是早期信号", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    out = FIG_DIR / "rank_loss_attnrank_reference_ablation_2pct_20261014.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------------------
# 6. attention-relation ranking: 3-epoch follow-up
# --------------------------------------------------------------------------------------
def fig_attnrank_3ep_followup() -> None:
    epochs = [1, 2, 3]
    runs = {
        "baseline λ=0": ([78.4180, 78.9160, 79.0740], GRAY),
        "FP teacher KL λ=0.01": ([78.1860, 78.5840, 78.7660], GREEN),
        "FP teacher rank λ=0.3": ([78.5040, 78.7680, 79.0320], RED),
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for name, (values, color) in runs.items():
        ax.plot(epochs, values, marker="o", color=color, linewidth=2.2, markersize=8, label=name)
        ax.annotate(f"{values[-1]:.3f}", (3, values[-1]), textcoords="offset points",
                    xytext=(8, -3), fontsize=9.5, color=color)
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.9, 3.45)
    ax.set_ylim(77.95, 79.35)
    ax.set_xlabel("epoch")
    ax.set_ylabel("50k 全量验证 Top-1 (%)")
    ax.set_title("3-epoch 参考系复现：2% 处的 +1.3~1.5 分早期信号消失\n"
                 "ep3：baseline 79.074 > teacher-rank 79.032 > teacher-KL 78.766", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "rank_loss_attnrank_3ep_followup_20261014.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------------------
# 7. one-page summary
# --------------------------------------------------------------------------------------
def fig_summary() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))

    ax = axes[0, 0]
    labels = ["baseline\nλ=0", "teacher KL\nλ=0.01", "teacher rank\nλ=0.3"]
    values = [79.074, 78.766, 79.032]
    bars = ax.bar(labels, values, color=[GRAY, GREEN, RED], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.3f}", ha="center", fontsize=9.5)
    ax.set_ylim(78.4, 79.3)
    ax.set_ylabel("epoch 3 Top-1 (%)")
    ax.set_title("A. attention 约束 3 epoch 复现（50k val）\n早期信号消失，baseline 反超", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    lam = [0.0, 0.01, 0.03, 0.1, 0.3, 3.0, 10.0, 30.0]
    top1 = [69.3640, 69.4820, 69.6500, 69.4760, 68.7480, 34.4520, 10.8540, 6.4320]
    ax.plot(range(len(lam)), top1, marker="o", color=RED, linewidth=2.0)
    ax.set_xticks(range(len(lam)))
    ax.set_xticklabels(["0", "0.01", "0.03", "0.1", "0.3", "3", "10", "30"])
    ax.set_ylim(0, 78)
    ax.set_xlabel("λ")
    ax.set_ylabel("Top-1 (%)")
    ax.set_title("B. prev-step attention rank 的 λ 悬崖\n69.4% → 34.5% → 6.4%", fontsize=10.5)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    names = ["参照\nλ=0", "原始\nλ=3", "冻结 s+\nmove", "hinge\nλ=3"]
    vals = [73.58, 38.48, 38.92, 58.39]
    bars = ax.bar(names, vals, color=[GRAY, RED, BLUE, ORANGE], width=0.55)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.2f}", ha="center", fontsize=9.5)
    ax.set_ylim(0, 82)
    ax.set_ylabel("Top-1 (%)")
    ax.set_title("C. 反作弊：堵住 s 和 move_qkx 无用\nhinge 只把崩溃从 38% 抬到 58%", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    epochs = list(range(1, 11))
    control = [78.4180, 78.9160, 79.0740, 79.3260, 79.2760, 79.3400, 79.5480, 79.5640, 79.4460, 79.6640]
    rankrun = [78.3120, 78.7800, 79.1800, 79.2960, 79.4260, 79.5080, 79.4780, 79.4920, 79.3720, 79.5620]
    ax.plot(epochs, control, marker="o", color=BLUE, linewidth=2.0, label="λ=0 同路径对照")
    ax.plot(epochs, rankrun, marker="s", color=RED, linewidth=2.0, label="logits-ranking λ=0.04")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Top-1 (%)")
    ax.set_title("D. logits ranking 同路径 10ep 对照\nep10 对照 79.664 vs ranking 79.562", fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Rank loss 实验总览（2026-09-23 ~ 2026-10-10）：目前没有任何一个 rank 变体拿到稳定正收益",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = FIG_DIR / "rank_loss_experiments_summary_20261014.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    fig_logitrank_lambda_calibration()
    fig_logitrank_same_path_control()
    fig_attnrank_prevstep_lambda_cliff()
    fig_attnrank_anticheat()
    fig_attnrank_reference_ablation()
    fig_attnrank_3ep_followup()
    fig_summary()


if __name__ == "__main__":
    main()
