#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/mlx_devbox/users/quyanyi/playground/QATs")
CSV_PATH = ROOT / "docs/ofq_100ep_kl_vs_nokl_fullval_curve_20260715.csv"
OUT_PATH = ROOT / "docs/ofq_100ep_kl_vs_nokl_acc1_curve_20260715.png"


def load_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    by_run: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_run.setdefault(row["run"], []).append(row)
    for rows in by_run.values():
        rows.sort(key=lambda item: int(item["checkpoint"]))
    return by_run


def main() -> None:
    import matplotlib.pyplot as plt

    by_run = load_rows(CSV_PATH)
    labels = {
        "kl": "100ep late sparse prev-step KL",
        "no_kl": "100ep no-KL control",
    }
    colors = {
        "kl": "#2563eb",
        "no_kl": "#dc2626",
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for run in ("kl", "no_kl"):
        rows = by_run[run]
        x = [int(row["checkpoint"]) for row in rows]
        acc1 = [float(row["acc1"]) for row in rows]
        best = [float(row["best_so_far_acc1"]) for row in rows]
        axes[0].plot(x, acc1, label=labels[run], color=colors[run], linewidth=1.8)
        axes[1].plot(x, best, label=labels[run], color=colors[run], linewidth=1.8)

    axes[0].axvline(51, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].text(51.5, 77.85, "KL observe boundary", fontsize=9, color="#374151")
    axes[0].set_ylabel("Acc@1")
    axes[0].set_title("OFQ 100epoch from-pretrained: KL vs no-KL full validation trajectory")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].axhline(80.7720, color=colors["kl"], linestyle=":", linewidth=1.0, alpha=0.8)
    axes[1].axhline(80.7920, color=colors["no_kl"], linestyle=":", linewidth=1.0, alpha=0.8)
    axes[1].set_xlabel("Checkpoint / epoch")
    axes[1].set_ylabel("Best-so-far Acc@1")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
