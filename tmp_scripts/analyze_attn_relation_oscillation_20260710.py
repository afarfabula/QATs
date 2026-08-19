#!/usr/bin/env python3
"""Offline Swin attention-relation oscillation analysis for QAT checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


QATS = Path("/mlx_devbox/users/quyanyi/playground/QATs")
OFQ = QATS / "third_party" / "OFQ"
if str(OFQ) not in sys.path:
    sys.path.insert(0, str(OFQ))
if str(OFQ / "tools") not in sys.path:
    sys.path.insert(0, str(OFQ / "tools"))

from offline_attention_oscillation import (  # noqa: E402
    build_args,
    build_model,
    build_probe_loader,
    get_block_metas,
)


RUNS = {
    "original": {
        "ckpt_dir": "/tmp/qat_public_repro/ofq_resume10_to60_original_ofq_public_20260710",
        "status_tsv": str(QATS / "docs" / "ofq_resume10_to60_original_ofq_public_status_20260710.tsv"),
    },
    "scheme_c": {
        "ckpt_dir": "/tmp/qat_public_repro/ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709",
        "status_tsv": str(QATS / "docs" / "ofq_sparse_pulse_prevstep_refkl_c_50epoch_status_20260709.tsv"),
    },
}


@dataclass(frozen=True)
class HeadKey:
    run: str
    global_block: int
    stage: int
    block: int
    head: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline attention relation oscillation analysis")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-dir", default="/tmp/imagenet1k_full_parquet")
    parser.add_argument("--train-args", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--start", type=int, default=48)
    parser.add_argument("--end", type=int, default=60)
    parser.add_argument("--runs", default="original,scheme_c")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_status_tsv(path: str) -> Dict[int, float]:
    acc = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ckpt = int(row["checkpoint"].split("-")[-1])
            acc[ckpt] = float(row["acc1"])
    return acc


def ckpt_path(ckpt_dir: str, ckpt: int) -> str:
    return str(Path(ckpt_dir) / f"checkpoint-{ckpt}.pth.tar")


def ensure_checkpoints(run_name: str, ckpt_dir: str, start: int, end: int) -> None:
    missing = [ckpt for ckpt in range(start, end + 1) if not Path(ckpt_path(ckpt_dir, ckpt)).exists()]
    if missing:
        raise FileNotFoundError(f"{run_name} missing checkpoints: {missing[:10]}")


def tensor_js(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = p.reshape(-1).astype(np.float64)
    q = q.reshape(-1).astype(np.float64)
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def tensor_cosine_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.reshape(-1).astype(np.float64)
    q = q.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(p) * np.linalg.norm(q) + eps
    return float(1.0 - np.dot(p, q) / denom)


def topk_change(p: np.ndarray, q: np.ndarray, k: int) -> float:
    k = min(k, p.shape[-1])
    p_top = np.argpartition(-p, kth=k - 1, axis=-1)[..., :k]
    q_top = np.argpartition(-q, kth=k - 1, axis=-1)[..., :k]
    changes = []
    for row_p, row_q in zip(p_top.reshape(-1, k), q_top.reshape(-1, k)):
        inter = len(set(row_p.tolist()) & set(row_q.tolist()))
        changes.append(1.0 - inter / float(k))
    return float(np.mean(changes))


def entropy(attn: np.ndarray, eps: float = 1e-8) -> float:
    p = np.maximum(attn.astype(np.float64), eps)
    p = p / p.sum(axis=-1, keepdims=True)
    ent = -np.sum(p * np.log(p), axis=-1)
    return float(np.mean(ent))


def collect_relation_summaries(model, loader, device: torch.device, max_samples: int) -> List[np.ndarray]:
    block_metas = get_block_metas()
    sums: List[torch.Tensor | None] = [None for _ in block_metas]
    counts = [0 for _ in block_metas]
    seen = 0
    with torch.no_grad():
        for images, _ in loader:
            if not isinstance(images, torch.Tensor):
                images = images[0]
            if seen >= max_samples:
                break
            if seen + images.shape[0] > max_samples:
                images = images[: max_samples - seen]
            seen += int(images.shape[0])
            images = images.to(device, non_blocking=True)
            _, attn_list = model(images)
            for block_idx, attn in enumerate(attn_list):
                if attn is None:
                    continue
                # attn: [batch*num_windows, heads, tokens, tokens]
                cur = attn.detach().float().sum(dim=0).cpu()
                if sums[block_idx] is None:
                    sums[block_idx] = cur
                else:
                    sums[block_idx] += cur
                counts[block_idx] += int(attn.shape[0])
            del images, attn_list
            torch.cuda.empty_cache()
    if seen <= 0:
        raise RuntimeError("no calibration samples were processed")
    out = []
    for block_idx, item in enumerate(sums):
        if item is None or counts[block_idx] <= 0:
            raise RuntimeError(f"missing attention for block {block_idx}")
        avg = (item / float(counts[block_idx])).numpy().astype(np.float32)
        out.append(avg)
    return out


def cache_path(output_dir: Path, run: str, ckpt: int) -> Path:
    return output_dir / "cache" / run / f"checkpoint-{ckpt}_relations.npz"


def save_relations(path: Path, relations: List[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"block_{i}": arr for i, arr in enumerate(relations)})


def load_relations(path: Path) -> List[np.ndarray]:
    data = np.load(path)
    keys = sorted(data.files, key=lambda x: int(x.split("_")[-1]))
    return [data[k] for k in keys]


def compute_pair_rows(
    run: str,
    ckpt_a: int,
    ckpt_b: int,
    rel_a: List[np.ndarray],
    rel_b: List[np.ndarray],
    acc: Dict[int, float],
    topk: int,
) -> List[Dict[str, object]]:
    rows = []
    metas = get_block_metas()
    acc_a = acc.get(ckpt_a, float("nan"))
    acc_b = acc.get(ckpt_b, float("nan"))
    acc_delta = acc_b - acc_a
    for block_idx, (a_block, b_block) in enumerate(zip(rel_a, rel_b)):
        meta = metas[block_idx]
        for head in range(a_block.shape[0]):
            a = a_block[head]
            b = b_block[head]
            ent_a = entropy(a)
            ent_b = entropy(b)
            js = tensor_js(a, b)
            cos = tensor_cosine_distance(a, b)
            topk_delta = topk_change(a, b, topk)
            ent_delta = abs(ent_b - ent_a)
            osc = js + cos + topk_delta + 0.1 * ent_delta
            rows.append({
                "run": run,
                "ckpt_a": ckpt_a,
                "ckpt_b": ckpt_b,
                "acc_a": acc_a,
                "acc_b": acc_b,
                "acc_delta": acc_delta,
                "global_block": block_idx,
                "stage": meta["stage"],
                "block": meta["block_in_stage"],
                "head": head,
                "js_divergence": js,
                "cosine_distance": cos,
                "topk_overlap_change": topk_delta,
                "entropy_a": ent_a,
                "entropy_b": ent_b,
                "entropy_delta": ent_delta,
                "oscillation_score": osc,
            })
    return rows


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def aggregate_heads(pair_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int], List[Dict[str, object]]] = {}
    for row in pair_rows:
        key = (str(row["run"]), int(row["global_block"]), int(row["head"]))
        grouped.setdefault(key, []).append(row)
    out = []
    for (run, global_block, head), rows in grouped.items():
        rows = sorted(rows, key=lambda r: (int(r["ckpt_a"]), int(r["ckpt_b"])))
        osc = [float(r["oscillation_score"]) for r in rows]
        acc_delta = [float(r["acc_delta"]) for r in rows]
        harmful_assoc = [o * max(0.0, -d) for o, d in zip(osc, acc_delta)]
        beneficial_assoc = [o * max(0.0, d) for o, d in zip(osc, acc_delta)]
        post_peak = [
            o * max(0.0, -d)
            for o, d, r in zip(osc, acc_delta, rows)
            if int(r["ckpt_a"]) >= 52
        ]
        vectors = []
        # reversal uses pairwise scalar oscillation slope as a cheap direction proxy.
        # The full relation deltas are intentionally not kept in memory for all heads.
        for i in range(1, len(osc)):
            vectors.append((osc[i] - osc[i - 1]) * (osc[i - 1]))
        reversal = float(np.mean([1.0 if v < 0 else 0.0 for v in vectors])) if vectors else 0.0
        first = rows[0]
        stage = int(first["stage"])
        block = int(first["block"])
        mean_osc = float(np.mean(osc))
        max_osc = float(np.max(osc))
        spike = max_osc / (mean_osc + 1e-12)
        harm = float(np.mean(harmful_assoc))
        bene = float(np.mean(beneficial_assoc))
        post = float(np.mean(post_peak)) if post_peak else 0.0
        corr_up = pearson(osc, acc_delta)
        corr_down = pearson(osc, [-d for d in acc_delta])
        harmful_score = mean_osc + 0.5 * spike + 2.0 * harm + 2.0 * post + 0.5 * max(0.0, corr_down) + 0.25 * reversal
        beneficial_score = mean_osc + 2.0 * bene + 0.5 * max(0.0, corr_up) - 0.5 * harm
        out.append({
            "run": run,
            "global_block": global_block,
            "stage": stage,
            "block": block,
            "head": head,
            "mean_js": float(np.mean([float(r["js_divergence"]) for r in rows])),
            "mean_cosine_distance": float(np.mean([float(r["cosine_distance"]) for r in rows])),
            "mean_topk_overlap_change": float(np.mean([float(r["topk_overlap_change"]) for r in rows])),
            "mean_entropy_delta": float(np.mean([float(r["entropy_delta"]) for r in rows])),
            "oscillation_score": mean_osc,
            "spike_score": spike,
            "reversal_score": reversal,
            "late_window_score": mean_osc,
            "post_peak_drop_score": post,
            "acc_up_corr": corr_up,
            "acc_down_corr": corr_down,
            "harmful_score": harmful_score,
            "beneficial_score": beneficial_score,
            "custom_subset_token": f"{global_block}:{head}",
            "stage_block_head": f"{stage}:{block}:{head}",
        })
    return out


def write_tsv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_float(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_report(
    path: Path,
    cli: argparse.Namespace,
    pair_rows: List[Dict[str, object]],
    summary_rows: List[Dict[str, object]],
    recommended: List[Dict[str, object]],
    avoid: List[Dict[str, object]],
    runs: List[str],
) -> None:
    top_osc = sorted(summary_rows, key=lambda r: float(r["oscillation_score"]), reverse=True)[:10]
    top_harm = sorted(summary_rows, key=lambda r: float(r["harmful_score"]), reverse=True)[:10]
    top_bene = sorted(summary_rows, key=lambda r: float(r["beneficial_score"]), reverse=True)[:10]
    rec_tokens = ",".join(str(r["custom_subset_token"]) for r in recommended[:5])
    avoid_tokens = ",".join(str(r["custom_subset_token"]) for r in avoid[:8])
    lines = []
    lines.append("# Attention relation 震荡检测报告")
    lines.append("")
    lines.append("## 数据来源")
    lines.append("")
    for run in runs:
        lines.append(f"- {run}: `{RUNS[run]['ckpt_dir']}`")
    lines.append(f"- checkpoint 范围: `{cli.start}` 到 `{cli.end}`")
    lines.append(f"- calibration 数据: `{cli.probe_dir}`")
    lines.append(f"- 样本数: `{cli.max_samples}`")
    lines.append(f"- top-k: `{cli.topk}`")
    lines.append("")
    lines.append("## 使用指标")
    lines.append("")
    lines.append("- `js_divergence`: 平均 attention relation 分布的 JS divergence")
    lines.append("- `cosine_distance`: 平均 relation matrix 的 cosine distance")
    lines.append("- `topk_overlap_change`: 每个 query 的 top-k key 集合变化")
    lines.append("- `entropy_delta`: head attention entropy 变化")
    lines.append("- `oscillation_score`: 综合震荡分数")
    lines.append("- `spike_score`: 最大震荡 / 平均震荡")
    lines.append("- `reversal_score`: 震荡强度方向来回变化的近似指标")
    lines.append("- `post_peak_drop_score`: checkpoint-52 以后与精度回落相乘的风险分数")
    lines.append("")

    def add_table(title: str, rows: List[Dict[str, object]], score_key: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| rank | run | global_block:head | stage:block:head | score | osc | spike | post_drop | acc_down_corr |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|")
        for i, row in enumerate(rows, 1):
            lines.append(
                f"| {i} | {row['run']} | {row['custom_subset_token']} | {row['stage_block_head']} | "
                f"{float(row[score_key]):.6f} | {float(row['oscillation_score']):.6f} | "
                f"{float(row['spike_score']):.6f} | {float(row['post_peak_drop_score']):.6f} | "
                f"{float(row['acc_down_corr']):.6f} |"
            )
        lines.append("")

    add_table("Top oscillating heads", top_osc, "oscillation_score")
    add_table("Top harmful heads", top_harm, "harmful_score")
    add_table("Top beneficial drift heads", top_bene, "beneficial_score")
    add_table("推荐用于 KL 的 heads", recommended[:10], "harmful_score")
    add_table("不建议约束的 heads", avoid[:10], "beneficial_score")
    lines.append("## 可复制 head 字符串")
    lines.append("")
    lines.append("本仓库已有 `custom_subset` 使用 `global_block:head` 编码，例如 `6:1,8:4`。")
    lines.append("")
    lines.append(f"- 推荐 KL heads: `custom_subset:{rec_tokens}`")
    lines.append(f"- 避让 heads: `custom_subset:{avoid_tokens}`")
    lines.append("")
    lines.append("## 对 10->110 dynamic sparse prev-step KL 的建议")
    lines.append("")
    lines.append("1. 不要再固定压制 49-52 的自然高点窗口。")
    lines.append("2. 只在验证精度低于 rolling best 且 head 的 harmful score 排名前列时触发 KL。")
    lines.append("3. 每次最多约束 1-2 个 head，weight 从 `1e-5` 起，上限 `3e-5`。")
    lines.append("4. 对 beneficial drift heads 设置 cooldown 或黑名单，避免压掉原版 OFQ 的自然上冲。")
    lines.append("5. 后续 10->110 的候选策略应偏向高点后稳定化，而不是高点形成前约束。")
    lines.append("")
    lines.append("## 完整性")
    lines.append("")
    lines.append(f"- pair metric rows: `{len(pair_rows)}`")
    lines.append(f"- head summary rows: `{len(summary_rows)}`")
    lines.append("- 本分析只做 checkpoint 前向和 attention 统计，没有启动训练。")
    lines.append("- 没有使用 soup / checkpoint averaging / ensemble。")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    cli = parse_args()
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cache").mkdir(exist_ok=True)
    runs = [x.strip() for x in cli.runs.split(",") if x.strip()]
    for run in runs:
        if run not in RUNS:
            raise ValueError(f"unknown run: {run}")
        ensure_checkpoints(run, RUNS[run]["ckpt_dir"], cli.start, cli.end)
        if not Path(RUNS[run]["status_tsv"]).exists():
            raise FileNotFoundError(RUNS[run]["status_tsv"])

    args = build_args(cli.train_args)
    args.data_dir = cli.probe_dir
    args.batch_size = cli.batch_size
    args.workers = cli.workers
    device = torch.device(cli.device)
    loader = build_probe_loader(args, cli.probe_dir)

    relation_paths: Dict[Tuple[str, int], Path] = {}
    for run in runs:
        for ckpt in range(cli.start, cli.end + 1):
            out_path = cache_path(output_dir, run, ckpt)
            relation_paths[(run, ckpt)] = out_path
            if out_path.exists() and not cli.force:
                continue
            print(f"[collect] run={run} checkpoint={ckpt} -> {out_path}", flush=True)
            model = build_model(args, ckpt_path(RUNS[run]["ckpt_dir"], ckpt), device)
            relations = collect_relation_summaries(model, loader, device, cli.max_samples)
            save_relations(out_path, relations)
            del model, relations
            torch.cuda.empty_cache()

    pair_rows: List[Dict[str, object]] = []
    for run in runs:
        acc = read_status_tsv(RUNS[run]["status_tsv"])
        for ckpt in range(cli.start, cli.end):
            rel_a = load_relations(relation_paths[(run, ckpt)])
            rel_b = load_relations(relation_paths[(run, ckpt + 1)])
            pair_rows.extend(compute_pair_rows(run, ckpt, ckpt + 1, rel_a, rel_b, acc, cli.topk))

    pair_fields = [
        "run", "ckpt_a", "ckpt_b", "acc_a", "acc_b", "acc_delta",
        "global_block", "stage", "block", "head",
        "js_divergence", "cosine_distance", "topk_overlap_change",
        "entropy_a", "entropy_b", "entropy_delta", "oscillation_score",
    ]
    write_tsv(output_dir / "checkpoint_pair_metrics.tsv", pair_rows, pair_fields)

    summary_rows = aggregate_heads(pair_rows)
    summary_fields = [
        "run", "global_block", "stage", "block", "head",
        "mean_js", "mean_cosine_distance", "mean_topk_overlap_change", "mean_entropy_delta",
        "oscillation_score", "spike_score", "reversal_score", "late_window_score",
        "post_peak_drop_score", "acc_up_corr", "acc_down_corr",
        "harmful_score", "beneficial_score", "custom_subset_token", "stage_block_head",
    ]
    write_tsv(output_dir / "head_oscillation_summary.tsv", summary_rows, summary_fields)
    with (output_dir / "head_oscillation_summary.json").open("w") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    # Prefer heads that are harmful in either run. The avoid list is the strongest
    # beneficial-drift list directly, even when harmful_score is also nonzero.
    recommended = sorted(
        [r for r in summary_rows if float(r["harmful_score"]) > float(r["beneficial_score"])],
        key=lambda r: float(r["harmful_score"]),
        reverse=True,
    )[:20]
    avoid = sorted(
        summary_rows,
        key=lambda r: float(r["beneficial_score"]),
        reverse=True,
    )[:20]
    write_tsv(output_dir / "recommended_kl_heads.tsv", recommended, summary_fields)
    write_tsv(output_dir / "avoid_kl_heads.tsv", avoid, summary_fields)
    write_report(output_dir / "analysis_report.md", cli, pair_rows, summary_rows, recommended, avoid, runs)

    manifest = {
        "runs": runs,
        "start": cli.start,
        "end": cli.end,
        "max_samples": cli.max_samples,
        "topk": cli.topk,
        "outputs": [
            "head_oscillation_summary.tsv",
            "head_oscillation_summary.json",
            "recommended_kl_heads.tsv",
            "avoid_kl_heads.tsv",
            "checkpoint_pair_metrics.tsv",
            "analysis_report.md",
        ],
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
