#!/usr/bin/env python3
"""Full-validation class/logit diagnosis for resume10 strict W4A4 checkpoints."""

from __future__ import annotations

import argparse
import csv
import functools
import io
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from timm.data import create_transform
from timm.models import create_model, safe_model_name


QATS = Path(__file__).resolve().parents[1]
OFQ_ROOT = QATS / "third_party" / "OFQ"
for path in (str(QATS), str(OFQ_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import qat_launch as ql  # noqa: E402


DEFAULT_CHECKPOINTS = {
    "ckpt10": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar",
    "phase2w": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_vartrust_epoch1_paramsinlate_gate_20260707/checkpoint-2.pth.tar",
    "best250": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_250upd_gate_20260708/checkpoint-3.pth.tar",
    "u300": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_paramsinlate_ckpt2_300upd_gate_20260708/checkpoint-3.pth.tar",
}


class IndexedImageNetParquetEvalDataset(torch.utils.data.IterableDataset):
    def __init__(self, root: str, split: str, transform=None, subset_ratio: float = 1.0):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.subset_ratio = float(subset_ratio)
        self.data_dir = os.path.join(root, "data") if os.path.isdir(os.path.join(root, "data")) else root
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"parquet data dir not found: {self.data_dir}")
        self.files = sorted(
            os.path.join(self.data_dir, item)
            for item in os.listdir(self.data_dir)
            if item.startswith(f"{split}-") and item.endswith(".parquet")
        )
        if not self.files:
            raise FileNotFoundError(f"no parquet files for split={split} under {self.data_dir}")

        self._segments: List[Tuple[int, int, int, int]] = []
        total_rows = 0
        for file_idx, path in enumerate(self.files):
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                rows = pf.metadata.row_group(rg_idx).num_rows
                self._segments.append((total_rows, total_rows + rows, file_idx, rg_idx))
                total_rows += rows
        self._total_rows = total_rows
        if not 0.0 < self.subset_ratio <= 1.0:
            raise ValueError(f"subset_ratio must be in (0, 1], got {self.subset_ratio}")
        if self.subset_ratio < 1.0:
            self._total_rows = max(1, int(math.ceil(self._total_rows * self.subset_ratio)))
            self._segments = [segment for segment in self._segments if segment[0] < self._total_rows]

    def __len__(self):
        return self._total_rows

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        worker_start = self._total_rows * worker_id // num_workers
        worker_end = self._total_rows * (worker_id + 1) // num_workers
        handles = {}
        for seg_start, seg_end, file_idx, rg_idx in self._segments:
            start = max(seg_start, worker_start)
            end = min(seg_end, worker_end, self._total_rows)
            if start >= end:
                continue
            path = self.files[file_idx]
            pf = handles.get(path)
            if pf is None:
                pf = pq.ParquetFile(path)
                handles[path] = pf
            table = pf.read_row_group(rg_idx, columns=["image", "label"])
            cols = table.to_pydict()
            images = cols["image"]
            labels = cols["label"]
            local_start = start - seg_start
            local_end = end - seg_start
            for local_idx in range(local_start, local_end):
                image = Image.open(io.BytesIO(images[local_idx]["bytes"])).convert("RGB")
                target = int(labels[local_idx])
                if self.transform is not None:
                    image = self.transform(image)
                yield image, target, seg_start + local_idx


def parse_checkpoints(items: Iterable[str]) -> Dict[str, str]:
    result = dict(DEFAULT_CHECKPOINTS)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--checkpoint must be label=path, got {item!r}")
        label, path = item.split("=", 1)
        result[label.strip()] = path.strip()
    return result


def build_runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    argv = [
        "qat_launch.py",
        "--method",
        "ofq",
        "--stage",
        "train",
        "--config",
        str(QATS / "third_party" / "OFQ" / "configs" / "swin_t_imagenet.attn_q.yml"),
        "--model",
        "swin_t",
        "--data",
        args.data,
        "--dataset-format",
        "parquet",
        "--output",
        str(args.out_dir.parent),
        "--experiment",
        args.out_dir.name,
        "--devices",
        args.devices,
        "--nproc-per-node",
        "1",
        "--master-port",
        str(args.master_port),
        "--model-type",
        "swin",
        "--teacher",
        "swin_t",
        "--teacher-type",
        "swin",
        "--teacher-checkpoint",
        args.teacher_checkpoint,
        "--teacher-pretrained",
        "--epochs",
        "0",
        "--start-epoch",
        "0",
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--lr",
        "1e-5",
        "--min-lr",
        "1e-5",
        "--weight-decay",
        "0.0",
        "--wbits",
        "4",
        "--abits",
        "4",
        "--wq-mode",
        args.wq_mode,
        "--aq-mode",
        args.aq_mode,
        "--wq-per-channel",
        "--aq-per-channel",
        "--aq-clip-learnable",
        "--pretrained",
        "--pretrained-initialized",
        "--use-kd",
        "--kd-hard-and-soft",
        "0",
        "--teacher-soft-temperature",
        "2.75",
        "--quantized",
        "--qk-reparam-type",
        str(args.qk_reparam_type),
        "--amp",
        "--amp-dtype",
        "bf16",
        "--extra-arg=--eval-only",
        "--extra-arg=--smoothing",
        "--extra-arg=0.1",
        "--extra-arg=--mixup",
        "--extra-arg=0.0",
        "--extra-arg=--cutmix",
        "--extra-arg=0.0",
        "--extra-arg=--aa",
        "--extra-arg=rand-m9-mstd0.5-inc1",
        "--extra-arg=--color-jitter",
        "--extra-arg=0.4",
        "--extra-arg=--reprob",
        "--extra-arg=0.25",
        "--extra-arg=--log-interval",
        "--extra-arg=50",
        "--extra-arg=--seed",
        "--extra-arg=42",
    ]
    if args.qk_reparam:
        argv.append("--qk-reparam")
    with ql.patched_argv(argv):
        parsed = ql.parse_args()
    runtime_args = ql.build_ofq_runtime_config(parsed)
    runtime_args.world_size = 1
    runtime_args.distributed = False
    runtime_args.rank = 0
    runtime_args.local_rank = 0
    runtime_args.gpu_id = args.device_index
    runtime_args.device = f"cuda:{args.device_index}"
    runtime_args.visible_gpu = args.devices
    runtime_args.workers = args.workers
    runtime_args.batch_size = args.batch_size
    runtime_args.use_kd = False
    return runtime_args


def build_model(runtime_args: SimpleNamespace) -> nn.Module:
    qqkkvv = (
        runtime_args.kd_hard_and_soft in {2, 3}
        or runtime_args.teacher_qk_rel_weight > 0
        or runtime_args.teacher_qkv_rel_weight > 0
    )
    model = create_model(
        runtime_args.model,
        drop_path=runtime_args.drop_path,
        num_classes=runtime_args.num_classes,
        pretrained=runtime_args.pretrained,
        qqkkvv=qqkkvv,
    )
    if runtime_args.quantized:
        model = ql.get_ofq_qat_model(model, runtime_args)
    model.cuda()
    if runtime_args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def make_train_loader(runtime_args: SimpleNamespace, data_config: Dict[str, object]):
    dataset_train = ql.create_dataset_compat(
        runtime_args.dataset,
        root=runtime_args.data_dir,
        split=runtime_args.train_split,
        is_training=True,
        batch_size=runtime_args.batch_size,
        subset_ratio=runtime_args.subset_ratio,
    )
    return ql.create_loader_compat(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=runtime_args.batch_size,
        is_training=True,
        use_prefetcher=runtime_args.prefetcher,
        no_aug=False,
        re_prob=runtime_args.reprob,
        re_mode=runtime_args.remode,
        re_count=runtime_args.recount,
        re_split=runtime_args.resplit,
        scale=runtime_args.scale,
        ratio=runtime_args.ratio,
        hflip=runtime_args.hflip,
        vflip=runtime_args.vflip,
        color_jitter=runtime_args.color_jitter,
        auto_augment=runtime_args.aa,
        num_aug_splits=runtime_args.aug_splits,
        num_aug_repeats=runtime_args.num_aug_repeats,
        interpolation=runtime_args.train_interpolation or data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=runtime_args.workers,
        distributed=False,
        collate_fn=None,
        pin_memory=runtime_args.pin_mem,
        use_multi_epochs_loader=runtime_args.use_multi_epochs_loader,
    )


def make_eval_loader(runtime_args: SimpleNamespace, data_config: Dict[str, object]):
    transform = create_transform(
        input_size=data_config["input_size"],
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        crop_pct=data_config["crop_pct"],
    )
    dataset = IndexedImageNetParquetEvalDataset(
        root=runtime_args.data_dir,
        split=runtime_args.val_split,
        transform=transform,
        subset_ratio=runtime_args.subset_ratio,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=runtime_args.batch_size,
        num_workers=runtime_args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def evaluate_checkpoint(
    label: str,
    path: str,
    runtime_args: SimpleNamespace,
    data_config: Dict[str, object],
    loader_train,
    loader_eval,
    num_samples: int,
    amp_autocast,
) -> Dict[str, object]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    model = build_model(runtime_args)
    ql.setup_alpha(model, loader_train, runtime_args, amp_autocast)
    ql.strict_resume_checkpoint(
        model,
        path,
        optimizer=None,
        loss_scaler=None,
        lr_scheduler=None,
        model_ema=None,
        restore_rng=False,
        log_info=True,
    )
    model.eval()
    loss_fn = nn.CrossEntropyLoss().cuda()
    target_arr = torch.full((num_samples,), -1, dtype=torch.long)
    pred_arr = torch.full((num_samples,), -1, dtype=torch.long)
    top5_arr = torch.zeros((num_samples,), dtype=torch.bool)
    conf_arr = torch.zeros((num_samples,), dtype=torch.float32)
    margin_arr = torch.zeros((num_samples,), dtype=torch.float32)
    true_prob_arr = torch.zeros((num_samples,), dtype=torch.float32)
    true_logit_arr = torch.zeros((num_samples,), dtype=torch.float32)
    entropy_arr = torch.zeros((num_samples,), dtype=torch.float32)
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    seen = 0
    start = ql.time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(loader_eval):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            if runtime_args.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss = loss_fn(outputs, targets)
            probs = torch.softmax(outputs.float(), dim=1)
            top_values, top_indices = outputs.float().topk(5, dim=1)
            pred = top_indices[:, 0]
            top1 = pred.eq(targets)
            top5 = top_indices.eq(targets[:, None]).any(dim=1)
            true_prob = probs.gather(1, targets[:, None]).squeeze(1)
            conf = probs.gather(1, pred[:, None]).squeeze(1)
            margin = top_values[:, 0] - top_values[:, 1]
            true_logit = outputs.float().gather(1, targets[:, None]).squeeze(1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
            idx_cpu = indices.long().cpu()
            target_arr[idx_cpu] = targets.cpu()
            pred_arr[idx_cpu] = pred.cpu()
            top5_arr[idx_cpu] = top5.cpu()
            conf_arr[idx_cpu] = conf.cpu()
            margin_arr[idx_cpu] = margin.cpu()
            true_prob_arr[idx_cpu] = true_prob.cpu()
            true_logit_arr[idx_cpu] = true_logit.cpu()
            entropy_arr[idx_cpu] = entropy.cpu()
            batch = int(targets.numel())
            loss_sum += float(loss.detach().item()) * batch
            top1_correct += int(top1.sum().item())
            top5_correct += int(top5.sum().item())
            seen += batch
            if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                print(
                    f"{label}: batch={batch_idx + 1} seen={seen} "
                    f"top1={100.0 * top1_correct / max(seen, 1):.4f}"
                )
    missing = int((target_arr < 0).sum().item())
    if missing:
        raise RuntimeError(f"{label}: missing {missing} validation samples")
    wall = ql.time.time() - start
    result = {
        "label": label,
        "path": path,
        "metrics": {
            "loss": loss_sum / max(seen, 1),
            "top1": 100.0 * top1_correct / max(seen, 1),
            "top5": 100.0 * top5_correct / max(seen, 1),
            "samples": seen,
            "wall_seconds": wall,
        },
        "target": target_arr,
        "pred": pred_arr,
        "top5_correct": top5_arr,
        "conf": conf_arr,
        "margin": margin_arr,
        "true_prob": true_prob_arr,
        "true_logit": true_logit_arr,
        "entropy": entropy_arr,
    }
    print(
        f"{label}: Loss={result['metrics']['loss']:.4f} "
        f"Acc@1={result['metrics']['top1']:.4f} Acc@5={result['metrics']['top5']:.4f} "
        f"Samples={seen} Time={wall:.1f}s"
    )
    del model
    torch.cuda.empty_cache()
    return result


def tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.0
    return float(values[mask].float().mean().item())


def pair_summary(ref: Dict[str, object], cmp: Dict[str, object]) -> Dict[str, object]:
    ref_correct = ref["pred"].eq(ref["target"])
    cmp_correct = cmp["pred"].eq(cmp["target"])
    improved = (~ref_correct) & cmp_correct
    regressed = ref_correct & (~cmp_correct)
    same_correct = ref_correct & cmp_correct
    same_wrong = (~ref_correct) & (~cmp_correct)
    true_prob_delta = cmp["true_prob"] - ref["true_prob"]
    margin_delta = cmp["margin"] - ref["margin"]
    return {
        "ref": ref["label"],
        "cmp": cmp["label"],
        "samples": int(ref_correct.numel()),
        "ref_top1": float(ref["metrics"]["top1"]),
        "cmp_top1": float(cmp["metrics"]["top1"]),
        "delta_top1": float(cmp["metrics"]["top1"] - ref["metrics"]["top1"]),
        "improved": int(improved.sum().item()),
        "regressed": int(regressed.sum().item()),
        "net_flips": int(improved.sum().item() - regressed.sum().item()),
        "same_correct": int(same_correct.sum().item()),
        "same_wrong": int(same_wrong.sum().item()),
        "avg_true_prob_delta": float(true_prob_delta.float().mean().item()),
        "avg_margin_delta": float(margin_delta.float().mean().item()),
        "improved_true_prob_delta": tensor_mean(true_prob_delta, improved),
        "regressed_true_prob_delta": tensor_mean(true_prob_delta, regressed),
    }


def class_rows(ref: Dict[str, object], cmp: Dict[str, object], pair: str, num_classes: int) -> List[Dict[str, object]]:
    rows = []
    target = ref["target"]
    ref_correct = ref["pred"].eq(target)
    cmp_correct = cmp["pred"].eq(target)
    true_prob_delta = cmp["true_prob"] - ref["true_prob"]
    margin_delta = cmp["margin"] - ref["margin"]
    for cls in range(num_classes):
        mask = target.eq(cls)
        total = int(mask.sum().item())
        if not total:
            continue
        ref_ok = int((ref_correct & mask).sum().item())
        cmp_ok = int((cmp_correct & mask).sum().item())
        rows.append(
            {
                "pair": pair,
                "class": cls,
                "total": total,
                "ref_correct": ref_ok,
                "cmp_correct": cmp_ok,
                "delta_correct": cmp_ok - ref_ok,
                "ref_acc": 100.0 * ref_ok / total,
                "cmp_acc": 100.0 * cmp_ok / total,
                "avg_true_prob_delta": tensor_mean(true_prob_delta, mask),
                "avg_margin_delta": tensor_mean(margin_delta, mask),
            }
        )
    rows.sort(key=lambda item: (item["pair"], item["delta_correct"], item["avg_true_prob_delta"]))
    return rows


def confidence_rows(ref: Dict[str, object], cmp: Dict[str, object], pair: str) -> List[Dict[str, object]]:
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.01)]
    ref_correct = ref["pred"].eq(ref["target"])
    cmp_correct = cmp["pred"].eq(cmp["target"])
    true_prob_delta = cmp["true_prob"] - ref["true_prob"]
    rows = []
    for low, high in bins:
        mask = ref["conf"].ge(low) & ref["conf"].lt(high)
        total = int(mask.sum().item())
        if not total:
            continue
        improved = int((mask & (~ref_correct) & cmp_correct).sum().item())
        regressed = int((mask & ref_correct & (~cmp_correct)).sum().item())
        rows.append(
            {
                "pair": pair,
                "ref_conf_bin": f"[{low:.2f},{high:.2f})",
                "total": total,
                "ref_correct": int((mask & ref_correct).sum().item()),
                "cmp_correct": int((mask & cmp_correct).sum().item()),
                "improved": improved,
                "regressed": regressed,
                "net_flips": improved - regressed,
                "avg_true_prob_delta": tensor_mean(true_prob_delta, mask),
            }
        )
    return rows


def flip_rows(ref: Dict[str, object], cmp: Dict[str, object], pair: str, topn: int) -> List[Dict[str, object]]:
    ref_correct = ref["pred"].eq(ref["target"])
    cmp_correct = cmp["pred"].eq(cmp["target"])
    changed = ref_correct.ne(cmp_correct)
    indices = torch.nonzero(changed, as_tuple=False).flatten()
    true_prob_delta = cmp["true_prob"] - ref["true_prob"]
    order = torch.argsort(true_prob_delta[indices].abs(), descending=True)
    selected = indices[order[:topn]]
    rows = []
    for idx in selected.tolist():
        direction = "improved" if (not bool(ref_correct[idx]) and bool(cmp_correct[idx])) else "regressed"
        rows.append(
            {
                "pair": pair,
                "sample_index": idx,
                "target": int(ref["target"][idx].item()),
                "direction": direction,
                "ref_pred": int(ref["pred"][idx].item()),
                "cmp_pred": int(cmp["pred"][idx].item()),
                "ref_conf": float(ref["conf"][idx].item()),
                "cmp_conf": float(cmp["conf"][idx].item()),
                "ref_true_prob": float(ref["true_prob"][idx].item()),
                "cmp_true_prob": float(cmp["true_prob"][idx].item()),
                "true_prob_delta": float(true_prob_delta[idx].item()),
                "ref_margin": float(ref["margin"][idx].item()),
                "cmp_margin": float(cmp["margin"][idx].item()),
            }
        )
    return rows


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/tmp/imagenet1k_full_parquet")
    parser.add_argument("--out-dir", type=Path, default=QATS / "docs" / "resume10_logit_class_diag_20260708")
    parser.add_argument("--checkpoint", action="append", default=[], help="label=path override/addition")
    parser.add_argument("--labels", default="ckpt10,phase2w,best250,u300")
    parser.add_argument("--compare-label", default="best250")
    parser.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=30731)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--teacher-checkpoint", default="/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth")
    parser.add_argument("--wq-mode", default="statsq", choices=["statsq", "lsq"])
    parser.add_argument("--aq-mode", default="lsq", choices=["statsq", "lsq"])
    parser.add_argument("--qk-reparam", dest="qk_reparam", action="store_true", default=True)
    parser.add_argument("--no-qk-reparam", dest="qk_reparam", action="store_false")
    parser.add_argument("--qk-reparam-type", type=int, default=0)
    parser.add_argument("--flip-topn", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict W4A4 logit diagnosis")
    torch.cuda.set_device(args.device_index)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    checkpoints = parse_checkpoints(args.checkpoint)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    missing_labels = [label for label in labels if label not in checkpoints]
    if missing_labels:
        raise ValueError(f"unknown checkpoint labels: {missing_labels}")
    if args.compare_label not in labels:
        raise ValueError(f"--compare-label {args.compare_label!r} must be included in --labels")

    runtime_args = build_runtime_args(args)
    ql.random_seed(runtime_args.seed, 0)
    import src  # noqa: F401

    probe_model = build_model(runtime_args)
    data_config = ql.resolve_data_config(vars(runtime_args), model=probe_model, verbose=True)
    print(f"Model {safe_model_name(runtime_args.model)} probe created.")
    del probe_model
    torch.cuda.empty_cache()
    use_amp = bool(runtime_args.amp or runtime_args.native_amp)
    amp_dtype = torch.bfloat16 if runtime_args.amp_dtype == "bf16" else torch.float16
    amp_autocast = functools.partial(torch.amp.autocast, "cuda", dtype=amp_dtype) if use_amp else ql.contextlib.suppress
    loader_train = make_train_loader(runtime_args, data_config)
    dataset_eval, loader_eval = make_eval_loader(runtime_args, data_config)
    print(f"diagnosis labels={labels}")
    print(f"validation_samples={len(dataset_eval)}")
    print(f"train_shards={len([p for p in Path(args.data, 'data').glob('train-*.parquet')])}")
    print(f"validation_shards={len([p for p in Path(args.data, 'data').glob('validation-*.parquet')])}")

    results = {}
    for label in labels:
        results[label] = evaluate_checkpoint(
            label,
            checkpoints[label],
            runtime_args,
            data_config,
            loader_train,
            loader_eval,
            len(dataset_eval),
            amp_autocast,
        )

    target = results[labels[0]]["target"]
    for label in labels[1:]:
        if not torch.equal(target, results[label]["target"]):
            raise RuntimeError(f"target order mismatch for {label}")

    cmp = results[args.compare_label]
    summaries = []
    all_class_rows = []
    all_conf_rows = []
    all_flip_rows = []
    for ref_label in labels:
        if ref_label == args.compare_label:
            continue
        ref = results[ref_label]
        pair = f"{args.compare_label}_vs_{ref_label}"
        summaries.append(pair_summary(ref, cmp))
        all_class_rows.extend(class_rows(ref, cmp, pair, runtime_args.num_classes))
        all_conf_rows.extend(confidence_rows(ref, cmp, pair))
        all_flip_rows.extend(flip_rows(ref, cmp, pair, args.flip_topn))

    metrics = {label: results[label]["metrics"] for label in labels}
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoints": {label: checkpoints[label] for label in labels},
        "compare_label": args.compare_label,
        "metrics": metrics,
        "pair_summaries": summaries,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "class_tsv": str(out_dir / "class_delta.tsv"),
            "confidence_tsv": str(out_dir / "confidence_bins.tsv"),
            "flip_tsv": str(out_dir / "flip_cases.tsv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(out_dir / "class_delta.tsv", all_class_rows)
    write_tsv(out_dir / "confidence_bins.tsv", all_conf_rows)
    write_tsv(out_dir / "flip_cases.tsv", all_flip_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
