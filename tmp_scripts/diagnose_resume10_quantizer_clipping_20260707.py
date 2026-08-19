#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


QATS = Path(__file__).resolve().parents[1]
OFQ = QATS / "third_party" / "OFQ"
sys.path.insert(0, str(QATS))
sys.path.insert(0, str(OFQ))

import qat_launch as ql  # noqa: E402


DEFAULT_CKPT10 = "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar"
DEFAULT_PHASE1F = "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar"
DEFAULT_PHASE1W = "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actpercentile_prerecon_gate_20260707/checkpoint-1.pth.tar"


def build_args(argv: Sequence[str]) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["qat_launch.py", *argv]
        return ql.parse_args()
    finally:
        sys.argv = old_argv


def runtime_for_args(args: argparse.Namespace) -> argparse.Namespace:
    argv = [
        "--method", "ofq",
        "--stage", "train",
        "--config", str(QATS / "third_party/OFQ/configs/swin_t_imagenet.attn_q.yml"),
        "--model", "swin_t",
        "--data", args.data,
        "--dataset-format", "parquet",
        "--output", "/tmp/qats_quant_clip_diag",
        "--experiment", "diag_quant_clip",
        "--devices", "0",
        "--nproc-per-node", "1",
        "--master-port", "30611",
        "--model-type", "swin",
        "--teacher", "swin_t",
        "--teacher-type", "swin",
        "--teacher-checkpoint", args.teacher_checkpoint,
        "--teacher-pretrained",
        "--epochs", "1",
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--lr", "1e-5",
        "--min-lr", "5e-6",
        "--weight-decay", "0.0",
        "--wbits", "4",
        "--abits", "4",
        "--wq-mode", "statsq",
        "--aq-mode", "lsq",
        "--wq-per-channel",
        "--aq-per-channel",
        "--aq-clip-learnable",
        "--pretrained",
        "--pretrained-initialized",
        "--use-kd",
        "--kd-hard-and-soft", "0",
        "--teacher-soft-temperature", "2.75",
        "--quantized",
        "--qk-reparam",
        "--qk-reparam-type", "0",
        "--extra-arg=--smoothing", "--extra-arg=0.1",
        "--extra-arg=--mixup", "--extra-arg=0.0",
        "--extra-arg=--cutmix", "--extra-arg=0.0",
        "--extra-arg=--aa", "--extra-arg=rand-m9-mstd0.5-inc1",
        "--extra-arg=--color-jitter", "--extra-arg=0.4",
        "--extra-arg=--reprob", "--extra-arg=0.25",
        "--extra-arg=--seed", "--extra-arg=42",
    ]
    parsed = build_args(argv)
    runtime = ql.build_ofq_runtime_config(parsed)
    runtime.local_rank = 0
    runtime.rank = 0
    runtime.world_size = 1
    runtime.distributed = False
    runtime.device = "cuda:0"
    runtime.prefetcher = False
    runtime.no_prefetcher = True
    runtime.setup_alpha_batches = args.setup_alpha_batches
    return runtime


def build_loader(runtime_args, args):
    data_config = ql.resolve_data_config(vars(runtime_args), model=None, verbose=False)
    dataset = ql.create_dataset_compat(
        runtime_args.dataset,
        root=runtime_args.data_dir,
        split=runtime_args.val_split,
        is_training=False,
        batch_size=runtime_args.batch_size,
        subset_ratio=runtime_args.subset_ratio,
        rank=0,
        world_size=1,
    )
    return ql.create_loader_compat(
        dataset,
        input_size=data_config["input_size"],
        batch_size=runtime_args.batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=False,
    )


def build_model(runtime_args):
    import src  # noqa: F401

    model = ql.create_model(
        runtime_args.model,
        drop_path=runtime_args.drop_path,
        num_classes=runtime_args.num_classes,
        pretrained=runtime_args.pretrained,
        qqkkvv=False,
    )
    model = ql.get_ofq_qat_model(model, runtime_args)
    return model.cuda().eval()


def parse_case(text: str) -> Tuple[str, str]:
    parts = text.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"case must be name=checkpoint, got {text!r}")
    return parts[0], parts[1]


def iter_activation_quantizers(model, module_names: Sequence[str]) -> Iterable[Tuple[str, torch.nn.Module]]:
    wanted = tuple(str(name) for name in module_names if str(name))
    for name, module in model.named_modules():
        if not ql.is_activation_quantizer_module_name(name):
            continue
        if wanted and not ql.parameter_belongs_to_any_module(name, wanted):
            continue
        if getattr(module, "s", None) is None:
            continue
        yield name, module


def channel_dim_for_scale(values: torch.Tensor, scale: torch.Tensor) -> Optional[int]:
    if scale.numel() <= 1:
        return None
    if values.ndim in {2, 3, 4} and scale.numel() == values.shape[-2]:
        return values.ndim - 2
    if values.ndim in {2, 3, 4} and scale.numel() == values.shape[-1]:
        return values.ndim - 1
    return None


def flatten_for_scale(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    channel_dim = channel_dim_for_scale(values, scale)
    if channel_dim is None:
        return values.reshape(1, -1)
    return values.movedim(channel_dim, 0).reshape(scale.numel(), -1)


def sample_flattened(data: torch.Tensor, max_values: int = 1048576) -> torch.Tensor:
    flat = data.reshape(-1)
    if flat.numel() <= max_values:
        return flat
    stride = max(1, flat.numel() // max_values)
    return flat[::stride][:max_values]


def channelwise_quantile(flattened: torch.Tensor, q: float, max_values_per_channel: int = 65536) -> torch.Tensor:
    if flattened.shape[1] <= max_values_per_channel:
        return torch.quantile(flattened, q, dim=1)
    stride = max(1, flattened.shape[1] // max_values_per_channel)
    sampled = flattened[:, ::stride][:, :max_values_per_channel]
    return torch.quantile(sampled, q, dim=1)


def summarize_quantizer_input(quantizer, x: torch.Tensor, percentiles: Sequence[float]) -> Optional[Dict[str, float]]:
    scale = getattr(quantizer, "s", None)
    if scale is None:
        return None
    threshold = float(getattr(quantizer, "thd_pos", 0.0))
    if threshold <= 0:
        return None
    x_float = x.detach().float()
    if bool(getattr(quantizer, "all_positive", False)):
        values = x_float.clamp_min(0.0)
    else:
        values = x_float.abs()
    flattened = flatten_for_scale(values, scale.detach().float())
    scale_flat = scale.detach().float().abs().reshape(-1)
    if scale_flat.numel() == 1 and flattened.shape[0] != 1:
        scale_flat = scale_flat.expand(flattened.shape[0])
    if flattened.shape[0] != scale_flat.numel():
        return None
    denom = (scale_flat[:, None] * threshold).clamp_min(1e-12)
    normalized = flattened / denom
    clipped = normalized > 1.0
    excess = (flattened - denom).clamp_min(0.0)
    quantized = torch.clamp(torch.round(flattened / scale_flat[:, None].clamp_min(1e-12)), 0.0, threshold)
    recon = quantized * scale_flat[:, None]
    abs_error = (flattened - recon).abs()
    normalized_sample = sample_flattened(normalized)
    abs_error_sample = sample_flattened(abs_error)
    row: Dict[str, float] = {
        "bit": float(getattr(quantizer, "bit", -1)),
        "all_positive": float(bool(getattr(quantizer, "all_positive", False))),
        "channels": float(scale_flat.numel()),
        "tokens": float(flattened.numel()),
        "scale_mean": float(scale_flat.mean().item()),
        "scale_min": float(scale_flat.min().item()),
        "scale_max": float(scale_flat.max().item()),
        "input_abs_mean": float(flattened.mean().item()),
        "input_abs_max": float(flattened.max().item()),
        "normalized_p95": float(torch.quantile(normalized_sample, 0.95).item()),
        "normalized_p99": float(torch.quantile(normalized_sample, 0.99).item()),
        "normalized_max": float(normalized.max().item()),
        "clip_rate": float(clipped.float().mean().item()),
        "clip_excess_mean": float(excess.mean().item()),
        "clip_excess_max": float(excess.max().item()),
        "quant_abs_error_mean": float(abs_error.mean().item()),
        "quant_abs_error_p99": float(torch.quantile(abs_error_sample, 0.99).item()),
    }
    for percentile in percentiles:
        target = channelwise_quantile(flattened, float(percentile)) / threshold
        ratio = target / scale_flat.clamp_min(1e-12)
        key = str(percentile).replace(".", "p")
        row[f"target_ratio_{key}_mean"] = float(ratio.mean().item())
        row[f"target_ratio_{key}_min"] = float(ratio.min().item())
        row[f"target_ratio_{key}_max"] = float(ratio.max().item())
    return row


def collect_case(case_name: str, checkpoint: str, runtime_args, args, layers: Sequence[str]) -> Dict[str, object]:
    model = build_model(runtime_args)
    loader = build_loader(runtime_args, args)
    try:
        first_images = None
        for batch_idx, (images, targets) in enumerate(loader):
            if batch_idx >= args.batches:
                break
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            if first_images is None:
                first_images = images
                ql.setup_alpha(model, [(images, targets)], runtime_args, nullcontext)
                ql.strict_resume_checkpoint(
                    model,
                    checkpoint,
                    optimizer=None,
                    loss_scaler=None,
                    lr_scheduler=None,
                    model_ema=None,
                    restore_rng=False,
                )
                ql.set_fake_quant_bits(model, 4, 4, rescale_lsq=True)
                model.eval()
            with torch.no_grad():
                model(images)
        if first_images is None:
            raise RuntimeError("diagnostic loader yielded no batches")

        rows: List[Dict[str, object]] = []
        handles = []
        captured: Dict[str, List[Dict[str, float]]] = {}

        def make_hook(name: str, module):
            def hook(_module, inputs):
                if not inputs or not torch.is_tensor(inputs[0]):
                    return
                row = summarize_quantizer_input(module, inputs[0], args.percentiles)
                if row is not None:
                    captured.setdefault(name, []).append(row)
            return hook

        for name, module in iter_activation_quantizers(model, layers):
            handles.append(module.register_forward_pre_hook(make_hook(name, module)))
        try:
            loader2 = build_loader(runtime_args, args)
            try:
                for batch_idx, (images, _targets) in enumerate(loader2):
                    if batch_idx >= args.batches:
                        break
                    images = images.cuda(non_blocking=True)
                    with torch.no_grad():
                        model(images)
            finally:
                ql.shutdown_data_loader(loader2)
        finally:
            for handle in handles:
                handle.remove()

        for name, batch_rows in captured.items():
            merged: Dict[str, object] = {"case": case_name, "checkpoint": checkpoint, "quantizer": name}
            keys = sorted(batch_rows[0].keys())
            for key in keys:
                values = [float(row[key]) for row in batch_rows]
                merged[key] = sum(values) / len(values)
            rows.append(merged)
        rows.sort(key=lambda row: (float(row.get("clip_rate", 0.0)), float(row.get("quant_abs_error_mean", 0.0))), reverse=True)
        return {"case": case_name, "checkpoint": checkpoint, "rows": rows}
    finally:
        ql.shutdown_data_loader(loader)
        del model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/tmp/imagenet1k_full_parquet")
    parser.add_argument("--teacher-checkpoint", default="/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--setup-alpha-batches", type=int, default=1)
    parser.add_argument("--layers", default="features.5.5,features.7.1")
    parser.add_argument("--percentiles", type=float, nargs="+", default=[0.99, 0.999])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--out-json", default=str(QATS / "docs/resume10_quantizer_clipping_diag_20260707.json"))
    parser.add_argument("--out-tsv", default=str(QATS / "docs/resume10_quantizer_clipping_diag_20260707.tsv"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run inside mlx worker login shell")
    torch.cuda.set_device(0)
    layers = tuple(layer.strip() for layer in args.layers.split(",") if layer.strip())
    runtime = runtime_for_args(args)
    cases = [parse_case(item) for item in args.case]
    if not cases:
        cases = [
            ("ckpt10_start", DEFAULT_CKPT10),
            ("phase1f_best", DEFAULT_PHASE1F),
            ("phase1w_actpercentile", DEFAULT_PHASE1W),
        ]

    results = []
    for name, checkpoint in cases:
        if not Path(checkpoint).is_file():
            print(f"[diag] skip missing checkpoint: {name} {checkpoint}", flush=True)
            continue
        print(f"[diag] case={name} checkpoint={checkpoint}", flush=True)
        results.append(collect_case(name, checkpoint, runtime, args, layers))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"layers": layers, "batches": args.batches, "results": results}, indent=2), encoding="utf-8")

    out_tsv = Path(args.out_tsv)
    fieldnames = ["case", "checkpoint", "quantizer"]
    for result in results:
        if result["rows"]:
            fieldnames.extend(key for key in result["rows"][0].keys() if key not in fieldnames)
            break
    seen = set()
    ordered = []
    for key in fieldnames:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    with out_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for result in results:
            for row in result["rows"]:
                writer.writerow(row)
    print(f"[diag] wrote {out_json}")
    print(f"[diag] wrote {out_tsv}")


if __name__ == "__main__":
    main()
