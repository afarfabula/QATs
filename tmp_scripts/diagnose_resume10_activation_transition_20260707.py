#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


QATS = Path(__file__).resolve().parents[1]
OFQ = QATS / "third_party" / "OFQ"
sys.path.insert(0, str(QATS))
sys.path.insert(0, str(OFQ))

import qat_launch as ql  # noqa: E402


DEFAULT_CHECKPOINTS = {
    "ckpt10_start": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar",
    "phase1h_w4a8_ckpt1": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_w4a8_to_w4a4_20260707/checkpoint-1.pth.tar",
    "phase1h_w4a4_ckpt2": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_w4a8_to_w4a4_20260707/checkpoint-2.pth.tar",
    "phase1i_w4a6_ckpt2": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_actcurr_w4a8_w4a6_w4a4_20260707/checkpoint-2.pth.tar",
    "phase1j_a4_recon_ckpt1": "/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_a8ckpt1_a4_recon100_gate_20260707/checkpoint-1.pth.tar",
}


def build_args(argv: Sequence[str]) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["qat_launch.py", *argv]
        return ql.parse_args()
    finally:
        sys.argv = old_argv


def runtime_for_bits(args: argparse.Namespace, wbits: int, abits: int) -> argparse.Namespace:
    argv = [
        "--method", "ofq",
        "--stage", "train",
        "--config", str(QATS / "third_party/OFQ/configs/swin_t_imagenet.attn_q.yml"),
        "--model", "swin_t",
        "--data", args.data,
        "--dataset-format", "parquet",
        "--output", "/tmp/qats_diag",
        "--experiment", f"diag_w{wbits}a{abits}",
        "--devices", "0",
        "--nproc-per-node", "1",
        "--master-port", "30600",
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
        "--wbits", str(wbits),
        "--abits", str(abits),
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


def first_batch(runtime_args):
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
    loader = ql.create_loader_compat(
        dataset,
        input_size=data_config["input_size"],
        batch_size=runtime_args.batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=runtime_args.workers,
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=False,
    )
    try:
        images, targets = next(iter(loader))
        return images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
    finally:
        ql.shutdown_data_loader(loader)


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


def build_teacher(runtime_args):
    import src  # noqa: F401

    teacher = ql.create_ofq_teacher_model(runtime_args).cuda().eval()
    return teacher


def tensor_summary(tensor: torch.Tensor) -> Dict[str, float]:
    data = tensor.detach().float().flatten()
    if data.numel() == 0:
        return {"numel": 0}
    return {
        "numel": int(data.numel()),
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()) if data.numel() > 1 else 0.0,
        "min": float(data.min().item()),
        "p05": float(torch.quantile(data, 0.05).item()),
        "p50": float(torch.quantile(data, 0.50).item()),
        "p95": float(torch.quantile(data, 0.95).item()),
        "max": float(data.max().item()),
    }


def iter_quantizers(model) -> Iterable[Tuple[str, str, object]]:
    for module_name, module in model.named_modules():
        for attr in ("input_quant_fn", "quant_x_4_qkv", "quan_a_qkx_fn", "quan_a_v_fn", "quan_a_softmax_fn"):
            quantizer = getattr(module, attr, None)
            if quantizer is not None and hasattr(quantizer, "bit"):
                yield module_name, attr, quantizer


def summarize_quantizers(model) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    grouped: Dict[str, List[torch.Tensor]] = {}
    for module_name, attr, quantizer in iter_quantizers(model):
        scale = getattr(quantizer, "s", None)
        if scale is None:
            continue
        scale_data = scale.detach().float().abs().flatten().cpu()
        key = attr
        grouped.setdefault(key, []).append(scale_data)
        row = {
            "module": module_name,
            "attr": attr,
            "bit": int(getattr(quantizer, "bit", -1)),
            "all_positive": bool(getattr(quantizer, "all_positive", False)),
            **{f"scale_{k}": v for k, v in tensor_summary(scale_data).items()},
        }
        rows.append(row)
    summary = {}
    for attr, tensors in grouped.items():
        summary[attr] = tensor_summary(torch.cat(tensors))
    return summary, rows


def feature_metrics(model, teacher, images, layers: Sequence[str]) -> Dict[str, Dict[str, float]]:
    with torch.no_grad():
        with ql.capture_named_module_outputs(teacher, layers, detach=True) as teacher_features:
            teacher_logits = teacher(images)
        with ql.capture_named_module_outputs(model, layers, detach=True) as student_features:
            student_logits = model(images)
    metrics = {}
    for idx, layer in enumerate(layers):
        student = student_features[idx].float()
        ref = teacher_features[idx].float()
        diff = student - ref
        metrics[layer] = {
            "mse": float(diff.pow(2).mean().item()),
            "mae": float(diff.abs().mean().item()),
            "rel_l2": float(diff.norm().div(ref.norm().clamp_min(1e-12)).item()),
            "student_std": float(student.std().item()),
            "teacher_std": float(ref.std().item()),
        }
    if isinstance(teacher_logits, tuple):
        teacher_logits = teacher_logits[0]
    if isinstance(student_logits, tuple):
        student_logits = student_logits[0]
    teacher_prob = F.softmax(teacher_logits.float(), dim=1)
    student_log_prob = F.log_softmax(student_logits.float(), dim=1)
    metrics["logits"] = {
        "kl_student_teacher": float(F.kl_div(student_log_prob, teacher_prob, reduction="batchmean").item()),
        "teacher_conf_mean": float(teacher_prob.max(dim=1).values.mean().item()),
        "student_top1_agree": float((student_logits.argmax(dim=1) == teacher_logits.argmax(dim=1)).float().mean().item()),
    }
    return metrics


def run_case(name: str, checkpoint: str, bits: Sequence[int], args, images, teacher, layers):
    wbits, abits = int(bits[0]), int(bits[1])
    runtime = runtime_for_bits(args, wbits, abits)
    model = build_model(runtime)
    ql.setup_alpha(model, [(images, torch.zeros(images.shape[0], dtype=torch.long, device=images.device))], runtime, nullcontext)
    ql.strict_resume_checkpoint(model, checkpoint, optimizer=None, loss_scaler=None, lr_scheduler=None, model_ema=None, restore_rng=False)
    ql.set_fake_quant_bits(model, wbits, abits, rescale_lsq=True)
    with torch.no_grad():
        model(images)
    q_summary, q_rows = summarize_quantizers(model)
    f_metrics = feature_metrics(model, teacher, images, layers)
    return {
        "name": name,
        "checkpoint": checkpoint,
        "bits": {"wbits": wbits, "abits": abits},
        "feature_metrics": f_metrics,
        "quantizer_summary": q_summary,
        "quantizer_rows": q_rows,
    }


def parse_case(text: str) -> Tuple[str, str, Tuple[int, int]]:
    parts = text.split("=")
    if len(parts) != 2:
        raise ValueError(f"case must be name=checkpoint:wbits:abits, got {text!r}")
    name, rest = parts
    checkpoint, wbits, abits = rest.rsplit(":", 2)
    return name, checkpoint, (int(wbits), int(abits))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/tmp/imagenet1k_full_parquet")
    parser.add_argument("--teacher-checkpoint", default="/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--setup-alpha-batches", type=int, default=1)
    parser.add_argument("--layers", default="features.5.5,features.7.1")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--out-json", default=str(QATS / "docs/resume10_activation_transition_diag_20260707.json"))
    parser.add_argument("--out-tsv", default=str(QATS / "docs/resume10_activation_transition_diag_20260707.tsv"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run inside mlx worker login shell")
    torch.cuda.set_device(0)
    layers = tuple(layer.strip() for layer in args.layers.split(",") if layer.strip())
    base_runtime = runtime_for_bits(args, 4, 4)
    images, _targets = first_batch(base_runtime)
    teacher = build_teacher(base_runtime)

    cases = [parse_case(item) for item in args.case]
    if not cases:
        cases = [
            ("ckpt10_start_w4a4", DEFAULT_CHECKPOINTS["ckpt10_start"], (4, 4)),
            ("phase1h_w4a8_ckpt1_as_w4a8", DEFAULT_CHECKPOINTS["phase1h_w4a8_ckpt1"], (4, 8)),
            ("phase1h_w4a8_ckpt1_as_w4a4", DEFAULT_CHECKPOINTS["phase1h_w4a8_ckpt1"], (4, 4)),
            ("phase1h_w4a4_ckpt2", DEFAULT_CHECKPOINTS["phase1h_w4a4_ckpt2"], (4, 4)),
            ("phase1i_w4a6_ckpt2_as_w4a6", DEFAULT_CHECKPOINTS["phase1i_w4a6_ckpt2"], (4, 6)),
            ("phase1i_w4a6_ckpt2_as_w4a4", DEFAULT_CHECKPOINTS["phase1i_w4a6_ckpt2"], (4, 4)),
            ("phase1j_a4_recon_ckpt1", DEFAULT_CHECKPOINTS["phase1j_a4_recon_ckpt1"], (4, 4)),
        ]

    results = []
    for name, checkpoint, bits in cases:
        print(f"[diag] case={name} checkpoint={checkpoint} bits={bits}", flush=True)
        if not Path(checkpoint).is_file():
            print(f"[diag] skip missing checkpoint: {checkpoint}", flush=True)
            continue
        results.append(run_case(name, checkpoint, bits, args, images, teacher, layers))
        torch.cuda.empty_cache()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"layers": layers, "results": results}, indent=2), encoding="utf-8")

    out_tsv = Path(args.out_tsv)
    with out_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["case", "wbits", "abits", "metric", "value"])
        for result in results:
            for layer, metrics in result["feature_metrics"].items():
                for key, value in metrics.items():
                    writer.writerow([result["name"], result["bits"]["wbits"], result["bits"]["abits"], f"{layer}.{key}", value])
            for attr, metrics in result["quantizer_summary"].items():
                for key, value in metrics.items():
                    writer.writerow([result["name"], result["bits"]["wbits"], result["bits"]["abits"], f"quantizer.{attr}.{key}", value])
    print(f"[diag] wrote {out_json}")
    print(f"[diag] wrote {out_tsv}")


if __name__ == "__main__":
    main()
