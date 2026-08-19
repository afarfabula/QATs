#!/usr/bin/env python3
"""Create single-checkpoint candidates using a small validation-gradient proxy."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn

QATS = Path(__file__).resolve().parents[1]
if str(QATS / "tmp_scripts") not in sys.path:
    sys.path.insert(0, str(QATS / "tmp_scripts"))

import diagnose_resume10_logit_classes_20260708 as diag  # noqa: E402


def load_checkpoint(path: str) -> dict:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    return torch.load(path, map_location="cpu", weights_only=False)


def state_dict_of(checkpoint: dict) -> Dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state_dict")
    return state


def get_tensor(state: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = state.get(key)
    if value is None:
        value = state.get(f"module.{key}")
    if value is None or not torch.is_tensor(value):
        raise KeyError(key)
    return value


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def build_diag_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data=args.data,
        out_dir=Path(args.out_dir),
        devices=args.devices,
        device_index=args.device_index,
        master_port=args.master_port,
        batch_size=args.batch_size,
        workers=args.workers,
        teacher_checkpoint=args.teacher_checkpoint,
        wq_mode="lsq",
        aq_mode="lsq",
        qk_reparam=False,
        qk_reparam_type=0,
    )


def load_model_for_proxy(path: str, runtime_args, loader_train, amp_autocast):
    model = diag.build_model(runtime_args)
    diag.ql.setup_alpha(model, loader_train, runtime_args, amp_autocast)
    diag.ql.strict_resume_checkpoint(
        model,
        path,
        optimizer=None,
        loss_scaler=None,
        lr_scheduler=None,
        model_ema=None,
        restore_rng=False,
        log_info=True,
    )
    return model


def collect_proxy_gradient(args: argparse.Namespace) -> torch.Tensor:
    diag_args = build_diag_args(args)
    runtime_args = diag.build_runtime_args(diag_args)
    diag.ql.random_seed(runtime_args.seed, 0)
    import src  # noqa: F401

    probe_model = diag.build_model(runtime_args)
    data_config = diag.ql.resolve_data_config(vars(runtime_args), model=probe_model, verbose=True)
    del probe_model
    torch.cuda.empty_cache()

    use_amp = bool(runtime_args.amp or runtime_args.native_amp)
    amp_dtype = torch.bfloat16 if runtime_args.amp_dtype == "bf16" else torch.float16
    amp_autocast = torch.amp.autocast if use_amp else None
    loader_train = diag.make_train_loader(runtime_args, data_config)
    _, loader_eval = diag.make_eval_loader(runtime_args, data_config)
    autocast_ctx = (
        (lambda: torch.amp.autocast("cuda", dtype=amp_dtype))
        if amp_autocast is not None
        else torch.enable_grad
    )
    model = load_model_for_proxy(args.base, runtime_args, loader_train, autocast_ctx)
    donor_model = None
    if args.proxy_mode != "ce":
        donor_model = load_model_for_proxy(args.donor, runtime_args, loader_train, autocast_ctx)
        donor_model.eval()
        for param in donor_model.parameters():
            param.requires_grad_(False)
    model.train(False)
    for param in model.parameters():
        param.requires_grad_(False)
    target_param = dict(model.named_parameters()).get(f"{args.module}.weight")
    if target_param is None:
        raise KeyError(f"{args.module}.weight")
    target_param.requires_grad_(True)
    target_param.grad = None
    loss_fn = nn.CrossEntropyLoss(reduction="none").cuda()
    seen = 0
    loss_sum = 0.0
    selected = 0
    for batch_idx, (inputs, targets, _indices) in enumerate(loader_eval):
        if batch_idx >= args.proxy_batches:
            break
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else torch.enable_grad():
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            per_sample_loss = loss_fn(outputs, targets)
            if args.proxy_mode == "ce":
                mask = torch.ones_like(targets, dtype=torch.bool)
            else:
                with torch.no_grad():
                    donor_outputs = donor_model(inputs)
                    if isinstance(donor_outputs, (tuple, list)):
                        donor_outputs = donor_outputs[0]
                    base_pred = outputs.detach().float().argmax(dim=1)
                    donor_pred = donor_outputs.detach().float().argmax(dim=1)
                    base_correct = base_pred.eq(targets)
                    donor_correct = donor_pred.eq(targets)
                    if args.proxy_mode == "flip_improve":
                        mask = (~base_correct) & donor_correct
                    elif args.proxy_mode == "flip_regress":
                        mask = base_correct & (~donor_correct)
                    elif args.proxy_mode == "flip_changed":
                        mask = base_correct.ne(donor_correct)
                    else:
                        base_prob = torch.softmax(outputs.detach().float(), dim=1)
                        base_conf = base_prob.gather(1, base_pred[:, None]).squeeze(1)
                        mask = base_conf.lt(args.proxy_conf_high)
            if int(mask.sum().item()) == 0:
                print(f"proxy batch={batch_idx + 1} selected=0")
                continue
            loss = per_sample_loss[mask].mean()
        loss.backward()
        batch = int(targets.numel())
        selected_batch = int(mask.sum().item())
        loss_sum += float(per_sample_loss[mask].detach().mean().item()) * selected_batch
        seen += batch
        selected += selected_batch
        print(
            f"proxy batch={batch_idx + 1} seen={seen} selected={selected} "
            f"avg_selected_loss={loss_sum / max(selected, 1):.6f}"
        )
    if target_param.grad is None:
        raise RuntimeError("proxy gradient was not computed")
    grad = target_param.grad.detach().float().cpu()
    del model
    del donor_model
    torch.cuda.empty_cache()
    return grad


def write_candidate(args: argparse.Namespace, grad: torch.Tensor, mode: str, output: str) -> Dict[str, object]:
    base = load_checkpoint(args.base)
    donor = load_checkpoint(args.donor)
    base_state = state_dict_of(base)
    donor_state = state_dict_of(donor)
    result = copy.deepcopy(base)
    result_state = state_dict_of(result)

    weight_key = f"{args.module}.weight"
    scale_key = f"{args.module}.lsqw_fn.s"
    base_weight = get_tensor(base_state, weight_key).float()
    donor_weight = get_tensor(donor_state, weight_key).float()
    scale = get_tensor(base_state, scale_key).float()
    if scale.ndim == 1 and base_weight.ndim >= 2:
        scale = scale.view(-1, *([1] * (base_weight.ndim - 1)))
    base_bin = torch.clamp(torch.round(base_weight / scale), args.qmin, args.qmax)
    donor_bin = torch.clamp(torch.round(donor_weight / scale), args.qmin, args.qmax)
    changed_mask = base_bin.ne(donor_bin)
    delta = donor_weight - base_weight
    score = grad.to(delta.device) * delta
    aligned = changed_mask & score.lt(0)
    anti = changed_mask & score.gt(0)
    if mode == "aligned":
        mask = aligned
    elif mode == "anti":
        mask = anti
    else:
        mask = changed_mask & score.eq(0)
    if int(mask.sum().item()) == 0:
        raise ValueError(f"empty mask for mode={mode}")

    out_weight = get_tensor(result_state, weight_key).clone()
    out_weight[mask] = get_tensor(donor_state, weight_key).to(out_weight.dtype)[mask]
    actual_weight_key = weight_key if weight_key in result_state else f"module.{weight_key}"
    result_state[actual_weight_key] = out_weight
    copied = [actual_weight_key]
    if parse_bool(args.include_move):
        for suffix in ("move_b4.bias", "move_aft.bias"):
            key = f"{args.module}.{suffix}"
            donor_value = get_tensor(donor_state, key)
            actual_key = key if key in result_state else f"module.{key}"
            result_state[actual_key] = donor_value.detach().clone()
            copied.append(actual_key)

    result["state_dict"] = result_state
    meta = result.setdefault("grad_proxy_assignment_20260709", {})
    meta.update(
        {
            "base": args.base,
            "donor": args.donor,
            "module": args.module,
            "mode": mode,
            "proxy_batches": args.proxy_batches,
            "proxy_mode": args.proxy_mode,
            "include_move": parse_bool(args.include_move),
            "changed_bin_elements": int(changed_mask.sum().item()),
            "aligned_elements": int(aligned.sum().item()),
            "anti_elements": int(anti.sum().item()),
            "assigned_weight_elements": int(mask.sum().item()),
            "total_weight_elements": int(mask.numel()),
            "assigned_fraction": float(mask.float().mean().item()),
            "copied_tensors": copied,
        }
    )
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    torch.save(result, tmp_path)
    tmp_path.replace(out_path)
    summary = {"output": str(out_path), **meta}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--aligned-output", required=True)
    parser.add_argument("--anti-output", required=True)
    parser.add_argument("--out-dir", default="/tmp/resume10_grad_proxy_assignment")
    parser.add_argument("--data", default="/tmp/imagenet1k_full_parquet")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=31496)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--proxy-batches", type=int, default=16)
    parser.add_argument("--proxy-mode", choices=["ce", "flip_improve", "flip_regress", "flip_changed", "low_conf"], default="ce")
    parser.add_argument("--proxy-conf-high", type=float, default=0.6)
    parser.add_argument("--teacher-checkpoint", default="/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth")
    parser.add_argument("--include-move", default="1")
    parser.add_argument("--qmin", type=int, default=-8)
    parser.add_argument("--qmax", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gradient proxy assignment")
    torch.cuda.set_device(args.device_index)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    grad = collect_proxy_gradient(args)
    summaries = [
        write_candidate(args, grad, "aligned", args.aligned_output),
        write_candidate(args, grad, "anti", args.anti_output),
    ]
    print(json.dumps({"summaries": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
