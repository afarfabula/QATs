#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import copy
import contextlib
import importlib
import importlib.util
import inspect
import io
import math
import os
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
import yaml
from PIL import Image
from timm.data import AugMixDataset, FastCollateMixup, Mixup, create_dataset, create_loader, resolve_data_config
from timm.loss import JsdCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model, load_checkpoint, model_parameters, resume_checkpoint, safe_model_name
from timm.utils import AverageMeter, NativeScaler, accuracy, dispatch_clip_grad, get_state_dict, random_seed, reduce_tensor, setup_default_logging


ROOT = Path(__file__).resolve().parent
THIRD_PARTY = ROOT / "third_party"
OFQ_ROOT = THIRD_PARTY / "OFQ"
_OFQ_TRAIN_MODULE = None

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([argparse.Namespace, SimpleNamespace])


class ImageNetParquetDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train", transform=None, subset_ratio: float = 1.0):
        self.root = root
        self.split = split
        self.transform = transform
        self.subset_ratio = float(subset_ratio)
        self.data_dir = os.path.join(root, "data") if os.path.isdir(os.path.join(root, "data")) else root
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"parquet data dir not found: {self.data_dir}")

        self.files = sorted(
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.startswith(f"{split}-") and f.endswith(".parquet")
        )
        if not self.files:
            raise FileNotFoundError(f"no parquet files for split={split} under {self.data_dir}")

        self._file_row_starts = []
        self._row_groups = []
        total_rows = 0
        for file_idx, path in enumerate(self.files):
            pf = pq.ParquetFile(path)
            self._file_row_starts.append(total_rows)
            file_total = 0
            for rg_idx in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                self._row_groups.append((total_rows + file_total, file_idx, rg_idx, rg_rows))
                file_total += rg_rows
            total_rows += file_total
        self._total_rows = total_rows
        self._file_handles = {}
        self._apply_subset_ratio()

    def _apply_subset_ratio(self) -> None:
        if self.subset_ratio <= 0 or self.subset_ratio > 1:
            raise ValueError(f"subset_ratio must be in (0, 1], got {self.subset_ratio}")
        if self.subset_ratio >= 1:
            return

        target_rows = max(1, int(math.ceil(self._total_rows * self.subset_ratio)))
        subset_row_groups = []
        subset_total_rows = 0
        for _, file_idx, rg_idx, rg_rows in self._row_groups:
            if subset_total_rows >= target_rows:
                break
            subset_row_groups.append((subset_total_rows, file_idx, rg_idx, rg_rows))
            subset_total_rows += rg_rows
        self._row_groups = subset_row_groups
        self._total_rows = min(subset_total_rows, target_rows)

    def __len__(self):
        return self._total_rows

    def __getitem__(self, index):
        if index < 0 or index >= self._total_rows:
            raise IndexError(index)
        starts = [rg[0] for rg in self._row_groups]
        rg_pos = max(0, bisect.bisect_right(starts, index) - 1)
        start, file_idx, rg_idx, _ = self._row_groups[rg_pos]
        path = self.files[file_idx]
        pf = self._file_handles.get(path)
        if pf is None:
            pf = pq.ParquetFile(path)
            self._file_handles[path] = pf
        table = pf.read_row_group(rg_idx, columns=["image", "label"])
        rows = table.to_pylist()
        sample = rows[index - start]
        image = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")
        target = int(sample["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class ImageNetParquetIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, root: str, split: str = "train", transform=None, shuffle: bool = True, seed: int = 42, subset_ratio: float = 1.0):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.subset_ratio = float(subset_ratio)
        self.data_dir = os.path.join(root, "data") if os.path.isdir(os.path.join(root, "data")) else root
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"parquet data dir not found: {self.data_dir}")

        self.files = sorted(
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.startswith(f"{split}-") and f.endswith(".parquet")
        )
        if not self.files:
            raise FileNotFoundError(f"no parquet files for split={split} under {self.data_dir}")

        self._row_groups = []
        total_rows = 0
        for path in self.files:
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                self._row_groups.append((path, rg_idx, rg_rows))
                total_rows += rg_rows
        self._total_rows = total_rows
        self._apply_subset_ratio()

    def _apply_subset_ratio(self) -> None:
        if self.subset_ratio <= 0 or self.subset_ratio > 1:
            raise ValueError(f"subset_ratio must be in (0, 1], got {self.subset_ratio}")
        if self.subset_ratio >= 1:
            return

        target_rows = max(1, int(math.ceil(self._total_rows * self.subset_ratio)))
        subset_row_groups = []
        subset_total_rows = 0
        for path, rg_idx, rg_rows in self._row_groups:
            if subset_total_rows >= target_rows:
                break
            subset_row_groups.append((path, rg_idx, rg_rows))
            subset_total_rows += rg_rows
        self._row_groups = subset_row_groups
        self._total_rows = min(subset_total_rows, target_rows)

    def __len__(self):
        return self._target_samples_per_rank()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _distributed_context(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()

        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world_size

    def _target_samples_per_rank(self) -> int:
        _, world_size = self._distributed_context()
        if world_size <= 1:
            return self._total_rows
        # Match DistributedSampler semantics: equal per-rank sample counts, with
        # minimal duplication when the dataset size is not divisible by world size.
        return int(math.ceil(self._total_rows / world_size))

    def _assigned_row_groups(self):
        rank, world_size = self._distributed_context()

        info = torch.utils.data.get_worker_info()
        num_workers = 1 if info is None else info.num_workers
        worker_id = 0 if info is None else info.id

        indices = list(range(len(self._row_groups)))
        if self.shuffle:
            import random as pyrandom

            rng = pyrandom.Random(self.seed + self.epoch)
            rng.shuffle(indices)

        rank_indices = indices[rank::world_size]
        worker_indices = rank_indices[worker_id::num_workers]
        if not worker_indices and rank_indices:
            worker_indices = [rank_indices[worker_id % len(rank_indices)]]
        return worker_indices, worker_id

    def __iter__(self):
        assigned, worker_id = self._assigned_row_groups()
        import random as pyrandom

        if not assigned:
            return

        rng = pyrandom.Random(self.seed + self.epoch * 1009 + worker_id)
        info = torch.utils.data.get_worker_info()
        num_workers = 1 if info is None else info.num_workers
        target_samples = self._target_samples_per_rank()
        worker_target = target_samples // num_workers
        if worker_id < (target_samples % num_workers):
            worker_target += 1

        handles = {}
        yielded = 0
        cycle_idx = 0
        while yielded < worker_target:
            rg_global_idx = assigned[cycle_idx % len(assigned)]
            cycle_idx += 1
            path, rg_idx, _ = self._row_groups[rg_global_idx]
            pf = handles.get(path)
            if pf is None:
                pf = pq.ParquetFile(path)
                handles[path] = pf
            table = pf.read_row_group(rg_idx, columns=["image", "label"])
            cols = table.to_pydict()
            images = cols["image"]
            labels = cols["label"]
            order = list(range(len(images)))
            if self.shuffle:
                rng.shuffle(order)
            for idx in order:
                image = Image.open(io.BytesIO(images[idx]["bytes"])).convert("RGB")
                target = int(labels[idx])
                if self.transform is not None:
                    image = self.transform(image)
                yield image, target
                yielded += 1
                if yielded >= worker_target:
                    break


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return str(Path(path).expanduser().resolve())


def count_devices(devices: Optional[str], fallback: Optional[int]) -> int:
    if fallback is not None:
        return fallback
    if not devices:
        return 1
    return max(1, len([item for item in devices.split(",") if item.strip()]))


def infer_ofq_model_type(model_name: str) -> str:
    lowered = model_name.lower()
    if "swin" in lowered:
        return "swin"
    return "deit"


def default_ofq_config(model_name: str) -> Optional[str]:
    lowered = model_name.lower()
    if "swin" in lowered:
        return str((THIRD_PARTY / "OFQ" / "configs" / "swin_t_imagenet.attn_q.yml").resolve())
    if "deit" in lowered:
        return str((THIRD_PARTY / "OFQ" / "configs" / "deit_default_imagent.attn_q.yml").resolve())
    return None


def qvit_model_name(arch: Optional[str], bits: Optional[int], explicit_model: Optional[str]) -> str:
    if explicit_model:
        return explicit_model
    if not arch:
        raise ValueError("Q-ViT 需要 --arch 或 --model")
    if arch == "swin_tiny":
        return "swin_tiny_patch4_window7_224"
    if bits not in {2, 3, 4}:
        raise ValueError("Q-ViT 的 DeiT 量化模型需要 --bits 为 2/3/4")
    bit_prefix = {2: "two", 3: "three", 4: "four"}[bits] + "bits"
    if arch == "deit_small":
        return f"{bit_prefix}_deit_small_patch16_224"
    if arch == "deit_tiny":
        return f"strict_{bit_prefix}_deit_tiny_patch16_224"
    raise ValueError(f"不支持的 Q-ViT arch: {arch}")


def qvit_teacher_name(arch: Optional[str], explicit_teacher: Optional[str]) -> Optional[str]:
    if explicit_teacher:
        return explicit_teacher
    if arch == "deit_small":
        return "vit_deit_small_distilled_patch16_224"
    if arch == "deit_tiny":
        return "vit_deit_tiny_distilled_patch16_224"
    return None


def qvit_dataset_name(dataset_format: str) -> str:
    mapping = {
        "folder": "IMNET",
        "parquet": "IMNET_PARQUET",
        "parquet-iter": "IMNET_PARQUET_ITER",
    }
    return mapping[dataset_format]


def append_optional_flag(command: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def append_optional_value(command: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def build_qvit(args: argparse.Namespace) -> Tuple[List[str], Path, Dict[str, str]]:
    repo = THIRD_PARTY / "Q-ViT"
    command: List[str]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices

    model_name = qvit_model_name(args.arch, args.bits, args.model)
    teacher_name = qvit_teacher_name(args.arch, args.teacher)
    dataset_name = qvit_dataset_name(args.dataset_format)

    if args.nproc_per_node and args.nproc_per_node > 1:
        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(args.nproc_per_node),
            "--master_port",
            str(args.master_port),
            "main.py",
        ]
    else:
        command = [sys.executable, "main.py"]

    command.extend([
        "--model", model_name,
        "--data-path", normalize_path(args.data) or "",
        "--data-set", dataset_name,
        "--output_dir", normalize_path(args.output) or str((ROOT / "outputs" / "qvit").resolve()),
    ])

    append_optional_value(command, "--epochs", args.epochs)
    append_optional_value(command, "--batch-size", args.batch_size)
    append_optional_value(command, "--batch-size-eval", args.batch_size_eval)
    append_optional_value(command, "--num_workers", args.workers)
    append_optional_value(command, "--lr", args.lr)
    append_optional_value(command, "--weight-decay", args.weight_decay)
    append_optional_value(command, "--warmup-epochs", args.warmup_epochs)
    append_optional_value(command, "--warmup-lr", args.warmup_lr)
    append_optional_value(command, "--resume", normalize_path(args.resume))
    append_optional_value(command, "--distillation-type", args.distillation_type)
    append_optional_value(command, "--teacher-model", teacher_name)
    append_optional_value(command, "--device", args.device)

    append_optional_flag(command, "--pretrained", args.pretrained)
    append_optional_flag(command, "--repeated-aug", args.repeated_aug)
    append_optional_flag(command, "--eval", args.eval)

    command.extend(args.extra_arg)
    return command, repo, env


def build_ofq(args: argparse.Namespace) -> Tuple[List[str], Path, Dict[str, str]]:
    repo = THIRD_PARTY / "OFQ"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    script_name = "cga.py" if args.stage == "cga" else "train.py"
    config_path = normalize_path(args.config) or default_ofq_config(args.model or "")
    if not config_path:
        raise ValueError("OFQ 需要 --config，或者通过 --model 命中内置默认 config")

    model_name = args.model or "swin_t"
    model_type = args.model_type or infer_ofq_model_type(model_name)
    teacher_name = args.teacher or model_name
    teacher_type = args.teacher_type or infer_ofq_model_type(teacher_name)
    world_size = count_devices(args.devices, args.nproc_per_node)
    visible_gpu = args.devices or "0"

    dataset_name = "hf-parquet-imagenet" if args.dataset_format != "folder" else "torch/imagenet"
    experiment = args.experiment or f"{model_name}_w{args.wbits or args.bits or 4}a{args.abits or args.bits or 4}_{args.stage}"

    command = [
        sys.executable,
        script_name,
        "-c", config_path,
        "--model", model_name,
        normalize_path(args.data) or "",
        "--dataset", dataset_name,
        "--output", normalize_path(args.output) or str((ROOT / "outputs" / "ofq").resolve()),
        "--experiment", experiment,
        "--visible_gpu", visible_gpu,
        "--world_size", str(world_size),
        "--tcp_port", str(args.master_port),
        "--model_type", model_type,
        "--teacher", teacher_name,
        "--teacher_type", teacher_type,
    ]

    append_optional_value(command, "--epochs", args.epochs)
    append_optional_value(command, "--batch-size", args.batch_size)
    append_optional_value(command, "--grad-accum-steps", args.grad_accum_steps)
    append_optional_value(command, "--workers", args.workers)
    append_optional_value(command, "--lr", args.lr)
    append_optional_value(command, "--weight-decay", args.weight_decay)
    append_optional_value(command, "--warmup-epochs", args.warmup_epochs)
    append_optional_value(command, "--warmup-lr", args.warmup_lr)
    append_optional_value(command, "--resume", normalize_path(args.resume))
    append_optional_value(command, "--checkpoint-hist", args.checkpoint_hist)
    append_optional_value(command, "--epoch-checkpoint-interval", args.epoch_checkpoint_interval)

    if args.wbits is not None or args.bits is not None:
        command.extend(["--wq-bitw", str(args.wbits if args.wbits is not None else args.bits)])
        command.append("--wq-enable")
    if args.abits is not None or args.bits is not None:
        command.extend(["--aq-bitw", str(args.abits if args.abits is not None else args.bits)])
        command.append("--aq-enable")

    append_optional_value(command, "--wq-mode", args.wq_mode)
    append_optional_value(command, "--aq-mode", args.aq_mode)
    append_optional_value(command, "--boundaryRange", args.boundary_range if args.stage == "cga" else None)
    append_optional_value(command, "--freeze_for_n_epochs", args.freeze_for_n_epochs if args.stage == "cga" else None)

    append_optional_flag(command, "--wq-per-channel", args.wq_per_channel)
    append_optional_flag(command, "--aq-per-channel", args.aq_per_channel)
    append_optional_flag(command, "--wq_clip_learnable", args.wq_clip_learnable)
    append_optional_flag(command, "--aq_clip_learnable", args.aq_clip_learnable)
    append_optional_flag(command, "--pretrained", args.pretrained)
    append_optional_flag(command, "--pretrained_initialized", args.pretrained_initialized)
    append_optional_flag(command, "--use-kd", args.use_kd)
    append_optional_value(command, "--kd_hard_and_soft", args.kd_hard_and_soft)
    append_optional_flag(command, "--teacher_pretrained", args.teacher_pretrained)
    append_optional_value(command, "--teacher-checkpoint", normalize_path(args.teacher_checkpoint))
    append_optional_flag(command, "--quantized", args.quantized)
    append_optional_flag(command, "--qk_reparam", args.qk_reparam)
    append_optional_value(command, "--qk_reparam_type", args.qk_reparam_type)
    append_optional_value(command, "--train-scheme", args.train_scheme)
    append_optional_value(command, "--ref-update", args.ref_update)
    append_optional_value(command, "--ref-momentum", args.ref_momentum)
    append_optional_value(command, "--ref-attn-kl-weight", args.ref_attn_kl_weight)
    append_optional_value(command, "--ref-head-mode", args.ref_head_mode)
    append_optional_value(command, "--ref-warmup-epochs", args.ref_warmup_epochs)

    command.extend(args.extra_arg)
    return command, repo, env


def build_aoq(args: argparse.Namespace) -> Tuple[List[str], Path, Dict[str, str]]:
    task = args.task or "imagenet"
    repo = THIRD_PARTY / "AOQ" / "AO_QAT" / ("resnet_imagenet" if task == "imagenet" else "resnet_cifar10")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.devices:
        primary_gpu = args.devices.split(",")[0].strip()
        env["CUDA_VISIBLE_DEVICES"] = primary_gpu
    env["QATS_DEVICE"] = args.device or "cuda:0"

    command = [
        sys.executable,
        "train.py",
        "--data", normalize_path(args.data) or "",
        "--save", normalize_path(args.output) or str((ROOT / "outputs" / "aoq").resolve()),
        "--student", args.model or ("resnet18" if task == "imagenet" else "resnet20"),
        "--teacher", args.teacher or ("resnet101" if task == "imagenet" else "resnet20"),
        "--n_bit", str(args.bits if args.bits is not None else 2),
        "--quantize_downsample", "True" if args.quantize_downsample else "False",
    ]

    append_optional_value(command, "--epochs", args.epochs)
    append_optional_value(command, "--batch_size", args.batch_size)
    append_optional_value(command, "--workers", args.workers)
    append_optional_value(command, "--learning_rate", args.lr)
    append_optional_value(command, "--weight_decay", args.weight_decay)
    append_optional_value(command, "--amp_dtype", args.amp_dtype)
    append_optional_value(command, "--compile_mode", args.compile_mode)
    append_optional_value(command, "--compile_backend", args.compile_backend)
    append_optional_value(command, "--prefetch_factor", args.prefetch_factor)
    append_optional_value(command, "--val_interval", args.val_interval)
    append_optional_value(command, "--plot_interval", args.plot_interval)
    append_optional_value(command, "--train_steps_per_epoch", args.train_steps_per_epoch)
    append_optional_value(command, "--val_steps", args.val_steps)
    append_optional_value(command, "--synthetic_train_size", args.synthetic_train_size)
    append_optional_value(command, "--synthetic_val_size", args.synthetic_val_size)
    append_optional_value(command, "--dataset_format", args.aoq_dataset_format)
    append_optional_flag(command, "--amp", args.amp)
    append_optional_flag(command, "--channels_last", args.channels_last)
    append_optional_flag(command, "--compile", args.compile)
    append_optional_flag(command, "--persistent_workers", args.persistent_workers)
    append_optional_flag(command, "--synthetic_data", args.synthetic_data)
    append_optional_flag(command, "--skip_teacher_val", args.skip_teacher_val)
    append_optional_flag(command, "--print_model", args.print_model)
    append_optional_flag(command, "--print_params", args.print_params)
    command.extend(args.extra_arg)
    return command, repo, env


def build_command(args: argparse.Namespace) -> Tuple[List[str], Path, Dict[str, str]]:
    if not args.data:
        raise ValueError("统一入口要求显式提供 --data")
    if args.method == "qvit":
        return build_qvit(args)
    if args.method == "ofq":
        return build_ofq(args)
    if args.method == "aoq":
        return build_aoq(args)
    raise ValueError(f"未知 method: {args.method}")


def load_ofq_training_module():
    global _OFQ_TRAIN_MODULE
    if _OFQ_TRAIN_MODULE is not None:
        return _OFQ_TRAIN_MODULE

    from src.quantization import (
        KDLossSoftandHard,
        KDLossSoftandHard_qk,
        KDLossSoftandHard_qkv,
        KLLossSoft,
        KLTokenMSELoss,
    )
    from src.quantization.modules.utils import replace_module_by_qmodule_deit, replace_module_by_qmodule_swin

    _OFQ_TRAIN_MODULE = SimpleNamespace(
        KDLossSoftandHard=KDLossSoftandHard,
        KDLossSoftandHard_qk=KDLossSoftandHard_qk,
        KDLossSoftandHard_qkv=KDLossSoftandHard_qkv,
        KLLossSoft=KLLossSoft,
        KLTokenMSELoss=KLTokenMSELoss,
        replace_module_by_qmodule_deit=replace_module_by_qmodule_deit,
        replace_module_by_qmodule_swin=replace_module_by_qmodule_swin,
    )
    return _OFQ_TRAIN_MODULE


def create_dataset_compat(dataset_name, root, split, is_training, batch_size, repeats=0, transform=None, subset_ratio: float = 1.0):
    if dataset_name == "hf-parquet-imagenet":
        if is_training:
            return ImageNetParquetIterableDataset(root=root, split=split, transform=transform, shuffle=True, subset_ratio=subset_ratio)
        return ImageNetParquetDataset(root=root, split=split, transform=transform, subset_ratio=subset_ratio)
    return create_dataset(dataset_name, root=root, split=split, is_training=is_training, batch_size=batch_size, repeats=repeats)


def create_loader_compat(dataset, **kwargs):
    sig = inspect.signature(create_loader)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return create_loader(dataset, **filtered)


def build_ofq_runtime_overrides(extra_args: Sequence[str]) -> Dict[str, object]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skip_validate", action="store_true")
    parser.add_argument("--eval-only", dest="eval_only", action="store_true")
    parser.add_argument("--max_train_updates", type=int)
    parser.add_argument("--log-interval", dest="log_interval", type=int)
    parser.add_argument("--save_step_checkpoints", action="store_true")
    parser.add_argument("--save_initial_step_checkpoint", action="store_true")
    parser.add_argument("--step_checkpoint_interval", type=int)
    parser.add_argument("--step_checkpoint_warmup_updates", type=int)
    parser.add_argument("--max_step_checkpoints_to_save", type=int)
    parser.add_argument("--collect_attention", action="store_true")
    parser.add_argument("--initial-checkpoint", dest="initial_checkpoint", type=str)
    parser.add_argument("--no-prefetcher", dest="no_prefetcher", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin-mem", dest="pin_mem", action="store_true")
    parser.add_argument("--channels-last", dest="channels_last", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mixup", type=float)
    parser.add_argument("--cutmix", type=float)
    parser.add_argument("--mixup-prob", dest="mixup_prob", type=float)
    parser.add_argument("--mixup-switch-prob", dest="mixup_switch_prob", type=float)
    parser.add_argument("--mixup-mode", dest="mixup_mode", type=str)
    parser.add_argument("--smoothing", type=float)
    parser.add_argument("--native-amp", dest="native_amp", action="store_true")
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--teacher-checkpoint", dest="teacher_checkpoint", type=str)
    parser.add_argument("--quant-teacher", dest="quant_teacher", action="store_true")
    parser.add_argument("--use-token-kd", dest="use_token_kd", action="store_true")
    parser.add_argument("--kd-alpha", dest="kd_alpha", type=float)
    parser.add_argument("--kd_hard_and_soft", type=int)
    parser.add_argument("--kd-type", dest="kd_type", type=str)
    parser.add_argument("--qk_reparam_type", type=int)
    parser.add_argument("--warmup-lr", dest="warmup_lr", type=float)
    parser.add_argument("--recovery-interval", dest="recovery_interval", type=int)
    parser.add_argument("--checkpoint-hist", dest="checkpoint_hist", type=int)
    parser.add_argument("--epoch-checkpoint-interval", dest="epoch_checkpoint_interval", type=int)
    parser.add_argument("--subset-ratio", dest="subset_ratio", type=float)
    parser.add_argument("--initial_checkpoint", dest="initial_checkpoint_alias", type=str)
    namespace, _ = parser.parse_known_args(list(extra_args))
    overrides = {k: v for k, v in vars(namespace).items() if v is not None and v is not False}
    if "initial_checkpoint_alias" in overrides:
        overrides["initial_checkpoint"] = overrides.pop("initial_checkpoint_alias")
    return overrides


def build_ofq_runtime_config(args: argparse.Namespace) -> SimpleNamespace:
    defaults = {
        "dataset": "hf-parquet-imagenet" if args.dataset_format != "folder" else "torch/imagenet",
        "train_split": "train",
        "val_split": "validation",
        "num_classes": 1000,
        "input_size": None,
        "crop_pct": None,
        "mean": None,
        "std": None,
        "interpolation": "bicubic",
        "train_interpolation": "random",
        "scale": [0.08, 1.0],
        "ratio": [0.75, 1.3333333333333333],
        "hflip": 0.5,
        "vflip": 0.0,
        "color_jitter": 0.4,
        "aa": None,
        "aug_splits": 0,
        "jsd": False,
        "reprob": 0.0,
        "remode": "const",
        "recount": 1,
        "resplit": False,
        "mixup": 0.0,
        "cutmix": 0.0,
        "cutmix_minmax": None,
        "mixup_prob": 1.0,
        "mixup_switch_prob": 0.5,
        "mixup_mode": "batch",
        "mixup_off_epoch": 0,
        "smoothing": 0.1,
        "drop": 0.0,
        "drop_path": 0.0,
        "drop_block": None,
        "num_aug_repeats": 0,
        "seed": 42,
        "log_interval": 50,
        "recovery_interval": 0,
        "checkpoint_hist": 10,
        "epoch_checkpoint_interval": 10,
        "subset_ratio": 1.0,
        "save_images": False,
        "amp": False,
        "apex_amp": False,
        "native_amp": False,
        "channels_last": False,
        "pin_mem": False,
        "no_prefetcher": False,
        "output": str((ROOT / "outputs" / "ofq").resolve()),
        "experiment": None,
        "eval_metric": "top1",
        "tta": 0,
        "use_multi_epochs_loader": False,
        "log_wandb": False,
        "wq_enable": False,
        "wq_mode": "statsq",
        "wq_bitw": 4,
        "wq_per_channel": False,
        "wq_asym": False,
        "wq_clip_learnable": False,
        "aq_enable": False,
        "aq_mode": "lsq",
        "aq_bitw": 4,
        "aq_per_channel": False,
        "aq_clip_learnable": False,
        "qmodules": [],
        "replace_ln_by_bn": False,
        "use_kd": False,
        "use_token_kd": False,
        "kd_alpha": 1.0,
        "teacher": args.model or "swin_t",
        "teacher_checkpoint": "",
        "teacher_pretrained": False,
        "quant_teacher": False,
        "kd_type": "last",
        "warmup_lr": 1e-6,
        "gpu_id": 0,
        "model_type": "swin",
        "quantized": False,
        "world_size": count_devices(args.devices, args.nproc_per_node),
        "visible_gpu": args.devices or "0",
        "tcp_port": str(args.master_port),
        "collect_attention": False,
        "max_train_updates": 0,
        "save_step_checkpoints": False,
        "save_initial_step_checkpoint": False,
        "step_checkpoint_interval": 1,
        "step_checkpoint_warmup_updates": 0,
        "max_step_checkpoints_to_save": 0,
        "skip_validate": False,
        "eval_only": False,
        "apply_q_attn_dropout": 0,
        "act_layer": "gelu",
        "kd_hard_and_soft": 1,
        "teacher_type": "swin",
        "pretrained_initialized": False,
        "qk_reparam": False,
        "qk_reparam_type": 0,
        "train_scheme": "baseline",
        "ref_update": "ema",
        "ref_momentum": 0.999,
        "ref_attn_kl_weight": 0.0,
        "ref_head_mode": "all",
        "ref_warmup_epochs": 0,
        "initial_checkpoint": "",
        "resume": "",
        "no_resume_opt": False,
        "opt": "adamw",
        "lr": 2e-4,
        "weight_decay": 0.0,
        "epochs": 300,
        "warmup_epochs": 0,
        "min_lr": 1e-5,
        "workers": 4,
        "batch_size": 32,
        "validation_batch_size_multiplier": 1,
        "grad_accum_steps": 1,
        "momentum": 0.9,
        "opt_betas": (0.9, 0.999),
        "clip_grad": None,
        "clip_mode": "norm",
    }

    config_path = normalize_path(args.config) or default_ofq_config(args.model or "")
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            defaults.update(loaded)

    defaults.update(
        {
            "data_dir": normalize_path(args.data) or "",
            "output": normalize_path(args.output) or defaults["output"],
            "model": args.model or defaults.get("model", "swin_t"),
            "teacher": args.teacher or defaults.get("teacher") or args.model or "swin_t",
            "experiment": args.experiment or defaults.get("experiment") or f"{args.model or 'swin_t'}_w{args.wbits or args.bits or 4}a{args.abits or args.bits or 4}_{args.stage}",
            "dataset": "hf-parquet-imagenet" if args.dataset_format != "folder" else "torch/imagenet",
            "visible_gpu": args.devices or defaults["visible_gpu"],
            "world_size": count_devices(args.devices, args.nproc_per_node),
            "tcp_port": str(args.master_port),
            "model_type": args.model_type or defaults.get("model_type") or infer_ofq_model_type(args.model or "swin_t"),
            "teacher_type": args.teacher_type or defaults.get("teacher_type") or infer_ofq_model_type(args.teacher or args.model or "swin_t"),
            "resume": normalize_path(args.resume) or defaults["resume"],
        }
    )

    if args.epochs is not None:
        defaults["epochs"] = args.epochs
    if args.batch_size is not None:
        defaults["batch_size"] = args.batch_size
    if args.workers is not None:
        defaults["workers"] = args.workers
    if args.lr is not None:
        defaults["lr"] = args.lr
    if args.weight_decay is not None:
        defaults["weight_decay"] = args.weight_decay
    if args.warmup_epochs is not None:
        defaults["warmup_epochs"] = args.warmup_epochs
    if args.warmup_lr is not None:
        defaults["warmup_lr"] = args.warmup_lr
    if args.grad_accum_steps is not None:
        defaults["grad_accum_steps"] = args.grad_accum_steps
    if args.checkpoint_hist is not None:
        defaults["checkpoint_hist"] = args.checkpoint_hist
    if args.epoch_checkpoint_interval is not None:
        defaults["epoch_checkpoint_interval"] = args.epoch_checkpoint_interval
    if args.kd_hard_and_soft is not None:
        defaults["kd_hard_and_soft"] = args.kd_hard_and_soft
    elif args.use_kd and defaults.get("kd_hard_and_soft", 0) == 0:
        defaults["kd_hard_and_soft"] = 1
    if args.qk_reparam_type is not None:
        defaults["qk_reparam_type"] = args.qk_reparam_type

    if args.wbits is not None or args.bits is not None:
        defaults["wq_bitw"] = args.wbits if args.wbits is not None else args.bits
        defaults["wq_enable"] = True
    if args.abits is not None or args.bits is not None:
        defaults["aq_bitw"] = args.abits if args.abits is not None else args.bits
        defaults["aq_enable"] = True

    defaults["wq_mode"] = args.wq_mode or defaults["wq_mode"]
    defaults["aq_mode"] = args.aq_mode or defaults["aq_mode"]
    defaults["wq_per_channel"] = bool(args.wq_per_channel or defaults.get("wq_per_channel", False))
    defaults["aq_per_channel"] = bool(args.aq_per_channel or defaults.get("aq_per_channel", False))
    defaults["wq_clip_learnable"] = bool(args.wq_clip_learnable or defaults.get("wq_clip_learnable", False))
    defaults["aq_clip_learnable"] = bool(args.aq_clip_learnable or defaults.get("aq_clip_learnable", False))
    defaults["pretrained"] = bool(args.pretrained or defaults.get("pretrained", False))
    defaults["pretrained_initialized"] = bool(args.pretrained_initialized or defaults.get("pretrained_initialized", False))
    defaults["use_kd"] = bool(args.use_kd or defaults.get("use_kd", False))
    if args.use_kd and defaults.get("kd_hard_and_soft", 0) == 0:
        defaults["kd_hard_and_soft"] = 1
    defaults["teacher_pretrained"] = bool(args.teacher_pretrained or defaults.get("teacher_pretrained", False))
    if args.teacher_checkpoint is not None:
        defaults["teacher_checkpoint"] = args.teacher_checkpoint
    defaults["quantized"] = bool(args.quantized or defaults.get("quantized", False))
    defaults["qk_reparam"] = bool(args.qk_reparam or defaults.get("qk_reparam", False))
    if args.train_scheme is not None:
        defaults["train_scheme"] = args.train_scheme
    if args.ref_update is not None:
        defaults["ref_update"] = args.ref_update
    if args.ref_momentum is not None:
        defaults["ref_momentum"] = args.ref_momentum
    if args.ref_attn_kl_weight is not None:
        defaults["ref_attn_kl_weight"] = args.ref_attn_kl_weight
    if args.ref_head_mode is not None:
        defaults["ref_head_mode"] = args.ref_head_mode
    if args.ref_warmup_epochs is not None:
        defaults["ref_warmup_epochs"] = args.ref_warmup_epochs

    defaults.update(build_ofq_runtime_overrides(args.extra_arg))
    defaults["world_size"] = int(defaults["world_size"])
    defaults["lr"] = float(defaults["lr"])
    defaults["warmup_lr"] = float(defaults["warmup_lr"])
    defaults["min_lr"] = float(defaults["min_lr"])
    defaults["weight_decay"] = float(defaults["weight_decay"])
    defaults["epochs"] = int(defaults["epochs"])
    defaults["batch_size"] = int(defaults["batch_size"])
    defaults["workers"] = int(defaults["workers"])
    defaults["grad_accum_steps"] = int(defaults["grad_accum_steps"])
    defaults["warmup_epochs"] = int(defaults["warmup_epochs"])
    defaults["num_classes"] = int(defaults["num_classes"])
    defaults["epoch_checkpoint_interval"] = int(defaults["epoch_checkpoint_interval"])
    defaults["subset_ratio"] = float(defaults["subset_ratio"])
    defaults["ref_warmup_epochs"] = int(defaults["ref_warmup_epochs"])
    defaults["ref_momentum"] = float(defaults["ref_momentum"])
    defaults["ref_attn_kl_weight"] = float(defaults["ref_attn_kl_weight"])
    defaults["no_prefetcher"] = bool(defaults.get("no_prefetcher", False))
    defaults["prefetcher"] = not defaults["no_prefetcher"]
    defaults["teacher"] = defaults["teacher"] or defaults["model"]
    defaults["experiment"] = defaults["experiment"] or safe_model_name(defaults["model"])
    defaults["opt_betas"] = tuple(defaults.get("opt_betas") or (0.9, 0.999))
    defaults["drop_path"] = 0.0 if defaults.get("drop_path") is None else defaults.get("drop_path")

    defaults["single_process_grad_accum_steps"] = defaults["grad_accum_steps"]
    defaults["single_process_effective_batch_size"] = defaults["batch_size"] * defaults["single_process_grad_accum_steps"]
    if defaults["world_size"] > 1:
        defaults["grad_accum_steps"] = max(1, int(math.ceil(defaults["single_process_grad_accum_steps"] / defaults["world_size"])))
    defaults["effective_batch_size"] = defaults["batch_size"] * defaults["world_size"] * defaults["grad_accum_steps"]

    return SimpleNamespace(**defaults)


def build_ofq_qconfigs(runtime_args: SimpleNamespace) -> Dict[str, Dict[str, object]]:
    act_layer_mappings = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "prelu": nn.PReLU,
        "rprelu": "rprelu",
        "None": "None",
    }
    qconfigs: Dict[str, Dict[str, object]] = {}
    for module_name in runtime_args.qmodules:
        wcfg = {
            "mode": runtime_args.wq_mode if runtime_args.wq_enable else "Identity",
            "bit": runtime_args.wq_bitw if runtime_args.wq_bitw < 32 and runtime_args.aq_enable else "identity",
            "all_positive": False,
            "symmetric": not runtime_args.wq_asym,
            "per_channel": runtime_args.wq_per_channel,
            "normalize_first": False,
            "learnable": runtime_args.wq_clip_learnable,
        }
        acfg = {
            "enable": runtime_args.aq_enable if runtime_args.aq_enable else "Identity",
            "mode": runtime_args.aq_mode if runtime_args.aq_bitw < 32 and runtime_args.aq_enable else "identity",
            "bit": runtime_args.aq_bitw,
            "per_channel": runtime_args.aq_per_channel,
            "normalize_first": False,
            "learnable": runtime_args.aq_clip_learnable,
        }
        qconfigs[module_name] = {
            "weight": wcfg,
            "act": acfg,
            "q_attn_dropout": runtime_args.apply_q_attn_dropout,
            "act_layer": act_layer_mappings[runtime_args.act_layer],
        }
    return qconfigs


def get_ofq_qat_model(model: nn.Module, runtime_args: SimpleNamespace) -> nn.Module:
    helpers = load_ofq_training_module()
    qconfigs = build_ofq_qconfigs(runtime_args)
    if runtime_args.model_type == "deit":
        return helpers.replace_module_by_qmodule_deit(
            model,
            qconfigs,
            pretrained_initialized=runtime_args.pretrained_initialized,
            qk_reparam=runtime_args.qk_reparam,
            qk_reparam_type=runtime_args.qk_reparam_type,
        )
    return helpers.replace_module_by_qmodule_swin(
        model,
        qconfigs,
        pretrained_initialized=runtime_args.pretrained_initialized,
        qk_reparam=runtime_args.qk_reparam,
        qk_reparam_type=runtime_args.qk_reparam_type,
    )


def enable_attention_collection(model: nn.Module) -> int:
    enabled = 0
    for module in model.modules():
        module_name = type(module).__name__
        if module_name == "ShiftedWindowAttention" or module_name.startswith("QAttention_swin"):
            setattr(module, "collect_attention", True)
            enabled += 1
    return enabled


def create_ofq_teacher_model(runtime_args: SimpleNamespace) -> nn.Module:
    qqkkvv = runtime_args.kd_hard_and_soft in {2, 3}
    if runtime_args.teacher_type == "deit":
        teacher = create_model(runtime_args.teacher, num_classes=runtime_args.num_classes, drop_rate=runtime_args.drop, pretrained=runtime_args.teacher_pretrained, qqkkvv=qqkkvv)
    else:
        teacher = create_model(runtime_args.teacher, num_classes=runtime_args.num_classes, drop_path=runtime_args.drop_path, pretrained=runtime_args.teacher_pretrained, qqkkvv=qqkkvv)
    if runtime_args.quant_teacher:
        teacher = get_ofq_qat_model(teacher, runtime_args)
    if runtime_args.teacher_checkpoint:
        load_checkpoint(teacher, runtime_args.teacher_checkpoint, strict=True)
    return teacher


def save_step_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, runtime_args: SimpleNamespace, output_dir: Path, step_tag: str, epoch: Optional[int] = None, batch_idx: Optional[int] = None, loss_scaler=None, metric=None) -> str:
    step_dir = output_dir / "step_checkpoints"
    step_dir.mkdir(parents=True, exist_ok=True)
    save_state = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "arch": runtime_args.model,
        "state_dict": get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "version": 2,
        "args": runtime_args,
        "step_tag": step_tag,
    }
    if loss_scaler is not None:
        save_state[loss_scaler.state_dict_key] = loss_scaler.state_dict()
    if metric is not None:
        save_state["metric"] = metric
    save_path = step_dir / f"{step_tag}.pth.tar"
    torch.save(save_state, save_path)
    return str(save_path)


def save_epoch_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, runtime_args: SimpleNamespace, output_dir: Path, epoch: int, loss_scaler=None) -> None:
    state = {
        "epoch": epoch + 1,
        "arch": runtime_args.model,
        "state_dict": get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "version": 2,
        "args": runtime_args,
    }
    if loss_scaler is not None:
        state[loss_scaler.state_dict_key] = loss_scaler.state_dict()
    checkpoint_path = output_dir / f"checkpoint-{epoch + 1}.pth.tar"
    last_path = output_dir / "last.pth.tar"
    torch.save(state, checkpoint_path)
    try:
        if last_path.exists() or last_path.is_symlink():
            last_path.unlink()
        os.link(checkpoint_path, last_path)
    except OSError:
        try:
            if last_path.exists() or last_path.is_symlink():
                last_path.unlink()
            os.symlink(checkpoint_path.name, last_path)
        except OSError:
            torch.save(state, last_path)


def maybe_unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def set_attention_mode(model: nn.Module, collect_attention: bool = False, qqkkvv: bool = False) -> None:
    for module in model.modules():
        module_name = type(module).__name__
        is_swin_attention = module_name == "ShiftedWindowAttention" or module_name.startswith("QAttention_swin")
        if hasattr(module, "collect_attention") or is_swin_attention:
            setattr(module, "collect_attention", collect_attention)
        if hasattr(module, "qqkkvv"):
            setattr(module, "qqkkvv", qqkkvv)


def clone_ref_model(student_model: nn.Module) -> nn.Module:
    student_core = maybe_unwrap_ddp(student_model)
    ref_model = copy.deepcopy(student_core)
    ref_model.cuda()
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    set_attention_mode(ref_model, collect_attention=True, qqkkvv=False)
    return ref_model


@torch.no_grad()
def update_ref_model(student_model: nn.Module, ref_model: nn.Module, momentum: float) -> None:
    student_core = maybe_unwrap_ddp(student_model)
    student_params = dict(student_core.named_parameters())
    for name, ref_param in ref_model.named_parameters():
        src = student_params[name]
        ref_param.data.mul_(momentum).add_(src.data, alpha=1.0 - momentum)

    student_buffers = dict(student_core.named_buffers())
    for name, ref_buffer in ref_model.named_buffers():
        src = student_buffers[name]
        if torch.is_floating_point(ref_buffer):
            ref_buffer.data.mul_(momentum).add_(src.data, alpha=1.0 - momentum)
        else:
            ref_buffer.data.copy_(src.data)


def extract_attn_prob_list(attn_info):
    if attn_info is None:
        return []
    extracted = []
    for layer_info in attn_info:
        if layer_info is None:
            continue
        if isinstance(layer_info, (tuple, list)):
            attn_tensor = layer_info[0]
        else:
            attn_tensor = layer_info
        if torch.is_tensor(attn_tensor):
            extracted.append(attn_tensor)
    return extracted


OSCILLATING_SWIN_HEADS = (
    (5, 2),
    (10, 14),
    (5, 1),
    (4, 1),
    (9, 10),
)


def attention_kl_consistency_loss(student_attn_info, ref_attn_info, head_mode: str = "all") -> torch.Tensor:
    student_list = extract_attn_prob_list(student_attn_info)
    ref_list = extract_attn_prob_list(ref_attn_info)
    if not student_list or not ref_list:
        if student_list:
            return student_list[0].new_zeros(())
        if ref_list:
            return ref_list[0].new_zeros(())
        return torch.zeros((), device="cuda")

    if head_mode not in {"all", "oscillating_top5"}:
        raise NotImplementedError(f"Unsupported ref head mode: {head_mode}")

    total = student_list[0].new_zeros(())
    count = 0
    selected_heads = None
    if head_mode == "oscillating_top5":
        selected_heads = OSCILLATING_SWIN_HEADS

    if selected_heads is None:
        selected_items = [
            (layer_idx, None)
            for layer_idx in range(min(len(student_list), len(ref_list)))
        ]
    else:
        selected_items = selected_heads

    for layer_idx, head_idx in selected_items:
        if layer_idx >= len(student_list) or layer_idx >= len(ref_list):
            continue
        student_attn = student_list[layer_idx]
        ref_attn = ref_list[layer_idx]
        if head_idx is not None:
            if student_attn.ndim < 4 or head_idx >= student_attn.shape[1] or head_idx >= ref_attn.shape[1]:
                continue
            student_attn = student_attn[:, head_idx : head_idx + 1]
            ref_attn = ref_attn[:, head_idx : head_idx + 1]
        student_log_prob = torch.log(student_attn.clamp_min(1e-8))
        ref_prob = ref_attn.clamp_min(1e-8)
        total = total + F.kl_div(student_log_prob, ref_prob, reduction="batchmean")
        count += 1
    return total / max(count, 1)


def setup_alpha(model: nn.Module, loader, runtime_args: SimpleNamespace, amp_autocast):
    model.eval()
    if runtime_args.local_rank == 0:
        print("setup alpha")
    with torch.no_grad():
        for input, target in loader:
            if not runtime_args.prefetcher:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            if runtime_args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                model(input)
            break


def create_ofq_loss(runtime_args: SimpleNamespace):
    helpers = load_ofq_training_module()
    if runtime_args.jsd:
        return JsdCrossEntropy(num_splits=runtime_args.aug_splits, smoothing=runtime_args.smoothing).cuda()
    if runtime_args.use_token_kd:
        return helpers.KLTokenMSELoss(alpha=runtime_args.kd_alpha, kd_type=runtime_args.kd_type).cuda()
    if runtime_args.use_kd:
        if runtime_args.kd_hard_and_soft == 0:
            return helpers.KLLossSoft().cuda()
        if runtime_args.kd_hard_and_soft == 1:
            return helpers.KDLossSoftandHard().cuda()
        if runtime_args.kd_hard_and_soft == 2:
            return helpers.KDLossSoftandHard_qk().cuda()
        if runtime_args.kd_hard_and_soft == 3:
            return helpers.KDLossSoftandHard_qkv().cuda()
    if runtime_args.mixup > 0 or runtime_args.cutmix > 0.0 or runtime_args.cutmix_minmax is not None:
        return SoftTargetCrossEntropy().cuda()
    if runtime_args.smoothing:
        return LabelSmoothingCrossEntropy(smoothing=runtime_args.smoothing).cuda()
    return nn.CrossEntropyLoss().cuda()


def create_ofq_optimizer(runtime_args: SimpleNamespace, model: nn.Module) -> torch.optim.Optimizer:
    if runtime_args.opt.lower() != "adamw":
        raise NotImplementedError(f"当前 unified OFQ path 仅支持 AdamW，收到: {runtime_args.opt}")
    return torch.optim.AdamW(model.parameters(), lr=runtime_args.lr, weight_decay=runtime_args.weight_decay, betas=runtime_args.opt_betas)


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, min_lr: float, warmup_updates: int, total_updates: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_updates = max(0, warmup_updates)
        self.total_updates = max(1, total_updates)

    def step_update(self, num_updates: int) -> None:
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            lr = self.base_lr * float(num_updates) / float(self.warmup_updates)
        else:
            progress = 0.0
            if self.total_updates > self.warmup_updates:
                progress = min(1.0, max(0.0, (num_updates - self.warmup_updates) / float(self.total_updates - self.warmup_updates)))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def validate_ofq(model: nn.Module, loader, loss_fn, runtime_args: SimpleNamespace, amp_autocast):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    model.eval()
    if runtime_args.local_rank == 0:
        print("model eval")
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not runtime_args.prefetcher:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            if runtime_args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            reduce_factor = runtime_args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if runtime_args.distributed:
                reduced_loss = reduce_tensor(loss.data, runtime_args.world_size)
                acc1 = reduce_tensor(acc1, runtime_args.world_size)
                acc5 = reduce_tensor(acc5, runtime_args.world_size)
            else:
                reduced_loss = loss.data
            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
            if runtime_args.local_rank == 0 and (last_batch or batch_idx % runtime_args.log_interval == 0):
                print(
                    f"Test: [{batch_idx:>4d}/{last_idx}]  Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                    f"Loss: {losses_m.val:>7.4f} ({losses_m.avg:>6.4f})  "
                    f"Acc@1: {top1_m.val:>7.4f} ({top1_m.avg:>7.4f})  "
                    f"Acc@5: {top5_m.val:>7.4f} ({top5_m.avg:>7.4f})"
                )
    return {"loss": losses_m.avg, "top1": top1_m.avg, "top5": top5_m.avg}


def train_one_epoch_ofq(epoch: int, model: nn.Module, loader, optimizer: torch.optim.Optimizer, loss_fn, runtime_args: SimpleNamespace, lr_scheduler: WarmupCosineScheduler, output_dir: Path, amp_autocast, loss_scaler, teacher: Optional[nn.Module], mixup_fn, ref_model: Optional[nn.Module] = None):
    if runtime_args.mixup_off_epoch and epoch >= runtime_args.mixup_off_epoch:
        if runtime_args.prefetcher and hasattr(loader, "mixup_enabled"):
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    base_losses_m = AverageMeter()
    ref_attn_kl_losses_m = AverageMeter()
    accum_steps = max(1, int(getattr(runtime_args, "grad_accum_steps", 1)))
    model.train()
    optimizer.zero_grad()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    local_update_count = 0
    saved_step_count = 0
    stopped_early = False
    warmup_updates = max(0, int(getattr(runtime_args, "step_checkpoint_warmup_updates", 0)))
    max_step_checkpoints_to_save = max(0, int(getattr(runtime_args, "max_step_checkpoints_to_save", 0)))

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        update_step = ((batch_idx + 1) % accum_steps == 0) or last_batch
        data_time_m.update(time.time() - end)
        if not runtime_args.prefetcher:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if runtime_args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        sync_context = contextlib.nullcontext()
        if runtime_args.distributed and not update_step and hasattr(model, "no_sync"):
            sync_context = model.no_sync()

        with sync_context:
            with amp_autocast():
                if runtime_args.model_type in {"deit", "swin"}:
                    student_logit, student_attn_info = model(input)
                else:
                    student_logit = model(input)
                    student_attn_info = None

                if runtime_args.use_kd:
                    if runtime_args.teacher_type in {"deit", "swin"}:
                        teacher_output = teacher(input)
                    else:
                        teacher_output = teacher(input)
                    if isinstance(teacher_output, tuple):
                        teacher_logit = teacher_output[0]
                        teacher_attn_info = teacher_output[1] if len(teacher_output) > 1 else None
                    else:
                        teacher_logit = teacher_output
                        teacher_attn_info = None

                    if runtime_args.kd_hard_and_soft == 0:
                        loss = loss_fn(student_logit, teacher_logit)
                    elif runtime_args.kd_hard_and_soft == 1:
                        loss = loss_fn(student_logit, target, teacher_logit)
                    elif runtime_args.kd_hard_and_soft == 2:
                        loss = loss_fn(student_logit, student_attn_info, target, teacher_logit, teacher_attn_info)
                    elif runtime_args.kd_hard_and_soft == 3:
                        loss = loss_fn(student_logit, student_attn_info, target, teacher_logit, teacher_attn_info)
                    else:
                        raise NotImplementedError(f"Unsupported kd_hard_and_soft={runtime_args.kd_hard_and_soft}")
                else:
                    student_logit = student_logit[0] if isinstance(student_logit, tuple) else student_logit
                    loss = loss_fn(student_logit, target)

                base_loss_for_log = loss.detach()
                ref_attn_kl_loss = loss.new_zeros(())
                use_ref_scheme = (
                    runtime_args.train_scheme == "ema_ref_attn_kl"
                    and ref_model is not None
                    and epoch >= runtime_args.ref_warmup_epochs
                    and runtime_args.ref_attn_kl_weight > 0
                )
                if use_ref_scheme:
                    with torch.no_grad():
                        _, ref_attn_info = ref_model(input)
                    ref_attn_kl_loss = attention_kl_consistency_loss(
                        student_attn_info,
                        ref_attn_info,
                        head_mode=runtime_args.ref_head_mode,
                    )
                    loss = loss + runtime_args.ref_attn_kl_weight * ref_attn_kl_loss

            loss_for_log = loss.detach()
            ref_attn_kl_loss_for_log = ref_attn_kl_loss.detach()
            if not runtime_args.distributed:
                losses_m.update(loss_for_log.item(), input.size(0))
                base_losses_m.update(base_loss_for_log.item(), input.size(0))
                ref_attn_kl_losses_m.update(ref_attn_kl_loss_for_log.item(), input.size(0))

            scaled_loss = loss / accum_steps
            if update_step and runtime_args.train_scheme == "ema_ref_attn_kl" and ref_model is not None and runtime_args.ref_update == "prev_step":
                update_ref_model(model, ref_model, 0.0)
            if loss_scaler is not None:
                loss_scaler(
                    scaled_loss,
                    optimizer,
                    clip_grad=runtime_args.clip_grad,
                    clip_mode=runtime_args.clip_mode,
                    parameters=model_parameters(model, exclude_head="agc" in runtime_args.clip_mode),
                    create_graph=second_order,
                    update_grad=update_step,
                )
            else:
                scaled_loss.backward(create_graph=second_order)
                if update_step:
                    if runtime_args.clip_grad is not None:
                        dispatch_clip_grad(model_parameters(model, exclude_head="agc" in runtime_args.clip_mode), value=runtime_args.clip_grad, mode=runtime_args.clip_mode)
                    optimizer.step()

        if update_step:
            optimizer.zero_grad()
            local_update_count += 1
            if runtime_args.train_scheme == "ema_ref_attn_kl" and ref_model is not None and runtime_args.ref_update == "ema":
                update_ref_model(model, ref_model, runtime_args.ref_momentum)
            if runtime_args.local_rank == 0 and runtime_args.save_step_checkpoints:
                interval = max(1, int(runtime_args.step_checkpoint_interval))
                if warmup_updates > 0:
                    if runtime_args.save_initial_step_checkpoint and local_update_count == warmup_updates:
                        save_step_checkpoint(model, optimizer, runtime_args, output_dir, f"step_{saved_step_count:04d}", epoch=epoch, batch_idx=batch_idx, loss_scaler=loss_scaler)
                        saved_step_count += 1
                    if local_update_count > warmup_updates and (local_update_count - warmup_updates) % interval == 0:
                        if max_step_checkpoints_to_save == 0 or saved_step_count < max_step_checkpoints_to_save:
                            save_step_checkpoint(model, optimizer, runtime_args, output_dir, f"step_{saved_step_count:04d}", epoch=epoch, batch_idx=batch_idx, loss_scaler=loss_scaler)
                            saved_step_count += 1
                    if max_step_checkpoints_to_save > 0 and saved_step_count >= max_step_checkpoints_to_save:
                        stopped_early = True
                        break
                elif local_update_count % interval == 0:
                    save_step_checkpoint(model, optimizer, runtime_args, output_dir, f"step_{local_update_count:04d}", epoch=epoch, batch_idx=batch_idx, loss_scaler=loss_scaler)

        torch.cuda.synchronize()
        if update_step:
            num_updates += 1
            lr_scheduler.step_update(num_updates)
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % runtime_args.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            if runtime_args.distributed:
                reduced_loss = reduce_tensor(loss.data, runtime_args.world_size)
                reduced_base_loss = reduce_tensor(base_loss_for_log, runtime_args.world_size)
                reduced_ref_attn_kl_loss = reduce_tensor(ref_attn_kl_loss_for_log, runtime_args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
                base_losses_m.update(reduced_base_loss.item(), input.size(0))
                ref_attn_kl_losses_m.update(reduced_ref_attn_kl_loss.item(), input.size(0))
            if runtime_args.local_rank == 0:
                print(
                    f"Train: {epoch} [{batch_idx:>4d}/{len(loader)} ({100. * batch_idx / last_idx:>3.0f}%)]  "
                    f"Loss: {losses_m.val:>9.6f} ({losses_m.avg:>6.4f})  "
                    f"BaseLoss: {base_losses_m.val:>9.6f} ({base_losses_m.avg:>6.4f})  "
                    f"RefAttnKL: {ref_attn_kl_losses_m.val:.3e} ({ref_attn_kl_losses_m.avg:.3e})  "
                    f"Time: {batch_time_m.val:.3f}s, {input.size(0) * runtime_args.world_size / batch_time_m.val:>7.2f}/s  "
                    f"({batch_time_m.avg:.3f}s, {input.size(0) * runtime_args.world_size / batch_time_m.avg:>7.2f}/s)  "
                    f"LR: {lr:.3e}  Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})"
                )

        if runtime_args.max_train_updates and local_update_count >= runtime_args.max_train_updates:
            stopped_early = True
            break
        end = time.time()

    return {"loss": losses_m.avg}, local_update_count, stopped_early


def run_unified_ofq(local_rank: int, runtime_args: SimpleNamespace) -> None:
    setup_default_logging()
    runtime_args.local_rank = local_rank
    runtime_args.distributed = runtime_args.world_size > 1
    runtime_args.rank = local_rank if runtime_args.distributed else 0
    runtime_args.device = f"cuda:{local_rank if runtime_args.distributed else runtime_args.gpu_id}"

    if runtime_args.distributed:
        dist.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{runtime_args.tcp_port}", rank=local_rank, world_size=runtime_args.world_size)
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(runtime_args.gpu_id)

    random_seed(runtime_args.seed, runtime_args.rank)
    import src  # noqa: F401

    qqkkvv = runtime_args.kd_hard_and_soft in {2, 3}
    if runtime_args.model_type == "deit":
        model = create_model(runtime_args.model, num_classes=runtime_args.num_classes, drop_rate=runtime_args.drop, pretrained=runtime_args.pretrained, qqkkvv=qqkkvv)
    else:
        model = create_model(runtime_args.model, drop_path=runtime_args.drop_path, num_classes=runtime_args.num_classes, pretrained=runtime_args.pretrained, qqkkvv=qqkkvv)

    if runtime_args.quantized:
        model = get_ofq_qat_model(model, runtime_args)
    if runtime_args.collect_attention:
        enabled_modules = enable_attention_collection(model)
        if runtime_args.local_rank == 0:
            print(f"Enabled attention collection for {enabled_modules} modules.")
    if runtime_args.train_scheme == "ema_ref_attn_kl":
        set_attention_mode(model, collect_attention=True, qqkkvv=False)
    if runtime_args.initial_checkpoint and not runtime_args.eval_only:
        load_checkpoint(model, runtime_args.initial_checkpoint, strict=False)

    teacher = None
    runtime_args.use_kd = runtime_args.use_kd or runtime_args.use_token_kd
    if runtime_args.use_kd:
        if runtime_args.local_rank == 0:
            print("create teacher model")
        teacher = create_ofq_teacher_model(runtime_args).cuda()
        teacher.eval()

    model.cuda()
    if runtime_args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    use_amp = bool(runtime_args.amp or runtime_args.native_amp)
    amp_autocast = torch.cuda.amp.autocast if use_amp else contextlib.suppress
    loss_scaler = NativeScaler() if use_amp else None

    data_config = resolve_data_config(vars(runtime_args), model=model, verbose=runtime_args.local_rank == 0)
    if runtime_args.eval_only:
        dataset_train = create_dataset_compat(
            runtime_args.dataset,
            root=runtime_args.data_dir,
            split=runtime_args.train_split,
            is_training=True,
            batch_size=runtime_args.batch_size,
            subset_ratio=runtime_args.subset_ratio,
        )
        train_interpolation = runtime_args.train_interpolation or data_config["interpolation"]
        loader_train = create_loader_compat(
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
            interpolation=train_interpolation,
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=runtime_args.workers,
            distributed=runtime_args.distributed,
            collate_fn=None,
            pin_memory=runtime_args.pin_mem,
            use_multi_epochs_loader=runtime_args.use_multi_epochs_loader,
        )
        if runtime_args.local_rank == 0:
            print(f"{len(dataset_train)}")
        setup_alpha(model, loader_train, runtime_args, amp_autocast)
        if runtime_args.initial_checkpoint:
            load_checkpoint(model, runtime_args.initial_checkpoint, strict=False)
        dataset_eval = create_dataset_compat(
            runtime_args.dataset,
            root=runtime_args.data_dir,
            split=runtime_args.val_split,
            is_training=False,
            batch_size=runtime_args.batch_size,
            subset_ratio=runtime_args.subset_ratio,
        )
        loader_eval = create_loader_compat(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=runtime_args.validation_batch_size_multiplier * runtime_args.batch_size,
            is_training=False,
            use_prefetcher=runtime_args.prefetcher,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=runtime_args.workers,
            distributed=runtime_args.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=runtime_args.pin_mem,
        )
        if runtime_args.local_rank == 0:
            print(f"{len(dataset_eval)}")
        if runtime_args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
        try:
            metrics = validate_ofq(model, loader_eval, validate_loss_fn, runtime_args, amp_autocast)
            if runtime_args.local_rank == 0:
                print(f"Eval-only metrics: {metrics}")
        finally:
            cleanup_torch_distributed()
        return

    dataset_train = create_dataset_compat(
        runtime_args.dataset,
        root=runtime_args.data_dir,
        split=runtime_args.train_split,
        is_training=True,
        batch_size=runtime_args.batch_size,
        subset_ratio=runtime_args.subset_ratio,
    )

    collate_fn = None
    mixup_fn = None
    mixup_active = runtime_args.mixup > 0 or runtime_args.cutmix > 0.0 or runtime_args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=runtime_args.mixup,
            cutmix_alpha=runtime_args.cutmix,
            cutmix_minmax=runtime_args.cutmix_minmax,
            prob=runtime_args.mixup_prob,
            switch_prob=runtime_args.mixup_switch_prob,
            mode=runtime_args.mixup_mode,
            label_smoothing=runtime_args.smoothing,
            num_classes=runtime_args.num_classes,
        )
        if runtime_args.prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if runtime_args.aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=runtime_args.aug_splits)

    train_interpolation = runtime_args.train_interpolation or data_config["interpolation"]
    loader_train = create_loader_compat(
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
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=runtime_args.workers,
        distributed=runtime_args.distributed,
        collate_fn=collate_fn,
        pin_memory=runtime_args.pin_mem,
        use_multi_epochs_loader=runtime_args.use_multi_epochs_loader,
    )
    if runtime_args.local_rank == 0:
        print(f"{len(dataset_train)}")

    loader_eval = None
    if not runtime_args.skip_validate:
        dataset_eval = create_dataset_compat(
            runtime_args.dataset,
            root=runtime_args.data_dir,
            split=runtime_args.val_split,
            is_training=False,
            batch_size=runtime_args.batch_size,
            subset_ratio=runtime_args.subset_ratio,
        )
        loader_eval = create_loader_compat(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=runtime_args.validation_batch_size_multiplier * runtime_args.batch_size,
            is_training=False,
            use_prefetcher=runtime_args.prefetcher,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=runtime_args.workers,
            distributed=runtime_args.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=runtime_args.pin_mem,
        )
        if runtime_args.local_rank == 0:
            print(f"{len(dataset_eval)}")

    setup_alpha(model, loader_train, runtime_args, amp_autocast)
    optimizer = create_ofq_optimizer(runtime_args, model)

    start_epoch = 0
    if runtime_args.resume:
        start_epoch = resume_checkpoint(model, runtime_args.resume, optimizer=None if runtime_args.no_resume_opt else optimizer, loss_scaler=None if runtime_args.no_resume_opt else loss_scaler, log_info=runtime_args.local_rank == 0) or 0

    if runtime_args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    ref_model = None
    if runtime_args.train_scheme == "ema_ref_attn_kl":
        ref_model = clone_ref_model(model)
        if runtime_args.local_rank == 0:
            print(
                "Enabled EMA refmodel attention-KL scheme: "
                f"ref_update={runtime_args.ref_update}, "
                f"momentum={runtime_args.ref_momentum}, "
                f"attn_kl_weight={runtime_args.ref_attn_kl_weight}, "
                f"head_mode={runtime_args.ref_head_mode}, "
                f"warmup_epochs={runtime_args.ref_warmup_epochs}"
            )

    updates_per_epoch = max(1, (len(loader_train) + max(1, runtime_args.grad_accum_steps) - 1) // max(1, runtime_args.grad_accum_steps))
    lr_scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=runtime_args.lr,
        min_lr=runtime_args.min_lr,
        warmup_updates=runtime_args.warmup_epochs * updates_per_epoch,
        total_updates=runtime_args.epochs * updates_per_epoch,
    )
    if start_epoch > 0:
        lr_scheduler.step_update(start_epoch * updates_per_epoch)

    if runtime_args.local_rank == 0:
        print(f"Model {safe_model_name(runtime_args.model)} created, param count:{sum(m.numel() for m in model.parameters())}")
        print(f"Scheduled epochs: {runtime_args.epochs}")
        print(
            "Effective batch alignment: "
            f"single-process batch={runtime_args.batch_size} x accum={runtime_args.single_process_grad_accum_steps} "
            f"= {runtime_args.single_process_effective_batch_size}; "
            f"distributed batch={runtime_args.batch_size} x world_size={runtime_args.world_size} x accum={runtime_args.grad_accum_steps} "
            f"= {runtime_args.effective_batch_size}"
        )

    train_loss_fn = create_ofq_loss(runtime_args)
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    output_dir = Path(runtime_args.output) / runtime_args.experiment
    if runtime_args.local_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "args.yaml", "w", encoding="utf-8") as handle:
            handle.write(yaml.safe_dump(vars(runtime_args), allow_unicode=True, sort_keys=True))
        if runtime_args.save_step_checkpoints and runtime_args.save_initial_step_checkpoint and int(runtime_args.step_checkpoint_warmup_updates) == 0:
            save_step_checkpoint(model, optimizer, runtime_args, output_dir, "step_0000", epoch=start_epoch, batch_idx=-1, loss_scaler=loss_scaler)

    try:
        for epoch in range(start_epoch, runtime_args.epochs):
            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(epoch)
            if runtime_args.distributed and hasattr(loader_train, "sampler") and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)
            train_metrics, local_update_count, stopped_early = train_one_epoch_ofq(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                runtime_args,
                lr_scheduler,
                output_dir,
                amp_autocast,
                loss_scaler,
                teacher,
                mixup_fn,
                ref_model,
            )
            if runtime_args.local_rank == 0:
                print("epoch: ", epoch, "g['lr']: ", optimizer.param_groups[0]["lr"])
            should_save_epoch_checkpoint = (
                (epoch + 1) % max(1, runtime_args.epoch_checkpoint_interval) == 0
                or (epoch + 1) == runtime_args.epochs
                or stopped_early
            )
            if runtime_args.local_rank == 0 and should_save_epoch_checkpoint:
                save_epoch_checkpoint(model, optimizer, runtime_args, output_dir, epoch, loss_scaler=loss_scaler)
            if runtime_args.distributed and loader_eval is not None:
                dist.barrier()
            if loader_eval is not None:
                validate_ofq(model, loader_eval, validate_loss_fn, runtime_args, amp_autocast)
            if stopped_early:
                if runtime_args.local_rank == 0:
                    print(f"Stopped early after {local_update_count} optimizer updates in epoch {epoch}.")
                break
    finally:
        cleanup_torch_distributed()


def ofq_spawn_entry_unified(local_rank: int, cwd_str: str, runtime_dict: Dict[str, object], env: Dict[str, str]) -> None:
    cwd = Path(cwd_str)
    with patched_environ(env), patched_sys_path([cwd]), patched_cwd(cwd):
        run_unified_ofq(local_rank, SimpleNamespace(**runtime_dict))



@contextlib.contextmanager
def patched_environ(overrides: Dict[str, str]) -> Iterator[None]:
    previous: Dict[str, Optional[str]] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextlib.contextmanager
def patched_argv(argv: Sequence[str]) -> Iterator[None]:
    old_argv = sys.argv[:]
    try:
        sys.argv = list(argv)
        yield
    finally:
        sys.argv = old_argv


@contextlib.contextmanager
def patched_sys_path(extra_paths: Sequence[Path]) -> Iterator[None]:
    originals = sys.path[:]
    try:
        for path in reversed([str(item) for item in extra_paths]):
            if path not in sys.path:
                sys.path.insert(0, path)
        yield
    finally:
        sys.path = originals


@contextlib.contextmanager
def patched_cwd(path: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def cleanup_torch_distributed() -> None:
    try:
        import torch.distributed as dist
    except ImportError:
        return

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def script_argv_from_command(method: str, command: Sequence[str]) -> List[str]:
    if method == "qvit":
        if "main.py" not in command:
            raise ValueError(f"无法从命令中解析 Q-ViT 参数: {command}")
        return ["main.py", *command[command.index("main.py") + 1:]]
    if method == "ofq":
        script_name = next((item for item in command if item.endswith(".py")), None)
        if script_name is None:
            raise ValueError(f"无法从命令中解析 OFQ 参数: {command}")
        return [script_name, *command[command.index(script_name) + 1:]]
    if method == "aoq":
        if "train.py" not in command:
            raise ValueError(f"无法从命令中解析 AOQ 参数: {command}")
        return ["train.py", *command[command.index("train.py") + 1:]]
    raise ValueError(f"未知 method: {method}")


def invoke_qvit(command: Sequence[str], cwd: Path, env: Dict[str, str]) -> int:
    argv = script_argv_from_command("qvit", command)
    with patched_environ(env), patched_sys_path([cwd]), patched_cwd(cwd), patched_argv(argv):
        module = load_module("qats_qvit_main", cwd / "main.py")
        parsed_args = module.get_args_parser().parse_args(argv[1:])
        if parsed_args.output_dir:
            Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
        module.main(parsed_args)
    return 0


def ofq_spawn_entry(local_rank: int, cwd_str: str, argv: Sequence[str], env: Dict[str, str]) -> None:
    cwd = Path(cwd_str)
    with patched_environ(env), patched_sys_path([cwd]), patched_cwd(cwd), patched_argv(argv):
        if "train" in sys.modules:
            del sys.modules["train"]
        module = importlib.import_module(Path(argv[0]).stem)
        args_tuple = module.parse_args()
        parsed_args, _ = args_tuple
        os.environ["CUDA_VISIBLE_DEVICES"] = parsed_args.visible_gpu
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = parsed_args.world_size
        try:
            module.main(local_rank, args_tuple)
        finally:
            cleanup_torch_distributed()


def invoke_ofq(args: argparse.Namespace, command: Sequence[str], cwd: Path, env: Dict[str, str]) -> int:
    world_size = count_devices(args.devices, args.nproc_per_node)
    runtime_args = build_ofq_runtime_config(args)
    env = env.copy()
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices
    env.setdefault("RANK", "0")
    env.setdefault("LOCAL_RANK", "0")
    env["WORLD_SIZE"] = str(world_size)
    env["NCCL_DEBUG"] = "WARN"
    with patched_environ(env), patched_sys_path([cwd]), patched_cwd(cwd):
        if world_size > 1:
            torch.multiprocessing.spawn(
                ofq_spawn_entry_unified,
                args=(str(cwd), vars(runtime_args), env),
                nprocs=world_size,
                join=True,
            )
        else:
            ofq_spawn_entry_unified(0, str(cwd), vars(runtime_args), env)
    return 0


def invoke_aoq(command: Sequence[str], cwd: Path, env: Dict[str, str]) -> int:
    argv = script_argv_from_command("aoq", command)
    with patched_environ(env), patched_sys_path([cwd, cwd.parent]), patched_cwd(cwd), patched_argv(argv):
        module = load_module("qats_aoq_train", cwd / "train.py")
        module.main()
    return 0


def execute_method(args: argparse.Namespace, command: Sequence[str], cwd: Path, env: Dict[str, str]) -> int:
    if args.method == "qvit":
        if args.nproc_per_node and args.nproc_per_node > 1:
            raise NotImplementedError("当前统一主 pipeline 仅支持单进程 Q-ViT；请先使用 --nproc-per-node 1。")
        return invoke_qvit(command, cwd, env)
    if args.method == "ofq":
        return invoke_ofq(args, command, cwd, env)
    if args.method == "aoq":
        return invoke_aoq(command, cwd, env)
    raise ValueError(f"未知 method: {args.method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QATs 统一训练启动入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=["qvit", "ofq", "aoq"], required=True, help="选择训练方法")
    parser.add_argument("--data", type=str, help="数据集根目录")
    parser.add_argument("--output", type=str, help="输出目录/保存目录")
    parser.add_argument("--model", type=str, help="显式模型名；对 AOQ 表示 student")
    parser.add_argument("--teacher", type=str, help="教师模型名")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--batch-size-eval", dest="batch_size_eval", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float)
    parser.add_argument("--warmup-epochs", dest="warmup_epochs", type=int)
    parser.add_argument("--warmup-lr", dest="warmup_lr", type=float)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--devices", type=str, help="GPU 列表，例如 0,1,2,3")
    parser.add_argument("--nproc-per-node", dest="nproc_per_node", type=int, help="Q-ViT torchrun 进程数；OFQ 也可用来推断 world_size")
    parser.add_argument("--master-port", dest="master_port", type=int, default=29500)
    parser.add_argument("--bits", type=int, help="统一位宽；Q-ViT/AOQ 使用它，OFQ 可作为 wbits/abits 默认值")
    parser.add_argument("--wbits", type=int, help="OFQ 权重量化位宽")
    parser.add_argument("--abits", type=int, help="OFQ 激活量化位宽")
    parser.add_argument("--dataset-format", choices=["folder", "parquet", "parquet-iter"], default="folder")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", type=str, help="训练设备字符串；AOQ 会写入 QATS_DEVICE")
    parser.add_argument("--eval", action="store_true", help="仅评估（当前主要用于 Q-ViT）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不实际执行")
    parser.add_argument("--extra-arg", action="append", default=[], help="透传给原始训练脚本的额外参数，可重复传入")

    parser.add_argument("--arch", choices=["deit_tiny", "deit_small", "swin_tiny"], help="Q-ViT 预设模型架构")
    parser.add_argument("--distillation-type", default="none", choices=["none", "soft", "hard"], help="Q-ViT 蒸馏模式")
    parser.add_argument("--repeated-aug", action="store_true", help="Q-ViT 是否启用 repeated augmentation")

    parser.add_argument("--stage", choices=["train", "cga"], default="train", help="OFQ 阶段")
    parser.add_argument("--task", choices=["imagenet", "cifar10"], default="imagenet", help="AOQ 任务类型")
    parser.add_argument("--config", type=str, help="OFQ 配置文件")
    parser.add_argument("--experiment", type=str, help="OFQ 实验名")
    parser.add_argument("--grad-accum-steps", dest="grad_accum_steps", type=int)
    parser.add_argument("--checkpoint-hist", dest="checkpoint_hist", type=int, help="OFQ 最多保留的 epoch checkpoint 数")
    parser.add_argument("--epoch-checkpoint-interval", dest="epoch_checkpoint_interval", type=int, help="OFQ 每隔多少个 epoch 保存一次 checkpoint")
    parser.add_argument("--model-type", dest="model_type", type=str, choices=["deit", "swin"], help="OFQ model_type")
    parser.add_argument("--teacher-type", dest="teacher_type", type=str, choices=["deit", "swin"], help="OFQ teacher_type")
    parser.add_argument("--wq-mode", dest="wq_mode", type=str, default="statsq")
    parser.add_argument("--aq-mode", dest="aq_mode", type=str, default="lsq")
    parser.add_argument("--wq-per-channel", dest="wq_per_channel", action="store_true")
    parser.add_argument("--aq-per-channel", dest="aq_per_channel", action="store_true")
    parser.add_argument("--wq-clip-learnable", dest="wq_clip_learnable", action="store_true")
    parser.add_argument("--aq-clip-learnable", dest="aq_clip_learnable", action="store_true")
    parser.add_argument("--use-kd", dest="use_kd", action="store_true")
    parser.add_argument("--kd-hard-and-soft", dest="kd_hard_and_soft", type=int)
    parser.add_argument("--teacher-pretrained", dest="teacher_pretrained", action="store_true")
    parser.add_argument("--teacher-checkpoint", dest="teacher_checkpoint", type=str)
    parser.add_argument("--pretrained-initialized", dest="pretrained_initialized", action="store_true")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--qk-reparam", dest="qk_reparam", action="store_true")
    parser.add_argument("--qk-reparam-type", dest="qk_reparam_type", type=int)
    parser.add_argument("--boundary-range", dest="boundary_range", type=float)
    parser.add_argument("--freeze-for-n-epochs", dest="freeze_for_n_epochs", type=int)
    parser.add_argument("--train-scheme", dest="train_scheme", choices=["baseline", "ema_ref_attn_kl"], default=None, help="OFQ 训练方案名")
    parser.add_argument("--ref-update", dest="ref_update", choices=["ema", "prev_step"], default=None, help="历史参考模型更新方式")
    parser.add_argument("--ref-momentum", dest="ref_momentum", type=float, default=None, help="EMA refmodel 动量")
    parser.add_argument("--ref-attn-kl-weight", dest="ref_attn_kl_weight", type=float, default=None, help="EMA refmodel attention KL 权重")
    parser.add_argument("--ref-head-mode", dest="ref_head_mode", choices=["all", "oscillating_top5"], default=None, help="refmodel head 级别接口")
    parser.add_argument("--ref-warmup-epochs", dest="ref_warmup_epochs", type=int, default=None, help="多少个 epoch 后再启用 refmodel attention KL")

    parser.add_argument("--quantize-downsample", dest="quantize_downsample", type=str2bool, default=True, help="AOQ 是否量化 downsample")
    parser.add_argument("--amp", action="store_true", help="AOQ mixed precision")
    parser.add_argument("--amp-dtype", dest="amp_dtype", choices=["bf16", "fp16"], default="bf16", help="AOQ mixed precision dtype")
    parser.add_argument("--channels-last", dest="channels_last", action="store_true", help="AOQ channels_last")
    parser.add_argument("--compile", action="store_true", help="AOQ torch.compile")
    parser.add_argument("--compile-mode", dest="compile_mode", type=str, default="default", help="AOQ torch.compile mode")
    parser.add_argument("--compile-backend", dest="compile_backend", type=str, default="inductor", help="AOQ torch.compile backend")
    parser.add_argument("--prefetch-factor", dest="prefetch_factor", type=int, default=4, help="AOQ dataloader prefetch factor")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", help="AOQ persistent workers")
    parser.add_argument("--val-interval", dest="val_interval", type=int, default=1, help="AOQ validation interval")
    parser.add_argument("--plot-interval", dest="plot_interval", type=int, default=0, help="AOQ histogram plot interval")
    parser.add_argument("--train-steps-per-epoch", dest="train_steps_per_epoch", type=int, default=0, help="AOQ max train steps per epoch")
    parser.add_argument("--val-steps", dest="val_steps", type=int, default=0, help="AOQ max val steps")
    parser.add_argument("--synthetic-data", dest="synthetic_data", action="store_true", help="AOQ use FakeData")
    parser.add_argument("--synthetic-train-size", dest="synthetic_train_size", type=int, default=32768, help="AOQ FakeData train size")
    parser.add_argument("--synthetic-val-size", dest="synthetic_val_size", type=int, default=4096, help="AOQ FakeData val size")
    parser.add_argument("--aoq-dataset-format", dest="aoq_dataset_format", choices=["imagefolder", "parquet", "parquet-iter"], default="imagefolder", help="AOQ dataset format")
    parser.add_argument("--skip-teacher-val", dest="skip_teacher_val", action="store_true", help="AOQ skip initial teacher validation")
    parser.add_argument("--print-model", dest="print_model", action="store_true", help="AOQ print full student model")
    parser.add_argument("--print-params", dest="print_params", action="store_true", help="AOQ print all params")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command, cwd, env = build_command(args)

    print(f"[QATs] method={args.method}")
    print(f"[QATs] cwd={cwd}")
    if "CUDA_VISIBLE_DEVICES" in env:
        print(f"[QATs] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    if "QATS_DEVICE" in env:
        print(f"[QATs] QATS_DEVICE={env['QATS_DEVICE']}")
    print(f"[QATs] command={shell_join(command)}")

    if args.dry_run:
        return 0

    return execute_method(args, command, cwd, env)


if __name__ == "__main__":
    raise SystemExit(main())
