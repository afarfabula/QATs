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
import json
import math
import functools
import os
import random
import shlex
import sys
import time
import types
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
from timm.models import create_model, load_checkpoint, model_parameters, safe_model_name
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


class ImageNetParquetEvalIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, root: str, split: str = "validation", transform=None, subset_ratio: float = 1.0, rank: Optional[int] = None, world_size: Optional[int] = None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.subset_ratio = float(subset_ratio)
        self.rank = rank
        self.world_size = world_size
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

        self._segments = []
        total_rows = 0
        for file_idx, path in enumerate(self.files):
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                self._segments.append((total_rows, total_rows + rg_rows, file_idx, rg_idx))
                total_rows += rg_rows
        self._total_rows = total_rows
        self._apply_subset_ratio()

    def _apply_subset_ratio(self) -> None:
        if self.subset_ratio <= 0 or self.subset_ratio > 1:
            raise ValueError(f"subset_ratio must be in (0, 1], got {self.subset_ratio}")
        if self.subset_ratio < 1:
            self._total_rows = max(1, int(math.ceil(self._total_rows * self.subset_ratio)))
            self._segments = [segment for segment in self._segments if segment[0] < self._total_rows]

    def _distributed_context(self):
        if self.rank is not None and self.world_size is not None:
            return int(self.rank), int(self.world_size)
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world_size

    def _rank_bounds(self) -> Tuple[int, int]:
        rank, world_size = self._distributed_context()
        start = self._total_rows * rank // world_size
        end = self._total_rows * (rank + 1) // world_size
        return start, end

    def __len__(self):
        start, end = self._rank_bounds()
        return max(0, end - start)

    def _iter_rank_segments(self):
        rank_start, rank_end = self._rank_bounds()
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        worker_start = rank_start + (rank_end - rank_start) * worker_id // num_workers
        worker_end = rank_start + (rank_end - rank_start) * (worker_id + 1) // num_workers
        for seg_start, seg_end, file_idx, rg_idx in self._segments:
            start = max(seg_start, worker_start)
            end = min(seg_end, worker_end)
            if start < end:
                yield file_idx, rg_idx, start - seg_start, end - seg_start

    def __iter__(self):
        handles = {}
        for file_idx, rg_idx, local_start, local_end in self._iter_rank_segments():
            path = self.files[file_idx]
            pf = handles.get(path)
            if pf is None:
                pf = pq.ParquetFile(path)
                handles[path] = pf
            table = pf.read_row_group(rg_idx, columns=["image", "label"])
            cols = table.to_pydict()
            images = cols["image"]
            labels = cols["label"]
            for idx in range(local_start, local_end):
                image = Image.open(io.BytesIO(images[idx]["bytes"])).convert("RGB")
                target = int(labels[idx])
                if self.transform is not None:
                    image = self.transform(image)
                yield image, target


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
    append_optional_value(command, "--scheduler-epochs", args.scheduler_epochs)
    append_optional_value(command, "--batch-size", args.batch_size)
    append_optional_value(command, "--grad-accum-steps", args.grad_accum_steps)
    append_optional_value(command, "--workers", args.workers)
    append_optional_value(command, "--lr", args.lr)
    append_optional_value(command, "--weight-decay", args.weight_decay)
    append_optional_value(command, "--warmup-epochs", args.warmup_epochs)
    append_optional_value(command, "--warmup-lr", args.warmup_lr)
    append_optional_value(command, "--min-lr", args.min_lr)
    append_optional_value(command, "--resume", normalize_path(args.resume))
    append_optional_flag(command, "--no-resume-opt", args.no_resume_opt)
    append_optional_flag(command, "--resume-opt-force-lr", args.resume_opt_force_lr)
    append_optional_value(command, "--start-epoch", args.start_epoch)
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
    append_optional_value(command, "--ref-update-interval", args.ref_update_interval)
    append_optional_value(command, "--ref-momentum", args.ref_momentum)
    append_optional_value(command, "--ref-attn-kl-weight", args.ref_attn_kl_weight)
    append_optional_value(command, "--ref-attn-kl-drop-prob", args.ref_attn_kl_drop_prob)
    append_optional_flag(command, "--ref-attn-kl-drop-scale", args.ref_attn_kl_drop_scale)
    append_optional_value(command, "--ref-attn-kl-clip", args.ref_attn_kl_clip)
    append_optional_value(command, "--ref-attn-loss", args.ref_attn_loss)
    append_optional_value(command, "--ref-logit-kl-weight", args.ref_logit_kl_weight)
    append_optional_value(command, "--ref-logit-kl-temperature", args.ref_logit_kl_temperature)
    append_optional_value(command, "--teacher-qk-rel-weight", args.teacher_qk_rel_weight)
    append_optional_value(command, "--teacher-qk-rel-warmup-epochs", args.teacher_qk_rel_warmup_epochs)
    append_optional_value(command, "--teacher-qkv-rel-weight", getattr(args, "teacher_qkv_rel_weight", None))
    append_optional_value(command, "--teacher-qkv-rel-warmup-epochs", getattr(args, "teacher_qkv_rel_warmup_epochs", None))
    append_optional_value(command, "--teacher-qkv-rel-layers", getattr(args, "teacher_qkv_rel_layers", None))
    append_optional_value(command, "--teacher-qkv-rel-components", getattr(args, "teacher_qkv_rel_components", None))
    append_optional_value(command, "--clean-start-target-loss-weight", args.clean_start_target_loss_weight)
    append_optional_value(command, "--ref-head-mode", args.ref_head_mode)
    append_optional_value(command, "--ref-warmup-epochs", args.ref_warmup_epochs)
    append_optional_value(command, "--ref-warmup-updates", args.ref_warmup_updates)
    append_optional_value(command, "--ref-stop-updates", args.ref_stop_updates)
    append_optional_value(command, "--anchor-ref-attn-kl-weight", args.anchor_ref_attn_kl_weight)
    append_optional_value(command, "--anchor-ref-warmup-epochs", args.anchor_ref_warmup_epochs)
    append_optional_value(command, "--anchor-ref-head-mode", args.anchor_ref_head_mode)
    append_optional_value(command, "--teacher-attn-kl-weight", args.teacher_attn_kl_weight)
    append_optional_value(command, "--teacher-attn-kl-warmup-epochs", args.teacher_attn_kl_warmup_epochs)
    append_optional_value(command, "--teacher-attn-output-weight", args.teacher_attn_output_weight)
    append_optional_value(command, "--teacher-attn-output-layers", args.teacher_attn_output_layers)
    append_optional_value(command, "--teacher-attn-output-warmup-epochs", args.teacher_attn_output_warmup_epochs)
    append_optional_value(command, "--teacher-attn-output-weight-epoch-overrides", getattr(args, "teacher_attn_output_weight_epoch_overrides", None))
    append_optional_value(command, "--teacher-feature-output-weight", args.teacher_feature_output_weight)
    append_optional_value(command, "--teacher-feature-output-layers", args.teacher_feature_output_layers)
    append_optional_value(command, "--teacher-feature-output-warmup-epochs", args.teacher_feature_output_warmup_epochs)
    append_optional_value(command, "--teacher-feature-output-loss", args.teacher_feature_output_loss)
    append_optional_value(command, "--bin-reg-weight", args.bin_reg_weight)
    append_optional_value(command, "--bin-reg-variance-weight", args.bin_reg_variance_weight)
    append_optional_value(command, "--bin-reg-layers", getattr(args, "bin_reg_layers", None))
    append_optional_flag(command, "--bin-reg-attn-only", getattr(args, "bin_reg_attn_only", False))
    append_optional_value(command, "--bin-reg-start-update", getattr(args, "bin_reg_start_update", None))
    append_optional_value(command, "--bin-reg-end-update", getattr(args, "bin_reg_end_update", None))
    append_optional_value(command, "--selective-bin-anchor-weight", getattr(args, "selective_bin_anchor_weight", None))
    append_optional_value(command, "--selective-bin-anchor-layers", getattr(args, "selective_bin_anchor_layers", None))
    append_optional_value(command, "--selective-bin-anchor-capture-update", getattr(args, "selective_bin_anchor_capture_update", None))
    append_optional_value(command, "--selective-bin-anchor-end-update", getattr(args, "selective_bin_anchor_end_update", None))
    append_optional_value(command, "--selective-bin-anchor-margin", getattr(args, "selective_bin_anchor_margin", None))
    append_optional_value(command, "--candidate-bin-anchor-weight", getattr(args, "candidate_bin_anchor_weight", None))
    append_optional_value(command, "--candidate-bin-anchor-layers", getattr(args, "candidate_bin_anchor_layers", None))
    append_optional_value(command, "--candidate-bin-anchor-capture-update", getattr(args, "candidate_bin_anchor_capture_update", None))
    append_optional_value(command, "--candidate-bin-anchor-end-update", getattr(args, "candidate_bin_anchor_end_update", None))
    append_optional_value(command, "--candidate-bin-anchor-source-checkpoint", getattr(args, "candidate_bin_anchor_source_checkpoint", None))
    append_optional_value(command, "--weight-bin-telemetry-layers", getattr(args, "weight_bin_telemetry_layers", None))
    append_optional_value(command, "--weight-bin-telemetry-start-update", getattr(args, "weight_bin_telemetry_start_update", None))
    append_optional_value(command, "--weight-bin-telemetry-end-update", getattr(args, "weight_bin_telemetry_end_update", None))
    append_optional_value(command, "--weight-bin-telemetry-interval", getattr(args, "weight_bin_telemetry_interval", None))
    append_optional_value(command, "--weight-bin-telemetry-margin", getattr(args, "weight_bin_telemetry_margin", None))
    append_optional_value(command, "--act-bin-margin-weight", getattr(args, "act_bin_margin_weight", None))
    append_optional_value(command, "--act-bin-margin-layers", getattr(args, "act_bin_margin_layers", None))
    append_optional_value(command, "--act-bin-margin-quantizers", getattr(args, "act_bin_margin_quantizers", None))
    append_optional_value(command, "--act-bin-margin", getattr(args, "act_bin_margin", None))
    append_optional_value(command, "--act-bin-margin-max-elements", getattr(args, "act_bin_margin_max_elements", None))
    append_optional_value(command, "--epoch1-acc-gate", args.epoch1_acc_gate)
    append_optional_value(command, "--teacher-confidence-kd-power", args.teacher_confidence_kd_power)
    append_optional_value(command, "--teacher-confidence-band-kd-weight", getattr(args, "teacher_confidence_band_kd_weight", None))
    append_optional_value(command, "--teacher-confidence-band-kd-low", getattr(args, "teacher_confidence_band_kd_low", None))
    append_optional_value(command, "--teacher-confidence-band-kd-high", getattr(args, "teacher_confidence_band_kd_high", None))
    append_optional_value(command, "--teacher-confidence-band-kd-temperature", getattr(args, "teacher_confidence_band_kd_temperature", None))
    append_optional_value(command, "--ref-confidence-band-kd-weight", getattr(args, "ref_confidence_band_kd_weight", None))
    append_optional_value(command, "--ref-confidence-band-kd-low", getattr(args, "ref_confidence_band_kd_low", None))
    append_optional_value(command, "--ref-confidence-band-kd-high", getattr(args, "ref_confidence_band_kd_high", None))
    append_optional_value(command, "--ref-confidence-band-kd-temperature", getattr(args, "ref_confidence_band_kd_temperature", None))
    append_optional_value(command, "--ref-confidence-band-kd-checkpoint", getattr(args, "ref_confidence_band_kd_checkpoint", None))
    append_optional_value(command, "--local-ref-confidence-band-kd-weight", getattr(args, "local_ref_confidence_band_kd_weight", None))
    append_optional_value(command, "--local-ref-confidence-band-kd-low", getattr(args, "local_ref_confidence_band_kd_low", None))
    append_optional_value(command, "--local-ref-confidence-band-kd-high", getattr(args, "local_ref_confidence_band_kd_high", None))
    append_optional_value(command, "--local-ref-confidence-band-kd-temperature", getattr(args, "local_ref_confidence_band_kd_temperature", None))
    append_optional_value(command, "--local-ref-confidence-band-kd-checkpoint", getattr(args, "local_ref_confidence_band_kd_checkpoint", None))
    append_optional_value(command, "--class-protect-ref-kl-weight", getattr(args, "class_protect_ref_kl_weight", None))
    append_optional_value(command, "--class-protect-ref-kl-classes", getattr(args, "class_protect_ref_kl_classes", None))
    append_optional_value(command, "--class-protect-ref-kl-temperature", getattr(args, "class_protect_ref_kl_temperature", None))
    append_optional_value(command, "--class-protect-ref-kl-checkpoint", getattr(args, "class_protect_ref_kl_checkpoint", None))
    append_optional_value(command, "--teacher-soft-temperature", args.teacher_soft_temperature)
    append_optional_value(command, "--quant-lr-multiplier", args.quant_lr_multiplier)
    append_optional_value(command, "--quant-lr-multiplier-epoch-overrides", getattr(args, "quant_lr_multiplier_epoch_overrides", None))
    append_optional_value(command, "--quant-slow-state-decay", getattr(args, "quant_slow_state_decay", None))
    append_optional_value(command, "--quant-slow-state-sync-interval", getattr(args, "quant_slow_state_sync_interval", None))
    append_optional_value(command, "--quant-slow-state-pull", getattr(args, "quant_slow_state_pull", None))
    append_optional_value(command, "--quant-slow-state-policy", getattr(args, "quant_slow_state_policy", None))
    append_optional_value(command, "--quant-slow-state-observe-start-epoch", getattr(args, "quant_slow_state_observe_start_epoch", None))
    append_optional_value(command, "--quant-slow-state-start-epoch", getattr(args, "quant_slow_state_start_epoch", None))
    append_optional_value(command, "--act-scale-anchor-weight", getattr(args, "act_scale_anchor_weight", None))
    append_optional_value(command, "--act-scale-anchor-layers", getattr(args, "act_scale_anchor_layers", None))
    append_optional_value(command, "--act-scale-anchor-start-epoch", getattr(args, "act_scale_anchor_start_epoch", None))
    append_optional_value(command, "--variation-trust-weight", getattr(args, "variation_trust_weight", None))
    append_optional_value(command, "--variation-trust-layers", getattr(args, "variation_trust_layers", None))
    append_optional_value(command, "--variation-trust-late-layers", getattr(args, "variation_trust_late_layers", None))
    append_optional_value(command, "--variation-trust-late-multiplier", getattr(args, "variation_trust_late_multiplier", None))
    append_optional_value(command, "--variation-trust-early-layers", getattr(args, "variation_trust_early_layers", None))
    append_optional_value(command, "--variation-trust-early-multiplier", getattr(args, "variation_trust_early_multiplier", None))
    append_optional_value(command, "--variation-trust-softmax-multiplier", getattr(args, "variation_trust_softmax_multiplier", None))
    append_optional_value(command, "--variation-trust-move-v-multiplier", getattr(args, "variation_trust_move_v_multiplier", None))
    append_optional_value(command, "--variation-trust-proj-move-multiplier", getattr(args, "variation_trust_proj_move_multiplier", None))
    append_optional_value(command, "--variation-trust-start-update", getattr(args, "variation_trust_start_update", None))
    append_optional_value(command, "--aoq-explore-scale-ratio", getattr(args, "aoq_explore_scale_ratio", None))
    append_optional_value(command, "--aoq-explore-threshold-ratio", getattr(args, "aoq_explore_threshold_ratio", None))
    append_optional_value(command, "--aoq-explore-layers", getattr(args, "aoq_explore_layers", None))
    append_optional_value(command, "--aoq-explore-layer-ratios", getattr(args, "aoq_explore_layer_ratios", None))
    append_optional_value(command, "--aoq-explore-selective-margin", getattr(args, "aoq_explore_selective_margin", None))
    append_optional_value(command, "--aoq-explore-quality-mode", getattr(args, "aoq_explore_quality_mode", None))
    append_optional_value(command, "--aoq-explore-quality-layers", getattr(args, "aoq_explore_quality_layers", None))
    append_optional_value(command, "--aoq-explore-quality-start-update", getattr(args, "aoq_explore_quality_start_update", None))
    append_optional_value(command, "--aoq-explore-quality-min-frac", getattr(args, "aoq_explore_quality_min_frac", None))
    append_optional_value(command, "--aoq-explore-anchor-checkpoint", getattr(args, "aoq_explore_anchor_checkpoint", None))
    append_optional_value(command, "--aoq-explore-start-update", getattr(args, "aoq_explore_start_update", None))
    append_optional_value(command, "--aoq-explore-end-update", getattr(args, "aoq_explore_end_update", None))
    append_optional_flag(command, "--aoq-explore-repeat-each-epoch", getattr(args, "aoq_explore_repeat_each_epoch", False))
    append_optional_value(command, "--aoq-explore-update-schedule", getattr(args, "aoq_explore_update_schedule", None))
    append_optional_value(command, "--delta-direction-anchor-weight", getattr(args, "delta_direction_anchor_weight", None))
    append_optional_value(command, "--delta-direction-anchor-base-checkpoint", getattr(args, "delta_direction_anchor_base_checkpoint", None))
    append_optional_value(command, "--delta-direction-anchor-target-checkpoint", getattr(args, "delta_direction_anchor_target_checkpoint", None))
    append_optional_value(command, "--delta-direction-anchor-params", getattr(args, "delta_direction_anchor_params", None))
    append_optional_value(command, "--delta-direction-anchor-start-update", getattr(args, "delta_direction_anchor_start_update", None))
    append_optional_value(command, "--pre-qat-act-percentile-calib-batches", getattr(args, "pre_qat_act_percentile_calib_batches", None))
    append_optional_value(command, "--pre-qat-act-percentile-calib-layers", getattr(args, "pre_qat_act_percentile_calib_layers", None))
    append_optional_value(command, "--pre-qat-act-percentile-calib-percentile", getattr(args, "pre_qat_act_percentile_calib_percentile", None))
    append_optional_value(command, "--pre-qat-act-percentile-calib-blend", getattr(args, "pre_qat_act_percentile_calib_blend", None))
    append_optional_value(command, "--pre-qat-act-mse-calib-batches", getattr(args, "pre_qat_act_mse_calib_batches", None))
    append_optional_value(command, "--pre-qat-act-mse-calib-layers", getattr(args, "pre_qat_act_mse_calib_layers", None))
    append_optional_value(command, "--pre-qat-act-mse-calib-quantizers", getattr(args, "pre_qat_act_mse_calib_quantizers", None))
    append_optional_value(command, "--pre-qat-act-mse-calib-grid", getattr(args, "pre_qat_act_mse_calib_grid", None))
    append_optional_value(command, "--pre-qat-act-mse-calib-blend", getattr(args, "pre_qat_act_mse_calib_blend", None))
    append_optional_value(command, "--pre-qat-recon-updates", getattr(args, "pre_qat_recon_updates", None))
    append_optional_value(command, "--pre-qat-recon-temperature", getattr(args, "pre_qat_recon_temperature", None))
    append_optional_value(command, "--pre-qat-feature-recon-updates", getattr(args, "pre_qat_feature_recon_updates", None))
    append_optional_value(command, "--pre-qat-feature-recon-layers", getattr(args, "pre_qat_feature_recon_layers", None))
    append_optional_value(command, "--pre-qat-feature-recon-policy", getattr(args, "pre_qat_feature_recon_policy", None))
    append_optional_value(command, "--pre-qat-feature-recon-confidence-power", getattr(args, "pre_qat_feature_recon_confidence_power", None))
    append_optional_value(command, "--pre-qat-feature-recon-weight-mode", getattr(args, "pre_qat_feature_recon_weight_mode", None))
    append_optional_value(command, "--pre-qat-feature-recon-qdrop-prob", getattr(args, "pre_qat_feature_recon_qdrop_prob", None))
    append_optional_value(command, "--pre-qat-feature-recon-qdrop-layers", getattr(args, "pre_qat_feature_recon_qdrop_layers", None))
    append_optional_value(command, "--pre-qat-feature-recon-anchor-kl-weight", getattr(args, "pre_qat_feature_recon_anchor_kl_weight", None))
    append_optional_value(command, "--pre-qat-feature-recon-anchor-kl-temperature", getattr(args, "pre_qat_feature_recon_anchor_kl_temperature", None))
    append_optional_value(command, "--post-epoch-feature-recon-updates", getattr(args, "post_epoch_feature_recon_updates", None))
    append_optional_value(command, "--pre-qat-seq-feature-recon-updates", getattr(args, "pre_qat_seq_feature_recon_updates", None))
    append_optional_value(command, "--pre-qat-seq-feature-recon-layers", getattr(args, "pre_qat_seq_feature_recon_layers", None))
    append_optional_value(command, "--pre-qat-seq-feature-recon-policy", getattr(args, "pre_qat_seq_feature_recon_policy", None))
    append_optional_value(command, "--ref-attn-kl-weight-epoch-overrides", args.ref_attn_kl_weight_epoch_overrides)
    append_optional_value(command, "--anchor-ref-attn-kl-weight-epoch-overrides", args.anchor_ref_attn_kl_weight_epoch_overrides)
    append_optional_value(command, "--ref-head-mode-epoch-overrides", getattr(args, "ref_head_mode_epoch_overrides", None))
    append_optional_flag(command, "--dynamic-sparse-prevstep-kl", getattr(args, "dynamic_sparse_prevstep_kl", False))
    append_optional_value(command, "--dynamic-kl-start-epoch", getattr(args, "dynamic_kl_start_epoch", None))
    append_optional_value(command, "--dynamic-kl-observe-until-epoch", getattr(args, "dynamic_kl_observe_until_epoch", None))
    append_optional_value(command, "--dynamic-kl-primary-heads", getattr(args, "dynamic_kl_primary_heads", None))
    append_optional_value(command, "--dynamic-kl-secondary-heads", getattr(args, "dynamic_kl_secondary_heads", None))
    append_optional_value(command, "--dynamic-kl-avoid-heads", getattr(args, "dynamic_kl_avoid_heads", None))
    append_optional_value(command, "--dynamic-kl-drop-threshold", getattr(args, "dynamic_kl_drop_threshold", None))
    append_optional_value(command, "--dynamic-kl-strong-drop-threshold", getattr(args, "dynamic_kl_strong_drop_threshold", None))
    append_optional_value(command, "--dynamic-kl-default-weight", getattr(args, "dynamic_kl_default_weight", None))
    append_optional_value(command, "--dynamic-kl-strong-weight", getattr(args, "dynamic_kl_strong_weight", None))
    append_optional_value(command, "--dynamic-kl-max-weight", getattr(args, "dynamic_kl_max_weight", None))
    append_optional_value(command, "--dynamic-kl-cooldown-epochs", getattr(args, "dynamic_kl_cooldown_epochs", None))
    append_optional_value(command, "--dynamic-kl-window-epochs", getattr(args, "dynamic_kl_window_epochs", None))
    append_optional_value(command, "--dynamic-kl-max-pulses-per-window", getattr(args, "dynamic_kl_max_pulses_per_window", None))
    append_optional_value(command, "--dynamic-kl-controller-tsv", getattr(args, "dynamic_kl_controller_tsv", None))
    append_optional_value(command, "--dynamic-kl-prior-source", getattr(args, "dynamic_kl_prior_source", None))
    append_optional_value(command, "--epoch-lr-overrides", args.epoch_lr_overrides)
    append_optional_value(command, "--progressive-bit-schedule", getattr(args, "progressive_bit_schedule", None))
    append_optional_flag(command, "--progressive-bit-rescale-lsq", getattr(args, "progressive_bit_rescale_lsq", False))
    append_optional_value(command, "--progressive-bit-recalibrate-epochs", getattr(args, "progressive_bit_recalibrate_epochs", None))
    append_optional_value(command, "--progressive-bit-recalibrate-batches", getattr(args, "progressive_bit_recalibrate_batches", None))
    append_optional_value(command, "--progressive-bit-transition-recon-updates", getattr(args, "progressive_bit_transition_recon_updates", None))
    append_optional_value(command, "--progressive-bit-transition-recon-epochs", getattr(args, "progressive_bit_transition_recon_epochs", None))
    append_optional_value(command, "--progressive-bit-transition-recon-layers", getattr(args, "progressive_bit_transition_recon_layers", None))
    append_optional_value(command, "--progressive-bit-transition-recon-policy", getattr(args, "progressive_bit_transition_recon_policy", None))
    append_optional_value(command, "--progressive-bit-transition-recon-confidence-power", getattr(args, "progressive_bit_transition_recon_confidence_power", None))
    append_optional_value(command, "--progressive-bit-transition-recon-weight-mode", getattr(args, "progressive_bit_transition_recon_weight_mode", None))
    append_optional_value(command, "--progressive-bit-transition-recon-qdrop-prob", getattr(args, "progressive_bit_transition_recon_qdrop_prob", None))
    append_optional_value(command, "--progressive-bit-transition-recon-qdrop-layers", getattr(args, "progressive_bit_transition_recon_qdrop_layers", None))
    append_optional_value(command, "--progressive-bit-transition-anchor-kl-weight", getattr(args, "progressive_bit_transition_anchor_kl_weight", None))
    append_optional_value(command, "--progressive-bit-transition-anchor-kl-temperature", getattr(args, "progressive_bit_transition_anchor_kl_temperature", None))
    append_optional_value(command, "--quant-only-start-epoch", args.quant_only_start_epoch)
    append_optional_value(command, "--trainable-policy", args.trainable_policy)
    append_optional_value(command, "--trainable-policy-freeze-act-except-layers", getattr(args, "trainable_policy_freeze_act_except_layers", None))
    append_optional_value(command, "--trainable-policy-update-overrides", args.trainable_policy_update_overrides)
    append_optional_value(command, "--trainable-policy-update-mode", args.trainable_policy_update_mode)
    append_optional_value(command, "--trainable-policy-grad-damp", getattr(args, "trainable_policy_grad_damp", None))
    append_optional_flag(command, "--model-ema", args.model_ema)
    append_optional_value(command, "--model-ema-decay", args.model_ema_decay)
    append_optional_value(command, "--post-resume-setup-alpha-batches", args.post_resume_setup_alpha_batches)
    use_native_amp = bool(getattr(args, "native_amp", False))
    append_optional_flag(command, "--amp", args.amp)
    append_optional_flag(command, "--native-amp", use_native_amp)
    append_optional_value(command, "--amp-dtype", args.amp_dtype if args.amp or use_native_amp else None)
    append_optional_flag(command, "--channels-last", args.channels_last)
    append_optional_flag(command, "--pin-mem", getattr(args, "pin_mem", False))

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


def create_dataset_compat(dataset_name, root, split, is_training, batch_size, repeats=0, transform=None, subset_ratio: float = 1.0, rank: Optional[int] = None, world_size: Optional[int] = None):
    if dataset_name == "hf-parquet-imagenet":
        if is_training:
            return ImageNetParquetIterableDataset(root=root, split=split, transform=transform, shuffle=True, subset_ratio=subset_ratio)
        return ImageNetParquetEvalIterableDataset(root=root, split=split, transform=transform, subset_ratio=subset_ratio, rank=rank, world_size=world_size)
    return create_dataset(dataset_name, root=root, split=split, is_training=is_training, batch_size=batch_size, repeats=repeats)


def create_loader_compat(dataset, **kwargs):
    sig = inspect.signature(create_loader)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if filtered.get("num_workers", 0) == 0 and "persistent_workers" in sig.parameters:
        filtered["persistent_workers"] = False
    return create_loader(dataset, **filtered)


def build_ofq_runtime_overrides(extra_args: Sequence[str]) -> Dict[str, object]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skip_validate", action="store_true")
    parser.add_argument("--eval-only", dest="eval_only", action="store_true")
    parser.add_argument("--max_train_updates", type=int)
    parser.add_argument("--start-epoch", dest="start_epoch", type=int)
    parser.add_argument("--log-interval", dest="log_interval", type=int)
    parser.add_argument("--save_step_checkpoints", action="store_true")
    parser.add_argument("--save_initial_step_checkpoint", action="store_true")
    parser.add_argument("--step_checkpoint_interval", type=int)
    parser.add_argument("--step_checkpoint_warmup_updates", type=int)
    parser.add_argument("--max_step_checkpoints_to_save", type=int)
    parser.add_argument("--collect_attention", action="store_true")
    parser.add_argument("--setup-alpha-batches", dest="setup_alpha_batches", type=int)
    parser.add_argument("--post-resume-setup-alpha-batches", dest="post_resume_setup_alpha_batches", type=int)
    parser.add_argument("--initial-checkpoint", dest="initial_checkpoint", type=str)
    parser.add_argument("--post-load-alpha", dest="post_load_alpha", action="store_true")
    parser.add_argument("--no-prefetcher", dest="no_prefetcher", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin-mem", dest="pin_mem", action="store_true")
    parser.add_argument("--sync-step-timing", dest="sync_step_timing", action="store_true")
    parser.add_argument("--static-graph", dest="static_graph", action="store_true")
    parser.add_argument("--no-gradient-as-bucket-view", dest="gradient_as_bucket_view", action="store_false")
    parser.add_argument("--compile", dest="compile", action="store_true")
    parser.add_argument("--compile-mode", dest="compile_mode", type=str)
    parser.add_argument("--channels-last", dest="channels_last", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aa", type=str)
    parser.add_argument("--reprob", type=float)
    parser.add_argument("--color-jitter", dest="color_jitter", type=float)
    parser.add_argument("--smoothing", type=float)
    parser.add_argument("--mixup", type=float)
    parser.add_argument("--cutmix", type=float)
    parser.add_argument("--mixup-prob", dest="mixup_prob", type=float)
    parser.add_argument("--mixup-switch-prob", dest="mixup_switch_prob", type=float)
    parser.add_argument("--mixup-mode", dest="mixup_mode", type=str)
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
    parser.add_argument("--scheduler-epochs", dest="scheduler_epochs", type=int)
    parser.add_argument("--min-lr", dest="min_lr", type=float)
    parser.add_argument("--recovery-interval", dest="recovery_interval", type=int)
    parser.add_argument("--checkpoint-hist", dest="checkpoint_hist", type=int)
    parser.add_argument("--epoch-checkpoint-interval", dest="epoch_checkpoint_interval", type=int)
    parser.add_argument("--subset-ratio", dest="subset_ratio", type=float)
    parser.add_argument("--initial_checkpoint", dest="initial_checkpoint_alias", type=str)
    parser.add_argument("--no-resume-opt", dest="no_resume_opt", action="store_true")
    parser.add_argument("--resume-opt-force-lr", dest="resume_opt_force_lr", action="store_true")
    parser.add_argument("--anchor-ref-attn-kl-weight", dest="anchor_ref_attn_kl_weight", type=float)
    parser.add_argument("--anchor-ref-warmup-epochs", dest="anchor_ref_warmup_epochs", type=int)
    parser.add_argument("--anchor-ref-head-mode", dest="anchor_ref_head_mode", type=str)
    parser.add_argument("--ref-update", dest="ref_update", type=str)
    parser.add_argument("--ref-update-interval", dest="ref_update_interval", type=int)
    parser.add_argument("--ref-warmup-updates", dest="ref_warmup_updates", type=int)
    parser.add_argument("--ref-stop-updates", dest="ref_stop_updates", type=int)
    parser.add_argument("--teacher-attn-kl-weight", dest="teacher_attn_kl_weight", type=float)
    parser.add_argument("--teacher-attn-kl-warmup-epochs", dest="teacher_attn_kl_warmup_epochs", type=int)
    parser.add_argument("--ref-head-mode-epoch-overrides", dest="ref_head_mode_epoch_overrides", type=str)
    parser.add_argument("--teacher-attn-output-weight", dest="teacher_attn_output_weight", type=float)
    parser.add_argument("--teacher-attn-output-layers", dest="teacher_attn_output_layers", type=str)
    parser.add_argument("--teacher-attn-output-warmup-epochs", dest="teacher_attn_output_warmup_epochs", type=int)
    parser.add_argument("--teacher-attn-output-weight-epoch-overrides", dest="teacher_attn_output_weight_epoch_overrides", type=str)
    parser.add_argument("--teacher-feature-output-weight", dest="teacher_feature_output_weight", type=float)
    parser.add_argument("--teacher-feature-output-layers", dest="teacher_feature_output_layers", type=str)
    parser.add_argument("--teacher-feature-output-warmup-epochs", dest="teacher_feature_output_warmup_epochs", type=int)
    parser.add_argument("--teacher-feature-output-loss", dest="teacher_feature_output_loss", type=str)
    parser.add_argument("--bin-reg-weight", dest="bin_reg_weight", type=float)
    parser.add_argument("--bin-reg-variance-weight", dest="bin_reg_variance_weight", type=float)
    parser.add_argument("--bin-reg-layers", dest="bin_reg_layers", type=str)
    parser.add_argument("--bin-reg-attn-only", dest="bin_reg_attn_only", action="store_true")
    parser.add_argument("--bin-reg-start-update", dest="bin_reg_start_update", type=int)
    parser.add_argument("--bin-reg-end-update", dest="bin_reg_end_update", type=int)
    parser.add_argument("--selective-bin-anchor-weight", dest="selective_bin_anchor_weight", type=float)
    parser.add_argument("--selective-bin-anchor-layers", dest="selective_bin_anchor_layers", type=str)
    parser.add_argument("--selective-bin-anchor-capture-update", dest="selective_bin_anchor_capture_update", type=int)
    parser.add_argument("--selective-bin-anchor-end-update", dest="selective_bin_anchor_end_update", type=int)
    parser.add_argument("--selective-bin-anchor-margin", dest="selective_bin_anchor_margin", type=float)
    parser.add_argument("--candidate-bin-anchor-weight", dest="candidate_bin_anchor_weight", type=float)
    parser.add_argument("--candidate-bin-anchor-layers", dest="candidate_bin_anchor_layers", type=str)
    parser.add_argument("--candidate-bin-anchor-capture-update", dest="candidate_bin_anchor_capture_update", type=int)
    parser.add_argument("--candidate-bin-anchor-end-update", dest="candidate_bin_anchor_end_update", type=int)
    parser.add_argument("--candidate-bin-anchor-source-checkpoint", dest="candidate_bin_anchor_source_checkpoint", type=str)
    parser.add_argument("--weight-bin-telemetry-layers", dest="weight_bin_telemetry_layers", type=str)
    parser.add_argument("--weight-bin-telemetry-start-update", dest="weight_bin_telemetry_start_update", type=int)
    parser.add_argument("--weight-bin-telemetry-end-update", dest="weight_bin_telemetry_end_update", type=int)
    parser.add_argument("--weight-bin-telemetry-interval", dest="weight_bin_telemetry_interval", type=int)
    parser.add_argument("--weight-bin-telemetry-margin", dest="weight_bin_telemetry_margin", type=float)
    parser.add_argument("--act-bin-margin-weight", dest="act_bin_margin_weight", type=float)
    parser.add_argument("--act-bin-margin-layers", dest="act_bin_margin_layers", type=str)
    parser.add_argument("--act-bin-margin-quantizers", dest="act_bin_margin_quantizers", type=str)
    parser.add_argument("--act-bin-margin", dest="act_bin_margin", type=float)
    parser.add_argument("--act-bin-margin-max-elements", dest="act_bin_margin_max_elements", type=int)
    parser.add_argument("--teacher-confidence-kd-power", dest="teacher_confidence_kd_power", type=float)
    parser.add_argument("--teacher-soft-temperature", dest="teacher_soft_temperature", type=float)
    parser.add_argument("--quant-lr-multiplier", dest="quant_lr_multiplier", type=float)
    parser.add_argument("--quant-lr-multiplier-epoch-overrides", dest="quant_lr_multiplier_epoch_overrides", type=str)
    parser.add_argument("--quant-slow-state-decay", dest="quant_slow_state_decay", type=float)
    parser.add_argument("--quant-slow-state-sync-interval", dest="quant_slow_state_sync_interval", type=int)
    parser.add_argument("--quant-slow-state-pull", dest="quant_slow_state_pull", type=float)
    parser.add_argument("--quant-slow-state-policy", dest="quant_slow_state_policy", type=str)
    parser.add_argument("--quant-slow-state-observe-start-epoch", dest="quant_slow_state_observe_start_epoch", type=int)
    parser.add_argument("--quant-slow-state-start-epoch", dest="quant_slow_state_start_epoch", type=int)
    parser.add_argument("--act-scale-anchor-weight", dest="act_scale_anchor_weight", type=float)
    parser.add_argument("--act-scale-anchor-layers", dest="act_scale_anchor_layers", type=str)
    parser.add_argument("--act-scale-anchor-start-epoch", dest="act_scale_anchor_start_epoch", type=int)
    parser.add_argument("--variation-trust-weight", dest="variation_trust_weight", type=float)
    parser.add_argument("--variation-trust-layers", dest="variation_trust_layers", type=str)
    parser.add_argument("--variation-trust-late-layers", dest="variation_trust_late_layers", type=str)
    parser.add_argument("--variation-trust-late-multiplier", dest="variation_trust_late_multiplier", type=float)
    parser.add_argument("--variation-trust-early-layers", dest="variation_trust_early_layers", type=str)
    parser.add_argument("--variation-trust-early-multiplier", dest="variation_trust_early_multiplier", type=float)
    parser.add_argument("--variation-trust-softmax-multiplier", dest="variation_trust_softmax_multiplier", type=float)
    parser.add_argument("--variation-trust-move-v-multiplier", dest="variation_trust_move_v_multiplier", type=float)
    parser.add_argument("--variation-trust-proj-move-multiplier", dest="variation_trust_proj_move_multiplier", type=float)
    parser.add_argument("--variation-trust-start-update", dest="variation_trust_start_update", type=int)
    parser.add_argument("--aoq-explore-scale-ratio", dest="aoq_explore_scale_ratio", type=float)
    parser.add_argument("--aoq-explore-threshold-ratio", dest="aoq_explore_threshold_ratio", type=float)
    parser.add_argument("--aoq-explore-layers", dest="aoq_explore_layers", type=str)
    parser.add_argument("--aoq-explore-layer-ratios", dest="aoq_explore_layer_ratios", type=str)
    parser.add_argument("--aoq-explore-selective-margin", dest="aoq_explore_selective_margin", type=float)
    parser.add_argument("--aoq-explore-quality-mode", dest="aoq_explore_quality_mode", type=str)
    parser.add_argument("--aoq-explore-quality-layers", dest="aoq_explore_quality_layers", type=str)
    parser.add_argument("--aoq-explore-quality-min-frac", dest="aoq_explore_quality_min_frac", type=float)
    parser.add_argument("--aoq-explore-anchor-checkpoint", dest="aoq_explore_anchor_checkpoint", type=str)
    parser.add_argument("--aoq-explore-start-update", dest="aoq_explore_start_update", type=int)
    parser.add_argument("--aoq-explore-end-update", dest="aoq_explore_end_update", type=int)
    parser.add_argument("--aoq-explore-repeat-each-epoch", dest="aoq_explore_repeat_each_epoch", action="store_true")
    parser.add_argument("--aoq-explore-update-schedule", dest="aoq_explore_update_schedule", type=str)
    parser.add_argument("--delta-direction-anchor-weight", dest="delta_direction_anchor_weight", type=float)
    parser.add_argument("--delta-direction-anchor-base-checkpoint", dest="delta_direction_anchor_base_checkpoint", type=str)
    parser.add_argument("--delta-direction-anchor-target-checkpoint", dest="delta_direction_anchor_target_checkpoint", type=str)
    parser.add_argument("--delta-direction-anchor-params", dest="delta_direction_anchor_params", type=str)
    parser.add_argument("--delta-direction-anchor-start-update", dest="delta_direction_anchor_start_update", type=int)
    parser.add_argument("--pre-qat-act-percentile-calib-batches", dest="pre_qat_act_percentile_calib_batches", type=int)
    parser.add_argument("--pre-qat-act-percentile-calib-layers", dest="pre_qat_act_percentile_calib_layers", type=str)
    parser.add_argument("--pre-qat-act-percentile-calib-percentile", dest="pre_qat_act_percentile_calib_percentile", type=float)
    parser.add_argument("--pre-qat-act-percentile-calib-blend", dest="pre_qat_act_percentile_calib_blend", type=float)
    parser.add_argument("--pre-qat-act-mse-calib-batches", dest="pre_qat_act_mse_calib_batches", type=int)
    parser.add_argument("--pre-qat-act-mse-calib-layers", dest="pre_qat_act_mse_calib_layers", type=str)
    parser.add_argument("--pre-qat-act-mse-calib-quantizers", dest="pre_qat_act_mse_calib_quantizers", type=str)
    parser.add_argument("--pre-qat-act-mse-calib-grid", dest="pre_qat_act_mse_calib_grid", type=str)
    parser.add_argument("--pre-qat-act-mse-calib-blend", dest="pre_qat_act_mse_calib_blend", type=float)
    parser.add_argument("--pre-qat-recon-updates", dest="pre_qat_recon_updates", type=int)
    parser.add_argument("--pre-qat-recon-temperature", dest="pre_qat_recon_temperature", type=float)
    parser.add_argument("--pre-qat-feature-recon-updates", dest="pre_qat_feature_recon_updates", type=int)
    parser.add_argument("--pre-qat-feature-recon-layers", dest="pre_qat_feature_recon_layers", type=str)
    parser.add_argument("--pre-qat-feature-recon-policy", dest="pre_qat_feature_recon_policy", type=str)
    parser.add_argument("--pre-qat-feature-recon-confidence-power", dest="pre_qat_feature_recon_confidence_power", type=float)
    parser.add_argument("--pre-qat-feature-recon-weight-mode", dest="pre_qat_feature_recon_weight_mode", type=str)
    parser.add_argument("--pre-qat-feature-recon-qdrop-prob", dest="pre_qat_feature_recon_qdrop_prob", type=float)
    parser.add_argument("--pre-qat-feature-recon-qdrop-layers", dest="pre_qat_feature_recon_qdrop_layers", type=str)
    parser.add_argument("--pre-qat-feature-recon-anchor-kl-weight", dest="pre_qat_feature_recon_anchor_kl_weight", type=float)
    parser.add_argument("--pre-qat-feature-recon-anchor-kl-temperature", dest="pre_qat_feature_recon_anchor_kl_temperature", type=float)
    parser.add_argument("--post-epoch-feature-recon-updates", dest="post_epoch_feature_recon_updates", type=int)
    parser.add_argument("--pre-qat-seq-feature-recon-updates", dest="pre_qat_seq_feature_recon_updates", type=int)
    parser.add_argument("--pre-qat-seq-feature-recon-layers", dest="pre_qat_seq_feature_recon_layers", type=str)
    parser.add_argument("--ref-attn-loss", dest="ref_attn_loss", type=str)
    parser.add_argument("--ref-attn-kl-drop-prob", dest="ref_attn_kl_drop_prob", type=float)
    parser.add_argument("--ref-attn-kl-drop-scale", dest="ref_attn_kl_drop_scale", type=str2bool, nargs="?", const=True)
    parser.add_argument("--ref-attn-kl-clip", dest="ref_attn_kl_clip", type=float)
    parser.add_argument("--ref-logit-kl-weight", dest="ref_logit_kl_weight", type=float)
    parser.add_argument("--ref-logit-kl-temperature", dest="ref_logit_kl_temperature", type=float)
    parser.add_argument("--teacher-qk-rel-weight", dest="teacher_qk_rel_weight", type=float)
    parser.add_argument("--teacher-qk-rel-warmup-epochs", dest="teacher_qk_rel_warmup_epochs", type=int)
    parser.add_argument("--teacher-qkv-rel-weight", dest="teacher_qkv_rel_weight", type=float)
    parser.add_argument("--teacher-qkv-rel-warmup-epochs", dest="teacher_qkv_rel_warmup_epochs", type=int)
    parser.add_argument("--teacher-qkv-rel-layers", dest="teacher_qkv_rel_layers", type=str)
    parser.add_argument("--teacher-qkv-rel-components", dest="teacher_qkv_rel_components", type=str)
    parser.add_argument("--clean-start-target-loss-weight", dest="clean_start_target_loss_weight", type=float)
    parser.add_argument("--ref-attn-kl-weight-epoch-overrides", dest="ref_attn_kl_weight_epoch_overrides", type=str)
    parser.add_argument("--anchor-ref-attn-kl-weight-epoch-overrides", dest="anchor_ref_attn_kl_weight_epoch_overrides", type=str)
    parser.add_argument("--dynamic-sparse-prevstep-kl", dest="dynamic_sparse_prevstep_kl", action="store_true")
    parser.add_argument("--dynamic-kl-start-epoch", dest="dynamic_kl_start_epoch", type=int)
    parser.add_argument("--dynamic-kl-observe-until-epoch", dest="dynamic_kl_observe_until_epoch", type=int)
    parser.add_argument("--dynamic-kl-primary-heads", dest="dynamic_kl_primary_heads", type=str)
    parser.add_argument("--dynamic-kl-secondary-heads", dest="dynamic_kl_secondary_heads", type=str)
    parser.add_argument("--dynamic-kl-avoid-heads", dest="dynamic_kl_avoid_heads", type=str)
    parser.add_argument("--dynamic-kl-drop-threshold", dest="dynamic_kl_drop_threshold", type=float)
    parser.add_argument("--dynamic-kl-strong-drop-threshold", dest="dynamic_kl_strong_drop_threshold", type=float)
    parser.add_argument("--dynamic-kl-default-weight", dest="dynamic_kl_default_weight", type=float)
    parser.add_argument("--dynamic-kl-strong-weight", dest="dynamic_kl_strong_weight", type=float)
    parser.add_argument("--dynamic-kl-max-weight", dest="dynamic_kl_max_weight", type=float)
    parser.add_argument("--dynamic-kl-cooldown-epochs", dest="dynamic_kl_cooldown_epochs", type=int)
    parser.add_argument("--dynamic-kl-window-epochs", dest="dynamic_kl_window_epochs", type=int)
    parser.add_argument("--dynamic-kl-max-pulses-per-window", dest="dynamic_kl_max_pulses_per_window", type=int)
    parser.add_argument("--dynamic-kl-controller-tsv", dest="dynamic_kl_controller_tsv", type=str)
    parser.add_argument("--dynamic-kl-prior-source", dest="dynamic_kl_prior_source", type=str)
    parser.add_argument("--epoch-lr-overrides", dest="epoch_lr_overrides", type=str)
    parser.add_argument("--progressive-bit-schedule", dest="progressive_bit_schedule", type=str)
    parser.add_argument("--progressive-bit-rescale-lsq", dest="progressive_bit_rescale_lsq", action="store_true")
    parser.add_argument("--progressive-bit-recalibrate-epochs", dest="progressive_bit_recalibrate_epochs", type=str)
    parser.add_argument("--progressive-bit-recalibrate-batches", dest="progressive_bit_recalibrate_batches", type=int)
    parser.add_argument("--progressive-bit-transition-recon-updates", dest="progressive_bit_transition_recon_updates", type=int)
    parser.add_argument("--progressive-bit-transition-recon-epochs", dest="progressive_bit_transition_recon_epochs", type=str)
    parser.add_argument("--progressive-bit-transition-recon-layers", dest="progressive_bit_transition_recon_layers", type=str)
    parser.add_argument("--progressive-bit-transition-recon-policy", dest="progressive_bit_transition_recon_policy", type=str)
    parser.add_argument("--progressive-bit-transition-recon-confidence-power", dest="progressive_bit_transition_recon_confidence_power", type=float)
    parser.add_argument("--progressive-bit-transition-recon-weight-mode", dest="progressive_bit_transition_recon_weight_mode", type=str)
    parser.add_argument("--progressive-bit-transition-recon-qdrop-prob", dest="progressive_bit_transition_recon_qdrop_prob", type=float)
    parser.add_argument("--progressive-bit-transition-recon-qdrop-layers", dest="progressive_bit_transition_recon_qdrop_layers", type=str)
    parser.add_argument("--progressive-bit-transition-anchor-kl-weight", dest="progressive_bit_transition_anchor_kl_weight", type=float)
    parser.add_argument("--progressive-bit-transition-anchor-kl-temperature", dest="progressive_bit_transition_anchor_kl_temperature", type=float)
    parser.add_argument("--quant-only-start-epoch", dest="quant_only_start_epoch", type=int)
    parser.add_argument("--trainable-policy", dest="trainable_policy", type=str)
    parser.add_argument("--trainable-policy-freeze-act-except-layers", dest="trainable_policy_freeze_act_except_layers", type=str)
    parser.add_argument("--trainable-policy-update-overrides", dest="trainable_policy_update_overrides", type=str)
    parser.add_argument("--trainable-policy-update-mode", dest="trainable_policy_update_mode", type=str)
    parser.add_argument("--trainable-policy-grad-damp", dest="trainable_policy_grad_damp", type=float)
    parser.add_argument("--model-ema", dest="model_ema", action="store_true")
    parser.add_argument("--model-ema-decay", dest="model_ema_decay", type=float)
    namespace, _ = parser.parse_known_args(list(extra_args))
    overrides = {k: v for k, v in vars(namespace).items() if v is not None and v is not False}
    if "initial_checkpoint_alias" in overrides:
        overrides["initial_checkpoint"] = overrides.pop("initial_checkpoint_alias")
    return overrides


def parse_epoch_float_overrides(spec: object) -> Dict[int, float]:
    if spec is None or spec == "":
        return {}
    if isinstance(spec, dict):
        return {int(k): float(v) for k, v in spec.items()}
    parsed = {}
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"epoch override must be epoch:value, got {item!r}")
        epoch_text, value_text = item.split(":", 1)
        parsed[int(epoch_text.strip())] = float(value_text.strip())
    return parsed


def parse_layer_float_overrides(spec: object) -> Dict[str, float]:
    if spec is None or spec == "":
        return {}
    if isinstance(spec, dict):
        return {str(k).strip(): float(v) for k, v in spec.items() if str(k).strip()}
    parsed = {}
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"layer override must be module:ratio, got {item!r}")
        layer_text, value_text = item.split(":", 1)
        layer_name = layer_text.strip()
        if not layer_name:
            raise ValueError(f"layer override has empty module name: {item!r}")
        parsed[layer_name] = float(value_text.strip())
    return parsed


def parse_aoq_update_schedule(spec: object) -> List[Tuple[int, float, float, float]]:
    if spec is None or spec == "":
        return []
    if isinstance(spec, (list, tuple)):
        schedule = []
        for item in spec:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise ValueError(f"AOQ update schedule entries must be update:scale:threshold:margin, got {item!r}")
            update, scale, threshold, margin = item
            schedule.append((int(update), float(scale), float(threshold), float(margin)))
        return sorted(schedule, key=lambda x: x[0])
    schedule: List[Tuple[int, float, float, float]] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(f"AOQ update schedule must be update:scale:threshold:margin, got {item!r}")
        update_text, scale_text, threshold_text, margin_text = (part.strip() for part in parts)
        schedule.append((int(update_text), float(scale_text), float(threshold_text), float(margin_text)))
    return sorted(schedule, key=lambda x: x[0])


def parse_progressive_bit_schedule(spec: object) -> List[Tuple[int, int, int]]:
    if spec is None or spec == "":
        return []
    schedule: List[Tuple[int, int, int]] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"progressive bit schedule must be epoch:wbits:abits, got {item!r}")
        epoch, wbits, abits = (int(part.strip()) for part in parts)
        if epoch < 0 or wbits < 1 or abits < 1:
            raise ValueError(f"invalid progressive bit schedule item: {item!r}")
        schedule.append((epoch, wbits, abits))
    return sorted(schedule, key=lambda x: x[0])


def progressive_bits_for_epoch(schedule: Sequence[Tuple[int, int, int]], epoch: int, default_wbits: int, default_abits: int) -> Tuple[int, int]:
    wbits = int(default_wbits)
    abits = int(default_abits)
    for start_epoch, scheduled_wbits, scheduled_abits in schedule:
        if epoch >= start_epoch:
            wbits = int(scheduled_wbits)
            abits = int(scheduled_abits)
        else:
            break
    return wbits, abits


def parse_policy_update_overrides(spec: object) -> Dict[int, str]:
    if spec is None or spec == "":
        return {}
    if isinstance(spec, dict):
        return {int(k): str(v) for k, v in spec.items()}
    parsed = {}
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"policy update override must be update:policy, got {item!r}")
        update_text, policy_text = item.split(":", 1)
        policy = policy_text.strip()
        if policy not in {"all", "quant", "quant_in_layers", "params_in_layers", "params_in_layers_attn_plus_quant", "params_in_layers_freeze_highdrift_act", "params_in_layers_freeze_move_v_shift", "head_norm_quant", "head_norm_proj_quant", "head_norm_attn_quant", "attn_quant", "freeze_act_except_layers"}:
            raise ValueError(f"Unsupported trainable policy override: {policy}")
        parsed[int(update_text.strip())] = policy
    return parsed


def update_policy_value(overrides: Dict[int, str], update: int, default: str) -> str:
    active = default
    for update_idx in sorted(overrides):
        if update >= update_idx:
            active = overrides[update_idx]
        else:
            break
    return active


def epoch_float_value(overrides: Dict[int, float], epoch: int, default: float) -> float:
    return float(overrides.get(int(epoch), default))


def parse_epoch_string_overrides(spec: object) -> Dict[int, str]:
    if spec is None or spec == "":
        return {}
    if isinstance(spec, dict):
        return {int(k): str(v).strip() for k, v in spec.items() if str(v).strip()}
    parsed = {}
    for item in str(spec).split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"epoch string override must be epoch=value, got {item!r}")
        epoch_text, value_text = item.split("=", 1)
        value = value_text.strip()
        if not value:
            raise ValueError(f"epoch string override has empty value: {item!r}")
        parsed[int(epoch_text.strip())] = value
    return parsed


def epoch_string_value(overrides: Dict[int, str], epoch: int, default: str) -> str:
    return str(overrides.get(int(epoch), default))


def normalize_optional_string(value: object) -> object:
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "false", "0"}:
        return None
    return value


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
        "val_interval": 1,
        "subset_ratio": 1.0,
        "save_images": False,
        "amp": False,
        "amp_dtype": "bf16",
        "apex_amp": False,
        "native_amp": False,
        "channels_last": False,
        "pin_mem": False,
        "no_prefetcher": False,
        "sync_step_timing": False,
        "static_graph": False,
        "gradient_as_bucket_view": True,
        "compile": False,
        "compile_mode": "reduce-overhead",
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
        "setup_alpha_batches": 1,
        "post_resume_setup_alpha_batches": 0,
        "max_train_updates": 0,
        "save_step_checkpoints": False,
        "save_initial_step_checkpoint": False,
        "step_checkpoint_interval": 1,
        "step_checkpoint_warmup_updates": 0,
        "max_step_checkpoints_to_save": 0,
        "skip_validate": False,
        "eval_only": False,
        "post_load_alpha": False,
        "apply_q_attn_dropout": 0,
        "act_layer": "gelu",
        "kd_hard_and_soft": 1,
        "teacher_type": "swin",
        "pretrained_initialized": False,
        "qk_reparam": False,
        "qk_reparam_type": 0,
        "train_scheme": "baseline",
        "ref_update": "ema",
        "ref_update_interval": 1,
        "ref_momentum": 0.999,
        "ref_attn_kl_weight": 0.0,
        "ref_attn_kl_drop_prob": 1.0,
        "ref_attn_kl_drop_scale": False,
        "ref_attn_kl_clip": 0.0,
        "ref_attn_loss": "kl_ref",
        "ref_logit_kl_weight": 0.0,
        "ref_logit_kl_temperature": 2.0,
        "teacher_qk_rel_weight": 0.0,
        "teacher_qk_rel_warmup_epochs": 0,
        "teacher_qkv_rel_weight": 0.0,
        "teacher_qkv_rel_warmup_epochs": 0,
        "teacher_qkv_rel_layers": "all",
        "teacher_qkv_rel_components": "q,k,v",
        "clean_start_target_loss_weight": 0.0,
        "ref_head_mode": "all",
        "ref_warmup_epochs": 0,
        "ref_warmup_updates": 0,
        "ref_stop_updates": 0,
        "anchor_ref_attn_kl_weight": 0.0,
        "anchor_ref_warmup_epochs": 0,
        "anchor_ref_head_mode": "",
        "teacher_attn_kl_weight": 0.0,
        "teacher_attn_kl_warmup_epochs": 0,
        "teacher_attn_output_weight": 0.0,
        "teacher_attn_output_layers": "all",
        "teacher_attn_output_warmup_epochs": 0,
        "teacher_attn_output_weight_epoch_overrides": "",
        "teacher_feature_output_weight": 0.0,
        "teacher_feature_output_layers": "",
        "teacher_feature_output_warmup_epochs": 0,
        "teacher_feature_output_loss": "mse",
        "bin_reg_weight": 0.0,
        "bin_reg_variance_weight": 1.0,
        "bin_reg_layers": "",
        "bin_reg_attn_only": False,
        "bin_reg_start_update": 0,
        "bin_reg_end_update": 0,
        "selective_bin_anchor_weight": 0.0,
        "selective_bin_anchor_layers": "",
        "selective_bin_anchor_capture_update": 0,
        "selective_bin_anchor_end_update": 0,
        "selective_bin_anchor_margin": 0.05,
        "candidate_bin_anchor_weight": 0.0,
        "candidate_bin_anchor_layers": "",
        "candidate_bin_anchor_capture_update": 0,
        "candidate_bin_anchor_end_update": 0,
        "candidate_bin_anchor_source_checkpoint": "",
        "weight_bin_telemetry_layers": "",
        "weight_bin_telemetry_start_update": 0,
        "weight_bin_telemetry_end_update": 0,
        "weight_bin_telemetry_interval": 0,
        "weight_bin_telemetry_margin": 0.05,
        "act_bin_margin_weight": 0.0,
        "act_bin_margin_layers": "",
        "act_bin_margin_quantizers": "",
        "act_bin_margin": 0.08,
        "act_bin_margin_max_elements": 65536,
        "epoch1_acc_gate": 0.0,
        "teacher_confidence_kd_power": 0.0,
        "teacher_confidence_band_kd_weight": 0.0,
        "teacher_confidence_band_kd_low": 0.2,
        "teacher_confidence_band_kd_high": 0.6,
        "teacher_confidence_band_kd_temperature": 2.75,
        "ref_confidence_band_kd_weight": 0.0,
        "ref_confidence_band_kd_low": 0.2,
        "ref_confidence_band_kd_high": 0.6,
        "ref_confidence_band_kd_temperature": 2.75,
        "ref_confidence_band_kd_checkpoint": "",
        "local_ref_confidence_band_kd_weight": 0.0,
        "local_ref_confidence_band_kd_low": 0.2,
        "local_ref_confidence_band_kd_high": 0.4,
        "local_ref_confidence_band_kd_temperature": 2.75,
        "local_ref_confidence_band_kd_checkpoint": "",
        "class_protect_ref_kl_weight": 0.0,
        "class_protect_ref_kl_classes": "",
        "class_protect_ref_kl_temperature": 2.75,
        "class_protect_ref_kl_checkpoint": "",
        "teacher_soft_temperature": 1.0,
        "quant_lr_multiplier": 1.0,
        "quant_lr_multiplier_epoch_overrides": "",
        "quant_slow_state_decay": 0.0,
        "quant_slow_state_sync_interval": 0,
        "quant_slow_state_pull": 0.0,
        "quant_slow_state_policy": "all",
        "quant_slow_state_observe_start_epoch": 0,
        "quant_slow_state_start_epoch": 0,
        "act_scale_anchor_weight": 0.0,
        "act_scale_anchor_layers": "",
        "act_scale_anchor_start_epoch": 0,
        "variation_trust_weight": 0.0,
        "variation_trust_layers": "",
        "variation_trust_late_layers": "features.5.5,features.7.1",
        "variation_trust_late_multiplier": 0.25,
        "variation_trust_early_layers": "features.0.0,features.1.0,features.1.1",
        "variation_trust_early_multiplier": 2.0,
        "variation_trust_softmax_multiplier": 2.0,
        "variation_trust_move_v_multiplier": 1.5,
        "variation_trust_proj_move_multiplier": 1.25,
        "variation_trust_start_update": 0,
        "aoq_explore_scale_ratio": 1.0,
        "aoq_explore_threshold_ratio": 0.0,
        "aoq_explore_layers": "",
        "aoq_explore_layer_ratios": "",
        "aoq_explore_selective_margin": 0.0,
        "aoq_explore_quality_mode": "none",
        "aoq_explore_quality_layers": "",
        "aoq_explore_quality_start_update": 0,
        "aoq_explore_quality_min_frac": 0.0,
        "aoq_explore_anchor_checkpoint": "",
        "aoq_explore_start_update": 0,
        "aoq_explore_end_update": 0,
        "aoq_explore_repeat_each_epoch": False,
        "aoq_explore_update_schedule": "",
        "delta_direction_anchor_weight": 0.0,
        "delta_direction_anchor_base_checkpoint": "",
        "delta_direction_anchor_target_checkpoint": "",
        "delta_direction_anchor_params": "",
        "delta_direction_anchor_start_update": 0,
        "pre_qat_act_percentile_calib_batches": 0,
        "pre_qat_act_percentile_calib_layers": "",
        "pre_qat_act_percentile_calib_percentile": 0.999,
        "pre_qat_act_percentile_calib_blend": 1.0,
        "pre_qat_act_mse_calib_batches": 0,
        "pre_qat_act_mse_calib_layers": "",
        "pre_qat_act_mse_calib_quantizers": "",
        "pre_qat_act_mse_calib_grid": "0.75,1.0,11",
        "pre_qat_act_mse_calib_blend": 1.0,
        "pre_qat_recon_updates": 0,
        "pre_qat_recon_temperature": 1.0,
        "pre_qat_feature_recon_updates": 0,
        "pre_qat_feature_recon_layers": "",
        "pre_qat_feature_recon_policy": "quant",
        "pre_qat_feature_recon_confidence_power": 0.0,
        "pre_qat_feature_recon_weight_mode": "none",
        "pre_qat_feature_recon_qdrop_prob": 0.0,
        "pre_qat_feature_recon_qdrop_layers": "",
        "pre_qat_feature_recon_anchor_kl_weight": 0.0,
        "pre_qat_feature_recon_anchor_kl_temperature": 2.75,
        "post_epoch_feature_recon_updates": 0,
        "pre_qat_seq_feature_recon_updates": 0,
        "pre_qat_seq_feature_recon_layers": "",
        "pre_qat_seq_feature_recon_policy": "quant",
        "ref_attn_kl_weight_epoch_overrides": "",
        "anchor_ref_attn_kl_weight_epoch_overrides": "",
        "ref_head_mode_epoch_overrides": "",
        "dynamic_sparse_prevstep_kl": False,
        "dynamic_kl_start_epoch": 61,
        "dynamic_kl_observe_until_epoch": 60,
        "dynamic_kl_primary_heads": "8:4",
        "dynamic_kl_secondary_heads": "5:7,4:11,6:1,11:18",
        "dynamic_kl_avoid_heads": "6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16",
        "dynamic_kl_drop_threshold": 0.06,
        "dynamic_kl_strong_drop_threshold": 0.12,
        "dynamic_kl_default_weight": 1e-5,
        "dynamic_kl_strong_weight": 2e-5,
        "dynamic_kl_max_weight": 3e-5,
        "dynamic_kl_cooldown_epochs": 5,
        "dynamic_kl_window_epochs": 10,
        "dynamic_kl_max_pulses_per_window": 3,
        "dynamic_kl_controller_tsv": "",
        "dynamic_kl_prior_source": "offline_attn_relation_oscillation_20260710",
        "epoch_lr_overrides": "",
        "progressive_bit_schedule": "",
        "progressive_bit_rescale_lsq": False,
        "progressive_bit_recalibrate_epochs": "",
        "progressive_bit_recalibrate_batches": 1,
        "progressive_bit_transition_recon_updates": 0,
        "progressive_bit_transition_recon_epochs": "",
        "progressive_bit_transition_recon_layers": "",
        "progressive_bit_transition_recon_policy": "module_all",
        "progressive_bit_transition_recon_confidence_power": 0.0,
        "progressive_bit_transition_recon_weight_mode": "none",
        "progressive_bit_transition_recon_qdrop_prob": 0.0,
        "progressive_bit_transition_recon_qdrop_layers": "",
        "progressive_bit_transition_anchor_kl_weight": 0.0,
        "progressive_bit_transition_anchor_kl_temperature": 2.75,
        "quant_only_start_epoch": None,
        "trainable_policy": "all",
        "trainable_policy_freeze_act_except_layers": "",
        "trainable_policy_update_overrides": "",
        "trainable_policy_update_mode": "requires_grad",
        "trainable_policy_grad_damp": 0.1,
        "model_ema": False,
        "model_ema_decay": 0.9999,
        "initial_checkpoint": "",
        "resume": "",
        "no_resume_opt": False,
        "resume_opt_force_lr": False,
        "start_epoch": None,
        "opt": "adamw",
        "lr": 2e-4,
        "weight_decay": 0.0,
        "epochs": 300,
        "warmup_epochs": 0,
        "scheduler_epochs": None,
        "min_lr": 1e-5,
        "workers": 4,
        "batch_size": 32,
        "validation_batch_size_multiplier": 1,
        "grad_accum_steps": 1,
        "forward_micro_batch_size": 0,
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
    if args.scheduler_epochs is not None:
        defaults["scheduler_epochs"] = args.scheduler_epochs
    if args.min_lr is not None:
        defaults["min_lr"] = args.min_lr
    if args.no_resume_opt:
        defaults["no_resume_opt"] = True
    if args.resume_opt_force_lr:
        defaults["resume_opt_force_lr"] = True
    if args.start_epoch is not None:
        defaults["start_epoch"] = args.start_epoch
    if args.grad_accum_steps is not None:
        defaults["grad_accum_steps"] = args.grad_accum_steps
    if args.forward_micro_batch_size is not None:
        defaults["forward_micro_batch_size"] = args.forward_micro_batch_size
    if args.checkpoint_hist is not None:
        defaults["checkpoint_hist"] = args.checkpoint_hist
    if args.epoch_checkpoint_interval is not None:
        defaults["epoch_checkpoint_interval"] = args.epoch_checkpoint_interval
    if getattr(args, "val_interval", None) is not None:
        defaults["val_interval"] = args.val_interval
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
    if args.kd_hard_and_soft is None and args.use_kd and defaults.get("kd_hard_and_soft", 0) == 0:
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
    if args.ref_update_interval is not None:
        defaults["ref_update_interval"] = args.ref_update_interval
    if args.ref_momentum is not None:
        defaults["ref_momentum"] = args.ref_momentum
    if args.ref_attn_kl_weight is not None:
        defaults["ref_attn_kl_weight"] = args.ref_attn_kl_weight
    if args.ref_attn_kl_drop_prob is not None:
        defaults["ref_attn_kl_drop_prob"] = args.ref_attn_kl_drop_prob
    if args.ref_attn_kl_drop_scale is not None:
        defaults["ref_attn_kl_drop_scale"] = args.ref_attn_kl_drop_scale
    if args.ref_attn_kl_clip is not None:
        defaults["ref_attn_kl_clip"] = args.ref_attn_kl_clip
    if args.ref_attn_loss is not None:
        defaults["ref_attn_loss"] = args.ref_attn_loss
    if args.ref_logit_kl_weight is not None:
        defaults["ref_logit_kl_weight"] = args.ref_logit_kl_weight
    if args.ref_logit_kl_temperature is not None:
        defaults["ref_logit_kl_temperature"] = args.ref_logit_kl_temperature
    if args.teacher_qk_rel_weight is not None:
        defaults["teacher_qk_rel_weight"] = args.teacher_qk_rel_weight
    if args.teacher_qk_rel_warmup_epochs is not None:
        defaults["teacher_qk_rel_warmup_epochs"] = args.teacher_qk_rel_warmup_epochs
    if getattr(args, "teacher_qkv_rel_weight", None) is not None:
        defaults["teacher_qkv_rel_weight"] = args.teacher_qkv_rel_weight
    if getattr(args, "teacher_qkv_rel_warmup_epochs", None) is not None:
        defaults["teacher_qkv_rel_warmup_epochs"] = args.teacher_qkv_rel_warmup_epochs
    if getattr(args, "teacher_qkv_rel_layers", None) is not None:
        defaults["teacher_qkv_rel_layers"] = args.teacher_qkv_rel_layers
    if getattr(args, "teacher_qkv_rel_components", None) is not None:
        defaults["teacher_qkv_rel_components"] = args.teacher_qkv_rel_components
    if args.clean_start_target_loss_weight is not None:
        defaults["clean_start_target_loss_weight"] = args.clean_start_target_loss_weight
    if args.ref_head_mode is not None:
        defaults["ref_head_mode"] = args.ref_head_mode
    if getattr(args, "ref_head_mode_epoch_overrides", None) is not None:
        defaults["ref_head_mode_epoch_overrides"] = args.ref_head_mode_epoch_overrides
    if args.ref_warmup_epochs is not None:
        defaults["ref_warmup_epochs"] = args.ref_warmup_epochs
    if args.ref_warmup_updates is not None:
        defaults["ref_warmup_updates"] = args.ref_warmup_updates
    if args.ref_stop_updates is not None:
        defaults["ref_stop_updates"] = args.ref_stop_updates
    if args.anchor_ref_attn_kl_weight is not None:
        defaults["anchor_ref_attn_kl_weight"] = args.anchor_ref_attn_kl_weight
    if args.anchor_ref_warmup_epochs is not None:
        defaults["anchor_ref_warmup_epochs"] = args.anchor_ref_warmup_epochs
    if args.anchor_ref_head_mode is not None:
        defaults["anchor_ref_head_mode"] = args.anchor_ref_head_mode
    if args.teacher_attn_kl_weight is not None:
        defaults["teacher_attn_kl_weight"] = args.teacher_attn_kl_weight
    if args.teacher_attn_kl_warmup_epochs is not None:
        defaults["teacher_attn_kl_warmup_epochs"] = args.teacher_attn_kl_warmup_epochs
    if args.teacher_attn_output_weight is not None:
        defaults["teacher_attn_output_weight"] = args.teacher_attn_output_weight
    if args.teacher_attn_output_layers is not None:
        defaults["teacher_attn_output_layers"] = args.teacher_attn_output_layers
    if args.teacher_attn_output_warmup_epochs is not None:
        defaults["teacher_attn_output_warmup_epochs"] = args.teacher_attn_output_warmup_epochs
    if getattr(args, "teacher_attn_output_weight_epoch_overrides", None) is not None:
        defaults["teacher_attn_output_weight_epoch_overrides"] = args.teacher_attn_output_weight_epoch_overrides
    if args.teacher_feature_output_weight is not None:
        defaults["teacher_feature_output_weight"] = args.teacher_feature_output_weight
    if args.teacher_feature_output_layers is not None:
        defaults["teacher_feature_output_layers"] = args.teacher_feature_output_layers
    if args.teacher_feature_output_warmup_epochs is not None:
        defaults["teacher_feature_output_warmup_epochs"] = args.teacher_feature_output_warmup_epochs
    if args.teacher_feature_output_loss is not None:
        defaults["teacher_feature_output_loss"] = args.teacher_feature_output_loss
    if args.bin_reg_weight is not None:
        defaults["bin_reg_weight"] = args.bin_reg_weight
    if args.bin_reg_variance_weight is not None:
        defaults["bin_reg_variance_weight"] = args.bin_reg_variance_weight
    if getattr(args, "bin_reg_layers", None) is not None:
        defaults["bin_reg_layers"] = args.bin_reg_layers
    if getattr(args, "bin_reg_attn_only", False):
        defaults["bin_reg_attn_only"] = True
    if getattr(args, "bin_reg_start_update", None) is not None:
        defaults["bin_reg_start_update"] = args.bin_reg_start_update
    if getattr(args, "bin_reg_end_update", None) is not None:
        defaults["bin_reg_end_update"] = args.bin_reg_end_update
    if getattr(args, "selective_bin_anchor_weight", None) is not None:
        defaults["selective_bin_anchor_weight"] = args.selective_bin_anchor_weight
    if getattr(args, "selective_bin_anchor_layers", None) is not None:
        defaults["selective_bin_anchor_layers"] = args.selective_bin_anchor_layers
    if getattr(args, "selective_bin_anchor_capture_update", None) is not None:
        defaults["selective_bin_anchor_capture_update"] = args.selective_bin_anchor_capture_update
    if getattr(args, "selective_bin_anchor_end_update", None) is not None:
        defaults["selective_bin_anchor_end_update"] = args.selective_bin_anchor_end_update
    if getattr(args, "selective_bin_anchor_margin", None) is not None:
        defaults["selective_bin_anchor_margin"] = args.selective_bin_anchor_margin
    if getattr(args, "candidate_bin_anchor_weight", None) is not None:
        defaults["candidate_bin_anchor_weight"] = args.candidate_bin_anchor_weight
    if getattr(args, "candidate_bin_anchor_layers", None) is not None:
        defaults["candidate_bin_anchor_layers"] = args.candidate_bin_anchor_layers
    if getattr(args, "candidate_bin_anchor_capture_update", None) is not None:
        defaults["candidate_bin_anchor_capture_update"] = args.candidate_bin_anchor_capture_update
    if getattr(args, "candidate_bin_anchor_end_update", None) is not None:
        defaults["candidate_bin_anchor_end_update"] = args.candidate_bin_anchor_end_update
    if getattr(args, "candidate_bin_anchor_source_checkpoint", None) is not None:
        defaults["candidate_bin_anchor_source_checkpoint"] = normalize_path(args.candidate_bin_anchor_source_checkpoint) or ""
    if getattr(args, "weight_bin_telemetry_layers", None) is not None:
        defaults["weight_bin_telemetry_layers"] = args.weight_bin_telemetry_layers
    if getattr(args, "weight_bin_telemetry_start_update", None) is not None:
        defaults["weight_bin_telemetry_start_update"] = args.weight_bin_telemetry_start_update
    if getattr(args, "weight_bin_telemetry_end_update", None) is not None:
        defaults["weight_bin_telemetry_end_update"] = args.weight_bin_telemetry_end_update
    if getattr(args, "weight_bin_telemetry_interval", None) is not None:
        defaults["weight_bin_telemetry_interval"] = args.weight_bin_telemetry_interval
    if getattr(args, "weight_bin_telemetry_margin", None) is not None:
        defaults["weight_bin_telemetry_margin"] = args.weight_bin_telemetry_margin
    if getattr(args, "act_bin_margin_weight", None) is not None:
        defaults["act_bin_margin_weight"] = args.act_bin_margin_weight
    if getattr(args, "act_bin_margin_layers", None) is not None:
        defaults["act_bin_margin_layers"] = args.act_bin_margin_layers
    if getattr(args, "act_bin_margin_quantizers", None) is not None:
        defaults["act_bin_margin_quantizers"] = args.act_bin_margin_quantizers
    if getattr(args, "act_bin_margin", None) is not None:
        defaults["act_bin_margin"] = args.act_bin_margin
    if getattr(args, "act_bin_margin_max_elements", None) is not None:
        defaults["act_bin_margin_max_elements"] = args.act_bin_margin_max_elements
    if args.epoch1_acc_gate is not None:
        defaults["epoch1_acc_gate"] = args.epoch1_acc_gate
    if args.teacher_confidence_kd_power is not None:
        defaults["teacher_confidence_kd_power"] = args.teacher_confidence_kd_power
    if getattr(args, "teacher_confidence_band_kd_weight", None) is not None:
        defaults["teacher_confidence_band_kd_weight"] = args.teacher_confidence_band_kd_weight
    if getattr(args, "teacher_confidence_band_kd_low", None) is not None:
        defaults["teacher_confidence_band_kd_low"] = args.teacher_confidence_band_kd_low
    if getattr(args, "teacher_confidence_band_kd_high", None) is not None:
        defaults["teacher_confidence_band_kd_high"] = args.teacher_confidence_band_kd_high
    if getattr(args, "teacher_confidence_band_kd_temperature", None) is not None:
        defaults["teacher_confidence_band_kd_temperature"] = args.teacher_confidence_band_kd_temperature
    if getattr(args, "ref_confidence_band_kd_weight", None) is not None:
        defaults["ref_confidence_band_kd_weight"] = args.ref_confidence_band_kd_weight
    if getattr(args, "ref_confidence_band_kd_low", None) is not None:
        defaults["ref_confidence_band_kd_low"] = args.ref_confidence_band_kd_low
    if getattr(args, "ref_confidence_band_kd_high", None) is not None:
        defaults["ref_confidence_band_kd_high"] = args.ref_confidence_band_kd_high
    if getattr(args, "ref_confidence_band_kd_temperature", None) is not None:
        defaults["ref_confidence_band_kd_temperature"] = args.ref_confidence_band_kd_temperature
    if getattr(args, "ref_confidence_band_kd_checkpoint", None) is not None:
        defaults["ref_confidence_band_kd_checkpoint"] = normalize_path(args.ref_confidence_band_kd_checkpoint) or ""
    if getattr(args, "local_ref_confidence_band_kd_weight", None) is not None:
        defaults["local_ref_confidence_band_kd_weight"] = args.local_ref_confidence_band_kd_weight
    if getattr(args, "local_ref_confidence_band_kd_low", None) is not None:
        defaults["local_ref_confidence_band_kd_low"] = args.local_ref_confidence_band_kd_low
    if getattr(args, "local_ref_confidence_band_kd_high", None) is not None:
        defaults["local_ref_confidence_band_kd_high"] = args.local_ref_confidence_band_kd_high
    if getattr(args, "local_ref_confidence_band_kd_temperature", None) is not None:
        defaults["local_ref_confidence_band_kd_temperature"] = args.local_ref_confidence_band_kd_temperature
    if getattr(args, "local_ref_confidence_band_kd_checkpoint", None) is not None:
        defaults["local_ref_confidence_band_kd_checkpoint"] = normalize_path(args.local_ref_confidence_band_kd_checkpoint) or ""
    if getattr(args, "class_protect_ref_kl_weight", None) is not None:
        defaults["class_protect_ref_kl_weight"] = args.class_protect_ref_kl_weight
    if getattr(args, "class_protect_ref_kl_classes", None) is not None:
        defaults["class_protect_ref_kl_classes"] = args.class_protect_ref_kl_classes
    if getattr(args, "class_protect_ref_kl_temperature", None) is not None:
        defaults["class_protect_ref_kl_temperature"] = args.class_protect_ref_kl_temperature
    if getattr(args, "class_protect_ref_kl_checkpoint", None) is not None:
        defaults["class_protect_ref_kl_checkpoint"] = normalize_path(args.class_protect_ref_kl_checkpoint) or ""
    if args.teacher_soft_temperature is not None:
        defaults["teacher_soft_temperature"] = args.teacher_soft_temperature
    if args.quant_lr_multiplier is not None:
        defaults["quant_lr_multiplier"] = args.quant_lr_multiplier
    if getattr(args, "quant_lr_multiplier_epoch_overrides", None) is not None:
        defaults["quant_lr_multiplier_epoch_overrides"] = args.quant_lr_multiplier_epoch_overrides
    if getattr(args, "quant_slow_state_decay", None) is not None:
        defaults["quant_slow_state_decay"] = args.quant_slow_state_decay
    if getattr(args, "quant_slow_state_sync_interval", None) is not None:
        defaults["quant_slow_state_sync_interval"] = args.quant_slow_state_sync_interval
    if getattr(args, "quant_slow_state_pull", None) is not None:
        defaults["quant_slow_state_pull"] = args.quant_slow_state_pull
    if getattr(args, "quant_slow_state_policy", None) is not None:
        defaults["quant_slow_state_policy"] = args.quant_slow_state_policy
    if getattr(args, "quant_slow_state_observe_start_epoch", None) is not None:
        defaults["quant_slow_state_observe_start_epoch"] = args.quant_slow_state_observe_start_epoch
    if getattr(args, "quant_slow_state_start_epoch", None) is not None:
        defaults["quant_slow_state_start_epoch"] = args.quant_slow_state_start_epoch
    if getattr(args, "act_scale_anchor_weight", None) is not None:
        defaults["act_scale_anchor_weight"] = args.act_scale_anchor_weight
    if getattr(args, "act_scale_anchor_layers", None) is not None:
        defaults["act_scale_anchor_layers"] = args.act_scale_anchor_layers
    if getattr(args, "act_scale_anchor_start_epoch", None) is not None:
        defaults["act_scale_anchor_start_epoch"] = args.act_scale_anchor_start_epoch
    if getattr(args, "variation_trust_weight", None) is not None:
        defaults["variation_trust_weight"] = args.variation_trust_weight
    if getattr(args, "variation_trust_layers", None) is not None:
        defaults["variation_trust_layers"] = args.variation_trust_layers
    if getattr(args, "variation_trust_late_layers", None) is not None:
        defaults["variation_trust_late_layers"] = args.variation_trust_late_layers
    if getattr(args, "variation_trust_late_multiplier", None) is not None:
        defaults["variation_trust_late_multiplier"] = args.variation_trust_late_multiplier
    if getattr(args, "variation_trust_early_layers", None) is not None:
        defaults["variation_trust_early_layers"] = args.variation_trust_early_layers
    if getattr(args, "variation_trust_early_multiplier", None) is not None:
        defaults["variation_trust_early_multiplier"] = args.variation_trust_early_multiplier
    if getattr(args, "variation_trust_softmax_multiplier", None) is not None:
        defaults["variation_trust_softmax_multiplier"] = args.variation_trust_softmax_multiplier
    if getattr(args, "variation_trust_move_v_multiplier", None) is not None:
        defaults["variation_trust_move_v_multiplier"] = args.variation_trust_move_v_multiplier
    if getattr(args, "variation_trust_proj_move_multiplier", None) is not None:
        defaults["variation_trust_proj_move_multiplier"] = args.variation_trust_proj_move_multiplier
    if getattr(args, "variation_trust_start_update", None) is not None:
        defaults["variation_trust_start_update"] = args.variation_trust_start_update
    if getattr(args, "aoq_explore_scale_ratio", None) is not None:
        defaults["aoq_explore_scale_ratio"] = args.aoq_explore_scale_ratio
    if getattr(args, "aoq_explore_threshold_ratio", None) is not None:
        defaults["aoq_explore_threshold_ratio"] = args.aoq_explore_threshold_ratio
    if getattr(args, "aoq_explore_layers", None) is not None:
        defaults["aoq_explore_layers"] = args.aoq_explore_layers
    if getattr(args, "aoq_explore_layer_ratios", None) is not None:
        defaults["aoq_explore_layer_ratios"] = args.aoq_explore_layer_ratios
    if getattr(args, "aoq_explore_selective_margin", None) is not None:
        defaults["aoq_explore_selective_margin"] = args.aoq_explore_selective_margin
    if getattr(args, "aoq_explore_quality_mode", None) is not None:
        defaults["aoq_explore_quality_mode"] = args.aoq_explore_quality_mode
    if getattr(args, "aoq_explore_quality_layers", None) is not None:
        defaults["aoq_explore_quality_layers"] = args.aoq_explore_quality_layers
    if getattr(args, "aoq_explore_quality_start_update", None) is not None:
        defaults["aoq_explore_quality_start_update"] = args.aoq_explore_quality_start_update
    if getattr(args, "aoq_explore_quality_min_frac", None) is not None:
        defaults["aoq_explore_quality_min_frac"] = args.aoq_explore_quality_min_frac
    if getattr(args, "aoq_explore_anchor_checkpoint", None) is not None:
        defaults["aoq_explore_anchor_checkpoint"] = normalize_path(args.aoq_explore_anchor_checkpoint) or ""
    if getattr(args, "aoq_explore_start_update", None) is not None:
        defaults["aoq_explore_start_update"] = args.aoq_explore_start_update
    if getattr(args, "aoq_explore_end_update", None) is not None:
        defaults["aoq_explore_end_update"] = args.aoq_explore_end_update
    if getattr(args, "aoq_explore_repeat_each_epoch", False):
        defaults["aoq_explore_repeat_each_epoch"] = True
    if getattr(args, "aoq_explore_update_schedule", None) is not None:
        defaults["aoq_explore_update_schedule"] = args.aoq_explore_update_schedule
    if getattr(args, "delta_direction_anchor_weight", None) is not None:
        defaults["delta_direction_anchor_weight"] = args.delta_direction_anchor_weight
    if getattr(args, "delta_direction_anchor_base_checkpoint", None) is not None:
        defaults["delta_direction_anchor_base_checkpoint"] = normalize_path(args.delta_direction_anchor_base_checkpoint) or ""
    if getattr(args, "delta_direction_anchor_target_checkpoint", None) is not None:
        defaults["delta_direction_anchor_target_checkpoint"] = normalize_path(args.delta_direction_anchor_target_checkpoint) or ""
    if getattr(args, "delta_direction_anchor_params", None) is not None:
        defaults["delta_direction_anchor_params"] = args.delta_direction_anchor_params
    if getattr(args, "delta_direction_anchor_start_update", None) is not None:
        defaults["delta_direction_anchor_start_update"] = args.delta_direction_anchor_start_update
    if getattr(args, "pre_qat_act_percentile_calib_batches", None) is not None:
        defaults["pre_qat_act_percentile_calib_batches"] = args.pre_qat_act_percentile_calib_batches
    if getattr(args, "pre_qat_act_percentile_calib_layers", None) is not None:
        defaults["pre_qat_act_percentile_calib_layers"] = args.pre_qat_act_percentile_calib_layers
    if getattr(args, "pre_qat_act_percentile_calib_percentile", None) is not None:
        defaults["pre_qat_act_percentile_calib_percentile"] = args.pre_qat_act_percentile_calib_percentile
    if getattr(args, "pre_qat_act_percentile_calib_blend", None) is not None:
        defaults["pre_qat_act_percentile_calib_blend"] = args.pre_qat_act_percentile_calib_blend
    if getattr(args, "pre_qat_act_mse_calib_batches", None) is not None:
        defaults["pre_qat_act_mse_calib_batches"] = args.pre_qat_act_mse_calib_batches
    if getattr(args, "pre_qat_act_mse_calib_layers", None) is not None:
        defaults["pre_qat_act_mse_calib_layers"] = args.pre_qat_act_mse_calib_layers
    if getattr(args, "pre_qat_act_mse_calib_quantizers", None) is not None:
        defaults["pre_qat_act_mse_calib_quantizers"] = args.pre_qat_act_mse_calib_quantizers
    if getattr(args, "pre_qat_act_mse_calib_grid", None) is not None:
        defaults["pre_qat_act_mse_calib_grid"] = args.pre_qat_act_mse_calib_grid
    if getattr(args, "pre_qat_act_mse_calib_blend", None) is not None:
        defaults["pre_qat_act_mse_calib_blend"] = args.pre_qat_act_mse_calib_blend
    if getattr(args, "pre_qat_recon_updates", None) is not None:
        defaults["pre_qat_recon_updates"] = args.pre_qat_recon_updates
    if getattr(args, "pre_qat_recon_temperature", None) is not None:
        defaults["pre_qat_recon_temperature"] = args.pre_qat_recon_temperature
    if getattr(args, "pre_qat_feature_recon_updates", None) is not None:
        defaults["pre_qat_feature_recon_updates"] = args.pre_qat_feature_recon_updates
    if getattr(args, "pre_qat_feature_recon_layers", None) is not None:
        defaults["pre_qat_feature_recon_layers"] = args.pre_qat_feature_recon_layers
    if getattr(args, "pre_qat_feature_recon_policy", None) is not None:
        defaults["pre_qat_feature_recon_policy"] = args.pre_qat_feature_recon_policy
    if getattr(args, "pre_qat_feature_recon_confidence_power", None) is not None:
        defaults["pre_qat_feature_recon_confidence_power"] = args.pre_qat_feature_recon_confidence_power
    if getattr(args, "pre_qat_feature_recon_weight_mode", None) is not None:
        defaults["pre_qat_feature_recon_weight_mode"] = args.pre_qat_feature_recon_weight_mode
    if getattr(args, "pre_qat_feature_recon_qdrop_prob", None) is not None:
        defaults["pre_qat_feature_recon_qdrop_prob"] = args.pre_qat_feature_recon_qdrop_prob
    if getattr(args, "pre_qat_feature_recon_qdrop_layers", None) is not None:
        defaults["pre_qat_feature_recon_qdrop_layers"] = args.pre_qat_feature_recon_qdrop_layers
    if getattr(args, "pre_qat_feature_recon_anchor_kl_weight", None) is not None:
        defaults["pre_qat_feature_recon_anchor_kl_weight"] = args.pre_qat_feature_recon_anchor_kl_weight
    if getattr(args, "pre_qat_feature_recon_anchor_kl_temperature", None) is not None:
        defaults["pre_qat_feature_recon_anchor_kl_temperature"] = args.pre_qat_feature_recon_anchor_kl_temperature
    if getattr(args, "post_epoch_feature_recon_updates", None) is not None:
        defaults["post_epoch_feature_recon_updates"] = args.post_epoch_feature_recon_updates
    if getattr(args, "pre_qat_seq_feature_recon_updates", None) is not None:
        defaults["pre_qat_seq_feature_recon_updates"] = args.pre_qat_seq_feature_recon_updates
    if getattr(args, "pre_qat_seq_feature_recon_layers", None) is not None:
        defaults["pre_qat_seq_feature_recon_layers"] = args.pre_qat_seq_feature_recon_layers
    if getattr(args, "pre_qat_seq_feature_recon_policy", None) is not None:
        defaults["pre_qat_seq_feature_recon_policy"] = args.pre_qat_seq_feature_recon_policy
    if args.ref_attn_kl_weight_epoch_overrides is not None:
        defaults["ref_attn_kl_weight_epoch_overrides"] = args.ref_attn_kl_weight_epoch_overrides
    if args.anchor_ref_attn_kl_weight_epoch_overrides is not None:
        defaults["anchor_ref_attn_kl_weight_epoch_overrides"] = args.anchor_ref_attn_kl_weight_epoch_overrides
    if getattr(args, "dynamic_sparse_prevstep_kl", False):
        defaults["dynamic_sparse_prevstep_kl"] = True
    if getattr(args, "dynamic_kl_start_epoch", None) is not None:
        defaults["dynamic_kl_start_epoch"] = args.dynamic_kl_start_epoch
    if getattr(args, "dynamic_kl_observe_until_epoch", None) is not None:
        defaults["dynamic_kl_observe_until_epoch"] = args.dynamic_kl_observe_until_epoch
    if getattr(args, "dynamic_kl_primary_heads", None) is not None:
        defaults["dynamic_kl_primary_heads"] = args.dynamic_kl_primary_heads
    if getattr(args, "dynamic_kl_secondary_heads", None) is not None:
        defaults["dynamic_kl_secondary_heads"] = args.dynamic_kl_secondary_heads
    if getattr(args, "dynamic_kl_avoid_heads", None) is not None:
        defaults["dynamic_kl_avoid_heads"] = args.dynamic_kl_avoid_heads
    if getattr(args, "dynamic_kl_drop_threshold", None) is not None:
        defaults["dynamic_kl_drop_threshold"] = args.dynamic_kl_drop_threshold
    if getattr(args, "dynamic_kl_strong_drop_threshold", None) is not None:
        defaults["dynamic_kl_strong_drop_threshold"] = args.dynamic_kl_strong_drop_threshold
    if getattr(args, "dynamic_kl_default_weight", None) is not None:
        defaults["dynamic_kl_default_weight"] = args.dynamic_kl_default_weight
    if getattr(args, "dynamic_kl_strong_weight", None) is not None:
        defaults["dynamic_kl_strong_weight"] = args.dynamic_kl_strong_weight
    if getattr(args, "dynamic_kl_max_weight", None) is not None:
        defaults["dynamic_kl_max_weight"] = args.dynamic_kl_max_weight
    if getattr(args, "dynamic_kl_cooldown_epochs", None) is not None:
        defaults["dynamic_kl_cooldown_epochs"] = args.dynamic_kl_cooldown_epochs
    if getattr(args, "dynamic_kl_window_epochs", None) is not None:
        defaults["dynamic_kl_window_epochs"] = args.dynamic_kl_window_epochs
    if getattr(args, "dynamic_kl_max_pulses_per_window", None) is not None:
        defaults["dynamic_kl_max_pulses_per_window"] = args.dynamic_kl_max_pulses_per_window
    if getattr(args, "dynamic_kl_controller_tsv", None) is not None:
        defaults["dynamic_kl_controller_tsv"] = args.dynamic_kl_controller_tsv
    if getattr(args, "dynamic_kl_prior_source", None) is not None:
        defaults["dynamic_kl_prior_source"] = args.dynamic_kl_prior_source
    if args.epoch_lr_overrides is not None:
        defaults["epoch_lr_overrides"] = args.epoch_lr_overrides
    if getattr(args, "progressive_bit_schedule", None) is not None:
        defaults["progressive_bit_schedule"] = args.progressive_bit_schedule
    if getattr(args, "progressive_bit_rescale_lsq", False):
        defaults["progressive_bit_rescale_lsq"] = True
    if getattr(args, "progressive_bit_recalibrate_epochs", None) is not None:
        defaults["progressive_bit_recalibrate_epochs"] = args.progressive_bit_recalibrate_epochs
    if getattr(args, "progressive_bit_recalibrate_batches", None) is not None:
        defaults["progressive_bit_recalibrate_batches"] = args.progressive_bit_recalibrate_batches
    if getattr(args, "progressive_bit_transition_recon_updates", None) is not None:
        defaults["progressive_bit_transition_recon_updates"] = args.progressive_bit_transition_recon_updates
    if getattr(args, "progressive_bit_transition_recon_epochs", None) is not None:
        defaults["progressive_bit_transition_recon_epochs"] = args.progressive_bit_transition_recon_epochs
    if getattr(args, "progressive_bit_transition_recon_layers", None) is not None:
        defaults["progressive_bit_transition_recon_layers"] = args.progressive_bit_transition_recon_layers
    if getattr(args, "progressive_bit_transition_recon_policy", None) is not None:
        defaults["progressive_bit_transition_recon_policy"] = args.progressive_bit_transition_recon_policy
    if getattr(args, "progressive_bit_transition_recon_confidence_power", None) is not None:
        defaults["progressive_bit_transition_recon_confidence_power"] = args.progressive_bit_transition_recon_confidence_power
    if getattr(args, "progressive_bit_transition_recon_weight_mode", None) is not None:
        defaults["progressive_bit_transition_recon_weight_mode"] = args.progressive_bit_transition_recon_weight_mode
    if getattr(args, "progressive_bit_transition_recon_qdrop_prob", None) is not None:
        defaults["progressive_bit_transition_recon_qdrop_prob"] = args.progressive_bit_transition_recon_qdrop_prob
    if getattr(args, "progressive_bit_transition_recon_qdrop_layers", None) is not None:
        defaults["progressive_bit_transition_recon_qdrop_layers"] = args.progressive_bit_transition_recon_qdrop_layers
    if getattr(args, "progressive_bit_transition_anchor_kl_weight", None) is not None:
        defaults["progressive_bit_transition_anchor_kl_weight"] = args.progressive_bit_transition_anchor_kl_weight
    if getattr(args, "progressive_bit_transition_anchor_kl_temperature", None) is not None:
        defaults["progressive_bit_transition_anchor_kl_temperature"] = args.progressive_bit_transition_anchor_kl_temperature
    if args.setup_alpha_batches is not None:
        defaults["setup_alpha_batches"] = args.setup_alpha_batches
    if args.post_resume_setup_alpha_batches is not None:
        defaults["post_resume_setup_alpha_batches"] = args.post_resume_setup_alpha_batches
    if args.quant_only_start_epoch is not None:
        defaults["quant_only_start_epoch"] = args.quant_only_start_epoch
    if args.trainable_policy is not None:
        defaults["trainable_policy"] = args.trainable_policy
    if getattr(args, "trainable_policy_freeze_act_except_layers", None) is not None:
        defaults["trainable_policy_freeze_act_except_layers"] = args.trainable_policy_freeze_act_except_layers
    if args.trainable_policy_update_overrides is not None:
        defaults["trainable_policy_update_overrides"] = args.trainable_policy_update_overrides
    if args.trainable_policy_update_mode is not None:
        defaults["trainable_policy_update_mode"] = args.trainable_policy_update_mode
    if getattr(args, "trainable_policy_grad_damp", None) is not None:
        defaults["trainable_policy_grad_damp"] = args.trainable_policy_grad_damp
    if args.model_ema:
        defaults["model_ema"] = True
    if args.model_ema_decay is not None:
        defaults["model_ema_decay"] = args.model_ema_decay
    if args.amp:
        defaults["amp"] = True
        defaults["native_amp"] = True
    if args.amp_dtype is not None:
        defaults["amp_dtype"] = args.amp_dtype
    if args.channels_last:
        defaults["channels_last"] = True
    if args.compile:
        defaults["compile"] = True
    if args.compile_mode is not None:
        defaults["compile_mode"] = args.compile_mode

    defaults.update(build_ofq_runtime_overrides(args.extra_arg))
    defaults["aa"] = normalize_optional_string(defaults.get("aa"))
    defaults["train_interpolation"] = normalize_optional_string(defaults.get("train_interpolation"))
    defaults["world_size"] = int(defaults["world_size"])
    defaults["lr"] = float(defaults["lr"])
    defaults["warmup_lr"] = float(defaults["warmup_lr"])
    defaults["min_lr"] = float(defaults["min_lr"])
    defaults["weight_decay"] = float(defaults["weight_decay"])
    defaults["epochs"] = int(defaults["epochs"])
    defaults["batch_size"] = int(defaults["batch_size"])
    defaults["workers"] = int(defaults["workers"])
    defaults["grad_accum_steps"] = int(defaults["grad_accum_steps"])
    defaults["forward_micro_batch_size"] = int(defaults.get("forward_micro_batch_size", 0) or 0)
    defaults["warmup_epochs"] = int(defaults["warmup_epochs"])
    if defaults.get("scheduler_epochs") is not None:
        defaults["scheduler_epochs"] = int(defaults["scheduler_epochs"])
    defaults["num_classes"] = int(defaults["num_classes"])
    defaults["epoch_checkpoint_interval"] = int(defaults["epoch_checkpoint_interval"])
    defaults["val_interval"] = int(defaults.get("val_interval", 1))
    defaults["setup_alpha_batches"] = int(defaults.get("setup_alpha_batches", 1))
    defaults["post_resume_setup_alpha_batches"] = int(defaults.get("post_resume_setup_alpha_batches", 0))
    defaults["subset_ratio"] = float(defaults["subset_ratio"])
    defaults["ref_warmup_epochs"] = int(defaults["ref_warmup_epochs"])
    defaults["ref_warmup_updates"] = int(defaults.get("ref_warmup_updates", 0))
    defaults["ref_stop_updates"] = int(defaults.get("ref_stop_updates", 0))
    if defaults.get("start_epoch") is not None:
        defaults["start_epoch"] = int(defaults["start_epoch"])
    defaults["ref_momentum"] = float(defaults["ref_momentum"])
    defaults["ref_update_interval"] = max(1, int(defaults.get("ref_update_interval", 1)))
    defaults["ref_attn_kl_weight"] = float(defaults["ref_attn_kl_weight"])
    defaults["ref_attn_kl_drop_prob"] = float(defaults.get("ref_attn_kl_drop_prob", 1.0))
    if not 0.0 <= defaults["ref_attn_kl_drop_prob"] <= 1.0:
        raise ValueError(f"ref_attn_kl_drop_prob must be in [0, 1], got {defaults['ref_attn_kl_drop_prob']}")
    defaults["ref_attn_kl_drop_scale"] = bool(defaults.get("ref_attn_kl_drop_scale", False))
    defaults["ref_attn_kl_clip"] = float(defaults.get("ref_attn_kl_clip", 0.0))
    defaults["ref_attn_loss"] = str(defaults.get("ref_attn_loss", "kl_ref"))
    defaults["ref_logit_kl_weight"] = float(defaults["ref_logit_kl_weight"])
    defaults["ref_logit_kl_temperature"] = float(defaults["ref_logit_kl_temperature"])
    defaults["teacher_qk_rel_weight"] = float(defaults["teacher_qk_rel_weight"])
    defaults["teacher_qk_rel_warmup_epochs"] = int(defaults["teacher_qk_rel_warmup_epochs"])
    defaults["teacher_qkv_rel_weight"] = float(defaults.get("teacher_qkv_rel_weight", 0.0))
    defaults["teacher_qkv_rel_warmup_epochs"] = int(defaults.get("teacher_qkv_rel_warmup_epochs", 0))
    defaults["teacher_qkv_rel_layers"] = str(defaults.get("teacher_qkv_rel_layers", "all"))
    defaults["teacher_qkv_rel_components"] = str(defaults.get("teacher_qkv_rel_components", "q,k,v"))
    defaults["clean_start_target_loss_weight"] = float(defaults.get("clean_start_target_loss_weight", 0.0))
    defaults["anchor_ref_attn_kl_weight"] = float(defaults["anchor_ref_attn_kl_weight"])
    defaults["anchor_ref_warmup_epochs"] = int(defaults["anchor_ref_warmup_epochs"])
    defaults["anchor_ref_head_mode"] = str(defaults.get("anchor_ref_head_mode") or "")
    defaults["teacher_attn_kl_weight"] = float(defaults["teacher_attn_kl_weight"])
    defaults["teacher_attn_kl_warmup_epochs"] = int(defaults["teacher_attn_kl_warmup_epochs"])
    defaults["teacher_attn_output_weight"] = float(defaults["teacher_attn_output_weight"])
    defaults["teacher_attn_output_layers"] = str(defaults.get("teacher_attn_output_layers", "all"))
    defaults["teacher_attn_output_warmup_epochs"] = int(defaults["teacher_attn_output_warmup_epochs"])
    defaults["teacher_attn_output_weight_epoch_overrides"] = parse_epoch_float_overrides(defaults.get("teacher_attn_output_weight_epoch_overrides"))
    defaults["teacher_feature_output_weight"] = float(defaults["teacher_feature_output_weight"])
    defaults["teacher_feature_output_layers"] = str(defaults.get("teacher_feature_output_layers", ""))
    defaults["teacher_feature_output_warmup_epochs"] = int(defaults["teacher_feature_output_warmup_epochs"])
    defaults["teacher_feature_output_loss"] = str(defaults.get("teacher_feature_output_loss", "mse"))
    defaults["bin_reg_weight"] = float(defaults.get("bin_reg_weight", 0.0))
    defaults["bin_reg_variance_weight"] = float(defaults.get("bin_reg_variance_weight", 1.0))
    defaults["bin_reg_layers"] = str(defaults.get("bin_reg_layers", ""))
    defaults["bin_reg_attn_only"] = bool(defaults.get("bin_reg_attn_only", False))
    defaults["bin_reg_start_update"] = int(defaults.get("bin_reg_start_update", 0))
    defaults["bin_reg_end_update"] = int(defaults.get("bin_reg_end_update", 0))
    if defaults["bin_reg_start_update"] < 0 or defaults["bin_reg_end_update"] < 0:
        raise ValueError("bin_reg_start_update and bin_reg_end_update must be non-negative")
    defaults["selective_bin_anchor_weight"] = float(defaults.get("selective_bin_anchor_weight", 0.0))
    defaults["selective_bin_anchor_layers"] = str(defaults.get("selective_bin_anchor_layers", ""))
    defaults["selective_bin_anchor_capture_update"] = int(defaults.get("selective_bin_anchor_capture_update", 0))
    defaults["selective_bin_anchor_end_update"] = int(defaults.get("selective_bin_anchor_end_update", 0))
    defaults["selective_bin_anchor_margin"] = float(defaults.get("selective_bin_anchor_margin", 0.05))
    if defaults["selective_bin_anchor_capture_update"] < 0 or defaults["selective_bin_anchor_end_update"] < 0:
        raise ValueError("selective_bin_anchor_capture_update and selective_bin_anchor_end_update must be non-negative")
    if not 0.0 <= defaults["selective_bin_anchor_margin"] <= 0.5:
        raise ValueError(
            f"selective_bin_anchor_margin must be in [0, 0.5], got {defaults['selective_bin_anchor_margin']}"
        )
    defaults["candidate_bin_anchor_weight"] = float(defaults.get("candidate_bin_anchor_weight", 0.0))
    defaults["candidate_bin_anchor_layers"] = str(defaults.get("candidate_bin_anchor_layers", ""))
    defaults["candidate_bin_anchor_capture_update"] = int(defaults.get("candidate_bin_anchor_capture_update", 0))
    defaults["candidate_bin_anchor_end_update"] = int(defaults.get("candidate_bin_anchor_end_update", 0))
    defaults["candidate_bin_anchor_source_checkpoint"] = str(defaults.get("candidate_bin_anchor_source_checkpoint", "") or "")
    if defaults["candidate_bin_anchor_capture_update"] < 0 or defaults["candidate_bin_anchor_end_update"] < 0:
        raise ValueError("candidate_bin_anchor_capture_update and candidate_bin_anchor_end_update must be non-negative")
    if defaults["candidate_bin_anchor_weight"] > 0 and not defaults["candidate_bin_anchor_source_checkpoint"]:
        raise ValueError("candidate_bin_anchor_source_checkpoint is required when candidate_bin_anchor_weight > 0")
    defaults["weight_bin_telemetry_layers"] = str(defaults.get("weight_bin_telemetry_layers", ""))
    defaults["weight_bin_telemetry_start_update"] = int(defaults.get("weight_bin_telemetry_start_update", 0))
    defaults["weight_bin_telemetry_end_update"] = int(defaults.get("weight_bin_telemetry_end_update", 0))
    defaults["weight_bin_telemetry_interval"] = int(defaults.get("weight_bin_telemetry_interval", 0))
    defaults["weight_bin_telemetry_margin"] = float(defaults.get("weight_bin_telemetry_margin", 0.05))
    if defaults["weight_bin_telemetry_start_update"] < 0 or defaults["weight_bin_telemetry_end_update"] < 0:
        raise ValueError("weight_bin_telemetry_start_update and weight_bin_telemetry_end_update must be non-negative")
    if defaults["weight_bin_telemetry_interval"] < 0:
        raise ValueError("weight_bin_telemetry_interval must be non-negative")
    if not 0.0 <= defaults["weight_bin_telemetry_margin"] <= 0.5:
        raise ValueError(
            f"weight_bin_telemetry_margin must be in [0, 0.5], got {defaults['weight_bin_telemetry_margin']}"
        )
    defaults["act_bin_margin_weight"] = float(defaults.get("act_bin_margin_weight", 0.0))
    defaults["act_bin_margin_layers"] = str(defaults.get("act_bin_margin_layers", ""))
    defaults["act_bin_margin_quantizers"] = str(defaults.get("act_bin_margin_quantizers", ""))
    defaults["act_bin_margin"] = float(defaults.get("act_bin_margin", 0.08))
    if not 0.0 <= defaults["act_bin_margin"] <= 0.5:
        raise ValueError(f"act_bin_margin must be in [0, 0.5], got {defaults['act_bin_margin']}")
    defaults["act_bin_margin_max_elements"] = int(defaults.get("act_bin_margin_max_elements", 65536))
    if defaults["act_bin_margin_max_elements"] <= 0:
        raise ValueError(f"act_bin_margin_max_elements must be positive, got {defaults['act_bin_margin_max_elements']}")
    defaults["aoq_explore_quality_start_update"] = int(defaults.get("aoq_explore_quality_start_update", 0))
    if defaults["aoq_explore_quality_start_update"] < 0:
        raise ValueError("aoq_explore_quality_start_update must be non-negative")
    defaults["epoch1_acc_gate"] = float(defaults.get("epoch1_acc_gate", 0.0))
    defaults["teacher_confidence_kd_power"] = float(defaults.get("teacher_confidence_kd_power", 0.0))
    defaults["teacher_confidence_band_kd_weight"] = float(defaults.get("teacher_confidence_band_kd_weight", 0.0))
    defaults["teacher_confidence_band_kd_low"] = float(defaults.get("teacher_confidence_band_kd_low", 0.2))
    defaults["teacher_confidence_band_kd_high"] = float(defaults.get("teacher_confidence_band_kd_high", 0.6))
    defaults["teacher_confidence_band_kd_temperature"] = float(defaults.get("teacher_confidence_band_kd_temperature", 2.75))
    if not 0.0 <= defaults["teacher_confidence_band_kd_low"] < defaults["teacher_confidence_band_kd_high"] <= 1.0:
        raise ValueError(
            "teacher confidence band must satisfy 0 <= low < high <= 1, got "
            f"{defaults['teacher_confidence_band_kd_low']}..{defaults['teacher_confidence_band_kd_high']}"
        )
    defaults["ref_confidence_band_kd_weight"] = float(defaults.get("ref_confidence_band_kd_weight", 0.0))
    defaults["ref_confidence_band_kd_low"] = float(defaults.get("ref_confidence_band_kd_low", 0.2))
    defaults["ref_confidence_band_kd_high"] = float(defaults.get("ref_confidence_band_kd_high", 0.6))
    defaults["ref_confidence_band_kd_temperature"] = float(defaults.get("ref_confidence_band_kd_temperature", 2.75))
    defaults["ref_confidence_band_kd_checkpoint"] = str(defaults.get("ref_confidence_band_kd_checkpoint", "") or "")
    if not 0.0 <= defaults["ref_confidence_band_kd_low"] < defaults["ref_confidence_band_kd_high"] <= 1.0:
        raise ValueError(
            "reference confidence band must satisfy 0 <= low < high <= 1, got "
            f"{defaults['ref_confidence_band_kd_low']}..{defaults['ref_confidence_band_kd_high']}"
        )
    defaults["local_ref_confidence_band_kd_weight"] = float(defaults.get("local_ref_confidence_band_kd_weight", 0.0))
    defaults["local_ref_confidence_band_kd_low"] = float(defaults.get("local_ref_confidence_band_kd_low", 0.2))
    defaults["local_ref_confidence_band_kd_high"] = float(defaults.get("local_ref_confidence_band_kd_high", 0.4))
    defaults["local_ref_confidence_band_kd_temperature"] = float(defaults.get("local_ref_confidence_band_kd_temperature", 2.75))
    defaults["local_ref_confidence_band_kd_checkpoint"] = str(defaults.get("local_ref_confidence_band_kd_checkpoint", "") or "")
    if not 0.0 <= defaults["local_ref_confidence_band_kd_low"] < defaults["local_ref_confidence_band_kd_high"] <= 1.0:
        raise ValueError(
            "local reference confidence band must satisfy 0 <= low < high <= 1, got "
            f"{defaults['local_ref_confidence_band_kd_low']}..{defaults['local_ref_confidence_band_kd_high']}"
        )
    defaults["class_protect_ref_kl_weight"] = float(defaults.get("class_protect_ref_kl_weight", 0.0))
    defaults["class_protect_ref_kl_classes"] = parse_int_set(defaults.get("class_protect_ref_kl_classes", ""))
    defaults["class_protect_ref_kl_temperature"] = float(defaults.get("class_protect_ref_kl_temperature", 2.75))
    defaults["class_protect_ref_kl_checkpoint"] = str(defaults.get("class_protect_ref_kl_checkpoint", "") or "")
    defaults["teacher_soft_temperature"] = float(defaults.get("teacher_soft_temperature", 1.0))
    defaults["quant_lr_multiplier"] = float(defaults.get("quant_lr_multiplier", 1.0))
    defaults["quant_lr_multiplier_epoch_overrides"] = parse_epoch_float_overrides(defaults.get("quant_lr_multiplier_epoch_overrides"))
    defaults["quant_slow_state_decay"] = float(defaults.get("quant_slow_state_decay", 0.0))
    defaults["quant_slow_state_sync_interval"] = int(defaults.get("quant_slow_state_sync_interval", 0))
    defaults["quant_slow_state_pull"] = float(defaults.get("quant_slow_state_pull", 0.0))
    defaults["quant_slow_state_policy"] = str(defaults.get("quant_slow_state_policy", "all"))
    if defaults["quant_slow_state_policy"] not in {"all", "activation"}:
        raise ValueError(f"Unsupported quant_slow_state_policy: {defaults['quant_slow_state_policy']}")
    defaults["quant_slow_state_observe_start_epoch"] = int(defaults.get("quant_slow_state_observe_start_epoch", 0))
    defaults["quant_slow_state_start_epoch"] = int(defaults.get("quant_slow_state_start_epoch", 0))
    defaults["act_scale_anchor_weight"] = float(defaults.get("act_scale_anchor_weight", 0.0))
    defaults["act_scale_anchor_layers"] = str(defaults.get("act_scale_anchor_layers", ""))
    defaults["act_scale_anchor_start_epoch"] = int(defaults.get("act_scale_anchor_start_epoch", 0))
    defaults["variation_trust_weight"] = float(defaults.get("variation_trust_weight", 0.0))
    defaults["variation_trust_layers"] = str(defaults.get("variation_trust_layers", ""))
    defaults["variation_trust_late_layers"] = str(defaults.get("variation_trust_late_layers", ""))
    defaults["variation_trust_late_multiplier"] = float(defaults.get("variation_trust_late_multiplier", 1.0))
    defaults["variation_trust_early_layers"] = str(defaults.get("variation_trust_early_layers", ""))
    defaults["variation_trust_early_multiplier"] = float(defaults.get("variation_trust_early_multiplier", 1.0))
    defaults["variation_trust_softmax_multiplier"] = float(defaults.get("variation_trust_softmax_multiplier", 1.0))
    defaults["variation_trust_move_v_multiplier"] = float(defaults.get("variation_trust_move_v_multiplier", 1.0))
    defaults["variation_trust_proj_move_multiplier"] = float(defaults.get("variation_trust_proj_move_multiplier", 1.0))
    defaults["variation_trust_start_update"] = int(defaults.get("variation_trust_start_update", 0))
    defaults["aoq_explore_scale_ratio"] = float(defaults.get("aoq_explore_scale_ratio", 1.0))
    defaults["aoq_explore_threshold_ratio"] = float(defaults.get("aoq_explore_threshold_ratio", 0.0))
    defaults["aoq_explore_layers"] = str(defaults.get("aoq_explore_layers", ""))
    defaults["aoq_explore_layer_ratios"] = str(defaults.get("aoq_explore_layer_ratios", ""))
    defaults["aoq_explore_selective_margin"] = float(defaults.get("aoq_explore_selective_margin", 0.0))
    defaults["aoq_explore_quality_mode"] = str(defaults.get("aoq_explore_quality_mode", "none") or "none")
    defaults["aoq_explore_quality_layers"] = str(defaults.get("aoq_explore_quality_layers", "") or "")
    defaults["aoq_explore_quality_min_frac"] = float(defaults.get("aoq_explore_quality_min_frac", 0.0))
    defaults["aoq_explore_anchor_checkpoint"] = str(defaults.get("aoq_explore_anchor_checkpoint", "") or "")
    defaults["aoq_explore_start_update"] = int(defaults.get("aoq_explore_start_update", 0))
    defaults["aoq_explore_end_update"] = int(defaults.get("aoq_explore_end_update", 0))
    defaults["aoq_explore_repeat_each_epoch"] = bool(defaults.get("aoq_explore_repeat_each_epoch", False))
    defaults["aoq_explore_update_schedule"] = parse_aoq_update_schedule(defaults.get("aoq_explore_update_schedule", ""))
    if defaults["aoq_explore_scale_ratio"] <= 0:
        raise ValueError(f"aoq_explore_scale_ratio must be positive, got {defaults['aoq_explore_scale_ratio']}")
    if defaults["aoq_explore_threshold_ratio"] < 0:
        raise ValueError(
            f"aoq_explore_threshold_ratio must be non-negative, got {defaults['aoq_explore_threshold_ratio']}"
        )
    if not 0.0 <= defaults["aoq_explore_selective_margin"] <= 0.5:
        raise ValueError(
            f"aoq_explore_selective_margin must be in [0, 0.5], got {defaults['aoq_explore_selective_margin']}"
        )
    if defaults["aoq_explore_quality_mode"] not in {"none", "grad_cross", "anchor_unmoved", "anchor_moved", "history_oscillating", "recent_oscillating"}:
        raise ValueError(
            "aoq_explore_quality_mode must be one of none, grad_cross, anchor_unmoved, anchor_moved, "
            "history_oscillating, recent_oscillating; "
            f"got {defaults['aoq_explore_quality_mode']}"
        )
    if defaults["aoq_explore_quality_mode"] in {"anchor_unmoved", "anchor_moved"} and not defaults["aoq_explore_anchor_checkpoint"]:
        raise ValueError(f"aoq_explore_anchor_checkpoint is required when aoq_explore_quality_mode={defaults['aoq_explore_quality_mode']}")
    if not 0.0 <= defaults["aoq_explore_quality_min_frac"] <= 1.0:
        raise ValueError(
            f"aoq_explore_quality_min_frac must be in [0, 1], got {defaults['aoq_explore_quality_min_frac']}"
        )
    for layer_name, layer_ratio in parse_layer_float_overrides(defaults["aoq_explore_layer_ratios"]).items():
        if layer_ratio <= 0:
            raise ValueError(f"aoq_explore_layer_ratios values must be positive, got {layer_name}:{layer_ratio}")
    if defaults["aoq_explore_start_update"] < 0 or defaults["aoq_explore_end_update"] < 0:
        raise ValueError("aoq_explore_start_update and aoq_explore_end_update must be non-negative")
    for update, scale_ratio, threshold_ratio, selective_margin in defaults["aoq_explore_update_schedule"]:
        if update < 0:
            raise ValueError(f"aoq_explore_update_schedule update must be non-negative, got {update}")
        if scale_ratio <= 0:
            raise ValueError(f"aoq_explore_update_schedule scale ratio must be positive, got {scale_ratio}")
        if threshold_ratio < 0:
            raise ValueError(f"aoq_explore_update_schedule threshold ratio must be non-negative, got {threshold_ratio}")
        if not 0.0 <= selective_margin <= 0.5:
            raise ValueError(
                f"aoq_explore_update_schedule margin must be in [0, 0.5], got {selective_margin}"
            )
    defaults["delta_direction_anchor_weight"] = float(defaults.get("delta_direction_anchor_weight", 0.0))
    defaults["delta_direction_anchor_base_checkpoint"] = str(defaults.get("delta_direction_anchor_base_checkpoint", "") or "")
    defaults["delta_direction_anchor_target_checkpoint"] = str(defaults.get("delta_direction_anchor_target_checkpoint", "") or "")
    defaults["delta_direction_anchor_params"] = str(defaults.get("delta_direction_anchor_params", "") or "")
    defaults["delta_direction_anchor_start_update"] = int(defaults.get("delta_direction_anchor_start_update", 0))
    defaults["pre_qat_act_percentile_calib_batches"] = int(defaults.get("pre_qat_act_percentile_calib_batches", 0))
    defaults["pre_qat_act_percentile_calib_layers"] = str(defaults.get("pre_qat_act_percentile_calib_layers", ""))
    defaults["pre_qat_act_percentile_calib_percentile"] = float(defaults.get("pre_qat_act_percentile_calib_percentile", 0.999))
    if not 0.0 < defaults["pre_qat_act_percentile_calib_percentile"] <= 1.0:
        raise ValueError(
            "pre_qat_act_percentile_calib_percentile must be in (0, 1], "
            f"got {defaults['pre_qat_act_percentile_calib_percentile']}"
        )
    defaults["pre_qat_act_percentile_calib_blend"] = float(defaults.get("pre_qat_act_percentile_calib_blend", 1.0))
    if not 0.0 <= defaults["pre_qat_act_percentile_calib_blend"] <= 1.0:
        raise ValueError(
            "pre_qat_act_percentile_calib_blend must be in [0, 1], "
            f"got {defaults['pre_qat_act_percentile_calib_blend']}"
        )
    defaults["pre_qat_act_mse_calib_batches"] = int(defaults.get("pre_qat_act_mse_calib_batches", 0))
    defaults["pre_qat_act_mse_calib_layers"] = str(defaults.get("pre_qat_act_mse_calib_layers", ""))
    defaults["pre_qat_act_mse_calib_quantizers"] = str(defaults.get("pre_qat_act_mse_calib_quantizers", ""))
    defaults["pre_qat_act_mse_calib_grid"] = str(defaults.get("pre_qat_act_mse_calib_grid", "0.75,1.0,11"))
    defaults["pre_qat_act_mse_calib_blend"] = float(defaults.get("pre_qat_act_mse_calib_blend", 1.0))
    if not 0.0 <= defaults["pre_qat_act_mse_calib_blend"] <= 1.0:
        raise ValueError(
            "pre_qat_act_mse_calib_blend must be in [0, 1], "
            f"got {defaults['pre_qat_act_mse_calib_blend']}"
        )
    defaults["pre_qat_recon_updates"] = int(defaults.get("pre_qat_recon_updates", 0))
    defaults["pre_qat_recon_temperature"] = float(defaults.get("pre_qat_recon_temperature", 1.0))
    defaults["pre_qat_feature_recon_updates"] = int(defaults.get("pre_qat_feature_recon_updates", 0))
    defaults["pre_qat_feature_recon_layers"] = str(defaults.get("pre_qat_feature_recon_layers", ""))
    defaults["pre_qat_feature_recon_policy"] = str(defaults.get("pre_qat_feature_recon_policy", "quant"))
    if defaults["pre_qat_feature_recon_policy"] not in {"quant", "module_all"}:
        raise ValueError(f"Unsupported pre_qat_feature_recon_policy: {defaults['pre_qat_feature_recon_policy']}")
    defaults["pre_qat_feature_recon_confidence_power"] = float(defaults.get("pre_qat_feature_recon_confidence_power", 0.0))
    defaults["pre_qat_feature_recon_weight_mode"] = str(defaults.get("pre_qat_feature_recon_weight_mode", "none"))
    if defaults["pre_qat_feature_recon_weight_mode"] not in {"none", "confidence", "disagreement"}:
        raise ValueError(f"Unsupported pre_qat_feature_recon_weight_mode: {defaults['pre_qat_feature_recon_weight_mode']}")
    defaults["pre_qat_feature_recon_qdrop_prob"] = float(defaults.get("pre_qat_feature_recon_qdrop_prob", 0.0))
    if not 0.0 <= defaults["pre_qat_feature_recon_qdrop_prob"] <= 1.0:
        raise ValueError(f"pre_qat_feature_recon_qdrop_prob must be in [0, 1], got {defaults['pre_qat_feature_recon_qdrop_prob']}")
    defaults["pre_qat_feature_recon_qdrop_layers"] = str(defaults.get("pre_qat_feature_recon_qdrop_layers", ""))
    defaults["pre_qat_feature_recon_anchor_kl_weight"] = float(defaults.get("pre_qat_feature_recon_anchor_kl_weight", 0.0))
    defaults["pre_qat_feature_recon_anchor_kl_temperature"] = float(defaults.get("pre_qat_feature_recon_anchor_kl_temperature", 2.75))
    defaults["post_epoch_feature_recon_updates"] = int(defaults.get("post_epoch_feature_recon_updates", 0))
    defaults["pre_qat_seq_feature_recon_updates"] = int(defaults.get("pre_qat_seq_feature_recon_updates", 0))
    defaults["pre_qat_seq_feature_recon_layers"] = str(defaults.get("pre_qat_seq_feature_recon_layers", ""))
    defaults["pre_qat_seq_feature_recon_policy"] = str(defaults.get("pre_qat_seq_feature_recon_policy", "quant"))
    if defaults["pre_qat_seq_feature_recon_policy"] not in {"quant", "module_all"}:
        raise ValueError(f"Unsupported pre_qat_seq_feature_recon_policy: {defaults['pre_qat_seq_feature_recon_policy']}")
    if defaults.get("quant_only_start_epoch") is not None:
        defaults["quant_only_start_epoch"] = int(defaults["quant_only_start_epoch"])
    defaults["trainable_policy"] = str(defaults.get("trainable_policy") or "all")
    defaults["trainable_policy_freeze_act_except_layers"] = str(defaults.get("trainable_policy_freeze_act_except_layers", ""))
    defaults["trainable_policy_update_overrides"] = parse_policy_update_overrides(defaults.get("trainable_policy_update_overrides"))
    defaults["trainable_policy_update_mode"] = str(defaults.get("trainable_policy_update_mode") or "requires_grad")
    if defaults["trainable_policy_update_mode"] not in {"requires_grad", "grad_mask", "grad_damp"}:
        raise ValueError(f"Unsupported trainable_policy_update_mode: {defaults['trainable_policy_update_mode']}")
    defaults["trainable_policy_grad_damp"] = float(defaults.get("trainable_policy_grad_damp", 0.1))
    if not 0.0 <= defaults["trainable_policy_grad_damp"] <= 1.0:
        raise ValueError(f"trainable_policy_grad_damp must be in [0, 1], got {defaults['trainable_policy_grad_damp']}")
    defaults["ref_attn_kl_weight_epoch_overrides"] = parse_epoch_float_overrides(defaults.get("ref_attn_kl_weight_epoch_overrides"))
    defaults["anchor_ref_attn_kl_weight_epoch_overrides"] = parse_epoch_float_overrides(defaults.get("anchor_ref_attn_kl_weight_epoch_overrides"))
    defaults["ref_head_mode_epoch_overrides"] = parse_epoch_string_overrides(defaults.get("ref_head_mode_epoch_overrides"))
    defaults["dynamic_sparse_prevstep_kl"] = bool(defaults.get("dynamic_sparse_prevstep_kl", False))
    defaults["dynamic_kl_start_epoch"] = int(defaults.get("dynamic_kl_start_epoch", 61))
    defaults["dynamic_kl_observe_until_epoch"] = int(defaults.get("dynamic_kl_observe_until_epoch", 60))
    defaults["dynamic_kl_primary_heads"] = str(defaults.get("dynamic_kl_primary_heads", "8:4") or "")
    defaults["dynamic_kl_secondary_heads"] = str(defaults.get("dynamic_kl_secondary_heads", "5:7,4:11,6:1,11:18") or "")
    defaults["dynamic_kl_avoid_heads"] = str(defaults.get("dynamic_kl_avoid_heads", "6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16") or "")
    defaults["dynamic_kl_drop_threshold"] = float(defaults.get("dynamic_kl_drop_threshold", 0.06))
    defaults["dynamic_kl_strong_drop_threshold"] = float(defaults.get("dynamic_kl_strong_drop_threshold", 0.12))
    defaults["dynamic_kl_default_weight"] = float(defaults.get("dynamic_kl_default_weight", 1e-5))
    defaults["dynamic_kl_strong_weight"] = float(defaults.get("dynamic_kl_strong_weight", 2e-5))
    defaults["dynamic_kl_max_weight"] = float(defaults.get("dynamic_kl_max_weight", 3e-5))
    defaults["dynamic_kl_cooldown_epochs"] = int(defaults.get("dynamic_kl_cooldown_epochs", 5))
    defaults["dynamic_kl_window_epochs"] = int(defaults.get("dynamic_kl_window_epochs", 10))
    defaults["dynamic_kl_max_pulses_per_window"] = int(defaults.get("dynamic_kl_max_pulses_per_window", 3))
    defaults["dynamic_kl_controller_tsv"] = str(defaults.get("dynamic_kl_controller_tsv", "") or "")
    defaults["dynamic_kl_prior_source"] = str(defaults.get("dynamic_kl_prior_source", "offline_attn_relation") or "offline_attn_relation")
    if defaults["dynamic_sparse_prevstep_kl"]:
        if defaults["train_scheme"] != "ema_ref_attn_kl":
            raise ValueError("dynamic sparse prev-step KL requires train_scheme=ema_ref_attn_kl")
        if defaults["ref_update"] != "prev_step":
            raise ValueError("dynamic sparse prev-step KL requires ref_update=prev_step")
        if defaults["ref_attn_kl_weight_epoch_overrides"]:
            raise ValueError("dynamic sparse prev-step KL must not be combined with ref_attn_kl_weight_epoch_overrides")
        if defaults["dynamic_kl_start_epoch"] <= defaults["dynamic_kl_observe_until_epoch"]:
            raise ValueError(
                "dynamic_kl_start_epoch must be greater than dynamic_kl_observe_until_epoch, got "
                f"{defaults['dynamic_kl_start_epoch']} <= {defaults['dynamic_kl_observe_until_epoch']}"
            )
        if defaults["dynamic_kl_drop_threshold"] < 0 or defaults["dynamic_kl_strong_drop_threshold"] < 0:
            raise ValueError("dynamic KL drop thresholds must be non-negative")
        if defaults["dynamic_kl_cooldown_epochs"] < 0:
            raise ValueError("dynamic_kl_cooldown_epochs must be non-negative")
        if defaults["dynamic_kl_window_epochs"] <= 0:
            raise ValueError("dynamic_kl_window_epochs must be positive")
        if defaults["dynamic_kl_max_pulses_per_window"] < 0:
            raise ValueError("dynamic_kl_max_pulses_per_window must be non-negative")
    defaults["epoch_lr_overrides"] = parse_epoch_float_overrides(defaults.get("epoch_lr_overrides"))
    defaults["progressive_bit_schedule"] = parse_progressive_bit_schedule(defaults.get("progressive_bit_schedule"))
    defaults["progressive_bit_rescale_lsq"] = bool(defaults.get("progressive_bit_rescale_lsq", False))
    defaults["progressive_bit_recalibrate_epochs"] = {
        int(item.strip()) for item in str(defaults.get("progressive_bit_recalibrate_epochs") or "").split(",") if item.strip()
    }
    defaults["progressive_bit_recalibrate_batches"] = int(defaults.get("progressive_bit_recalibrate_batches", 1))
    defaults["progressive_bit_transition_recon_updates"] = int(defaults.get("progressive_bit_transition_recon_updates", 0))
    defaults["progressive_bit_transition_recon_epochs"] = {
        int(item.strip()) for item in str(defaults.get("progressive_bit_transition_recon_epochs") or "").split(",") if item.strip()
    }
    defaults["progressive_bit_transition_recon_layers"] = str(defaults.get("progressive_bit_transition_recon_layers", ""))
    defaults["progressive_bit_transition_recon_policy"] = str(defaults.get("progressive_bit_transition_recon_policy", "module_all"))
    if defaults["progressive_bit_transition_recon_policy"] not in {"quant", "module_all"}:
        raise ValueError(
            "Unsupported progressive_bit_transition_recon_policy: "
            f"{defaults['progressive_bit_transition_recon_policy']}"
        )
    defaults["progressive_bit_transition_recon_confidence_power"] = float(
        defaults.get("progressive_bit_transition_recon_confidence_power", 0.0)
    )
    defaults["progressive_bit_transition_recon_weight_mode"] = str(defaults.get("progressive_bit_transition_recon_weight_mode", "none"))
    if defaults["progressive_bit_transition_recon_weight_mode"] not in {"none", "confidence", "disagreement"}:
        raise ValueError(
            "Unsupported progressive_bit_transition_recon_weight_mode: "
            f"{defaults['progressive_bit_transition_recon_weight_mode']}"
        )
    defaults["progressive_bit_transition_recon_qdrop_prob"] = float(defaults.get("progressive_bit_transition_recon_qdrop_prob", 0.0))
    if not 0.0 <= defaults["progressive_bit_transition_recon_qdrop_prob"] <= 1.0:
        raise ValueError(
            "progressive_bit_transition_recon_qdrop_prob must be in [0, 1], "
            f"got {defaults['progressive_bit_transition_recon_qdrop_prob']}"
        )
    defaults["progressive_bit_transition_recon_qdrop_layers"] = str(defaults.get("progressive_bit_transition_recon_qdrop_layers", ""))
    defaults["progressive_bit_transition_anchor_kl_weight"] = float(defaults.get("progressive_bit_transition_anchor_kl_weight", 0.0))
    defaults["progressive_bit_transition_anchor_kl_temperature"] = float(
        defaults.get("progressive_bit_transition_anchor_kl_temperature", 2.75)
    )
    defaults["model_ema"] = bool(defaults.get("model_ema", False))
    defaults["model_ema_decay"] = float(defaults.get("model_ema_decay", 0.9999))
    defaults["no_prefetcher"] = bool(defaults.get("no_prefetcher", False))
    defaults["prefetcher"] = not defaults["no_prefetcher"]
    defaults["sync_step_timing"] = bool(defaults.get("sync_step_timing", False))
    defaults["static_graph"] = bool(defaults.get("static_graph", False))
    defaults["gradient_as_bucket_view"] = bool(defaults.get("gradient_as_bucket_view", True))
    defaults["compile"] = bool(defaults.get("compile", False))
    defaults["compile_mode"] = str(defaults.get("compile_mode") or "reduce-overhead")
    defaults["amp"] = bool(defaults.get("amp", False))
    defaults["native_amp"] = bool(defaults.get("native_amp", False) or defaults["amp"])
    defaults["amp_dtype"] = str(defaults.get("amp_dtype") or "bf16").lower()
    if defaults["amp_dtype"] not in {"bf16", "fp16"}:
        raise ValueError(f"OFQ amp_dtype must be bf16 or fp16, got {defaults['amp_dtype']!r}")
    defaults["channels_last"] = bool(defaults.get("channels_last", False))
    defaults["teacher"] = defaults["teacher"] or defaults["model"]
    defaults["experiment"] = defaults["experiment"] or safe_model_name(defaults["model"])
    defaults["opt_betas"] = tuple(defaults.get("opt_betas") or (0.9, 0.999))
    defaults["drop_path"] = 0.0 if defaults.get("drop_path") is None else defaults.get("drop_path")

    defaults["single_process_grad_accum_steps"] = defaults["grad_accum_steps"]
    defaults["single_process_effective_batch_size"] = defaults["batch_size"] * defaults["single_process_grad_accum_steps"]
    if defaults["world_size"] > 1:
        defaults["grad_accum_steps"] = max(1, int(math.ceil(defaults["single_process_grad_accum_steps"] / defaults["world_size"])))
    if defaults.get("forward_micro_batch_size", 0) > 0 and defaults["forward_micro_batch_size"] < defaults["batch_size"]:
        defaults["grad_accum_steps"] = max(defaults["grad_accum_steps"], int(math.ceil(defaults["batch_size"] / defaults["forward_micro_batch_size"])))
    defaults["effective_batch_size"] = defaults["batch_size"] * defaults["world_size"]

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
    qqkkvv = runtime_args.kd_hard_and_soft in {2, 3} or runtime_args.teacher_qk_rel_weight > 0 or runtime_args.teacher_qkv_rel_weight > 0
    if runtime_args.teacher_type == "deit":
        teacher = create_model(runtime_args.teacher, num_classes=runtime_args.num_classes, drop_rate=runtime_args.drop, pretrained=runtime_args.teacher_pretrained, qqkkvv=qqkkvv)
    else:
        teacher = create_model(runtime_args.teacher, num_classes=runtime_args.num_classes, drop_path=runtime_args.drop_path, pretrained=runtime_args.teacher_pretrained, qqkkvv=qqkkvv)
    if runtime_args.quant_teacher:
        teacher = get_ofq_qat_model(teacher, runtime_args)
    if runtime_args.teacher_checkpoint:
        load_checkpoint(teacher, runtime_args.teacher_checkpoint, strict=True)
    if (
        runtime_args.teacher_attn_kl_weight > 0
        or runtime_args.teacher_qk_rel_weight > 0
        or str(getattr(runtime_args, "ref_head_mode", "")).startswith("dynamic_teacher_agree_top")
    ):
        set_attention_mode(teacher, collect_attention=True, qqkkvv=qqkkvv)
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def save_step_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, runtime_args: SimpleNamespace, output_dir: Path, step_tag: str, epoch: Optional[int] = None, batch_idx: Optional[int] = None, loss_scaler=None, metric=None, lr_scheduler: Optional["WarmupCosineScheduler"] = None, model_ema: Optional[nn.Module] = None) -> str:
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
        "rng_state": _rng_state_dict(),
    }
    if loss_scaler is not None:
        save_state[loss_scaler.state_dict_key] = loss_scaler.state_dict()
    if metric is not None:
        save_state["metric"] = metric
    if lr_scheduler is not None:
        save_state["lr_scheduler"] = lr_scheduler.state_dict()
    if model_ema is not None:
        save_state["state_dict_ema"] = get_state_dict(model_ema)
    save_path = step_dir / f"{step_tag}.pth.tar"
    tmp_path = step_dir / f".{save_path.name}.tmp"
    torch.save(save_state, tmp_path)
    os.replace(tmp_path, save_path)
    return str(save_path)


def _safe_torch_load(path: str, map_location="cpu"):
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


def _rng_state_dict() -> Dict[str, object]:
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _load_rng_state(state: Optional[Dict[str, object]]) -> None:
    if not state:
        return
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)


def _prune_epoch_checkpoints(output_dir: Path, runtime_args: SimpleNamespace, suffix: str = "") -> None:
    keep = int(getattr(runtime_args, "checkpoint_hist", 0) or 0)
    if keep <= 0:
        return
    pattern = f"checkpoint-*{suffix}.pth.tar"
    checkpoints = [path for path in output_dir.glob(pattern) if path.is_file()]
    if not suffix:
        checkpoints = [path for path in checkpoints if ".ema." not in path.name]
    checkpoints = sorted(checkpoints, key=lambda path: path.stat().st_mtime)
    excess = len(checkpoints) - keep
    for path in checkpoints[: max(0, excess)]:
        try:
            path.unlink()
        except OSError:
            pass


def save_epoch_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, runtime_args: SimpleNamespace, output_dir: Path, epoch: int, loss_scaler=None, suffix: str = "", lr_scheduler: Optional["WarmupCosineScheduler"] = None, model_ema: Optional[nn.Module] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch + 1,
        "arch": runtime_args.model,
        "state_dict": get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "version": 2,
        "args": runtime_args,
        "rng_state": _rng_state_dict(),
    }
    if loss_scaler is not None:
        state[loss_scaler.state_dict_key] = loss_scaler.state_dict()
    if lr_scheduler is not None:
        state["lr_scheduler"] = lr_scheduler.state_dict()
    if model_ema is not None:
        state["state_dict_ema"] = get_state_dict(model_ema)
    checkpoint_path = output_dir / f"checkpoint-{epoch + 1}{suffix}.pth.tar"
    last_path = output_dir / ("last.pth.tar" if not suffix else f"last{suffix}.pth.tar")
    tmp_path = output_dir / f".{checkpoint_path.name}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, checkpoint_path)
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
    _prune_epoch_checkpoints(output_dir, runtime_args, suffix=suffix)


def strict_resume_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_scaler=None,
    lr_scheduler: Optional["WarmupCosineScheduler"] = None,
    model_ema: Optional[nn.Module] = None,
    restore_rng: bool = True,
    log_info: bool = True,
) -> int:
    checkpoint = _safe_torch_load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing and unexpected:
        candidates = []
        if all(isinstance(key, str) for key in state_dict.keys()):
            if any(key.startswith("module.") for key in state_dict.keys()):
                stripped = {key[len("module.") :] if key.startswith("module.") else key: value for key, value in state_dict.items()}
                candidates.append(("strip-module-prefix", stripped))
            else:
                prefixed = {f"module.{key}": value for key, value in state_dict.items()}
                candidates.append(("add-module-prefix", prefixed))
        best_missing, best_unexpected = missing, unexpected
        best_state_dict = state_dict
        best_label = ""
        for label, candidate_state in candidates:
            candidate_missing, candidate_unexpected = model.load_state_dict(candidate_state, strict=False)
            if len(candidate_missing) + len(candidate_unexpected) < len(best_missing) + len(best_unexpected):
                best_missing, best_unexpected = candidate_missing, candidate_unexpected
                best_state_dict = candidate_state
                best_label = label
        if best_state_dict is not state_dict:
            state_dict = best_state_dict
            missing, unexpected = best_missing, best_unexpected
            if log_info:
                print(f"Strict resume: adjusted state_dict keys via {best_label}.")
    if log_info:
        print(
            f"Strict resume: loaded model from {checkpoint_path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    if not isinstance(checkpoint, dict):
        return 0

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if log_info:
            opt_states = len(optimizer.state_dict().get("state", {}))
            print(f"Strict resume: restored optimizer state entries={opt_states}")
    elif optimizer is not None and log_info:
        print("Strict resume: optimizer state missing in checkpoint.")

    if loss_scaler is not None and getattr(loss_scaler, "state_dict_key", None) in checkpoint:
        loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])
        if log_info:
            print(f"Strict resume: restored loss scaler key={loss_scaler.state_dict_key}")
    elif loss_scaler is not None and log_info:
        print("Strict resume: loss scaler state missing in checkpoint.")

    if lr_scheduler is not None and "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if log_info:
            print(f"Strict resume: restored lr scheduler state={lr_scheduler.state_dict()}")
    elif lr_scheduler is not None and log_info:
        print("Strict resume: lr scheduler state missing; scheduler will be positioned by epoch.")

    if model_ema is not None and "state_dict_ema" in checkpoint:
        ema_missing, ema_unexpected = model_ema.load_state_dict(checkpoint["state_dict_ema"], strict=False)
        if log_info:
            print(
                "Strict resume: restored EMA model; "
                f"missing={len(ema_missing)}, unexpected={len(ema_unexpected)}"
            )
    elif model_ema is not None:
        ema_missing, ema_unexpected = model_ema.load_state_dict(state_dict, strict=False)
        if log_info:
            print(
                "Strict resume: EMA state missing; initialized EMA from raw model state; "
                f"missing={len(ema_missing)}, unexpected={len(ema_unexpected)}"
            )

    if restore_rng:
        _load_rng_state(checkpoint.get("rng_state"))
        if log_info:
            print(f"Strict resume: restored RNG state={bool(checkpoint.get('rng_state'))}")

    return int(checkpoint.get("epoch", 0) or 0)


def maybe_unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def all_reduce_parameter_grads(model: nn.Module) -> int:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return 0
    world_size = max(1, torch.distributed.get_world_size())
    reduced = 0
    for param in maybe_unwrap_ddp(model).parameters():
        if param.grad is None:
            continue
        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
        param.grad.div_(world_size)
        reduced += param.numel()
    return reduced


def set_attention_mode(model: nn.Module, collect_attention: bool = False, qqkkvv: bool = False) -> None:
    for module in model.modules():
        module_name = type(module).__name__
        is_swin_attention = module_name == "ShiftedWindowAttention" or module_name.startswith("QAttention_swin")
        if hasattr(module, "collect_attention") or is_swin_attention:
            setattr(module, "collect_attention", collect_attention)
        if hasattr(module, "qqkkvv"):
            setattr(module, "qqkkvv", qqkkvv)


def is_attention_module(module: nn.Module) -> bool:
    module_name = type(module).__name__
    return module_name == "ShiftedWindowAttention" or module_name.startswith("QAttention_swin")


def parse_layer_indices(spec: str) -> Optional[Tuple[int, ...]]:
    text = str(spec or "all").strip().lower()
    if text in {"", "all", "*"}:
        return None
    return tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))


@contextlib.contextmanager
def capture_attention_outputs(model: nn.Module, layer_indices: Optional[Tuple[int, ...]] = None, detach: bool = False) -> Iterator[List[torch.Tensor]]:
    outputs: List[torch.Tensor] = []
    handles = []
    layer_set = set(layer_indices) if layer_indices is not None else None
    layer_idx = 0

    def make_hook():
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(tensor):
                outputs.append(tensor.detach() if detach else tensor)
        return hook

    for module in maybe_unwrap_ddp(model).modules():
        if not is_attention_module(module):
            continue
        if layer_set is None or layer_idx in layer_set:
            handles.append(module.register_forward_hook(make_hook()))
        layer_idx += 1
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def parse_name_list(spec: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in str(spec or "").split(",") if item.strip())


def parse_int_set(spec: str) -> Tuple[int, ...]:
    return tuple(sorted({int(item.strip()) for item in str(spec or "").split(",") if item.strip()}))


def first_tensor_from_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = first_tensor_from_output(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(output, dict):
        for item in output.values():
            tensor = first_tensor_from_output(item)
            if tensor is not None:
                return tensor
        return None
    return None


def module_name_matches(name: str, wanted_name: str) -> bool:
    return name == wanted_name or name.endswith(f".{wanted_name}")


def matched_named_modules(model: nn.Module, module_names: Sequence[str]) -> Tuple[str, ...]:
    wanted = tuple(str(name) for name in module_names)
    matches = []
    for name, _module in maybe_unwrap_ddp(model).named_modules():
        if any(module_name_matches(name, wanted_name) for wanted_name in wanted):
            matches.append(name)
    return tuple(matches)


@contextlib.contextmanager
def capture_named_module_outputs(model: nn.Module, module_names: Sequence[str], detach: bool = False) -> Iterator[List[torch.Tensor]]:
    outputs: List[torch.Tensor] = []
    handles = []
    wanted = tuple(str(name) for name in module_names)

    def hook(_module, _inputs, output):
        tensor = first_tensor_from_output(output)
        if torch.is_tensor(tensor):
            outputs.append(tensor.detach() if detach else tensor)

    for name, module in maybe_unwrap_ddp(model).named_modules():
        if any(module_name_matches(name, wanted_name) for wanted_name in wanted):
            handles.append(module.register_forward_hook(hook))
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def is_activation_quantizer_module_name(name: str) -> bool:
    return (
        name.endswith("input_quant_fn")
        or ".input_quant_fn" in name
        or "quan_a_" in name
    )


def activation_quantizer_selected(name: str, wanted_layers: Sequence[str], wanted_quantizers: Sequence[str]) -> bool:
    if wanted_layers and not parameter_belongs_to_any_module(name, wanted_layers):
        return False
    if wanted_quantizers and not any(name == wanted or name.endswith(f".{wanted}") for wanted in wanted_quantizers):
        return False
    return True


@contextlib.contextmanager
def stochastic_activation_quant_bypass(model: nn.Module, drop_prob: float, module_names: Sequence[str] = ()) -> Iterator[int]:
    prob = float(drop_prob)
    if prob <= 0.0:
        yield 0
        return
    patched = []
    wanted = tuple(str(name) for name in module_names if str(name))

    for name, module in maybe_unwrap_ddp(model).named_modules():
        if not is_activation_quantizer_module_name(name):
            continue
        if wanted and not parameter_belongs_to_any_module(name, wanted):
            continue
        original_forward = module.forward

        def qdrop_forward(self, input, *args, _original_forward=original_forward, **kwargs):
            if self.training and torch.rand((), device=input.device).item() < prob:
                return input
            return _original_forward(input, *args, **kwargs)

        module.forward = types.MethodType(qdrop_forward, module)
        patched.append((module, original_forward))
    try:
        yield len(patched)
    finally:
        for module, original_forward in patched:
            module.forward = original_forward


@contextlib.contextmanager
def capture_activation_quantizer_inputs(
    model: nn.Module,
    module_names: Sequence[str],
    quantizer_names: Sequence[str] = (),
    detach: bool = False,
) -> Iterator[List[Tuple[str, nn.Module, torch.Tensor]]]:
    captures: List[Tuple[str, nn.Module, torch.Tensor]] = []
    handles = []
    wanted_layers = tuple(str(name) for name in module_names if str(name))
    wanted_quantizers = tuple(str(name) for name in quantizer_names if str(name))

    def make_hook(module_name: str):
        def hook(module, inputs, _output):
            if not inputs:
                return
            tensor = first_tensor_from_output(inputs[0])
            if torch.is_tensor(tensor):
                captures.append((module_name, module, tensor.detach() if detach else tensor))

        return hook

    for name, module in maybe_unwrap_ddp(model).named_modules():
        if not is_activation_quantizer_module_name(name):
            continue
        if not activation_quantizer_selected(name, wanted_layers, wanted_quantizers):
            continue
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def install_activation_quantizer_input_hooks(
    model: nn.Module,
    module_names: Sequence[str],
    quantizer_names: Sequence[str] = (),
    detach: bool = False,
) -> Tuple[List[Tuple[str, nn.Module, torch.Tensor]], List[torch.utils.hooks.RemovableHandle]]:
    captures: List[Tuple[str, nn.Module, torch.Tensor]] = []
    handles: List[torch.utils.hooks.RemovableHandle] = []
    wanted_layers = tuple(str(name) for name in module_names if str(name))
    wanted_quantizers = tuple(str(name) for name in quantizer_names if str(name))

    def make_hook(module_name: str):
        def hook(module, inputs, _output):
            if not inputs:
                return
            tensor = first_tensor_from_output(inputs[0])
            if torch.is_tensor(tensor):
                captures.append((module_name, module, tensor.detach() if detach else tensor))

        return hook

    for name, module in maybe_unwrap_ddp(model).named_modules():
        if not is_activation_quantizer_module_name(name):
            continue
        if not activation_quantizer_selected(name, wanted_layers, wanted_quantizers):
            continue
        handles.append(module.register_forward_hook(make_hook(name)))
    return captures, handles


def activation_bin_margin_loss(
    captures: Sequence[Tuple[str, nn.Module, torch.Tensor]],
    margin: float,
    max_elements: int,
) -> Tuple[torch.Tensor, int]:
    total_loss = None
    total_pairs = 0
    margin_value = float(margin)
    limit = max(1, int(max_elements))
    for _name, quantizer, activation in captures:
        scale = getattr(quantizer, "s", None)
        if scale is None or not torch.is_tensor(scale):
            continue
        threshold_pos = float(getattr(quantizer, "thd_pos", 0.0))
        threshold_neg = float(getattr(quantizer, "thd_neg", -threshold_pos))
        if threshold_pos <= threshold_neg:
            continue
        act = activation.float()
        if act.numel() > limit:
            stride = max(1, act.numel() // limit)
            act = act.reshape(-1)[::stride][:limit]
            scale_for_act = scale.float().mean()
        else:
            scale_for_act = scale.float()
        if scale_for_act.numel() > 1:
            view_shape = [1] * act.dim()
            view_shape[-2] = scale_for_act.numel()
            scaled = act / scale_for_act.clamp_min(1e-5).reshape(view_shape)
        else:
            scaled = act / scale_for_act.clamp_min(1e-5)
        clipped = scaled.clamp(threshold_neg, threshold_pos)
        center_dist = torch.abs(clipped - torch.round(clipped)).clamp(max=0.5)
        boundary_dist = 0.5 - center_dist
        penalty = F.relu(margin_value - boundary_dist).pow(2)
        pair_loss = penalty.mean()
        total_loss = pair_loss if total_loss is None else total_loss + pair_loss
        total_pairs += 1
    if total_loss is None:
        if captures:
            return captures[0][2].new_zeros(()), 0
        raise ValueError("activation_bin_margin_loss received no captures")
    return total_loss, total_pairs


def set_selected_attention_heads(model: nn.Module, head_map: Optional[Dict[int, Sequence[int]]] = None) -> None:
    if head_map is None:
        for module in model.modules():
            if hasattr(module, "collect_attention_head_indices"):
                delattr(module, "collect_attention_head_indices")
        return

    layer_idx = 0
    for module in model.modules():
        module_name = type(module).__name__
        is_swin_attention = module_name == "ShiftedWindowAttention" or module_name.startswith("QAttention_swin")
        if not is_swin_attention:
            continue
        heads = tuple(sorted(set(int(head) for head in head_map.get(layer_idx, ()))))
        setattr(module, "collect_attention_head_indices", heads)
        layer_idx += 1


def clone_ref_model(student_model: nn.Module) -> nn.Module:
    student_core = maybe_unwrap_ddp(student_model)
    ref_model = copy.deepcopy(student_core)
    ref_model.cuda()
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    set_attention_mode(ref_model, collect_attention=True, qqkkvv=False)
    return ref_model


def clone_model_ema(student_model: nn.Module) -> nn.Module:
    student_core = maybe_unwrap_ddp(student_model)
    ema_model = copy.deepcopy(student_core)
    ema_model.cuda()
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    set_attention_mode(ema_model, collect_attention=False, qqkkvv=False)
    return ema_model


def clone_fixed_logit_ref_model(student_model: nn.Module, checkpoint_path: str = "") -> nn.Module:
    student_core = maybe_unwrap_ddp(student_model)
    ref_model = copy.deepcopy(student_core)
    ref_model.cuda()
    ref_model.eval()
    if checkpoint_path:
        strict_resume_checkpoint(
            ref_model,
            checkpoint_path,
            optimizer=None,
            loss_scaler=None,
            lr_scheduler=None,
            model_ema=None,
            restore_rng=False,
            log_info=False,
        )
    for param in ref_model.parameters():
        param.requires_grad_(False)
    set_attention_mode(ref_model, collect_attention=False, qqkkvv=False)
    return ref_model


@torch.no_grad()
def update_model_ema(student_model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    student_core = maybe_unwrap_ddp(student_model)
    student_state = student_core.state_dict()
    ema_state = ema_model.state_dict()
    for name, ema_value in ema_state.items():
        src = student_state[name]
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(src, alpha=1.0 - decay)
        else:
            ema_value.copy_(src)


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


def is_quant_or_shift_parameter(name: str) -> bool:
    quant_tokens = (
        "input_quant_fn",
        "lsqw_fn",
        "statsq_fn",
        "qk_quant",
        "v_quant",
        "quan_a_",
        "move_",
    )
    return any(token in name for token in quant_tokens)


def is_activation_quant_or_shift_parameter(name: str) -> bool:
    act_quant_tokens = (
        "input_quant_fn",
        "quan_a_",
        "move_",
    )
    return any(token in name for token in act_quant_tokens)


def is_high_drift_late_attention_parameter(name: str) -> bool:
    high_drift_tokens = (
        "move_v_",
        ".attn.proj.move_",
        ".attn.quan_a_softmax",
        ".attn.quant_x_4_qkv.move_",
    )
    return any(token in name for token in high_drift_tokens)


def is_move_v_shift_parameter(name: str) -> bool:
    return "move_v_" in name


def matches_quant_slow_state_policy(name: str, policy: str) -> bool:
    if policy == "all":
        return is_quant_or_shift_parameter(name)
    if policy == "activation":
        return is_activation_quant_or_shift_parameter(name)
    raise ValueError(f"Unsupported quant slow state policy: {policy}")


def collect_activation_scale_anchor_state(model: nn.Module, module_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    wanted = tuple(str(name) for name in module_names if str(name))
    for name, param in maybe_unwrap_ddp(model).named_parameters():
        if not name.endswith(".s"):
            continue
        if not is_activation_quant_or_shift_parameter(name):
            continue
        if wanted and not parameter_belongs_to_any_module(name, wanted):
            continue
        state[name] = param.detach().clone()
    return state


def activation_scale_anchor_loss(model: nn.Module, anchor_state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int]:
    if not anchor_state:
        first_param = next(maybe_unwrap_ddp(model).parameters())
        return first_param.new_zeros(()), 0
    total_loss = None
    count = 0
    for name, param in maybe_unwrap_ddp(model).named_parameters():
        ref = anchor_state.get(name)
        if ref is None:
            continue
        denom = ref.detach().float().pow(2).mean().clamp_min(1e-12)
        loss = F.mse_loss(param.float(), ref.detach().float()) / denom
        total_loss = loss if total_loss is None else total_loss + loss
        count += 1
    if total_loss is None:
        first_param = next(maybe_unwrap_ddp(model).parameters())
        total_loss = first_param.new_zeros(())
    else:
        total_loss = total_loss / max(count, 1)
    return total_loss, count


def collect_variation_trust_state(model: nn.Module, module_names: Sequence[str] = ()) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    wanted = tuple(str(name) for name in module_names if str(name))
    for name, param in maybe_unwrap_ddp(model).named_parameters():
        if not is_activation_quant_or_shift_parameter(name):
            continue
        if wanted and not parameter_belongs_to_any_module(name, wanted):
            continue
        state[name] = param.detach().clone()
    return state


def variation_trust_multiplier(name: str, runtime_args: SimpleNamespace) -> float:
    multiplier = 1.0
    late_layers = parse_name_list(getattr(runtime_args, "variation_trust_late_layers", ""))
    early_layers = parse_name_list(getattr(runtime_args, "variation_trust_early_layers", ""))
    if late_layers and parameter_belongs_to_any_module(name, late_layers):
        multiplier *= float(getattr(runtime_args, "variation_trust_late_multiplier", 1.0))
    if early_layers and parameter_belongs_to_any_module(name, early_layers):
        multiplier *= float(getattr(runtime_args, "variation_trust_early_multiplier", 1.0))
    if "quan_a_softmax" in name:
        multiplier *= float(getattr(runtime_args, "variation_trust_softmax_multiplier", 1.0))
    if "move_v_" in name:
        multiplier *= float(getattr(runtime_args, "variation_trust_move_v_multiplier", 1.0))
    if ".proj.move_" in name:
        multiplier *= float(getattr(runtime_args, "variation_trust_proj_move_multiplier", 1.0))
    return multiplier


def variation_trust_loss(model: nn.Module, trust_state: Dict[str, torch.Tensor], runtime_args: SimpleNamespace) -> Tuple[torch.Tensor, int, float]:
    if not trust_state:
        first_param = next(maybe_unwrap_ddp(model).parameters())
        return first_param.new_zeros(()), 0, 0.0
    total_loss = None
    count = 0
    multiplier_sum = 0.0
    for name, param in maybe_unwrap_ddp(model).named_parameters():
        ref = trust_state.get(name)
        if ref is None:
            continue
        multiplier = variation_trust_multiplier(name, runtime_args)
        if multiplier <= 0:
            continue
        denom = ref.detach().float().pow(2).mean().clamp_min(1e-8)
        loss = F.mse_loss(param.float(), ref.detach().float()) / denom
        loss = float(multiplier) * loss
        total_loss = loss if total_loss is None else total_loss + loss
        count += 1
        multiplier_sum += float(multiplier)
    if total_loss is None:
        first_param = next(maybe_unwrap_ddp(model).parameters())
        total_loss = first_param.new_zeros(())
    else:
        total_loss = total_loss / max(count, 1)
    return total_loss, count, multiplier_sum / max(count, 1)


def load_float_state_dict_from_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = _safe_torch_load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    result = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        name = key[len("module.") :] if key.startswith("module.") else key
        result[name] = value.detach().cpu()
    return result


def parameter_matches_any_pattern(name: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    return any(pattern and pattern in name for pattern in patterns)


def initialize_delta_direction_anchor(runtime_args: SimpleNamespace) -> None:
    if runtime_args.delta_direction_anchor_weight <= 0:
        return
    if hasattr(runtime_args, "_delta_direction_anchor_state"):
        return
    base_path = runtime_args.delta_direction_anchor_base_checkpoint
    target_path = runtime_args.delta_direction_anchor_target_checkpoint
    if not base_path or not target_path:
        raise ValueError("delta direction anchor requires both base and target checkpoints")
    patterns = parse_name_list(runtime_args.delta_direction_anchor_params)
    if not patterns:
        raise ValueError("delta direction anchor requires --delta-direction-anchor-params")
    base_state = load_float_state_dict_from_checkpoint(base_path)
    target_state = load_float_state_dict_from_checkpoint(target_path)
    anchor_state = {}
    for name, base_tensor in base_state.items():
        target_tensor = target_state.get(name)
        if target_tensor is None or target_tensor.shape != base_tensor.shape:
            continue
        if not parameter_matches_any_pattern(name, patterns):
            continue
        target_delta = target_tensor.float() - base_tensor.float()
        if float(target_delta.norm().item()) <= 0:
            continue
        anchor_state[name] = {
            "base": base_tensor.float(),
            "target_delta": target_delta,
        }
    if not anchor_state:
        raise ValueError(f"delta direction anchor matched no parameters for patterns={patterns}")
    runtime_args._delta_direction_anchor_state = anchor_state
    if runtime_args.local_rank == 0:
        print(
            "Initialized delta direction anchor: "
            f"params={len(anchor_state)}, weight={runtime_args.delta_direction_anchor_weight}, "
            f"patterns={patterns}, base={base_path}, target={target_path}, "
            f"start_update={runtime_args.delta_direction_anchor_start_update}"
        )


def delta_direction_anchor_loss(
    model: nn.Module,
    anchor_state: Dict[str, Dict[str, torch.Tensor]],
    runtime_args: SimpleNamespace,
    local_update_count: int,
) -> Tuple[torch.Tensor, int]:
    if not anchor_state or local_update_count < int(getattr(runtime_args, "delta_direction_anchor_start_update", 0)):
        first_param = next(maybe_unwrap_ddp(model).parameters())
        return first_param.new_zeros(()), 0
    total_loss = None
    count = 0
    for name, param in maybe_unwrap_ddp(model).named_parameters():
        entry = anchor_state.get(name)
        if entry is None:
            continue
        base = entry["base"].to(device=param.device, dtype=torch.float32)
        target_delta = entry["target_delta"].to(device=param.device, dtype=torch.float32)
        current_delta = param.float() - base
        if current_delta.numel() == 0:
            continue
        loss = 1.0 - F.cosine_similarity(
            current_delta.reshape(1, -1),
            target_delta.reshape(1, -1).detach(),
            dim=1,
            eps=1e-8,
        ).mean()
        total_loss = loss if total_loss is None else total_loss + loss
        count += 1
    if total_loss is None:
        first_param = next(maybe_unwrap_ddp(model).parameters())
        total_loss = first_param.new_zeros(())
    else:
        total_loss = total_loss / max(count, 1)
    return total_loss, count


def maybe_initialize_variation_trust_anchor(
    model: nn.Module,
    runtime_args: SimpleNamespace,
    local_update_count: int,
    force: bool = False,
) -> bool:
    if runtime_args.variation_trust_weight <= 0:
        return False
    if hasattr(runtime_args, "_variation_trust_state"):
        return False
    start_update = int(getattr(runtime_args, "variation_trust_start_update", 0))
    if not force and local_update_count < start_update:
        return False
    variation_trust_layers = parse_name_list(runtime_args.variation_trust_layers)
    runtime_args._variation_trust_state = collect_variation_trust_state(model, variation_trust_layers)
    if runtime_args.local_rank == 0:
        print(
            "Initialized variation trust anchor: "
            f"params={len(runtime_args._variation_trust_state)}, "
            f"weight={runtime_args.variation_trust_weight}, "
            f"layers={variation_trust_layers}, "
            f"late_layers={runtime_args.variation_trust_late_layers}, "
            f"early_layers={runtime_args.variation_trust_early_layers}, "
            f"start_update={start_update}, "
            f"current_update={local_update_count}"
        )
    return True


def activation_percentile_scale_from_input(quantizer: nn.Module, x: torch.Tensor, percentile: float) -> Optional[torch.Tensor]:
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
    q = float(percentile)
    if getattr(quantizer, "per_channel", False) and scale.numel() > 1:
        if values.ndim == 3 and scale.numel() == values.shape[-2]:
            channel_dim = values.ndim - 2
        elif values.ndim == 4 and scale.numel() == values.shape[-2]:
            channel_dim = values.ndim - 2
        elif values.ndim == 3 and scale.numel() == values.shape[-1]:
            channel_dim = values.ndim - 1
        elif values.ndim == 4 and scale.numel() == values.shape[-1]:
            channel_dim = values.ndim - 1
        elif values.ndim == 2 and scale.numel() == values.shape[-2]:
            channel_dim = values.ndim - 2
        elif values.ndim == 2 and scale.numel() == values.shape[-1]:
            channel_dim = values.ndim - 1
        else:
            return None
        flattened = values.movedim(channel_dim, 0).reshape(scale.numel(), -1)
        stat = torch.quantile(flattened, q, dim=1)
        if stat.numel() != scale.numel():
            return None
        return (stat.reshape_as(scale.detach()) / threshold).clamp_min(1e-5)
    stat = torch.quantile(values.reshape(-1), q)
    return (stat.reshape_as(scale.detach()) / threshold).clamp_min(1e-5)


def parse_mse_calib_grid(spec: str) -> Tuple[float, float, int]:
    parts = [part.strip() for part in str(spec or "").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"pre_qat_act_mse_calib_grid must be min,max,steps, got {spec!r}")
    min_ratio = float(parts[0])
    max_ratio = float(parts[1])
    steps = int(parts[2])
    if min_ratio <= 0 or max_ratio <= 0 or max_ratio < min_ratio or steps < 1:
        raise ValueError(f"invalid pre_qat_act_mse_calib_grid={spec!r}")
    return min_ratio, max_ratio, steps


def activation_mse_scale_from_input(quantizer: nn.Module, x: torch.Tensor, grid_spec: str) -> Optional[torch.Tensor]:
    scale = getattr(quantizer, "s", None)
    if scale is None:
        return None
    threshold = float(getattr(quantizer, "thd_pos", 0.0))
    if threshold <= 0:
        return None
    x_float = x.detach().float()
    if bool(getattr(quantizer, "all_positive", False)):
        values = x_float.clamp_min(0.0)
        signed = False
    else:
        values = x_float
        signed = True
    if getattr(quantizer, "per_channel", False) and scale.numel() > 1:
        if values.ndim == 3 and scale.numel() == values.shape[-2]:
            channel_dim = values.ndim - 2
        elif values.ndim == 4 and scale.numel() == values.shape[-2]:
            channel_dim = values.ndim - 2
        elif values.ndim == 3 and scale.numel() == values.shape[-1]:
            channel_dim = values.ndim - 1
        elif values.ndim == 4 and scale.numel() == values.shape[-1]:
            channel_dim = values.ndim - 1
        elif values.ndim == 2 and scale.numel() == values.shape[-2]:
            channel_dim = values.ndim - 2
        elif values.ndim == 2 and scale.numel() == values.shape[-1]:
            channel_dim = values.ndim - 1
        else:
            return None
        flattened = values.movedim(channel_dim, 0).reshape(scale.numel(), -1)
    else:
        flattened = values.reshape(1, -1)
    scale_flat = scale.detach().float().abs().reshape(-1)
    if scale_flat.numel() == 1 and flattened.shape[0] != 1:
        scale_flat = scale_flat.expand(flattened.shape[0])
    if flattened.shape[0] != scale_flat.numel():
        return None
    min_ratio, max_ratio, steps = parse_mse_calib_grid(grid_spec)
    ratios = torch.linspace(min_ratio, max_ratio, steps, device=flattened.device, dtype=flattened.dtype)
    best_error = None
    best_scale = scale_flat.clone()
    for ratio in ratios:
        candidate = (scale_flat * ratio).clamp_min(1e-5)
        normalized = flattened / candidate[:, None]
        if signed and int(getattr(quantizer, "bit", 0)) == 1:
            quant = torch.sign(normalized)
        else:
            quant = torch.round(normalized).clamp(float(getattr(quantizer, "thd_neg", 0.0)), threshold)
        recon = quant * candidate[:, None]
        error = (recon - flattened).pow(2).mean(dim=1)
        if best_error is None:
            best_error = error
            best_scale = candidate
        else:
            take = error < best_error
            best_error = torch.where(take, error, best_error)
            best_scale = torch.where(take, candidate, best_scale)
    return best_scale.reshape_as(scale.detach()).clamp_min(1e-5)


@torch.no_grad()
def run_pre_qat_activation_percentile_calibration(
    model: nn.Module,
    loader,
    runtime_args: SimpleNamespace,
    amp_autocast,
) -> None:
    batches = max(0, int(getattr(runtime_args, "pre_qat_act_percentile_calib_batches", 0)))
    if batches <= 0:
        return
    percentile = float(getattr(runtime_args, "pre_qat_act_percentile_calib_percentile", 0.999))
    blend = float(getattr(runtime_args, "pre_qat_act_percentile_calib_blend", 1.0))
    wanted_layers = parse_name_list(getattr(runtime_args, "pre_qat_act_percentile_calib_layers", ""))
    stats: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def make_hook(name: str, module: nn.Module):
        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            target_scale = activation_percentile_scale_from_input(module, x, percentile)
            if target_scale is not None:
                stats.setdefault(name, []).append(target_scale.detach().cpu())
        return hook

    root = maybe_unwrap_ddp(model)
    for name, module in root.named_modules():
        if not is_activation_quantizer_module_name(name):
            continue
        if wanted_layers and not parameter_belongs_to_any_module(name, wanted_layers):
            continue
        if getattr(module, "s", None) is None:
            continue
        handles.append(module.register_forward_pre_hook(make_hook(name, module)))

    if runtime_args.local_rank == 0:
        print(
            "Starting pre-QAT activation percentile calibration: "
            f"batches={batches}, percentile={percentile}, blend={blend}, "
            f"layers={wanted_layers}, quantizers={len(handles)}"
        )
    if not handles:
        if runtime_args.local_rank == 0:
            print("Skipped pre-QAT activation percentile calibration: no matched initialized activation quantizers")
        return

    model.train()
    seen = 0
    try:
        for input, _target in loader:
            if seen >= batches:
                break
            if not runtime_args.prefetcher:
                input = input.cuda(non_blocking=True)
            if runtime_args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                model(input)
            seen += 1
    finally:
        for handle in handles:
            handle.remove()

    updated = 0
    ratios = []
    module_map = dict(root.named_modules())
    for name, values in stats.items():
        module = module_map.get(name)
        scale = getattr(module, "s", None) if module is not None else None
        if scale is None or not values:
            continue
        target = torch.stack(values, dim=0).median(dim=0).values.to(device=scale.device, dtype=scale.dtype)
        old = scale.detach().clone()
        blended = old.mul(1.0 - blend).add(target, alpha=blend).clamp_min(1e-5)
        scale.data.copy_(blended)
        ratios.append((blended.detach().float().mean() / old.detach().float().mean().clamp_min(1e-12)).item())
        updated += 1

    if runtime_args.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if runtime_args.local_rank == 0:
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        min_ratio = min(ratios) if ratios else 0.0
        max_ratio = max(ratios) if ratios else 0.0
        print(
            "Finished pre-QAT activation percentile calibration: "
            f"batches={seen}, updated={updated}, mean_scale_ratio={mean_ratio:.4f}, "
            f"min_ratio={min_ratio:.4f}, max_ratio={max_ratio:.4f}"
        )


@torch.no_grad()
def run_pre_qat_activation_mse_calibration(
    model: nn.Module,
    loader,
    runtime_args: SimpleNamespace,
    amp_autocast,
) -> None:
    batches = max(0, int(getattr(runtime_args, "pre_qat_act_mse_calib_batches", 0)))
    if batches <= 0:
        return
    grid_spec = str(getattr(runtime_args, "pre_qat_act_mse_calib_grid", "0.75,1.0,11"))
    blend = float(getattr(runtime_args, "pre_qat_act_mse_calib_blend", 1.0))
    wanted_layers = parse_name_list(getattr(runtime_args, "pre_qat_act_mse_calib_layers", ""))
    wanted_quantizers = parse_name_list(getattr(runtime_args, "pre_qat_act_mse_calib_quantizers", ""))
    targets: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def quantizer_selected(name: str) -> bool:
        if wanted_quantizers and not any(name == wanted or name.endswith(f".{wanted}") for wanted in wanted_quantizers):
            return False
        if wanted_layers and not parameter_belongs_to_any_module(name, wanted_layers):
            return False
        return True

    def make_hook(name: str, module: nn.Module):
        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            target_scale = activation_mse_scale_from_input(module, x, grid_spec)
            if target_scale is not None:
                targets.setdefault(name, []).append(target_scale.detach().cpu())
        return hook

    root = maybe_unwrap_ddp(model)
    for name, module in root.named_modules():
        if not is_activation_quantizer_module_name(name):
            continue
        if not quantizer_selected(name):
            continue
        if getattr(module, "s", None) is None:
            continue
        handles.append(module.register_forward_pre_hook(make_hook(name, module)))

    if runtime_args.local_rank == 0:
        print(
            "Starting pre-QAT activation MSE calibration: "
            f"batches={batches}, grid={grid_spec}, blend={blend}, layers={wanted_layers}, "
            f"quantizers={wanted_quantizers}, matched={len(handles)}"
        )
    if not handles:
        if runtime_args.local_rank == 0:
            print("Skipped pre-QAT activation MSE calibration: no matched initialized activation quantizers")
        return

    model.train()
    seen = 0
    try:
        for input, _target in loader:
            if seen >= batches:
                break
            if not runtime_args.prefetcher:
                input = input.cuda(non_blocking=True)
            if runtime_args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                model(input)
            seen += 1
    finally:
        for handle in handles:
            handle.remove()

    updated = 0
    ratios = []
    module_map = dict(root.named_modules())
    for name, values in targets.items():
        module = module_map.get(name)
        scale = getattr(module, "s", None) if module is not None else None
        if scale is None or not values:
            continue
        target = torch.stack(values, dim=0).median(dim=0).values.to(device=scale.device, dtype=scale.dtype)
        old = scale.detach().clone()
        blended = old.mul(1.0 - blend).add(target, alpha=blend).clamp_min(1e-5)
        scale.data.copy_(blended)
        ratios.append((blended.detach().float().mean() / old.detach().float().mean().clamp_min(1e-12)).item())
        updated += 1

    if runtime_args.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if runtime_args.local_rank == 0:
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        min_ratio = min(ratios) if ratios else 0.0
        max_ratio = max(ratios) if ratios else 0.0
        print(
            "Finished pre-QAT activation MSE calibration: "
            f"batches={seen}, updated={updated}, mean_scale_ratio={mean_ratio:.4f}, "
            f"min_ratio={min_ratio:.4f}, max_ratio={max_ratio:.4f}"
        )


def is_head_norm_parameter(name: str) -> bool:
    return name.startswith("head.") or ".norm" in name or name.startswith("norm.")


def is_attention_projection_parameter(name: str) -> bool:
    return (
        ".attn.q." in name
        or ".attn.k." in name
        or ".attn.v." in name
        or ".attn.qkv." in name
        or ".attn.proj." in name
    )


def is_attention_proj_only_parameter(name: str) -> bool:
    return ".attn.proj." in name


def is_attention_submodule_parameter(name: str) -> bool:
    return ".attn." in name


def parameter_belongs_to_layer_prefix(name: str, layer_names: Sequence[str]) -> bool:
    for layer_name in layer_names:
        if name == layer_name or name.startswith(f"{layer_name}.") or name.endswith(f".{layer_name}") or f".{layer_name}." in name:
            return True
    return False


def set_trainable_policy(model: nn.Module, policy: str, runtime_args: Optional[SimpleNamespace] = None) -> Tuple[int, int]:
    policy = str(policy or "all")
    if policy not in {"all", "non_quant", "freeze_act_quant", "freeze_act_except_layers", "quant", "quant_in_layers", "params_in_layers", "params_in_layers_attn_plus_quant", "params_in_layers_freeze_highdrift_act", "params_in_layers_freeze_move_v_shift", "head_norm_quant", "head_norm_proj_quant", "head_norm_attn_quant", "attn_quant"}:
        raise ValueError(f"Unsupported trainable policy: {policy}")
    selected_layers = parse_name_list(getattr(runtime_args, "trainable_policy_freeze_act_except_layers", "")) if runtime_args is not None else ()
    trainable = 0
    frozen = 0
    for name, param in maybe_unwrap_ddp(model).named_parameters():
        if policy == "all":
            should_train = True
        elif policy == "non_quant":
            should_train = not is_quant_or_shift_parameter(name)
        elif policy == "freeze_act_quant":
            should_train = not is_activation_quant_or_shift_parameter(name)
        elif policy == "freeze_act_except_layers":
            should_train = (not is_activation_quant_or_shift_parameter(name)) or parameter_belongs_to_layer_prefix(name, selected_layers)
        elif policy == "quant":
            should_train = is_quant_or_shift_parameter(name)
        elif policy == "quant_in_layers":
            should_train = is_quant_or_shift_parameter(name) and parameter_belongs_to_layer_prefix(name, selected_layers)
        elif policy == "params_in_layers":
            should_train = parameter_belongs_to_layer_prefix(name, selected_layers)
        elif policy == "params_in_layers_attn_plus_quant":
            in_selected_layer = parameter_belongs_to_layer_prefix(name, selected_layers)
            should_train = in_selected_layer and (is_attention_submodule_parameter(name) or is_quant_or_shift_parameter(name))
        elif policy == "params_in_layers_freeze_highdrift_act":
            should_train = parameter_belongs_to_layer_prefix(name, selected_layers) and not is_high_drift_late_attention_parameter(name)
        elif policy == "params_in_layers_freeze_move_v_shift":
            should_train = parameter_belongs_to_layer_prefix(name, selected_layers) and not is_move_v_shift_parameter(name)
        elif policy == "head_norm_quant":
            should_train = is_quant_or_shift_parameter(name) or is_head_norm_parameter(name)
        elif policy == "head_norm_proj_quant":
            should_train = is_quant_or_shift_parameter(name) or is_head_norm_parameter(name) or is_attention_proj_only_parameter(name)
        elif policy == "attn_quant":
            should_train = is_quant_or_shift_parameter(name) or is_attention_projection_parameter(name)
        else:
            should_train = is_quant_or_shift_parameter(name) or is_head_norm_parameter(name) or is_attention_projection_parameter(name)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return trainable, frozen


def parameter_matches_trainable_policy(name: str, policy: str, runtime_args: Optional[SimpleNamespace] = None) -> bool:
    policy = str(policy or "all")
    if policy == "all":
        return True
    if policy == "non_quant":
        return not is_quant_or_shift_parameter(name)
    if policy == "freeze_act_quant":
        return not is_activation_quant_or_shift_parameter(name)
    if policy == "freeze_act_except_layers":
        selected_layers = parse_name_list(getattr(runtime_args, "trainable_policy_freeze_act_except_layers", "")) if runtime_args is not None else ()
        return (not is_activation_quant_or_shift_parameter(name)) or parameter_belongs_to_layer_prefix(name, selected_layers)
    if policy == "quant":
        return is_quant_or_shift_parameter(name)
    if policy == "quant_in_layers":
        selected_layers = parse_name_list(getattr(runtime_args, "trainable_policy_freeze_act_except_layers", "")) if runtime_args is not None else ()
        return is_quant_or_shift_parameter(name) and parameter_belongs_to_layer_prefix(name, selected_layers)
    if policy == "params_in_layers":
        selected_layers = parse_name_list(getattr(runtime_args, "trainable_policy_freeze_act_except_layers", "")) if runtime_args is not None else ()
        return parameter_belongs_to_layer_prefix(name, selected_layers)
    if policy == "params_in_layers_attn_plus_quant":
        selected_layers = parse_name_list(getattr(runtime_args, "trainable_policy_freeze_act_except_layers", "")) if runtime_args is not None else ()
        return parameter_belongs_to_layer_prefix(name, selected_layers) and (is_attention_submodule_parameter(name) or is_quant_or_shift_parameter(name))
    if policy == "params_in_layers_freeze_highdrift_act":
        selected_layers = parse_name_list(getattr(runtime_args, "trainable_policy_freeze_act_except_layers", "")) if runtime_args is not None else ()
        return parameter_belongs_to_layer_prefix(name, selected_layers) and not is_high_drift_late_attention_parameter(name)
    if policy == "params_in_layers_freeze_move_v_shift":
        selected_layers = parse_name_list(getattr(runtime_args, "trainable_policy_freeze_act_except_layers", "")) if runtime_args is not None else ()
        return parameter_belongs_to_layer_prefix(name, selected_layers) and not is_move_v_shift_parameter(name)
    if policy == "head_norm_quant":
        return is_quant_or_shift_parameter(name) or is_head_norm_parameter(name)
    if policy == "head_norm_proj_quant":
        return is_quant_or_shift_parameter(name) or is_head_norm_parameter(name) or is_attention_proj_only_parameter(name)
    if policy == "attn_quant":
        return is_quant_or_shift_parameter(name) or is_attention_projection_parameter(name)
    if policy == "head_norm_attn_quant":
        return is_quant_or_shift_parameter(name) or is_head_norm_parameter(name) or is_attention_projection_parameter(name)
    raise ValueError(f"Unsupported trainable policy: {policy}")


def apply_gradient_mask_policy(model: nn.Module, policy: str, runtime_args: Optional[SimpleNamespace] = None) -> None:
    if str(policy or "all") == "all":
        return
    for name, param in maybe_unwrap_ddp(model).named_parameters():
        if param.grad is not None and not parameter_matches_trainable_policy(name, policy, runtime_args=runtime_args):
            param.grad = None


def apply_gradient_damp_policy(model: nn.Module, policy: str, runtime_args: Optional[SimpleNamespace] = None) -> Tuple[int, int]:
    if str(policy or "all") == "all":
        return 0, 0
    damp = float(getattr(runtime_args, "trainable_policy_grad_damp", 0.1) if runtime_args is not None else 0.1)
    base_policy = str(getattr(runtime_args, "trainable_policy", "all") if runtime_args is not None else "all")
    damped = 0
    masked = 0
    for name, param in maybe_unwrap_ddp(model).named_parameters():
        if param.grad is None:
            continue
        if parameter_matches_trainable_policy(name, policy, runtime_args=runtime_args):
            continue
        if parameter_matches_trainable_policy(name, base_policy, runtime_args=runtime_args):
            param.grad.mul_(damp)
            damped += param.numel()
        else:
            param.grad = None
            masked += param.numel()
    return damped, masked


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr)


def set_quant_lr_multiplier(optimizer: torch.optim.Optimizer, multiplier: float) -> int:
    updated = 0
    for param_group in optimizer.param_groups:
        if float(param_group.get("lr_scale", 1.0)) != 1.0:
            param_group["lr_scale"] = float(multiplier)
            updated += 1
    return updated


def quantized_weight_for_bin_regularizer(module: nn.Module, weight: torch.Tensor) -> Optional[torch.Tensor]:
    if hasattr(module, "statsq_fn"):
        return module.statsq_fn(weight)
    if hasattr(module, "lsqw_fn"):
        return module.lsqw_fn(weight)
    return None


def bin_variance_regularizer_for_pair(
    weight: torch.Tensor,
    weight_q: torch.Tensor,
    variance_weight: float,
    num_bits: int = 4,
) -> torch.Tensor:
    weight_q_detached = weight_q.detach()
    loss = F.mse_loss(weight.float(), weight_q_detached.float())
    if variance_weight <= 0:
        return loss

    with torch.no_grad():
        qmax = max(1, (2 ** (int(num_bits) - 1)) - 1)
        scale = weight_q_detached.detach().abs().amax().clamp_min(1e-12) / float(qmax)
        int_bins = torch.round(weight_q_detached.detach() / scale).clamp(-qmax - 1, qmax).to(torch.int16)
    variance_terms = []
    flat_weight = weight.float().reshape(-1)
    flat_bins = int_bins.reshape(-1)
    for bin_value in range(-qmax - 1, qmax + 1):
        mask = flat_bins == bin_value
        if torch.count_nonzero(mask) > 1:
            variance_terms.append(torch.var(flat_weight[mask], unbiased=False))
    if variance_terms:
        loss = loss + float(variance_weight) * torch.stack(variance_terms).mean()
    return loss


def bin_regularizer_loss(
    model: nn.Module,
    variance_weight: float = 1.0,
    module_names: Sequence[str] = (),
    attn_only: bool = False,
) -> Tuple[torch.Tensor, int]:
    unwrapped = maybe_unwrap_ddp(model)
    total_loss = None
    total_pairs = 0
    wanted = tuple(str(name) for name in module_names if str(name))
    for module_name, module in unwrapped.named_modules():
        in_scope = (not wanted) or parameter_belongs_to_any_module(module_name, wanted)
        if not in_scope:
            continue
        if not attn_only and hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor):
            weight_q = quantized_weight_for_bin_regularizer(module, module.weight)
            if weight_q is not None:
                pair_loss = bin_variance_regularizer_for_pair(module.weight, weight_q, variance_weight, getattr(module, "weight_bits", 4))
                total_loss = pair_loss if total_loss is None else total_loss + pair_loss
                total_pairs += 1
        if hasattr(module, "qk_quant"):
            for attr in ("q", "k"):
                linear = getattr(module, attr, None)
                if linear is not None and hasattr(linear, "weight"):
                    weight_q = module.qk_quant(linear.weight)
                    pair_loss = bin_variance_regularizer_for_pair(linear.weight, weight_q, variance_weight, getattr(module, "weight_bits", 4))
                    total_loss = pair_loss if total_loss is None else total_loss + pair_loss
                    total_pairs += 1
        if hasattr(module, "v_quant"):
            linear = getattr(module, "v", None)
            if linear is not None and hasattr(linear, "weight"):
                weight_q = module.v_quant(linear.weight)
                pair_loss = bin_variance_regularizer_for_pair(linear.weight, weight_q, variance_weight, getattr(module, "weight_bits", 4))
                total_loss = pair_loss if total_loss is None else total_loss + pair_loss
                total_pairs += 1
    if total_loss is None:
        first_param = next(unwrapped.parameters())
        total_loss = first_param.new_zeros(())
    return total_loss, total_pairs


def _lsq_weight_boundary_mask(weight: torch.Tensor, quantizer: nn.Module, margin: float) -> torch.Tensor:
    scale = getattr(quantizer, "s", None)
    if scale is None:
        return torch.ones_like(weight, dtype=torch.bool)
    scale = scale.detach().float().clamp_min(1e-12)
    if scale.ndim == 1 and weight.ndim >= 2 and scale.numel() == weight.shape[0]:
        scale = scale.reshape(-1, *([1] * (weight.ndim - 1)))
    normalized = weight.detach().float() / scale
    thd_neg = float(getattr(quantizer, "thd_neg", -8))
    thd_pos = float(getattr(quantizer, "thd_pos", 7))
    clipped = normalized.clamp(thd_neg, thd_pos)
    center_dist = (clipped - torch.round(clipped)).abs().clamp(max=0.5)
    boundary_dist = 0.5 - center_dist
    return boundary_dist <= float(margin)


def _lsq_weight_bins_and_boundary(
    weight: torch.Tensor,
    quantizer: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    scale = getattr(quantizer, "s", None)
    if scale is None:
        return None
    scale = scale.detach().float().clamp_min(1e-12)
    if scale.ndim == 1 and weight.ndim >= 2 and scale.numel() == weight.shape[0]:
        scale = scale.reshape(-1, *([1] * (weight.ndim - 1)))
    thd_neg = float(getattr(quantizer, "thd_neg", -8))
    thd_pos = float(getattr(quantizer, "thd_pos", 7))
    normalized = weight.detach().float() / scale
    clipped = normalized.clamp(thd_neg, thd_pos)
    bins = torch.round(clipped).clamp(thd_neg, thd_pos).to(torch.int16)
    center_dist = (clipped - torch.round(clipped)).abs().clamp(max=0.5)
    boundary_dist = 0.5 - center_dist
    return bins, boundary_dist


def capture_selective_bin_anchor_state(
    model: nn.Module,
    module_names: Sequence[str],
    margin: float,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], int, int, int]:
    root = maybe_unwrap_ddp(model)
    wanted = tuple(str(name) for name in module_names if str(name))
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    pairs = 0
    masked = 0
    total = 0
    for module_name, module in root.named_modules():
        if wanted and not parameter_belongs_to_any_module(module_name, wanted):
            continue
        weight = getattr(module, "weight", None)
        quantizer = getattr(module, "lsqw_fn", None)
        if weight is None or quantizer is None or not isinstance(weight, torch.Tensor):
            continue
        with torch.no_grad():
            target = quantizer(weight).detach().clone()
            mask = _lsq_weight_boundary_mask(weight, quantizer, margin).detach().clone()
        state[f"{module_name}.weight"] = {"target": target, "mask": mask}
        pairs += 1
        masked += int(mask.sum().item())
        total += int(mask.numel())
    return state, pairs, masked, total


def selective_bin_anchor_loss(
    model: nn.Module,
    anchor_state: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, int, int, int]:
    root = maybe_unwrap_ddp(model)
    total_loss = None
    pairs = 0
    masked = 0
    total = 0
    modules = dict(root.named_modules())
    for key, entry in anchor_state.items():
        if not key.endswith(".weight"):
            continue
        module_name = key[: -len(".weight")]
        module = modules.get(module_name)
        if module is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, torch.Tensor):
            continue
        target = entry["target"].to(device=weight.device, dtype=weight.dtype)
        mask = entry["mask"].to(device=weight.device, dtype=torch.bool)
        if mask.shape != weight.shape:
            continue
        diff = (weight.float() - target.float()).pow(2)
        if bool(mask.any()):
            pair_loss = diff[mask].mean()
        else:
            pair_loss = diff.mean() * 0.0
        total_loss = pair_loss if total_loss is None else total_loss + pair_loss
        pairs += 1
        masked += int(mask.sum().item())
        total += int(mask.numel())
    if total_loss is None:
        first_param = next(root.parameters())
        total_loss = first_param.new_zeros(())
    return total_loss, pairs, masked, total


def capture_candidate_bin_anchor_state(
    model: nn.Module,
    module_names: Sequence[str],
    source_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], int, int, int, int]:
    root = maybe_unwrap_ddp(model)
    wanted = tuple(str(name) for name in module_names if str(name))
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    pairs = 0
    masked = 0
    total = 0
    missing = 0
    for module_name, module in root.named_modules():
        if wanted and not parameter_belongs_to_any_module(module_name, wanted):
            continue
        weight = getattr(module, "weight", None)
        quantizer = getattr(module, "lsqw_fn", None)
        scale = getattr(quantizer, "s", None) if quantizer is not None else None
        source_weight = source_state.get(f"{module_name}.weight")
        source_scale = source_state.get(f"{module_name}.lsqw_fn.s")
        if weight is None or quantizer is None or scale is None or not isinstance(weight, torch.Tensor):
            continue
        if source_weight is None or source_scale is None or tuple(source_weight.shape) != tuple(weight.shape):
            missing += 1
            continue
        with torch.no_grad():
            current_bins = lsq_int_bins_for_mask(weight.detach(), scale.detach(), quantizer)
            source_bins = lsq_int_bins_for_mask(source_weight.to(device=weight.device), source_scale, quantizer)
            if source_bins.shape != current_bins.shape:
                missing += 1
                continue
            mask = (current_bins != source_bins).detach().clone()
            target = quantizer(weight).detach().clone()
        state[f"{module_name}.weight"] = {"target": target, "mask": mask}
        pairs += 1
        masked += int(mask.sum().item())
        total += int(mask.numel())
    return state, pairs, masked, total, missing


def weight_bin_telemetry_snapshot(
    model: nn.Module,
    module_names: Sequence[str],
    margin: float,
) -> Dict[str, Dict[str, torch.Tensor]]:
    root = maybe_unwrap_ddp(model)
    wanted = tuple(str(name) for name in module_names if str(name))
    snapshot: Dict[str, Dict[str, torch.Tensor]] = {}
    for module_name, module in root.named_modules():
        if wanted and not parameter_belongs_to_any_module(module_name, wanted):
            continue
        weight = getattr(module, "weight", None)
        quantizer = getattr(module, "lsqw_fn", None)
        if weight is None or quantizer is None or not isinstance(weight, torch.Tensor):
            continue
        bins_boundary = _lsq_weight_bins_and_boundary(weight, quantizer)
        if bins_boundary is None:
            continue
        bins, boundary = bins_boundary
        snapshot[f"{module_name}.weight"] = {
            "bins": bins.detach().cpu(),
            "near": (boundary.detach().cpu() <= float(margin)),
        }
    return snapshot


def update_weight_bin_telemetry(
    model: nn.Module,
    runtime_args: SimpleNamespace,
    local_update_count: int,
) -> Optional[Dict[str, float]]:
    interval = int(getattr(runtime_args, "weight_bin_telemetry_interval", 0))
    if interval <= 0:
        return None
    start = int(getattr(runtime_args, "weight_bin_telemetry_start_update", 0))
    end = int(getattr(runtime_args, "weight_bin_telemetry_end_update", 0))
    if local_update_count < start or (end > 0 and local_update_count > end):
        return None
    if (local_update_count - start) % interval != 0:
        return None

    layers = parse_name_list(getattr(runtime_args, "weight_bin_telemetry_layers", ""))
    current = weight_bin_telemetry_snapshot(
        model,
        layers,
        float(getattr(runtime_args, "weight_bin_telemetry_margin", 0.05)),
    )
    prev = getattr(runtime_args, "_weight_bin_telemetry_prev", None)
    prev_delta = getattr(runtime_args, "_weight_bin_telemetry_prev_delta", {})
    runtime_args._weight_bin_telemetry_prev = current

    total = sum(int(entry["bins"].numel()) for entry in current.values())
    near = sum(int(entry["near"].sum().item()) for entry in current.values())
    if prev is None:
        return {
            "pairs": float(len(current)),
            "total": float(total),
            "near_fraction": near / max(1, total),
            "switch_fraction": 0.0,
            "oscillation_fraction": 0.0,
            "mean_abs_delta": 0.0,
        }

    switched = 0
    oscillated = 0
    weighted_abs_delta = 0.0
    compared = 0
    next_delta = {}
    for name, entry in current.items():
        if name not in prev:
            continue
        curr_bins = entry["bins"]
        prev_bins = prev[name]["bins"]
        if curr_bins.shape != prev_bins.shape:
            continue
        delta = curr_bins.int() - prev_bins.int()
        changed = delta != 0
        abs_delta = delta.abs().float()
        switched += int(changed.sum().item())
        weighted_abs_delta += float(abs_delta.sum().item())
        compared += int(delta.numel())
        old_delta = prev_delta.get(name)
        if old_delta is not None and old_delta.shape == delta.shape:
            oscillated += int(((old_delta * delta) < 0).sum().item())
        next_delta[name] = delta.detach().cpu()
    runtime_args._weight_bin_telemetry_prev_delta = next_delta
    return {
        "pairs": float(len(current)),
        "total": float(total),
        "near_fraction": near / max(1, total),
        "switch_fraction": switched / max(1, compared),
        "oscillation_fraction": oscillated / max(1, compared),
        "mean_abs_delta": weighted_abs_delta / max(1, compared),
    }


def set_aoq_explore_scale_ratio(
    model: nn.Module,
    module_names: Sequence[str],
    scale_ratio: float,
    selective_margin: float = 0.0,
    threshold_ratio: float = 0.0,
) -> int:
    root = maybe_unwrap_ddp(model)
    wanted = tuple(str(name) for name in module_names if str(name))
    ratio = float(scale_ratio)
    threshold = None if float(threshold_ratio) <= 0.0 else float(threshold_ratio)
    margin = float(selective_margin)
    updated = 0
    for module_name, module in root.named_modules():
        if wanted and not parameter_belongs_to_any_module(module_name, wanted):
            continue
        for attr in ("statsq_fn", "qk_quant", "v_quant", "lsqw_fn"):
            quantizer = getattr(module, attr, None)
            if quantizer is not None and hasattr(quantizer, "aoq_scale_ratio"):
                quantizer.aoq_scale_ratio = ratio
                if hasattr(quantizer, "aoq_threshold_ratio"):
                    quantizer.aoq_threshold_ratio = threshold
                if hasattr(quantizer, "aoq_selective_margin"):
                    quantizer.aoq_selective_margin = margin
                updated += 1
    return updated


def set_aoq_explore_layer_ratios(
    model: nn.Module,
    layer_ratios: Dict[str, float],
    selective_margin: float = 0.0,
    threshold_ratio: float = 0.0,
) -> Tuple[int, Dict[str, int]]:
    root = maybe_unwrap_ddp(model)
    updated = 0
    counts: Dict[str, int] = {}
    margin = float(selective_margin)
    threshold = None if float(threshold_ratio) <= 0.0 else float(threshold_ratio)
    for layer_name, scale_ratio in layer_ratios.items():
        wanted = (str(layer_name),)
        ratio = float(scale_ratio)
        layer_updated = 0
        for module_name, module in root.named_modules():
            if not parameter_belongs_to_any_module(module_name, wanted):
                continue
            for attr in ("statsq_fn", "qk_quant", "v_quant", "lsqw_fn"):
                quantizer = getattr(module, attr, None)
                if quantizer is not None and hasattr(quantizer, "aoq_scale_ratio"):
                    quantizer.aoq_scale_ratio = ratio
                    if hasattr(quantizer, "aoq_threshold_ratio"):
                        quantizer.aoq_threshold_ratio = threshold
                    if hasattr(quantizer, "aoq_selective_margin"):
                        quantizer.aoq_selective_margin = margin
                    layer_updated += 1
        counts[str(layer_name)] = layer_updated
        updated += layer_updated
    return updated, counts


def clear_aoq_explore_quality_masks(model: nn.Module) -> int:
    root = maybe_unwrap_ddp(model)
    cleared = 0
    for module in root.modules():
        for attr in ("statsq_fn", "qk_quant", "v_quant", "lsqw_fn"):
            quantizer = getattr(module, attr, None)
            if quantizer is not None and hasattr(quantizer, "aoq_quality_mask"):
                quantizer.aoq_quality_mask = None
                if hasattr(quantizer, "aoq_quality_mode"):
                    quantizer.aoq_quality_mode = "none"
                cleared += 1
    return cleared


def _strip_module_prefix_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key[len("module.") :] if isinstance(key, str) and key.startswith("module.") else key: value
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    }


def load_aoq_anchor_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = _safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"anchor checkpoint does not contain a state_dict: {checkpoint_path}")
    return _strip_module_prefix_state_dict(state_dict)


def lsq_int_bins_for_mask(weight: torch.Tensor, scale: torch.Tensor, quantizer) -> torch.Tensor:
    scale = scale.detach().to(device=weight.device, dtype=torch.float32).clamp_min(1e-5)
    if getattr(quantizer, "per_channel", False):
        if weight.ndim == 2 and scale.ndim == 1:
            scale = scale.view(-1, 1)
        elif weight.ndim == 4 and scale.ndim == 1:
            scale = scale.view(-1, 1, 1, 1)
    normalized = weight.detach().float() / scale
    thd_neg = int(getattr(quantizer, "thd_neg", -8))
    thd_pos = int(getattr(quantizer, "thd_pos", 7))
    return torch.round(normalized.clamp(float(thd_neg), float(thd_pos))).clamp(thd_neg, thd_pos).to(torch.int16)


def update_aoq_anchor_unmoved_masks(
    model: nn.Module,
    module_names: Sequence[str],
    anchor_state: Dict[str, torch.Tensor],
    selective_margin: float,
    select_moved: bool = False,
) -> Dict[str, float]:
    root = maybe_unwrap_ddp(model)
    wanted = tuple(str(name) for name in module_names if str(name))
    margin = float(selective_margin)
    stats = {"pairs": 0.0, "near": 0.0, "selected": 0.0, "moved": 0.0, "missing": 0.0}
    for module_name, module in root.named_modules():
        if wanted and not parameter_belongs_to_any_module(module_name, wanted):
            continue
        weight = getattr(module, "weight", None)
        quantizer = getattr(module, "lsqw_fn", None)
        if weight is None or quantizer is None or not hasattr(quantizer, "aoq_quality_mask"):
            continue
        scale_param = getattr(quantizer, "s", None)
        anchor_weight = anchor_state.get(f"{module_name}.weight")
        anchor_scale = anchor_state.get(f"{module_name}.lsqw_fn.s")
        if scale_param is None or anchor_weight is None or anchor_scale is None:
            stats["missing"] += 1.0
            continue
        if tuple(anchor_weight.shape) != tuple(weight.shape):
            stats["missing"] += 1.0
            continue
        with torch.no_grad():
            weight_data = weight.detach()
            scale = scale_param.detach().clamp_min(1e-5)
            if getattr(quantizer, "per_channel", False):
                if weight_data.ndim == 2 and scale.ndim == 1:
                    scale_for_dist = scale.view(-1, 1)
                elif weight_data.ndim == 4 and scale.ndim == 1:
                    scale_for_dist = scale.view(-1, 1, 1, 1)
                else:
                    scale_for_dist = scale
            else:
                scale_for_dist = scale
            try:
                x_base = weight_data / scale_for_dist
            except RuntimeError:
                stats["missing"] += 1.0
                continue
            center_delta = x_base - torch.round(x_base)
            boundary_dist = 0.5 - torch.abs(center_delta).clamp(max=0.5)
            near_boundary = boundary_dist <= margin

            current_bins = lsq_int_bins_for_mask(weight_data, scale_param.detach(), quantizer)
            anchor_bins = lsq_int_bins_for_mask(anchor_weight.to(device=weight_data.device), anchor_scale, quantizer)
            if anchor_bins.shape != current_bins.shape:
                stats["missing"] += 1.0
                continue
            moved_from_anchor = anchor_bins != current_bins
            quality_mask = near_boundary & (moved_from_anchor if select_moved else (~moved_from_anchor))
            quantizer.aoq_quality_mode = "anchor_moved" if select_moved else "anchor_unmoved"
            quantizer.aoq_quality_mask = quality_mask.detach()
            stats["pairs"] += 1.0
            stats["near"] += float(near_boundary.sum().item())
            stats["selected"] += float(quality_mask.sum().item())
            stats["moved"] += float((near_boundary & moved_from_anchor).sum().item())
    return stats


def update_aoq_history_oscillating_masks(
    model: nn.Module,
    module_names: Sequence[str],
    selective_margin: float,
    min_frac: float = 0.0,
    history_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    recent_only: bool = False,
) -> Dict[str, float]:
    root = maybe_unwrap_ddp(model)
    wanted = tuple(str(name) for name in module_names if str(name))
    margin = float(selective_margin)
    floor_frac = float(min_frac)
    state = history_state if history_state is not None else {}
    stats = {
        "pairs": 0.0,
        "near": 0.0,
        "selected": 0.0,
        "moved": 0.0,
        "switched": 0.0,
        "oscillating": 0.0,
        "missing": 0.0,
    }
    for module_name, module in root.named_modules():
        if wanted and not parameter_belongs_to_any_module(module_name, wanted):
            continue
        weight = getattr(module, "weight", None)
        quantizer = getattr(module, "lsqw_fn", None)
        scale_param = getattr(quantizer, "s", None) if quantizer is not None else None
        if weight is None or quantizer is None or scale_param is None or not hasattr(quantizer, "aoq_quality_mask"):
            continue
        with torch.no_grad():
            weight_data = weight.detach()
            scale = scale_param.detach().clamp_min(1e-5)
            if getattr(quantizer, "per_channel", False):
                if weight_data.ndim == 2 and scale.ndim == 1:
                    scale_for_dist = scale.view(-1, 1)
                elif weight_data.ndim == 4 and scale.ndim == 1:
                    scale_for_dist = scale.view(-1, 1, 1, 1)
                else:
                    scale_for_dist = scale
            else:
                scale_for_dist = scale
            try:
                x_base = weight_data / scale_for_dist
            except RuntimeError:
                stats["missing"] += 1.0
                continue
            center_delta = x_base - torch.round(x_base)
            boundary_dist = 0.5 - torch.abs(center_delta).clamp(max=0.5)
            near_boundary = boundary_dist <= margin
            current_bins = lsq_int_bins_for_mask(weight_data, scale_param.detach(), quantizer)
            key = f"{module_name}.weight"
            entry = state.get(key)
            if entry is None or entry.get("prev_bins") is None or entry["prev_bins"].shape != current_bins.shape:
                entry = {
                    "prev_bins": current_bins.detach().clone(),
                    "prev_delta": torch.zeros_like(current_bins, dtype=torch.int8),
                    "switch_count": torch.zeros_like(current_bins, dtype=torch.int16),
                    "osc_count": torch.zeros_like(current_bins, dtype=torch.int16),
                }
                state[key] = entry
                quality_mask = torch.zeros_like(near_boundary, dtype=torch.bool)
                switched = torch.zeros_like(near_boundary, dtype=torch.bool)
                oscillated = torch.zeros_like(near_boundary, dtype=torch.bool)
            else:
                prev_bins = entry["prev_bins"].to(device=current_bins.device)
                prev_delta = entry["prev_delta"].to(device=current_bins.device)
                switch_count = entry["switch_count"].to(device=current_bins.device)
                osc_count = entry["osc_count"].to(device=current_bins.device)
                delta = (current_bins.to(torch.int16) - prev_bins.to(torch.int16)).clamp(-128, 127).to(torch.int8)
                switched = delta != 0
                oscillated = switched & ((prev_delta.to(torch.int16) * delta.to(torch.int16)) < 0)
                switch_count = (switch_count + switched.to(torch.int16)).clamp(max=32767)
                osc_count = (osc_count + oscillated.to(torch.int16)).clamp(max=32767)
                prev_delta = torch.where(switched, delta, prev_delta).to(torch.int8)
                entry["prev_bins"] = current_bins.detach().clone()
                entry["prev_delta"] = prev_delta.detach().clone()
                entry["switch_count"] = switch_count.detach().clone()
                entry["osc_count"] = osc_count.detach().clone()
                quality_mask = near_boundary & (oscillated if recent_only else (osc_count > 0))
                if floor_frac > 0.0:
                    near_count = int(near_boundary.sum().item())
                    selected_count = int(quality_mask.sum().item())
                    min_count = int(math.ceil(near_count * floor_frac))
                    if selected_count < min_count and min_count > 0:
                        candidate = near_boundary & (~quality_mask) & (switched if recent_only else (switch_count > 0))
                        candidate_scores = (
                            torch.abs(delta.float())
                            if recent_only
                            else switch_count.float() + (2.0 * osc_count.float())
                        )
                        flat_scores = candidate_scores[candidate]
                        if flat_scores.numel() > 0:
                            add_count = min(min_count - selected_count, flat_scores.numel())
                            threshold = torch.topk(flat_scores, k=add_count, largest=True).values[-1]
                            quality_mask = quality_mask | (candidate & (candidate_scores >= threshold))
            quantizer.aoq_quality_mode = "recent_oscillating" if recent_only else "history_oscillating"
            quantizer.aoq_quality_mask = quality_mask.detach()
            stats["pairs"] += 1.0
            stats["near"] += float(near_boundary.sum().item())
            stats["selected"] += float(quality_mask.sum().item())
            stats["moved"] += float(oscillated.sum().item())
            stats["switched"] += float(switched.sum().item())
            stats["oscillating"] += float((near_boundary & quality_mask).sum().item())
    return stats


def update_aoq_explore_quality_masks(
    model: nn.Module,
    module_names: Sequence[str],
    quality_mode: str,
    selective_margin: float,
    min_frac: float = 0.0,
    anchor_state: Optional[Dict[str, torch.Tensor]] = None,
    history_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, float]:
    mode = str(quality_mode or "none")
    if mode == "none":
        return {"pairs": 0.0, "near": 0.0, "selected": 0.0}
    if mode in {"history_oscillating", "recent_oscillating"}:
        return update_aoq_history_oscillating_masks(
            model,
            module_names,
            selective_margin,
            min_frac=min_frac,
            history_state=history_state,
            recent_only=mode == "recent_oscillating",
        )
    if mode in {"anchor_unmoved", "anchor_moved"}:
        if anchor_state is None:
            raise ValueError(f"{mode} requires aoq anchor state")
        return update_aoq_anchor_unmoved_masks(
            model,
            module_names,
            anchor_state,
            selective_margin,
            select_moved=mode == "anchor_moved",
        )
    if mode != "grad_cross":
        raise ValueError(f"unsupported aoq_explore_quality_mode={quality_mode}")
    root = maybe_unwrap_ddp(model)
    wanted = tuple(str(name) for name in module_names if str(name))
    margin = float(selective_margin)
    floor_frac = float(min_frac)
    stats = {"pairs": 0.0, "near": 0.0, "selected": 0.0}
    for module_name, module in root.named_modules():
        if wanted and not parameter_belongs_to_any_module(module_name, wanted):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.grad is None:
            continue
        for attr in ("statsq_fn", "qk_quant", "v_quant", "lsqw_fn"):
            quantizer = getattr(module, attr, None)
            if quantizer is None or not hasattr(quantizer, "aoq_quality_mask"):
                continue
            scale_param = getattr(quantizer, "s", None)
            if scale_param is None:
                continue
            with torch.no_grad():
                weight_data = weight.detach()
                grad = weight.grad.detach()
                scale = scale_param.detach().clamp_min(1e-5)
                if getattr(quantizer, "per_channel", False):
                    if weight_data.ndim == 2:
                        scale = scale.view(-1, 1)
                    elif weight_data.ndim == 4:
                        scale = scale.view(-1, 1, 1, 1)
                try:
                    x_base = weight_data / scale
                except RuntimeError:
                    continue
                rounded = torch.round(x_base)
                center_delta = x_base - rounded
                boundary_dist = (0.5 - torch.abs(center_delta).clamp(max=0.5))
                near_boundary = boundary_dist <= margin
                if not bool(near_boundary.any().item()):
                    quality_mask = torch.zeros_like(near_boundary, dtype=torch.bool)
                else:
                    # A negative gradient step moves opposite to grad; keep near-boundary
                    # elements whose predicted movement points away from the current bin center.
                    step_dir = -torch.sign(grad)
                    quality_mask = near_boundary & ((center_delta * step_dir) > 0)
                    if floor_frac > 0.0:
                        near_count = int(near_boundary.sum().item())
                        selected_count = int(quality_mask.sum().item())
                        min_count = int(math.ceil(near_count * floor_frac))
                        if selected_count < min_count and min_count > 0:
                            candidate = near_boundary & (~quality_mask)
                            candidate_scores = torch.abs(grad * center_delta)
                            flat_scores = candidate_scores[candidate]
                            if flat_scores.numel() > 0:
                                add_count = min(min_count - selected_count, flat_scores.numel())
                                threshold = torch.topk(flat_scores, k=add_count, largest=True).values[-1]
                                quality_mask = quality_mask | (candidate & (candidate_scores >= threshold))
                quantizer.aoq_quality_mode = mode
                quantizer.aoq_quality_mask = quality_mask.detach()
                stats["pairs"] += 1.0
                stats["near"] += float(near_boundary.sum().item())
                stats["selected"] += float(quality_mask.sum().item())
    return stats


def aoq_explore_enabled(runtime_args: SimpleNamespace, local_update_count: int) -> bool:
    schedule = getattr(runtime_args, "aoq_explore_update_schedule", None) or []
    scheduled = aoq_explore_schedule_value(runtime_args, local_update_count)
    if schedule and scheduled is None:
        return False
    if scheduled is not None:
        ratio, threshold_ratio, selective_margin = scheduled
        threshold_enabled = threshold_ratio > 0.0 and abs(threshold_ratio - 1.0) >= 1e-12
        return (
            abs(ratio - 1.0) >= 1e-12
            or threshold_enabled
            or (selective_margin > 0.0 and abs(ratio - 1.0) >= 1e-12)
        )
    ratio = float(getattr(runtime_args, "aoq_explore_scale_ratio", 1.0))
    layer_ratios = parse_layer_float_overrides(getattr(runtime_args, "aoq_explore_layer_ratios", ""))
    threshold_ratio = float(getattr(runtime_args, "aoq_explore_threshold_ratio", 0.0))
    threshold_enabled = threshold_ratio > 0.0 and abs(threshold_ratio - 1.0) >= 1e-12
    if abs(ratio - 1.0) < 1e-12 and not threshold_enabled and not layer_ratios:
        return False
    start = int(getattr(runtime_args, "aoq_explore_start_update", 0))
    end = int(getattr(runtime_args, "aoq_explore_end_update", 0))
    if local_update_count < start:
        return False
    if end > 0 and local_update_count >= end:
        return False
    return True


def aoq_explore_schedule_value(
    runtime_args: SimpleNamespace,
    local_update_count: int,
) -> Optional[Tuple[float, float, float]]:
    schedule = getattr(runtime_args, "aoq_explore_update_schedule", None) or []
    active = None
    for update, scale_ratio, threshold_ratio, selective_margin in schedule:
        if local_update_count >= int(update):
            active = (float(scale_ratio), float(threshold_ratio), float(selective_margin))
        else:
            break
    return active


def maybe_init_quant_slow_state(model: nn.Module, runtime_args: SimpleNamespace) -> None:
    if getattr(runtime_args, "_quant_slow_state", None) is not None:
        return
    if float(getattr(runtime_args, "quant_slow_state_decay", 0.0)) <= 0:
        runtime_args._quant_slow_state = {}
        return
    state = {}
    policy = str(getattr(runtime_args, "quant_slow_state_policy", "all"))
    for name, param in maybe_unwrap_ddp(model).named_parameters():
        if matches_quant_slow_state_policy(name, policy):
            state[name] = param.detach().clone()
    runtime_args._quant_slow_state = state
    if runtime_args.local_rank == 0:
        print(
            "Initialized quant slow state: "
            f"params={len(state)}, policy={policy}, decay={runtime_args.quant_slow_state_decay}, "
            f"sync_interval={runtime_args.quant_slow_state_sync_interval}, pull={runtime_args.quant_slow_state_pull}"
        )


def update_quant_slow_state(model: nn.Module, runtime_args: SimpleNamespace, global_update: int, pull_enabled: bool = True) -> None:
    decay = float(getattr(runtime_args, "quant_slow_state_decay", 0.0))
    sync_interval = int(getattr(runtime_args, "quant_slow_state_sync_interval", 0))
    pull = float(getattr(runtime_args, "quant_slow_state_pull", 0.0))
    if decay <= 0 or sync_interval <= 0 or pull <= 0:
        return
    maybe_init_quant_slow_state(model, runtime_args)
    state = runtime_args._quant_slow_state
    synced = 0
    with torch.no_grad():
        for name, param in maybe_unwrap_ddp(model).named_parameters():
            if name not in state:
                continue
            state[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
            if pull_enabled and global_update > 0 and global_update % sync_interval == 0:
                param.data.mul_(1.0 - pull).add_(state[name], alpha=pull)
                synced += 1
    if synced and runtime_args.local_rank == 0:
        print(f"Applied quant slow state pull: update={global_update}, tensors={synced}, pull={pull}")


def quant_thresholds(bit: int, all_positive: bool = False) -> Tuple[int, int]:
    bit = int(bit)
    if all_positive:
        if bit == 1:
            return 0, 1
        return 0, 2 ** bit - 1
    if bit == 1:
        return -1, 1
    return -2 ** (bit - 1), 2 ** (bit - 1) - 1


def set_fake_quant_bits(model: nn.Module, wbits: int, abits: int, rescale_lsq: bool = False) -> Tuple[int, int]:
    root = maybe_unwrap_ddp(model)
    wbits = int(wbits)
    abits = int(abits)
    weight_modules = 0
    act_modules = 0

    def update_bit_quantizer(quantizer, new_bit: int) -> None:
        old_bit = int(getattr(quantizer, "bit", new_bit))
        old_all_positive = bool(getattr(quantizer, "all_positive", False))
        old_thd_pos = getattr(quantizer, "thd_pos", quant_thresholds(old_bit, old_all_positive)[1])
        new_thd_neg, new_thd_pos = quant_thresholds(new_bit, old_all_positive)
        quantizer.bit = int(new_bit)
        if hasattr(quantizer, "thd_neg"):
            quantizer.thd_neg = new_thd_neg
        if hasattr(quantizer, "thd_pos"):
            quantizer.thd_pos = new_thd_pos
        scale = getattr(quantizer, "s", None)
        if rescale_lsq and scale is not None and old_bit != int(new_bit):
            try:
                ratio = math.sqrt(float(old_thd_pos) / float(new_thd_pos))
                with torch.no_grad():
                    scale.mul_(ratio)
            except Exception:
                pass

    for module in root.modules():
        if hasattr(module, "weight_bits"):
            module.weight_bits = wbits
            weight_modules += 1
        if hasattr(module, "input_bits"):
            module.input_bits = abits
        for attr in ("statsq_fn", "qk_quant", "v_quant"):
            quantizer = getattr(module, attr, None)
            if quantizer is not None and hasattr(quantizer, "num_bits"):
                quantizer.num_bits = wbits
                weight_modules += 1
        for attr in ("lsqw_fn",):
            quantizer = getattr(module, attr, None)
            if quantizer is not None and hasattr(quantizer, "bit"):
                update_bit_quantizer(quantizer, wbits)
                weight_modules += 1
        for attr in ("input_quant_fn", "quant_x_4_qkv", "quan_a_qkx_fn"):
            quantizer = getattr(module, attr, None)
            if quantizer is not None and hasattr(quantizer, "bit"):
                update_bit_quantizer(quantizer, abits)
                act_modules += 1
    return weight_modules, act_modules


def extract_attn_prob_list(attn_info):
    if attn_info is None:
        return []
    extracted = []
    for layer_info in attn_info:
        if layer_info is None:
            extracted.append(None)
            continue
        if isinstance(layer_info, (tuple, list)):
            attn_tensor = layer_info[0]
        else:
            attn_tensor = layer_info
        if torch.is_tensor(attn_tensor):
            extracted.append(attn_tensor)
        else:
            extracted.append(None)
    return extracted


OSCILLATING_SWIN_HEADS = (
    (5, 2),
    (10, 14),
    (5, 1),
    (4, 1),
    (9, 10),
)

OSCILLATING_SWIN_HEADS_TOP10 = OSCILLATING_SWIN_HEADS + (
    (10, 13),
    (10, 15),
    (9, 9),
    (9, 11),
    (5, 3),
)

OSCILLATING_SWIN_HEADS_TOP15 = OSCILLATING_SWIN_HEADS_TOP10 + (
    (4, 2),
    (6, 1),
    (7, 4),
    (8, 8),
    (11, 14),
)


def parse_ref_head_mode(head_mode: str):
    if head_mode == "all":
        return None
    if head_mode == "oscillating_top5":
        return OSCILLATING_SWIN_HEADS
    if head_mode == "oscillating_top10":
        return OSCILLATING_SWIN_HEADS_TOP10
    if head_mode == "oscillating_top15":
        return OSCILLATING_SWIN_HEADS_TOP15
    if head_mode.startswith("custom:"):
        items = []
        raw_items = [item.strip() for item in head_mode[len("custom:") :].split(",") if item.strip()]
        for raw_item in raw_items:
            if ":" not in raw_item:
                raise ValueError(f"Invalid custom ref head item: {raw_item}")
            layer_idx, head_idx = raw_item.split(":", 1)
            items.append((int(layer_idx), int(head_idx)))
        if len(items) < len(OSCILLATING_SWIN_HEADS):
            raise ValueError("custom ref head mode must include at least five heads")
        missing = [head for head in OSCILLATING_SWIN_HEADS if head not in items]
        if missing:
            raise ValueError(f"custom ref head mode must include oscillating_top5 heads: {missing}")
        return tuple(items)
    if head_mode.startswith("custom_subset:"):
        items = []
        raw_items = [item.strip() for item in head_mode[len("custom_subset:") :].split(",") if item.strip()]
        for raw_item in raw_items:
            if ":" not in raw_item:
                raise ValueError(f"Invalid custom_subset ref head item: {raw_item}")
            layer_idx, head_idx = raw_item.split(":", 1)
            items.append((int(layer_idx), int(head_idx)))
        if not items:
            raise ValueError("custom_subset ref head mode must include at least one head")
        return tuple(items)
    raise NotImplementedError(f"Unsupported ref head mode: {head_mode}")


def parse_dynamic_kl_head_list(heads: object) -> Tuple[Tuple[int, int], ...]:
    if heads is None:
        return tuple()
    raw = str(heads or "").strip()
    if not raw:
        return tuple()
    if raw.startswith("custom_subset:"):
        raw = raw[len("custom_subset:") :]
    items: List[Tuple[int, int]] = []
    for raw_item in raw.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid dynamic KL head item: {item}")
        layer_idx, head_idx = item.split(":", 1)
        items.append((int(layer_idx), int(head_idx)))
    seen = set()
    deduped = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return tuple(deduped)


def format_ref_head_mode_from_head(head: Optional[Tuple[int, int]]) -> str:
    if head is None:
        return "all"
    return f"custom_subset:{int(head[0])}:{int(head[1])}"


def format_dynamic_kl_head(head: Optional[Tuple[int, int]]) -> str:
    if head is None:
        return ""
    return f"{int(head[0])}:{int(head[1])}"


def apply_ref_head_mode_to_models(
    head_mode: str,
    model: nn.Module,
    teacher: Optional[nn.Module],
    ref_model: Optional[nn.Module],
    anchor_ref_model: Optional[nn.Module],
    runtime_args: SimpleNamespace,
) -> Tuple[Optional[Dict[int, Tuple[int, ...]]], Optional[Dict[int, Tuple[int, ...]]]]:
    selected_head_map = ref_head_map(head_mode)
    anchor_ref_head_mode = runtime_args.anchor_ref_head_mode or head_mode
    anchor_selected_head_map = ref_head_map(anchor_ref_head_mode)
    set_selected_attention_heads(model, selected_head_map)
    if teacher is not None:
        set_selected_attention_heads(teacher, selected_head_map)
    if ref_model is not None:
        set_selected_attention_heads(ref_model, selected_head_map)
    if anchor_ref_model is not None:
        set_selected_attention_heads(anchor_ref_model, anchor_selected_head_map)
    return selected_head_map, anchor_selected_head_map


def broadcast_dynamic_kl_epoch_decision(
    runtime_args: SimpleNamespace,
    head_mode: str,
    weight: float,
    head: Optional[Tuple[int, int]],
    spike_score: float,
    reason: str,
) -> Tuple[str, float, Optional[Tuple[int, int]], float, str]:
    if not getattr(runtime_args, "distributed", False):
        return head_mode, weight, head, spike_score, reason
    payload = [head_mode, float(weight), list(head) if head is not None else None, float(spike_score), str(reason)]
    dist.broadcast_object_list(payload, src=0)
    received_head_mode, received_weight, received_head, received_spike_score, received_reason = payload
    if received_head is not None:
        received_head = (int(received_head[0]), int(received_head[1]))
    return str(received_head_mode), float(received_weight), received_head, float(received_spike_score), str(received_reason)


class DynamicSparsePrevStepKLController:
    TSV_HEADER = (
        "epoch\tphase\ttop1\ttop5\tsamples\trolling_best\tdrop\t"
        "applied_head\tapplied_weight\tapplied_spike_score\tnext_head\tnext_weight\tnext_spike_score\t"
        "triggered\treason\tprior_source\tcooldown_state\twindow_pulses\n"
    )

    def __init__(self, runtime_args: SimpleNamespace, output_dir: Path):
        self.enabled = bool(getattr(runtime_args, "dynamic_sparse_prevstep_kl", False))
        self.start_epoch = int(getattr(runtime_args, "dynamic_kl_start_epoch", 61))
        self.observe_until_epoch = int(getattr(runtime_args, "dynamic_kl_observe_until_epoch", self.start_epoch - 1))
        self.primary_heads = parse_dynamic_kl_head_list(getattr(runtime_args, "dynamic_kl_primary_heads", ""))
        self.secondary_heads = parse_dynamic_kl_head_list(getattr(runtime_args, "dynamic_kl_secondary_heads", ""))
        self.avoid_heads = set(parse_dynamic_kl_head_list(getattr(runtime_args, "dynamic_kl_avoid_heads", "")))
        self.drop_threshold = float(getattr(runtime_args, "dynamic_kl_drop_threshold", 0.06))
        self.strong_drop_threshold = float(getattr(runtime_args, "dynamic_kl_strong_drop_threshold", 0.12))
        self.default_weight = float(getattr(runtime_args, "dynamic_kl_default_weight", 1e-5))
        self.strong_weight = float(getattr(runtime_args, "dynamic_kl_strong_weight", 2e-5))
        self.max_weight = float(getattr(runtime_args, "dynamic_kl_max_weight", 3e-5))
        self.cooldown_epochs = int(getattr(runtime_args, "dynamic_kl_cooldown_epochs", 5))
        self.window_epochs = max(1, int(getattr(runtime_args, "dynamic_kl_window_epochs", 10)))
        self.max_pulses_per_window = int(getattr(runtime_args, "dynamic_kl_max_pulses_per_window", 3))
        self.prior_source = str(getattr(runtime_args, "dynamic_kl_prior_source", "offline_attn_relation") or "offline_attn_relation")
        self.base_head_mode = str(getattr(runtime_args, "ref_head_mode", "all") or "all")
        self.base_weight = float(getattr(runtime_args, "ref_attn_kl_weight", 0.0))
        self.rolling_best: Optional[float] = None
        self.next_head: Optional[Tuple[int, int]] = None
        self.next_weight = 0.0
        self.next_spike_score = 0.0
        self.next_reason = "init"
        self.cooldown_until: Dict[Tuple[int, int], int] = {}
        self.pulse_epochs: List[int] = []
        tsv_path = str(getattr(runtime_args, "dynamic_kl_controller_tsv", "") or "")
        self.tsv_path = Path(tsv_path) if tsv_path else output_dir / "dynamic_sparse_prevstep_kl_controller.tsv"

        if self.enabled:
            overlap = [head for head in self.primary_heads + self.secondary_heads if head in self.avoid_heads]
            if overlap:
                raise ValueError(f"dynamic KL candidate heads overlap avoid heads: {overlap}")
            if not (self.primary_heads or self.secondary_heads):
                raise ValueError("dynamic sparse prev-step KL requires at least one candidate head")
            if self.start_epoch <= self.observe_until_epoch:
                raise ValueError(
                    "dynamic_kl_start_epoch must be greater than dynamic_kl_observe_until_epoch; "
                    f"got start={self.start_epoch}, observe_until={self.observe_until_epoch}"
                )
            if self.default_weight < 0 or self.strong_weight < 0 or self.max_weight < 0:
                raise ValueError("dynamic KL weights must be non-negative")
            if max(self.default_weight, self.strong_weight) > self.max_weight:
                raise ValueError(
                    "dynamic KL default/strong weights must not exceed max weight; "
                    f"default={self.default_weight}, strong={self.strong_weight}, max={self.max_weight}"
                )

    def initialize_log(self, local_rank: int) -> None:
        if not self.enabled or local_rank != 0:
            return
        self.tsv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.tsv_path.exists():
            with open(self.tsv_path, "w", encoding="utf-8") as handle:
                handle.write(self.TSV_HEADER)

    def decision_for_epoch(self, epoch: int) -> Tuple[str, float, Optional[Tuple[int, int]], float, str]:
        if not self.enabled:
            return self.base_head_mode, self.base_weight, None, 0.0, "disabled"
        if epoch <= self.observe_until_epoch or epoch < self.start_epoch:
            return self.base_head_mode, 0.0, None, 0.0, "observe_only"
        if self.next_head is None or self.next_weight <= 0:
            return self.base_head_mode, 0.0, None, 0.0, self.next_reason or "no_pulse"
        head = self.next_head
        return format_ref_head_mode_from_head(head), self.next_weight, head, self.next_spike_score, self.next_reason

    def update_after_validation(
        self,
        epoch: int,
        val_metrics: Optional[Dict[str, object]],
        applied_head: Optional[Tuple[int, int]],
        applied_weight: float,
        applied_spike_score: float,
        local_rank: int,
    ) -> None:
        if not self.enabled or val_metrics is None:
            return
        top1 = float(val_metrics.get("top1", 0.0))
        top5 = float(val_metrics.get("top5", 0.0))
        samples = int(val_metrics.get("samples", 0) or 0)
        previous_best = self.rolling_best
        if self.rolling_best is None or top1 > self.rolling_best:
            self.rolling_best = top1
        rolling_best_for_drop = previous_best if previous_best is not None else top1
        drop = max(0.0, float(rolling_best_for_drop) - top1)
        next_head, next_weight, next_spike_score, triggered, reason = self._choose_next(epoch, drop)
        self.next_head = next_head
        self.next_weight = next_weight
        self.next_spike_score = next_spike_score
        self.next_reason = reason
        if triggered and next_head is not None:
            self.cooldown_until[next_head] = epoch + self.cooldown_epochs + 1
            self.pulse_epochs.append(epoch + 1)

        if local_rank == 0:
            phase = "observe" if epoch <= self.observe_until_epoch or epoch < self.start_epoch else "dynamic"
            cooldown_state = json.dumps(
                {format_dynamic_kl_head(head): until for head, until in sorted(self.cooldown_until.items())},
                sort_keys=True,
            )
            window_pulses = len(self._recent_pulse_epochs(epoch))
            line = (
                f"{epoch}\t{phase}\t{top1:.4f}\t{top5:.4f}\t{samples}\t"
                f"{float(self.rolling_best):.4f}\t{drop:.4f}\t"
                f"{format_dynamic_kl_head(applied_head)}\t{float(applied_weight):.8g}\t{float(applied_spike_score):.6f}\t"
                f"{format_dynamic_kl_head(next_head)}\t{float(next_weight):.8g}\t{float(next_spike_score):.6f}\t"
                f"{int(triggered)}\t{reason}\t{self.prior_source}\t{cooldown_state}\t{window_pulses}\n"
            )
            with open(self.tsv_path, "a", encoding="utf-8") as handle:
                handle.write(line)
            print(
                "DynamicSparsePrevStepKLController: "
                f"epoch={epoch}, phase={phase}, top1={top1:.4f}, rolling_best={float(self.rolling_best):.4f}, "
                f"drop={drop:.4f}, applied_head={format_dynamic_kl_head(applied_head)}, "
                f"applied_weight={float(applied_weight):.3e}, applied_spike_score={float(applied_spike_score):.6f}, "
                f"next_head={format_dynamic_kl_head(next_head)}, next_weight={float(next_weight):.3e}, "
                f"next_spike_score={float(next_spike_score):.6f}, triggered={triggered}, reason={reason}, "
                f"prior_source={self.prior_source}, window_pulses={window_pulses}, cooldown={cooldown_state}"
            )

    def _choose_next(self, epoch: int, drop: float) -> Tuple[Optional[Tuple[int, int]], float, float, bool, str]:
        if epoch < self.start_epoch:
            return None, 0.0, 0.0, False, "observe_only_before_start"
        if drop < self.drop_threshold:
            return None, 0.0, 0.0, False, f"drop_below_threshold:{drop:.4f}<{self.drop_threshold:.4f}"
        if len(self._recent_pulse_epochs(epoch)) >= self.max_pulses_per_window:
            return None, 0.0, 0.0, False, f"window_limit:{self.max_pulses_per_window}/{self.window_epochs}"

        for head in self.primary_heads + self.secondary_heads:
            if head in self.avoid_heads:
                continue
            cooldown_until = int(self.cooldown_until.get(head, -1))
            if (epoch + 1) < cooldown_until:
                continue
            weight = self.strong_weight if drop >= self.strong_drop_threshold else self.default_weight
            weight = min(weight, self.max_weight)
            spike_score = self._offline_spike_score(head)
            reason = (
                "offline_prior_validation_drop:"
                f"head={format_dynamic_kl_head(head)},spike_score={spike_score:.6f},"
                f"drop={drop:.4f},threshold={self.drop_threshold:.4f}"
            )
            return head, weight, spike_score, True, reason
        return None, 0.0, 0.0, False, "no_candidate_after_avoid_or_cooldown"

    def _recent_pulse_epochs(self, epoch: int) -> List[int]:
        lower = epoch + 1 - self.window_epochs
        return [pulse_epoch for pulse_epoch in self.pulse_epochs if lower <= pulse_epoch <= epoch]

    def _offline_spike_score(self, head: Tuple[int, int]) -> float:
        candidates = list(self.primary_heads + self.secondary_heads)
        if head not in candidates:
            return 0.0
        if head in self.primary_heads:
            return 1.0
        secondary_count = max(1, len(self.secondary_heads))
        secondary_rank = self.secondary_heads.index(head)
        return max(0.1, 0.8 - 0.1 * secondary_rank / secondary_count)


def ref_head_map(head_mode: str) -> Optional[Dict[int, Tuple[int, ...]]]:
    if str(head_mode).startswith("dynamic_teacher_agree_top"):
        _, pool_mode, _, _ = dynamic_ref_head_mode(str(head_mode))
        return ref_head_map(pool_mode)
    if str(head_mode).startswith("dynamic_top"):
        return None
    if str(head_mode).startswith("dynamic_custom_layer_top"):
        _, pool_mode, _, _ = dynamic_ref_head_mode(str(head_mode))
        return ref_head_map(pool_mode)
    if str(head_mode).startswith("dynamic_ema_custom_top"):
        _, pool_mode, _, _ = dynamic_ref_head_mode(str(head_mode))
        return ref_head_map(pool_mode)
    if str(head_mode).startswith("dynamic_custom_top"):
        _, pool_mode, _, _ = dynamic_ref_head_mode(str(head_mode))
        return ref_head_map(pool_mode)
    selected_heads = parse_ref_head_mode(head_mode)
    if selected_heads is None:
        return None
    head_map: Dict[int, List[int]] = {}
    for layer_idx, head_idx in selected_heads:
        if head_idx is None:
            return None
        head_map.setdefault(int(layer_idx), []).append(int(head_idx))
    return {layer_idx: tuple(sorted(set(heads))) for layer_idx, heads in head_map.items()}


def dynamic_ref_head_mode(head_mode: str) -> Tuple[Optional[int], str, bool, bool]:
    mode = str(head_mode or "all")
    if mode.startswith("dynamic_teacher_agree_top"):
        top_part, _, pool_mode = mode.partition(":")
        if not pool_mode:
            raise ValueError(f"dynamic teacher-agree ref head mode requires a pool mode after ':', got {head_mode!r}")
        return int(top_part[len("dynamic_teacher_agree_top") :]), pool_mode, False, False
    if mode.startswith("dynamic_top"):
        return int(mode[len("dynamic_top") :]), "all", False, False
    if mode.startswith("dynamic_custom_layer_top"):
        top_part, _, pool_mode = mode.partition(":")
        if not pool_mode:
            raise ValueError(f"dynamic custom layer ref head mode requires a pool mode after ':', got {head_mode!r}")
        return int(top_part[len("dynamic_custom_layer_top") :]), pool_mode, True, False
    if mode.startswith("dynamic_ema_custom_top"):
        top_part, _, pool_mode = mode.partition(":")
        if not pool_mode:
            raise ValueError(f"dynamic EMA custom ref head mode requires a pool mode after ':', got {head_mode!r}")
        return int(top_part[len("dynamic_ema_custom_top") :]), pool_mode, False, True
    if mode.startswith("dynamic_custom_top"):
        top_part, _, pool_mode = mode.partition(":")
        if not pool_mode:
            raise ValueError(f"dynamic custom ref head mode requires a pool mode after ':', got {head_mode!r}")
        return int(top_part[len("dynamic_custom_top") :]), pool_mode, False, False
    return None, mode, False, False


def attention_kl_pair_loss(student_attn: torch.Tensor, ref_attn: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type in {"cosine", "centered_cosine"}:
        student_vec = student_attn.float().flatten(1)
        ref_vec = ref_attn.float().flatten(1)
        if loss_type == "centered_cosine":
            student_vec = student_vec - student_vec.mean(dim=1, keepdim=True)
            ref_vec = ref_vec - ref_vec.mean(dim=1, keepdim=True)
        return (1.0 - F.cosine_similarity(student_vec, ref_vec.detach(), dim=1, eps=1e-8)).mean()
    student_prob = student_attn.clamp_min(1e-8)
    ref_prob = ref_attn.clamp_min(1e-8)
    if loss_type == "kl_ref":
        return F.kl_div(torch.log(student_prob), ref_prob, reduction="batchmean")
    if loss_type == "symmetric_kl":
        student_to_ref = F.kl_div(torch.log(student_prob), ref_prob, reduction="batchmean")
        ref_to_student = F.kl_div(torch.log(ref_prob), student_prob, reduction="batchmean")
        return 0.5 * (student_to_ref + ref_to_student)
    if loss_type == "js":
        mixed_prob = 0.5 * (student_prob + ref_prob)
        student_js = F.kl_div(torch.log(student_prob), mixed_prob, reduction="batchmean")
        ref_js = F.kl_div(torch.log(ref_prob), mixed_prob, reduction="batchmean")
        return 0.5 * (student_js + ref_js)
    raise NotImplementedError(f"Unsupported ref attention loss: {loss_type}")


def maybe_clip_ref_loss(loss: torch.Tensor, clip_value: float = 0.0) -> torch.Tensor:
    if float(clip_value or 0.0) <= 0:
        return loss
    return torch.clamp(loss, max=float(clip_value))


_DYNAMIC_HEAD_SCORE_EMA: Dict[str, torch.Tensor] = {}


def attention_kl_consistency_loss(student_attn_info, ref_attn_info, head_mode: str = "all", loss_type: str = "kl_ref", clip_value: float = 0.0) -> torch.Tensor:
    student_list = extract_attn_prob_list(student_attn_info)
    ref_list = extract_attn_prob_list(ref_attn_info)
    first_tensor = next((x for x in student_list if torch.is_tensor(x)), None)
    if first_tensor is None:
        first_tensor = next((x for x in ref_list if torch.is_tensor(x)), None)
    if not student_list or not ref_list or first_tensor is None:
        if first_tensor is not None:
            return first_tensor.new_zeros(())
        return torch.zeros((), device="cuda")

    total = first_tensor.new_zeros(())
    count = 0
    dynamic_topk, static_head_mode, dynamic_layerwise, dynamic_ema = dynamic_ref_head_mode(head_mode)
    head_mode = static_head_mode
    selected_heads = parse_ref_head_mode(head_mode)

    selected_head_map = None
    if selected_heads is not None:
        selected_head_map = ref_head_map(head_mode)
        if selected_head_map is not None:
            selected_heads = tuple((layer_idx, compact_idx) for layer_idx, heads in selected_head_map.items() for compact_idx, _ in enumerate(heads))

    if selected_heads is None:
        selected_items = [
            (layer_idx, None)
            for layer_idx in range(min(len(student_list), len(ref_list)))
        ]
    else:
        selected_items = selected_heads

    if dynamic_topk is not None:
        head_losses = []
        head_keys = []
        layer_losses: Dict[int, List[torch.Tensor]] = {}
        for layer_idx, head_idx in selected_items:
            if layer_idx >= len(student_list) or layer_idx >= len(ref_list):
                continue
            student_attn = student_list[layer_idx]
            ref_attn = ref_list[layer_idx]
            if student_attn is None or ref_attn is None or student_attn.ndim < 4:
                continue
            if head_idx is None:
                max_heads = min(student_attn.shape[1], ref_attn.shape[1])
                head_range = range(max_heads)
            else:
                if head_idx >= student_attn.shape[1] or head_idx >= ref_attn.shape[1]:
                    continue
                head_range = (head_idx,)
            for current_head in head_range:
                student_head = student_attn[:, current_head : current_head + 1]
                ref_head = ref_attn[:, current_head : current_head + 1]
                head_loss = maybe_clip_ref_loss(attention_kl_pair_loss(student_head, ref_head, loss_type), clip_value)
                head_losses.append(head_loss)
                head_keys.append((int(layer_idx), int(current_head)))
                layer_losses.setdefault(layer_idx, []).append(head_loss)
        if dynamic_layerwise:
            selected_layer_losses = []
            for losses in layer_losses.values():
                if not losses:
                    continue
                stacked_layer_losses = torch.stack(losses)
                topk = min(max(1, int(dynamic_topk)), stacked_layer_losses.numel())
                selected_layer_losses.append(torch.topk(stacked_layer_losses, k=topk).values.mean())
            if not selected_layer_losses:
                return total
            return torch.stack(selected_layer_losses).mean()
        if not head_losses:
            return total
        stacked_losses = torch.stack(head_losses)
        topk = min(max(1, int(dynamic_topk)), stacked_losses.numel())
        if dynamic_ema:
            ema_key = f"{head_mode}|{loss_type}|{clip_value}"
            current_scores = stacked_losses.detach()
            previous_scores = _DYNAMIC_HEAD_SCORE_EMA.get(ema_key)
            if previous_scores is None or previous_scores.shape != current_scores.shape or previous_scores.device != current_scores.device:
                smoothed_scores = current_scores
            else:
                smoothed_scores = previous_scores.mul(0.9).add(current_scores, alpha=0.1)
            _DYNAMIC_HEAD_SCORE_EMA[ema_key] = smoothed_scores.detach()
            selected_idx = torch.topk(smoothed_scores, k=topk).indices
            return stacked_losses.index_select(0, selected_idx).mean()
        return torch.topk(stacked_losses, k=topk).values.mean()

    for layer_idx, head_idx in selected_items:
        if layer_idx >= len(student_list) or layer_idx >= len(ref_list):
            continue
        student_attn = student_list[layer_idx]
        ref_attn = ref_list[layer_idx]
        if student_attn is None or ref_attn is None:
            continue
        if head_idx is not None:
            if student_attn.ndim < 4 or head_idx >= student_attn.shape[1] or head_idx >= ref_attn.shape[1]:
                continue
            student_attn = student_attn[:, head_idx : head_idx + 1]
            ref_attn = ref_attn[:, head_idx : head_idx + 1]
        total = total + maybe_clip_ref_loss(attention_kl_pair_loss(student_attn, ref_attn, loss_type), clip_value)
        count += 1
    return total / max(count, 1)


def attention_teacher_agree_consistency_loss(student_attn_info, ref_attn_info, teacher_attn_info, head_mode: str, loss_type: str = "kl_ref", clip_value: float = 0.0) -> torch.Tensor:
    mode = str(head_mode or "")
    if not mode.startswith("dynamic_teacher_agree_top"):
        return attention_kl_consistency_loss(student_attn_info, ref_attn_info, head_mode=head_mode, loss_type=loss_type, clip_value=clip_value)
    dynamic_topk, static_head_mode, _, _ = dynamic_ref_head_mode(mode)
    if dynamic_topk is None:
        return attention_kl_consistency_loss(student_attn_info, ref_attn_info, head_mode=head_mode, loss_type=loss_type, clip_value=clip_value)

    student_list = extract_attn_prob_list(student_attn_info)
    ref_list = extract_attn_prob_list(ref_attn_info)
    teacher_list = extract_attn_prob_list(teacher_attn_info)
    first_tensor = next((x for x in student_list if torch.is_tensor(x)), None)
    if first_tensor is None:
        return torch.zeros((), device="cuda")

    selected_heads = parse_ref_head_mode(static_head_mode)
    selected_head_map = ref_head_map(static_head_mode) if selected_heads is not None else None
    if selected_heads is not None and selected_head_map is not None:
        selected_items = tuple((layer_idx, compact_idx) for layer_idx, heads in selected_head_map.items() for compact_idx, _ in enumerate(heads))
    elif selected_heads is None:
        selected_items = tuple((layer_idx, None) for layer_idx in range(min(len(student_list), len(ref_list), len(teacher_list))))
    else:
        selected_items = selected_heads

    candidates = []
    for layer_idx, head_idx in selected_items:
        if layer_idx >= len(student_list) or layer_idx >= len(ref_list) or layer_idx >= len(teacher_list):
            continue
        student_attn = student_list[layer_idx]
        ref_attn = ref_list[layer_idx]
        teacher_attn = teacher_list[layer_idx]
        if student_attn is None or ref_attn is None or teacher_attn is None or student_attn.ndim < 4:
            continue
        if head_idx is None:
            head_range = range(min(student_attn.shape[1], ref_attn.shape[1], teacher_attn.shape[1]))
        else:
            if head_idx >= student_attn.shape[1] or head_idx >= ref_attn.shape[1] or head_idx >= teacher_attn.shape[1]:
                continue
            head_range = (head_idx,)
        for current_head in head_range:
            student_head = student_attn[:, current_head : current_head + 1]
            ref_head = ref_attn[:, current_head : current_head + 1]
            teacher_head = teacher_attn[:, current_head : current_head + 1]
            ref_loss = maybe_clip_ref_loss(attention_kl_pair_loss(student_head, ref_head, loss_type), clip_value)
            teacher_loss = attention_kl_pair_loss(student_head, teacher_head, loss_type).detach()
            candidates.append((ref_loss, teacher_loss))

    if not candidates:
        return first_tensor.new_zeros(())
    ref_losses = torch.stack([item[0] for item in candidates])
    teacher_losses = torch.stack([item[1] for item in candidates])
    shortlist_k = min(max(1, int(dynamic_topk) * 2), ref_losses.numel())
    shortlist_idx = torch.topk(ref_losses.detach(), k=shortlist_k).indices
    shortlist_teacher = teacher_losses[shortlist_idx]
    topk = min(max(1, int(dynamic_topk)), shortlist_idx.numel())
    selected_idx = shortlist_idx[torch.topk(-shortlist_teacher, k=topk).indices]
    return ref_losses[selected_idx].mean()


def logits_kl_consistency_loss(student_logits: torch.Tensor, ref_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    student_log_prob = F.log_softmax(student_logits / temp, dim=-1)
    ref_prob = F.softmax(ref_logits / temp, dim=-1)
    return F.kl_div(student_log_prob, ref_prob, reduction="batchmean") * (temp * temp)


def teacher_confidence_weighted_soft_kd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, power: float = 1.0, temperature: float = 1.0) -> torch.Tensor:
    student_logits = student_logits[0] if isinstance(student_logits, tuple) else student_logits
    teacher_logits = teacher_logits[0] if isinstance(teacher_logits, tuple) else teacher_logits
    temp = max(float(temperature), 1e-6)
    teacher_prob = F.softmax(teacher_logits / temp, dim=1)
    student_log_prob = F.log_softmax(student_logits / temp, dim=1)
    per_sample_loss = -(teacher_prob * student_log_prob).sum(dim=1) * (temp * temp)
    confidence = teacher_prob.max(dim=1).values.detach().clamp_min(1e-6)
    weights = confidence.pow(float(power))
    weights = weights / weights.mean().clamp_min(1e-6)
    return (per_sample_loss * weights).mean()


def teacher_soft_kd_with_temperature(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return teacher_confidence_weighted_soft_kd(
        student_logits,
        teacher_logits,
        power=0.0,
        temperature=temperature,
    )


def teacher_confidence_band_soft_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    low: float,
    high: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    student_logits = student_logits[0] if isinstance(student_logits, tuple) else student_logits
    teacher_logits = teacher_logits[0] if isinstance(teacher_logits, tuple) else teacher_logits
    temp = max(float(temperature), 1e-6)
    teacher_prob = F.softmax(teacher_logits / temp, dim=1)
    student_log_prob = F.log_softmax(student_logits / temp, dim=1)
    per_sample_loss = -(teacher_prob * student_log_prob).sum(dim=1) * (temp * temp)
    teacher_conf = teacher_prob.max(dim=1).values.detach()
    mask = teacher_conf.ge(float(low)) & teacher_conf.lt(float(high))
    if not bool(mask.any()):
        return per_sample_loss.mean() * 0.0
    return per_sample_loss[mask].mean()


def reference_confidence_band_soft_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    low: float,
    high: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    student_logits = student_logits[0] if isinstance(student_logits, tuple) else student_logits
    teacher_logits = teacher_logits[0] if isinstance(teacher_logits, tuple) else teacher_logits
    ref_logits = ref_logits[0] if isinstance(ref_logits, tuple) else ref_logits
    temp = max(float(temperature), 1e-6)
    teacher_prob = F.softmax(teacher_logits / temp, dim=1)
    student_log_prob = F.log_softmax(student_logits / temp, dim=1)
    per_sample_loss = -(teacher_prob * student_log_prob).sum(dim=1) * (temp * temp)
    ref_conf = F.softmax(ref_logits.float(), dim=1).max(dim=1).values.detach()
    mask = ref_conf.ge(float(low)) & ref_conf.lt(float(high))
    if not bool(mask.any()):
        return per_sample_loss.mean() * 0.0
    return per_sample_loss[mask].mean()


def local_reference_confidence_band_soft_kd(
    student_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    low: float,
    high: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    student_logits = student_logits[0] if isinstance(student_logits, tuple) else student_logits
    ref_logits = ref_logits[0] if isinstance(ref_logits, tuple) else ref_logits
    temp = max(float(temperature), 1e-6)
    ref_prob = F.softmax(ref_logits / temp, dim=1)
    student_log_prob = F.log_softmax(student_logits / temp, dim=1)
    per_sample_loss = -(ref_prob * student_log_prob).sum(dim=1) * (temp * temp)
    ref_conf = F.softmax(ref_logits.float(), dim=1).max(dim=1).values.detach()
    mask = ref_conf.ge(float(low)) & ref_conf.lt(float(high))
    if not bool(mask.any()):
        return per_sample_loss.mean() * 0.0
    return per_sample_loss[mask].mean()


def class_protect_ref_kl_loss(
    student_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    target: torch.Tensor,
    classes: Sequence[int],
    temperature: float = 2.0,
) -> torch.Tensor:
    student_logits = student_logits[0] if isinstance(student_logits, tuple) else student_logits
    ref_logits = ref_logits[0] if isinstance(ref_logits, tuple) else ref_logits
    if not classes:
        return student_logits.float().sum() * 0.0
    class_tensor = torch.as_tensor(tuple(int(item) for item in classes), device=target.device, dtype=target.dtype)
    mask = target.unsqueeze(1).eq(class_tensor.unsqueeze(0)).any(dim=1)
    if not bool(mask.any()):
        return student_logits.float().sum() * 0.0
    return logits_kl_consistency_loss(
        student_logits[mask],
        ref_logits[mask],
        temperature=temperature,
    )


def teacher_qk_relation_loss(student_attn_info, teacher_attn_info) -> torch.Tensor:
    if student_attn_info is None or teacher_attn_info is None:
        return torch.zeros((), device="cuda")
    total = None
    count = 0
    for student_layer, teacher_layer in zip(student_attn_info, teacher_attn_info):
        if not isinstance(student_layer, (tuple, list)) or not isinstance(teacher_layer, (tuple, list)):
            continue
        if len(student_layer) < 3 or len(teacher_layer) < 3:
            continue
        for idx in (1, 2):
            student_tensor = student_layer[idx]
            teacher_tensor = teacher_layer[idx]
            if not torch.is_tensor(student_tensor) or not torch.is_tensor(teacher_tensor):
                continue
            student_rel = F.normalize(student_tensor.flatten(1).float(), dim=1)
            teacher_rel = F.normalize(teacher_tensor.flatten(1).float(), dim=1)
            layer_loss = F.mse_loss(student_rel, teacher_rel)
            total = layer_loss if total is None else total + layer_loss
            count += 1
    if total is None:
        if isinstance(student_attn_info, (tuple, list)) and student_attn_info and isinstance(student_attn_info[0], (tuple, list)):
            first_tensor = next((item for item in student_attn_info[0] if torch.is_tensor(item)), None)
            if first_tensor is not None:
                return first_tensor.new_zeros(())
        return torch.zeros((), device="cuda")
    return total / max(count, 1)


def teacher_qkv_relation_loss(
    student_attn_info,
    teacher_attn_info,
    layer_indices: Optional[Tuple[int, ...]] = None,
    components: str = "q,k,v",
) -> torch.Tensor:
    if student_attn_info is None or teacher_attn_info is None:
        return torch.zeros((), device="cuda")
    component_to_index = {"q": 1, "k": 2, "v": 3}
    component_indices = tuple(
        component_to_index[item.strip().lower()]
        for item in str(components or "q,k,v").split(",")
        if item.strip().lower() in component_to_index
    )
    if not component_indices:
        component_indices = (1, 2, 3)
    layer_set = set(layer_indices) if layer_indices is not None else None
    total = None
    count = 0
    first_tensor = None
    for layer_idx, (student_layer, teacher_layer) in enumerate(zip(student_attn_info, teacher_attn_info)):
        if layer_set is not None and layer_idx not in layer_set:
            continue
        if not isinstance(student_layer, (tuple, list)) or not isinstance(teacher_layer, (tuple, list)):
            continue
        if len(student_layer) < 4 or len(teacher_layer) < 4:
            continue
        for idx in component_indices:
            student_tensor = student_layer[idx]
            teacher_tensor = teacher_layer[idx]
            if not torch.is_tensor(student_tensor) or not torch.is_tensor(teacher_tensor):
                continue
            first_tensor = student_tensor
            student_vec = student_tensor.float().flatten(2)
            teacher_vec = teacher_tensor.detach().float().flatten(2)
            if student_vec.shape != teacher_vec.shape:
                continue
            student_vec = F.normalize(student_vec, dim=-1)
            teacher_vec = F.normalize(teacher_vec, dim=-1)
            layer_loss = F.mse_loss(student_vec, teacher_vec)
            total = layer_loss if total is None else total + layer_loss
            count += 1
    if total is None:
        if first_tensor is not None:
            return first_tensor.new_zeros(())
        if isinstance(student_attn_info, (tuple, list)) and student_attn_info and isinstance(student_attn_info[0], (tuple, list)):
            first_tensor = next((item for item in student_attn_info[0] if torch.is_tensor(item)), None)
            if first_tensor is not None:
                return first_tensor.new_zeros(())
        return torch.zeros((), device="cuda")
    return total / max(count, 1)


def attention_output_mse_loss(student_outputs: Sequence[torch.Tensor], teacher_outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    total = None
    count = 0
    for student_output, teacher_output in zip(student_outputs, teacher_outputs):
        if not torch.is_tensor(student_output) or not torch.is_tensor(teacher_output):
            continue
        if student_output.shape != teacher_output.shape:
            continue
        layer_loss = F.mse_loss(student_output.float(), teacher_output.detach().float())
        total = layer_loss if total is None else total + layer_loss
        count += 1
    if total is None:
        if student_outputs:
            return student_outputs[0].new_zeros(())
        return torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    return total / max(count, 1)


def attention_output_normalized_mse_loss(student_outputs: Sequence[torch.Tensor], teacher_outputs: Sequence[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    total = None
    count = 0
    for student_output, teacher_output in zip(student_outputs, teacher_outputs):
        if not torch.is_tensor(student_output) or not torch.is_tensor(teacher_output):
            continue
        if student_output.shape != teacher_output.shape:
            continue
        student_float = student_output.float()
        teacher_float = teacher_output.detach().float()
        mse = F.mse_loss(student_float, teacher_float)
        scale = teacher_float.pow(2).mean().clamp_min(eps)
        layer_loss = mse / scale
        total = layer_loss if total is None else total + layer_loss
        count += 1
    if total is None:
        if student_outputs:
            return student_outputs[0].new_zeros(())
        return torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    return total / max(count, 1)


def attention_output_confidence_weighted_normalized_mse_loss(
    student_outputs: Sequence[torch.Tensor],
    teacher_outputs: Sequence[torch.Tensor],
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    total = None
    count = 0
    weights = weights.detach().float()
    weights = weights / weights.mean().clamp_min(eps)
    for student_output, teacher_output in zip(student_outputs, teacher_outputs):
        if not torch.is_tensor(student_output) or not torch.is_tensor(teacher_output):
            continue
        if student_output.shape != teacher_output.shape or student_output.shape[0] != weights.shape[0]:
            continue
        student_float = student_output.float()
        teacher_float = teacher_output.detach().float()
        reduce_dims = tuple(range(1, student_float.ndim))
        per_sample_mse = (student_float - teacher_float).pow(2).mean(dim=reduce_dims)
        per_sample_scale = teacher_float.pow(2).mean(dim=reduce_dims).clamp_min(eps)
        layer_loss = (weights * (per_sample_mse / per_sample_scale)).mean()
        total = layer_loss if total is None else total + layer_loss
        count += 1
    if total is None:
        if student_outputs:
            return student_outputs[0].new_zeros(())
        return torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    return total / max(count, 1)


def describe_output_pairs(student_outputs: Sequence[torch.Tensor], teacher_outputs: Sequence[torch.Tensor]) -> str:
    parts = []
    for idx, (student_output, teacher_output) in enumerate(zip(student_outputs, teacher_outputs)):
        if not torch.is_tensor(student_output) or not torch.is_tensor(teacher_output):
            parts.append(f"{idx}:non_tensor")
            continue
        same_shape = student_output.shape == teacher_output.shape
        if same_shape:
            with torch.no_grad():
                mse = F.mse_loss(student_output.detach().float(), teacher_output.detach().float()).item()
            parts.append(f"{idx}:s={tuple(student_output.shape)} t={tuple(teacher_output.shape)} mse={mse:.3e}")
        else:
            parts.append(f"{idx}:s={tuple(student_output.shape)} t={tuple(teacher_output.shape)} mismatch")
    if len(student_outputs) != len(teacher_outputs):
        parts.append(f"count_mismatch student={len(student_outputs)} teacher={len(teacher_outputs)}")
    return "; ".join(parts) if parts else "no_pairs"


def setup_alpha(model: nn.Module, loader, runtime_args: SimpleNamespace, amp_autocast):
    model.eval()
    setup_batches = int(getattr(runtime_args, "setup_alpha_batches", 1))
    if runtime_args.local_rank == 0:
        print(f"setup alpha batches={setup_batches}")
    if setup_batches <= 0:
        if runtime_args.local_rank == 0:
            print("setup alpha skipped")
        return
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if not runtime_args.prefetcher:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            if runtime_args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                model(input)
            if batch_idx + 1 >= setup_batches:
                break


def recalibrate_lsq_alpha_preserve_params(model: nn.Module, loader, runtime_args: SimpleNamespace, amp_autocast, batches: int) -> int:
    root = maybe_unwrap_ddp(model)
    quantizers = []
    for module in root.modules():
        for attr in ("lsqw_fn", "input_quant_fn", "quant_x_4_qkv", "quan_a_qkx_fn"):
            quantizer = getattr(module, attr, None)
            if quantizer is not None and hasattr(quantizer, "initialized_alpha") and hasattr(quantizer, "s"):
                quantizers.append(quantizer)
    if runtime_args.local_rank == 0:
        print(f"progressive bit recalibrate alpha batches={batches} quantizers={len(quantizers)}")
    if batches <= 0 or not quantizers:
        return len(quantizers)
    old_params = {id(quantizer): getattr(quantizer, "s", None) for quantizer in quantizers}
    for quantizer in quantizers:
        quantizer.initialized_alpha = False
    old_setup_batches = getattr(runtime_args, "setup_alpha_batches", 1)
    runtime_args.setup_alpha_batches = int(batches)
    setup_alpha(model, loader, runtime_args, amp_autocast)
    runtime_args.setup_alpha_batches = old_setup_batches
    for quantizer in quantizers:
        old_param = old_params[id(quantizer)]
        new_param = getattr(quantizer, "s", None)
        if old_param is not None and new_param is not None:
            with torch.no_grad():
                old_param.data.copy_(new_param.data)
            quantizer.s = old_param
            quantizer.initialized_alpha = True
    return len(quantizers)


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


def run_pre_qat_reconstruction(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    runtime_args: SimpleNamespace,
    amp_autocast,
    teacher: Optional[nn.Module],
) -> None:
    updates = max(0, int(getattr(runtime_args, "pre_qat_recon_updates", 0)))
    if updates <= 0:
        return
    if teacher is None:
        raise ValueError("pre-QAT reconstruction requires --use-kd teacher")
    model.train()
    teacher.eval()
    if runtime_args.local_rank == 0:
        print(
            "Starting pre-QAT teacher-logit reconstruction: "
            f"updates={updates}, policy=quant, temperature={runtime_args.pre_qat_recon_temperature}"
        )
    completed = 0
    optimizer.zero_grad(set_to_none=True)
    while completed < updates:
        for input, target in loader:
            if completed >= updates:
                break
            if not runtime_args.prefetcher:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            if runtime_args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            with torch.no_grad(), amp_autocast():
                teacher_output = teacher(input)
                teacher_logit = teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output
            with amp_autocast():
                student_output = model(input)
                student_logit = student_output[0] if isinstance(student_output, tuple) else student_output
                loss = teacher_soft_kd_with_temperature(
                    student_logit,
                    teacher_logit,
                    temperature=runtime_args.pre_qat_recon_temperature,
                )
            loss.backward()
            apply_gradient_mask_policy(model, "quant")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed += 1
            if runtime_args.local_rank == 0 and (completed == 1 or completed % 50 == 0 or completed == updates):
                print(f"PreQATRecon: update={completed}/{updates} loss={loss.detach().float().item():.6f}")
        if len(loader) == 0:
            break
    if runtime_args.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if runtime_args.local_rank == 0:
        print(f"Finished pre-QAT teacher-logit reconstruction: updates={completed}")


def run_pre_qat_feature_reconstruction(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    runtime_args: SimpleNamespace,
    amp_autocast,
    teacher: Optional[nn.Module],
    updates_override: Optional[int] = None,
    label: str = "pre-QAT feature reconstruction",
    bypass_ddp: bool = False,
    anchor_model: Optional[nn.Module] = None,
    anchor_kl_weight: float = 0.0,
    anchor_kl_temperature: float = 2.75,
) -> None:
    updates = max(0, int(getattr(runtime_args, "pre_qat_feature_recon_updates", 0) if updates_override is None else updates_override))
    if updates <= 0:
        return
    if teacher is None:
        raise ValueError(f"{label} requires --use-kd teacher")
    feature_layers = parse_name_list(getattr(runtime_args, "pre_qat_feature_recon_layers", ""))
    if not feature_layers:
        raise ValueError("pre-QAT feature reconstruction requires --pre-qat-feature-recon-layers")
    policy = str(getattr(runtime_args, "pre_qat_feature_recon_policy", "quant"))
    confidence_power = float(getattr(runtime_args, "pre_qat_feature_recon_confidence_power", 0.0))
    weight_mode = str(getattr(runtime_args, "pre_qat_feature_recon_weight_mode", "none"))
    qdrop_prob = float(getattr(runtime_args, "pre_qat_feature_recon_qdrop_prob", 0.0))
    qdrop_layers = parse_name_list(getattr(runtime_args, "pre_qat_feature_recon_qdrop_layers", ""))
    anchor_kl_weight = float(anchor_kl_weight)
    active_model = maybe_unwrap_ddp(model) if bypass_ddp else model
    active_model.train()
    teacher.eval()
    if anchor_model is not None:
        anchor_model.eval()
    if runtime_args.local_rank == 0:
        print(
            f"Starting {label}: "
            f"updates={updates}, policy={policy}, weight_mode={weight_mode}, confidence_power={confidence_power}, "
            f"qdrop_prob={qdrop_prob}, qdrop_layers={qdrop_layers}, bypass_ddp={bypass_ddp}, "
            f"anchor_kl_weight={anchor_kl_weight}, anchor_kl_temperature={anchor_kl_temperature}, "
            f"layers={feature_layers}, student_matches={matched_named_modules(active_model, feature_layers)}, "
            f"teacher_matches={matched_named_modules(teacher, feature_layers)}"
        )
    completed = 0
    optimizer.zero_grad(set_to_none=True)
    while completed < updates:
        for input, target in loader:
            if completed >= updates:
                break
            if not runtime_args.prefetcher:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            if runtime_args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            student_features = []
            teacher_features = []
            with torch.no_grad(), capture_named_module_outputs(teacher, feature_layers, detach=True) as captured_teacher, amp_autocast():
                teacher_output = teacher(input)
                teacher_features = list(captured_teacher)
            anchor_output = None
            if anchor_model is not None and anchor_kl_weight > 0:
                with torch.no_grad(), amp_autocast():
                    anchor_output = anchor_model(input)
            with stochastic_activation_quant_bypass(active_model, qdrop_prob, qdrop_layers) as qdrop_modules:
                if runtime_args.local_rank == 0 and completed == 0 and qdrop_prob > 0:
                    print(f"PreQATFeatRecon QDrop: prob={qdrop_prob}, layers={qdrop_layers}, activation_quantizers={qdrop_modules}")
                with capture_named_module_outputs(active_model, feature_layers, detach=False) as captured_student, amp_autocast():
                    student_output = active_model(input)
                    student_features = list(captured_student)
            if len(student_features) != len(teacher_features) or not student_features:
                raise RuntimeError(
                    "pre-QAT feature reconstruction captured mismatched features: "
                    f"student={len(student_features)} teacher={len(teacher_features)}"
                )
            if weight_mode == "confidence" or (weight_mode == "none" and confidence_power > 0):
                teacher_logit = teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output
                power = confidence_power if confidence_power > 0 else 1.0
                teacher_confidence = F.softmax(teacher_logit.float(), dim=1).max(dim=1).values.pow(power)
                loss = attention_output_confidence_weighted_normalized_mse_loss(student_features, teacher_features, teacher_confidence)
            elif weight_mode == "disagreement":
                teacher_logit = teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output
                student_logit = student_output[0] if isinstance(student_output, tuple) else student_output
                teacher_prob = F.softmax(teacher_logit.detach().float(), dim=1)
                student_log_prob = F.log_softmax(student_logit.float(), dim=1)
                disagreement = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1).clamp_min(1e-8)
                loss = attention_output_confidence_weighted_normalized_mse_loss(student_features, teacher_features, disagreement)
            else:
                loss = attention_output_normalized_mse_loss(student_features, teacher_features)
            if anchor_output is not None:
                student_logit = student_output[0] if isinstance(student_output, tuple) else student_output
                anchor_logit = anchor_output[0] if isinstance(anchor_output, tuple) else anchor_output
                loss = loss + anchor_kl_weight * logits_kl_consistency_loss(
                    student_logit,
                    anchor_logit,
                    temperature=anchor_kl_temperature,
                )
            loss.backward()
            kept, masked = apply_feature_recon_gradient_mask(active_model, feature_layers, policy)
            reduced = all_reduce_parameter_grads(active_model) if bypass_ddp else 0
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed += 1
            if runtime_args.local_rank == 0 and (completed == 1 or completed % 50 == 0 or completed == updates):
                print(
                    f"{label}: update={completed}/{updates} "
                    f"loss={loss.detach().float().item():.6f} kept={kept} masked={masked} reduced={reduced}"
                )
        if len(loader) == 0:
            break
    if runtime_args.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if runtime_args.local_rank == 0:
        print(f"Finished {label}: updates={completed}")


def run_progressive_bit_transition_reconstruction(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    runtime_args: SimpleNamespace,
    amp_autocast,
    teacher: Optional[nn.Module],
    epoch: int,
    previous_bits: Tuple[int, int],
    current_bits: Tuple[int, int],
    anchor_model: Optional[nn.Module] = None,
) -> None:
    updates = max(0, int(getattr(runtime_args, "progressive_bit_transition_recon_updates", 0)))
    if updates <= 0:
        return
    forced_epochs = set(getattr(runtime_args, "progressive_bit_transition_recon_epochs", set()) or set())
    if forced_epochs and epoch not in forced_epochs:
        return
    previous_wbits, previous_abits = previous_bits
    current_wbits, current_abits = current_bits
    is_down_transition = current_wbits < previous_wbits or current_abits < previous_abits
    if not is_down_transition and not forced_epochs:
        return
    layers = str(getattr(runtime_args, "progressive_bit_transition_recon_layers", ""))
    if not layers:
        raise ValueError("progressive bit transition reconstruction requires --progressive-bit-transition-recon-layers")
    saved = {
        "pre_qat_feature_recon_layers": getattr(runtime_args, "pre_qat_feature_recon_layers", ""),
        "pre_qat_feature_recon_policy": getattr(runtime_args, "pre_qat_feature_recon_policy", "quant"),
        "pre_qat_feature_recon_confidence_power": getattr(runtime_args, "pre_qat_feature_recon_confidence_power", 0.0),
        "pre_qat_feature_recon_weight_mode": getattr(runtime_args, "pre_qat_feature_recon_weight_mode", "none"),
        "pre_qat_feature_recon_qdrop_prob": getattr(runtime_args, "pre_qat_feature_recon_qdrop_prob", 0.0),
        "pre_qat_feature_recon_qdrop_layers": getattr(runtime_args, "pre_qat_feature_recon_qdrop_layers", ""),
    }
    runtime_args.pre_qat_feature_recon_layers = layers
    runtime_args.pre_qat_feature_recon_policy = str(getattr(runtime_args, "progressive_bit_transition_recon_policy", "module_all"))
    runtime_args.pre_qat_feature_recon_confidence_power = float(
        getattr(runtime_args, "progressive_bit_transition_recon_confidence_power", 0.0)
    )
    runtime_args.pre_qat_feature_recon_weight_mode = str(getattr(runtime_args, "progressive_bit_transition_recon_weight_mode", "none"))
    runtime_args.pre_qat_feature_recon_qdrop_prob = float(getattr(runtime_args, "progressive_bit_transition_recon_qdrop_prob", 0.0))
    runtime_args.pre_qat_feature_recon_qdrop_layers = str(getattr(runtime_args, "progressive_bit_transition_recon_qdrop_layers", ""))
    if runtime_args.local_rank == 0:
        print(
            "Starting progressive bit transition reconstruction: "
            f"epoch={epoch}, from=W{previous_wbits}A{previous_abits}, to=W{current_wbits}A{current_abits}, "
            f"updates={updates}, layers={parse_name_list(layers)}, "
            f"policy={runtime_args.pre_qat_feature_recon_policy}, "
            f"weight_mode={runtime_args.pre_qat_feature_recon_weight_mode}, "
            f"qdrop_prob={runtime_args.pre_qat_feature_recon_qdrop_prob}, "
            f"anchor_kl_weight={getattr(runtime_args, 'progressive_bit_transition_anchor_kl_weight', 0.0)}, "
            f"anchor_kl_temperature={getattr(runtime_args, 'progressive_bit_transition_anchor_kl_temperature', 2.75)}"
        )
    try:
        run_pre_qat_feature_reconstruction(
            model,
            loader,
            optimizer,
            runtime_args,
            amp_autocast,
            teacher,
            updates_override=updates,
            label=f"progressive bit transition feature reconstruction epoch={epoch}",
            bypass_ddp=True,
            anchor_model=anchor_model,
            anchor_kl_weight=float(getattr(runtime_args, "progressive_bit_transition_anchor_kl_weight", 0.0)),
            anchor_kl_temperature=float(getattr(runtime_args, "progressive_bit_transition_anchor_kl_temperature", 2.75)),
        )
    finally:
        for name, value in saved.items():
            setattr(runtime_args, name, value)


def apply_named_quant_gradient_mask(model: nn.Module, module_name: str) -> Tuple[int, int]:
    root = maybe_unwrap_ddp(model)
    kept = 0
    masked = 0
    for name, param in root.named_parameters():
        if param.grad is None:
            continue
        belongs_to_module = name == module_name or name.startswith(f"{module_name}.") or name.endswith(f".{module_name}") or f".{module_name}." in name
        keep = belongs_to_module and is_quant_or_shift_parameter(name)
        if keep:
            kept += param.numel()
        else:
            param.grad = None
            masked += param.numel()
    return kept, masked


def parameter_belongs_to_any_module(name: str, module_names: Sequence[str]) -> bool:
    for module_name in module_names:
        if name == module_name or name.startswith(f"{module_name}.") or name.endswith(f".{module_name}") or f".{module_name}." in name:
            return True
    return False


def apply_feature_recon_gradient_mask(model: nn.Module, module_names: Sequence[str], policy: str) -> Tuple[int, int]:
    root = maybe_unwrap_ddp(model)
    kept = 0
    masked = 0
    for name, param in root.named_parameters():
        if param.grad is None:
            continue
        in_module = parameter_belongs_to_any_module(name, module_names)
        if policy == "quant":
            keep = in_module and is_quant_or_shift_parameter(name)
        elif policy == "module_all":
            keep = in_module
        else:
            raise ValueError(f"Unsupported feature recon policy: {policy}")
        if keep:
            kept += param.numel()
        else:
            param.grad = None
            masked += param.numel()
    return kept, masked


def run_pre_qat_sequential_feature_reconstruction(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    runtime_args: SimpleNamespace,
    amp_autocast,
    teacher: Optional[nn.Module],
) -> None:
    updates_per_layer = max(0, int(getattr(runtime_args, "pre_qat_seq_feature_recon_updates", 0)))
    if updates_per_layer <= 0:
        return
    if teacher is None:
        raise ValueError("pre-QAT sequential feature reconstruction requires --use-kd teacher")
    feature_layers = parse_name_list(getattr(runtime_args, "pre_qat_seq_feature_recon_layers", ""))
    if not feature_layers:
        raise ValueError("pre-QAT sequential feature reconstruction requires --pre-qat-seq-feature-recon-layers")
    policy = str(getattr(runtime_args, "pre_qat_seq_feature_recon_policy", "quant"))
    model.train()
    teacher.eval()
    if runtime_args.local_rank == 0:
        print(
            "Starting pre-QAT sequential feature reconstruction: "
            f"updates_per_layer={updates_per_layer}, policy={policy}, layers={feature_layers}, "
            f"student_matches={matched_named_modules(model, feature_layers)}, "
            f"teacher_matches={matched_named_modules(teacher, feature_layers)}"
        )
    cached_batches = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if batch_idx >= updates_per_layer:
                break
            if not runtime_args.prefetcher:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            if runtime_args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            cached_batches.append(input.detach())
    for layer_name in feature_layers:
        completed = 0
        optimizer.zero_grad(set_to_none=True)
        for input in cached_batches:
            student_features = []
            teacher_features = []
            with torch.no_grad(), capture_named_module_outputs(teacher, (layer_name,), detach=True) as captured_teacher, amp_autocast():
                teacher(input)
                teacher_features = list(captured_teacher)
            with capture_named_module_outputs(model, (layer_name,), detach=False) as captured_student, amp_autocast():
                model(input)
                student_features = list(captured_student)
            if len(student_features) != 1 or len(teacher_features) != 1:
                raise RuntimeError(
                    "pre-QAT sequential feature reconstruction captured mismatched features: "
                    f"layer={layer_name} student={len(student_features)} teacher={len(teacher_features)}"
            )
            loss = attention_output_normalized_mse_loss(student_features, teacher_features)
            loss.backward()
            kept, masked = apply_feature_recon_gradient_mask(model, (layer_name,), policy)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed += 1
            if runtime_args.local_rank == 0 and (completed == 1 or completed % 25 == 0 or completed == updates_per_layer):
                print(
                    f"PreQATSeqFeatRecon: layer={layer_name} update={completed}/{updates_per_layer} "
                    f"loss={loss.detach().float().item():.6f} kept={kept} masked={masked}"
                )
    if runtime_args.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if runtime_args.local_rank == 0:
        print("Finished pre-QAT sequential feature reconstruction")


def create_ofq_optimizer(runtime_args: SimpleNamespace, model: nn.Module) -> torch.optim.Optimizer:
    if runtime_args.opt.lower() != "adamw":
        raise NotImplementedError(f"当前 unified OFQ path 仅支持 AdamW，收到: {runtime_args.opt}")
    quant_lr_multiplier = float(getattr(runtime_args, "quant_lr_multiplier", 1.0))
    if quant_lr_multiplier == 1.0:
        return torch.optim.AdamW(model.parameters(), lr=runtime_args.lr, weight_decay=runtime_args.weight_decay, betas=runtime_args.opt_betas, fused=True)

    quant_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_quant_or_shift_parameter(name):
            quant_params.append(param)
        else:
            base_params.append(param)
    param_groups = [
        {"params": base_params, "lr": runtime_args.lr, "lr_scale": 1.0},
        {"params": quant_params, "lr": runtime_args.lr * quant_lr_multiplier, "lr_scale": quant_lr_multiplier},
    ]
    if runtime_args.local_rank == 0:
        print(
            "Using grouped LR: "
            f"base_params={sum(p.numel() for p in base_params)}, "
            f"quant_params={sum(p.numel() for p in quant_params)}, "
            f"quant_lr_multiplier={quant_lr_multiplier}"
        )
    return torch.optim.AdamW(param_groups, lr=runtime_args.lr, weight_decay=runtime_args.weight_decay, betas=runtime_args.opt_betas, fused=True)


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
            param_group["lr"] = lr * float(param_group.get("lr_scale", 1.0))

    def state_dict(self) -> Dict[str, object]:
        return {
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_updates": self.warmup_updates,
            "total_updates": self.total_updates,
            "last_lr": [group.get("lr") for group in self.optimizer.param_groups],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.min_lr = float(state.get("min_lr", self.min_lr))
        self.warmup_updates = int(state.get("warmup_updates", self.warmup_updates))
        self.total_updates = int(state.get("total_updates", self.total_updates))
        last_lr = state.get("last_lr")
        if isinstance(last_lr, (list, tuple)):
            for param_group, lr in zip(self.optimizer.param_groups, last_lr):
                if lr is not None:
                    param_group["lr"] = float(lr)


def validate_ofq(model: nn.Module, loader, loss_fn, runtime_args: SimpleNamespace, amp_autocast):
    batch_time_m = AverageMeter()
    model.eval()
    if runtime_args.local_rank == 0:
        print("model eval")
    local_start_time = time.time()
    end = local_start_time
    last_idx = len(loader) - 1
    local_loss_sum = 0.0
    local_top1_correct = 0.0
    local_top5_correct = 0.0
    local_samples = 0.0
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
            batch_size = float(output.size(0))
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))
            local_loss_sum += float(loss.detach().item()) * batch_size
            local_top1_correct += float(correct[:1].reshape(-1).float().sum().item())
            local_top5_correct += float(correct[:5].reshape(-1).float().sum().item())
            local_samples += batch_size
            batch_time_m.update(time.time() - end)
            end = time.time()
            if runtime_args.local_rank == 0 and (last_batch or batch_idx % runtime_args.log_interval == 0):
                local_loss = local_loss_sum / max(local_samples, 1.0)
                local_top1 = 100.0 * local_top1_correct / max(local_samples, 1.0)
                local_top5 = 100.0 * local_top5_correct / max(local_samples, 1.0)
                print(
                    f"TestLocal: [{batch_idx:>4d}/{last_idx}]  Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                    f"Loss: {local_loss:>7.4f}  Acc@1: {local_top1:>7.4f}  "
                    f"Acc@5: {local_top5:>7.4f}  Samples: {int(local_samples)}"
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Validation can run immediately after large-batch training on nearly full GPUs.
        # Free cached training activations before tiny distributed metric reductions;
        # otherwise NCCL may fail allocating its communication buffers.
        torch.cuda.empty_cache()
    local_wall = time.time() - local_start_time
    device = torch.device(runtime_args.device if torch.cuda.is_available() else "cpu")
    stats = torch.tensor(
        [local_loss_sum, local_top1_correct, local_top5_correct, local_samples, local_wall],
        dtype=torch.float64,
        device=device,
    )
    if runtime_args.distributed:
        # Reduce all scalar validation statistics in one collective. Avoid a separate
        # gather for per-rank sample counts: it is not needed for raw Top-1/Top-5 and
        # has been observed to OOM with NCCL after 8xH100 large-batch QAT epochs.
        dist.all_reduce(stats[:4], op=dist.ReduceOp.SUM)
        wall_tensor = stats[4:5].clone()
        dist.all_reduce(wall_tensor, op=dist.ReduceOp.MAX)
        global_wall = float(wall_tensor.item())
    else:
        global_wall = local_wall

    global_samples = max(float(stats[3].item()), 1.0)
    metrics = {
        "loss": float(stats[0].item() / global_samples),
        "top1": float(100.0 * stats[1].item() / global_samples),
        "top5": float(100.0 * stats[2].item() / global_samples),
        "samples": int(stats[3].item()),
        "local_samples": int(local_samples),
        "wall_seconds": global_wall,
    }
    if runtime_args.local_rank == 0:
        print(
            f"Test: [distributed-summary]  Time: {global_wall:.3f}s  "
            f"Loss: {metrics['loss']:.4f}  Acc@1: {metrics['top1']:.4f}  "
            f"Acc@5: {metrics['top5']:.4f}  Samples: {metrics['samples']}"
        )
    return metrics


def compute_ofq_batch_loss(input, target, model: nn.Module, loss_fn, runtime_args: SimpleNamespace, amp_autocast, teacher: Optional[nn.Module], ref_model: Optional[nn.Module], anchor_ref_model: Optional[nn.Module], ref_attn_kl_weight: float, anchor_ref_attn_kl_weight: float):
    with amp_autocast():
        if runtime_args.model_type in {"deit", "swin"}:
            student_output = model(input)
            if isinstance(student_output, tuple):
                student_logit = student_output[0]
                student_attn_info = student_output[1] if len(student_output) > 1 else None
            else:
                student_logit = student_output
                student_attn_info = None
        else:
            student_logit = model(input)
            student_attn_info = None

        if runtime_args.use_kd:
            with torch.no_grad():
                teacher_output = teacher(input)
            if isinstance(teacher_output, tuple):
                teacher_logit = teacher_output[0]
                teacher_attn_info = teacher_output[1] if len(teacher_output) > 1 else None
            else:
                teacher_logit = teacher_output
                teacher_attn_info = None
            if runtime_args.kd_hard_and_soft == 0:
                if runtime_args.teacher_confidence_kd_power > 0:
                    loss = teacher_confidence_weighted_soft_kd(
                        student_logit,
                        teacher_logit,
                        power=runtime_args.teacher_confidence_kd_power,
                        temperature=runtime_args.teacher_soft_temperature,
                    )
                elif runtime_args.teacher_soft_temperature != 1.0:
                    loss = teacher_soft_kd_with_temperature(
                        student_logit,
                        teacher_logit,
                        temperature=runtime_args.teacher_soft_temperature,
                    )
                else:
                    loss = loss_fn(student_logit, teacher_logit)
            elif runtime_args.kd_hard_and_soft == 1:
                loss = loss_fn(student_logit, target, teacher_logit)
            elif runtime_args.kd_hard_and_soft == 2:
                loss = loss_fn(student_logit, student_attn_info, target, teacher_logit, teacher_attn_info)
            elif runtime_args.kd_hard_and_soft == 3:
                loss = loss_fn(student_logit, student_attn_info, target, teacher_logit, teacher_attn_info)
            else:
                raise NotImplementedError(f"Unsupported kd_hard_and_soft={runtime_args.kd_hard_and_soft}")
            if runtime_args.teacher_confidence_band_kd_weight > 0:
                loss = loss + runtime_args.teacher_confidence_band_kd_weight * teacher_confidence_band_soft_kd(
                    student_logit,
                    teacher_logit,
                    low=runtime_args.teacher_confidence_band_kd_low,
                    high=runtime_args.teacher_confidence_band_kd_high,
                    temperature=runtime_args.teacher_confidence_band_kd_temperature,
                )
            if runtime_args.clean_start_target_loss_weight > 0:
                student_logit_for_ce = student_logit[0] if isinstance(student_logit, tuple) else student_logit
                loss = loss + runtime_args.clean_start_target_loss_weight * F.cross_entropy(student_logit_for_ce, target)
        else:
            student_logit = student_logit[0] if isinstance(student_logit, tuple) else student_logit
            loss = loss_fn(student_logit, target)

        base_loss_for_log = loss.detach()
        ref_attn_kl_loss = loss.new_zeros(())
        ref_logit_kl_loss = loss.new_zeros(())
        anchor_ref_attn_kl_loss = loss.new_zeros(())
        teacher_attn_kl_loss = loss.new_zeros(())
        teacher_qk_rel_loss = loss.new_zeros(())
        use_ref_scheme = (
            runtime_args.train_scheme == "ema_ref_attn_kl"
            and ref_model is not None
            and (ref_attn_kl_weight > 0 or runtime_args.ref_logit_kl_weight > 0)
        )
        if use_ref_scheme:
            with torch.no_grad():
                ref_logits, ref_attn_info = ref_model(input)
            if ref_attn_kl_weight > 0:
                ref_attn_kl_loss = attention_kl_consistency_loss(
                    student_attn_info,
                    ref_attn_info,
                    head_mode=runtime_args.ref_head_mode,
                    loss_type=runtime_args.ref_attn_loss,
                    clip_value=runtime_args.ref_attn_kl_clip,
                )
                if runtime_args.ref_attn_kl_drop_prob < 1.0:
                    if runtime_args.ref_attn_kl_drop_prob <= 0.0:
                        ref_attn_kl_loss = ref_attn_kl_loss * 0.0
                    else:
                        kl_gate = (torch.rand((), device=ref_attn_kl_loss.device) < runtime_args.ref_attn_kl_drop_prob).to(ref_attn_kl_loss.dtype)
                        if runtime_args.ref_attn_kl_drop_scale:
                            kl_gate = kl_gate / runtime_args.ref_attn_kl_drop_prob
                        ref_attn_kl_loss = ref_attn_kl_loss * kl_gate
                loss = loss + ref_attn_kl_weight * ref_attn_kl_loss
            if runtime_args.ref_logit_kl_weight > 0:
                ref_logit_kl_loss = logits_kl_consistency_loss(
                    student_logit,
                    ref_logits,
                    temperature=runtime_args.ref_logit_kl_temperature,
                )
                loss = loss + runtime_args.ref_logit_kl_weight * ref_logit_kl_loss
        return loss, base_loss_for_log, ref_attn_kl_loss.detach(), ref_logit_kl_loss.detach(), anchor_ref_attn_kl_loss.detach(), teacher_attn_kl_loss.detach(), teacher_qk_rel_loss.detach()

def train_one_epoch_ofq(epoch: int, model: nn.Module, loader, optimizer: torch.optim.Optimizer, loss_fn, runtime_args: SimpleNamespace, lr_scheduler: WarmupCosineScheduler, output_dir: Path, amp_autocast, loss_scaler, teacher: Optional[nn.Module], mixup_fn, ref_model: Optional[nn.Module] = None, anchor_ref_model: Optional[nn.Module] = None, model_ema: Optional[nn.Module] = None, confidence_ref_model: Optional[nn.Module] = None):
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
    ref_logit_kl_losses_m = AverageMeter()
    anchor_ref_attn_kl_losses_m = AverageMeter()
    teacher_attn_kl_losses_m = AverageMeter()
    teacher_qk_rel_losses_m = AverageMeter()
    teacher_attn_output_losses_m = AverageMeter()
    teacher_feature_output_losses_m = AverageMeter()
    act_scale_anchor_losses_m = AverageMeter()
    variation_trust_losses_m = AverageMeter()
    delta_direction_anchor_losses_m = AverageMeter()
    bin_reg_losses_m = AverageMeter()
    selective_bin_anchor_losses_m = AverageMeter()
    act_bin_margin_losses_m = AverageMeter()
    if epoch >= int(getattr(runtime_args, "quant_slow_state_observe_start_epoch", 0)):
        maybe_init_quant_slow_state(model, runtime_args)
    if not hasattr(runtime_args, "_global_train_update"):
        runtime_args._global_train_update = 0
    accum_steps = max(1, int(getattr(runtime_args, "grad_accum_steps", 1)))
    teacher_attn_output_layers = parse_layer_indices(runtime_args.teacher_attn_output_layers)
    teacher_feature_output_layers = parse_name_list(runtime_args.teacher_feature_output_layers)
    teacher_qkv_rel_layers = parse_layer_indices(getattr(runtime_args, "teacher_qkv_rel_layers", "all"))
    anchor_ref_head_mode = runtime_args.anchor_ref_head_mode or runtime_args.ref_head_mode
    act_bin_margin_layers = parse_name_list(getattr(runtime_args, "act_bin_margin_layers", ""))
    act_bin_margin_quantizers = parse_name_list(getattr(runtime_args, "act_bin_margin_quantizers", ""))
    capture_act_bin_margin = runtime_args.act_bin_margin_weight > 0
    if runtime_args.local_rank == 0 and teacher is not None and teacher_feature_output_layers and runtime_args.teacher_feature_output_weight > 0:
        print(
            "Teacher feature-output hooks: "
            f"layers={teacher_feature_output_layers}, "
            f"student_matches={matched_named_modules(model, teacher_feature_output_layers)}, "
            f"teacher_matches={matched_named_modules(teacher, teacher_feature_output_layers)}"
        )
    ref_attn_kl_weight = epoch_float_value(
        runtime_args.ref_attn_kl_weight_epoch_overrides,
        epoch,
        runtime_args.ref_attn_kl_weight,
    )
    anchor_ref_attn_kl_weight = epoch_float_value(
        runtime_args.anchor_ref_attn_kl_weight_epoch_overrides,
        epoch,
        runtime_args.anchor_ref_attn_kl_weight,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    local_update_count = 0
    saved_step_count = 0
    stopped_early = False
    current_trainable_policy = None
    warmup_updates = max(0, int(getattr(runtime_args, "step_checkpoint_warmup_updates", 0)))
    max_step_checkpoints_to_save = max(0, int(getattr(runtime_args, "max_step_checkpoints_to_save", 0)))
    logged_teacher_feature_debug = False
    logged_grad_damp = False
    aoq_explore_layers = parse_name_list(getattr(runtime_args, "aoq_explore_layers", ""))
    aoq_explore_layer_ratios = parse_layer_float_overrides(getattr(runtime_args, "aoq_explore_layer_ratios", ""))
    aoq_explore_active = None
    aoq_explore_state = None
    aoq_current_selective_margin = float(getattr(runtime_args, "aoq_explore_selective_margin", 0.0))
    aoq_quality_mode = str(getattr(runtime_args, "aoq_explore_quality_mode", "none") or "none")
    aoq_quality_layers = parse_name_list(getattr(runtime_args, "aoq_explore_quality_layers", "")) or aoq_explore_layers
    aoq_quality_start_update = int(getattr(runtime_args, "aoq_explore_quality_start_update", 0))
    aoq_quality_min_frac = float(getattr(runtime_args, "aoq_explore_quality_min_frac", 0.0))
    aoq_anchor_state = getattr(runtime_args, "_aoq_explore_anchor_state", None)
    aoq_history_state = getattr(runtime_args, "_aoq_explore_history_state", None)
    if aoq_history_state is None and aoq_quality_mode in {"history_oscillating", "recent_oscillating"}:
        aoq_history_state = {}
        runtime_args._aoq_explore_history_state = aoq_history_state

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        update_step = ((batch_idx + 1) % accum_steps == 0) or last_batch
        aoq_schedule = getattr(runtime_args, "aoq_explore_update_schedule", None) or []
        scheduled_aoq = aoq_explore_schedule_value(runtime_args, local_update_count)
        desired_aoq_explore_active = aoq_explore_enabled(runtime_args, local_update_count)
        if aoq_schedule and scheduled_aoq is None:
            base_ratio, base_threshold_ratio, base_selective_margin = 1.0, 0.0, 0.0
        elif scheduled_aoq is None:
            base_ratio = float(runtime_args.aoq_explore_scale_ratio)
            base_threshold_ratio = float(getattr(runtime_args, "aoq_explore_threshold_ratio", 0.0))
            base_selective_margin = float(getattr(runtime_args, "aoq_explore_selective_margin", 0.0))
        else:
            base_ratio, base_threshold_ratio, base_selective_margin = scheduled_aoq
        desired_aoq_state = (
            bool(desired_aoq_explore_active),
            float(base_ratio) if desired_aoq_explore_active else 1.0,
            float(base_threshold_ratio) if desired_aoq_explore_active else 0.0,
            float(base_selective_margin) if desired_aoq_explore_active else 0.0,
        )
        if desired_aoq_state != aoq_explore_state:
            quality_active = aoq_quality_mode != "none" and local_update_count >= aoq_quality_start_update
            active_threshold_ratio = desired_aoq_state[2]
            ratio = desired_aoq_state[1]
            selective_margin = desired_aoq_state[3]
            was_active = bool(aoq_explore_state[0]) if aoq_explore_state is not None else False
            apply_base_ratio = (
                abs(ratio - 1.0) >= 1e-12
                or active_threshold_ratio > 0.0
                or was_active
            )
            aoq_pairs = (
                set_aoq_explore_scale_ratio(model, aoq_explore_layers, ratio, selective_margin, active_threshold_ratio)
                if apply_base_ratio
                else 0
            )
            active_layer_ratios = aoq_explore_layer_ratios if desired_aoq_explore_active else {
                name: 1.0 for name in aoq_explore_layer_ratios
            }
            aoq_override_pairs, aoq_override_counts = set_aoq_explore_layer_ratios(
                model,
                active_layer_ratios,
                selective_margin,
                active_threshold_ratio,
            )
            aoq_explore_active = desired_aoq_explore_active
            aoq_explore_state = desired_aoq_state
            aoq_current_selective_margin = selective_margin
            if not aoq_explore_active or not quality_active:
                clear_aoq_explore_quality_masks(model)
            elif aoq_quality_mode != "none":
                clear_aoq_explore_quality_masks(model)
                initial_quality_stats = update_aoq_explore_quality_masks(
                    model,
                    aoq_quality_layers,
                    aoq_quality_mode,
                    aoq_current_selective_margin,
                    aoq_quality_min_frac,
                    anchor_state=aoq_anchor_state,
                    history_state=aoq_history_state,
                )
                if runtime_args.local_rank == 0:
                    near = initial_quality_stats.get("near", 0.0)
                    selected = initial_quality_stats.get("selected", 0.0)
                    moved = initial_quality_stats.get("moved", 0.0)
                    switched = initial_quality_stats.get("switched", 0.0)
                    oscillating = initial_quality_stats.get("oscillating", 0.0)
                    missing = initial_quality_stats.get("missing", 0.0)
                    print(
                        "AOQ crossing-quality selector init: "
                        f"epoch={epoch}, update={local_update_count}, mode={aoq_quality_mode}, "
                        f"pairs={initial_quality_stats.get('pairs', 0.0):.0f}, near={near:.0f}, "
                        f"selected={selected:.0f}, selected_over_near={selected / max(1.0, near):.6f}, "
                        f"moved_excluded={moved:.0f}, switched={switched:.0f}, "
                        f"oscillating={oscillating:.0f}, missing_pairs={missing:.0f}"
                    )
            if runtime_args.local_rank == 0:
                print(
                    "AOQ explore scale ratio update: "
                    f"epoch={epoch}, update={local_update_count}, active={aoq_explore_active}, "
                    f"base_ratio={ratio}, threshold_ratio={active_threshold_ratio}, base_layers={aoq_explore_layers}, "
                    f"selective_margin={selective_margin}, "
                    f"base_quantizers={aoq_pairs}, layer_ratios={active_layer_ratios}, "
                    f"layer_quantizers={aoq_override_pairs}, layer_counts={aoq_override_counts}, "
                    f"quality_mode={aoq_quality_mode}, quality_layers={aoq_quality_layers}, "
                    f"quality_start_update={aoq_quality_start_update}, "
                    f"quality_min_frac={aoq_quality_min_frac}, "
                    f"start_update={runtime_args.aoq_explore_start_update}, "
                    f"end_update={runtime_args.aoq_explore_end_update}"
                )
        desired_policy = update_policy_value(runtime_args.trainable_policy_update_overrides, local_update_count, runtime_args.trainable_policy)
        if desired_policy != current_trainable_policy:
            if runtime_args.trainable_policy_update_mode == "requires_grad":
                trainable_params, frozen_params = set_trainable_policy(model, desired_policy, runtime_args=runtime_args)
            else:
                trainable_params, frozen_params = set_trainable_policy(model, "all", runtime_args=runtime_args)
            current_trainable_policy = desired_policy
            if runtime_args.local_rank == 0:
                print(
                    f"Trainable parameter update policy: epoch={epoch}, update={local_update_count}, "
                    f"mode={runtime_args.trainable_policy_update_mode}, policy={desired_policy}, "
                    f"trainable={trainable_params}, frozen={frozen_params}"
                )
        maybe_initialize_variation_trust_anchor(model, runtime_args, local_update_count=local_update_count)
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
                student_attn_outputs = []
                teacher_attn_outputs = []
                student_feature_outputs = []
                teacher_feature_outputs = []
                act_bin_margin_captures = []
                act_bin_margin_handles = []
                capture_teacher_attn_output = (
                    teacher is not None
                    and epoch >= runtime_args.teacher_attn_output_warmup_epochs
                    and runtime_args.teacher_attn_output_weight > 0
                )
                capture_teacher_feature_output = (
                    teacher is not None
                    and teacher_feature_output_layers
                    and epoch >= runtime_args.teacher_feature_output_warmup_epochs
                    and runtime_args.teacher_feature_output_weight > 0
                )
                if capture_act_bin_margin:
                    act_bin_margin_captures, act_bin_margin_handles = install_activation_quantizer_input_hooks(
                        model,
                        act_bin_margin_layers,
                        quantizer_names=act_bin_margin_quantizers,
                        detach=False,
                    )
                try:
                    if runtime_args.model_type in {"deit", "swin"}:
                        if capture_teacher_attn_output and capture_teacher_feature_output:
                            with capture_attention_outputs(model, teacher_attn_output_layers, detach=False) as captured_student_outputs, capture_named_module_outputs(model, teacher_feature_output_layers, detach=False) as captured_student_feature_outputs:
                                student_output = model(input)
                                student_attn_outputs = list(captured_student_outputs)
                                student_feature_outputs = list(captured_student_feature_outputs)
                        elif capture_teacher_attn_output:
                            with capture_attention_outputs(model, teacher_attn_output_layers, detach=False) as captured_student_outputs:
                                student_output = model(input)
                                student_attn_outputs = list(captured_student_outputs)
                        elif capture_teacher_feature_output:
                            with capture_named_module_outputs(model, teacher_feature_output_layers, detach=False) as captured_student_feature_outputs:
                                student_output = model(input)
                                student_feature_outputs = list(captured_student_feature_outputs)
                        else:
                            student_output = model(input)
                        if isinstance(student_output, tuple):
                            student_logit = student_output[0]
                            student_attn_info = student_output[1] if len(student_output) > 1 else None
                        else:
                            student_logit = student_output
                            student_attn_info = None
                    else:
                        student_logit = model(input)
                        student_attn_info = None
                finally:
                    for handle in act_bin_margin_handles:
                        handle.remove()

                teacher_attn_info = None
                if runtime_args.use_kd:
                    with torch.no_grad():
                        if runtime_args.teacher_type in {"deit", "swin"}:
                            if capture_teacher_attn_output and capture_teacher_feature_output:
                                with capture_attention_outputs(teacher, teacher_attn_output_layers, detach=True) as captured_teacher_outputs, capture_named_module_outputs(teacher, teacher_feature_output_layers, detach=True) as captured_teacher_feature_outputs:
                                    teacher_output = teacher(input)
                                    teacher_attn_outputs = list(captured_teacher_outputs)
                                    teacher_feature_outputs = list(captured_teacher_feature_outputs)
                            elif capture_teacher_attn_output:
                                with capture_attention_outputs(teacher, teacher_attn_output_layers, detach=True) as captured_teacher_outputs:
                                    teacher_output = teacher(input)
                                    teacher_attn_outputs = list(captured_teacher_outputs)
                            elif capture_teacher_feature_output:
                                with capture_named_module_outputs(teacher, teacher_feature_output_layers, detach=True) as captured_teacher_feature_outputs:
                                    teacher_output = teacher(input)
                                    teacher_feature_outputs = list(captured_teacher_feature_outputs)
                            else:
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
                        if runtime_args.teacher_confidence_kd_power > 0:
                            loss = teacher_confidence_weighted_soft_kd(
                                student_logit,
                                teacher_logit,
                                power=runtime_args.teacher_confidence_kd_power,
                                temperature=runtime_args.teacher_soft_temperature,
                            )
                        elif runtime_args.teacher_soft_temperature != 1.0:
                            loss = teacher_soft_kd_with_temperature(
                                student_logit,
                                teacher_logit,
                                temperature=runtime_args.teacher_soft_temperature,
                            )
                        else:
                            loss = loss_fn(student_logit, teacher_logit)
                    elif runtime_args.kd_hard_and_soft == 1:
                        loss = loss_fn(student_logit, target, teacher_logit)
                    elif runtime_args.kd_hard_and_soft == 2:
                        loss = loss_fn(student_logit, student_attn_info, target, teacher_logit, teacher_attn_info)
                    elif runtime_args.kd_hard_and_soft == 3:
                        loss = loss_fn(student_logit, student_attn_info, target, teacher_logit, teacher_attn_info)
                    else:
                        raise NotImplementedError(f"Unsupported kd_hard_and_soft={runtime_args.kd_hard_and_soft}")
                    if runtime_args.teacher_confidence_band_kd_weight > 0:
                        loss = loss + runtime_args.teacher_confidence_band_kd_weight * teacher_confidence_band_soft_kd(
                            student_logit,
                            teacher_logit,
                            low=runtime_args.teacher_confidence_band_kd_low,
                            high=runtime_args.teacher_confidence_band_kd_high,
                            temperature=runtime_args.teacher_confidence_band_kd_temperature,
                        )
                    if runtime_args.ref_confidence_band_kd_weight > 0 and confidence_ref_model is not None:
                        with torch.no_grad():
                            ref_conf_output = confidence_ref_model(input)
                        ref_conf_logit = ref_conf_output[0] if isinstance(ref_conf_output, (tuple, list)) else ref_conf_output
                        loss = loss + runtime_args.ref_confidence_band_kd_weight * reference_confidence_band_soft_kd(
                            student_logit,
                            teacher_logit,
                            ref_conf_logit,
                            low=runtime_args.ref_confidence_band_kd_low,
                            high=runtime_args.ref_confidence_band_kd_high,
                            temperature=runtime_args.ref_confidence_band_kd_temperature,
                        )
                    if runtime_args.local_ref_confidence_band_kd_weight > 0 and confidence_ref_model is not None:
                        with torch.no_grad():
                            local_ref_output = confidence_ref_model(input)
                        local_ref_logit = local_ref_output[0] if isinstance(local_ref_output, (tuple, list)) else local_ref_output
                        loss = loss + runtime_args.local_ref_confidence_band_kd_weight * local_reference_confidence_band_soft_kd(
                            student_logit,
                            local_ref_logit,
                            low=runtime_args.local_ref_confidence_band_kd_low,
                            high=runtime_args.local_ref_confidence_band_kd_high,
                            temperature=runtime_args.local_ref_confidence_band_kd_temperature,
                        )
                    if runtime_args.class_protect_ref_kl_weight > 0 and confidence_ref_model is not None:
                        with torch.no_grad():
                            ref_class_output = confidence_ref_model(input)
                        ref_class_logit = ref_class_output[0] if isinstance(ref_class_output, (tuple, list)) else ref_class_output
                        loss = loss + runtime_args.class_protect_ref_kl_weight * class_protect_ref_kl_loss(
                            student_logit,
                            ref_class_logit,
                            target,
                            runtime_args.class_protect_ref_kl_classes,
                            temperature=runtime_args.class_protect_ref_kl_temperature,
                        )
                    if runtime_args.clean_start_target_loss_weight > 0:
                        student_logit_for_ce = student_logit[0] if isinstance(student_logit, tuple) else student_logit
                        loss = loss + runtime_args.clean_start_target_loss_weight * F.cross_entropy(student_logit_for_ce, target)
                else:
                    student_logit = student_logit[0] if isinstance(student_logit, tuple) else student_logit
                    loss = loss_fn(student_logit, target)

                base_loss_for_log = loss.detach()
                ref_attn_kl_loss = loss.new_zeros(())
                ref_logit_kl_loss = loss.new_zeros(())
                anchor_ref_attn_kl_loss = loss.new_zeros(())
                teacher_attn_kl_loss = loss.new_zeros(())
                teacher_qk_rel_loss = loss.new_zeros(())
                teacher_attn_output_loss = loss.new_zeros(())
                teacher_feature_output_loss = loss.new_zeros(())
                act_scale_anchor_loss_value = loss.new_zeros(())
                variation_trust_loss_value = loss.new_zeros(())
                delta_direction_anchor_loss_value = loss.new_zeros(())
                bin_reg_loss = loss.new_zeros(())
                selective_bin_anchor_loss_value = loss.new_zeros(())
                act_bin_margin_loss_value = loss.new_zeros(())
                current_ref_attn_kl_weight = ref_attn_kl_weight
                current_ref_logit_kl_weight = runtime_args.ref_logit_kl_weight
                if runtime_args.ref_stop_updates > 0 and local_update_count >= runtime_args.ref_stop_updates:
                    current_ref_attn_kl_weight = 0.0
                    current_ref_logit_kl_weight = 0.0
                use_ref_scheme = (
                    runtime_args.train_scheme == "ema_ref_attn_kl"
                    and ref_model is not None
                    and epoch >= runtime_args.ref_warmup_epochs
                    and local_update_count >= runtime_args.ref_warmup_updates
                    and (current_ref_attn_kl_weight > 0 or current_ref_logit_kl_weight > 0)
                )
                if use_ref_scheme:
                    with torch.no_grad():
                        ref_logits, ref_attn_info = ref_model(input)
                    if current_ref_attn_kl_weight > 0:
                        if str(runtime_args.ref_head_mode).startswith("dynamic_teacher_agree_top") and teacher_attn_info is not None:
                            ref_attn_kl_loss = attention_teacher_agree_consistency_loss(
                                student_attn_info,
                                ref_attn_info,
                                teacher_attn_info,
                                head_mode=runtime_args.ref_head_mode,
                                loss_type=runtime_args.ref_attn_loss,
                                clip_value=runtime_args.ref_attn_kl_clip,
                            )
                        else:
                            ref_attn_kl_loss = attention_kl_consistency_loss(
                                student_attn_info,
                                ref_attn_info,
                                head_mode=runtime_args.ref_head_mode,
                                loss_type=runtime_args.ref_attn_loss,
                                clip_value=runtime_args.ref_attn_kl_clip,
                            )
                        if runtime_args.ref_attn_kl_drop_prob < 1.0:
                            if runtime_args.ref_attn_kl_drop_prob <= 0.0:
                                ref_attn_kl_loss = ref_attn_kl_loss * 0.0
                            else:
                                kl_gate = (torch.rand((), device=ref_attn_kl_loss.device) < runtime_args.ref_attn_kl_drop_prob).to(ref_attn_kl_loss.dtype)
                                if runtime_args.ref_attn_kl_drop_scale:
                                    kl_gate = kl_gate / runtime_args.ref_attn_kl_drop_prob
                                ref_attn_kl_loss = ref_attn_kl_loss * kl_gate
                        loss = loss + current_ref_attn_kl_weight * ref_attn_kl_loss
                    if current_ref_logit_kl_weight > 0:
                        ref_logit_kl_loss = logits_kl_consistency_loss(
                            student_logit,
                            ref_logits,
                            temperature=runtime_args.ref_logit_kl_temperature,
                        )
                        loss = loss + current_ref_logit_kl_weight * ref_logit_kl_loss
                use_anchor_ref_scheme = (
                    runtime_args.train_scheme == "ema_ref_attn_kl"
                    and anchor_ref_model is not None
                    and epoch >= runtime_args.anchor_ref_warmup_epochs
                    and anchor_ref_attn_kl_weight > 0
                )
                if use_anchor_ref_scheme:
                    with torch.no_grad():
                        _, anchor_ref_attn_info = anchor_ref_model(input)
                    anchor_ref_attn_kl_loss = attention_kl_consistency_loss(
                        student_attn_info,
                        anchor_ref_attn_info,
                        head_mode=anchor_ref_head_mode,
                        loss_type=runtime_args.ref_attn_loss,
                        clip_value=runtime_args.ref_attn_kl_clip,
                    )
                    loss = loss + anchor_ref_attn_kl_weight * anchor_ref_attn_kl_loss
                use_teacher_attn_scheme = (
                    runtime_args.train_scheme == "ema_ref_attn_kl"
                    and teacher is not None
                    and epoch >= runtime_args.teacher_attn_kl_warmup_epochs
                    and runtime_args.teacher_attn_kl_weight > 0
                )
                if use_teacher_attn_scheme:
                    if teacher_attn_info is not None:
                        teacher_attn_info_for_kl = teacher_attn_info
                    else:
                        with torch.no_grad():
                            teacher_attn_output = teacher(input)
                        if isinstance(teacher_attn_output, tuple):
                            teacher_attn_info_for_kl = teacher_attn_output[1] if len(teacher_attn_output) > 1 else None
                        else:
                            teacher_attn_info_for_kl = None
                    teacher_attn_kl_loss = attention_kl_consistency_loss(
                        student_attn_info,
                        teacher_attn_info_for_kl,
                        head_mode=runtime_args.ref_head_mode,
                        loss_type=runtime_args.ref_attn_loss,
                        clip_value=runtime_args.ref_attn_kl_clip,
                    )
                    loss = loss + runtime_args.teacher_attn_kl_weight * teacher_attn_kl_loss
                use_teacher_qk_rel_scheme = (
                    runtime_args.train_scheme == "ema_ref_attn_kl"
                    and teacher_attn_info is not None
                    and epoch >= runtime_args.teacher_qk_rel_warmup_epochs
                    and runtime_args.teacher_qk_rel_weight > 0
                )
                if use_teacher_qk_rel_scheme:
                    teacher_qk_rel_loss = teacher_qk_relation_loss(student_attn_info, teacher_attn_info)
                    loss = loss + runtime_args.teacher_qk_rel_weight * teacher_qk_rel_loss
                use_teacher_qkv_rel_scheme = (
                    runtime_args.train_scheme == "ema_ref_attn_kl"
                    and teacher_attn_info is not None
                    and epoch >= runtime_args.teacher_qkv_rel_warmup_epochs
                    and runtime_args.teacher_qkv_rel_weight > 0
                )
                if use_teacher_qkv_rel_scheme:
                    teacher_qk_rel_loss = teacher_qkv_relation_loss(
                        student_attn_info,
                        teacher_attn_info,
                        layer_indices=teacher_qkv_rel_layers,
                        components=runtime_args.teacher_qkv_rel_components,
                    )
                    loss = loss + runtime_args.teacher_qkv_rel_weight * teacher_qk_rel_loss
                if capture_teacher_attn_output:
                    teacher_attn_output_loss = attention_output_mse_loss(student_attn_outputs, teacher_attn_outputs)
                    loss = loss + runtime_args.teacher_attn_output_weight * teacher_attn_output_loss
                if capture_teacher_feature_output:
                    if runtime_args.local_rank == 0 and not logged_teacher_feature_debug:
                        print(
                            "Teacher feature-output debug: "
                            f"student_count={len(student_feature_outputs)}, "
                            f"teacher_count={len(teacher_feature_outputs)}, "
                            f"{describe_output_pairs(student_feature_outputs, teacher_feature_outputs)}"
                        )
                        logged_teacher_feature_debug = True
                    if runtime_args.teacher_feature_output_loss == "norm_mse":
                        teacher_feature_output_loss = attention_output_normalized_mse_loss(student_feature_outputs, teacher_feature_outputs)
                    elif runtime_args.teacher_feature_output_loss == "mse":
                        teacher_feature_output_loss = attention_output_mse_loss(student_feature_outputs, teacher_feature_outputs)
                    else:
                        raise ValueError(f"Unsupported teacher_feature_output_loss={runtime_args.teacher_feature_output_loss}")
                    loss = loss + runtime_args.teacher_feature_output_weight * teacher_feature_output_loss
                bin_reg_start_update = int(getattr(runtime_args, "bin_reg_start_update", 0))
                bin_reg_end_update = int(getattr(runtime_args, "bin_reg_end_update", 0))
                bin_reg_active = (
                    runtime_args.bin_reg_weight > 0
                    and local_update_count >= bin_reg_start_update
                    and (bin_reg_end_update <= 0 or local_update_count < bin_reg_end_update)
                )
                if bin_reg_active:
                    bin_reg_layers = parse_name_list(getattr(runtime_args, "bin_reg_layers", ""))
                    bin_reg_loss, bin_reg_pairs = bin_regularizer_loss(
                        model,
                        runtime_args.bin_reg_variance_weight,
                        module_names=bin_reg_layers,
                        attn_only=getattr(runtime_args, "bin_reg_attn_only", False),
                    )
                    loss = loss + runtime_args.bin_reg_weight * bin_reg_loss
                    if runtime_args.local_rank == 0 and local_update_count == bin_reg_start_update:
                        print(
                            "Enabled bin regularizer: "
                            f"weight={runtime_args.bin_reg_weight}, "
                            f"variance_weight={runtime_args.bin_reg_variance_weight}, "
                            f"layers={bin_reg_layers}, attn_only={runtime_args.bin_reg_attn_only}, "
                            f"pairs={bin_reg_pairs}, start_update={bin_reg_start_update}, end_update={bin_reg_end_update}"
                        )
                selective_anchor_weight = float(getattr(runtime_args, "selective_bin_anchor_weight", 0.0))
                selective_capture_update = int(getattr(runtime_args, "selective_bin_anchor_capture_update", 0))
                selective_end_update = int(getattr(runtime_args, "selective_bin_anchor_end_update", 0))
                if selective_anchor_weight > 0 and local_update_count >= selective_capture_update:
                    if not hasattr(runtime_args, "_selective_bin_anchor_state"):
                        selective_layers = parse_name_list(getattr(runtime_args, "selective_bin_anchor_layers", ""))
                        anchor_state, anchor_pairs, anchor_masked, anchor_total = capture_selective_bin_anchor_state(
                            model,
                            selective_layers,
                            float(getattr(runtime_args, "selective_bin_anchor_margin", 0.05)),
                        )
                        runtime_args._selective_bin_anchor_state = anchor_state
                        if runtime_args.local_rank == 0:
                            mask_fraction = anchor_masked / max(1, anchor_total)
                            print(
                                "Captured selective bin anchor: "
                                f"weight={selective_anchor_weight}, layers={selective_layers}, "
                                f"pairs={anchor_pairs}, masked={anchor_masked}, total={anchor_total}, "
                                f"mask_fraction={mask_fraction:.6f}, "
                                f"capture_update={selective_capture_update}, end_update={selective_end_update}, "
                                f"margin={runtime_args.selective_bin_anchor_margin}"
                            )
                    selective_active = selective_end_update <= 0 or local_update_count < selective_end_update
                    if selective_active:
                        selective_bin_anchor_loss_value, selective_pairs, selective_masked, selective_total = selective_bin_anchor_loss(
                            model,
                            runtime_args._selective_bin_anchor_state,
                        )
                        loss = loss + selective_anchor_weight * selective_bin_anchor_loss_value
                        if runtime_args.local_rank == 0 and local_update_count == selective_capture_update:
                            mask_fraction = selective_masked / max(1, selective_total)
                            print(
                                "Enabled selective bin anchor: "
                                f"weight={selective_anchor_weight}, pairs={selective_pairs}, "
                                f"masked={selective_masked}, total={selective_total}, "
                                f"mask_fraction={mask_fraction:.6f}, "
                                f"capture_update={selective_capture_update}, end_update={selective_end_update}"
                            )
                candidate_anchor_weight = float(getattr(runtime_args, "candidate_bin_anchor_weight", 0.0))
                candidate_capture_update = int(getattr(runtime_args, "candidate_bin_anchor_capture_update", 0))
                candidate_end_update = int(getattr(runtime_args, "candidate_bin_anchor_end_update", 0))
                if candidate_anchor_weight > 0 and local_update_count >= candidate_capture_update:
                    if not hasattr(runtime_args, "_candidate_bin_anchor_state"):
                        candidate_layers = parse_name_list(getattr(runtime_args, "candidate_bin_anchor_layers", ""))
                        candidate_state, candidate_pairs, candidate_masked, candidate_total, candidate_missing = (
                            capture_candidate_bin_anchor_state(
                                model,
                                candidate_layers,
                                runtime_args._candidate_bin_anchor_source_state,
                            )
                        )
                        runtime_args._candidate_bin_anchor_state = candidate_state
                        if runtime_args.local_rank == 0:
                            mask_fraction = candidate_masked / max(1, candidate_total)
                            print(
                                "Captured candidate-bin anchor: "
                                f"weight={candidate_anchor_weight}, layers={candidate_layers}, "
                                f"pairs={candidate_pairs}, masked={candidate_masked}, total={candidate_total}, "
                                f"mask_fraction={mask_fraction:.6f}, missing_pairs={candidate_missing}, "
                                f"capture_update={candidate_capture_update}, end_update={candidate_end_update}, "
                                f"source={runtime_args.candidate_bin_anchor_source_checkpoint}"
                            )
                    candidate_active = candidate_end_update <= 0 or local_update_count < candidate_end_update
                    if candidate_active:
                        candidate_bin_anchor_loss_value, candidate_pairs, candidate_masked, candidate_total = selective_bin_anchor_loss(
                            model,
                            runtime_args._candidate_bin_anchor_state,
                        )
                        loss = loss + candidate_anchor_weight * candidate_bin_anchor_loss_value
                        if runtime_args.local_rank == 0 and local_update_count == candidate_capture_update:
                            mask_fraction = candidate_masked / max(1, candidate_total)
                            print(
                                "Enabled candidate-bin anchor: "
                                f"weight={candidate_anchor_weight}, pairs={candidate_pairs}, "
                                f"masked={candidate_masked}, total={candidate_total}, "
                                f"mask_fraction={mask_fraction:.6f}, "
                                f"capture_update={candidate_capture_update}, end_update={candidate_end_update}"
                            )
                if capture_act_bin_margin:
                    act_bin_margin_loss_value, act_bin_margin_pairs = activation_bin_margin_loss(
                        act_bin_margin_captures,
                        margin=runtime_args.act_bin_margin,
                        max_elements=runtime_args.act_bin_margin_max_elements,
                    )
                    loss = loss + runtime_args.act_bin_margin_weight * act_bin_margin_loss_value
                    if runtime_args.local_rank == 0 and local_update_count == 0:
                        print(
                            "Enabled activation bin-margin regularizer: "
                            f"weight={runtime_args.act_bin_margin_weight}, "
                            f"layers={act_bin_margin_layers}, quantizers={act_bin_margin_quantizers}, "
                            f"margin={runtime_args.act_bin_margin}, pairs={act_bin_margin_pairs}"
                        )
                act_scale_anchor_state = getattr(runtime_args, "_act_scale_anchor_state", {})
                if (
                    runtime_args.act_scale_anchor_weight > 0
                    and epoch >= runtime_args.act_scale_anchor_start_epoch
                    and act_scale_anchor_state
                ):
                    act_scale_anchor_loss_value, act_scale_anchor_pairs = activation_scale_anchor_loss(model, act_scale_anchor_state)
                    loss = loss + runtime_args.act_scale_anchor_weight * act_scale_anchor_loss_value
                    if runtime_args.local_rank == 0 and local_update_count == 0:
                        print(
                            "Enabled activation scale anchor: "
                            f"weight={runtime_args.act_scale_anchor_weight}, "
                            f"layers={runtime_args.act_scale_anchor_layers}, pairs={act_scale_anchor_pairs}"
                        )
                variation_trust_state = getattr(runtime_args, "_variation_trust_state", {})
                if runtime_args.variation_trust_weight > 0 and variation_trust_state:
                    variation_trust_loss_value, variation_trust_pairs, variation_trust_avg_multiplier = variation_trust_loss(
                        model,
                        variation_trust_state,
                        runtime_args,
                    )
                    loss = loss + runtime_args.variation_trust_weight * variation_trust_loss_value
                    if runtime_args.local_rank == 0 and local_update_count == 0:
                        print(
                            "Enabled variation trust regularizer: "
                            f"weight={runtime_args.variation_trust_weight}, pairs={variation_trust_pairs}, "
                            f"avg_multiplier={variation_trust_avg_multiplier:.3f}, "
                            f"late_layers={runtime_args.variation_trust_late_layers}, "
                            f"early_layers={runtime_args.variation_trust_early_layers}"
                        )
                delta_direction_anchor_state = getattr(runtime_args, "_delta_direction_anchor_state", {})
                if runtime_args.delta_direction_anchor_weight > 0 and delta_direction_anchor_state:
                    delta_direction_anchor_loss_value, delta_direction_anchor_pairs = delta_direction_anchor_loss(
                        model,
                        delta_direction_anchor_state,
                        runtime_args,
                        local_update_count,
                    )
                    loss = loss + runtime_args.delta_direction_anchor_weight * delta_direction_anchor_loss_value
                    if (
                        runtime_args.local_rank == 0
                        and local_update_count == int(getattr(runtime_args, "delta_direction_anchor_start_update", 0))
                    ):
                        print(
                            "Enabled delta direction anchor: "
                            f"weight={runtime_args.delta_direction_anchor_weight}, "
                            f"pairs={delta_direction_anchor_pairs}, "
                            f"start_update={runtime_args.delta_direction_anchor_start_update}"
                        )

            loss_for_log = loss.detach()
            ref_attn_kl_loss_for_log = ref_attn_kl_loss.detach()
            ref_logit_kl_loss_for_log = ref_logit_kl_loss.detach()
            anchor_ref_attn_kl_loss_for_log = anchor_ref_attn_kl_loss.detach()
            teacher_attn_kl_loss_for_log = teacher_attn_kl_loss.detach()
            teacher_qk_rel_loss_for_log = teacher_qk_rel_loss.detach()
            teacher_attn_output_loss_for_log = teacher_attn_output_loss.detach()
            teacher_feature_output_loss_for_log = teacher_feature_output_loss.detach()
            act_scale_anchor_loss_for_log = act_scale_anchor_loss_value.detach()
            variation_trust_loss_for_log = variation_trust_loss_value.detach()
            delta_direction_anchor_loss_for_log = delta_direction_anchor_loss_value.detach()
            bin_reg_loss_for_log = bin_reg_loss.detach()
            selective_bin_anchor_loss_for_log = selective_bin_anchor_loss_value.detach()
            act_bin_margin_loss_for_log = act_bin_margin_loss_value.detach()
            if not runtime_args.distributed:
                losses_m.update(loss_for_log.item(), input.size(0))
                base_losses_m.update(base_loss_for_log.item(), input.size(0))
                ref_attn_kl_losses_m.update(ref_attn_kl_loss_for_log.item(), input.size(0))
                ref_logit_kl_losses_m.update(ref_logit_kl_loss_for_log.item(), input.size(0))
                anchor_ref_attn_kl_losses_m.update(anchor_ref_attn_kl_loss_for_log.item(), input.size(0))
                teacher_attn_kl_losses_m.update(teacher_attn_kl_loss_for_log.item(), input.size(0))
                teacher_qk_rel_losses_m.update(teacher_qk_rel_loss_for_log.item(), input.size(0))
                teacher_attn_output_losses_m.update(teacher_attn_output_loss_for_log.item(), input.size(0))
                teacher_feature_output_losses_m.update(teacher_feature_output_loss_for_log.item(), input.size(0))
                act_scale_anchor_losses_m.update(act_scale_anchor_loss_for_log.item(), input.size(0))
                variation_trust_losses_m.update(variation_trust_loss_for_log.item(), input.size(0))
                delta_direction_anchor_losses_m.update(delta_direction_anchor_loss_for_log.item(), input.size(0))
                bin_reg_losses_m.update(bin_reg_loss_for_log.item(), input.size(0))
                act_bin_margin_losses_m.update(act_bin_margin_loss_for_log.item(), input.size(0))

            scaled_loss = loss / accum_steps
            if (
                update_step
                and runtime_args.train_scheme == "ema_ref_attn_kl"
                and ref_model is not None
                and runtime_args.ref_update == "prev_step"
                and local_update_count % runtime_args.ref_update_interval == 0
            ):
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
                    if runtime_args.trainable_policy_update_mode == "grad_mask":
                        apply_gradient_mask_policy(model, current_trainable_policy, runtime_args=runtime_args)
                    elif runtime_args.trainable_policy_update_mode == "grad_damp":
                        damped_params, masked_params = apply_gradient_damp_policy(model, current_trainable_policy, runtime_args=runtime_args)
                        if runtime_args.local_rank == 0 and not logged_grad_damp and str(current_trainable_policy or "all") != "all":
                            print(
                                "Applied gradient damping policy: "
                                f"policy={current_trainable_policy}, damp={runtime_args.trainable_policy_grad_damp}, "
                                f"damped_params={damped_params}, masked_params={masked_params}"
                            )
                            logged_grad_damp = True
                    if runtime_args.clip_grad is not None:
                        dispatch_clip_grad(model_parameters(model, exclude_head="agc" in runtime_args.clip_mode), value=runtime_args.clip_grad, mode=runtime_args.clip_mode)
                    quality_active = aoq_quality_mode != "none" and local_update_count >= aoq_quality_start_update
                    if aoq_explore_active and quality_active:
                        clear_aoq_explore_quality_masks(model)
                        quality_stats = update_aoq_explore_quality_masks(
                            model,
                            aoq_quality_layers,
                            aoq_quality_mode,
                            aoq_current_selective_margin,
                            aoq_quality_min_frac,
                            anchor_state=aoq_anchor_state,
                            history_state=aoq_history_state,
                        )
                        if runtime_args.local_rank == 0 and (local_update_count < 3 or local_update_count % 200 == 0):
                            near = quality_stats.get("near", 0.0)
                            selected = quality_stats.get("selected", 0.0)
                            moved = quality_stats.get("moved", 0.0)
                            switched = quality_stats.get("switched", 0.0)
                            oscillating = quality_stats.get("oscillating", 0.0)
                            missing = quality_stats.get("missing", 0.0)
                            print(
                                "AOQ crossing-quality selector: "
                                f"epoch={epoch}, update={local_update_count}, mode={aoq_quality_mode}, "
                                f"pairs={quality_stats.get('pairs', 0.0):.0f}, near={near:.0f}, "
                                f"selected={selected:.0f}, selected_over_near={selected / max(1.0, near):.6f}, "
                                f"moved_excluded={moved:.0f}, switched={switched:.0f}, "
                                f"oscillating={oscillating:.0f}, missing_pairs={missing:.0f}"
                            )
                    elif aoq_quality_mode != "none" and not quality_active:
                        clear_aoq_explore_quality_masks(model)
                    optimizer.step()

        if update_step:
            optimizer.zero_grad(set_to_none=True)
            local_update_count += 1
            runtime_args._global_train_update += 1
            if epoch >= int(getattr(runtime_args, "quant_slow_state_observe_start_epoch", 0)):
                update_quant_slow_state(
                    model,
                    runtime_args,
                    runtime_args._global_train_update,
                    pull_enabled=epoch >= int(getattr(runtime_args, "quant_slow_state_start_epoch", 0)),
                )
            if model_ema is not None:
                update_model_ema(model, model_ema, runtime_args.model_ema_decay)
            if runtime_args.train_scheme == "ema_ref_attn_kl" and ref_model is not None and runtime_args.ref_update == "ema":
                update_ref_model(model, ref_model, runtime_args.ref_momentum)
            telemetry = update_weight_bin_telemetry(model, runtime_args, local_update_count)
            if runtime_args.local_rank == 0 and telemetry is not None:
                print(
                    "WeightBinTelemetry: "
                    f"epoch={epoch}, update={local_update_count}, "
                    f"pairs={int(telemetry['pairs'])}, total={int(telemetry['total'])}, "
                    f"near_fraction={telemetry['near_fraction']:.6f}, "
                    f"switch_fraction={telemetry['switch_fraction']:.6f}, "
                    f"oscillation_fraction={telemetry['oscillation_fraction']:.6f}, "
                    f"mean_abs_delta={telemetry['mean_abs_delta']:.6f}"
                )
            if runtime_args.local_rank == 0 and runtime_args.save_step_checkpoints:
                interval = max(1, int(runtime_args.step_checkpoint_interval))
                if warmup_updates > 0:
                    if runtime_args.save_initial_step_checkpoint and local_update_count == warmup_updates:
                        save_step_checkpoint(model, optimizer, runtime_args, output_dir, f"step_{saved_step_count:04d}", epoch=epoch, batch_idx=batch_idx, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, model_ema=model_ema)
                        saved_step_count += 1
                    if local_update_count > warmup_updates and (local_update_count - warmup_updates) % interval == 0:
                        if max_step_checkpoints_to_save == 0 or saved_step_count < max_step_checkpoints_to_save:
                            save_step_checkpoint(model, optimizer, runtime_args, output_dir, f"step_{saved_step_count:04d}", epoch=epoch, batch_idx=batch_idx, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, model_ema=model_ema)
                            saved_step_count += 1
                    if max_step_checkpoints_to_save > 0 and saved_step_count >= max_step_checkpoints_to_save:
                        stopped_early = True
                        break
                elif local_update_count % interval == 0:
                    save_step_checkpoint(model, optimizer, runtime_args, output_dir, f"step_{local_update_count:04d}", epoch=epoch, batch_idx=batch_idx, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, model_ema=model_ema)

        if runtime_args.sync_step_timing:
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
                reduced_ref_logit_kl_loss = reduce_tensor(ref_logit_kl_loss_for_log, runtime_args.world_size)
                reduced_anchor_ref_attn_kl_loss = reduce_tensor(anchor_ref_attn_kl_loss_for_log, runtime_args.world_size)
                reduced_teacher_attn_kl_loss = reduce_tensor(teacher_attn_kl_loss_for_log, runtime_args.world_size)
                reduced_teacher_qk_rel_loss = reduce_tensor(teacher_qk_rel_loss_for_log, runtime_args.world_size)
                reduced_teacher_attn_output_loss = reduce_tensor(teacher_attn_output_loss_for_log, runtime_args.world_size)
                reduced_teacher_feature_output_loss = reduce_tensor(teacher_feature_output_loss_for_log, runtime_args.world_size)
                reduced_act_scale_anchor_loss = reduce_tensor(act_scale_anchor_loss_for_log, runtime_args.world_size)
                reduced_variation_trust_loss = reduce_tensor(variation_trust_loss_for_log, runtime_args.world_size)
                reduced_delta_direction_anchor_loss = reduce_tensor(delta_direction_anchor_loss_for_log, runtime_args.world_size)
                reduced_bin_reg_loss = reduce_tensor(bin_reg_loss_for_log, runtime_args.world_size)
                reduced_selective_bin_anchor_loss = reduce_tensor(selective_bin_anchor_loss_for_log, runtime_args.world_size)
                reduced_act_bin_margin_loss = reduce_tensor(act_bin_margin_loss_for_log, runtime_args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
                base_losses_m.update(reduced_base_loss.item(), input.size(0))
                ref_attn_kl_losses_m.update(reduced_ref_attn_kl_loss.item(), input.size(0))
                ref_logit_kl_losses_m.update(reduced_ref_logit_kl_loss.item(), input.size(0))
                anchor_ref_attn_kl_losses_m.update(reduced_anchor_ref_attn_kl_loss.item(), input.size(0))
                teacher_attn_kl_losses_m.update(reduced_teacher_attn_kl_loss.item(), input.size(0))
                teacher_qk_rel_losses_m.update(reduced_teacher_qk_rel_loss.item(), input.size(0))
                teacher_attn_output_losses_m.update(reduced_teacher_attn_output_loss.item(), input.size(0))
                teacher_feature_output_losses_m.update(reduced_teacher_feature_output_loss.item(), input.size(0))
                act_scale_anchor_losses_m.update(reduced_act_scale_anchor_loss.item(), input.size(0))
                variation_trust_losses_m.update(reduced_variation_trust_loss.item(), input.size(0))
                delta_direction_anchor_losses_m.update(reduced_delta_direction_anchor_loss.item(), input.size(0))
                bin_reg_losses_m.update(reduced_bin_reg_loss.item(), input.size(0))
                selective_bin_anchor_losses_m.update(reduced_selective_bin_anchor_loss.item(), input.size(0))
                act_bin_margin_losses_m.update(reduced_act_bin_margin_loss.item(), input.size(0))
            if runtime_args.local_rank == 0:
                print(
                    f"Train: {epoch} [{batch_idx:>4d}/{len(loader)} ({100. * batch_idx / last_idx:>3.0f}%)]  "
                    f"Loss: {losses_m.val:>9.6f} ({losses_m.avg:>6.4f})  "
                    f"BaseLoss: {base_losses_m.val:>9.6f} ({base_losses_m.avg:>6.4f})  "
                    f"RefAttnKL: {ref_attn_kl_losses_m.val:.3e} ({ref_attn_kl_losses_m.avg:.3e})  "
                    f"RefLogitKL: {ref_logit_kl_losses_m.val:.3e} ({ref_logit_kl_losses_m.avg:.3e})  "
                    f"AnchorRefAttnKL: {anchor_ref_attn_kl_losses_m.val:.3e} ({anchor_ref_attn_kl_losses_m.avg:.3e})  "
                    f"TeacherAttnKL: {teacher_attn_kl_losses_m.val:.3e} ({teacher_attn_kl_losses_m.avg:.3e})  "
                    f"TeacherRel: {teacher_qk_rel_losses_m.val:.3e} ({teacher_qk_rel_losses_m.avg:.3e})  "
                    f"TeacherAttnOut: {teacher_attn_output_losses_m.val:.3e} ({teacher_attn_output_losses_m.avg:.3e})  "
                    f"TeacherFeatOut: {teacher_feature_output_losses_m.val:.3e} ({teacher_feature_output_losses_m.avg:.3e})  "
                    f"ActScaleAnchor: {act_scale_anchor_losses_m.val:.3e} ({act_scale_anchor_losses_m.avg:.3e})  "
                    f"VarTrust: {variation_trust_losses_m.val:.3e} ({variation_trust_losses_m.avg:.3e})  "
                    f"DirAnchor: {delta_direction_anchor_losses_m.val:.3e} ({delta_direction_anchor_losses_m.avg:.3e})  "
                    f"BinReg: {bin_reg_losses_m.val:.3e} ({bin_reg_losses_m.avg:.3e})  "
                    f"SelBinAnchor: {selective_bin_anchor_losses_m.val:.3e} ({selective_bin_anchor_losses_m.avg:.3e})  "
                    f"ActBinMargin: {act_bin_margin_losses_m.val:.3e} ({act_bin_margin_losses_m.avg:.3e})  "
                    f"Time: {batch_time_m.val:.3f}s, {input.size(0) * runtime_args.world_size / batch_time_m.val:>7.2f}/s  "
                    f"({batch_time_m.avg:.3f}s, {input.size(0) * runtime_args.world_size / batch_time_m.avg:>7.2f}/s)  "
                    f"LR: {lr:.3e}  RefW: {current_ref_attn_kl_weight:.3e}  AnchorRefW: {anchor_ref_attn_kl_weight:.3e}  "
                    f"Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})"
                )

        if runtime_args.max_train_updates and local_update_count >= runtime_args.max_train_updates:
            stopped_early = True
            break
        end = time.time()

    if runtime_args.local_rank == 0:
        samples_per_step = runtime_args.batch_size * runtime_args.world_size
        throughput = samples_per_step / batch_time_m.avg if batch_time_m.avg > 0 else 0.0
        print(
            f"TrainSummary: epoch={epoch} updates={local_update_count} "
            f"avg_step_time={batch_time_m.avg:.6f}s "
            f"samples_per_step={samples_per_step} samples_per_sec={throughput:.2f}"
        )

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

    qqkkvv = runtime_args.kd_hard_and_soft in {2, 3} or runtime_args.teacher_qk_rel_weight > 0 or runtime_args.teacher_qkv_rel_weight > 0
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
        set_attention_mode(model, collect_attention=True, qqkkvv=qqkkvv)
    load_initial_after_alpha = bool(runtime_args.initial_checkpoint and not runtime_args.eval_only)

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
    if runtime_args.compile:
        if runtime_args.local_rank == 0:
            print(f"Compiling OFQ model with torch.compile mode={runtime_args.compile_mode}")
        model = torch.compile(model, mode=runtime_args.compile_mode)

    use_amp = bool(runtime_args.amp or runtime_args.native_amp)
    amp_dtype = torch.bfloat16 if runtime_args.amp_dtype == "bf16" else torch.float16
    amp_autocast = functools.partial(torch.amp.autocast, "cuda", dtype=amp_dtype) if use_amp else contextlib.suppress
    loss_scaler = NativeScaler() if use_amp and amp_dtype is torch.float16 else None
    if runtime_args.local_rank == 0 and use_amp:
        scaler_state = "enabled" if loss_scaler is not None else "disabled"
        print(f"Using OFQ CUDA AMP dtype={runtime_args.amp_dtype}, loss_scaler={scaler_state}")

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
        if runtime_args.progressive_bit_schedule:
            initial_wbits, initial_abits = progressive_bits_for_epoch(
                runtime_args.progressive_bit_schedule,
                0,
                runtime_args.wq_bitw,
                runtime_args.aq_bitw,
            )
            weight_modules, act_modules = set_fake_quant_bits(model, initial_wbits, initial_abits)
            runtime_args.wq_bitw = initial_wbits
            runtime_args.aq_bitw = initial_abits
            if runtime_args.local_rank == 0:
                print(
                    "Applied progressive fake-quant bits before setup_alpha: "
                    f"epoch=0 wbits={initial_wbits} abits={initial_abits} "
                    f"weight_modules={weight_modules} act_modules={act_modules}"
                )
        setup_alpha(model, loader_train, runtime_args, amp_autocast)
        if runtime_args.initial_checkpoint:
            load_checkpoint(model, runtime_args.initial_checkpoint, strict=False)
            if runtime_args.post_load_alpha:
                setup_alpha(model, loader_train, runtime_args, amp_autocast)
        if runtime_args.resume:
            strict_resume_checkpoint(
                model,
                runtime_args.resume,
                optimizer=None,
                loss_scaler=None,
                lr_scheduler=None,
                model_ema=None,
                restore_rng=False,
                log_info=runtime_args.local_rank == 0,
            )
        dataset_eval = create_dataset_compat(
            runtime_args.dataset,
            root=runtime_args.data_dir,
            split=runtime_args.val_split,
            is_training=False,
            batch_size=runtime_args.batch_size,
            subset_ratio=runtime_args.subset_ratio,
            rank=runtime_args.rank,
            world_size=runtime_args.world_size,
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
            shutdown_data_loader(loader_eval)
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
    train_loader_batch_size = runtime_args.batch_size
    if runtime_args.forward_micro_batch_size > 0 and runtime_args.forward_micro_batch_size < runtime_args.batch_size:
        train_loader_batch_size = runtime_args.forward_micro_batch_size
        if runtime_args.local_rank == 0:
            print(
                f"Using forward micro-batch: loader_batch_size={train_loader_batch_size}, "
                f"grad_accum_steps={runtime_args.grad_accum_steps}, effective_per_gpu_batch={runtime_args.batch_size}"
            )
    loader_train = create_loader_compat(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=train_loader_batch_size,
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
            rank=runtime_args.rank,
            world_size=runtime_args.world_size,
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
    if load_initial_after_alpha:
        load_checkpoint(model, runtime_args.initial_checkpoint, strict=False)
    model_ema = None
    if runtime_args.model_ema:
        model_ema = clone_model_ema(model)
        if runtime_args.local_rank == 0:
            print(f"Enabled student weight EMA: decay={runtime_args.model_ema_decay}")

    optimizer = create_ofq_optimizer(runtime_args, model)
    updates_per_epoch = max(1, (len(loader_train) + max(1, runtime_args.grad_accum_steps) - 1) // max(1, runtime_args.grad_accum_steps))
    lr_scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=runtime_args.lr,
        min_lr=runtime_args.min_lr,
        warmup_updates=runtime_args.warmup_epochs * updates_per_epoch,
        total_updates=(runtime_args.scheduler_epochs or runtime_args.epochs) * updates_per_epoch,
    )

    start_epoch = 0
    if runtime_args.resume:
        if runtime_args.no_resume_opt:
            start_epoch = strict_resume_checkpoint(
                model,
                runtime_args.resume,
                optimizer=None,
                loss_scaler=None,
                lr_scheduler=None,
                model_ema=None,
                restore_rng=False,
                log_info=runtime_args.local_rank == 0,
            ) or 0
            if runtime_args.local_rank == 0:
                print("Strict resume: --no-resume-opt set; optimizer/scheduler/scaler/RNG intentionally not restored.")
        else:
            start_epoch = strict_resume_checkpoint(
                model,
                runtime_args.resume,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                lr_scheduler=lr_scheduler,
                model_ema=model_ema,
                restore_rng=True,
                log_info=runtime_args.local_rank == 0,
            ) or 0
            if runtime_args.resume_opt_force_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = runtime_args.lr
                    if "initial_lr" in param_group:
                        param_group["initial_lr"] = runtime_args.lr
                if runtime_args.local_rank == 0:
                    print(f"Strict resume: forced restored optimizer lr to {runtime_args.lr}")
        post_resume_setup_batches = max(0, int(getattr(runtime_args, "post_resume_setup_alpha_batches", 0)))
        if post_resume_setup_batches > 0:
            original_setup_batches = runtime_args.setup_alpha_batches
            runtime_args.setup_alpha_batches = post_resume_setup_batches
            if runtime_args.local_rank == 0:
                print(f"post-resume setup alpha batches={post_resume_setup_batches}")
            setup_alpha(model, loader_train, runtime_args, amp_autocast)
            runtime_args.setup_alpha_batches = original_setup_batches
    if runtime_args.start_epoch is not None:
        start_epoch = int(runtime_args.start_epoch)
    if start_epoch > 0 and (not runtime_args.resume or runtime_args.no_resume_opt):
        lr_scheduler.step_update(start_epoch * updates_per_epoch)

    if start_epoch == 0:
        run_pre_qat_activation_percentile_calibration(model, loader_train, runtime_args, amp_autocast)
        run_pre_qat_activation_mse_calibration(model, loader_train, runtime_args, amp_autocast)
        run_pre_qat_reconstruction(model, loader_train, optimizer, runtime_args, amp_autocast, teacher)
        pre_qat_feature_anchor_model = None
        pre_qat_feature_anchor_weight = float(getattr(runtime_args, "pre_qat_feature_recon_anchor_kl_weight", 0.0))
        if runtime_args.pre_qat_feature_recon_updates > 0 and pre_qat_feature_anchor_weight > 0:
            pre_qat_feature_anchor_model = copy.deepcopy(maybe_unwrap_ddp(model)).cuda()
            pre_qat_feature_anchor_model.eval()
            for param in pre_qat_feature_anchor_model.parameters():
                param.requires_grad_(False)
            if runtime_args.local_rank == 0:
                print(
                    "Captured pre-QAT feature reconstruction anchor model: "
                    f"kl_weight={pre_qat_feature_anchor_weight}, "
                    f"temperature={runtime_args.pre_qat_feature_recon_anchor_kl_temperature}"
                )
        run_pre_qat_feature_reconstruction(
            model,
            loader_train,
            optimizer,
            runtime_args,
            amp_autocast,
            teacher,
            anchor_model=pre_qat_feature_anchor_model,
            anchor_kl_weight=pre_qat_feature_anchor_weight,
            anchor_kl_temperature=runtime_args.pre_qat_feature_recon_anchor_kl_temperature,
        )
        del pre_qat_feature_anchor_model
        run_pre_qat_sequential_feature_reconstruction(model, loader_train, optimizer, runtime_args, amp_autocast, teacher)
        if runtime_args.act_scale_anchor_weight > 0:
            anchor_layers = parse_name_list(runtime_args.act_scale_anchor_layers)
            runtime_args._act_scale_anchor_state = collect_activation_scale_anchor_state(model, anchor_layers)
            if runtime_args.local_rank == 0:
                print(
                    "Initialized activation scale anchor: "
                    f"params={len(runtime_args._act_scale_anchor_state)}, "
                    f"layers={anchor_layers}, weight={runtime_args.act_scale_anchor_weight}, "
                    f"start_epoch={runtime_args.act_scale_anchor_start_epoch}"
                )
        if runtime_args.variation_trust_weight > 0:
            maybe_initialize_variation_trust_anchor(model, runtime_args, local_update_count=0, force=runtime_args.variation_trust_start_update <= 0)
    elif (
        runtime_args.no_resume_opt
        and (
            getattr(runtime_args, "pre_qat_act_percentile_calib_batches", 0) > 0
            or getattr(runtime_args, "pre_qat_act_mse_calib_batches", 0) > 0
        )
    ):
        run_pre_qat_activation_percentile_calibration(model, loader_train, runtime_args, amp_autocast)
        run_pre_qat_activation_mse_calibration(model, loader_train, runtime_args, amp_autocast)
        if runtime_args.local_rank == 0:
            print(f"Applied pre-QAT activation calibration after strict resume: start_epoch={start_epoch}")

    if runtime_args.variation_trust_weight > 0 and not hasattr(runtime_args, "_variation_trust_state"):
        maybe_initialize_variation_trust_anchor(model, runtime_args, local_update_count=0, force=runtime_args.variation_trust_start_update <= 0)
    if runtime_args.delta_direction_anchor_weight > 0:
        initialize_delta_direction_anchor(runtime_args)

    if runtime_args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=runtime_args.quant_only_start_epoch is not None,
            static_graph=runtime_args.static_graph,
            gradient_as_bucket_view=runtime_args.gradient_as_bucket_view,
        )

    ref_model = None
    anchor_ref_model = None
    confidence_ref_model = None
    if (
        runtime_args.ref_confidence_band_kd_weight > 0
        or runtime_args.local_ref_confidence_band_kd_weight > 0
        or runtime_args.class_protect_ref_kl_weight > 0
    ):
        ref_checkpoint = (
            runtime_args.local_ref_confidence_band_kd_checkpoint
            or runtime_args.ref_confidence_band_kd_checkpoint
            or runtime_args.class_protect_ref_kl_checkpoint
        )
        confidence_ref_model = clone_fixed_logit_ref_model(
            model,
            ref_checkpoint,
        )
        if runtime_args.local_rank == 0:
            source = ref_checkpoint or "current model"
            if runtime_args.ref_confidence_band_kd_weight > 0:
                print(
                    "Enabled reference-confidence band KD: "
                    f"weight={runtime_args.ref_confidence_band_kd_weight}, "
                    f"band=[{runtime_args.ref_confidence_band_kd_low}, {runtime_args.ref_confidence_band_kd_high}), "
                    f"temperature={runtime_args.ref_confidence_band_kd_temperature}, source={source}"
                )
            if runtime_args.local_ref_confidence_band_kd_weight > 0:
                print(
                    "Enabled local-reference confidence band KD: "
                    f"weight={runtime_args.local_ref_confidence_band_kd_weight}, "
                    f"band=[{runtime_args.local_ref_confidence_band_kd_low}, {runtime_args.local_ref_confidence_band_kd_high}), "
                    f"temperature={runtime_args.local_ref_confidence_band_kd_temperature}, source={source}"
                )
            if runtime_args.class_protect_ref_kl_weight > 0:
                print(
                    "Enabled class-protect ref KL: "
                    f"weight={runtime_args.class_protect_ref_kl_weight}, "
                    f"classes={runtime_args.class_protect_ref_kl_classes}, "
                    f"temperature={runtime_args.class_protect_ref_kl_temperature}, source={source}"
                )
    if runtime_args.train_scheme == "ema_ref_attn_kl":
        ref_model = clone_ref_model(model)
        if runtime_args.anchor_ref_attn_kl_weight > 0 or runtime_args.anchor_ref_attn_kl_weight_epoch_overrides:
            anchor_ref_model = clone_ref_model(model)
        selected_head_map, anchor_selected_head_map = apply_ref_head_mode_to_models(
            runtime_args.ref_head_mode,
            model,
            teacher,
            ref_model,
            anchor_ref_model,
            runtime_args,
        )
        anchor_ref_head_mode = runtime_args.anchor_ref_head_mode or runtime_args.ref_head_mode
        if runtime_args.local_rank == 0:
            print(
                "Enabled EMA refmodel attention-KL scheme: "
                f"ref_update={runtime_args.ref_update}, "
                f"ref_update_interval={runtime_args.ref_update_interval}, "
                f"momentum={runtime_args.ref_momentum}, "
                f"attn_kl_weight={runtime_args.ref_attn_kl_weight}, "
                f"anchor_attn_kl_weight={runtime_args.anchor_ref_attn_kl_weight}, "
                f"head_mode={runtime_args.ref_head_mode}, "
                f"anchor_head_mode={anchor_ref_head_mode}, "
                f"selected_head_map={selected_head_map}, "
                f"anchor_selected_head_map={anchor_selected_head_map}, "
                f"warmup_epochs={runtime_args.ref_warmup_epochs}, "
                f"anchor_warmup_epochs={runtime_args.anchor_ref_warmup_epochs}"
            )
    if runtime_args.local_rank == 0:
        print(f"Model {safe_model_name(runtime_args.model)} created, param count:{sum(m.numel() for m in model.parameters())}")
        print(f"Scheduled epochs: {runtime_args.epochs}")
        print(
            "Effective batch alignment: "
            f"per_gpu_effective_batch={runtime_args.batch_size}, "
            f"loader_batch={runtime_args.forward_micro_batch_size if runtime_args.forward_micro_batch_size > 0 else runtime_args.batch_size}, "
            f"accum={runtime_args.grad_accum_steps}, world_size={runtime_args.world_size}, "
            f"global_effective_batch={runtime_args.effective_batch_size}"
        )

    train_loss_fn = create_ofq_loss(runtime_args)
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    output_dir = Path(runtime_args.output) / runtime_args.experiment
    dynamic_kl_controller = DynamicSparsePrevStepKLController(runtime_args, output_dir)
    if runtime_args.local_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "args.yaml", "w", encoding="utf-8") as handle:
            public_args = {key: value for key, value in vars(runtime_args).items() if not str(key).startswith("_")}
            handle.write(yaml.safe_dump(public_args, allow_unicode=True, sort_keys=True))
        dynamic_kl_controller.initialize_log(runtime_args.local_rank)
        if dynamic_kl_controller.enabled:
            print(
                "Enabled dynamic sparse prev-step KL controller: "
                f"start_epoch={dynamic_kl_controller.start_epoch}, "
                f"observe_until={dynamic_kl_controller.observe_until_epoch}, "
                f"primary_heads={[format_dynamic_kl_head(h) for h in dynamic_kl_controller.primary_heads]}, "
                f"secondary_heads={[format_dynamic_kl_head(h) for h in dynamic_kl_controller.secondary_heads]}, "
                f"avoid_heads={[format_dynamic_kl_head(h) for h in sorted(dynamic_kl_controller.avoid_heads)]}, "
                f"drop_threshold={dynamic_kl_controller.drop_threshold}, "
                f"strong_drop_threshold={dynamic_kl_controller.strong_drop_threshold}, "
                f"default_weight={dynamic_kl_controller.default_weight}, "
                f"strong_weight={dynamic_kl_controller.strong_weight}, "
                f"max_weight={dynamic_kl_controller.max_weight}, "
                f"prior_source={dynamic_kl_controller.prior_source}, "
                f"controller_tsv={dynamic_kl_controller.tsv_path}"
            )
        if runtime_args.save_step_checkpoints and runtime_args.save_initial_step_checkpoint and int(runtime_args.step_checkpoint_warmup_updates) == 0:
            save_step_checkpoint(model, optimizer, runtime_args, output_dir, "step_0000", epoch=start_epoch, batch_idx=-1, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, model_ema=model_ema)

    if runtime_args.aoq_explore_quality_mode in {"anchor_unmoved", "anchor_moved"}:
        runtime_args._aoq_explore_anchor_state = load_aoq_anchor_state(runtime_args.aoq_explore_anchor_checkpoint)
        if runtime_args.local_rank == 0:
            print(
                f"Loaded AOQ anchor checkpoint for {runtime_args.aoq_explore_quality_mode} selector: "
                f"path={runtime_args.aoq_explore_anchor_checkpoint}, tensors={len(runtime_args._aoq_explore_anchor_state)}"
            )
    if float(getattr(runtime_args, "candidate_bin_anchor_weight", 0.0)) > 0:
        runtime_args._candidate_bin_anchor_source_state = load_aoq_anchor_state(runtime_args.candidate_bin_anchor_source_checkpoint)
        if runtime_args.local_rank == 0:
            print(
                "Loaded candidate-bin anchor source checkpoint: "
                f"path={runtime_args.candidate_bin_anchor_source_checkpoint}, "
                f"tensors={len(runtime_args._candidate_bin_anchor_source_state)}"
            )

    try:
        active_wbits = int(runtime_args.wq_bitw)
        active_abits = int(runtime_args.aq_bitw)
        base_ref_head_mode = str(runtime_args.ref_head_mode)
        base_ref_attn_kl_weight = float(runtime_args.ref_attn_kl_weight)
        for epoch in range(start_epoch, runtime_args.epochs):
            if runtime_args.local_rank == 0:
                epoch_ref_head_mode, epoch_ref_attn_kl_weight, epoch_dynamic_head, epoch_dynamic_spike_score, epoch_dynamic_reason = (
                    dynamic_kl_controller.decision_for_epoch(epoch)
                )
            else:
                epoch_ref_head_mode, epoch_ref_attn_kl_weight, epoch_dynamic_head, epoch_dynamic_spike_score, epoch_dynamic_reason = (
                    base_ref_head_mode,
                    base_ref_attn_kl_weight,
                    None,
                    0.0,
                    "rank_wait",
                )
            epoch_ref_head_mode, epoch_ref_attn_kl_weight, epoch_dynamic_head, epoch_dynamic_spike_score, epoch_dynamic_reason = (
                broadcast_dynamic_kl_epoch_decision(
                    runtime_args,
                    epoch_ref_head_mode,
                    epoch_ref_attn_kl_weight,
                    epoch_dynamic_head,
                    epoch_dynamic_spike_score,
                    epoch_dynamic_reason,
                )
            )
            if not dynamic_kl_controller.enabled:
                epoch_ref_head_mode = epoch_string_value(
                    runtime_args.ref_head_mode_epoch_overrides,
                    epoch,
                    base_ref_head_mode,
                )
            runtime_args.ref_head_mode = epoch_ref_head_mode
            runtime_args.ref_attn_kl_weight = float(epoch_ref_attn_kl_weight)
            if runtime_args.train_scheme == "ema_ref_attn_kl":
                selected_head_map, anchor_selected_head_map = apply_ref_head_mode_to_models(
                    runtime_args.ref_head_mode,
                    model,
                    teacher,
                    ref_model,
                    anchor_ref_model,
                    runtime_args,
                )
                if runtime_args.local_rank == 0 and dynamic_kl_controller.enabled:
                    print(
                        "DynamicSparsePrevStepKLControllerApply: "
                        f"epoch={epoch}, head_mode={runtime_args.ref_head_mode}, "
                        f"head={format_dynamic_kl_head(epoch_dynamic_head)}, "
                        f"weight={float(epoch_ref_attn_kl_weight):.3e}, spike_score={float(epoch_dynamic_spike_score):.6f}, "
                        f"reason={epoch_dynamic_reason}, "
                        f"selected_head_map={selected_head_map}, anchor_selected_head_map={anchor_selected_head_map}"
                    )
            teacher_attn_output_weight_override = runtime_args.teacher_attn_output_weight_epoch_overrides.get(epoch)
            if teacher_attn_output_weight_override is not None:
                runtime_args.teacher_attn_output_weight = float(teacher_attn_output_weight_override)
                if runtime_args.local_rank == 0:
                    print(
                        "Applied teacher attention-output weight override: "
                        f"epoch={epoch}, weight={teacher_attn_output_weight_override}"
                    )
            if runtime_args.progressive_bit_schedule:
                scheduled_wbits, scheduled_abits = progressive_bits_for_epoch(
                    runtime_args.progressive_bit_schedule,
                    epoch,
                    runtime_args.wq_bitw,
                    runtime_args.aq_bitw,
                )
                transition_anchor_model = None
                transition_anchor_weight = float(getattr(runtime_args, "progressive_bit_transition_anchor_kl_weight", 0.0))
                is_down_transition = scheduled_wbits < active_wbits or scheduled_abits < active_abits
                forced_transition_epochs = set(getattr(runtime_args, "progressive_bit_transition_recon_epochs", set()) or set())
                force_transition_recon = epoch in forced_transition_epochs
                if transition_anchor_weight > 0 and (is_down_transition or force_transition_recon):
                    transition_anchor_model = copy.deepcopy(maybe_unwrap_ddp(model)).cuda()
                    transition_anchor_model.eval()
                    for param in transition_anchor_model.parameters():
                        param.requires_grad_(False)
                    if runtime_args.local_rank == 0:
                        print(
                            "Captured progressive bit transition anchor model: "
                            f"epoch={epoch}, from=W{active_wbits}A{active_abits}, "
                            f"to=W{scheduled_wbits}A{scheduled_abits}, "
                            f"kl_weight={transition_anchor_weight}, "
                            f"temperature={runtime_args.progressive_bit_transition_anchor_kl_temperature}"
                        )
                weight_modules, act_modules = set_fake_quant_bits(
                    model,
                    scheduled_wbits,
                    scheduled_abits,
                    rescale_lsq=runtime_args.progressive_bit_rescale_lsq,
                )
                runtime_args.wq_bitw = scheduled_wbits
                runtime_args.aq_bitw = scheduled_abits
                if runtime_args.local_rank == 0:
                    print(
                        "Applied progressive fake-quant bits: "
                        f"epoch={epoch} wbits={scheduled_wbits} abits={scheduled_abits} "
                        f"weight_modules={weight_modules} act_modules={act_modules}"
                    )
                if epoch in runtime_args.progressive_bit_recalibrate_epochs:
                    recalibrated = recalibrate_lsq_alpha_preserve_params(
                        model,
                        loader_train,
                        runtime_args,
                        amp_autocast,
                        runtime_args.progressive_bit_recalibrate_batches,
                    )
                    if runtime_args.local_rank == 0:
                        print(
                            "Applied progressive bit alpha recalibration: "
                            f"epoch={epoch} batches={runtime_args.progressive_bit_recalibrate_batches} "
                            f"quantizers={recalibrated}"
                        )
                run_progressive_bit_transition_reconstruction(
                    model,
                    loader_train,
                    optimizer,
                    runtime_args,
                    amp_autocast,
                    teacher,
                    epoch,
                    previous_bits=(active_wbits, active_abits),
                    current_bits=(scheduled_wbits, scheduled_abits),
                    anchor_model=transition_anchor_model,
                )
                del transition_anchor_model
                active_wbits = scheduled_wbits
                active_abits = scheduled_abits
            epoch_lr_override = runtime_args.epoch_lr_overrides.get(epoch)
            if epoch_lr_override is not None:
                lr_scheduler.base_lr = float(epoch_lr_override)
                lr_scheduler.min_lr = float(epoch_lr_override)
                set_optimizer_lr(optimizer, float(epoch_lr_override))
                if runtime_args.local_rank == 0:
                    print(f"Applied epoch LR override: epoch={epoch}, lr={epoch_lr_override}")
            quant_lr_multiplier_override = runtime_args.quant_lr_multiplier_epoch_overrides.get(epoch)
            if quant_lr_multiplier_override is not None:
                updated_groups = set_quant_lr_multiplier(optimizer, float(quant_lr_multiplier_override))
                if runtime_args.local_rank == 0:
                    print(
                        "Applied quant LR multiplier override: "
                        f"epoch={epoch}, multiplier={quant_lr_multiplier_override}, groups={updated_groups}"
                    )
            quant_only_enabled = runtime_args.quant_only_start_epoch is not None and epoch >= runtime_args.quant_only_start_epoch
            active_trainable_policy = runtime_args.trainable_policy if quant_only_enabled else "all"
            trainable_params, frozen_params = set_trainable_policy(model, active_trainable_policy, runtime_args=runtime_args)
            if runtime_args.local_rank == 0:
                print(
                    "Trainable parameter policy: "
                    f"epoch={epoch}, quant_only={quant_only_enabled}, policy={active_trainable_policy}, "
                    f"trainable={trainable_params}, frozen={frozen_params}"
                )
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
                anchor_ref_model,
                model_ema,
                confidence_ref_model,
            )
            if runtime_args.local_rank == 0:
                print("epoch: ", epoch, "g['lr']: ", optimizer.param_groups[0]["lr"])
            post_epoch_feature_recon_updates = int(getattr(runtime_args, "post_epoch_feature_recon_updates", 0))
            if post_epoch_feature_recon_updates > 0:
                run_pre_qat_feature_reconstruction(
                    model,
                    loader_train,
                    optimizer,
                    runtime_args,
                    amp_autocast,
                    teacher,
                    updates_override=post_epoch_feature_recon_updates,
                    label=f"post-epoch feature reconstruction epoch={epoch}",
                    bypass_ddp=True,
                )
            should_save_epoch_checkpoint = (
                (epoch + 1) % max(1, runtime_args.epoch_checkpoint_interval) == 0
                or (epoch + 1) == runtime_args.epochs
                or stopped_early
            )
            if runtime_args.local_rank == 0 and should_save_epoch_checkpoint:
                save_epoch_checkpoint(
                    model,
                    optimizer,
                    runtime_args,
                    output_dir,
                    epoch,
                    loss_scaler=loss_scaler,
                    lr_scheduler=lr_scheduler,
                    model_ema=model_ema,
                )
                if model_ema is not None:
                    save_epoch_checkpoint(
                        model_ema,
                        optimizer,
                        runtime_args,
                        output_dir,
                        epoch,
                        loss_scaler=loss_scaler,
                        suffix=".ema",
                        lr_scheduler=lr_scheduler,
                    )
            should_validate = (
                loader_eval is not None
                and (
                    ((epoch + 1) % max(1, int(getattr(runtime_args, "val_interval", 1))) == 0)
                    or ((epoch + 1) == runtime_args.epochs)
                    or stopped_early
                )
            )
            if runtime_args.distributed and loader_eval is not None and should_validate:
                dist.barrier()
            val_metrics = None
            if should_validate:
                val_metrics = validate_ofq(model, loader_eval, validate_loss_fn, runtime_args, amp_autocast)
                dynamic_kl_controller.update_after_validation(
                    epoch,
                    val_metrics,
                    epoch_dynamic_head,
                    float(epoch_ref_attn_kl_weight),
                    float(epoch_dynamic_spike_score),
                    runtime_args.local_rank,
                )
            if (
                val_metrics is not None
                and epoch == 0
                and float(getattr(runtime_args, "epoch1_acc_gate", 0.0)) > 0.0
            ):
                epoch1_top1 = float(val_metrics.get("top1", 0.0))
                gate_top1 = float(runtime_args.epoch1_acc_gate)
                failed_gate = epoch1_top1 <= gate_top1
                if runtime_args.local_rank == 0:
                    status = "failed" if failed_gate else "passed"
                    print(
                        "Epoch1AccGate: "
                        f"{status} top1={epoch1_top1:.4f} threshold>{gate_top1:.4f} "
                        f"samples={val_metrics.get('samples')}"
                    )
                if failed_gate:
                    stopped_early = True
            if stopped_early:
                if runtime_args.local_rank == 0:
                    print(f"Stopped early after {local_update_count} optimizer updates in epoch {epoch}.")
                runtime_args.ref_head_mode = base_ref_head_mode
                runtime_args.ref_attn_kl_weight = base_ref_attn_kl_weight
                break
            runtime_args.ref_head_mode = base_ref_head_mode
            runtime_args.ref_attn_kl_weight = base_ref_attn_kl_weight
    finally:
        shutdown_data_loader(loader_eval)
        shutdown_data_loader(loader_train)
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
        backend = None
        try:
            try:
                backend = dist.get_backend()
            except Exception:
                backend = None
            current_device = None
            if torch.cuda.is_available():
                try:
                    current_device = torch.cuda.current_device()
                except Exception:
                    current_device = None
            if backend == "nccl" and current_device is not None:
                dist.barrier(device_ids=[current_device])
            else:
                dist.barrier()
        except Exception as exc:
            print(f"[qat_launch] warning: dist.barrier() before destroy_process_group failed: {exc}", flush=True)
        if backend == "nccl":
            print("[qat_launch] skip dist.destroy_process_group() for NCCL to avoid teardown hang on process exit", flush=True)
            return
        try:
            dist.destroy_process_group()
        except Exception as exc:
            print(f"[qat_launch] warning: dist.destroy_process_group() failed: {exc}", flush=True)


def _close_dataset_resources(dataset) -> None:
    if dataset is None:
        return
    seen = set()

    def _close(obj) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))

        file_handles = getattr(obj, "_file_handles", None)
        if isinstance(file_handles, dict):
            for handle in list(file_handles.values()):
                close_fn = getattr(handle, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception as exc:
                        print(f"[qat_launch] warning: dataset handle close failed: {exc}", flush=True)
            try:
                file_handles.clear()
            except Exception:
                pass

        close_fn = getattr(obj, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

        nested = getattr(obj, "dataset", None)
        if nested is not None and nested is not obj:
            _close(nested)

        nested_datasets = getattr(obj, "datasets", None)
        if isinstance(nested_datasets, (list, tuple)):
            for item in nested_datasets:
                _close(item)

    _close(dataset)


def shutdown_data_loader(loader) -> None:
    if loader is None:
        return

    seen = set()

    def _shutdown(obj) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))

        nested_loader = getattr(obj, "loader", None)
        if nested_loader is not None and nested_loader is not obj:
            _shutdown(nested_loader)
        nested_loader = getattr(obj, "_loader", None)
        if nested_loader is not None and nested_loader is not obj:
            _shutdown(nested_loader)

        iterator = getattr(obj, "_iterator", None)
        shutdown_fn = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except Exception as exc:
                print(f"[qat_launch] warning: dataloader worker shutdown failed: {exc}", flush=True)
            try:
                obj._iterator = None
            except Exception:
                pass

        dataset = getattr(obj, "dataset", None)
        _close_dataset_resources(dataset)

    _shutdown(loader)


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
    parser.add_argument("--forward-micro-batch-size", dest="forward_micro_batch_size", type=int, default=None, help="split each train batch into micro forwards with gradient accumulation")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float)
    parser.add_argument("--warmup-epochs", dest="warmup_epochs", type=int)
    parser.add_argument("--warmup-lr", dest="warmup_lr", type=float)
    parser.add_argument("--scheduler-epochs", dest="scheduler_epochs", type=int)
    parser.add_argument("--min-lr", dest="min_lr", type=float)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--no-resume-opt", dest="no_resume_opt", action="store_true")
    parser.add_argument("--resume-opt-force-lr", dest="resume_opt_force_lr", action="store_true", help="恢复 optimizer state 后强制把 param group lr 改回命令行 --lr")
    parser.add_argument("--start-epoch", dest="start_epoch", type=int)
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
    parser.add_argument("--quant-teacher", dest="quant_teacher", action="store_true", help="quantize teacher before loading teacher checkpoint")
    parser.add_argument("--pretrained-initialized", dest="pretrained_initialized", action="store_true")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--qk-reparam", dest="qk_reparam", action="store_true")
    parser.add_argument("--qk-reparam-type", dest="qk_reparam_type", type=int)
    parser.add_argument("--boundary-range", dest="boundary_range", type=float)
    parser.add_argument("--freeze-for-n-epochs", dest="freeze_for_n_epochs", type=int)
    parser.add_argument("--train-scheme", dest="train_scheme", choices=["baseline", "ema_ref_attn_kl"], default=None, help="OFQ 训练方案名")
    parser.add_argument("--ref-update", dest="ref_update", choices=["ema", "prev_step", "fixed"], default=None, help="历史参考模型更新方式")
    parser.add_argument("--ref-update-interval", dest="ref_update_interval", type=int, default=None, help="prev_step refmodel 每隔多少个 optimizer update 刷新一次")
    parser.add_argument("--ref-momentum", dest="ref_momentum", type=float, default=None, help="EMA refmodel 动量")
    parser.add_argument("--ref-attn-kl-weight", dest="ref_attn_kl_weight", type=float, default=None, help="EMA refmodel attention KL 权重")
    parser.add_argument("--ref-attn-kl-drop-prob", dest="ref_attn_kl_drop_prob", type=float, default=None, help="probability of applying prev-step attention KL on each update; 1 keeps old behavior")
    parser.add_argument("--ref-attn-kl-drop-scale", dest="ref_attn_kl_drop_scale", type=str2bool, nargs="?", const=True, default=None, help="when KL dropout is enabled, scale active KL by 1/p to preserve expected weight")
    parser.add_argument("--ref-attn-kl-clip", dest="ref_attn_kl_clip", type=float, default=None, help="clip per-head ref attention KL before top-k/averaging; <=0 disables")
    parser.add_argument("--ref-attn-loss", dest="ref_attn_loss", choices=["kl_ref", "symmetric_kl", "js", "cosine", "centered_cosine"], default=None, help="refmodel attention consistency loss")
    parser.add_argument("--ref-logit-kl-weight", dest="ref_logit_kl_weight", type=float, default=None, help="refmodel logits KL 权重")
    parser.add_argument("--ref-logit-kl-temperature", dest="ref_logit_kl_temperature", type=float, default=None, help="refmodel logits KL temperature")
    parser.add_argument("--teacher-qk-rel-weight", dest="teacher_qk_rel_weight", type=float, default=None, help="FP teacher Q/K relation MSE 权重")
    parser.add_argument("--teacher-qk-rel-warmup-epochs", dest="teacher_qk_rel_warmup_epochs", type=int, default=None, help="多少个 epoch 后启用 FP teacher Q/K relation MSE")
    parser.add_argument("--teacher-qkv-rel-weight", dest="teacher_qkv_rel_weight", type=float, default=None, help="FP teacher 局部 Q/K/V relation distillation 权重")
    parser.add_argument("--teacher-qkv-rel-warmup-epochs", dest="teacher_qkv_rel_warmup_epochs", type=int, default=None, help="多少个 epoch 后启用局部 Q/K/V relation distillation")
    parser.add_argument("--teacher-qkv-rel-layers", dest="teacher_qkv_rel_layers", type=str, default=None, help="局部 Q/K/V relation distillation 的 attention layer index 列表；all 表示全部")
    parser.add_argument("--teacher-qkv-rel-components", dest="teacher_qkv_rel_components", type=str, default=None, help="局部 relation 使用哪些分量，逗号分隔 q,k,v")
    parser.add_argument("--clean-start-target-loss-weight", dest="clean_start_target_loss_weight", type=float, default=None, help="early stage clean-start hard-label CE auxiliary weight for KD runs")
    parser.add_argument("--ref-head-mode", dest="ref_head_mode", type=str, default=None, help="refmodel head 级别接口: all, oscillating_top5/top10/top15, or custom:layer:head,...")
    parser.add_argument("--ref-warmup-epochs", dest="ref_warmup_epochs", type=int, default=None, help="多少个 epoch 后再启用 refmodel attention KL")
    parser.add_argument("--ref-warmup-updates", dest="ref_warmup_updates", type=int, default=None, help="多少个 optimizer update 后再启用 refmodel consistency loss")
    parser.add_argument("--ref-stop-updates", dest="ref_stop_updates", type=int, default=None, help="达到多少个 optimizer update 后关闭 refmodel consistency loss；0 表示不关闭")
    parser.add_argument("--anchor-ref-attn-kl-weight", dest="anchor_ref_attn_kl_weight", type=float, default=None, help="固定 anchor refmodel attention KL 权重")
    parser.add_argument("--anchor-ref-warmup-epochs", dest="anchor_ref_warmup_epochs", type=int, default=None, help="多少个 epoch 后启用 anchor refmodel attention KL")
    parser.add_argument("--anchor-ref-head-mode", dest="anchor_ref_head_mode", type=str, default=None, help="固定 anchor refmodel 单独使用的 head mode；为空则沿用 --ref-head-mode")
    parser.add_argument("--teacher-attn-kl-weight", dest="teacher_attn_kl_weight", type=float, default=None, help="FP teacher attention KL 权重")
    parser.add_argument("--teacher-attn-kl-warmup-epochs", dest="teacher_attn_kl_warmup_epochs", type=int, default=None, help="多少个 epoch 后启用 FP teacher attention KL")
    parser.add_argument("--teacher-attn-output-weight", dest="teacher_attn_output_weight", type=float, default=None, help="FP teacher attention module output MSE 权重")
    parser.add_argument("--teacher-attn-output-layers", dest="teacher_attn_output_layers", type=str, default=None, help="teacher attention output MSE 层选择: all 或逗号分隔 attention layer index")
    parser.add_argument("--teacher-attn-output-warmup-epochs", dest="teacher_attn_output_warmup_epochs", type=int, default=None, help="多少个 epoch 后启用 FP teacher attention output MSE")
    parser.add_argument("--teacher-attn-output-weight-epoch-overrides", dest="teacher_attn_output_weight_epoch_overrides", type=str, default=None, help="按 epoch 覆盖 teacher attention output MSE 权重，格式 epoch:value,epoch:value")
    parser.add_argument("--teacher-feature-output-weight", dest="teacher_feature_output_weight", type=float, default=None, help="FP teacher named feature output MSE 权重")
    parser.add_argument("--teacher-feature-output-layers", dest="teacher_feature_output_layers", type=str, default=None, help="逗号分隔 module 名称，例如 features.1,features.3,features.5,features.7")
    parser.add_argument("--teacher-feature-output-warmup-epochs", dest="teacher_feature_output_warmup_epochs", type=int, default=None, help="多少个 epoch 后启用 FP teacher feature output MSE")
    parser.add_argument("--teacher-feature-output-loss", dest="teacher_feature_output_loss", choices=["mse", "norm_mse"], default=None, help="FP teacher feature output loss: raw MSE or energy-normalized MSE")
    parser.add_argument("--bin-reg-weight", dest="bin_reg_weight", type=float, default=None, help="VVTQ-style quantized weight bin regularizer weight; 0 disables")
    parser.add_argument("--bin-reg-variance-weight", dest="bin_reg_variance_weight", type=float, default=None, help="relative weight for within-bin FP weight variance term")
    parser.add_argument("--bin-reg-layers", dest="bin_reg_layers", type=str, default=None, help="BinReg 作用的 named module 列表；空表示全模型")
    parser.add_argument("--bin-reg-attn-only", dest="bin_reg_attn_only", action="store_true", help="BinReg 只作用于 attention q/k/v quantized weights，不作用于普通 QLinear weights")
    parser.add_argument("--bin-reg-start-update", dest="bin_reg_start_update", type=int, default=None, help="延迟到第几个 local optimizer update 开始启用 BinReg；0 表示训练开始")
    parser.add_argument("--bin-reg-end-update", dest="bin_reg_end_update", type=int, default=None, help="第几个 local optimizer update 后关闭 BinReg；0 表示不关闭")
    parser.add_argument("--selective-bin-anchor-weight", dest="selective_bin_anchor_weight", type=float, default=None, help="AOQ 后选择性离散 bin anchor 权重；0 关闭")
    parser.add_argument("--selective-bin-anchor-layers", dest="selective_bin_anchor_layers", type=str, default=None, help="Selective bin anchor 作用的 LSQ weight module 列表；空表示全模型 LSQ weight")
    parser.add_argument("--selective-bin-anchor-capture-update", dest="selective_bin_anchor_capture_update", type=int, default=None, help="第几个 local optimizer update 捕获 selective bin anchor")
    parser.add_argument("--selective-bin-anchor-end-update", dest="selective_bin_anchor_end_update", type=int, default=None, help="第几个 local optimizer update 后关闭 selective bin anchor；0 表示不关闭")
    parser.add_argument("--selective-bin-anchor-margin", dest="selective_bin_anchor_margin", type=float, default=None, help="捕获 anchor mask 的 bin-boundary 距离阈值，范围 [0, 0.5]")
    parser.add_argument("--candidate-bin-anchor-weight", dest="candidate_bin_anchor_weight", type=float, default=None, help="AOQ 后保留相对 source 已跨 bin 候选权重的 anchor 权重；0 关闭")
    parser.add_argument("--candidate-bin-anchor-layers", dest="candidate_bin_anchor_layers", type=str, default=None, help="Candidate-bin anchor 作用的 LSQ weight module 列表；空表示全模型 LSQ weight")
    parser.add_argument("--candidate-bin-anchor-capture-update", dest="candidate_bin_anchor_capture_update", type=int, default=None, help="第几个 local optimizer update 捕获 candidate-bin anchor")
    parser.add_argument("--candidate-bin-anchor-end-update", dest="candidate_bin_anchor_end_update", type=int, default=None, help="第几个 local optimizer update 后关闭 candidate-bin anchor；0 表示不关闭")
    parser.add_argument("--candidate-bin-anchor-source-checkpoint", dest="candidate_bin_anchor_source_checkpoint", type=str, default=None, help="candidate-bin anchor 用于判定已跨 bin 的 source checkpoint")
    parser.add_argument("--weight-bin-telemetry-layers", dest="weight_bin_telemetry_layers", type=str, default=None, help="记录 integer-bin telemetry 的 LSQ weight module 列表；空表示全模型 LSQ weight")
    parser.add_argument("--weight-bin-telemetry-start-update", dest="weight_bin_telemetry_start_update", type=int, default=None, help="第几个 local optimizer update 开始记录 weight-bin telemetry")
    parser.add_argument("--weight-bin-telemetry-end-update", dest="weight_bin_telemetry_end_update", type=int, default=None, help="第几个 local optimizer update 后停止记录；0 表示不停止")
    parser.add_argument("--weight-bin-telemetry-interval", dest="weight_bin_telemetry_interval", type=int, default=None, help="每隔多少 local optimizer update 记录一次；0 关闭")
    parser.add_argument("--weight-bin-telemetry-margin", dest="weight_bin_telemetry_margin", type=float, default=None, help="near-boundary 统计的 bin-boundary 距离阈值，范围 [0, 0.5]")
    parser.add_argument("--epoch1-acc-gate", dest="epoch1_acc_gate", type=float, default=None, help="stop after first epoch validation if full-val Top-1 is below this threshold; 0 disables")
    parser.add_argument("--teacher-confidence-kd-power", dest="teacher_confidence_kd_power", type=float, default=None, help="teacher soft KD 样本权重的置信度幂次；0 关闭")
    parser.add_argument("--teacher-confidence-band-kd-weight", dest="teacher_confidence_band_kd_weight", type=float, default=None, help="extra soft KD weight for samples whose teacher confidence falls in a selected band")
    parser.add_argument("--teacher-confidence-band-kd-low", dest="teacher_confidence_band_kd_low", type=float, default=None, help="lower bound for teacher confidence band KD")
    parser.add_argument("--teacher-confidence-band-kd-high", dest="teacher_confidence_band_kd_high", type=float, default=None, help="upper bound for teacher confidence band KD")
    parser.add_argument("--teacher-confidence-band-kd-temperature", dest="teacher_confidence_band_kd_temperature", type=float, default=None, help="temperature for teacher confidence band KD")
    parser.add_argument("--ref-confidence-band-kd-weight", dest="ref_confidence_band_kd_weight", type=float, default=None, help="extra soft KD weight selected by a fixed reference model confidence band")
    parser.add_argument("--ref-confidence-band-kd-low", dest="ref_confidence_band_kd_low", type=float, default=None, help="lower bound for reference confidence band KD")
    parser.add_argument("--ref-confidence-band-kd-high", dest="ref_confidence_band_kd_high", type=float, default=None, help="upper bound for reference confidence band KD")
    parser.add_argument("--ref-confidence-band-kd-temperature", dest="ref_confidence_band_kd_temperature", type=float, default=None, help="temperature for reference confidence band KD")
    parser.add_argument("--ref-confidence-band-kd-checkpoint", dest="ref_confidence_band_kd_checkpoint", type=str, default=None, help="optional fixed checkpoint used only to select confidence-band samples")
    parser.add_argument("--local-ref-confidence-band-kd-weight", dest="local_ref_confidence_band_kd_weight", type=float, default=None, help="fixed local reference soft KD weight for samples whose local reference confidence falls in a selected band")
    parser.add_argument("--local-ref-confidence-band-kd-low", dest="local_ref_confidence_band_kd_low", type=float, default=None, help="lower bound for local reference confidence band KD")
    parser.add_argument("--local-ref-confidence-band-kd-high", dest="local_ref_confidence_band_kd_high", type=float, default=None, help="upper bound for local reference confidence band KD")
    parser.add_argument("--local-ref-confidence-band-kd-temperature", dest="local_ref_confidence_band_kd_temperature", type=float, default=None, help="temperature for local reference confidence band KD")
    parser.add_argument("--local-ref-confidence-band-kd-checkpoint", dest="local_ref_confidence_band_kd_checkpoint", type=str, default=None, help="fixed checkpoint used as local teacher and confidence-band selector")
    parser.add_argument("--class-protect-ref-kl-weight", dest="class_protect_ref_kl_weight", type=float, default=None, help="fixed reference logit KL weight for selected target classes")
    parser.add_argument("--class-protect-ref-kl-classes", dest="class_protect_ref_kl_classes", type=str, default=None, help="comma-separated target class ids protected by fixed reference logit KL")
    parser.add_argument("--class-protect-ref-kl-temperature", dest="class_protect_ref_kl_temperature", type=float, default=None, help="temperature for class-protect fixed reference logit KL")
    parser.add_argument("--class-protect-ref-kl-checkpoint", dest="class_protect_ref_kl_checkpoint", type=str, default=None, help="optional fixed checkpoint used for class-protect ref KL")
    parser.add_argument("--teacher-soft-temperature", dest="teacher_soft_temperature", type=float, default=None, help="teacher soft KD temperature；1 使用原始 OFQ soft KD")
    parser.add_argument("--quant-lr-multiplier", dest="quant_lr_multiplier", type=float, default=None, help="quant/shift 参数相对 base lr 的倍率；1 关闭分组 LR")
    parser.add_argument("--quant-lr-multiplier-epoch-overrides", dest="quant_lr_multiplier_epoch_overrides", type=str, default=None, help="按 epoch 覆盖 quant/shift 参数 LR 倍率，格式 epoch:value,epoch:value")
    parser.add_argument("--quant-slow-state-decay", dest="quant_slow_state_decay", type=float, default=None, help="QSS quant/shift shadow EMA decay；0 关闭")
    parser.add_argument("--quant-slow-state-sync-interval", dest="quant_slow_state_sync_interval", type=int, default=None, help="QSS 每隔多少 optimizer updates 将 student quant/shift 拉向 shadow")
    parser.add_argument("--quant-slow-state-pull", dest="quant_slow_state_pull", type=float, default=None, help="QSS 同步时 shadow 注入比例")
    parser.add_argument("--quant-slow-state-policy", dest="quant_slow_state_policy", choices=["all", "activation"], default=None, help="QSS 维护哪些 quant/shift 参数")
    parser.add_argument("--quant-slow-state-observe-start-epoch", dest="quant_slow_state_observe_start_epoch", type=int, default=None, help="QSS 从第几个 epoch 开始维护 shadow EMA")
    parser.add_argument("--quant-slow-state-start-epoch", dest="quant_slow_state_start_epoch", type=int, default=None, help="QSS 从第几个 epoch 开始启用")
    parser.add_argument("--act-scale-anchor-weight", dest="act_scale_anchor_weight", type=float, default=None, help="activation LSQ scale anchor regularizer weight; 0 disables")
    parser.add_argument("--act-scale-anchor-layers", dest="act_scale_anchor_layers", type=str, default=None, help="activation scale anchor 作用的 named module 列表；空表示全部 activation quantizers")
    parser.add_argument("--act-scale-anchor-start-epoch", dest="act_scale_anchor_start_epoch", type=int, default=None, help="从第几个 epoch 开始启用 activation scale anchor")
    parser.add_argument("--variation-trust-weight", dest="variation_trust_weight", type=float, default=None, help="VVTQ-inspired variation-weighted trust regularizer weight; 0 disables")
    parser.add_argument("--variation-trust-layers", dest="variation_trust_layers", type=str, default=None, help="variation trust 只采集这些 named module 下的 activation/shift 参数；空表示全部")
    parser.add_argument("--variation-trust-late-layers", dest="variation_trust_late_layers", type=str, default=None, help="variation trust 中弱 anchor 的 late named module 列表")
    parser.add_argument("--variation-trust-late-multiplier", dest="variation_trust_late_multiplier", type=float, default=None, help="late layers 的 trust multiplier，<1 表示弱 anchor")
    parser.add_argument("--variation-trust-early-layers", dest="variation_trust_early_layers", type=str, default=None, help="variation trust 中强 anchor 的 early/high-risk named module 列表")
    parser.add_argument("--variation-trust-early-multiplier", dest="variation_trust_early_multiplier", type=float, default=None, help="early/high-risk layers 的 trust multiplier")
    parser.add_argument("--variation-trust-softmax-multiplier", dest="variation_trust_softmax_multiplier", type=float, default=None, help="attention softmax activation quantizer 的 trust multiplier")
    parser.add_argument("--variation-trust-move-v-multiplier", dest="variation_trust_move_v_multiplier", type=float, default=None, help="move_v shift 参数的 trust multiplier")
    parser.add_argument("--variation-trust-proj-move-multiplier", dest="variation_trust_proj_move_multiplier", type=float, default=None, help="attention proj move 参数的 trust multiplier")
    parser.add_argument("--variation-trust-start-update", dest="variation_trust_start_update", type=int, default=None, help="延迟到第几个 local optimizer update 捕获 variation trust anchor；0 表示训练开始前捕获")
    parser.add_argument("--aoq-explore-scale-ratio", dest="aoq_explore_scale_ratio", type=float, default=None, help="AOQ-inspired 探索阶段对 selected weight quantizer scale 的临时倍率；1 关闭，<1 缩窄 threshold/level 间隔")
    parser.add_argument("--aoq-explore-threshold-ratio", dest="aoq_explore_threshold_ratio", type=float, default=None, help="AOQ-inspired 探索阶段仅对 threshold interval 的临时倍率；0 表示跟随 scale ratio")
    parser.add_argument("--aoq-explore-layers", dest="aoq_explore_layers", type=str, default=None, help="AOQ explore 只作用于这些 named module；空表示全部 StatsQ/qk/v/LSQ weight quantizers")
    parser.add_argument("--aoq-explore-layer-ratios", dest="aoq_explore_layer_ratios", type=str, default=None, help="AOQ explore 按 named module 覆盖 scale ratio，格式 module:ratio,module:ratio；覆盖会在基础 ratio 之后应用")
    parser.add_argument("--aoq-explore-selective-margin", dest="aoq_explore_selective_margin", type=float, default=None, help="AOQ explore 只对距离 weight bin boundary 不超过该 margin 的元素应用 scale ratio；0 表示全量应用")
    parser.add_argument("--aoq-explore-quality-mode", dest="aoq_explore_quality_mode", choices=["none", "grad_cross", "anchor_unmoved", "anchor_moved", "history_oscillating", "recent_oscillating"], default=None, help="AOQ crossing-quality selector；grad_cross 用当前梯度方向筛 near-boundary crossing；anchor_unmoved/anchor_moved 按相对 anchor checkpoint 是否已跨 bin 筛选；history_oscillating 按累计 per-weight bin switch/oscillation 历史筛选；recent_oscillating 只按当前 update 新发生的方向反转筛选")
    parser.add_argument("--aoq-explore-quality-layers", dest="aoq_explore_quality_layers", type=str, default=None, help="AOQ crossing-quality selector 只作用于这些 named module；空表示沿用 aoq-explore-layers")
    parser.add_argument("--aoq-explore-quality-start-update", dest="aoq_explore_quality_start_update", type=int, default=None, help="第几个 local optimizer update 后才启用 AOQ crossing-quality selector；用于尾段状态驱动探索")
    parser.add_argument("--aoq-explore-quality-min-frac", dest="aoq_explore_quality_min_frac", type=float, default=None, help="grad_cross selector 每个 quantizer 至少保留的 near-boundary fraction，0 表示不强制保留")
    parser.add_argument("--aoq-explore-anchor-checkpoint", dest="aoq_explore_anchor_checkpoint", type=str, default=None, help="anchor_unmoved selector 的 anchor/source checkpoint；已相对该 checkpoint 跨 bin 的权重不再参与 AOQ explore")
    parser.add_argument("--aoq-explore-start-update", dest="aoq_explore_start_update", type=int, default=None, help="AOQ explore 从第几个 local optimizer update 开始")
    parser.add_argument("--aoq-explore-end-update", dest="aoq_explore_end_update", type=int, default=None, help="AOQ explore 到第几个 local optimizer update 前结束；0 表示不自动结束")
    parser.add_argument("--aoq-explore-repeat-each-epoch", dest="aoq_explore_repeat_each_epoch", action="store_true", help="每个 epoch 都按 local update 重新启用 AOQ explore window")
    parser.add_argument("--aoq-explore-update-schedule", dest="aoq_explore_update_schedule", type=str, default=None, help="按 local update 切换 AOQ ratio，格式 update:scale:threshold:margin,update:scale:threshold:margin；scale=1,threshold=0,margin=0 表示关闭")
    parser.add_argument("--delta-direction-anchor-weight", dest="delta_direction_anchor_weight", type=float, default=None, help="parameter delta direction cosine anchor weight; 0 disables")
    parser.add_argument("--delta-direction-anchor-base-checkpoint", dest="delta_direction_anchor_base_checkpoint", type=str, default=None, help="base checkpoint for delta direction anchor")
    parser.add_argument("--delta-direction-anchor-target-checkpoint", dest="delta_direction_anchor_target_checkpoint", type=str, default=None, help="target checkpoint defining desired parameter delta direction")
    parser.add_argument("--delta-direction-anchor-params", dest="delta_direction_anchor_params", type=str, default=None, help="comma-separated parameter name substrings for delta direction anchor")
    parser.add_argument("--delta-direction-anchor-start-update", dest="delta_direction_anchor_start_update", type=int, default=None, help="optimizer update to start applying delta direction anchor")
    parser.add_argument("--act-bin-margin-weight", dest="act_bin_margin_weight", type=float, default=None, help="activation bin-boundary margin regularizer weight; 0 disables")
    parser.add_argument("--act-bin-margin-layers", dest="act_bin_margin_layers", type=str, default=None, help="activation bin-margin 作用的 named module 列表；空表示全部 activation quantizers")
    parser.add_argument("--act-bin-margin-quantizers", dest="act_bin_margin_quantizers", type=str, default=None, help="精确或后缀匹配的 activation quantizer 名称列表")
    parser.add_argument("--act-bin-margin", dest="act_bin_margin", type=float, default=None, help="归一化量化 bin 边界安全距离，范围 [0, 0.5]")
    parser.add_argument("--act-bin-margin-max-elements", dest="act_bin_margin_max_elements", type=int, default=None, help="每个 quantizer 最多采样多少 activation 元素计算 margin")
    parser.add_argument("--pre-qat-act-percentile-calib-batches", dest="pre_qat_act_percentile_calib_batches", type=int, default=None, help="pre-QAT 阶段用多少 train batches 按 activation percentile 重校准 LSQ scale；0 关闭")
    parser.add_argument("--pre-qat-act-percentile-calib-layers", dest="pre_qat_act_percentile_calib_layers", type=str, default=None, help="pre-QAT activation percentile calibration 作用的 named module 列表；空表示全部 activation quantizers")
    parser.add_argument("--pre-qat-act-percentile-calib-percentile", dest="pre_qat_act_percentile_calib_percentile", type=float, default=None, help="pre-QAT activation scale percentile，例如 0.999")
    parser.add_argument("--pre-qat-act-percentile-calib-blend", dest="pre_qat_act_percentile_calib_blend", type=float, default=None, help="新 percentile scale 与旧 scale 的混合比例，1.0 表示完全替换")
    parser.add_argument("--pre-qat-act-mse-calib-batches", dest="pre_qat_act_mse_calib_batches", type=int, default=None, help="pre-QAT 阶段用多少 train batches 做 activation MSE scale search；0 关闭")
    parser.add_argument("--pre-qat-act-mse-calib-layers", dest="pre_qat_act_mse_calib_layers", type=str, default=None, help="pre-QAT activation MSE calibration 作用的 named module 列表")
    parser.add_argument("--pre-qat-act-mse-calib-quantizers", dest="pre_qat_act_mse_calib_quantizers", type=str, default=None, help="精确或后缀匹配的 activation quantizer 名称列表；空表示匹配 layers 下全部")
    parser.add_argument("--pre-qat-act-mse-calib-grid", dest="pre_qat_act_mse_calib_grid", type=str, default=None, help="activation MSE scale search ratio 网格，格式 min,max,steps")
    parser.add_argument("--pre-qat-act-mse-calib-blend", dest="pre_qat_act_mse_calib_blend", type=float, default=None, help="MSE-opt scale 与旧 scale 的混合比例，1.0 表示完全替换")
    parser.add_argument("--pre-qat-recon-updates", dest="pre_qat_recon_updates", type=int, default=None, help="正式 QAT epoch 前执行多少步 teacher-logit reconstruction，仅更新量化/shift 参数")
    parser.add_argument("--pre-qat-recon-temperature", dest="pre_qat_recon_temperature", type=float, default=None, help="pre-QAT teacher-logit reconstruction temperature")
    parser.add_argument("--pre-qat-feature-recon-updates", dest="pre_qat_feature_recon_updates", type=int, default=None, help="正式 QAT epoch 前执行多少步 teacher feature/block-output reconstruction，仅更新量化/shift 参数")
    parser.add_argument("--pre-qat-feature-recon-layers", dest="pre_qat_feature_recon_layers", type=str, default=None, help="pre-QAT feature reconstruction 的 named module 列表")
    parser.add_argument("--pre-qat-feature-recon-policy", dest="pre_qat_feature_recon_policy", choices=["quant", "module_all"], default=None, help="pre-QAT feature reconstruction 更新参数范围")
    parser.add_argument("--pre-qat-feature-recon-confidence-power", dest="pre_qat_feature_recon_confidence_power", type=float, default=None, help="pre-QAT feature reconstruction 按 teacher confidence 加权的幂次，0 关闭")
    parser.add_argument("--pre-qat-feature-recon-weight-mode", dest="pre_qat_feature_recon_weight_mode", choices=["none", "confidence", "disagreement"], default=None, help="pre-QAT feature reconstruction 样本加权方式")
    parser.add_argument("--pre-qat-feature-recon-qdrop-prob", dest="pre_qat_feature_recon_qdrop_prob", type=float, default=None, help="pre-QAT feature reconstruction 期间随机 bypass activation quantizer 的概率")
    parser.add_argument("--pre-qat-feature-recon-qdrop-layers", dest="pre_qat_feature_recon_qdrop_layers", type=str, default=None, help="pre-QAT QDrop 只作用于这些 named module 下的 activation quantizer；空表示全模型")
    parser.add_argument("--pre-qat-feature-recon-anchor-kl-weight", dest="pre_qat_feature_recon_anchor_kl_weight", type=float, default=None, help="pre-QAT feature reconstruction 期间固定起点 student logit KL 权重；0 关闭")
    parser.add_argument("--pre-qat-feature-recon-anchor-kl-temperature", dest="pre_qat_feature_recon_anchor_kl_temperature", type=float, default=None, help="pre-QAT feature reconstruction 起点 student logit KL temperature")
    parser.add_argument("--post-epoch-feature-recon-updates", dest="post_epoch_feature_recon_updates", type=int, default=None, help="每个正式 QAT epoch 后、保存和验证前额外执行多少步 teacher feature reconstruction；0 关闭")
    parser.add_argument("--pre-qat-seq-feature-recon-updates", dest="pre_qat_seq_feature_recon_updates", type=int, default=None, help="正式 QAT epoch 前逐层执行 feature reconstruction，每层更新步数")
    parser.add_argument("--pre-qat-seq-feature-recon-layers", dest="pre_qat_seq_feature_recon_layers", type=str, default=None, help="pre-QAT sequential feature reconstruction 的 named module 列表")
    parser.add_argument("--pre-qat-seq-feature-recon-policy", dest="pre_qat_seq_feature_recon_policy", choices=["quant", "module_all"], default=None, help="pre-QAT sequential feature reconstruction 更新参数范围")
    parser.add_argument("--setup-alpha-batches", dest="setup_alpha_batches", type=int, default=None, help="number of train batches used for quantizer alpha initialization")
    parser.add_argument("--post-resume-setup-alpha-batches", dest="post_resume_setup_alpha_batches", type=int, default=None, help="number of train batches used for post-resume quantizer calibration; 0 disables")
    parser.add_argument("--ref-attn-kl-weight-epoch-overrides", dest="ref_attn_kl_weight_epoch_overrides", type=str, default=None, help="按 epoch 覆盖 prev-step KL 权重，格式 epoch:value,epoch:value")
    parser.add_argument("--anchor-ref-attn-kl-weight-epoch-overrides", dest="anchor_ref_attn_kl_weight_epoch_overrides", type=str, default=None, help="按 epoch 覆盖 anchor KL 权重，格式 epoch:value,epoch:value")
    parser.add_argument("--ref-head-mode-epoch-overrides", dest="ref_head_mode_epoch_overrides", type=str, default=None, help="按 epoch 覆盖 prev-step KL head mode，格式 epoch=value;epoch=value")
    parser.add_argument("--dynamic-sparse-prevstep-kl", dest="dynamic_sparse_prevstep_kl", action="store_true", help="启用基于 full-val drop 和离线 head prior 的 dynamic sparse prev-step KL controller")
    parser.add_argument("--dynamic-kl-start-epoch", dest="dynamic_kl_start_epoch", type=int, default=None, help="dynamic sparse prev-step KL 开始允许触发的 epoch；默认 61")
    parser.add_argument("--dynamic-kl-observe-until-epoch", dest="dynamic_kl_observe_until_epoch", type=int, default=None, help="只观测不触发 KL 的最后 epoch；默认 60")
    parser.add_argument("--dynamic-kl-primary-heads", dest="dynamic_kl_primary_heads", type=str, default=None, help="dynamic KL primary head 列表，例如 8:4")
    parser.add_argument("--dynamic-kl-secondary-heads", dest="dynamic_kl_secondary_heads", type=str, default=None, help="dynamic KL secondary head 列表，例如 5:7,4:11,6:1,11:18")
    parser.add_argument("--dynamic-kl-avoid-heads", dest="dynamic_kl_avoid_heads", type=str, default=None, help="dynamic KL 禁止选择的 head 列表")
    parser.add_argument("--dynamic-kl-drop-threshold", dest="dynamic_kl_drop_threshold", type=float, default=None, help="rolling best 与当前 Top-1 的最小 drop 触发阈值")
    parser.add_argument("--dynamic-kl-strong-drop-threshold", dest="dynamic_kl_strong_drop_threshold", type=float, default=None, help="使用 strong KL 权重的 Top-1 drop 阈值")
    parser.add_argument("--dynamic-kl-default-weight", dest="dynamic_kl_default_weight", type=float, default=None, help="dynamic KL 默认 pulse 权重")
    parser.add_argument("--dynamic-kl-strong-weight", dest="dynamic_kl_strong_weight", type=float, default=None, help="dynamic KL strong pulse 权重")
    parser.add_argument("--dynamic-kl-max-weight", dest="dynamic_kl_max_weight", type=float, default=None, help="dynamic KL 最大允许权重")
    parser.add_argument("--dynamic-kl-cooldown-epochs", dest="dynamic_kl_cooldown_epochs", type=int, default=None, help="同一 head 触发后的 cooldown epoch 数")
    parser.add_argument("--dynamic-kl-window-epochs", dest="dynamic_kl_window_epochs", type=int, default=None, help="pulse 频率限制窗口长度")
    parser.add_argument("--dynamic-kl-max-pulses-per-window", dest="dynamic_kl_max_pulses_per_window", type=int, default=None, help="每个窗口最多允许的 pulse 次数")
    parser.add_argument("--dynamic-kl-controller-tsv", dest="dynamic_kl_controller_tsv", type=str, default=None, help="dynamic KL controller TSV 输出路径")
    parser.add_argument("--dynamic-kl-prior-source", dest="dynamic_kl_prior_source", type=str, default=None, help="controller 使用的 head prior 来源说明")
    parser.add_argument("--epoch-lr-overrides", dest="epoch_lr_overrides", type=str, default=None, help="按 epoch 固定 LR，格式 epoch:value,epoch:value")
    parser.add_argument("--progressive-bit-schedule", dest="progressive_bit_schedule", type=str, default=None, help="按 epoch 切换 fake-quant bit-width，格式 epoch:wbits:abits,epoch:wbits:abits")
    parser.add_argument("--progressive-bit-rescale-lsq", dest="progressive_bit_rescale_lsq", action="store_true", help="切换 LSQ bit-width 时按 sqrt(old_thd_pos/new_thd_pos) 重缩放 learned scale")
    parser.add_argument("--progressive-bit-recalibrate-epochs", dest="progressive_bit_recalibrate_epochs", type=str, default=None, help="bit 切换后重校准 LSQ alpha 的 epoch 列表，例如 2,3")
    parser.add_argument("--progressive-bit-recalibrate-batches", dest="progressive_bit_recalibrate_batches", type=int, default=None, help="每次 bit 切换重校准使用的 train batches")
    parser.add_argument("--progressive-bit-transition-recon-updates", dest="progressive_bit_transition_recon_updates", type=int, default=None, help="降 bit 切换后、正式训练前执行多少步局部 teacher feature reconstruction；0 关闭")
    parser.add_argument("--progressive-bit-transition-recon-epochs", dest="progressive_bit_transition_recon_epochs", type=str, default=None, help="只在这些 epoch 执行 transition reconstruction；空表示所有降 bit 切换点")
    parser.add_argument("--progressive-bit-transition-recon-layers", dest="progressive_bit_transition_recon_layers", type=str, default=None, help="transition reconstruction 的 named module 列表")
    parser.add_argument("--progressive-bit-transition-recon-policy", dest="progressive_bit_transition_recon_policy", choices=["quant", "module_all"], default=None, help="transition reconstruction 更新参数范围")
    parser.add_argument("--progressive-bit-transition-recon-confidence-power", dest="progressive_bit_transition_recon_confidence_power", type=float, default=None, help="transition reconstruction 按 teacher confidence 加权的幂次")
    parser.add_argument("--progressive-bit-transition-recon-weight-mode", dest="progressive_bit_transition_recon_weight_mode", choices=["none", "confidence", "disagreement"], default=None, help="transition reconstruction 样本加权方式")
    parser.add_argument("--progressive-bit-transition-recon-qdrop-prob", dest="progressive_bit_transition_recon_qdrop_prob", type=float, default=None, help="transition reconstruction 期间随机 bypass activation quantizer 的概率")
    parser.add_argument("--progressive-bit-transition-recon-qdrop-layers", dest="progressive_bit_transition_recon_qdrop_layers", type=str, default=None, help="transition reconstruction QDrop 作用的 named module 列表")
    parser.add_argument("--progressive-bit-transition-anchor-kl-weight", dest="progressive_bit_transition_anchor_kl_weight", type=float, default=None, help="降 bit 前 student anchor logit KL 权重；0 关闭")
    parser.add_argument("--progressive-bit-transition-anchor-kl-temperature", dest="progressive_bit_transition_anchor_kl_temperature", type=float, default=None, help="降 bit 前 student anchor logit KL temperature")
    parser.add_argument("--quant-only-start-epoch", dest="quant_only_start_epoch", type=int, default=None, help="从该 epoch 起只训练量化和 shift 参数")
    parser.add_argument("--trainable-policy", dest="trainable_policy", choices=["all", "non_quant", "freeze_act_quant", "freeze_act_except_layers", "quant", "quant_in_layers", "params_in_layers", "params_in_layers_attn_plus_quant", "params_in_layers_freeze_highdrift_act", "params_in_layers_freeze_move_v_shift", "head_norm_quant", "head_norm_proj_quant", "head_norm_attn_quant", "attn_quant"], default=None, help="quant-only 阶段的可训练参数集合")
    parser.add_argument("--trainable-policy-freeze-act-except-layers", dest="trainable_policy_freeze_act_except_layers", type=str, default=None, help="freeze_act_except_layers 策略下允许 activation quant/shift 继续训练的 named module 列表")
    parser.add_argument("--trainable-policy-update-overrides", dest="trainable_policy_update_overrides", type=str, default=None, help="按 optimizer update 切换可训练参数集合，格式 update:policy,update:policy")
    parser.add_argument("--trainable-policy-update-mode", dest="trainable_policy_update_mode", choices=["requires_grad", "grad_mask", "grad_damp"], default=None, help="update 级 policy 的执行方式")
    parser.add_argument("--trainable-policy-grad-damp", dest="trainable_policy_grad_damp", type=float, default=None, help="grad_damp 模式下被当前 policy 排除的参数梯度倍率")
    parser.add_argument("--model-ema", dest="model_ema", action="store_true", help="训练时维护 student 权重 EMA 并保存 .ema checkpoint")
    parser.add_argument("--model-ema-decay", dest="model_ema_decay", type=float, default=None, help="student 权重 EMA decay")

    parser.add_argument("--quantize-downsample", dest="quantize_downsample", type=str2bool, default=True, help="AOQ 是否量化 downsample")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision for supported pipelines; OFQ uses CUDA autocast")
    parser.add_argument("--amp-dtype", dest="amp_dtype", choices=["bf16", "fp16"], default="bf16", help="Mixed precision dtype for supported pipelines")
    parser.add_argument("--channels-last", dest="channels_last", action="store_true", help="Use channels_last memory format where supported")
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
