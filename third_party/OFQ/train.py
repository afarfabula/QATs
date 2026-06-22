#!/usr/bin/env python3
# This is a slightly modified version of timm's training script
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import copy
import bisect
import io
import inspect
import types
import sys
import time
import yaml
import os
import shutil
import logging
import random as pyrandom
from collections import OrderedDict
from contextlib import nullcontext, suppress
from datetime import datetime, timedelta

import torch
import torch.serialization
import torch.nn as nn
import torchvision.utils
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.multiprocessing as mp
from torch.utils.data import Dataset
import pyarrow.parquet as pq

import numpy as np

from timm.data import (
    create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
)
from timm.models import (
    create_model, safe_model_name, resume_checkpoint, load_checkpoint,
    convert_splitbn_model, model_parameters
)
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.scheduler.scheduler import Scheduler
from timm.utils import ApexScaler, NativeScaler

from src import *
import math

if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([argparse.Namespace, types.SimpleNamespace])

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


def format_eta_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class ImageNetParquetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data_dir = os.path.join(root, 'data') if os.path.isdir(os.path.join(root, 'data')) else root
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f'parquet data dir not found: {self.data_dir}')

        self.files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.startswith(f'{split}-') and f.endswith('.parquet')
        ])
        if not self.files:
            raise FileNotFoundError(f'no parquet files for split={split} under {self.data_dir}')

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
        self._current_cache_key = None
        self._current_cache = None

    def __len__(self):
        return self._total_rows

    def set_transform(self, transform):
        self.transform = transform

    def _locate_row_group(self, index):
        pos = bisect.bisect_right(self._row_groups, (index, float('inf'), float('inf'), float('inf'))) - 1
        if pos < 0:
            raise IndexError(index)
        start, file_idx, rg_idx, rg_rows = self._row_groups[pos]
        if index >= start + rg_rows:
            raise IndexError(index)
        return start, file_idx, rg_idx

    def _load_row_group(self, file_idx, rg_idx):
        key = (file_idx, rg_idx)
        if self._current_cache_key != key:
            table = pq.ParquetFile(self.files[file_idx]).read_row_group(rg_idx, columns=['image', 'label'])
            self._current_cache = table.to_pylist()
            self._current_cache_key = key
        return self._current_cache

    def __getitem__(self, index):
        start, file_idx, rg_idx = self._locate_row_group(index)
        rows = self._load_row_group(file_idx, rg_idx)
        sample = rows[index - start]
        image_bytes = sample['image']['bytes']
        target = int(sample['label'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class ImageNetParquetIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            root,
            split='train',
            transform=None,
            shuffle=True,
            seed=42,
            batch_size=None,
            drop_last=False,
            distributed_world_size=None,
            distributed_rank=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.distributed_world_size = distributed_world_size
        self.distributed_rank = distributed_rank
        self.data_dir = os.path.join(root, 'data') if os.path.isdir(os.path.join(root, 'data')) else root
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f'parquet data dir not found: {self.data_dir}')

        self.files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.startswith(f'{split}-') and f.endswith('.parquet')
        ])
        if not self.files:
            raise FileNotFoundError(f'no parquet files for split={split} under {self.data_dir}')

        self._row_groups = []
        total_rows = 0
        for path in self.files:
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                self._row_groups.append((path, rg_idx, rg_rows))
                total_rows += rg_rows
        self._total_rows = total_rows

    def __len__(self):
        world_size = self._world_size()
        if not self.batch_size:
            return self._total_rows

        if self.drop_last:
            global_batch = self.batch_size * world_size
            if global_batch <= 0:
                return self._total_rows
            full_batches = self._total_rows // global_batch
            return full_batches * self.batch_size

        return int(math.ceil(self._total_rows / float(world_size)))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _world_size(self):
        if self.distributed_world_size is not None:
            return int(self.distributed_world_size)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    def _rank(self):
        if self.distributed_rank is not None:
            return int(self.distributed_rank)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _ordered_row_group_indices(self):
        indices = list(range(len(self._row_groups)))
        if self.shuffle:
            rng = pyrandom.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        return indices

    def _effective_total_rows(self, world_size):
        if not self.batch_size:
            return self._total_rows
        if self.drop_last:
            global_batch = self.batch_size * world_size
            if global_batch <= 0:
                return self._total_rows
            return (self._total_rows // global_batch) * global_batch
        return self._total_rows

    def _partition_row_groups(self, ordered_indices, target_rows):
        if target_rows <= 0:
            return [[]]

        partitions = []
        current = []
        remaining = int(target_rows)
        for rg_global_idx in ordered_indices:
            if remaining <= 0:
                partitions.append(current)
                current = []
                remaining = int(target_rows)

            path, rg_idx, rg_rows = self._row_groups[rg_global_idx]
            start = 0
            while rg_rows > 0:
                if remaining <= 0:
                    partitions.append(current)
                    current = []
                    remaining = int(target_rows)

                take = min(rg_rows, remaining)
                current.append((rg_global_idx, path, rg_idx, start, start + take))
                rg_rows -= take
                start += take
                remaining -= take

        if current:
            partitions.append(current)
        return partitions

    def _rank_segments(self, world_size, rank):
        ordered_indices = self._ordered_row_group_indices()
        effective_total_rows = self._effective_total_rows(world_size)
        if effective_total_rows <= 0:
            return []

        if self.batch_size and self.drop_last:
            target_rows = effective_total_rows // world_size
        else:
            target_rows = int(math.ceil(effective_total_rows / float(world_size)))

        partitions = self._partition_row_groups(ordered_indices, target_rows)
        if rank >= len(partitions):
            return []
        return partitions[rank]

    @staticmethod
    def _partition_segments_for_worker(segments, num_workers, worker_id):
        if num_workers <= 1 or not segments:
            return segments

        total_rows = sum(end - start for _, _, _, start, end in segments)
        if total_rows <= 0:
            return []

        worker_target = int(math.ceil(total_rows / float(num_workers)))
        worker_partitions = []
        current = []
        remaining = worker_target

        for rg_global_idx, path, rg_idx, start, end in segments:
            seg_start = start
            seg_rows = end - start
            while seg_rows > 0:
                if remaining <= 0:
                    worker_partitions.append(current)
                    current = []
                    remaining = worker_target

                take = min(seg_rows, remaining)
                current.append((rg_global_idx, path, rg_idx, seg_start, seg_start + take))
                seg_rows -= take
                seg_start += take
                remaining -= take

        if current:
            worker_partitions.append(current)

        if worker_id >= len(worker_partitions):
            return []
        return worker_partitions[worker_id]

    def _iter_segment_rows(self, segment, rng):
        rg_global_idx, path, rg_idx, start, end = segment
        table = pq.ParquetFile(path).read_row_group(rg_idx, columns=['image', 'label'])
        cols = table.to_pydict()
        images = cols['image']
        labels = cols['label']
        order = list(range(start, end))
        if self.shuffle:
            rng.shuffle(order)

        for idx in order:
            image_bytes = images[idx]['bytes']
            target = int(labels[idx])
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            yield image, target

    def __iter__(self):
        world_size = self._world_size()
        rank = self._rank()
        rank_segments = self._rank_segments(world_size, rank)

        info = torch.utils.data.get_worker_info()
        if info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = info.num_workers
            worker_id = info.id

        worker_segments = self._partition_segments_for_worker(rank_segments, num_workers, worker_id)
        rng = pyrandom.Random(self.seed + self.epoch * 1009 + rank * 53 + worker_id)

        for segment in worker_segments:
            yield from self._iter_segment_rows(segment, rng)


def create_dataset_compat(
        dataset_name,
        root,
        split,
        is_training,
        batch_size,
        repeats=0,
        transform=None,
        distributed=False,
        distributed_world_size=None,
        distributed_rank=None):
    if dataset_name == 'hf-parquet-imagenet':
        # NOTE:
        # For training we keep the high-throughput iterable parquet path, but we
        # partition the globally shuffled row-group stream into equal full-batch
        # sized chunks per rank. This preserves sequential row-group reads while
        # guaranteeing every rank executes exactly the same number of training
        # steps, avoiding the train/validate collective mismatch seen before.
        if is_training:
            return ImageNetParquetIterableDataset(
                root=root,
                split=split,
                transform=transform,
                shuffle=True,
                batch_size=batch_size,
                drop_last=True,
                distributed_world_size=distributed_world_size,
                distributed_rank=distributed_rank,
            )
        return ImageNetParquetDataset(root=root, split=split, transform=transform)
    return create_dataset(dataset_name, root=root, split=split, is_training=is_training,
                          batch_size=batch_size, repeats=repeats)


def create_loader_compat(dataset, **kwargs):
    sig = inspect.signature(create_loader)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return create_loader(dataset, **filtered)

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--num_aug_repeats', type= int, default=0) ## if 0 then no Repeated Aug, 3 is for deit-t

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('--epoch-checkpoint-interval', type=int, default=10, metavar='N',
                    help='save one epoch checkpoint every N epochs (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                    help='number of gradient accumulation steps (default: 1)')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
parser.add_argument('--wq-enable', action='store_true', default=False,
                    help='enable weight quantization')
parser.add_argument('--wq-mode', default='LSQ', type=str,
                    help='weight quantization mode')
parser.add_argument('--wq-bitw', default=2, type=int,
                    help='weight quantization bit width')
parser.add_argument('--wq-per-channel', default=False, action='store_true',
                    help='per channel weight quantization')
parser.add_argument('--wq-asym', action='store_true', default=False,
                    help='asymmetric quantization for weight')
parser.add_argument('--wq_clip_learnable', action='store_true', default=False,
                    help='learbable step size / clip value for weight quant')
parser.add_argument('--aq-enable', action='store_true', default=False,
                    help='enable act quantization')
parser.add_argument('--aq-mode', default='lsq', type=str,
                    help='act quantization mode')
parser.add_argument('--aq-bitw', default=2, type=int,
                    help='act quantization bit width')
parser.add_argument('--aq-per-channel', default=False, action='store_true',
                    help='per channel activation quantization')
parser.add_argument('--aq_clip_learnable', action='store_true', default=False,
                    help='learbable step size / clip value for activation quant')
parser.add_argument('--qmodules', type=str, nargs='+', default=None, metavar='N',
                    help='Quantized modules')
parser.add_argument('--replace-ln-by-bn', action='store_true', default=False,
                    help='whether to use bn instead of layernorm')
parser.add_argument('--use-kd', action='store_true', default=False,
                    help='whether to use kd')
parser.add_argument('--use-token-kd', action='store_true', default=False,
                    help='whether to use token level kd')
parser.add_argument('--kd-alpha', default=1.0, type=float,
                    help='KD alpha, soft loss portion (default: 1.0)')
parser.add_argument('--teacher', default='resnet101', type=str, metavar='MODEL',
                    help='Name of teacher model (default: "countception"')
parser.add_argument('--teacher-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize teacher model from this checkpoint (default: none)')
parser.add_argument('--teacher_pretrained', default=False, action='store_true')
parser.add_argument('--quant-teacher', action='store_true', default=False,
                    help='whether to quantize the teacher')
parser.add_argument('--use-distill-head', action='store_true', default=False,
                    help='whether to use two separate heads for distillation')
parser.add_argument('--use-layer-scale', action='store_true', default=False,
                    help='whether to use scaler for attention and mlp')
parser.add_argument('--use-skip', action='store_true', default=False,
                    help='whether to use skip for connection')
parser.add_argument('--use-relu', action='store_true', default=False,
                    help='whether to use relu for the model')
parser.add_argument('--kd-type', type=str, default="last",
                    help='whether to match the last or all the embeddings')

parser.add_argument('--gpu_id', default=0, type=int,
                    help='gpu id to use for single gpu training')
parser.add_argument('--model_type', type=str, default="deit", help='model type to quantize: deit or swin')
parser.add_argument('--quantized', action='store_true', default=False,
                    help='whether to quantize model')

parser.add_argument('--world_size', type=str, default='1', help='number of gpu to use for training')
parser.add_argument('--visible_gpu', type=str, default='0', help='indicate which gpus to use for distributed training')
parser.add_argument('--tcp_port', type=str, default='37879')
parser.add_argument('--collect_attention', action='store_true', default=False,
                    help='return attention matrices for offline probing')
parser.add_argument('--max_train_updates', type=int, default=0,
                    help='stop after this many optimizer updates in current run, 0 means no limit')
parser.add_argument('--save_step_checkpoints', action='store_true', default=False,
                    help='save step-level checkpoints during training updates')
parser.add_argument('--save_initial_step_checkpoint', action='store_true', default=False,
                    help='save step checkpoint before any new optimizer update')
parser.add_argument('--step_checkpoint_interval', type=int, default=1,
                    help='save one step checkpoint every N optimizer updates')
parser.add_argument('--step_checkpoint_warmup_updates', type=int, default=0,
                    help='run this many optimizer updates before starting step checkpoint collection')
parser.add_argument('--max_step_checkpoints_to_save', type=int, default=0,
                    help='maximum number of step checkpoints to save after warmup, 0 means no limit')
parser.add_argument('--skip_validate', action='store_true', default=False,
                    help='skip validation pass for quick replay runs')
parser.add_argument('--eval-only', action='store_true', default=False,
                    help='load checkpoint, run validation once, and exit without further training')

parser.add_argument('--apply_q_attn_dropout', type= int, default= 0
                    ,help="0: quantized attn and apply dropout, 1 don't q attn and apply dropout, 2 don't q attn and don't dropout, 3 quantized attn but no dropout")
parser.add_argument('--act_layer', type= str, default= 'gelu', help='activation function to use for the fc1 output of MLP') # ACT_LAYER_MAPPINGS, ['relu', 'gelu', 'prelu', 'rprelu']
parser.add_argument('--kd_hard_and_soft', type = int, default=0, help="0: only use soft label, 1 hard label + soft label") 
parser.add_argument('--teacher_type', type=str, default='deit',help='teacher type: deit or swin')
parser.add_argument('--pretrained_initialized', action='store_true', default=False, help="false = does not use pretrained weight to initialize") 

## apply qk_reparam
parser.add_argument( '--qk_reparam', action='store_true', default= False, help="True: use QKR, False: doesn't use QKR")
parser.add_argument( '--qk_reparam_type', type=int, default=0, help="0 for normal training, 1 if for cga finetuning")


def parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    if args.qmodules is None:
        args.qmodules = []

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def get_qat_model(model, args):
    qat_model = copy.deepcopy(model)
    qat_model.train()
    qconfigs = {}

    ACT_LAYER_MAPPINGS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'prelu' : nn.PReLU,
    'rprelu' : 'rprelu',
    'None': 'None'
    }

    for m in args.qmodules:
        wcfg = {
            "mode": args.wq_mode if args.wq_enable else "Identity",
            "bit": args.wq_bitw if args.wq_bitw < 32 and args.aq_enable else "identity",
            "all_positive": False,
            "symmetric": not args.wq_asym,
            "per_channel": args.wq_per_channel,
            "normalize_first": False,
            "learnable": args.wq_clip_learnable
        }
        acfg = {
            "enable": args.aq_enable if args.aq_enable else "Identity",
            "mode": args.aq_mode if args.aq_bitw < 32 and args.aq_enable else "identity",
            "bit": args.aq_bitw,
            "per_channel": args.aq_per_channel,
            "normalize_first": False,
            "learnable": args.aq_clip_learnable
        }
        qconfigs[m] = {"weight": wcfg, "act": acfg, "q_attn_dropout": args.apply_q_attn_dropout, "act_layer": ACT_LAYER_MAPPINGS[args.act_layer] }

    if args.model_type == 'deit':
        qat_model = replace_module_by_qmodule_deit(model, qconfigs, pretrained_initialized = args.pretrained_initialized,  \
            qk_reparam=args.qk_reparam, qk_reparam_type = args.qk_reparam_type)
    elif args.model_type == 'swin':
        qat_model = replace_module_by_qmodule_swin(model, qconfigs, pretrained_initialized = args.pretrained_initialized,  \
        qk_reparam=args.qk_reparam, qk_reparam_type = args.qk_reparam_type)
    
    return qat_model


def enable_attention_collection(model):
    enabled = 0
    for module in model.modules():
        module_name = type(module).__name__
        if module_name == 'ShiftedWindowAttention' or module_name.startswith('QAttention_swin'):
            setattr(module, 'collect_attention', True)
            enabled += 1
    return enabled


def _write_sync_marker(path):
    with open(path, 'w') as sync_f:
        sync_f.write('1')


def _wait_for_sync_markers(paths, timeout_seconds, poll_interval=1.0):
    deadline = time.monotonic() + timeout_seconds
    pending = set(paths)
    while pending:
        ready = {path for path in pending if os.path.exists(path)}
        pending.difference_update(ready)
        if not pending:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f'Timed out waiting for sync markers: {sorted(pending)}')
        time.sleep(poll_interval)


def _cleanup_sync_markers(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _save_epoch_checkpoint_no_metric(saver, epoch):
    if saver is None:
        raise RuntimeError('Checkpoint save requested on rank 0 but saver is not initialized.')

    tmp_save_path = os.path.join(saver.checkpoint_dir, 'tmp' + saver.extension)
    last_save_path = os.path.join(saver.checkpoint_dir, 'last' + saver.extension)
    checkpoint_name = '-'.join([saver.save_prefix, str(epoch)]) + saver.extension
    checkpoint_path = os.path.join(saver.checkpoint_dir, checkpoint_name)

    saver._save(tmp_save_path, epoch, metric=None)
    if os.path.exists(last_save_path):
        os.unlink(last_save_path)
    os.replace(tmp_save_path, last_save_path)

    if os.path.exists(checkpoint_path):
        os.unlink(checkpoint_path)
    shutil.copy2(last_save_path, checkpoint_path)

    saver.checkpoint_files = [
        (path, metric) for path, metric in saver.checkpoint_files
        if os.path.exists(path) and path != checkpoint_path
    ]
    saver.checkpoint_files.append((checkpoint_path, None))
    _logger.info(f'Saved checkpoint artifacts without eval metric: {checkpoint_path}')
    return None, None


def save_step_checkpoint(model, optimizer, args, output_dir, step_tag, epoch=None, batch_idx=None, loss_scaler=None, metric=None):
    step_dir = os.path.join(output_dir, 'step_checkpoints')
    os.makedirs(step_dir, exist_ok=True)
    save_state = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'arch': args.model,
        'state_dict': get_state_dict(model),
        'optimizer': optimizer.state_dict(),
        'version': 2,
        'args': args,
        'step_tag': step_tag,
    }
    if loss_scaler is not None:
        save_state[loss_scaler.state_dict_key] = loss_scaler.state_dict()
    if metric is not None:
        save_state['metric'] = metric
    save_path = os.path.join(step_dir, f'{step_tag}.pth.tar')
    torch.save(save_state, save_path)
    return save_path

def create_teacher_model(args):
    if args.teacher_type == "deit":
        teacher = create_model(args.teacher,num_classes=args.num_classes,
        drop_rate=args.drop, pretrained=args.teacher_pretrained, qqkkvv = args.kd_hard_and_soft == 2 or args.kd_hard_and_soft == 3
        )
    elif args.teacher_type == "swin":
        teacher = create_model(args.teacher, num_classes=args.num_classes, drop_path=args.drop_path, pretrained=args.teacher_pretrained, qqkkvv = args.kd_hard_and_soft == 2 or args.kd_hard_and_soft == 3)
        
    if args.quant_teacher:
        args.wq_bitw = 4
        args.aq_bitw = 4
        teacher = get_qat_model(teacher, args)
    if args.teacher_pretrained and args.teacher_checkpoint != "":
        load_checkpoint(teacher, args.teacher_checkpoint, strict=True)
    return teacher

def main(local_rank, args):
    setup_default_logging()
    args, args_text = args

    args.use_kd = args.use_kd or args.use_token_kd


    args.prefetcher = not args.no_prefetcher
    args.local_rank = local_rank
    if args.log_wandb:
        if has_wandb:
            if args.local_rank == 0:
                print("init wandb")
                wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    print("args.local_rank: "+str(args.local_rank))
    print(int(os.environ['WORLD_SIZE']))
    args.device = 'cuda:'+str(args.gpu_id)
    args.world_size = 1
    args.rank = 0  # global rank
    
    
    if args.distributed:
        
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:'+args.tcp_port,rank=args.local_rank, world_size=int(os.environ['WORLD_SIZE']))
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)
    if args.model_type == "deit":
        model = create_model(args.model,num_classes=args.num_classes, 
        drop_rate=args.drop, pretrained=args.pretrained,  qqkkvv = args.kd_hard_and_soft == 2 or args.kd_hard_and_soft == 3
        )
    elif args.model_type == "swin":
        model = create_model(args.model, drop_path=args.drop_path, num_classes=args.num_classes, pretrained=args.pretrained,
                             qqkkvv = args.kd_hard_and_soft == 2 or args.kd_hard_and_soft == 3 )
    else:
        print("check other branch")

    if args.quantized:
        model = get_qat_model(model, args)

    if args.collect_attention:
        enabled_attention_modules = enable_attention_collection(model)
        if args.local_rank == 0:
            _logger.info(f'Enabled attention collection for {enabled_attention_modules} modules.')
    
    if args.initial_checkpoint != "":
        load_checkpoint(model, args.initial_checkpoint, strict=False)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly
    if args.replace_ln_by_bn:
        model = replace_ln_by_bn1d(model)
    

    # distillation
    teacher = None
    if args.use_kd:
        if args.local_rank == 0:
            print("create teacher model")
        teacher = create_teacher_model(args)
        teacher.cuda()

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != 'native':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    ## create datasets
    dataset_train = create_dataset_compat(
        args.dataset,
        root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats,
        distributed=args.distributed,
        distributed_world_size=args.world_size,
        distributed_rank=args.rank)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader_compat(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        num_aug_repeats = args.num_aug_repeats,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    if args.local_rank == 0:
        _logger.info(f"Trainer transform: {loader_train.dataset.transform}")
        
    dataset_eval = create_dataset_compat(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        batch_size=args.batch_size,
        distributed=args.distributed,
        distributed_world_size=args.world_size,
        distributed_rank=args.rank)

    loader_eval = create_loader_compat(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )        
    if args.local_rank == 0:
        print(dataset_train.__len__())
        print(dataset_eval.__len__())

    

    ## have to create and init all the parameters before send to optimizer (important!)
    setup_alpha(model=model,loader=loader_train,args=args)
    # when creating optimizer, only weights and emb have weight-decay
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')
    
    # optionally resume from a checkpoint        
    resume_epoch = None
    output_dir_resume = None
    if args.experiment:
        try:
            output_dir_resume = os.path.join(args.output, args.experiment)
        except:
            print('please indicate args.resume if you would like to resume the checkpoint')
    
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)
    
    elif args.experiment:
        if os.path.exists(os.path.join(output_dir_resume,'last.pth.tar')):
            print("resume from last.pth.tar")
            resume_path = os.path.join(output_dir_resume,'last.pth.tar')
            resume_epoch = resume_checkpoint(
                model, resume_path,
                optimizer=None if args.no_resume_opt else optimizer,
                loss_scaler=None if args.no_resume_opt else loss_scaler,
                log_info=args.local_rank == 0)
    
    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    if (
        resume_epoch is not None
        and args.start_epoch is None
        and getattr(args, 'save_step_checkpoints', False)
        and int(getattr(args, 'max_train_updates', 0)) > 0
        and int(getattr(args, 'epochs', 0)) <= int(resume_epoch)
    ):
        adjusted_epochs = int(resume_epoch) + 1
        if args.local_rank == 0:
            _logger.info(
                'Adjusted args.epochs from %s to %s so resumed replay training can advance past epoch %s.',
                args.epochs,
                adjusted_epochs,
                resume_epoch,
            )
        args.epochs = adjusted_epochs

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))
        print(args.dataset)




    # setup loss function
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif args.use_token_kd:
        train_loss_fn = KLTokenMSELoss(alpha=args.kd_alpha, kd_type=args.kd_type)
    elif args.use_kd:
        if args.kd_hard_and_soft == 0:
            train_loss_fn = KLLossSoft()
        elif args.kd_hard_and_soft == 1: ## this loss function will automatically apply soft cross entropy if the label probability is given
            train_loss_fn = KDLossSoftandHard().cuda() # work with mixup as well
        elif args.kd_hard_and_soft == 2:
            train_loss_fn = KDLossSoftandHard_qk().cuda()
        elif args.kd_hard_and_soft == 3:
            train_loss_fn = KDLossSoftandHard_qkv().cuda()
            
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    
    if args.local_rank == 0:
        _logger.info(f"Train loss {train_loss_fn}")


    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    

    if args.resume:
        if not args.skip_validate or args.eval_only:
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,model_type=args.model_type)
            if args.eval_only:
                if args.local_rank == 0:
                    _logger.info('Eval-only metrics: {}'.format(eval_metrics))
                return
    elif args.experiment:
        if os.path.exists(os.path.join(output_dir_resume,'last.pth.tar')):
            if not args.skip_validate or args.eval_only:
                eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,model_type=args.model_type)
                if args.eval_only:
                    if args.local_rank == 0:
                        _logger.info('Eval-only metrics: {}'.format(eval_metrics))
                    return

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        
        
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

        if args.save_step_checkpoints and args.save_initial_step_checkpoint and int(getattr(args, 'step_checkpoint_warmup_updates', 0)) == 0:
            initial_step_path = save_step_checkpoint(
                model=model,
                optimizer=optimizer,
                args=args,
                output_dir=output_dir,
                step_tag='step_0000',
                epoch=start_epoch,
                batch_idx=-1,
                loss_scaler=loss_scaler,
            )
            _logger.info(f'Saved initial step checkpoint to {initial_step_path}')

    try:
            
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics, local_update_count, stopped_early = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema,
                mixup_fn=mixup_fn, teacher=teacher,teacher_type=args.teacher_type, log_wandb=args.log_wandb and has_wandb )
            if args.local_rank == 0:
                print("epoch: ", epoch, "g['lr']: ",optimizer.param_groups[0]['lr'])

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = {'loss': None, eval_metric: None}
            if not args.skip_validate:
                eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,epoch=epoch, log_wandb=args.log_wandb and has_wandb)
                
            if not args.skip_validate and model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                if args.skip_validate:
                    lr_scheduler.step(epoch + 1)
                else:
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None and not args.skip_validate:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            # save proper checkpoint with eval metric when available; otherwise still
            # persist a last/checkpoint-N artifact for later eval-only validation.
            save_metric = None if args.skip_validate else eval_metrics[eval_metric]
            should_save_checkpoint = (
                ((epoch + 1) % max(1, int(getattr(args, 'epoch_checkpoint_interval', 10))) == 0)
                or ((epoch + 1) == num_epochs)
                or stopped_early
            )
            if should_save_checkpoint:
                save_sync_path = None
                sync_output_dir = output_dir if output_dir is not None else output_dir_resume
                dist_ready = (
                    args.distributed
                    and torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                )
                sync_timeout_seconds = 1800
                ready_markers = []
                ack_markers = []
                cleanup_marker_path = None
                rank_ready_path = None
                rank_ack_path = None
                distributed_world_size = int(args.world_size)
                if dist_ready and sync_output_dir is not None:
                    save_sync_path = os.path.join(sync_output_dir, f'.checkpoint_sync_epoch_{epoch}.done')
                    cleanup_marker_path = os.path.join(sync_output_dir, f'.checkpoint_sync_epoch_{epoch}.cleanup')
                    ready_markers = [
                        os.path.join(sync_output_dir, f'.checkpoint_sync_epoch_{epoch}.rank{rank}.ready')
                        for rank in range(distributed_world_size)
                    ]
                    ack_markers = [
                        os.path.join(sync_output_dir, f'.checkpoint_sync_epoch_{epoch}.rank{rank}.ack')
                        for rank in range(distributed_world_size)
                    ]
                    rank_ready_path = ready_markers[args.rank]
                    rank_ack_path = ack_markers[args.rank]
                    if args.rank == 0:
                        _cleanup_sync_markers([cleanup_marker_path, save_sync_path, *ready_markers, *ack_markers])
                        _write_sync_marker(cleanup_marker_path)
                    else:
                        _wait_for_sync_markers([cleanup_marker_path], sync_timeout_seconds)
                    _write_sync_marker(rank_ready_path)
                    if args.rank == 0:
                        _wait_for_sync_markers(ready_markers, sync_timeout_seconds)

                if args.rank == 0:
                    if save_metric is None:
                        best_metric, best_epoch = _save_epoch_checkpoint_no_metric(saver, epoch)
                    else:
                        if saver is None:
                            raise RuntimeError('Checkpoint save requested on rank 0 but saver is not initialized.')
                        best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                    if save_sync_path is not None:
                        _write_sync_marker(save_sync_path)
                        _logger.info(f'Checkpoint sync marker written for epoch {epoch}: {save_sync_path}')
                elif save_sync_path is not None:
                    _wait_for_sync_markers([save_sync_path], sync_timeout_seconds)

                if dist_ready and rank_ack_path is not None:
                    _write_sync_marker(rank_ack_path)
                    if args.rank == 0:
                        _wait_for_sync_markers(ack_markers, sync_timeout_seconds)
                        _cleanup_sync_markers([cleanup_marker_path, save_sync_path, *ready_markers, *ack_markers])

            if stopped_early:
                if args.local_rank == 0:
                    _logger.info(f'Stopped early after {local_update_count} optimizer updates in epoch {epoch}.')
                break

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    if args.log_wandb and has_wandb:
        wandb.finish()

    if args.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass

def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, teacher=None, teacher_type = 'deit', log_wandb = False, annealing_schedule_for_dampen = None
        , annealing_schedule_for_iterative_freezing = None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    accum_steps = max(1, int(getattr(args, 'grad_accum_steps', 1)))

    model.train()
    optimizer.zero_grad()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    local_update_count = 0
    saved_step_count = 0
    stopped_early = False
    warmup_updates = max(0, int(getattr(args, 'step_checkpoint_warmup_updates', 0)))
    max_step_checkpoints_to_save = max(0, int(getattr(args, 'max_step_checkpoints_to_save', 0)))
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        update_step = ((batch_idx + 1) % accum_steps == 0) or last_batch
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        ddp_sync_context = nullcontext()
        if args.distributed and not update_step and hasattr(model, 'no_sync'):
            ddp_sync_context = model.no_sync()

        with ddp_sync_context:
            with amp_autocast():
                if args.model_type == 'deit' or args.model_type == 'swin':
                    student_logit, student_attn_info = model(input)
                else:
                    student_logit = model(input)
                if args.use_kd:
                    if args.kd_hard_and_soft == 0:
                        if args.teacher_type == 'deit':
                            soft_target, _ = teacher(input)
                        else:
                            soft_target = teacher(input)
                        loss = loss_fn(student_logit, soft_target)
                    elif args.kd_hard_and_soft == 1:
                        if args.teacher_type == 'deit' or args.teacher_type == 'swin':
                            soft_target, teacher_attn_info = teacher(input)
                        else:
                            soft_target = teacher(input)

                        loss = loss_fn(student_logit, target, soft_target)
                else:
                    student_logit = student_logit[0] if isinstance(student_logit, tuple) else student_logit
                    loss = loss_fn(student_logit, target)

            loss_for_log = loss.detach()

            if not args.distributed:
                losses_m.update(loss_for_log.item(), input.size(0))

            loss = loss / accum_steps
            if loss_scaler is not None:
                loss_scaler(
                    loss, optimizer,
                    clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    update_grad=update_step)
            else:
                loss.backward(create_graph=second_order)

                if update_step:
                    if args.clip_grad is not None:
                        dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad, mode=args.clip_mode)
                    optimizer.step()

        if update_step:
            optimizer.zero_grad()
            local_update_count += 1

            if args.rank == 0 and args.save_step_checkpoints and output_dir is not None:
                interval = max(1, int(args.step_checkpoint_interval))
                if warmup_updates > 0:
                    if args.save_initial_step_checkpoint and local_update_count == warmup_updates:
                        save_path = save_step_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            args=args,
                            output_dir=output_dir,
                            step_tag=f'step_{saved_step_count:04d}',
                            epoch=epoch,
                            batch_idx=batch_idx,
                            loss_scaler=loss_scaler,
                        )
                        saved_step_count += 1
                        _logger.info(f'Saved warmup-complete step checkpoint to {save_path}')

                    if local_update_count > warmup_updates and (local_update_count - warmup_updates) % interval == 0:
                        if max_step_checkpoints_to_save == 0 or saved_step_count < max_step_checkpoints_to_save:
                            save_path = save_step_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                args=args,
                                output_dir=output_dir,
                                step_tag=f'step_{saved_step_count:04d}',
                                epoch=epoch,
                                batch_idx=batch_idx,
                                loss_scaler=loss_scaler,
                            )
                            saved_step_count += 1
                            _logger.info(f'Saved step checkpoint to {save_path}')

                    if max_step_checkpoints_to_save > 0 and saved_step_count >= max_step_checkpoints_to_save:
                        stopped_early = True
                        if args.local_rank == 0:
                            _logger.info(
                                f'Stopped after saving {saved_step_count} step checkpoints '
                                f'following {warmup_updates} warmup updates.'
                            )
                        break
                elif local_update_count % interval == 0:
                    save_path = save_step_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        output_dir=output_dir,
                        step_tag=f'step_{local_update_count:04d}',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        loss_scaler=loss_scaler,
                    )
                    _logger.info(f'Saved step checkpoint to {save_path}')


        if args.local_rank == 0 and args.log_wandb and has_wandb and batch_idx == 0 and epoch == 0:
            wandb.watch(models = model, log_freq = 500)



        if update_step and model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        if update_step:
            num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            remaining_batches_in_epoch = max(last_idx - batch_idx, 0)
            remaining_epochs = max(int(getattr(args, 'epochs', epoch + 1)) - epoch - 1, 0)
            remaining_batches_total = remaining_batches_in_epoch + remaining_epochs * len(loader)
            eta_seconds = batch_time_m.avg * remaining_batches_total
            eta_duration = format_eta_duration(eta_seconds)
            eta_finish_time = (datetime.now() + timedelta(seconds=eta_seconds)).strftime('%Y-%m-%d %H:%M:%S')

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
                
            if args.local_rank == 0 and log_wandb:
                wandb.log({'train_loss': losses_m.avg}, step=num_updates)

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'ETA: {eta_duration}  '
                    'Done: {eta_finish_time}  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        eta_duration=eta_duration,
                        eta_finish_time=eta_finish_time,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None and update_step:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        if args.max_train_updates and local_update_count >= args.max_train_updates:
            stopped_early = True
            break

        end = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)]), local_update_count, stopped_early

def setup_alpha(model, loader, args, amp_autocast=suppress):
    model.eval()
    if getattr(args, 'local_rank', 0) == 0:
        print("setup alpha")
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            break
    
def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', model_type = 'deit', epoch=0, log_wandb = False):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    if getattr(args, 'local_rank', 0) == 0:
        print("model eval")
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                if model_type == 'rednet':
                    output = model.forward_test(input, softmax = False, post_process=False)
                else:
                    output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
                
        if args.local_rank == 0 and log_wandb:
            wandb.log({'val_loss': losses_m.avg, 'epoch': epoch})
            wandb.log({'val_top1': top1_m.avg, 'epoch': epoch})
            wandb.log({'val_top5': top5_m.avg, 'epoch': epoch})
            ## log the scaling factor of weight and activation
                
            

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

if __name__ == '__main__':
    args, args_text = parse_args()
    
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    os.environ['WORLD_SIZE'] = args.world_size
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"

    mp.spawn( main,
             args=((args, args_text),),
             nprocs=int(os.environ['WORLD_SIZE']),
             join=True)
   
