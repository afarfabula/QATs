# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import glob
import random
from io import BytesIO
from bisect import bisect_right

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import numpy as np
from PIL import Image

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class ParquetImageNetDataset(torch.utils.data.Dataset):
    """
    Minimal parquet-backed ImageNet dataset for HF-style shards:
    - train-*.parquet
    - validation-*.parquet
    """
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required for parquet dataset. Install with: pip install pyarrow"
            ) from exc

        self._pq = pq
        self.files = sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet")))
        if not self.files:
            raise RuntimeError(
                f"No parquet files found for split '{split}' under {os.path.join(root, 'data')}"
            )

        self._parquet_files = []
        self._row_groups = []  # (file_index, row_group_index, num_rows)
        cumulative = 0
        self._cumulative_rows = []
        for file_idx, path in enumerate(self.files):
            pf = self._pq.ParquetFile(path)
            self._parquet_files.append(pf)
            for rg_idx in range(pf.num_row_groups):
                num_rows = pf.metadata.row_group(rg_idx).num_rows
                self._row_groups.append((file_idx, rg_idx, num_rows))
                cumulative += num_rows
                self._cumulative_rows.append(cumulative)

        self._cache_key = None
        self._cache_rows = None

    def __len__(self):
        return self._cumulative_rows[-1]

    def _get_row_group(self, group_idx):
        file_idx, rg_idx, _ = self._row_groups[group_idx]
        cache_key = (file_idx, rg_idx)
        if self._cache_key == cache_key and self._cache_rows is not None:
            return self._cache_rows

        table = self._parquet_files[file_idx].read_row_group(rg_idx, columns=["image", "label"])
        rows = table.to_pylist()
        self._cache_key = cache_key
        self._cache_rows = rows
        return rows

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset length {len(self)}")

        group_idx = bisect_right(self._cumulative_rows, index)
        prev_cum = 0 if group_idx == 0 else self._cumulative_rows[group_idx - 1]
        row_idx = index - prev_cum
        row = self._get_row_group(group_idx)[row_idx]

        image_bytes = row["image"]["bytes"]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


class ParquetIterableImageNet(torch.utils.data.IterableDataset):
    """
    Streaming parquet ImageNet loader optimized for throughput:
    - Iterates parquet files row-group by row-group (sequential disk I/O,
      no random per-sample row-group reload).
    - Splits the row-group list across (DDP rank, DataLoader worker), so each
      worker reads its own private slice of files exactly once per epoch.
    - Shuffles the row-group order each epoch and shuffles rows within each
      row group; for ImageNet (HF shards already class-mixed) this is enough
      randomness for training.
    - Returns transformed (image, label) tuples just like ImageFolder.

    Set the epoch via `dataset.set_epoch(epoch)` before iterating to get a
    new deterministic shuffle.
    """

    def __init__(self, root, split, transform=None, shuffle=True, seed=0):
        super().__init__()
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required for parquet dataset. Install with: pip install pyarrow"
            ) from exc

        self._pq = pq
        self.root = root
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.files = sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet")))
        if not self.files:
            raise RuntimeError(
                f"No parquet files found for split '{split}' under {os.path.join(root, 'data')}"
            )

        # Pre-scan row-group sizes so we know dataset length without re-opening
        # parquet handles inside workers.
        self._row_groups = []  # list of (file_path, rg_idx, num_rows)
        total = 0
        for path in self.files:
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                num_rows = pf.metadata.row_group(rg_idx).num_rows
                self._row_groups.append((path, rg_idx, num_rows))
                total += num_rows
        self._length = total

    def __len__(self):
        # Accurate per-epoch length. DataLoader / progress meters use this.
        return self._length

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _shard_row_groups(self):
        """Partition row groups across (rank, worker)."""
        # DDP rank
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0

        # DataLoader worker
        info = torch.utils.data.get_worker_info()
        if info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = info.num_workers
            worker_id = info.id

        total_shards = world_size * num_workers
        global_id = rank * num_workers + worker_id

        rgs = list(range(len(self._row_groups)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(rgs)

        # Round-robin assignment so each shard gets a roughly equal count
        # regardless of total length (last shard may have one fewer rg).
        assigned = rgs[global_id::total_shards]
        return assigned

    def __iter__(self):
        assigned = self._shard_row_groups()

        # Per-iter rng so within-rg shuffle differs per epoch / per worker.
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = random.Random(self.seed + self.epoch * 1009 + worker_id)

        # Cache parquet handles inside the worker process.
        handles = {}
        for rg_global_idx in assigned:
            path, rg_idx, _ = self._row_groups[rg_global_idx]
            pf = handles.get(path)
            if pf is None:
                pf = self._pq.ParquetFile(path)
                handles[path] = pf

            # Read the whole row group once, sequential I/O, no Python list
            # materialization of every row's metadata.
            table = pf.read_row_group(rg_idx, columns=["image", "label"])
            # to_pydict gives us 2 columns of length N; cheaper than to_pylist
            # because we skip the per-row dict allocations.
            cols = table.to_pydict()
            images = cols["image"]
            labels = cols["label"]
            n = len(images)

            order = list(range(n))
            if self.shuffle:
                rng.shuffle(order)

            for i in order:
                img_struct = images[i]
                # HF imagenet stores image as struct {'bytes': <bytes>, 'path': str}
                if isinstance(img_struct, dict):
                    img_bytes = img_struct["bytes"]
                else:
                    img_bytes = img_struct
                with Image.open(BytesIO(img_bytes)) as im:
                    im = im.convert("RGB")
                    if self.transform is not None:
                        im = self.transform(im)
                yield im, int(labels[i])


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET_PARQUET':
        split = 'train' if is_train else 'validation'
        dataset = ParquetImageNetDataset(args.data_path, split=split, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET_PARQUET_ITER':
        split = 'train' if is_train else 'validation'
        dataset = ParquetIterableImageNet(
            args.data_path,
            split=split,
            transform=transform,
            shuffle=is_train,
        )
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
