import os
import sys
import shutil
import io
import bisect
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
from contextlib import nullcontext
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import matplotlib.pyplot as plt

sys.path.append("..")
from utils.utils import *
from utils import KD_loss
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models
from collections import OrderedDict
from PIL import Image
import pyarrow.parquet as pq
import quan
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from globalVal import globalVal


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser("gmmq")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--epochs", type=int, default=256, help="num of training epochs")
parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="init learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
parser.add_argument(
    "--save", type=str, default="./models", help="path for saving trained models"
)
parser.add_argument("--data", metavar="DIR", help="path to dataset")
parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing")
parser.add_argument("--teacher", type=str, default="resnet101", help="teacher model")
parser.add_argument("--student", type=str, default="resnet18", help="student model")
parser.add_argument("--n_bit", type=int, default=2, help="number of bits")
parser.add_argument(
    "--quantize_downsample",
    type=str,
    default="True",
    help="quantize downsampling layer or not",
)
parser.add_argument(
    "-j",
    "--workers",
    default=6,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument("--amp", action="store_true", help="enable autocast mixed precision")
parser.add_argument(
    "--amp_dtype",
    type=str,
    default="bf16",
    choices=["bf16", "fp16"],
    help="mixed precision dtype",
)
parser.add_argument("--channels_last", action="store_true", help="use channels_last memory format")
parser.add_argument("--compile", action="store_true", help="compile teacher/student models")
parser.add_argument("--compile_mode", type=str, default="default", help="torch.compile mode")
parser.add_argument("--compile_backend", type=str, default="inductor", help="torch.compile backend")
parser.add_argument("--prefetch_factor", type=int, default=4, help="dataloader prefetch factor")
parser.add_argument("--persistent_workers", action="store_true", help="use persistent dataloader workers")
parser.add_argument("--val_interval", type=int, default=1, help="validation interval in epochs")
parser.add_argument("--plot_interval", type=int, default=0, help="histogram save interval, 0 disables plotting")
parser.add_argument("--train_steps_per_epoch", type=int, default=0, help="limit train steps per epoch, 0 means full epoch")
parser.add_argument("--val_steps", type=int, default=0, help="limit validation steps, 0 means full eval")
parser.add_argument("--synthetic_data", action="store_true", help="use torchvision FakeData instead of on-disk ImageNet")
parser.add_argument("--synthetic_train_size", type=int, default=32768, help="synthetic train dataset size")
parser.add_argument("--synthetic_val_size", type=int, default=4096, help="synthetic validation dataset size")
parser.add_argument(
    "--dataset_format",
    type=str,
    default="imagefolder",
    choices=["imagefolder", "parquet", "parquet-iter"],
    help="dataset layout",
)
parser.add_argument("--skip_teacher_val", action="store_true", help="skip initial teacher validation")
parser.add_argument("--print_model", action="store_true", help="print full student model")
parser.add_argument("--print_params", action="store_true", help="print all parameter groups")
args = parser.parse_args()

CLASSES = 1000


class ImageNetParquetDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data_dir = os.path.join(root, "data") if os.path.isdir(os.path.join(root, "data")) else root
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"parquet data dir not found: {self.data_dir}")

        split_prefix = "validation" if split == "val" else split
        self.files = sorted(
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.startswith(f"{split_prefix}-") and f.endswith(".parquet")
        )
        if not self.files:
            raise FileNotFoundError(f"no parquet files for split={split_prefix} under {self.data_dir}")

        self._row_groups = []
        total_rows = 0
        for file_idx, path in enumerate(self.files):
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                self._row_groups.append((total_rows, file_idx, rg_idx, rg_rows))
                total_rows += rg_rows
        self._total_rows = total_rows
        self._cache_key = None
        self._cache_rows = None

    def __len__(self):
        return self._total_rows

    def _locate(self, index):
        pos = bisect.bisect_right(self._row_groups, (index, float("inf"), float("inf"), float("inf"))) - 1
        if pos < 0:
            raise IndexError(index)
        start, file_idx, rg_idx, rg_rows = self._row_groups[pos]
        if index >= start + rg_rows:
            raise IndexError(index)
        return start, file_idx, rg_idx

    def _load_row_group(self, file_idx, rg_idx):
        key = (file_idx, rg_idx)
        if self._cache_key != key:
            table = pq.ParquetFile(self.files[file_idx]).read_row_group(rg_idx, columns=["image", "label"])
            self._cache_rows = table.to_pylist()
            self._cache_key = key
        return self._cache_rows

    def __getitem__(self, index):
        start, file_idx, rg_idx = self._locate(index)
        rows = self._load_row_group(file_idx, rg_idx)
        sample = rows[index - start]
        image = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")
        label = int(sample["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ImageNetParquetIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, split="train", transform=None, shuffle=True, seed=42):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.data_dir = os.path.join(root, "data") if os.path.isdir(os.path.join(root, "data")) else root
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"parquet data dir not found: {self.data_dir}")

        split_prefix = "validation" if split == "val" else split
        self.files = sorted(
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.startswith(f"{split_prefix}-") and f.endswith(".parquet")
        )
        if not self.files:
            raise FileNotFoundError(f"no parquet files for split={split_prefix} under {self.data_dir}")

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
        return self._total_rows

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0

        indices = list(range(len(self._row_groups)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        assigned = indices[worker_id::num_workers]
        local_rng = random.Random(self.seed + self.epoch * 1009 + worker_id)

        handles = {}
        for rg_global_idx in assigned:
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
                local_rng.shuffle(order)
            for idx in order:
                image = Image.open(io.BytesIO(images[idx]["bytes"])).convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                yield image, int(labels[idx])

if not os.path.exists("log"):
    os.mkdir("log")

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join("log/log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
device = torch.device(globalVal.device)


def main():
    setup_seed(42)
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info("args = %s", args)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.amp_dtype == "fp16")

    # load model
    model_teacher = models.__dict__[args.teacher](weights="IMAGENET1K_V1")
    # model_teacher = nn.DataParallel(model_teacher, device_ids=device_ids).cuda()
    model_teacher = model_teacher.to(device)
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    if args.quantize_downsample == "True" or args.quantize_downsample == "1":
        args.quantize_downsample = True
    else:
        args.quantize_downsample = False

    model_student = models.__dict__[args.student](weights="IMAGENET1K_V1")
    modules_to_replace = quan.find_modules_to_quantize(model_student, args.n_bit)
    model_student = quan.replace_module_by_names(model_student, modules_to_replace)
    model_student = model_student.to(device)

    if args.channels_last:
        model_teacher = model_teacher.to(memory_format=torch.channels_last)
        model_student = model_student.to(memory_format=torch.channels_last)

    if args.compile and hasattr(torch, "compile"):
        model_teacher = torch.compile(
            model_teacher,
            mode=args.compile_mode,
            backend=args.compile_backend,
        )
        model_student = torch.compile(
            model_student,
            mode=args.compile_mode,
            backend=args.compile_backend,
        )

    if args.print_model:
        logging.info("student:")
        logging.info(model_student)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.to(device)
    criterion_kd = KD_loss.DistributionLoss()

    all_parameters = model_student.parameters()
    weight_parameters = []
    alpha_parameters = []
    beta_parameters = []
    theta_parameters = []

    for pname, p in model_student.named_parameters():
        if p.ndimension() == 4 and "bias" not in pname:
            if args.print_params:
                print("weight_param:", pname)
            weight_parameters.append(p)
        elif "quan_w_fn.mean_interval" in pname:
            if args.print_params:
                print("alpha_param:", pname)
            alpha_parameters.append(p)
        elif "quan_a_fn.mean_interval" in pname:
            if args.print_params:
                print("beta_param:", pname)
            beta_parameters.append(p)
        elif "quan_w_fn.alpha" in pname:
            if args.print_params:
                print("theta_param:", pname)
            theta_parameters.append(p)
        else:
            if args.print_params:
                print("other_param:", pname)

    weight_parameters_id = list(map(id, weight_parameters))
    alpha_parameters_id = list(map(id, alpha_parameters))
    beta_parameters_id = list(map(id, beta_parameters))
    theta_parameters_id = list(map(id, theta_parameters))
    other_parameters1 = list(
        filter(lambda p: id(p) not in weight_parameters_id, all_parameters)
    )
    other_parameters2 = list(
        filter(lambda p: id(p) not in alpha_parameters_id, other_parameters1)
    )
    other_parameters3 = list(
        filter(lambda p: id(p) not in beta_parameters_id, other_parameters2)
    )
    other_parameters = list(
        filter(lambda p: id(p) not in theta_parameters_id, other_parameters3)
    )

    optimizer = torch.optim.Adam(
        [
            {"params": alpha_parameters, "lr": args.learning_rate / 200},
            {"params": beta_parameters, "lr": args.learning_rate / 10},
            {"params": theta_parameters, "lr": args.learning_rate},
            {"params": other_parameters, "lr": args.learning_rate},
            {
                "params": weight_parameters,
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate,
            },
        ],
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (1.0 - step / args.epochs), last_epoch=-1
    )
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lambda step: 1.0, last_epoch=-1
    # )
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=1)
    start_epoch = 0
    best_top1_acc = 0

    checkpoint_tar = os.path.join(
        args.save,
        args.student
        + "_"
        + str(args.n_bit)
        + "bit_quantize_downsample_"
        + str(args.quantize_downsample),
        "checkpoint.pth.tar",
    )
    if os.path.exists(checkpoint_tar):
        logging.info("loading checkpoint {} ..........".format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint["epoch"] + 1
        best_top1_acc = checkpoint["best_top1_acc"]
        model_student.load_state_dict(checkpoint["state_dict"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        logging.info(
            "loaded checkpoint {} epoch = {}".format(
                checkpoint_tar, checkpoint["epoch"]
            )
        )

    # load training data
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # data augmentation
    crop_scale = 0.08
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if args.synthetic_data:
        train_dataset = datasets.FakeData(
            size=args.synthetic_train_size,
            image_size=(3, 224, 224),
            num_classes=CLASSES,
            transform=train_transforms,
        )
    elif args.dataset_format == "parquet-iter":
        train_dataset = ImageNetParquetIterableDataset(
            root=args.data,
            split="train",
            transform=train_transforms,
            shuffle=True,
        )
    elif args.dataset_format == "parquet":
        train_dataset = ImageNetParquetDataset(
            root=args.data,
            split="train",
            transform=train_transforms,
        )
    else:
        train_dataset = datasets.ImageFolder(traindir, transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not isinstance(train_dataset, torch.utils.data.IterableDataset),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.persistent_workers and args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )

    # load validation data
    if args.synthetic_data:
        val_dataset = datasets.FakeData(
            size=args.synthetic_val_size,
            image_size=(3, 224, 224),
            num_classes=CLASSES,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
    elif args.dataset_format in {"parquet", "parquet-iter"}:
        val_dataset = ImageNetParquetDataset(
            root=args.data,
            split="val",
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
    else:
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.persistent_workers and args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )

    if not args.skip_teacher_val:
        print("教师网络的精度")
        validate(-2, val_loader, model_teacher, criterion, args, amp_dtype)
    # train the model
    model_student = model_student.to(device)
    epoch = start_epoch

    while epoch < args.epochs:
        if args.plot_interval > 0 and epoch % args.plot_interval == 0:
            fname = "epoch" + str(epoch) + ".png"
            plt.figure(1)
            plt.hist(
                model_student.state_dict()["layer3.0.conv1.weight"].reshape(-1).cpu(),
                bins=200,
                range=(-0.8, 0.8),
            )
            plt.savefig(fname)
        train_obj, train_top1_acc, train_top5_acc = train(
            epoch,
            train_loader,
            model_student,
            model_teacher,
            criterion_kd,
            # criterion,
            optimizer,
            scaler,
            amp_dtype,
            args,
        )
        if args.val_interval > 0 and epoch % args.val_interval == 0:
            valid_obj, valid_top1_acc, valid_top5_acc = validate(
                epoch, val_loader, model_student, criterion, args, amp_dtype
            )
        else:
            valid_obj, valid_top1_acc, valid_top5_acc = train_obj, train_top1_acc, train_top5_acc

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        scheduler.step()

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model_student.state_dict(),
                "best_top1_acc": best_top1_acc,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            os.path.join(
                args.save,
                args.student
                + "_"
                + str(args.n_bit)
                + "bit_quantize_downsample_"
                + str(args.quantize_downsample),
            ),
        )

        epoch += 1

    training_time = (time.time() - start_t) / 3600
    print("total training time = {} hours".format(training_time))


def train(
    epoch,
    train_loader,
    model_student,
    model_teacher,
    criterion,
    optimizer,
    scaler,
    amp_dtype,
    args,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # constraint = ParaConstraint()

    model_student.train()
    # model_student.eval()
    model_teacher.eval()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group["lr"]
    print("learning_rate:", cur_lr)

    for i, (images, target) in enumerate(train_loader):
        if args.train_steps_per_epoch > 0 and i >= args.train_steps_per_epoch:
            break
        data_time.update(time.time() - end)
        # images = images.cuda()
        images = images.to(device, non_blocking=True)
        if args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        # target = target.cuda()
        target = target.to(device, non_blocking=True)

        # compute outputy
        amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp)
        with amp_context:
            logits_student = model_student(images)
            with torch.no_grad():
                logits_teacher = model_teacher(images)
            if globalVal.epoch <= 150:
                globalVal.loss = 0.0
                loss = criterion(logits_student, logits_teacher)
            else:
                loss1 = criterion(logits_student, logits_teacher)
                loss2 = globalVal.loss
                globalVal.loss = 0.0
                loss = loss1 + 0.01 * loss2

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

    return losses.avg, top1.avg, top5.avg


def validate(epoch, val_loader, model, criterion, args, amp_dtype):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.val_steps > 0 and i >= args.val_steps:
                break
            # images = images.cuda()
            images = images.to(device, non_blocking=True)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            # target = target.cuda()
            target = target.to(device, non_blocking=True)

            # compute output
            amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp)
            with amp_context:
                logits = model(images)
                loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        print(
            " * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
