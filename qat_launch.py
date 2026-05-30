#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
THIRD_PARTY = ROOT / "third_party"


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
    append_optional_value(command, "--resume", normalize_path(args.resume))

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
    append_optional_flag(command, "--teacher_pretrained", args.teacher_pretrained)
    append_optional_flag(command, "--quantized", args.quantized)
    append_optional_flag(command, "--qk_reparam", args.qk_reparam)

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
    parser.add_argument("--model-type", dest="model_type", type=str, choices=["deit", "swin"], help="OFQ model_type")
    parser.add_argument("--teacher-type", dest="teacher_type", type=str, choices=["deit", "swin"], help="OFQ teacher_type")
    parser.add_argument("--wq-mode", dest="wq_mode", type=str, default="statsq")
    parser.add_argument("--aq-mode", dest="aq_mode", type=str, default="lsq")
    parser.add_argument("--wq-per-channel", dest="wq_per_channel", action="store_true")
    parser.add_argument("--aq-per-channel", dest="aq_per_channel", action="store_true")
    parser.add_argument("--wq-clip-learnable", dest="wq_clip_learnable", action="store_true")
    parser.add_argument("--aq-clip-learnable", dest="aq_clip_learnable", action="store_true")
    parser.add_argument("--use-kd", dest="use_kd", action="store_true")
    parser.add_argument("--teacher-pretrained", dest="teacher_pretrained", action="store_true")
    parser.add_argument("--pretrained-initialized", dest="pretrained_initialized", action="store_true")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--qk-reparam", dest="qk_reparam", action="store_true")
    parser.add_argument("--boundary-range", dest="boundary_range", type=float)
    parser.add_argument("--freeze-for-n-epochs", dest="freeze_for_n_epochs", type=int)

    parser.add_argument("--quantize-downsample", dest="quantize_downsample", type=str2bool, default=True, help="AOQ 是否量化 downsample")
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

    completed = subprocess.run(command, cwd=str(cwd), env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
