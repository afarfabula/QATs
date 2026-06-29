#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import signal
import time
from typing import List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用大矩阵乘法持续拉高指定 GPU 的 utilization"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="4,5,6,7",
        help="要打满的 GPU 编号，逗号分隔，默认 4,5,6,7",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=12288,
        help="方阵边长，默认 12288；不够满可以继续调大",
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=4,
        help="每张卡并发 CUDA stream 数，默认 4",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
        help="计算精度，默认 bf16",
    )
    parser.add_argument(
        "--mode",
        choices=["matmul", "cuda_sleep"],
        default="matmul",
        help="负载模式；cuda_sleep 用 CUDA busy-wait 拉高 util 且占用显存更少",
    )
    parser.add_argument(
        "--sleep-cycles",
        type=int,
        default=200_000_000,
        help="cuda_sleep 每个 stream 每轮 busy-wait cycles，默认 200000000",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=0,
        help="运行时长；0 表示一直跑到 Ctrl+C",
    )
    parser.add_argument(
        "--sleep-ms",
        type=float,
        default=0.0,
        help="每轮同步后额外 sleep 的毫秒数，默认 0",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=50,
        help="每多少轮打印一次进度，默认 50",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


def burn_worker(
    gpu_id: int,
    matrix_size: int,
    num_streams: int,
    dtype_name: str,
    mode: str,
    sleep_cycles: int,
    seconds: int,
    sleep_ms: float,
    report_every: int,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    dtype = resolve_dtype(dtype_name)
    props = torch.cuda.get_device_properties(device)
    total_mem_gb = props.total_memory / 1024**3

    print(
        f"[GPU {gpu_id}] start burn: name={props.name}, mem={total_mem_gb:.1f}GB, "
        f"mode={mode}, matrix={matrix_size}, streams={num_streams}, dtype={dtype_name}",
        flush=True,
    )

    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    if mode == "cuda_sleep":
        start_time = time.time()
        step = 0
        while True:
            if seconds > 0 and time.time() - start_time >= seconds:
                break

            for stream in streams:
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(sleep_cycles)

            torch.cuda.synchronize(device)
            step += 1

            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

            if report_every > 0 and step % report_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[GPU {gpu_id}] steps={step}, elapsed={elapsed:.1f}s, "
                    f"avg_step={elapsed / step:.3f}s",
                    flush=True,
                )

        torch.cuda.synchronize(device)
        print(f"[GPU {gpu_id}] done", flush=True)
        return

    left: List[torch.Tensor] = []
    right: List[torch.Tensor] = []
    out: List[torch.Tensor] = []

    for _ in range(num_streams):
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        c = torch.empty(matrix_size, matrix_size, device=device, dtype=dtype)
        left.append(a)
        right.append(b)
        out.append(c)

    for idx, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            out[idx] = left[idx] @ right[idx]
            left[idx], right[idx] = out[idx], left[idx]

    torch.cuda.synchronize(device)

    start_time = time.time()
    step = 0

    while True:
        if seconds > 0 and time.time() - start_time >= seconds:
            break

        for idx, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                out[idx] = left[idx] @ right[idx]
                left[idx], right[idx] = out[idx], left[idx]

        torch.cuda.synchronize(device)
        step += 1

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        if report_every > 0 and step % report_every == 0:
            elapsed = time.time() - start_time
            print(
                f"[GPU {gpu_id}] steps={step}, elapsed={elapsed:.1f}s, "
                f"avg_step={elapsed / step:.3f}s",
                flush=True,
            )

    torch.cuda.synchronize(device)
    print(f"[GPU {gpu_id}] done", flush=True)


def main() -> None:
    args = parse_args()

    gpu_ids = [int(part.strip()) for part in args.gpus.split(",") if part.strip()]
    if not gpu_ids:
        raise ValueError("--gpus 不能为空")

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA")

    visible = torch.cuda.device_count()
    bad = [gpu for gpu in gpu_ids if gpu < 0 or gpu >= visible]
    if bad:
        raise ValueError(f"GPU 编号越界: {bad}，当前可见 GPU 数={visible}")

    print(
        f"PID={os.getpid()} visible_gpus={visible} target={gpu_ids} "
        f"matrix={args.matrix_size} streams={args.streams} dtype={args.dtype}",
        flush=True,
    )
    print("按 Ctrl+C 可以停止全部子进程", flush=True)

    ctx = mp.get_context("spawn")
    workers: List[mp.Process] = []

    try:
        for gpu_id in gpu_ids:
            p = ctx.Process(
                target=burn_worker,
                args=(
                    gpu_id,
                    args.matrix_size,
                    args.streams,
                    args.dtype,
                    args.mode,
                    args.sleep_cycles,
                    args.seconds,
                    args.sleep_ms,
                    args.report_every,
                ),
                daemon=False,
            )
            p.start()
            workers.append(p)

        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在停止所有 GPU 负载进程...", flush=True)
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join(timeout=5)
    finally:
        for p in workers:
            if p.is_alive():
                p.kill()


if __name__ == "__main__":
    main()
