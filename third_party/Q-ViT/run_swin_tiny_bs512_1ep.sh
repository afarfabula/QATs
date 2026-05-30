#!/usr/bin/env bash
# Launch single-GPU Swin-Tiny FP training, 1 epoch, BS=512 (try max).
# Foreground = false (nohup background). Log lives inside repo for IDE viewing.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU="${GPU:-0}"
DATA_PATH="${DATA_PATH:-/tmp/imagenet1k_full_parquet}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/out/swin_tiny_bs512_1ep}"

BS="${BS:-512}"
BS_EVAL="${BS_EVAL:-512}"
NUM_WORKERS="${NUM_WORKERS:-16}"

mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 nohup python3 -u main.py \
  --model swin_tiny_patch4_window7_224 \
  --pretrained \
  --data-set IMNET_PARQUET \
  --data-path "${DATA_PATH}" \
  --epochs 1 \
  --warmup-epochs 0 \
  --weight-decay 0.05 \
  --drop-path 0.2 \
  --batch-size "${BS}" \
  --batch-size-eval "${BS_EVAL}" \
  --num_workers "${NUM_WORKERS}" \
  --output_dir "${OUT_DIR}" \
  > "${OUT_DIR}/console.log" 2>&1 &

PID=$!
echo $PID > "${OUT_DIR}/pid.txt"
echo "launched_pid=${PID}"
echo "out_dir=${OUT_DIR}"
echo "log=${OUT_DIR}/console.log"
echo "batch_size=${BS} batch_size_eval=${BS_EVAL} num_workers=${NUM_WORKERS}"
