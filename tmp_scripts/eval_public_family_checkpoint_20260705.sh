#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:?usage: $0 CHECKPOINT [EXP]}"
QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
OUT="${OUT:-/tmp/qat_public_repro/evals_20260705}"
EXP="${2:-eval_$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}" .pth.tar)_20260705}"
TEACHER="${TEACHER:-/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30495}"
LOG="${LOG:-/tmp/${EXP}.log}"
SECONDS=0

mkdir -p "${OUT}"

{
  echo "===== eval public-style W4A4-family checkpoint $(date '+%F %T') ====="
  echo "CKPT=${CKPT}"
  echo "EXP=${EXP}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}"
  echo "DEVICES=${DEVICES}"
  echo "MASTER_PORT=${MASTER_PORT}"
  git -C "${QATS}" rev-parse --short HEAD || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  echo
} | tee "${LOG}"

PYTHONUNBUFFERED=1 python3 "${QATS}/qat_launch.py" \
  --method ofq --stage train \
  --config "${QATS}/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data "${DATA}" --dataset-format parquet \
  --output "${OUT}" --experiment "${EXP}" \
  --devices "${DEVICES}" --nproc-per-node 8 --master-port "${MASTER_PORT}" --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-checkpoint "${TEACHER}" --teacher-pretrained \
  --resume "${CKPT}" --epochs 0 --start-epoch 0 \
  --batch-size 128 --workers 8 --lr 1e-5 --min-lr 1e-5 --weight-decay 0.0 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--eval-only \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1 \
  --extra-arg=--color-jitter --extra-arg=0.4 \
  --extra-arg=--reprob --extra-arg=0.25 \
  --extra-arg=--log-interval --extra-arg=50 \
  --extra-arg=--seed --extra-arg=42 \
  2>&1 | tee -a "${LOG}"

{
  echo "wall_seconds=${SECONDS}"
  echo "eval_log=${LOG}"
} | tee -a "${LOG}"
