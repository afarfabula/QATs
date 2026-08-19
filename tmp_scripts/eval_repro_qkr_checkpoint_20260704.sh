#!/usr/bin/env bash
set -euo pipefail

QATS=/mlx_devbox/users/quyanyi/playground/QATs
OUT=/tmp/qat_recipe1_runs
TRAIN_EXP=repro_qkr_50ep_history_keepckpt_20260704
CKPT="${1:-${OUT}/${TRAIN_EXP}/checkpoint-49.pth.tar}"
EXP="eval_$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pth.tar)_20260704"
LOG=/tmp/${EXP}.log
SECONDS=0

if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 2
fi

echo "checkpoint=$CKPT"
echo "log=$LOG"

PYTHONUNBUFFERED=1 python3 "$QATS/qat_launch.py" \
  --method ofq --stage train \
  --config "$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "$OUT" --experiment "$EXP" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30409 --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-pretrained \
  --epochs 50 --batch-size 64 --workers 8 \
  --warmup-lr 1e-6 --weight-decay 0.0 \
  --resume "$CKPT" \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --extra-arg=--eval-only \
  --extra-arg=--cooldown-epochs --extra-arg=0 \
  --extra-arg=--log-interval --extra-arg=50 \
  2>&1 | tee "$LOG"

echo "wall_seconds=$SECONDS" | tee -a "$LOG"
echo "eval log: $LOG" | tee -a "$LOG"
