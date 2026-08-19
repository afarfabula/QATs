#!/usr/bin/env bash
set -euo pipefail

QATS=/mlx_devbox/users/quyanyi/playground/QATs
OUT=/tmp/qat_recipe1_runs
EXP=repro_qkr_50ep_history_keepckpt_20260704
LOG=/tmp/train_${EXP}.log
SECONDS=0

rm -rf "${OUT}/${EXP}"
mkdir -p "$OUT"
echo "output=${OUT}/${EXP}"
echo "log=$LOG"

# Reproduce the historical QKR 50-epoch source run as closely as possible while
# keeping enough checkpoint history for verification. The historical run used
# skip_validate and was later evaluated via a resume/eval path.
PYTHONUNBUFFERED=1 python3 "$QATS/qat_launch.py" \
  --method ofq --stage train \
  --config "$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "$OUT" --experiment "$EXP" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30408 --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-pretrained \
  --epochs 50 --batch-size 64 --grad-accum-steps 1 --workers 8 \
  --warmup-lr 1e-6 --weight-decay 0.0 \
  --epoch-checkpoint-interval 10 --checkpoint-hist 20 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --extra-arg=--skip_validate \
  --extra-arg=--cooldown-epochs --extra-arg=0 \
  --extra-arg=--log-interval --extra-arg=50 \
  2>&1 | tee "$LOG"

echo "wall_seconds=$SECONDS" | tee -a "$LOG"
echo "train log: $LOG" | tee -a "$LOG"
echo "output: ${OUT}/${EXP}" | tee -a "$LOG"
