#!/usr/bin/env bash
set -euo pipefail

QATS=/mlx_devbox/users/quyanyi/playground/QATs
OUT=/tmp/qat_recipe1_runs
EXP=w3_strict_resume10_to20_20260704
TEACHER=/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth
CKPT=/tmp/qat_recipe1_runs/recipe1_w3_best_10ep_prefeatrecon100_q4_final8_20260703/checkpoint-10.pth.tar
LOG=/tmp/train_${EXP}.log
SECONDS=0

if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 2
fi

rm -rf "${OUT}/${EXP}"
mkdir -p "$OUT"
echo "checkpoint=$CKPT"
echo "output=${OUT}/${EXP}"
echo "log=$LOG"

PYTHONUNBUFFERED=1 python3 "$QATS/qat_launch.py" \
  --method ofq --stage train \
  --config "$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "$OUT" --experiment "$EXP" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30414 --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-checkpoint "$TEACHER" --teacher-pretrained \
  --resume "$CKPT" \
  --epochs 20 --scheduler-epochs 20 --batch-size 256 --workers 8 \
  --lr 2e-4 --min-lr 1e-5 --weight-decay 0.0 \
  --quant-lr-multiplier 4 \
  --quant-lr-multiplier-epoch-overrides 19:8 \
  --epoch-checkpoint-interval 10 --checkpoint-hist 2 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 \
  --quantized --amp --amp-dtype bf16 \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing --extra-arg=0.0 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=none \
  --extra-arg=--color-jitter --extra-arg=0.0 \
  --extra-arg=--reprob --extra-arg=0.0 \
  --extra-arg=--log-interval --extra-arg=50 \
  --extra-arg=--seed --extra-arg=42 \
  2>&1 | tee "$LOG"

echo "wall_seconds=$SECONDS" | tee -a "$LOG"
echo "train log: $LOG" | tee -a "$LOG"
echo "output: ${OUT}/${EXP}" | tee -a "$LOG"
