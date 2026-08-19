#!/usr/bin/env bash
set -euo pipefail
QATS=/mlx_devbox/users/quyanyi/playground/QATs
OUT=/tmp/qat_recipe1_runs
EXP=recipe1_n_quantlr4_confkd_p1_5ep_20260702
TEACHER=/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth
LOG=/tmp/train_${EXP}.log
SECONDS=0
rm -rf "${OUT}/${EXP}"
mkdir -p "$OUT"
echo "output=${OUT}/${EXP}"
echo "log=$LOG"
PYTHONUNBUFFERED=1 python3 "$QATS/qat_launch.py" \
  --method ofq --stage train \
  --config "$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "$OUT" --experiment "$EXP" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30330 --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-checkpoint "$TEACHER" --teacher-pretrained \
  --epochs 5 --scheduler-epochs 5 --batch-size 256 --workers 8 \
  --lr 2e-4 --min-lr 1e-5 --weight-decay 0.0 \
  --quant-lr-multiplier 4 \
  --teacher-confidence-kd-power 1.0 \
  --teacher-soft-temperature 1.0 \
  --epoch-checkpoint-interval 5 --checkpoint-hist 2 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 \
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
