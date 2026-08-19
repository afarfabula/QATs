#!/usr/bin/env bash
set -euo pipefail
QATS=/mlx_devbox/users/quyanyi/playground/QATs
OUT=/tmp/qat_recipe1_runs
EXP=smoke_recipe1_tsd_quantteacher_1update_20260703
TEACHER=/tmp/qat_recipe1_runs/recipe1_w3_best_10ep_prefeatrecon100_q4_final8_20260703/checkpoint-10.pth.tar
LOG=/tmp/train_${EXP}.log
rm -rf "${OUT}/${EXP}"
mkdir -p "$OUT"
PYTHONUNBUFFERED=1 python3 "$QATS/qat_launch.py" \
  --method ofq --stage train \
  --config "$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "$OUT" --experiment "$EXP" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30348 --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-checkpoint "$TEACHER" --quant-teacher \
  --epochs 1 --scheduler-epochs 1 --batch-size 256 --workers 8 \
  --lr 2e-4 --min-lr 1e-5 --weight-decay 0.0 \
  --quant-lr-multiplier 4 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 \
  --quantized --amp --amp-dtype bf16 \
  --extra-arg=--max_train_updates --extra-arg=1 \
  --extra-arg=--skip_validate \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing --extra-arg=0.0 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=none \
  --extra-arg=--color-jitter --extra-arg=0.0 \
  --extra-arg=--reprob --extra-arg=0.0 \
  --extra-arg=--log-interval --extra-arg=1 \
  --extra-arg=--seed --extra-arg=42 \
  2>&1 | tee "$LOG"
