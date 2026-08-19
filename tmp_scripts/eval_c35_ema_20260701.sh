#!/usr/bin/env bash
set -euo pipefail
cd /mlx_devbox/users/quyanyi/playground/QATs
EXP=eval_c35_ema_20260701
CKPT=/tmp/qats_stage2_outputs/stage2_50to51_teacherref_20260701/swin_t_w4a4_stage2_c35_teacher_softonly_partial_ema999_from50_to51_20260701/checkpoint-51.ema.pth.tar
OUT=/tmp/qats_stage2_outputs/evals_20260701
PYTHONUNBUFFERED=1 python3 /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  --method ofq --stage train \
  --config /mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "$OUT" --experiment "$EXP" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30149 \
  --model-type swin --teacher swin_t --teacher-type swin --teacher-checkpoint /home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth --teacher-pretrained \
  --resume "$CKPT" --epochs 0 --start-epoch 0 \
  --batch-size 128 --workers 8 --lr 5e-6 --min-lr 5e-6 --weight-decay 0.0 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 \
  --quantized --amp --amp-dtype bf16 --train-scheme baseline \
  --extra-arg=--eval-only \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing --extra-arg=0.0 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=none \
  --extra-arg=--color-jitter --extra-arg=0.0 \
  --extra-arg=--reprob --extra-arg=0.0 \
  --extra-arg=--log-interval --extra-arg=50 \
  --extra-arg=--seed --extra-arg=42 2>&1 | tee /tmp/eval_c35_ema_20260701.log
