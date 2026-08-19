#!/usr/bin/env bash
set -euo pipefail
QATS=/mlx_devbox/users/quyanyi/playground/QATs
OUT=/tmp/qats_stage1_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624
TEACHER=/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth
EXP=swin_t_w4a4_stage1_bs256_kd_noaug_setupalpha16_1ep_fastval_20260630
LOG=/tmp/train_${EXP}.log
SECONDS=0
PYTHONUNBUFFERED=1 python3 "$QATS/qat_launch.py" \
  --method ofq --stage train \
  --config "$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "$OUT" --experiment "$EXP" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 29984 --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-checkpoint "$TEACHER" --teacher-pretrained \
  --epochs 1 --scheduler-epochs 50 --batch-size 256 --workers 8 \
  --lr 2e-4 --min-lr 1e-5 --weight-decay 0.0 --epoch-checkpoint-interval 1 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 \
  --quantized --amp --amp-dtype bf16 --setup-alpha-batches 16 \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing --extra-arg=0.0 --extra-arg=--mixup --extra-arg=0.0 --extra-arg=--cutmix --extra-arg=0.0 --extra-arg=--aa --extra-arg=none --extra-arg=--color-jitter --extra-arg=0.0 --extra-arg=--reprob --extra-arg=0.0 \
  --extra-arg=--log-interval --extra-arg=50 --extra-arg=--seed --extra-arg=42 \
  2>&1 | tee "$LOG"
echo "wall_seconds=$SECONDS"
echo "train log: $LOG"
