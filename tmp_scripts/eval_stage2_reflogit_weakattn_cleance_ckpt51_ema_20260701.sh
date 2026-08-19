#!/usr/bin/env bash
set -euo pipefail
QATS=/mlx_devbox/users/quyanyi/playground/QATs
BASE=/tmp/qats_stage2_outputs/stage2_50to51_algo_20260701
SRC=${BASE}/swin_t_w4a4_stage2_reflogit_weakattn_cleance_from50_to51_bsz128_kd_ema999_20260701
CKPT=${SRC}/checkpoint-51.ema.pth.tar
EXP=eval_stage2_reflogit_weakattn_cleance_ckpt51_ema_20260701
LOG=/tmp/${EXP}.log
ls -lh "${CKPT}"
PYTHONUNBUFFERED=1 python3 "$QATS/qat_launch.py" \
  --method ofq --stage train \
  --config "$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "$BASE" --experiment "$EXP" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30105 --model-type swin \
  --epochs 1 --batch-size 256 --workers 8 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --quantized --amp --amp-dtype bf16 \
  --extra-arg=--eval-only --extra-arg=--initial-checkpoint --extra-arg="$CKPT" \
  --extra-arg=--static-graph --extra-arg=--log-interval --extra-arg=10 --extra-arg=--seed --extra-arg=42 \
  2>&1 | tee "$LOG"
