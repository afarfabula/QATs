#!/usr/bin/env bash
set -euo pipefail
QATS=/mlx_devbox/users/quyanyi/playground/QATs
OUT=/tmp/qats_stage1_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624
EXP=eval_init_quant_fastval_20260630
LOG=/tmp/${EXP}.log
SECONDS=0
PYTHONUNBUFFERED=1 python3 "$QATS/qat_launch.py" \
  --method ofq --stage train \
  --config "$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "$OUT" --experiment "$EXP" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 29985 --model-type swin \
  --epochs 1 --batch-size 256 --workers 8 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --quantized --amp --amp-dtype bf16 \
  --extra-arg=--eval-only --extra-arg=--static-graph --extra-arg=--log-interval --extra-arg=10 --extra-arg=--seed --extra-arg=42 \
  2>&1 | tee "$LOG"
echo "wall_seconds=$SECONDS"
echo "eval log: $LOG"
