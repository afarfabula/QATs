#!/usr/bin/env bash
set -euo pipefail
cd /mlx_devbox/users/quyanyi/playground/QATs
EXP=swin_t_w4a4_stage1_resume_diag_continue50_to60_20260701
LOG=/tmp/train_${EXP}.log
OUT=/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624
CKPT=/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630/checkpoint-50.pth.tar
TEACHER=/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth
rm -rf "${OUT}/${EXP}"
mkdir -p "${OUT}"
SECONDS=0
PYTHONUNBUFFERED=1 python3 /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  --method ofq --stage train \
  --config /mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "${OUT}" --experiment "${EXP}" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30138 \
  --model-type swin --teacher swin_t --teacher-type swin --teacher-checkpoint "${TEACHER}" --teacher-pretrained \
  --resume "${CKPT}" --no-resume-opt --epochs 60 --start-epoch 50 --scheduler-epochs 60 \
  --batch-size 256 --workers 8 --lr 2e-4 --min-lr 1e-5 --weight-decay 0.0 --epoch-checkpoint-interval 1 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 \
  --quantized --amp --amp-dtype bf16 \
  --train-scheme baseline \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing --extra-arg=0.0 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=none \
  --extra-arg=--color-jitter --extra-arg=0.0 \
  --extra-arg=--reprob --extra-arg=0.0 \
  --extra-arg=--log-interval --extra-arg=50 \
  --extra-arg=--seed --extra-arg=42 2>&1 | tee "${LOG}"
echo "wall_seconds=${SECONDS}"
