#!/usr/bin/env bash
set -euo pipefail
cd /mlx_devbox/users/quyanyi/playground/QATs
EXP=swin_t_w4a4_stage2_partial_attnquant_weakref_lr3e5_noema_from50_to51_bsz128_kd_20260701
LOG=/tmp/${EXP}.log
OUT=/tmp/qats_stage2_outputs/stage2_50to51_algo_20260701
CKPT=/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630/checkpoint-50.pth.tar
rm -rf "${OUT}/${EXP}"
mkdir -p "${OUT}"
python3 /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  --method ofq --stage train \
  --config /mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml \
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \
  --output "${OUT}" --experiment "${EXP}" \
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 30108 \
  --model-type swin --teacher swin_t --teacher-type swin \
  --teacher-checkpoint /home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth --teacher-pretrained \
  --resume "${CKPT}" --no-resume-opt --epochs 51 --start-epoch 50 --scheduler-epochs 100 \
  --batch-size 128 --workers 8 --lr 3e-5 --min-lr 3e-5 --weight-decay 0.0 \
  --epoch-checkpoint-interval 1 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 --quantized \
  --amp --amp-dtype bf16 \
  --train-scheme ema_ref_attn_kl --ref-update prev_step \
  --ref-attn-kl-weight 0.0001 --ref-attn-loss kl_ref \
  --ref-head-mode custom:5:2,10:14,5:1,4:1,9:10,6:1,8:4,8:9,11:18,11:4 \
  --ref-warmup-epochs 50 \
  --quant-only-start-epoch 50 --trainable-policy head_norm_attn_quant \
  --extra-arg=--smoothing --extra-arg=0.0 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=none \
  --extra-arg=--color-jitter --extra-arg=0.0 \
  --extra-arg=--reprob --extra-arg=0.0 \
  --extra-arg=--log-interval --extra-arg=50 \
  --extra-arg=--seed --extra-arg=42 2>&1 | tee "${LOG}"
