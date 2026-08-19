#!/usr/bin/env bash
set -euo pipefail
cd /mlx_devbox/users/quyanyi/playground/QATs
EXP=swin_t_w4a4_stage2_prevstep_custom10_from50_bsz224_nokd_selheads_100step_bench_20260630
LOG=/tmp/${EXP}.log
OUT=/tmp/qats_stage2_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624
CKPT=/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630/checkpoint-50.pth.tar
rm -rf "${OUT}/${EXP}"
mkdir -p "${OUT}"
python3 /mlx_devbox/users/quyanyi/playground/QATs/qat_launch.py \
  --method ofq \
  --stage train \
  --config /mlx_devbox/users/quyanyi/playground/QATs/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml \
  --model swin_t \
  --data /tmp/imagenet1k_full_parquet \
  --dataset-format parquet \
  --output "${OUT}" \
  --experiment "${EXP}" \
  --devices 0,1,2,3,4,5,6,7 \
  --nproc-per-node 8 \
  --master-port 30005 \
  --model-type swin \
  --resume "${CKPT}" \
  --no-resume-opt \
  --epochs 52 \
  --start-epoch 50 \
  --scheduler-epochs 100 \
  --batch-size 224 \
  --workers 8 \
  --lr 5e-6 \
  --min-lr 5e-6 \
  --weight-decay 0.0 \
  --epoch-checkpoint-interval 1 \
  --wbits 4 \
  --abits 4 \
  --wq-mode statsq \
  --aq-mode lsq \
  --wq-per-channel \
  --aq-per-channel \
  --aq-clip-learnable \
  --pretrained \
  --pretrained-initialized \
  --quantized \
  --amp \
  --amp-dtype bf16 \
  --train-scheme ema_ref_attn_kl \
  --ref-update prev_step \
  --ref-attn-kl-weight 0.001 \
  --ref-attn-loss kl_ref \
  --ref-head-mode custom:5:2,10:14,5:1,4:1,9:10,6:1,8:4,8:9,11:18,11:4 \
  --ref-warmup-epochs 50 \
  --extra-arg=--max_train_updates \
  --extra-arg=100 \
  --extra-arg=--skip_validate \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing \
  --extra-arg=0.0 \
  --extra-arg=--mixup \
  --extra-arg=0.0 \
  --extra-arg=--cutmix \
  --extra-arg=0.0 \
  --extra-arg=--aa \
  --extra-arg=none \
  --extra-arg=--color-jitter \
  --extra-arg=0.0 \
  --extra-arg=--reprob \
  --extra-arg=0.0 \
  --extra-arg=--log-interval \
  --extra-arg=10 \
  --extra-arg=--seed \
  --extra-arg=42 2>&1 | tee "${LOG}"
