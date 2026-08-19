#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
EXP="${EXP:-ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
CKPT="${CKPT:-/mlx_devbox/users/quyanyi/playground/QATs/checkpoints/resume10_public_family/checkpoint-10.pth.tar}"
TEACHER="${TEACHER:-/mlx_devbox/users/quyanyi/playground/QATs/checkpoints/pretrained/swin_t-704ceda3.pth}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-31561}"
REF_HEAD_MODE="${REF_HEAD_MODE:-custom_subset:6:1,8:4,8:9,11:18,11:4}"
REF_WEIGHT_OVERRIDES="${REF_WEIGHT_OVERRIDES:-28:0.00030,29:0.00030,36:0.00035,37:0.00035,44:0.00035,45:0.00035,52:0.00030,53:0.00030}"
SECONDS=0

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== Public-family OFQ resume10->60 + sparse pulse prev-step refKL C $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "CKPT=${CKPT}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "REF_HEAD_MODE=${REF_HEAD_MODE}"
  echo "REF_WEIGHT_OVERRIDES=${REF_WEIGHT_OVERRIDES}"
  test -f "${CKPT}" && ls -lh "${CKPT}"
  test -f "${TEACHER}" && ls -lh "${TEACHER}" || true
  echo "train_shards=$(find "${DATA}/data" -maxdepth 1 -name 'train-*.parquet' | wc -l)"
  echo "validation_shards=$(find "${DATA}/data" -maxdepth 1 -name 'validation-*.parquet' | wc -l)"
  git -C "${QATS}" rev-parse --short HEAD || true
  test -e /dev/nvidia0 && echo gpu-device-present || echo no-gpu-device
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  echo
} | tee "${LOG}"

PYTHONUNBUFFERED=1 python3 "${QATS}/qat_launch.py" \
  --method ofq --stage train \
  --config "${QATS}/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data "${DATA}" --dataset-format parquet \
  --output "${OUT}" --experiment "${EXP}" \
  --devices "${DEVICES}" --nproc-per-node 8 --master-port "${MASTER_PORT}" \
  --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-checkpoint "${TEACHER}" --teacher-pretrained \
  --resume "${CKPT}" --no-resume-opt \
  --epochs 60 --scheduler-epochs 60 \
  --batch-size 64 --workers 8 --lr 1.5e-5 --min-lr 5e-6 --weight-decay 0.0 \
  --epoch-checkpoint-interval 1 --checkpoint-hist 60 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --train-scheme ema_ref_attn_kl \
  --ref-update prev_step --ref-update-interval 50 \
  --ref-attn-loss kl_ref \
  --ref-attn-kl-weight 0.0 \
  --ref-attn-kl-weight-epoch-overrides "${REF_WEIGHT_OVERRIDES}" \
  --ref-head-mode "${REF_HEAD_MODE}" \
  --ref-warmup-epochs 28 \
  --ref-attn-kl-drop-prob 0.50 \
  --extra-arg=--static-graph \
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1 \
  --extra-arg=--color-jitter --extra-arg=0.4 \
  --extra-arg=--reprob --extra-arg=0.25 \
  --extra-arg=--log-interval --extra-arg=50 \
  --extra-arg=--seed --extra-arg=42 \
  2>&1 | tee -a "${LOG}"

{
  echo "wall_seconds=${SECONDS}"
  echo "train_log=${LOG}"
  echo "output=${OUT}/${EXP}"
} | tee -a "${LOG}"
