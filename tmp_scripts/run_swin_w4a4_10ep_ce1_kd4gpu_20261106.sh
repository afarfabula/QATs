#!/usr/bin/env bash
# Swin-T W4A4 10-epoch ground-truth audit on 4x RTX3090.
#
# Controlled against:
#   tmp_scripts/run_swin_w4a4_lambda0_control_10ep_4gpu_20260929.sh
#
# The only intentional objective change is:
#   pure soft KD -> soft KD at T=2.75 + 1.0 * hard-label CE
# All data, seed, augmentation, quantization, optimizer and effective batch settings
# remain the same as the 10-epoch lambda=0 control.
#
# Usage:
#   DEVICES=0,1,2,4 bash tmp_scripts/run_swin_w4a4_10ep_ce1_kd4gpu_20261106.sh
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-ofq_swin_t_w4a4_10ep_ce1_kd_t275_20261106}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/swin_t-704ceda3.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-0,1,2,4}"
MASTER_PORT="${MASTER_PORT:-30692}"
CE_WEIGHT="${CE_WEIGHT:-1.0}"
MAX_UPDATES="${MAX_UPDATES:-25030}"  # 10 epochs x 2503 optimizer updates/epoch
SECONDS=0

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== Swin-T W4A4 ground-truth audit: CE + KD, 4x3090, $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "git_head=$(git -C "${QATS}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty_files=$(git -C "${QATS}" status --porcelain 2>/dev/null | wc -l)"
  echo "DATA=${DATA} (folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}  batch=32/GPU  effective_batch=512 images/optimizer step"
  echo "loss=T^2*KD(T=2.75) + ${CE_WEIGHT}*CE(ground_truth)"
  echo "max_train_updates=${MAX_UPDATES} (=> 10 epochs)"
  echo "PY=${PY}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo "start_epoch=$(date +%s)"
  echo
} | tee "${LOG}"

set +e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
"${PY}" "${QATS}/qat_launch.py" \
  --method ofq --stage train \
  --config "${QATS}/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data "${DATA}" --dataset-format folder \
  --output "${OUT}" --experiment "${EXP}" \
  --devices "${DEVICES}" --nproc-per-node 4 --master-port "${MASTER_PORT}" \
  --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-pretrained \
  --teacher-checkpoint "${TEACHER}" \
  --epochs 100 --scheduler-epochs 100 \
  --batch-size 32 --workers 8 --lr 2e-4 --min-lr 5e-6 --weight-decay 0.0 \
  --grad-accum-steps 16 \
  --epoch-checkpoint-interval 1 --checkpoint-hist 2 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
  --clean-start-target-loss-weight "${CE_WEIGHT}" \
  --logit-rank-weight 0 --logit-rank-topk 5 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1 \
  --extra-arg=--color-jitter --extra-arg=0.4 \
  --extra-arg=--reprob --extra-arg=0.25 \
  --extra-arg=--log-interval --extra-arg=50 \
  --extra-arg=--seed --extra-arg=42 \
  --extra-arg=--max_train_updates --extra-arg="${MAX_UPDATES}" \
  >> "${LOG}" 2>&1
status=$?
set -e

{
  echo
  echo "exit_status=${status}"
  echo "end_epoch=$(date +%s)"
  echo "wall_seconds=${SECONDS}"
  echo "train_log=${LOG}"
  echo "output=${OUT}/${EXP}"
} | tee -a "${LOG}"
