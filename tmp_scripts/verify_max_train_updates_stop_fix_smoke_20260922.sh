#!/usr/bin/env bash
# Smoke test for the max_train_updates / LR-counter fix (2026-09-22).
#
# Before the fix, train_one_epoch_ofq reset local_update_count every epoch and compared it
# against --max_train_updates, so a cap of 5004 (= 2 epochs of 2503 optimizer updates) could
# never trigger and the run kept going to --epochs. Here we pass a tiny cap (3) with
# --epochs 100: a fixed build stops after 3 optimizer updates, a broken build keeps training.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-fixsmoke_maxupdates3_20260922}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/swin_t-704ceda3.pth}"
PY="${PY:-/tmp/qat_env/bin/python}"
DEVICES="${DEVICES:-5}"
MASTER_PORT="${MASTER_PORT:-30693}"
MAX_UPDATES="${MAX_UPDATES:-3}"
SECONDS=0

rm -rf "${OUT}/${EXP}"
mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== max_train_updates stop fix smoke $(date '+%F %T') ====="
  echo "EXP=${EXP} MAX_UPDATES=${MAX_UPDATES} DEVICES=${DEVICES}"
  echo "PY=${PY}"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
  echo
} | tee "${LOG}"

set +e
PYTHONUNBUFFERED=1 "${PY}" "${QATS}/qat_launch.py" \
  --method ofq --stage train \
  --config "${QATS}/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data "${DATA}" --dataset-format folder \
  --output "${OUT}" --experiment "${EXP}" \
  --devices "${DEVICES}" --nproc-per-node 1 --master-port "${MASTER_PORT}" \
  --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-pretrained \
  --teacher-checkpoint "${TEACHER}" \
  --epochs 100 --scheduler-epochs 100 \
  --batch-size 32 --workers 8 --lr 2e-4 --min-lr 5e-6 --weight-decay 0.0 \
  --grad-accum-steps 4 \
  --epoch-checkpoint-interval 1 --checkpoint-hist 2 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--skip_validate \
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=none \
  --extra-arg=--color-jitter --extra-arg=0.0 \
  --extra-arg=--reprob --extra-arg=0.0 \
  --extra-arg=--log-interval --extra-arg=1 \
  --extra-arg=--seed --extra-arg=42 \
  --extra-arg=--max_train_updates --extra-arg="${MAX_UPDATES}" \
  >> "${LOG}" 2>&1
status=$?
set -e

{
  echo
  echo "exit_status=${status}"
  echo "wall_seconds=${SECONDS}"
  echo "stopped_early_lines=$(grep -c 'Stopped early' "${LOG}" || true)"
  grep 'Stopped early' "${LOG}" || true
} | tee -a "${LOG}"
