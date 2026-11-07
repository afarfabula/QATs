#!/usr/bin/env bash
# DeiT-Tiny W4A4-family QAT baseline on the two RTX 3090s currently allocated
# to this project (GPU 4 and 7).
#
# Recipe:
#   pure teacher soft KD, no CE or auxiliary ranking losses
#   W4A4 main blocks, StatsQ weights, LSQ activations, QK reparameterization
#   patch embedding and classifier heads W8A8, matching the existing Swin path
#   64 images/GPU x 2 ranks x 4 accumulation steps = 512 images/update
#   100 epochs on the standard DeiT augmentation recipe
#
# Usage:
#   DEVICES=4,7 bash tmp_scripts/run_deit_tiny_w4a4_baseline_100ep_2gpu_20261107.sh
#
# Short smoke:
#   MAX_UPDATES=20 SKIP_VALIDATE=1 CHECKPOINT_HIST=0 \
#     EXP=deit_tiny_w4a4_baseline_smoke_20261107 \
#     bash tmp_scripts/run_deit_tiny_w4a4_baseline_100ep_2gpu_20261107.sh
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-deit_tiny_w4a4_baseline_100ep_2gpu_20261107}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-4,7}"
MASTER_PORT="${MASTER_PORT:-30693}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"  # divided by 2 ranks -> 4/GPU
EPOCHS="${EPOCHS:-100}"
MAX_UPDATES="${MAX_UPDATES:-0}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-2}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
SECONDS=0

EXTRA_ARGS=(
  --seed 42
  --log-interval 50
)
if [[ "${MAX_UPDATES}" -gt 0 ]]; then
  EXTRA_ARGS+=(--max_train_updates "${MAX_UPDATES}")
fi
if [[ "${SKIP_VALIDATE}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_validate)
fi

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== DeiT-Tiny W4A4 QAT baseline, 2x RTX3090, $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "git_head=$(git -C "${QATS}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty_files=$(git -C "${QATS}" status --porcelain 2>/dev/null | wc -l)"
  echo "DATA=${DATA} (folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}  batch=${BATCH_SIZE}/GPU  requested_accum=${GRAD_ACCUM_STEPS}  ranks=2"
  echo "expected_effective_batch=$((BATCH_SIZE * 2 * (GRAD_ACCUM_STEPS / 2)))"
  echo "epochs=${EPOCHS}  max_train_updates=${MAX_UPDATES:-full}  checkpoint_hist=${CHECKPOINT_HIST}"
  echo "objective=T^2*KD(T=2.75), pure soft KD; no CE or auxiliary ranking loss"
  echo "quant=W4A4 main blocks + QK reparam; patch/head W8A8"
  echo "PY=${PY}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo "start_epoch=$(date +%s)"
  echo
} | tee "${LOG}"

set +e
TORCH_HOME="${TORCH_HOME:-/datadisk2/quyanyi/cache/torch}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
"${PY}" "${QATS}/qat_launch.py" \
  --method ofq --stage train \
  --config "${QATS}/third_party/OFQ/configs/deit_tiny_qat_baseline_100ep.yml" \
  --model deit_tiny_distilled_patch16_224 --model-type deit \
  --teacher deit_tiny_distilled_patch16_224 --teacher-type deit \
  --teacher-checkpoint "${TEACHER}" --teacher-pretrained \
  --data "${DATA}" --dataset-format folder \
  --output "${OUT}" --experiment "${EXP}" \
  --devices "${DEVICES}" --nproc-per-node 2 --master-port "${MASTER_PORT}" \
  --epochs "${EPOCHS}" --scheduler-epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" --workers 8 \
  --lr 5e-4 --min-lr 1e-5 --weight-decay 5e-2 --warmup-epochs 5 \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epoch-checkpoint-interval 1 --checkpoint-hist "${CHECKPOINT_HIST}" \
  --num-aug-repeats 0 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
  --logit-rank-weight 0 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.8 \
  --extra-arg=--cutmix --extra-arg=1.0 \
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1 \
  --extra-arg=--color-jitter --extra-arg=0.4 \
  --extra-arg=--reprob --extra-arg=0.25 \
  "${EXTRA_ARGS[@]/#/--extra-arg=}" \
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

exit "${status}"
