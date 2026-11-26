#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-deit_tiny_qat_sub24_v2shape_20ep_4gpu_20261126}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-0,1,2,3}"
NPROC="${NPROC:-$(echo "${DEVICES}" | tr ',' '\n' | grep -c .)}"
MASTER_PORT="${MASTER_PORT:-30985}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EPOCHS="${EPOCHS:-20}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-20}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-3}"
DEPTH_BIT_PROFILE="${DEPTH_BIT_PROFILE:-T,3,2,3,T,T,3,2,3,T,2,T,3,T,3,T,T,T,T,T,2,T,3,T}"

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== DeiT-Tiny static 24-submodule retrain of v2-discovered profile, $(date '+%F %T') ====="
  echo "EXP=${EXP}"
  echo "DEVICES=${DEVICES} NPROC=${NPROC} MASTER_PORT=${MASTER_PORT}"
  echo "DEPTH_BIT_PROFILE=${DEPTH_BIT_PROFILE}"
} | tee "${LOG}"

"${PY}" "${QATS}/qat_launch.py" \
  --method ofq --stage train \
  --config "${QATS}/third_party/OFQ/configs/deit_tiny_qat_baseline_100ep.yml" \
  --model deit_tiny_distilled_patch16_224 --model-type deit \
  --teacher deit_tiny_distilled_patch16_224 --teacher-type deit \
  --teacher-checkpoint "${TEACHER}" --teacher-pretrained \
  --data "${DATA}" --dataset-format folder \
  --output "${OUT}" --experiment "${EXP}" \
  --devices "${DEVICES}" --nproc-per-node "${NPROC}" --master-port "${MASTER_PORT}" \
  --epochs "${EPOCHS}" --scheduler-epochs "${SCHEDULER_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" --workers 8 \
  --lr 5e-4 --min-lr 1e-5 --weight-decay 5e-2 --warmup-epochs 5 \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epoch-checkpoint-interval 1 --checkpoint-hist "${CHECKPOINT_HIST}" \
  --num-aug-repeats 0 \
  --wbits 2 --abits 2 --wq-mode statsq --aq-mode lsq \
  --depth-bit-profile "${DEPTH_BIT_PROFILE}" \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 1 --ofq-official-kd-heads \
  --teacher-soft-temperature 1.0 \
  --logit-rank-weight 0 --attn-rank-weight 0 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --model-ema --model-ema-decay 0.999 \
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1 \
  --extra-arg=--color-jitter --extra-arg=0.4 \
  --extra-arg=--reprob --extra-arg=0.25 \
  --extra-arg=--seed --extra-arg=42 \
  --extra-arg=--log-interval --extra-arg=50 \
  >> "${LOG}" 2>&1
