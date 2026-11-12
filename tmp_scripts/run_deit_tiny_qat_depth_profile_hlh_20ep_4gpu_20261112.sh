#!/usr/bin/env bash
# Layer-wise precision-profile QAT gate for DeiT-Tiny.
#
# Default profile: H-L-H with equal average block precision to uniform W3A3.
#   blocks 0-2:   W4A4
#   blocks 3-8:   W2A2
#   blocks 9-11:  W4A4
# Mean block precision = (6*4 + 6*2) / 12 = 3 bit.
#
# Patch embedding and classifier heads remain W8A8 through the OFQ wrapper.
# All other training settings match the current official-objective W2A2 gate:
# CE(cls, GT) + KL(dist, FP teacher), QKR, 20 epochs, global batch 512.
#
# Uniform-3 local control:
#   DEPTH_BIT_PROFILE=3,3,3,3,3,3,3,3,3,3,3,3 \
#   EXP=deit_tiny_uniform_w3a3_20ep_4gpu_20261112 \
#   bash tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh
#
# Ternary + 3-bit H-L-H, effective mean about 2.057 bit/block:
#   DEPTH_BIT_PROFILE=3,3,T,T,T,T,T,T,T,T,3,3 \
#   EXP=deit_tiny_qat_hlh_w3a3_mid_ternary_20ep_4gpu_20261112 \
#   bash tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-deit_tiny_qat_hlh_w4a4_mid_w2a2_avg3_20ep_4gpu_20261112}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-1,4,5,7}"
NPROC="${NPROC:-$(echo "${DEVICES}" | tr ',' '\n' | grep -c .)}"
MASTER_PORT="${MASTER_PORT:-30713}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EPOCHS="${EPOCHS:-20}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-20}"
MAX_UPDATES="${MAX_UPDATES:-0}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-2}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
DEPTH_BIT_PROFILE="${DEPTH_BIT_PROFILE:-4,4,4,2,2,2,2,2,2,4,4,4}"
DRY_RUN="${DRY_RUN:-0}"
SECONDS=0

IFS=',' read -r -a PROFILE_VALUES <<< "${DEPTH_BIT_PROFILE}"
if [[ "${#PROFILE_VALUES[@]}" -ne 12 ]]; then
  echo "DEPTH_BIT_PROFILE must contain 12 block entries, got ${#PROFILE_VALUES[@]}: ${DEPTH_BIT_PROFILE}" >&2
  exit 2
fi
PROFILE_EFFECTIVE_MILLIBITS=0
for value in "${PROFILE_VALUES[@]}"; do
  value="${value//[[:space:]]/}"
  normalized="$(echo "${value}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${normalized}" =~ ^(t|ternary|1\.58)$ ]]; then
    PROFILE_EFFECTIVE_MILLIBITS=$((PROFILE_EFFECTIVE_MILLIBITS + 1585))
  elif [[ "${value}" =~ ^[0-9]+$ ]]; then
    PROFILE_EFFECTIVE_MILLIBITS=$((PROFILE_EFFECTIVE_MILLIBITS + value * 1000))
  else
    echo "Unsupported DEPTH_BIT_PROFILE entry '${value}'; use integer or T/ternary" >&2
    exit 2
  fi
done
PROFILE_EFFECTIVE_AVG_X1000=$((PROFILE_EFFECTIVE_MILLIBITS / ${#PROFILE_VALUES[@]}))

EXTRA_ARGS=(
  --seed 42
  --log-interval 50
)
DRY_RUN_ARGS=()
if [[ "${MAX_UPDATES}" -gt 0 ]]; then
  EXTRA_ARGS+=(--max_train_updates "${MAX_UPDATES}")
fi
if [[ "${SKIP_VALIDATE}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_validate)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_RUN_ARGS+=(--dry-run)
fi

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== DeiT-Tiny QAT depth precision profile, $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "git_head=$(git -C "${QATS}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty_files=$(git -C "${QATS}" status --porcelain 2>/dev/null | wc -l)"
  echo "DATA=${DATA} (folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}  nproc=${NPROC}  batch=${BATCH_SIZE}/GPU  requested_accum=${GRAD_ACCUM_STEPS}"
  echo "expected_effective_batch=$((BATCH_SIZE * NPROC * (GRAD_ACCUM_STEPS / NPROC)))"
  echo "epochs=${EPOCHS}  scheduler_epochs=${SCHEDULER_EPOCHS}"
  echo "objective=CE(cls,GT) + KL(dist,teacher), equal weight, KD temperature=1.0"
  echo "augmentation=official noaug: MixUp=0, CutMix=0"
  echo "depth_profile=${DEPTH_BIT_PROFILE}"
  echo "depth_profile_effective_average_x1000=${PROFILE_EFFECTIVE_AVG_X1000}"
  echo "quant=W/A depth profile on 12 blocks + QK reparam; patch/head W8A8"
  echo "dry_run=${DRY_RUN}"
  echo "PY=${PY}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo "start_epoch=$(date +%s)"
  echo
} | tee "${LOG}"

set +e
TORCH_HOME="${TORCH_HOME:-/datadisk2/quyanyi/cache/torch}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
"${PY}" "${QATS}/qat_launch.py" \
  "${DRY_RUN_ARGS[@]}" \
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
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
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
