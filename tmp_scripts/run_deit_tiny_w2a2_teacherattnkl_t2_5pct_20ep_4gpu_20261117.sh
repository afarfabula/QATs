#!/usr/bin/env bash
# DeiT-Tiny W2A2 with corrected OFQ CE+logit-KL objective and a calibrated
# FP-teacher post-softmax attention KL auxiliary loss.
#
# Objective:
#   CE(student cls head, ground truth)
#   + KL(student dist head, FP teacher), temperature=1.0
#   + lambda * KL(A_student^T || A_teacher^T), attention temperature=2.0
#
# The selected layers match the previous DeiT teacher-attention-KL gate:
# blocks 8-11, all three heads. Patch/head remain W8A8; main blocks are W2A2.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-deit_tiny_w2a2_teacherattnkl_t2_5pct_20ep_4gpu_20261117}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-1,4,5,7}"
NPROC="${NPROC:-$(echo "${DEVICES}" | tr ',' '\n' | grep -c .)}"
MASTER_PORT="${MASTER_PORT:-30723}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EPOCHS="${EPOCHS:-20}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-20}"
MAX_UPDATES="${MAX_UPDATES:-0}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-2}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
ATTN_KL_TEMPERATURE="${ATTN_KL_TEMPERATURE:-2.0}"
ATTN_KL_TARGET_SHARE="${ATTN_KL_TARGET_SHARE:-0.05}"
TEACHER_ATTN_KL_WEIGHT="${TEACHER_ATTN_KL_WEIGHT:-0.001064}"
ATTN_KL_LAYERS="${ATTN_KL_LAYERS:-custom_subset:8:0,8:1,8:2,9:0,9:1,9:2,10:0,10:1,10:2,11:0,11:1,11:2}"
SECONDS=0

if [[ -z "${TEACHER_ATTN_KL_WEIGHT}" ]]; then
  echo "TEACHER_ATTN_KL_WEIGHT is required" >&2
  exit 2
fi
if [[ "${NPROC}" -ne 4 ]]; then
  echo "This experiment is fixed to four ranks; got DEVICES=${DEVICES}" >&2
  exit 2
fi

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
  echo "===== DeiT-Tiny W2A2 CE+KL + FP-teacher attention KL, 4x RTX3090, $(date '+%F %T') ====="
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
  echo "base_objective=CE(cls,GT) + KL(dist,teacher), equal weight, logit KD temperature=1.0"
  echo "teacher_attention_kl_weight=${TEACHER_ATTN_KL_WEIGHT}"
  echo "teacher_attention_kl_temperature=${ATTN_KL_TEMPERATURE}"
  echo "teacher_attention_kl_target_total_loss_share=${ATTN_KL_TARGET_SHARE}"
  echo "teacher_attention_kl_layers=${ATTN_KL_LAYERS}"
  echo "attention_target=post_quant student attention vs frozen FP teacher attention"
  echo "augmentation=official W2A2: MixUp=0, CutMix=0"
  echo "quant=W2A2 main blocks + QK reparam; patch/head W8A8"
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
  --devices "${DEVICES}" --nproc-per-node "${NPROC}" --master-port "${MASTER_PORT}" \
  --epochs "${EPOCHS}" --scheduler-epochs "${SCHEDULER_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" --workers 8 \
  --lr 5e-4 --min-lr 1e-5 --weight-decay 5e-2 --warmup-epochs 5 \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epoch-checkpoint-interval 1 --checkpoint-hist "${CHECKPOINT_HIST}" \
  --num-aug-repeats 0 \
  --wbits 2 --abits 2 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 1 --ofq-official-kd-heads \
  --teacher-soft-temperature 1.0 \
  --logit-rank-weight 0 --attn-rank-weight 0 \
  --attn-rank-target post_quant \
  --train-scheme ema_ref_attn_kl \
  --ref-update prev_step --ref-update-interval 50 \
  --ref-attn-kl-weight 0 --ref-logit-kl-weight 0 \
  --ref-head-mode "${ATTN_KL_LAYERS}" \
  --teacher-attn-kl-weight "${TEACHER_ATTN_KL_WEIGHT}" \
  --teacher-attn-kl-temperature "${ATTN_KL_TEMPERATURE}" \
  --teacher-attn-kl-warmup-epochs 0 \
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
