#!/usr/bin/env bash
# DeiT-Tiny W2A2 QAT + fixed-FP-teacher attention constraint.
#
# Paired against:
#   deit_tiny_w2a2_baseline_20ep_2gpu_20261109
#
# Only objective changes relative to the baseline:
#   + attention-relation ranking from the frozen FP teacher
#   + lambda=0.3, top-k=5, min_attn=1e-4
#   + last four blocks (8-11), all 3 heads in each block
#
# This is the best-performing teacher-rank point in the repository's short
# reference-ablation experiment. It uses the original softplus form.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-deit_tiny_w2a2_teacherrank_w03_l8to11_20ep_2gpu_20261110}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-4,7}"
MASTER_PORT="${MASTER_PORT:-30695}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"  # divided by 2 ranks -> 4/GPU
EPOCHS="${EPOCHS:-20}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-${EPOCHS}}"
RANK_W="${RANK_W:-0.3}"
RANK_TOPK="${RANK_TOPK:-5}"
RANK_MIN_ATTN="${RANK_MIN_ATTN:-1e-4}"
TEACHER_KL_W="${TEACHER_KL_W:-0}"
HEADS="${HEADS:-custom_subset:8:0,8:1,8:2,9:0,9:1,9:2,10:0,10:1,10:2,11:0,11:1,11:2}"
MAX_UPDATES="${MAX_UPDATES:-0}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-2}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
SECONDS=0

EXTRA_ARGS=(
  --seed 42
  --log-interval 50
)
TEACHER_KL_ARGS=()
if [[ "${TEACHER_KL_W}" != "0" && "${TEACHER_KL_W}" != "0.0" ]]; then
  TEACHER_KL_ARGS+=(
    --train-scheme ema_ref_attn_kl
    --ref-update prev_step
    --ref-update-interval 50
    --ref-attn-kl-weight 0
    --ref-logit-kl-weight 0
    --ref-attn-kl-clip 0
    --ref-attn-kl-drop-prob 1.0
    --ref-attn-loss kl_ref
    --teacher-attn-kl-weight "${TEACHER_KL_W}"
    --teacher-attn-kl-warmup-epochs 0
  )
fi
if [[ "${MAX_UPDATES}" -gt 0 ]]; then
  EXTRA_ARGS+=(--max_train_updates "${MAX_UPDATES}")
fi
if [[ "${SKIP_VALIDATE}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_validate)
fi

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== DeiT-Tiny W2A2 + FP-teacher attention constraint, 2x RTX3090, $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "git_head=$(git -C "${QATS}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty_files=$(git -C "${QATS}" status --porcelain 2>/dev/null | wc -l)"
  echo "DATA=${DATA} (folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}  batch=${BATCH_SIZE}/GPU  requested_accum=${GRAD_ACCUM_STEPS}  ranks=2"
  echo "expected_effective_batch=$((BATCH_SIZE * 2 * (GRAD_ACCUM_STEPS / 2)))"
  echo "epochs=${EPOCHS}  scheduler_epochs=${SCHEDULER_EPOCHS}  max_train_updates=${MAX_UPDATES:-full}  checkpoint_hist=${CHECKPOINT_HIST}"
  echo "objective=T^2*KD(T=2.75) + ${RANK_W}*teacher-attention-rank + ${TEACHER_KL_W}*teacher-attention-KL"
  echo "rank=source=teacher source=post_quant topk=${RANK_TOPK} min_attn=${RANK_MIN_ATTN} loss=softplus heads=blocks8-11/all"
  echo "quant=W2A2 main blocks + QK reparam; patch/head W8A8"
  echo "stage=direct train only; no CGA"
  echo "paired_baseline=/datadisk2/quyanyi/qat_runs/deit_tiny_w2a2_baseline_20ep_2gpu_20261109.log"
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
  --epochs "${EPOCHS}" --scheduler-epochs "${SCHEDULER_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" --workers 8 \
  --lr 5e-4 --min-lr 1e-5 --weight-decay 5e-2 --warmup-epochs 5 \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epoch-checkpoint-interval 1 --checkpoint-hist "${CHECKPOINT_HIST}" \
  --num-aug-repeats 0 \
  --wbits 2 --abits 2 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
  --logit-rank-weight 0 \
  --attn-rank-weight "${RANK_W}" --attn-rank-source teacher \
  --attn-rank-topk "${RANK_TOPK}" --attn-rank-min-attn "${RANK_MIN_ATTN}" \
  --attn-rank-target post_quant --ref-head-mode "${HEADS}" \
  "${TEACHER_KL_ARGS[@]}" \
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
