#!/usr/bin/env bash
# W2A2 + CE+logit-KL + teacher attention KL on pre-quant student attention.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-deit_tiny_w2a2_preqattnkl_t2_19ep_4gpu_20261118}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-1,4,5,7}"
NPROC="${NPROC:-$(echo "${DEVICES}" | tr ',' '\n' | grep -c .)}"
MASTER_PORT="${MASTER_PORT:-30751}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EPOCHS="${EPOCHS:-19}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-19}"
MAX_UPDATES="${MAX_UPDATES:-0}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-2}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
TEACHER_ATTN_KL_WEIGHT="${TEACHER_ATTN_KL_WEIGHT:-}"
ATTN_KL_TEMPERATURE="${ATTN_KL_TEMPERATURE:-2.0}"
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

EXTRA_ARGS=(--seed 42 --log-interval 50)
if [[ "${MAX_UPDATES}" -gt 0 ]]; then
  EXTRA_ARGS+=(--max_train_updates "${MAX_UPDATES}")
fi
if [[ "${SKIP_VALIDATE}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_validate)
fi

mkdir -p "${OUT}"
rm -f "${LOG}"
{
  echo "===== DeiT-Tiny W2A2 pre-quant teacher attention KL, 4x RTX3090, $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "git_head=$(git -C "${QATS}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "DATA=${DATA} (folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "DEVICES=${DEVICES}  nproc=${NPROC}  batch=${BATCH_SIZE}/GPU  requested_accum=${GRAD_ACCUM_STEPS}"
  echo "epochs=${EPOCHS}  scheduler_epochs=${SCHEDULER_EPOCHS}"
  echo "objective=CE(cls,GT) + KL(dist,teacher) + teacher-attention-KL"
  echo "student_attention=pre_quant (before post-softmax A2 quantizer)"
  echo "teacher_attention=fp"
  echo "teacher_attention_kl_weight=${TEACHER_ATTN_KL_WEIGHT}"
  echo "teacher_attention_kl_temperature=${ATTN_KL_TEMPERATURE}"
  echo "teacher_attention_kl_layers=${ATTN_KL_LAYERS}"
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
  --attn-rank-target pre_quant \
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
  echo "exit_status=${status}"
  echo "end_epoch=$(date +%s)"
  echo "wall_seconds=${SECONDS}"
  echo "train_log=${LOG}"
  echo "output=${OUT}/${EXP}"
} | tee -a "${LOG}"
exit "${status}"
