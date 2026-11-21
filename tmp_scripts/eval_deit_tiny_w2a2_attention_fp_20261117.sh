#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
CKPT="${CKPT:?CKPT is required}"
EXP="${EXP:?EXP is required}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-0,2}"
NPROC="${NPROC:-$(echo "${DEVICES}" | tr ',' '\n' | grep -c .)}"
MASTER_PORT="${MASTER_PORT:-30731}"
ATTENTION_SOFTMAX_FP="${ATTENTION_SOFTMAX_FP:-0}"
ATTENTION_SOFTMAX_POST_QUANT_RENORM="${ATTENTION_SOFTMAX_POST_QUANT_RENORM:-0}"
SECONDS=0

EXTRA_ARGS=(--seed 42 --log-interval 50)
if [[ "${ATTENTION_SOFTMAX_FP}" == "1" ]]; then
  EXTRA_ARGS+=(--attention-softmax-fp)
fi
if [[ "${ATTENTION_SOFTMAX_POST_QUANT_RENORM}" == "1" ]]; then
  EXTRA_ARGS+=(--attention-softmax-post-quant-renorm)
fi

mkdir -p "${OUT}"
rm -f "${LOG}"
{
  echo "===== DeiT-Tiny W2A2 full-val eval, $(date '+%F %T') ====="
  echo "checkpoint=${CKPT}"
  echo "attention_softmax_fp=${ATTENTION_SOFTMAX_FP}"
  echo "attention_softmax_post_quant_renorm=${ATTENTION_SOFTMAX_POST_QUANT_RENORM}"
  echo "devices=${DEVICES}"
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
  --epochs 20 --scheduler-epochs 20 \
  --batch-size 64 --workers 8 \
  --lr 5e-4 --min-lr 1e-5 --weight-decay 5e-2 --warmup-epochs 5 \
  --grad-accum-steps 8 --num-aug-repeats 0 \
  --wbits 2 --abits 2 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 1 --ofq-official-kd-heads \
  --teacher-soft-temperature 1.0 \
  --logit-rank-weight 0 --attn-rank-weight 0 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--eval-only \
  --extra-arg=--initial-checkpoint --extra-arg="${CKPT}" \
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
  echo "wall_seconds=${SECONDS}"
  echo "eval_log=${LOG}"
} | tee -a "${LOG}"
exit "${status}"
