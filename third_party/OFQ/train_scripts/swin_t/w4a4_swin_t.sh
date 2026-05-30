#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

DATA_PATH="${DATA_PATH:-/tmp/imagenet1k_full_parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mlx_devbox/users/quyanyi/playground/OFQ/out}"
EXPERIMENT="${EXPERIMENT:-swin_t_w4a4_qkr_30ep}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${EXPERIMENT}.log}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-16}"
VISIBLE_GPU="${VISIBLE_GPU:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
TCP_PORT="${TCP_PORT:-12346}"

if (( BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
  echo "BATCH_SIZE (${BATCH_SIZE}) must be divisible by MICRO_BATCH_SIZE (${MICRO_BATCH_SIZE})" >&2
  exit 1
fi

GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$((BATCH_SIZE / MICRO_BATCH_SIZE))}"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

CMD=(
  python3 -u train.py
  -c ./configs/swin_t_imagenet.attn_q.yml
  --model swin_t
  "${DATA_PATH}"
  --dataset hf-parquet-imagenet
  --epochs "${EPOCHS}"
  --batch-size "${MICRO_BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --workers "${WORKERS}"
  --weight-decay 0.0
  --warmup-lr 1.0e-6
  --lr 2.0e-4
  --warmup-epochs 0
  --aq-enable
  --aq-mode lsq
  --aq-per-channel
  --aq_clip_learnable
  --aq-bitw 4
  --wq-enable
  --wq-per-channel
  --wq-bitw 4
  --wq-mode statsq
  --model_type swin
  --teacher_type swin
  --quantized
  --pretrained
  --pretrained_initialized
  --use-kd
  --teacher swin_t
  --kd_hard_and_soft 1
  --qk_reparam
  --qk_reparam_type 0
  --teacher_pretrained
  --output "${OUTPUT_ROOT}"
  --experiment "${EXPERIMENT}"
  --visible_gpu "${VISIBLE_GPU}"
  --world_size "${WORLD_SIZE}"
  --tcp_port "${TCP_PORT}"
)

printf '===== launch at %s =====\n' "$(date '+%F %T')" >> "${LOG_FILE}"
printf 'repo=%s\n' "${REPO_DIR}" >> "${LOG_FILE}"
printf 'data=%s\n' "${DATA_PATH}" >> "${LOG_FILE}"
printf 'output=%s\n' "${OUTPUT_ROOT}" >> "${LOG_FILE}"
printf 'experiment=%s\n' "${EXPERIMENT}" >> "${LOG_FILE}"
printf 'effective_batch_size=%s\n' "${BATCH_SIZE}" >> "${LOG_FILE}"
printf 'micro_batch_size=%s\n' "${MICRO_BATCH_SIZE}" >> "${LOG_FILE}"
printf 'grad_accum_steps=%s\n' "${GRAD_ACCUM_STEPS}" >> "${LOG_FILE}"

PYTHONUNBUFFERED=1 "${CMD[@]}" >> "${LOG_FILE}" 2>&1
