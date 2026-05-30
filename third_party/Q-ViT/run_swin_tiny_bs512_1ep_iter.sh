#!/usr/bin/env bash
# Single-GPU Swin-Tiny FP launcher using the iterable parquet loader.
# Defaults match the validated BS=512 setup, but epochs/output dir can be
# overridden via environment variables.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU="${GPU:-0}"
DATA_PATH="${DATA_PATH:-/tmp/imagenet1k_full_parquet}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/out/swin_tiny_bs512_1ep_iter}"
RESUME="${RESUME:-}"

BS="${BS:-512}"
BS_EVAL="${BS_EVAL:-512}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EPOCHS="${EPOCHS:-1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/console.log}"
LOG_MODE="${LOG_MODE:-truncate}"

mkdir -p "${OUT_DIR}"
cd "${REPO_DIR}"

CMD=(
  python3 -u main.py
  --model swin_tiny_patch4_window7_224
  --pretrained
  --data-set IMNET_PARQUET_ITER
  --data-path "${DATA_PATH}"
  --epochs "${EPOCHS}"
  --warmup-epochs "${WARMUP_EPOCHS}"
  --weight-decay 0.05
  --drop-path 0.2
  --batch-size "${BS}"
  --batch-size-eval "${BS_EVAL}"
  --num_workers "${NUM_WORKERS}"
  --output_dir "${OUT_DIR}"
)

if [[ -n "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi

if [[ "${LOG_MODE}" == "append" ]]; then
  printf '\n===== launch at %s | epochs=%s | resume=%s =====\n' "$(date '+%F %T')" "${EPOCHS}" "${RESUME:-<none>}" >> "${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${CMD[@]}" >> "${LOG_FILE}" 2>&1
else
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${CMD[@]}" > "${LOG_FILE}" 2>&1
fi
