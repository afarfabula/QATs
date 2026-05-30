#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
METHOD="${METHOD:-aoq}"
DATA_ROOT="${DATA_ROOT:-/tmp/qats/imagenet1k}"
IMG_ROOT="${IMG_ROOT:-${DATA_ROOT}/imagefolder}"
PARQUET_ROOT="${PARQUET_ROOT:-${DATA_ROOT}/parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/${METHOD}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_REQUIREMENTS="${RUN_REQUIREMENTS:-1}"
RUN_DATASET="${RUN_DATASET:-1}"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${RUN_REQUIREMENTS}" == "1" ]]; then
  bash "${ROOT_DIR}/check_requirements.sh"
fi

if [[ "${RUN_DATASET}" == "1" ]]; then
  bash "${ROOT_DIR}/check_dataset.sh"
fi

case "${METHOD}" in
  qvit)
    DATA_PATH="${QVIT_DATA_PATH:-${PARQUET_ROOT}}"
    exec "${PYTHON_BIN}" "${ROOT_DIR}/qat_launch.py" \
      --method qvit \
      --arch "${QVIT_ARCH:-swin_tiny}" \
      --data "${DATA_PATH}" \
      --dataset-format "${QVIT_DATASET_FORMAT:-parquet}" \
      --output "${OUTPUT_ROOT}" \
      --epochs "${EPOCHS:-300}" \
      --batch-size "${BATCH_SIZE:-64}" \
      --workers "${WORKERS:-16}" \
      --nproc-per-node "${NPROC_PER_NODE:-1}" \
      --master-port "${MASTER_PORT:-29541}" \
      --pretrained \
      ${QVIT_EXTRA_ARGS:-}
    ;;
  ofq)
    DATA_PATH="${OFQ_DATA_PATH:-${PARQUET_ROOT}}"
    exec "${PYTHON_BIN}" "${ROOT_DIR}/qat_launch.py" \
      --method ofq \
      --stage "${OFQ_STAGE:-train}" \
      --model "${OFQ_MODEL:-swin_t}" \
      --bits "${BITS:-4}" \
      --data "${DATA_PATH}" \
      --dataset-format "${OFQ_DATASET_FORMAT:-parquet}" \
      --output "${OUTPUT_ROOT}" \
      --epochs "${EPOCHS:-30}" \
      --batch-size "${MICRO_BATCH_SIZE:-64}" \
      --grad-accum-steps "${GRAD_ACCUM_STEPS:-8}" \
      --workers "${WORKERS:-16}" \
      --devices "${DEVICES:-0}" \
      --master-port "${MASTER_PORT:-12346}" \
      --pretrained \
      --pretrained-initialized \
      --use-kd \
      --teacher-pretrained \
      --quantized \
      --qk-reparam \
      --wq-per-channel \
      --aq-per-channel \
      --aq-clip-learnable \
      ${OFQ_EXTRA_ARGS:-}
    ;;
  aoq)
    DATA_PATH="${AOQ_DATA_PATH:-${PARQUET_ROOT}}"
    CMD=(
      "${PYTHON_BIN}" "${ROOT_DIR}/qat_launch.py"
      --method aoq
      --task "${AOQ_TASK:-imagenet}"
      --model "${AOQ_MODEL:-resnet18}"
      --teacher "${AOQ_TEACHER:-resnet101}"
      --bits "${BITS:-2}"
      --data "${DATA_PATH}"
      --output "${OUTPUT_ROOT}"
      --aoq-dataset-format "${AOQ_DATASET_FORMAT:-parquet-iter}"
      --epochs "${EPOCHS:-300}"
      --batch-size "${BATCH_SIZE:-256}"
      --workers "${WORKERS:-16}"
      --lr "${LR:-0.00125}"
      --weight-decay "${WEIGHT_DECAY:-0}"
      --devices "${DEVICES:-0}"
      --quantize-downsample "${QUANTIZE_DOWNSAMPLE:-true}"
      --amp-dtype "${AMP_DTYPE:-bf16}"
      --compile-mode "${COMPILE_MODE:-max-autotune}"
      --prefetch-factor "${PREFETCH_FACTOR:-8}"
      --plot-interval "${PLOT_INTERVAL:-0}"
      --val-interval "${VAL_INTERVAL:-1}"
      --train-steps-per-epoch "${TRAIN_STEPS_PER_EPOCH:-0}"
      --val-steps "${VAL_STEPS:-0}"
    )

    if [[ "${ENABLE_AMP:-1}" == "1" ]]; then
      CMD+=(--amp)
    fi
    if [[ "${ENABLE_CHANNELS_LAST:-1}" == "1" ]]; then
      CMD+=(--channels-last)
    fi
    if [[ "${ENABLE_COMPILE:-1}" == "1" ]]; then
      CMD+=(--compile)
    fi
    if [[ "${ENABLE_PERSISTENT_WORKERS:-1}" == "1" ]]; then
      CMD+=(--persistent-workers)
    fi
    if [[ "${SKIP_TEACHER_VAL:-1}" == "1" ]]; then
      CMD+=(--skip-teacher-val)
    fi
    if [[ "${ENABLE_SYNTHETIC_DATA:-0}" == "1" ]]; then
      CMD+=(
        --synthetic-data
        --synthetic-train-size "${SYNTHETIC_TRAIN_SIZE:-32768}"
        --synthetic-val-size "${SYNTHETIC_VAL_SIZE:-4096}"
      )
    fi

    exec "${CMD[@]}"
    ;;
  *)
    echo "Unsupported METHOD=${METHOD}" >&2
    exit 1
    ;;
esac
