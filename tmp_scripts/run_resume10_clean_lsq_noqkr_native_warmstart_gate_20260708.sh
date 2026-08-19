#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_native_warmstart_gate_20260708}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
TEACHER="${TEACHER:-/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30795}"
EPOCHS="${EPOCHS:-1}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-1e-5}"
QUANT_LR_MULTIPLIER="${QUANT_LR_MULTIPLIER:-4}"
PRE_QAT_RECON_UPDATES="${PRE_QAT_RECON_UPDATES:-100}"
PRE_QAT_RECON_TEMPERATURE="${PRE_QAT_RECON_TEMPERATURE:-2.75}"
PRE_QAT_FEATURE_RECON_UPDATES="${PRE_QAT_FEATURE_RECON_UPDATES:-100}"
PRE_QAT_FEATURE_RECON_LAYERS="${PRE_QAT_FEATURE_RECON_LAYERS:-features.1.1,features.3.1,features.5.5,features.7.1}"
PRE_QAT_FEATURE_RECON_POLICY="${PRE_QAT_FEATURE_RECON_POLICY:-quant}"
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-1}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
SECONDS=0

rm -rf "${OUT}/${EXP}"
mkdir -p "${OUT}"
rm -f "${LOG}"

cmd=(
  python3 "${QATS}/qat_launch.py"
  --method ofq --stage train
  --config "${QATS}/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml"
  --model swin_t --data "${DATA}" --dataset-format parquet
  --output "${OUT}" --experiment "${EXP}"
  --devices "${DEVICES}" --nproc-per-node 8 --master-port "${MASTER_PORT}" --model-type swin
  --teacher swin_t --teacher-type swin --teacher-checkpoint "${TEACHER}" --teacher-pretrained
  --epochs "${EPOCHS}" --scheduler-epochs "${SCHEDULER_EPOCHS}" --batch-size "${BATCH_SIZE}" --workers 8
  --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay 0.0
  --quant-lr-multiplier "${QUANT_LR_MULTIPLIER}"
  --epoch-checkpoint-interval 1 --checkpoint-hist 5
  --wbits 4 --abits 4 --wq-mode lsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized
  --amp --amp-dtype bf16
  --pre-qat-recon-updates "${PRE_QAT_RECON_UPDATES}"
  --pre-qat-recon-temperature "${PRE_QAT_RECON_TEMPERATURE}"
  --pre-qat-feature-recon-updates "${PRE_QAT_FEATURE_RECON_UPDATES}"
  --pre-qat-feature-recon-layers "${PRE_QAT_FEATURE_RECON_LAYERS}"
  --pre-qat-feature-recon-policy "${PRE_QAT_FEATURE_RECON_POLICY}"
  --extra-arg=--static-graph
  --extra-arg=--smoothing --extra-arg=0.0
  --extra-arg=--mixup --extra-arg=0.0
  --extra-arg=--cutmix --extra-arg=0.0
  --extra-arg=--aa --extra-arg=none
  --extra-arg=--color-jitter --extra-arg=0.0
  --extra-arg=--reprob --extra-arg=0.0
  --extra-arg=--log-interval --extra-arg=20
  --extra-arg=--seed --extra-arg=42
)

if [[ -n "${MAX_TRAIN_UPDATES}" && "${MAX_TRAIN_UPDATES}" != "0" ]]; then
  cmd+=(--extra-arg=--max_train_updates --extra-arg="${MAX_TRAIN_UPDATES}")
  if [[ "${SKIP_VALIDATE}" == "1" ]]; then
    cmd+=(--extra-arg=--skip_validate)
  fi
fi

{
  echo "===== Clean LSQ no-QKR native warm-start gate $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "EPOCHS=${EPOCHS}"
  echo "SCHEDULER_EPOCHS=${SCHEDULER_EPOCHS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "LR=${LR}"
  echo "MIN_LR=${MIN_LR}"
  echo "QUANT_LR_MULTIPLIER=${QUANT_LR_MULTIPLIER}"
  echo "WQ_MODE=lsq"
  echo "AQ_MODE=lsq"
  echo "QK_REPARAM=0"
  echo "PRE_QAT_RECON_UPDATES=${PRE_QAT_RECON_UPDATES}"
  echo "PRE_QAT_RECON_TEMPERATURE=${PRE_QAT_RECON_TEMPERATURE}"
  echo "PRE_QAT_FEATURE_RECON_UPDATES=${PRE_QAT_FEATURE_RECON_UPDATES}"
  echo "PRE_QAT_FEATURE_RECON_LAYERS=${PRE_QAT_FEATURE_RECON_LAYERS}"
  echo "PRE_QAT_FEATURE_RECON_POLICY=${PRE_QAT_FEATURE_RECON_POLICY}"
  echo "MAX_TRAIN_UPDATES=${MAX_TRAIN_UPDATES}"
  echo "SKIP_VALIDATE=${SKIP_VALIDATE}"
  python3 - <<'PY'
from pathlib import Path
root = Path('/tmp/imagenet1k_full_parquet/data')
print('train_shards=' + str(len(list(root.glob('train-*.parquet')))))
print('validation_shards=' + str(len(list(root.glob('validation-*.parquet')))))
PY
  git -C "${QATS}" rev-parse --short HEAD || true
  test -e /dev/nvidia0 && echo gpu-device-present || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  printf 'command='
  printf '%q ' "${cmd[@]}"
  echo
} | tee "${LOG}"

"${cmd[@]}" 2>&1 | tee -a "${LOG}"

echo "wall_seconds=${SECONDS}" | tee -a "${LOG}"
echo "train_log=${LOG}" | tee -a "${LOG}"
echo "output=${OUT}/${EXP}" | tee -a "${LOG}"
