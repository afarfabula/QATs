#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_lsqaoq090_late5571_delayedbinreg_gate_20260708}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
TEACHER="${TEACHER:-/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth}"
RESUME="${RESUME:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30863}"
START_EPOCH="${START_EPOCH:-3}"
EPOCHS="${EPOCHS:-4}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-1e-5}"
QUANT_LR_MULTIPLIER="${QUANT_LR_MULTIPLIER:-4}"
AOQ_EXPLORE_SCALE_RATIO="${AOQ_EXPLORE_SCALE_RATIO:-0.90}"
AOQ_EXPLORE_LAYERS="${AOQ_EXPLORE_LAYERS:-features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2}"
AOQ_EXPLORE_START_UPDATE="${AOQ_EXPLORE_START_UPDATE:-0}"
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-1800}"
BIN_REG_WEIGHT="${BIN_REG_WEIGHT:-1e-5}"
BIN_REG_VARIANCE_WEIGHT="${BIN_REG_VARIANCE_WEIGHT:-1.0}"
BIN_REG_LAYERS="${BIN_REG_LAYERS:-features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2}"
BIN_REG_START_UPDATE="${BIN_REG_START_UPDATE:-1800}"
BIN_REG_END_UPDATE="${BIN_REG_END_UPDATE:-0}"
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-0}"
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
  --resume "${RESUME}" --no-resume-opt --start-epoch "${START_EPOCH}"
  --epochs "${EPOCHS}" --scheduler-epochs "${SCHEDULER_EPOCHS}" --batch-size "${BATCH_SIZE}" --workers "${WORKERS}"
  --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay 0.0
  --quant-lr-multiplier "${QUANT_LR_MULTIPLIER}"
  --epoch-checkpoint-interval 1 --checkpoint-hist 5
  --wbits 4 --abits 4 --wq-mode lsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized
  --amp --amp-dtype bf16
  --aoq-explore-scale-ratio "${AOQ_EXPLORE_SCALE_RATIO}"
  --aoq-explore-layers "${AOQ_EXPLORE_LAYERS}"
  --aoq-explore-start-update "${AOQ_EXPLORE_START_UPDATE}"
  --aoq-explore-end-update "${AOQ_EXPLORE_END_UPDATE}"
  --bin-reg-weight "${BIN_REG_WEIGHT}"
  --bin-reg-variance-weight "${BIN_REG_VARIANCE_WEIGHT}"
  --bin-reg-layers "${BIN_REG_LAYERS}"
  --bin-reg-start-update "${BIN_REG_START_UPDATE}"
  --bin-reg-end-update "${BIN_REG_END_UPDATE}"
  --extra-arg=--static-graph
  --extra-arg=--smoothing --extra-arg=0.0
  --extra-arg=--mixup --extra-arg=0.0
  --extra-arg=--cutmix --extra-arg=0.0
  --extra-arg=--aa --extra-arg=none
  --extra-arg=--color-jitter --extra-arg=0.0
  --extra-arg=--reprob --extra-arg=0.0
  --extra-arg=--log-interval --extra-arg=50
  --extra-arg=--seed --extra-arg=42
)

if [[ -n "${MAX_TRAIN_UPDATES}" && "${MAX_TRAIN_UPDATES}" != "0" ]]; then
  cmd+=(--extra-arg=--max_train_updates --extra-arg="${MAX_TRAIN_UPDATES}")
  if [[ "${SKIP_VALIDATE}" == "1" ]]; then
    cmd+=(--extra-arg=--skip_validate)
  fi
fi

{
  echo "===== Clean LSQ no-QKR LSQ-AOQ090 late5571 delayed-BinReg gate $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "RESUME=${RESUME}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "START_EPOCH=${START_EPOCH}"
  echo "EPOCHS=${EPOCHS}"
  echo "SCHEDULER_EPOCHS=${SCHEDULER_EPOCHS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "WORKERS=${WORKERS}"
  echo "LR=${LR}"
  echo "MIN_LR=${MIN_LR}"
  echo "QUANT_LR_MULTIPLIER=${QUANT_LR_MULTIPLIER}"
  echo "WQ_MODE=lsq"
  echo "AQ_MODE=lsq"
  echo "QK_REPARAM=0"
  echo "AOQ_EXPLORE_SCALE_RATIO=${AOQ_EXPLORE_SCALE_RATIO}"
  echo "AOQ_EXPLORE_LAYERS=${AOQ_EXPLORE_LAYERS}"
  echo "AOQ_EXPLORE_START_UPDATE=${AOQ_EXPLORE_START_UPDATE}"
  echo "AOQ_EXPLORE_END_UPDATE=${AOQ_EXPLORE_END_UPDATE}"
  echo "BIN_REG_WEIGHT=${BIN_REG_WEIGHT}"
  echo "BIN_REG_VARIANCE_WEIGHT=${BIN_REG_VARIANCE_WEIGHT}"
  echo "BIN_REG_LAYERS=${BIN_REG_LAYERS}"
  echo "BIN_REG_START_UPDATE=${BIN_REG_START_UPDATE}"
  echo "BIN_REG_END_UPDATE=${BIN_REG_END_UPDATE}"
  echo "MAX_TRAIN_UPDATES=${MAX_TRAIN_UPDATES}"
  echo "SKIP_VALIDATE=${SKIP_VALIDATE}"
  ls -lh "${RESUME}"
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
