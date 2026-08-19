#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_gate_20260708}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
TEACHER="${TEACHER:-/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth}"
RESUME="${RESUME:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30787}"
START_EPOCH="${START_EPOCH:-10}"
EPOCHS="${EPOCHS:-11}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-11}"
LR="${LR:-1.5e-5}"
MIN_LR="${MIN_LR:-5e-6}"
QUANT_LR_MULTIPLIER="${QUANT_LR_MULTIPLIER:-2}"
TEACHER_FEATURE_OUTPUT_WEIGHT="${TEACHER_FEATURE_OUTPUT_WEIGHT:-0.003}"
TEACHER_FEATURE_OUTPUT_LAYERS="${TEACHER_FEATURE_OUTPUT_LAYERS:-features.5.5,features.7.1}"
QUANT_ONLY_START_EPOCH="${QUANT_ONLY_START_EPOCH:-10}"
TRAINABLE_POLICY="${TRAINABLE_POLICY:-params_in_layers}"
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS="${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS:-features.5.5,features.7.1}"
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
  --epochs "${EPOCHS}" --scheduler-epochs "${SCHEDULER_EPOCHS}" --batch-size 64 --workers 8
  --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay 0.0
  --quant-lr-multiplier "${QUANT_LR_MULTIPLIER}"
  --epoch-checkpoint-interval 1 --checkpoint-hist 10
  --wbits 4 --abits 4 --wq-mode lsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized
  --amp --amp-dtype bf16
  --extra-arg=--static-graph
  --extra-arg=--smoothing --extra-arg=0.1
  --extra-arg=--mixup --extra-arg=0.0
  --extra-arg=--cutmix --extra-arg=0.0
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1
  --extra-arg=--color-jitter --extra-arg=0.4
  --extra-arg=--reprob --extra-arg=0.25
  --extra-arg=--log-interval --extra-arg=50
  --extra-arg=--seed --extra-arg=42
)

if [[ -n "${TEACHER_FEATURE_OUTPUT_WEIGHT}" && "${TEACHER_FEATURE_OUTPUT_WEIGHT}" != "0" ]]; then
  cmd+=(
    --teacher-feature-output-weight "${TEACHER_FEATURE_OUTPUT_WEIGHT}"
    --teacher-feature-output-layers "${TEACHER_FEATURE_OUTPUT_LAYERS}"
    --teacher-feature-output-loss norm_mse
  )
fi
if [[ -n "${QUANT_ONLY_START_EPOCH}" ]]; then
  cmd+=(--quant-only-start-epoch "${QUANT_ONLY_START_EPOCH}")
fi
if [[ -n "${TRAINABLE_POLICY}" ]]; then
  cmd+=(--trainable-policy "${TRAINABLE_POLICY}")
fi
if [[ -n "${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS}" ]]; then
  cmd+=(--trainable-policy-freeze-act-except-layers "${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS}")
fi
if [[ -n "${MAX_TRAIN_UPDATES}" && "${MAX_TRAIN_UPDATES}" != "0" ]]; then
  cmd+=(--extra-arg=--max_train_updates --extra-arg="${MAX_TRAIN_UPDATES}")
  if [[ "${SKIP_VALIDATE}" == "1" ]]; then
    cmd+=(--extra-arg=--skip_validate)
  fi
fi

{
  echo "===== Resume10 clean LSQ no-QKR gate $(date '+%F %T') ====="
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
  echo "LR=${LR}"
  echo "MIN_LR=${MIN_LR}"
  echo "QUANT_LR_MULTIPLIER=${QUANT_LR_MULTIPLIER}"
  echo "WQ_MODE=lsq"
  echo "AQ_MODE=lsq"
  echo "QK_REPARAM=0"
  echo "TEACHER_FEATURE_OUTPUT_WEIGHT=${TEACHER_FEATURE_OUTPUT_WEIGHT}"
  echo "TEACHER_FEATURE_OUTPUT_LAYERS=${TEACHER_FEATURE_OUTPUT_LAYERS}"
  echo "QUANT_ONLY_START_EPOCH=${QUANT_ONLY_START_EPOCH}"
  echo "TRAINABLE_POLICY=${TRAINABLE_POLICY}"
  echo "TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS}"
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
