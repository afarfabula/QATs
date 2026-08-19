#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
EXP="${EXP:-recipe_phase1f_ckpt2_anchor_continuation_gate_20260707}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
TEACHER="${TEACHER:-/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth}"
RESUME="${RESUME:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30625}"
EPOCHS="${EPOCHS:-1}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-1}"
LR="${LR:-1.0e-5}"
MIN_LR="${MIN_LR:-5e-6}"
QUANT_LR_MULTIPLIER="${QUANT_LR_MULTIPLIER:-1.5}"
VARIATION_TRUST_WEIGHT="${VARIATION_TRUST_WEIGHT:-0.003}"
VARIATION_TRUST_LATE_LAYERS="${VARIATION_TRUST_LATE_LAYERS:-features.5.5,features.7.1}"
VARIATION_TRUST_LATE_MULTIPLIER="${VARIATION_TRUST_LATE_MULTIPLIER:-0.25}"
VARIATION_TRUST_EARLY_LAYERS="${VARIATION_TRUST_EARLY_LAYERS:-features.0.0,features.1.0,features.1.1}"
VARIATION_TRUST_EARLY_MULTIPLIER="${VARIATION_TRUST_EARLY_MULTIPLIER:-3.0}"
VARIATION_TRUST_SOFTMAX_MULTIPLIER="${VARIATION_TRUST_SOFTMAX_MULTIPLIER:-2.0}"
VARIATION_TRUST_MOVE_V_MULTIPLIER="${VARIATION_TRUST_MOVE_V_MULTIPLIER:-2.0}"
VARIATION_TRUST_PROJ_MOVE_MULTIPLIER="${VARIATION_TRUST_PROJ_MOVE_MULTIPLIER:-1.5}"
TEACHER_FEATURE_OUTPUT_WEIGHT="${TEACHER_FEATURE_OUTPUT_WEIGHT:-0.003}"
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-}"
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
  --resume "${RESUME}" --no-resume-opt --start-epoch 0
  --epochs "${EPOCHS}" --scheduler-epochs "${SCHEDULER_EPOCHS}" --batch-size 64 --workers 8
  --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay 0.0
  --quant-lr-multiplier "${QUANT_LR_MULTIPLIER}"
  --variation-trust-weight "${VARIATION_TRUST_WEIGHT}"
  --variation-trust-late-layers "${VARIATION_TRUST_LATE_LAYERS}"
  --variation-trust-late-multiplier "${VARIATION_TRUST_LATE_MULTIPLIER}"
  --variation-trust-early-layers "${VARIATION_TRUST_EARLY_LAYERS}"
  --variation-trust-early-multiplier "${VARIATION_TRUST_EARLY_MULTIPLIER}"
  --variation-trust-softmax-multiplier "${VARIATION_TRUST_SOFTMAX_MULTIPLIER}"
  --variation-trust-move-v-multiplier "${VARIATION_TRUST_MOVE_V_MULTIPLIER}"
  --variation-trust-proj-move-multiplier "${VARIATION_TRUST_PROJ_MOVE_MULTIPLIER}"
  --epoch-checkpoint-interval 1 --checkpoint-hist 10
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized --qk-reparam --qk-reparam-type 0
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

if [[ -n "${MAX_TRAIN_UPDATES}" && "${MAX_TRAIN_UPDATES}" != "0" ]]; then
  cmd+=(--extra-arg=--max_train_updates --extra-arg="${MAX_TRAIN_UPDATES}" --extra-arg=--skip_validate)
fi

if [[ -n "${TEACHER_FEATURE_OUTPUT_WEIGHT}" && "${TEACHER_FEATURE_OUTPUT_WEIGHT}" != "0" ]]; then
  cmd+=(
    --teacher-feature-output-weight "${TEACHER_FEATURE_OUTPUT_WEIGHT}"
    --teacher-feature-output-layers features.5.5,features.7.1
    --teacher-feature-output-loss norm_mse
  )
fi

{
  echo "===== Phase 2I Phase1F ckpt2 anchor-preserved continuation gate $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "RESUME=${RESUME}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "EPOCHS=${EPOCHS}"
  echo "SCHEDULER_EPOCHS=${SCHEDULER_EPOCHS}"
  echo "LR=${LR}"
  echo "MIN_LR=${MIN_LR}"
  echo "QUANT_LR_MULTIPLIER=${QUANT_LR_MULTIPLIER}"
  echo "VARIATION_TRUST_WEIGHT=${VARIATION_TRUST_WEIGHT}"
  echo "VARIATION_TRUST_LATE_LAYERS=${VARIATION_TRUST_LATE_LAYERS}"
  echo "VARIATION_TRUST_LATE_MULTIPLIER=${VARIATION_TRUST_LATE_MULTIPLIER}"
  echo "VARIATION_TRUST_EARLY_LAYERS=${VARIATION_TRUST_EARLY_LAYERS}"
  echo "VARIATION_TRUST_EARLY_MULTIPLIER=${VARIATION_TRUST_EARLY_MULTIPLIER}"
  echo "TEACHER_FEATURE_OUTPUT_WEIGHT=${TEACHER_FEATURE_OUTPUT_WEIGHT}"
  echo "MAX_TRAIN_UPDATES=${MAX_TRAIN_UPDATES}"
  test -f "${RESUME}" && ls -lh "${RESUME}"
  echo "train_shards=$(find "${DATA}/data" -maxdepth 1 -name 'train-*.parquet' | wc -l)"
  echo "validation_shards=$(find "${DATA}/data" -maxdepth 1 -name 'validation-*.parquet' | wc -l)"
  git -C "${QATS}" rev-parse --short HEAD || true
  test -e /dev/nvidia0 && echo gpu-device-present || echo no-gpu-device
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  printf 'command='
  printf '%q ' "${cmd[@]}"
  printf '\n\n'
} | tee "${LOG}"

PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee -a "${LOG}"

{
  echo "wall_seconds=${SECONDS}"
  echo "train_log=${LOG}"
  echo "output=${OUT}/${EXP}"
} | tee -a "${LOG}"
