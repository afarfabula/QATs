#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
EXP="${EXP:-recipe_resume10_actcurr_transition_recon_20260707}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
TEACHER="${TEACHER:-/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth}"
RESUME="${RESUME:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-30647}"
EPOCHS="${EPOCHS:-4}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-4}"
LR="${LR:-1.5e-5}"
MIN_LR="${MIN_LR:-5e-6}"
QUANT_LR_MULTIPLIER="${QUANT_LR_MULTIPLIER:-2}"
PROGRESSIVE_BIT_SCHEDULE="${PROGRESSIVE_BIT_SCHEDULE:-0:4:8,1:4:8,2:4:6,3:4:4}"
PROGRESSIVE_BIT_RECALIBRATE_EPOCHS="${PROGRESSIVE_BIT_RECALIBRATE_EPOCHS:-2,3}"
PROGRESSIVE_BIT_RECALIBRATE_BATCHES="${PROGRESSIVE_BIT_RECALIBRATE_BATCHES:-4}"
TRANSITION_RECON_UPDATES="${TRANSITION_RECON_UPDATES:-80}"
TRANSITION_RECON_EPOCHS="${TRANSITION_RECON_EPOCHS:-2,3}"
TRANSITION_RECON_LAYERS="${TRANSITION_RECON_LAYERS:-features.5.5,features.7.1}"
TRANSITION_RECON_POLICY="${TRANSITION_RECON_POLICY:-module_all}"
TRANSITION_RECON_WEIGHT_MODE="${TRANSITION_RECON_WEIGHT_MODE:-none}"
TRANSITION_RECON_CONFIDENCE_POWER="${TRANSITION_RECON_CONFIDENCE_POWER:-0}"
TRANSITION_RECON_QDROP_PROB="${TRANSITION_RECON_QDROP_PROB:-0}"
TRANSITION_RECON_QDROP_LAYERS="${TRANSITION_RECON_QDROP_LAYERS:-}"
TRANSITION_ANCHOR_KL_WEIGHT="${TRANSITION_ANCHOR_KL_WEIGHT:-0}"
TRANSITION_ANCHOR_KL_TEMPERATURE="${TRANSITION_ANCHOR_KL_TEMPERATURE:-2.75}"
TEACHER_FEATURE_OUTPUT_WEIGHT="${TEACHER_FEATURE_OUTPUT_WEIGHT:-0.003}"
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
USE_STATIC_GRAPH="${USE_STATIC_GRAPH:-0}"
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
  --devices "${DEVICES}" --nproc-per-node "${NPROC_PER_NODE}" --master-port "${MASTER_PORT}" --model-type swin
  --teacher swin_t --teacher-type swin --teacher-checkpoint "${TEACHER}" --teacher-pretrained
  --resume "${RESUME}" --no-resume-opt --start-epoch 0
  --epochs "${EPOCHS}" --scheduler-epochs "${SCHEDULER_EPOCHS}" --batch-size 64 --workers 8
  --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay 0.0
  --quant-lr-multiplier "${QUANT_LR_MULTIPLIER}"
  --progressive-bit-schedule "${PROGRESSIVE_BIT_SCHEDULE}"
  --progressive-bit-rescale-lsq
  --progressive-bit-recalibrate-epochs "${PROGRESSIVE_BIT_RECALIBRATE_EPOCHS}"
  --progressive-bit-recalibrate-batches "${PROGRESSIVE_BIT_RECALIBRATE_BATCHES}"
  --progressive-bit-transition-recon-updates "${TRANSITION_RECON_UPDATES}"
  --progressive-bit-transition-recon-epochs "${TRANSITION_RECON_EPOCHS}"
  --progressive-bit-transition-recon-layers "${TRANSITION_RECON_LAYERS}"
  --progressive-bit-transition-recon-policy "${TRANSITION_RECON_POLICY}"
  --progressive-bit-transition-recon-weight-mode "${TRANSITION_RECON_WEIGHT_MODE}"
  --progressive-bit-transition-recon-confidence-power "${TRANSITION_RECON_CONFIDENCE_POWER}"
  --progressive-bit-transition-recon-qdrop-prob "${TRANSITION_RECON_QDROP_PROB}"
  --progressive-bit-transition-anchor-kl-weight "${TRANSITION_ANCHOR_KL_WEIGHT}"
  --progressive-bit-transition-anchor-kl-temperature "${TRANSITION_ANCHOR_KL_TEMPERATURE}"
  --epoch-checkpoint-interval 1 --checkpoint-hist 10
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized --qk-reparam --qk-reparam-type 0
  --amp --amp-dtype bf16
  --extra-arg=--smoothing --extra-arg=0.1
  --extra-arg=--mixup --extra-arg=0.0
  --extra-arg=--cutmix --extra-arg=0.0
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1
  --extra-arg=--color-jitter --extra-arg=0.4
  --extra-arg=--reprob --extra-arg=0.25
  --extra-arg=--log-interval --extra-arg=50
  --extra-arg=--seed --extra-arg=42
)

if [[ "${USE_STATIC_GRAPH}" == "1" ]]; then
  cmd+=(--extra-arg=--static-graph)
fi

if [[ -n "${TRANSITION_RECON_QDROP_LAYERS}" ]]; then
  cmd+=(--progressive-bit-transition-recon-qdrop-layers "${TRANSITION_RECON_QDROP_LAYERS}")
fi

if [[ -n "${TEACHER_FEATURE_OUTPUT_WEIGHT}" && "${TEACHER_FEATURE_OUTPUT_WEIGHT}" != "0" ]]; then
  cmd+=(
    --teacher-feature-output-weight "${TEACHER_FEATURE_OUTPUT_WEIGHT}"
    --teacher-feature-output-layers features.5.5,features.7.1
    --teacher-feature-output-loss norm_mse
  )
fi

if [[ "${MAX_TRAIN_UPDATES}" != "0" ]]; then
  cmd+=(--extra-arg=--max_train_updates --extra-arg="${MAX_TRAIN_UPDATES}")
fi

if [[ "${SKIP_VALIDATE}" == "1" ]]; then
  cmd+=(--extra-arg=--skip_validate)
fi

{
  echo "===== Resume10 activation-curriculum transition-local reconstruction gate $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "RESUME=${RESUME}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "EPOCHS=${EPOCHS}"
  echo "SCHEDULER_EPOCHS=${SCHEDULER_EPOCHS}"
  echo "LR=${LR}"
  echo "MIN_LR=${MIN_LR}"
  echo "QUANT_LR_MULTIPLIER=${QUANT_LR_MULTIPLIER}"
  echo "PROGRESSIVE_BIT_SCHEDULE=${PROGRESSIVE_BIT_SCHEDULE}"
  echo "PROGRESSIVE_BIT_RECALIBRATE_EPOCHS=${PROGRESSIVE_BIT_RECALIBRATE_EPOCHS}"
  echo "PROGRESSIVE_BIT_RECALIBRATE_BATCHES=${PROGRESSIVE_BIT_RECALIBRATE_BATCHES}"
  echo "TRANSITION_RECON_UPDATES=${TRANSITION_RECON_UPDATES}"
  echo "TRANSITION_RECON_EPOCHS=${TRANSITION_RECON_EPOCHS}"
  echo "TRANSITION_RECON_LAYERS=${TRANSITION_RECON_LAYERS}"
  echo "TRANSITION_RECON_POLICY=${TRANSITION_RECON_POLICY}"
  echo "TRANSITION_RECON_WEIGHT_MODE=${TRANSITION_RECON_WEIGHT_MODE}"
  echo "TRANSITION_RECON_CONFIDENCE_POWER=${TRANSITION_RECON_CONFIDENCE_POWER}"
  echo "TRANSITION_RECON_QDROP_PROB=${TRANSITION_RECON_QDROP_PROB}"
  echo "TRANSITION_RECON_QDROP_LAYERS=${TRANSITION_RECON_QDROP_LAYERS}"
  echo "TRANSITION_ANCHOR_KL_WEIGHT=${TRANSITION_ANCHOR_KL_WEIGHT}"
  echo "TRANSITION_ANCHOR_KL_TEMPERATURE=${TRANSITION_ANCHOR_KL_TEMPERATURE}"
  echo "TEACHER_FEATURE_OUTPUT_WEIGHT=${TEACHER_FEATURE_OUTPUT_WEIGHT}"
  echo "MAX_TRAIN_UPDATES=${MAX_TRAIN_UPDATES}"
  echo "SKIP_VALIDATE=${SKIP_VALIDATE}"
  echo "USE_STATIC_GRAPH=${USE_STATIC_GRAPH}"
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
