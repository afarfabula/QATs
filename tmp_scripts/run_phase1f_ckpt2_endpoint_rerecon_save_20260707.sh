#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
EXP="${EXP:-recipe_phase1f_ckpt2_endpoint_rerecon100_20260707}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
TEACHER="${TEACHER:-/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth}"
RESUME="${RESUME:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30621}"
PRE_QAT_FEATURE_RECON_UPDATES="${PRE_QAT_FEATURE_RECON_UPDATES:-100}"
PRE_QAT_FEATURE_RECON_LAYERS="${PRE_QAT_FEATURE_RECON_LAYERS:-features.5.5,features.7.1}"
PRE_QAT_FEATURE_RECON_POLICY="${PRE_QAT_FEATURE_RECON_POLICY:-quant}"
TEACHER_FEATURE_OUTPUT_WEIGHT="${TEACHER_FEATURE_OUTPUT_WEIGHT:-0.003}"
LR="${LR:-1.5e-5}"
MIN_LR="${MIN_LR:-5e-6}"
QUANT_LR_MULTIPLIER="${QUANT_LR_MULTIPLIER:-2}"
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
  --epochs 0 --scheduler-epochs 3 --batch-size 64 --workers 8
  --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay 0.0
  --quant-lr-multiplier "${QUANT_LR_MULTIPLIER}"
  --pre-qat-feature-recon-updates "${PRE_QAT_FEATURE_RECON_UPDATES}"
  --pre-qat-feature-recon-layers "${PRE_QAT_FEATURE_RECON_LAYERS}"
  --pre-qat-feature-recon-policy "${PRE_QAT_FEATURE_RECON_POLICY}"
  --epoch-checkpoint-interval 1 --checkpoint-hist 10
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized --qk-reparam --qk-reparam-type 0
  --amp --amp-dtype bf16
  --extra-arg=--static-graph
  --extra-arg=--save_step_checkpoints
  --extra-arg=--save_initial_step_checkpoint
  --extra-arg=--step_checkpoint_warmup_updates --extra-arg=0
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
    --teacher-feature-output-layers features.5.5,features.7.1
    --teacher-feature-output-loss norm_mse
  )
fi

{
  echo "===== Phase 2H endpoint re-reconstruction save $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "RESUME=${RESUME}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "PRE_QAT_FEATURE_RECON_UPDATES=${PRE_QAT_FEATURE_RECON_UPDATES}"
  echo "PRE_QAT_FEATURE_RECON_LAYERS=${PRE_QAT_FEATURE_RECON_LAYERS}"
  echo "PRE_QAT_FEATURE_RECON_POLICY=${PRE_QAT_FEATURE_RECON_POLICY}"
  echo "TEACHER_FEATURE_OUTPUT_WEIGHT=${TEACHER_FEATURE_OUTPUT_WEIGHT}"
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
  echo "step_checkpoint=${OUT}/${EXP}/step_checkpoints/step_0000.pth.tar"
} | tee -a "${LOG}"
