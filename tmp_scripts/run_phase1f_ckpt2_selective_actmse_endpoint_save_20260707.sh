#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
EXP="${EXP:-recipe_phase1f_ckpt2_selective_actmse_endpoint_20260707}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
TEACHER="${TEACHER:-/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth}"
RESUME="${RESUME:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_prerecon100_lowlr_b_20260706/checkpoint-2.pth.tar}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-30658}"
ACT_MSE_BATCHES="${ACT_MSE_BATCHES:-8}"
ACT_MSE_LAYERS="${ACT_MSE_LAYERS:-features.5.5,features.7.1}"
ACT_MSE_QUANTIZERS="${ACT_MSE_QUANTIZERS:-features.5.5.attn.quan_a_qkx_fn,features.7.1.attn.quant_x_4_qkv.input_quant_fn,features.5.5.attn.quan_a_v_fn}"
ACT_MSE_GRID="${ACT_MSE_GRID:-0.85,1.25,17}"
ACT_MSE_BLEND="${ACT_MSE_BLEND:-0.35}"
LR="${LR:-8.0e-6}"
MIN_LR="${MIN_LR:-4.0e-6}"
QUANT_LR_MULTIPLIER="${QUANT_LR_MULTIPLIER:-1.5}"
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
  --epochs 0 --scheduler-epochs 1 --batch-size 64 --workers 8
  --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay 0.0
  --quant-lr-multiplier "${QUANT_LR_MULTIPLIER}"
  --pre-qat-act-mse-calib-batches "${ACT_MSE_BATCHES}"
  --pre-qat-act-mse-calib-layers "${ACT_MSE_LAYERS}"
  --pre-qat-act-mse-calib-quantizers "${ACT_MSE_QUANTIZERS}"
  --pre-qat-act-mse-calib-grid "${ACT_MSE_GRID}"
  --pre-qat-act-mse-calib-blend "${ACT_MSE_BLEND}"
  --epoch-checkpoint-interval 1 --checkpoint-hist 10
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized --qk-reparam --qk-reparam-type 0
  --amp --amp-dtype bf16
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

{
  echo "===== Phase1F ckpt2 selective activation-MSE endpoint save $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "RESUME=${RESUME}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "ACT_MSE_BATCHES=${ACT_MSE_BATCHES}"
  echo "ACT_MSE_LAYERS=${ACT_MSE_LAYERS}"
  echo "ACT_MSE_QUANTIZERS=${ACT_MSE_QUANTIZERS}"
  echo "ACT_MSE_GRID=${ACT_MSE_GRID}"
  echo "ACT_MSE_BLEND=${ACT_MSE_BLEND}"
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
