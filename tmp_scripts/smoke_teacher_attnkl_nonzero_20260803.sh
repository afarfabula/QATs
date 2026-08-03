#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
OUT="${OUT:-/tmp/qat_public_repro}"
EXP="${EXP:-smoke_teacher_attnkl_nonzero_20260803}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
TEACHER="${TEACHER:-/mlx_devbox/users/quyanyi/playground/QATs/checkpoints/pretrained/swin_t-704ceda3.pth}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-31967}"
PRIMARY_HEADS="custom_subset:8:4,11:18,6:1"

rm -f "${LOG}"
mkdir -p "${OUT}"

CMD=(python3 "${QATS}/qat_launch.py"
  --method ofq --stage train
  --config "${QATS}/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml"
  --model swin_t --data "${DATA}" --dataset-format parquet
  --output "${OUT}" --experiment "${EXP}"
  --devices "${DEVICES}" --nproc-per-node 8 --master-port "${MASTER_PORT}"
  --model-type swin
  --teacher swin_t --teacher-type swin --teacher-checkpoint "${TEACHER}" --teacher-pretrained
  --epochs 1 --scheduler-epochs 100
  --batch-size 64 --workers 8 --lr 2e-4 --min-lr 5e-6 --weight-decay 0.0
  --epoch-checkpoint-interval 1 --checkpoint-hist 2
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized --qk-reparam --qk-reparam-type 0
  --amp --amp-dtype bf16
  --train-scheme ema_ref_attn_kl
  --ref-update prev_step --ref-update-interval 50
  --ref-attn-loss kl_ref
  --ref-attn-kl-weight 0.0
  --ref-head-mode "${PRIMARY_HEADS}"
  --ref-head-mode-epoch-overrides "0=${PRIMARY_HEADS}"
  --ref-warmup-epochs 0
  --ref-attn-kl-drop-prob 1.0
  --ref-attn-kl-clip 20.0
  --teacher-attn-kl-weight 0.0
  --teacher-attn-kl-warmup-epochs 0
  --teacher-attn-kl-weight-epoch-overrides "0:2e-6"
  --extra-arg=--static-graph
  --extra-arg=--smoothing --extra-arg=0.1
  --extra-arg=--mixup --extra-arg=0.0
  --extra-arg=--cutmix --extra-arg=0.0
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1
  --extra-arg=--color-jitter --extra-arg=0.4
  --extra-arg=--reprob --extra-arg=0.25
  --extra-arg=--log-interval --extra-arg=1
  --extra-arg=--max_train_updates --extra-arg=20
  --extra-arg=--seed --extra-arg=42
)

{
  echo "===== smoke teacher attn KL nonzero $(date '+%F %T') ====="
  git -C "${QATS}" rev-parse --short HEAD || true
  printf '%q ' "${CMD[@]}"
  echo
} | tee "${LOG}"

PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee -a "${LOG}"
