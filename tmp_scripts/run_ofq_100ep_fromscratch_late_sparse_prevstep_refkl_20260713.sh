#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
OUT="${OUT:-/tmp/qat_public_repro}"
EXP="${EXP:-ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
TEACHER="${TEACHER:-/mlx_devbox/users/quyanyi/playground/QATs/checkpoints/pretrained/swin_t-704ceda3.pth}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-31851}"
DOC_DIR="${DOC_DIR:-/mlx_devbox/users/quyanyi/playground/QATs/docs}"
CONTROLLER_TSV="${CONTROLLER_TSV:-${DOC_DIR}/ofq_100ep_fromscratch_late_sparse_prevstep_refkl_controller_20260713.tsv}"
DRY_RUN="${DRY_RUN:-0}"
SECONDS=0

mkdir -p "${OUT}" "${DOC_DIR}"
rm -f "${LOG}" "${CONTROLLER_TSV}"

{
  echo "===== OFQ 100ep from-scratch late sparse prev-step refKL $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "CONTROLLER_TSV=${CONTROLLER_TSV}"
  echo "DRY_RUN=${DRY_RUN}"
  test -f "${TEACHER}" && ls -lh "${TEACHER}" || true
  echo "train_shards=$(find "${DATA}/data" -maxdepth 1 -name 'train-*.parquet' | wc -l)"
  echo "validation_shards=$(find "${DATA}/data" -maxdepth 1 -name 'validation-*.parquet' | wc -l)"
  git -C "${QATS}" rev-parse --short HEAD || true
  test -e /dev/nvidia0 && echo gpu-device-present || echo no-gpu-device
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  echo
} | tee "${LOG}"

CMD=(python3 "${QATS}/qat_launch.py"
  --method ofq --stage train
  --config "${QATS}/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml"
  --model swin_t --data "${DATA}" --dataset-format parquet
  --output "${OUT}" --experiment "${EXP}"
  --devices "${DEVICES}" --nproc-per-node 8 --master-port "${MASTER_PORT}"
  --model-type swin
  --teacher swin_t --teacher-type swin --teacher-checkpoint "${TEACHER}" --teacher-pretrained
  --epochs 100 --scheduler-epochs 100
  --batch-size 64 --workers 8 --lr 2e-4 --min-lr 5e-6 --weight-decay 0.0
  --epoch-checkpoint-interval 1 --checkpoint-hist 100
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized --qk-reparam --qk-reparam-type 0
  --amp --amp-dtype bf16
  --train-scheme ema_ref_attn_kl
  --ref-update prev_step --ref-update-interval 50
  --ref-attn-loss kl_ref
  --ref-attn-kl-weight 0.0
  --ref-head-mode custom_subset:8:4
  --ref-warmup-epochs 0
  --ref-attn-kl-drop-prob 0.50
  --ref-attn-kl-clip 20.0
  --dynamic-sparse-prevstep-kl
  --dynamic-kl-start-epoch 51
  --dynamic-kl-observe-until-epoch 50
  --dynamic-kl-primary-heads 8:4,5:7,4:11
  --dynamic-kl-secondary-heads 11:18,6:1
  --dynamic-kl-avoid-heads 6:6,7:7,4:1,2:4,10:13,11:4,6:7,11:16
  --dynamic-kl-drop-threshold 0.06
  --dynamic-kl-strong-drop-threshold 0.14
  --dynamic-kl-default-weight 0.00001
  --dynamic-kl-strong-weight 0.00002
  --dynamic-kl-max-weight 0.00002
  --dynamic-kl-cooldown-epochs 6
  --dynamic-kl-window-epochs 10
  --dynamic-kl-max-pulses-per-window 3
  --dynamic-kl-controller-tsv "${CONTROLLER_TSV}"
  --dynamic-kl-prior-source ofq_100ep_fromscratch_late_sparse_prevstep_refkl_20260713_static_controller
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

if [[ "${DRY_RUN}" == "1" ]]; then
  "${CMD[@]}" --dry-run 2>&1 | tee -a "${LOG}"
  exit 0
fi

PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee -a "${LOG}"

{
  echo "wall_seconds=${SECONDS}"
  echo "train_log=${LOG}"
  echo "output=${OUT}/${EXP}"
  echo "controller_tsv=${CONTROLLER_TSV}"
} | tee -a "${LOG}"
