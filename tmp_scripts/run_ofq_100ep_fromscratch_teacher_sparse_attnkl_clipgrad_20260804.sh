#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
OUT="${OUT:-/tmp/qat_public_repro}"
EXP="${EXP:-ofq_100ep_fromscratch_teacher_sparse_attnkl_clipgrad_20260804}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
TEACHER="${TEACHER:-/mlx_devbox/users/quyanyi/playground/QATs/checkpoints/pretrained/swin_t-704ceda3.pth}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-31983}"
DRY_RUN="${DRY_RUN:-0}"
SECONDS=0

PRIMARY_HEADS="custom_subset:8:4,11:18,6:1"
PRIMARY_SECONDARY_HEADS="custom_subset:8:4,11:18,6:1,5:7,4:11"

build_teacher_attn_schedules() {
  python3 - <<'PY'
weight = {}
head = {}
primary = "custom_subset:8:4,11:18,6:1"
primary_secondary = "custom_subset:8:4,11:18,6:1,5:7,4:11"

for ep in range(100):
    if ep < 5:
        weight[ep] = 0.0
        head[ep] = primary
    elif ep < 30:
        weight[ep] = 1e-6
        head[ep] = primary
    elif ep < 70:
        weight[ep] = 2e-6
        head[ep] = primary_secondary
    elif ep < 90:
        weight[ep] = 1e-6
        head[ep] = primary
    else:
        weight[ep] = 0.0
        head[ep] = primary

weight_spec = ",".join(f"{ep}:{weight[ep]:.8g}" for ep in range(100))
head_spec = ";".join(f"{ep}={head[ep]}" for ep in range(100))
print(weight_spec)
print(head_spec)
PY
}

mapfile -t SCHEDULES < <(build_teacher_attn_schedules)
TEACHER_ATTN_KL_WEIGHT_OVERRIDES="${TEACHER_ATTN_KL_WEIGHT_OVERRIDES:-${SCHEDULES[0]}}"
REF_HEAD_MODE_OVERRIDES="${REF_HEAD_MODE_OVERRIDES:-${SCHEDULES[1]}}"

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== OFQ 100ep from-scratch teacher sparse attention KL clipgrad $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "PRIMARY_HEADS=${PRIMARY_HEADS}"
  echo "PRIMARY_SECONDARY_HEADS=${PRIMARY_SECONDARY_HEADS}"
  echo "TEACHER_ATTN_KL_WEIGHT_OVERRIDES=${TEACHER_ATTN_KL_WEIGHT_OVERRIDES}"
  echo "REF_HEAD_MODE_OVERRIDES=${REF_HEAD_MODE_OVERRIDES}"
  test -f "${TEACHER}" && ls -lh "${TEACHER}" || { echo "missing TEACHER=${TEACHER}"; exit 2; }
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
  --epoch-checkpoint-interval 1 --checkpoint-hist 30
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized --qk-reparam --qk-reparam-type 0
  --amp --amp-dtype bf16
  --train-scheme ema_ref_attn_kl
  --ref-update prev_step --ref-update-interval 50
  --ref-attn-loss kl_ref
  --ref-attn-kl-weight 0.0
  --ref-head-mode "${PRIMARY_HEADS}"
  --ref-head-mode-epoch-overrides "${REF_HEAD_MODE_OVERRIDES}"
  --ref-warmup-epochs 0
  --ref-attn-kl-drop-prob 1.0
  --ref-attn-kl-clip 20.0
  --teacher-attn-kl-weight 0.0
  --teacher-attn-kl-warmup-epochs 0
  --teacher-attn-kl-weight-epoch-overrides "${TEACHER_ATTN_KL_WEIGHT_OVERRIDES}"
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
} | tee -a "${LOG}"
