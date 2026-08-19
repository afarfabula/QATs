#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
OUT="${OUT:-/tmp/qat_public_repro}"
PREV_EXP="${PREV_EXP:-ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731}"
EXP="${EXP:-ofq_resume200_to300_fixedcycle_sparse_prevstep_refkl_20260802}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
TEACHER="${TEACHER:-/mlx_devbox/users/quyanyi/playground/QATs/checkpoints/pretrained/swin_t-704ceda3.pth}"
RESUME="${RESUME:-/mlx_devbox/users/quyanyi/playground/QATs/checkpoints/resume_sources/ofq_200ep_fromscratch_fixedcycle_group_sparse_prevstep_refkl_20260731/checkpoint-200.pth.tar}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-31937}"
DRY_RUN="${DRY_RUN:-0}"
SECONDS=0

PRIMARY_HEADS="custom_subset:5:7,4:11,8:4"
PRIMARY_SECONDARY_HEADS="custom_subset:5:7,4:11,8:4,11:18,6:1"

build_late_resume_overrides() {
  python3 - <<'PY'
primary = "custom_subset:5:7,4:11,8:4"
primary_secondary = "custom_subset:5:7,4:11,8:4,11:18,6:1"
weight = {}
head = {}

# Strict-resume continuation from checkpoint-200.
# Epoch numbers are zero-based inside qat_launch.py.
# Human-readable epochs 201-300: one sparse pulse every 10 epochs.
# Keep this fixed and independent of validation accuracy.
for pulse_idx, ep in enumerate(range(200, 300, 10), start=1):
    if pulse_idx in {3, 7}:
        weight[ep] = 4e-6
        head[ep] = primary_secondary
    else:
        weight[ep] = 5e-6
        head[ep] = primary

weight_spec = ",".join(f"{ep}:{weight[ep]:.8g}" for ep in sorted(weight))
head_spec = ";".join(f"{ep}={head[ep]}" for ep in sorted(head))
print(weight_spec)
print(head_spec)
PY
}

mapfile -t OVERRIDES < <(build_late_resume_overrides)
REF_WEIGHT_OVERRIDES="${REF_WEIGHT_OVERRIDES:-${OVERRIDES[0]}}"
REF_HEAD_OVERRIDES="${REF_HEAD_OVERRIDES:-${OVERRIDES[1]}}"

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== OFQ resume200->300 fixed-cycle sparse prev-step refKL $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "PREV_EXP=${PREV_EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "RESUME=${RESUME}"
  echo "DEVICES=${DEVICES}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "PRIMARY_HEADS=${PRIMARY_HEADS}"
  echo "PRIMARY_SECONDARY_HEADS=${PRIMARY_SECONDARY_HEADS}"
  echo "REF_WEIGHT_OVERRIDES=${REF_WEIGHT_OVERRIDES}"
  echo "REF_HEAD_OVERRIDES=${REF_HEAD_OVERRIDES}"
  test -f "${TEACHER}" && ls -lh "${TEACHER}" || true
  test -f "${RESUME}" && ls -lh "${RESUME}" || { echo "missing RESUME=${RESUME}"; exit 2; }
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
  --resume "${RESUME}"
  --epochs 300 --scheduler-epochs 300
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
  --ref-attn-kl-weight-epoch-overrides "${REF_WEIGHT_OVERRIDES}"
  --ref-head-mode "${PRIMARY_HEADS}"
  --ref-head-mode-epoch-overrides "${REF_HEAD_OVERRIDES}"
  --ref-warmup-epochs 0
  --ref-attn-kl-drop-prob 1.0
  --ref-attn-kl-clip 20.0
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
