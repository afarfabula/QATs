#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-deit_tiny_mixed_bit_gate_v2_20ep_4gpu_20261126}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-0,1,2,3}"
NPROC="${NPROC:-$(echo "${DEVICES}" | tr ',' '\n' | grep -c .)}"
MASTER_PORT="${MASTER_PORT:-30983}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EPOCHS="${EPOCHS:-20}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-20}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-3}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

export MIXED_BIT_GATE_ENABLE="${MIXED_BIT_GATE_ENABLE:-1}"
export MIXED_BIT_TARGET_BITS="${MIXED_BIT_TARGET_BITS:-2.0}"
export MIXED_BIT_BUDGET_TOL="${MIXED_BIT_BUDGET_TOL:-0.015}"
export MIXED_BIT_BUDGET_WEIGHT="${MIXED_BIT_BUDGET_WEIGHT:-25.0}"
export MIXED_BIT_SEARCH_START_EPOCH="${MIXED_BIT_SEARCH_START_EPOCH:-2}"
export MIXED_BIT_HARD_START_EPOCH="${MIXED_BIT_HARD_START_EPOCH:-6}"
export MIXED_BIT_FREEZE_EPOCH="${MIXED_BIT_FREEZE_EPOCH:-12}"
export MIXED_BIT_WEIGHT_RAMP_EPOCHS="${MIXED_BIT_WEIGHT_RAMP_EPOCHS:-3}"
export MIXED_BIT_GATE_LR="${MIXED_BIT_GATE_LR:-0.003}"
export MIXED_BIT_TAU_START="${MIXED_BIT_TAU_START:-1.0}"
export MIXED_BIT_TAU_MID="${MIXED_BIT_TAU_MID:-0.35}"
export MIXED_BIT_TAU_END="${MIXED_BIT_TAU_END:-0.15}"
export MIXED_BIT_INIT_LOGITS="${MIXED_BIT_INIT_LOGITS:-0.5,1.5,-0.4}"
export MIXED_BIT_HARD_BUDGET="${MIXED_BIT_HARD_BUDGET:-1}"
export MIXED_BIT_SMOKE_BATCHES="${MIXED_BIT_SMOKE_BATCHES:-0}"

EXTRA_ARGS=(
  --seed 42
  --log-interval 50
)
if [[ "${SKIP_VALIDATE}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_validate)
fi

EXTRA_CLI_FLAGS=()
for arg in "${EXTRA_ARGS[@]}"; do
  EXTRA_CLI_FLAGS+=("--extra-arg=${arg}")
done

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== DeiT-Tiny Learnable Mixed-Bit Gated QAT, $(date '+%F %T') ====="
  echo "EXP=${EXP}"
  echo "DEVICES=${DEVICES} NPROC=${NPROC}"
  echo "EPOCHS=${EPOCHS} SCHEDULER_EPOCHS=${SCHEDULER_EPOCHS}"
  echo "MIXED_BIT_TARGET_BITS=${MIXED_BIT_TARGET_BITS} TOL=${MIXED_BIT_BUDGET_TOL} WEIGHT=${MIXED_BIT_BUDGET_WEIGHT}"
  echo "PHASES=warmup:0-${MIXED_BIT_SEARCH_START_EPOCH} soft:${MIXED_BIT_SEARCH_START_EPOCH}-${MIXED_BIT_HARD_START_EPOCH} hard:${MIXED_BIT_HARD_START_EPOCH}-${MIXED_BIT_FREEZE_EPOCH} frozen:${MIXED_BIT_FREEZE_EPOCH}-${EPOCHS}"
  echo "GATE_LR=${MIXED_BIT_GATE_LR} INIT_LOGITS=${MIXED_BIT_INIT_LOGITS} SMOKE_BATCHES=${MIXED_BIT_SMOKE_BATCHES}"
} | tee -a "${LOG}"

"${PY}" "${QATS}/qat_launch.py" \
  --method ofq \
  --stage train \
  --config "${QATS}/third_party/OFQ/configs/deit_tiny_qat_baseline_100ep.yml" \
  --model deit_tiny_distilled_patch16_224 \
  --model-type deit \
  --teacher deit_tiny_distilled_patch16_224 \
  --teacher-type deit \
  --teacher-checkpoint "${TEACHER}" \
  --teacher-pretrained \
  --data "${DATA}" \
  --dataset-format folder \
  --output "${OUT}" \
  --experiment "${EXP}" \
  --devices "${DEVICES}" \
  --nproc-per-node "${NPROC}" \
  --master-port "${MASTER_PORT}" \
  --epochs "${EPOCHS}" \
  --scheduler-epochs "${SCHEDULER_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --workers 8 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --weight-decay 5e-2 \
  --warmup-epochs 5 \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epoch-checkpoint-interval 1 \
  --checkpoint-hist "${CHECKPOINT_HIST}" \
  --num-aug-repeats 0 \
  --wbits 2 \
  --abits 2 \
  --wq-mode statsq \
  --aq-mode lsq \
  --wq-per-channel \
  --aq-per-channel \
  --aq-clip-learnable \
  --pretrained \
  --pretrained-initialized \
  --use-kd \
  --kd-hard-and-soft 1 \
  --ofq-official-kd-heads \
  --teacher-soft-temperature 1.0 \
  --logit-rank-weight 0 \
  --attn-rank-weight 0 \
  --quantized \
  --qk-reparam \
  --qk-reparam-type 0 \
  --amp \
  --amp-dtype bf16 \
  --model-ema \
  --model-ema-decay 0.999 \
  "${EXTRA_CLI_FLAGS[@]}" \
  2>&1 | tee -a "${LOG}"