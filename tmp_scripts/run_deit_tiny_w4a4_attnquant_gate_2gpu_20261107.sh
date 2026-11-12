#!/usr/bin/env bash
# Teacher-free attention quantization-fidelity gate on DeiT-Tiny W4A4.
#
# Objective:
#   T^2 * KD(T=2.75)
#   + ATTENTION_QUANT_WEIGHT * [
#         JSD(normalize(post_quant_attn), normalize(stopgrad(pre_quant_attn)))
#         + ATTENTION_QUANT_MASS_WEIGHT * relative_row_mass_error^2
#     ]
#
# This uses only the current student forward. No teacher attention and no temporal
# reference attention are used by the auxiliary loss.
#
# Usage:
#   DEVICES=0,1 ATTENTION_QUANT_WEIGHT=0.03 MAX_UPDATES=2503 \
#     bash tmp_scripts/run_deit_tiny_w4a4_attnquant_gate_2gpu_20261107.sh
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-deit_tiny_w4a4_attnquant_gate_2gpu_20261107}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-30694}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EPOCHS="${EPOCHS:-100}"
MAX_UPDATES="${MAX_UPDATES:-2503}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-1}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
ATTENTION_QUANT_WEIGHT="${ATTENTION_QUANT_WEIGHT:-0.03}"
ATTENTION_QUANT_JSD_WEIGHT="${ATTENTION_QUANT_JSD_WEIGHT:-1.0}"
ATTENTION_QUANT_MASS_WEIGHT="${ATTENTION_QUANT_MASS_WEIGHT:-1.0}"
ATTENTION_QUANT_TOPK_MASS_WEIGHT="${ATTENTION_QUANT_TOPK_MASS_WEIGHT:-0}"
ATTENTION_QUANT_TOPK="${ATTENTION_QUANT_TOPK:-5}"
ATTENTION_QUANT_PROBE="${ATTENTION_QUANT_PROBE:-0}"
ATTENTION_SOFTMAX_POST_QUANT_RENORM="${ATTENTION_SOFTMAX_POST_QUANT_RENORM:-0}"
ATTENTION_SOFTMAX_FP="${ATTENTION_SOFTMAX_FP:-0}"
ATTENTION_SOFTMAX_FP_LAYERS="${ATTENTION_SOFTMAX_FP_LAYERS:-}"
ACTIVATION_QUANT_FP="${ACTIVATION_QUANT_FP:-0}"
ACTIVATION_QUANT_FP_SUFFIXES="${ACTIVATION_QUANT_FP_SUFFIXES:-}"
ATTENTION_OUTPUT_WEIGHT="${ATTENTION_OUTPUT_WEIGHT:-0}"
ATTENTION_OUTPUT_PROBE="${ATTENTION_OUTPUT_PROBE:-0}"
QSS_DECAY="${QSS_DECAY:-0}"
QSS_SYNC_INTERVAL="${QSS_SYNC_INTERVAL:-0}"
QSS_PULL="${QSS_PULL:-0}"
QSS_POLICY="${QSS_POLICY:-attention_softmax}"
QSS_OBSERVE_START_EPOCH="${QSS_OBSERVE_START_EPOCH:-0}"
QSS_START_EPOCH="${QSS_START_EPOCH:-0}"
LSQ_SCALE_FLOOR="${LSQ_SCALE_FLOOR:-0}"
LSQ_SCALE_FLOOR_TARGET="${LSQ_SCALE_FLOOR_TARGET:-attention_softmax}"
AOQ_EXPLORE_SCALE_RATIO="${AOQ_EXPLORE_SCALE_RATIO:-1.0}"
AOQ_EXPLORE_THRESHOLD_RATIO="${AOQ_EXPLORE_THRESHOLD_RATIO:-0}"
AOQ_EXPLORE_START_UPDATE="${AOQ_EXPLORE_START_UPDATE:-0}"
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-0}"
AOQ_EXPLORE_ATTENTION_SOFTMAX="${AOQ_EXPLORE_ATTENTION_SOFTMAX:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
CKPT="${CKPT:-}"
RESUME="${RESUME:-}"
START_EPOCH="${START_EPOCH:-0}"
SECONDS=0

EXTRA_ARGS=(
  --seed 42
  --log-interval 50
  --attention-quant-weight "${ATTENTION_QUANT_WEIGHT}"
  --attention-quant-jsd-weight "${ATTENTION_QUANT_JSD_WEIGHT}"
  --attention-quant-mass-weight "${ATTENTION_QUANT_MASS_WEIGHT}"
  --attention-quant-topk-mass-weight "${ATTENTION_QUANT_TOPK_MASS_WEIGHT}"
  --attention-quant-topk "${ATTENTION_QUANT_TOPK}"
)
RESUME_ARGS=()
if [[ -n "${RESUME}" ]]; then
  RESUME_ARGS=(--resume "${RESUME}" --start-epoch "${START_EPOCH}")
fi
if [[ "${ATTENTION_QUANT_PROBE}" == "1" ]]; then
  EXTRA_ARGS+=(--attention-quant-probe)
fi
if [[ "${ATTENTION_SOFTMAX_POST_QUANT_RENORM}" == "1" ]]; then
  EXTRA_ARGS+=(--attention-softmax-post-quant-renorm)
fi
if [[ "${ATTENTION_SOFTMAX_FP}" == "1" ]]; then
  EXTRA_ARGS+=(--attention-softmax-fp)
  if [[ -n "${ATTENTION_SOFTMAX_FP_LAYERS}" ]]; then
    EXTRA_ARGS+=(--attention-softmax-fp-layers "${ATTENTION_SOFTMAX_FP_LAYERS}")
  fi
fi
if [[ "${ACTIVATION_QUANT_FP}" == "1" ]]; then
  EXTRA_ARGS+=(--activation-quant-fp)
  if [[ -n "${ACTIVATION_QUANT_FP_SUFFIXES}" ]]; then
    EXTRA_ARGS+=(--activation-quant-fp-suffixes "${ACTIVATION_QUANT_FP_SUFFIXES}")
  fi
fi
if awk "BEGIN {exit !(${QSS_DECAY} > 0)}"; then
  EXTRA_ARGS+=(
    --quant-slow-state-decay "${QSS_DECAY}"
    --quant-slow-state-sync-interval "${QSS_SYNC_INTERVAL}"
    --quant-slow-state-pull "${QSS_PULL}"
    --quant-slow-state-policy "${QSS_POLICY}"
    --quant-slow-state-observe-start-epoch "${QSS_OBSERVE_START_EPOCH}"
    --quant-slow-state-start-epoch "${QSS_START_EPOCH}"
  )
fi
if awk "BEGIN {exit !(${LSQ_SCALE_FLOOR} > 0)}"; then
  EXTRA_ARGS+=(
    --lsq-scale-floor "${LSQ_SCALE_FLOOR}"
    --lsq-scale-floor-target "${LSQ_SCALE_FLOOR_TARGET}"
  )
fi
if [[ "${AOQ_EXPLORE_ATTENTION_SOFTMAX}" == "1" ]]; then
  EXTRA_ARGS+=(
    --aoq-explore-attention-softmax
    --aoq-explore-scale-ratio "${AOQ_EXPLORE_SCALE_RATIO}"
    --aoq-explore-threshold-ratio "${AOQ_EXPLORE_THRESHOLD_RATIO}"
    --aoq-explore-start-update "${AOQ_EXPLORE_START_UPDATE}"
    --aoq-explore-end-update "${AOQ_EXPLORE_END_UPDATE}"
  )
fi
if awk "BEGIN {exit !(${ATTENTION_OUTPUT_WEIGHT} > 0)}"; then
  EXTRA_ARGS+=(--attention-output-weight "${ATTENTION_OUTPUT_WEIGHT}")
fi
if [[ "${ATTENTION_OUTPUT_PROBE}" == "1" ]]; then
  EXTRA_ARGS+=(--attention-output-probe)
fi
if [[ "${EVAL_ONLY}" == "1" ]]; then
  if [[ -z "${CKPT}" ]]; then
    echo "EVAL_ONLY=1 requires CKPT" >&2
    exit 2
  fi
  EXTRA_ARGS+=(--eval-only --initial-checkpoint "${CKPT}")
elif [[ "${MAX_UPDATES}" -gt 0 ]]; then
  EXTRA_ARGS+=(--max_train_updates "${MAX_UPDATES}")
fi
if [[ "${SKIP_VALIDATE}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_validate)
fi

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== DeiT-Tiny W4A4 attention-quant fidelity gate, 2x RTX3090, $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "git_head=$(git -C "${QATS}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty_files=$(git -C "${QATS}" status --porcelain 2>/dev/null | wc -l)"
  echo "DATA=${DATA} (folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}  batch=${BATCH_SIZE}/GPU  requested_accum=${GRAD_ACCUM_STEPS}  ranks=2"
  echo "epochs=${EPOCHS}  max_train_updates=${MAX_UPDATES}  checkpoint_hist=${CHECKPOINT_HIST}"
  echo "base_objective=T^2*KD(T=2.75)"
  echo "attention_quant_weight=${ATTENTION_QUANT_WEIGHT}  jsd_weight=${ATTENTION_QUANT_JSD_WEIGHT}  mass_weight=${ATTENTION_QUANT_MASS_WEIGHT}  topk_mass_weight=${ATTENTION_QUANT_TOPK_MASS_WEIGHT}  topk=${ATTENTION_QUANT_TOPK}"
  echo "attention_quant_probe=${ATTENTION_QUANT_PROBE}"
  echo "attention_softmax_post_quant_renorm=${ATTENTION_SOFTMAX_POST_QUANT_RENORM}"
  echo "attention_softmax_fp=${ATTENTION_SOFTMAX_FP}"
  echo "attention_softmax_fp_layers=${ATTENTION_SOFTMAX_FP_LAYERS:-all}"
  echo "activation_quant_fp=${ACTIVATION_QUANT_FP} suffixes=${ACTIVATION_QUANT_FP_SUFFIXES:-none}"
  echo "quant_slow_state=${QSS_DECAY}/${QSS_SYNC_INTERVAL}/${QSS_PULL}/${QSS_POLICY} observe_epoch=${QSS_OBSERVE_START_EPOCH} start_epoch=${QSS_START_EPOCH}"
  echo "lsq_scale_floor=${LSQ_SCALE_FLOOR} target=${LSQ_SCALE_FLOOR_TARGET}"
  echo "aoq_attention_softmax=${AOQ_EXPLORE_ATTENTION_SOFTMAX} scale=${AOQ_EXPLORE_SCALE_RATIO} threshold=${AOQ_EXPLORE_THRESHOLD_RATIO} updates=${AOQ_EXPLORE_START_UPDATE}..${AOQ_EXPLORE_END_UPDATE}"
  echo "attention_output_weight=${ATTENTION_OUTPUT_WEIGHT}  probe=${ATTENTION_OUTPUT_PROBE}"
  echo "eval_only=${EVAL_ONLY}  checkpoint=${CKPT:-none}"
  echo "resume=${RESUME:-none}  start_epoch=${START_EPOCH}"
  echo "quant=W4A4 main blocks + QK reparam; patch/head W8A8"
  echo "PY=${PY}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo "start_epoch=$(date +%s)"
  echo
} | tee "${LOG}"

set +e
TORCH_HOME="${TORCH_HOME:-/datadisk2/quyanyi/cache/torch}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
"${PY}" "${QATS}/qat_launch.py" \
  --method ofq --stage train \
  --config "${QATS}/third_party/OFQ/configs/deit_tiny_qat_baseline_100ep.yml" \
  --model deit_tiny_distilled_patch16_224 --model-type deit \
  --teacher deit_tiny_distilled_patch16_224 --teacher-type deit \
  --teacher-checkpoint "${TEACHER}" --teacher-pretrained \
  --data "${DATA}" --dataset-format folder \
  --output "${OUT}" --experiment "${EXP}" \
  --devices "${DEVICES}" --nproc-per-node 2 --master-port "${MASTER_PORT}" \
  --epochs "${EPOCHS}" --scheduler-epochs "${EPOCHS}" \
  "${RESUME_ARGS[@]}" \
  --batch-size "${BATCH_SIZE}" --workers 8 \
  --lr 5e-4 --min-lr 1e-5 --weight-decay 5e-2 --warmup-epochs 5 \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epoch-checkpoint-interval 1 --checkpoint-hist "${CHECKPOINT_HIST}" \
  --num-aug-repeats 0 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
  --logit-rank-weight 0 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.8 \
  --extra-arg=--cutmix --extra-arg=1.0 \
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1 \
  --extra-arg=--color-jitter --extra-arg=0.4 \
  --extra-arg=--reprob --extra-arg=0.25 \
  "${EXTRA_ARGS[@]/#/--extra-arg=}" \
  >> "${LOG}" 2>&1
status=$?
set -e

{
  echo
  echo "exit_status=${status}"
  echo "end_epoch=$(date +%s)"
  echo "wall_seconds=${SECONDS}"
  echo "train_log=${LOG}"
  echo "output=${OUT}/${EXP}"
} | tee -a "${LOG}"

exit "${status}"
