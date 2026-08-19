#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
TEACHER="${TEACHER:-/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth}"
RESUME="${RESUME:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30943}"
START_EPOCH="${START_EPOCH:-3}"
EPOCHS="${EPOCHS:-4}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-1e-5}"
QUANT_LR_MULTIPLIER="${QUANT_LR_MULTIPLIER:-4}"
AOQ_EXPLORE_SCALE_RATIO="${AOQ_EXPLORE_SCALE_RATIO:-0.90}"
AOQ_EXPLORE_THRESHOLD_RATIO="${AOQ_EXPLORE_THRESHOLD_RATIO:-}"
AOQ_EXPLORE_SELECTIVE_MARGIN="${AOQ_EXPLORE_SELECTIVE_MARGIN:-0.08}"
AOQ_EXPLORE_QUALITY_MODE="${AOQ_EXPLORE_QUALITY_MODE:-none}"
AOQ_EXPLORE_QUALITY_LAYERS="${AOQ_EXPLORE_QUALITY_LAYERS:-}"
AOQ_EXPLORE_QUALITY_START_UPDATE="${AOQ_EXPLORE_QUALITY_START_UPDATE:-0}"
AOQ_EXPLORE_QUALITY_MIN_FRAC="${AOQ_EXPLORE_QUALITY_MIN_FRAC:-0}"
AOQ_EXPLORE_ANCHOR_CHECKPOINT="${AOQ_EXPLORE_ANCHOR_CHECKPOINT:-}"
AOQ_EXPLORE_LAYERS="${AOQ_EXPLORE_LAYERS:-features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2}"
AOQ_EXPLORE_LAYER_RATIOS="${AOQ_EXPLORE_LAYER_RATIOS:-}"
AOQ_EXPLORE_START_UPDATE="${AOQ_EXPLORE_START_UPDATE:-0}"
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-1800}"
AOQ_EXPLORE_UPDATE_SCHEDULE="${AOQ_EXPLORE_UPDATE_SCHEDULE:-}"
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
SAVE_STEP_CHECKPOINTS="${SAVE_STEP_CHECKPOINTS:-0}"
SAVE_INITIAL_STEP_CHECKPOINT="${SAVE_INITIAL_STEP_CHECKPOINT:-0}"
STEP_CHECKPOINT_INTERVAL="${STEP_CHECKPOINT_INTERVAL:-0}"
STEP_CHECKPOINT_WARMUP_UPDATES="${STEP_CHECKPOINT_WARMUP_UPDATES:-0}"
MAX_STEP_CHECKPOINTS_TO_SAVE="${MAX_STEP_CHECKPOINTS_TO_SAVE:-0}"
TRAINABLE_POLICY="${TRAINABLE_POLICY:-}"
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS="${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS:-}"
TRAINABLE_POLICY_UPDATE_OVERRIDES="${TRAINABLE_POLICY_UPDATE_OVERRIDES:-}"
TRAINABLE_POLICY_UPDATE_MODE="${TRAINABLE_POLICY_UPDATE_MODE:-}"
TRAINABLE_POLICY_GRAD_DAMP="${TRAINABLE_POLICY_GRAD_DAMP:-}"
TEACHER_FEATURE_OUTPUT_WEIGHT="${TEACHER_FEATURE_OUTPUT_WEIGHT:-}"
TEACHER_FEATURE_OUTPUT_LAYERS="${TEACHER_FEATURE_OUTPUT_LAYERS:-}"
TEACHER_FEATURE_OUTPUT_LOSS="${TEACHER_FEATURE_OUTPUT_LOSS:-norm_mse}"
TEACHER_CONFIDENCE_BAND_KD_WEIGHT="${TEACHER_CONFIDENCE_BAND_KD_WEIGHT:-}"
TEACHER_CONFIDENCE_BAND_KD_LOW="${TEACHER_CONFIDENCE_BAND_KD_LOW:-0.2}"
TEACHER_CONFIDENCE_BAND_KD_HIGH="${TEACHER_CONFIDENCE_BAND_KD_HIGH:-0.6}"
TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE="${TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE:-2.75}"
REF_CONFIDENCE_BAND_KD_WEIGHT="${REF_CONFIDENCE_BAND_KD_WEIGHT:-}"
REF_CONFIDENCE_BAND_KD_LOW="${REF_CONFIDENCE_BAND_KD_LOW:-0.2}"
REF_CONFIDENCE_BAND_KD_HIGH="${REF_CONFIDENCE_BAND_KD_HIGH:-0.6}"
REF_CONFIDENCE_BAND_KD_TEMPERATURE="${REF_CONFIDENCE_BAND_KD_TEMPERATURE:-2.75}"
REF_CONFIDENCE_BAND_KD_CHECKPOINT="${REF_CONFIDENCE_BAND_KD_CHECKPOINT:-}"
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT="${LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT:-}"
LOCAL_REF_CONFIDENCE_BAND_KD_LOW="${LOCAL_REF_CONFIDENCE_BAND_KD_LOW:-0.2}"
LOCAL_REF_CONFIDENCE_BAND_KD_HIGH="${LOCAL_REF_CONFIDENCE_BAND_KD_HIGH:-0.6}"
LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE="${LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE:-2.75}"
LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT="${LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT:-}"
SELECTIVE_BIN_ANCHOR_WEIGHT="${SELECTIVE_BIN_ANCHOR_WEIGHT:-}"
SELECTIVE_BIN_ANCHOR_LAYERS="${SELECTIVE_BIN_ANCHOR_LAYERS:-}"
SELECTIVE_BIN_ANCHOR_CAPTURE_UPDATE="${SELECTIVE_BIN_ANCHOR_CAPTURE_UPDATE:-}"
SELECTIVE_BIN_ANCHOR_END_UPDATE="${SELECTIVE_BIN_ANCHOR_END_UPDATE:-}"
SELECTIVE_BIN_ANCHOR_MARGIN="${SELECTIVE_BIN_ANCHOR_MARGIN:-}"
CANDIDATE_BIN_ANCHOR_WEIGHT="${CANDIDATE_BIN_ANCHOR_WEIGHT:-}"
CANDIDATE_BIN_ANCHOR_LAYERS="${CANDIDATE_BIN_ANCHOR_LAYERS:-}"
CANDIDATE_BIN_ANCHOR_CAPTURE_UPDATE="${CANDIDATE_BIN_ANCHOR_CAPTURE_UPDATE:-}"
CANDIDATE_BIN_ANCHOR_END_UPDATE="${CANDIDATE_BIN_ANCHOR_END_UPDATE:-}"
CANDIDATE_BIN_ANCHOR_SOURCE_CHECKPOINT="${CANDIDATE_BIN_ANCHOR_SOURCE_CHECKPOINT:-}"
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
  --resume "${RESUME}" --no-resume-opt --start-epoch "${START_EPOCH}"
  --epochs "${EPOCHS}" --scheduler-epochs "${SCHEDULER_EPOCHS}" --batch-size "${BATCH_SIZE}" --workers "${WORKERS}"
  --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay 0.0
  --quant-lr-multiplier "${QUANT_LR_MULTIPLIER}"
  --epoch-checkpoint-interval 1 --checkpoint-hist 5
  --wbits 4 --abits 4 --wq-mode lsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized
  --amp --amp-dtype bf16
  --aoq-explore-scale-ratio "${AOQ_EXPLORE_SCALE_RATIO}"
  --aoq-explore-selective-margin "${AOQ_EXPLORE_SELECTIVE_MARGIN}"
  --aoq-explore-quality-mode "${AOQ_EXPLORE_QUALITY_MODE}"
  --aoq-explore-quality-layers "${AOQ_EXPLORE_QUALITY_LAYERS}"
  --aoq-explore-quality-start-update "${AOQ_EXPLORE_QUALITY_START_UPDATE}"
  --aoq-explore-quality-min-frac "${AOQ_EXPLORE_QUALITY_MIN_FRAC}"
  --aoq-explore-layers "${AOQ_EXPLORE_LAYERS}"
  --aoq-explore-start-update "${AOQ_EXPLORE_START_UPDATE}"
  --aoq-explore-end-update "${AOQ_EXPLORE_END_UPDATE}"
  --extra-arg=--static-graph
  --extra-arg=--smoothing --extra-arg=0.0
  --extra-arg=--mixup --extra-arg=0.0
  --extra-arg=--cutmix --extra-arg=0.0
  --extra-arg=--aa --extra-arg=none
  --extra-arg=--color-jitter --extra-arg=0.0
  --extra-arg=--reprob --extra-arg=0.0
  --extra-arg=--log-interval --extra-arg=50
  --extra-arg=--seed --extra-arg=42
)

if [[ -n "${AOQ_EXPLORE_THRESHOLD_RATIO}" ]]; then
  cmd+=(--aoq-explore-threshold-ratio "${AOQ_EXPLORE_THRESHOLD_RATIO}")
fi
if [[ -n "${AOQ_EXPLORE_LAYER_RATIOS}" ]]; then
  cmd+=(--aoq-explore-layer-ratios "${AOQ_EXPLORE_LAYER_RATIOS}")
fi
if [[ -n "${AOQ_EXPLORE_ANCHOR_CHECKPOINT}" ]]; then
  cmd+=(--aoq-explore-anchor-checkpoint "${AOQ_EXPLORE_ANCHOR_CHECKPOINT}")
fi
if [[ -n "${AOQ_EXPLORE_UPDATE_SCHEDULE}" ]]; then
  cmd+=(--aoq-explore-update-schedule "${AOQ_EXPLORE_UPDATE_SCHEDULE}")
fi
if [[ -n "${MAX_TRAIN_UPDATES}" && "${MAX_TRAIN_UPDATES}" != "0" ]]; then
  cmd+=(--extra-arg=--max_train_updates --extra-arg="${MAX_TRAIN_UPDATES}")
  if [[ "${SKIP_VALIDATE}" == "1" ]]; then
    cmd+=(--extra-arg=--skip_validate)
  fi
fi
if [[ "${SAVE_STEP_CHECKPOINTS}" == "1" ]]; then
  cmd+=(--extra-arg=--save_step_checkpoints)
  if [[ "${SAVE_INITIAL_STEP_CHECKPOINT}" == "1" ]]; then
    cmd+=(--extra-arg=--save_initial_step_checkpoint)
  fi
  if [[ -n "${STEP_CHECKPOINT_INTERVAL}" && "${STEP_CHECKPOINT_INTERVAL}" != "0" ]]; then
    cmd+=(--extra-arg=--step_checkpoint_interval --extra-arg="${STEP_CHECKPOINT_INTERVAL}")
  fi
  if [[ -n "${STEP_CHECKPOINT_WARMUP_UPDATES}" && "${STEP_CHECKPOINT_WARMUP_UPDATES}" != "0" ]]; then
    cmd+=(--extra-arg=--step_checkpoint_warmup_updates --extra-arg="${STEP_CHECKPOINT_WARMUP_UPDATES}")
  fi
  if [[ -n "${MAX_STEP_CHECKPOINTS_TO_SAVE}" && "${MAX_STEP_CHECKPOINTS_TO_SAVE}" != "0" ]]; then
    cmd+=(--extra-arg=--max_step_checkpoints_to_save --extra-arg="${MAX_STEP_CHECKPOINTS_TO_SAVE}")
  fi
fi
if [[ -n "${TRAINABLE_POLICY}" ]]; then
  cmd+=(--trainable-policy "${TRAINABLE_POLICY}")
fi
if [[ -n "${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS}" ]]; then
  cmd+=(--trainable-policy-freeze-act-except-layers "${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS}")
fi
if [[ -n "${TRAINABLE_POLICY_UPDATE_OVERRIDES}" ]]; then
  cmd+=(--trainable-policy-update-overrides "${TRAINABLE_POLICY_UPDATE_OVERRIDES}")
fi
if [[ -n "${TRAINABLE_POLICY_UPDATE_MODE}" ]]; then
  cmd+=(--trainable-policy-update-mode "${TRAINABLE_POLICY_UPDATE_MODE}")
fi
if [[ -n "${TRAINABLE_POLICY_GRAD_DAMP}" ]]; then
  cmd+=(--trainable-policy-grad-damp "${TRAINABLE_POLICY_GRAD_DAMP}")
fi
if [[ -n "${TEACHER_FEATURE_OUTPUT_WEIGHT}" ]]; then
  cmd+=(--teacher-feature-output-weight "${TEACHER_FEATURE_OUTPUT_WEIGHT}")
fi
if [[ -n "${TEACHER_FEATURE_OUTPUT_LAYERS}" ]]; then
  cmd+=(--teacher-feature-output-layers "${TEACHER_FEATURE_OUTPUT_LAYERS}")
  cmd+=(--teacher-feature-output-loss "${TEACHER_FEATURE_OUTPUT_LOSS}")
fi
if [[ -n "${TEACHER_CONFIDENCE_BAND_KD_WEIGHT}" ]]; then
  cmd+=(--teacher-confidence-band-kd-weight "${TEACHER_CONFIDENCE_BAND_KD_WEIGHT}")
  cmd+=(--teacher-confidence-band-kd-low "${TEACHER_CONFIDENCE_BAND_KD_LOW}")
  cmd+=(--teacher-confidence-band-kd-high "${TEACHER_CONFIDENCE_BAND_KD_HIGH}")
  cmd+=(--teacher-confidence-band-kd-temperature "${TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE}")
fi
if [[ -n "${REF_CONFIDENCE_BAND_KD_WEIGHT}" ]]; then
  cmd+=(--ref-confidence-band-kd-weight "${REF_CONFIDENCE_BAND_KD_WEIGHT}")
  cmd+=(--ref-confidence-band-kd-low "${REF_CONFIDENCE_BAND_KD_LOW}")
  cmd+=(--ref-confidence-band-kd-high "${REF_CONFIDENCE_BAND_KD_HIGH}")
  cmd+=(--ref-confidence-band-kd-temperature "${REF_CONFIDENCE_BAND_KD_TEMPERATURE}")
  if [[ -n "${REF_CONFIDENCE_BAND_KD_CHECKPOINT}" ]]; then
    cmd+=(--ref-confidence-band-kd-checkpoint "${REF_CONFIDENCE_BAND_KD_CHECKPOINT}")
  fi
fi
if [[ -n "${LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT}" ]]; then
  cmd+=(--local-ref-confidence-band-kd-weight "${LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT}")
  cmd+=(--local-ref-confidence-band-kd-low "${LOCAL_REF_CONFIDENCE_BAND_KD_LOW}")
  cmd+=(--local-ref-confidence-band-kd-high "${LOCAL_REF_CONFIDENCE_BAND_KD_HIGH}")
  cmd+=(--local-ref-confidence-band-kd-temperature "${LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE}")
  if [[ -n "${LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT}" ]]; then
    cmd+=(--local-ref-confidence-band-kd-checkpoint "${LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT}")
  fi
fi
if [[ -n "${SELECTIVE_BIN_ANCHOR_WEIGHT}" ]]; then
  cmd+=(--selective-bin-anchor-weight "${SELECTIVE_BIN_ANCHOR_WEIGHT}")
fi
if [[ -n "${SELECTIVE_BIN_ANCHOR_LAYERS}" ]]; then
  cmd+=(--selective-bin-anchor-layers "${SELECTIVE_BIN_ANCHOR_LAYERS}")
fi
if [[ -n "${SELECTIVE_BIN_ANCHOR_CAPTURE_UPDATE}" ]]; then
  cmd+=(--selective-bin-anchor-capture-update "${SELECTIVE_BIN_ANCHOR_CAPTURE_UPDATE}")
fi
if [[ -n "${SELECTIVE_BIN_ANCHOR_END_UPDATE}" ]]; then
  cmd+=(--selective-bin-anchor-end-update "${SELECTIVE_BIN_ANCHOR_END_UPDATE}")
fi
if [[ -n "${SELECTIVE_BIN_ANCHOR_MARGIN}" ]]; then
  cmd+=(--selective-bin-anchor-margin "${SELECTIVE_BIN_ANCHOR_MARGIN}")
fi
if [[ -n "${CANDIDATE_BIN_ANCHOR_WEIGHT}" ]]; then
  cmd+=(--candidate-bin-anchor-weight "${CANDIDATE_BIN_ANCHOR_WEIGHT}")
fi
if [[ -n "${CANDIDATE_BIN_ANCHOR_LAYERS}" ]]; then
  cmd+=(--candidate-bin-anchor-layers "${CANDIDATE_BIN_ANCHOR_LAYERS}")
fi
if [[ -n "${CANDIDATE_BIN_ANCHOR_CAPTURE_UPDATE}" ]]; then
  cmd+=(--candidate-bin-anchor-capture-update "${CANDIDATE_BIN_ANCHOR_CAPTURE_UPDATE}")
fi
if [[ -n "${CANDIDATE_BIN_ANCHOR_END_UPDATE}" ]]; then
  cmd+=(--candidate-bin-anchor-end-update "${CANDIDATE_BIN_ANCHOR_END_UPDATE}")
fi
if [[ -n "${CANDIDATE_BIN_ANCHOR_SOURCE_CHECKPOINT}" ]]; then
  cmd+=(--candidate-bin-anchor-source-checkpoint "${CANDIDATE_BIN_ANCHOR_SOURCE_CHECKPOINT}")
fi

{
  echo "===== Clean LSQ no-QKR LSQ-AOQ090 selective-margin08 gate $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA}"
  echo "OUT=${OUT}/${EXP}"
  echo "RESUME=${RESUME}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "START_EPOCH=${START_EPOCH}"
  echo "EPOCHS=${EPOCHS}"
  echo "SCHEDULER_EPOCHS=${SCHEDULER_EPOCHS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "WORKERS=${WORKERS}"
  echo "LR=${LR}"
  echo "MIN_LR=${MIN_LR}"
  echo "QUANT_LR_MULTIPLIER=${QUANT_LR_MULTIPLIER}"
  echo "WQ_MODE=lsq"
  echo "AQ_MODE=lsq"
  echo "QK_REPARAM=0"
  echo "AOQ_EXPLORE_SCALE_RATIO=${AOQ_EXPLORE_SCALE_RATIO}"
  echo "AOQ_EXPLORE_THRESHOLD_RATIO=${AOQ_EXPLORE_THRESHOLD_RATIO}"
  echo "AOQ_EXPLORE_SELECTIVE_MARGIN=${AOQ_EXPLORE_SELECTIVE_MARGIN}"
  echo "AOQ_EXPLORE_QUALITY_MODE=${AOQ_EXPLORE_QUALITY_MODE}"
  echo "AOQ_EXPLORE_QUALITY_LAYERS=${AOQ_EXPLORE_QUALITY_LAYERS}"
  echo "AOQ_EXPLORE_QUALITY_START_UPDATE=${AOQ_EXPLORE_QUALITY_START_UPDATE}"
  echo "AOQ_EXPLORE_QUALITY_MIN_FRAC=${AOQ_EXPLORE_QUALITY_MIN_FRAC}"
  echo "AOQ_EXPLORE_ANCHOR_CHECKPOINT=${AOQ_EXPLORE_ANCHOR_CHECKPOINT}"
  echo "AOQ_EXPLORE_LAYERS=${AOQ_EXPLORE_LAYERS}"
  echo "AOQ_EXPLORE_LAYER_RATIOS=${AOQ_EXPLORE_LAYER_RATIOS}"
  echo "AOQ_EXPLORE_START_UPDATE=${AOQ_EXPLORE_START_UPDATE}"
  echo "AOQ_EXPLORE_END_UPDATE=${AOQ_EXPLORE_END_UPDATE}"
  echo "AOQ_EXPLORE_UPDATE_SCHEDULE=${AOQ_EXPLORE_UPDATE_SCHEDULE}"
  echo "SAVE_STEP_CHECKPOINTS=${SAVE_STEP_CHECKPOINTS}"
  echo "SAVE_INITIAL_STEP_CHECKPOINT=${SAVE_INITIAL_STEP_CHECKPOINT}"
  echo "STEP_CHECKPOINT_INTERVAL=${STEP_CHECKPOINT_INTERVAL}"
  echo "STEP_CHECKPOINT_WARMUP_UPDATES=${STEP_CHECKPOINT_WARMUP_UPDATES}"
  echo "MAX_STEP_CHECKPOINTS_TO_SAVE=${MAX_STEP_CHECKPOINTS_TO_SAVE}"
  echo "TRAINABLE_POLICY=${TRAINABLE_POLICY}"
  echo "TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS}"
  echo "TRAINABLE_POLICY_UPDATE_OVERRIDES=${TRAINABLE_POLICY_UPDATE_OVERRIDES}"
  echo "TRAINABLE_POLICY_UPDATE_MODE=${TRAINABLE_POLICY_UPDATE_MODE}"
  echo "TRAINABLE_POLICY_GRAD_DAMP=${TRAINABLE_POLICY_GRAD_DAMP}"
  echo "TEACHER_FEATURE_OUTPUT_WEIGHT=${TEACHER_FEATURE_OUTPUT_WEIGHT}"
  echo "TEACHER_FEATURE_OUTPUT_LAYERS=${TEACHER_FEATURE_OUTPUT_LAYERS}"
  echo "TEACHER_FEATURE_OUTPUT_LOSS=${TEACHER_FEATURE_OUTPUT_LOSS}"
  echo "TEACHER_CONFIDENCE_BAND_KD_WEIGHT=${TEACHER_CONFIDENCE_BAND_KD_WEIGHT}"
  echo "TEACHER_CONFIDENCE_BAND_KD_LOW=${TEACHER_CONFIDENCE_BAND_KD_LOW}"
  echo "TEACHER_CONFIDENCE_BAND_KD_HIGH=${TEACHER_CONFIDENCE_BAND_KD_HIGH}"
  echo "TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE=${TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE}"
  echo "REF_CONFIDENCE_BAND_KD_WEIGHT=${REF_CONFIDENCE_BAND_KD_WEIGHT}"
  echo "REF_CONFIDENCE_BAND_KD_LOW=${REF_CONFIDENCE_BAND_KD_LOW}"
  echo "REF_CONFIDENCE_BAND_KD_HIGH=${REF_CONFIDENCE_BAND_KD_HIGH}"
  echo "REF_CONFIDENCE_BAND_KD_TEMPERATURE=${REF_CONFIDENCE_BAND_KD_TEMPERATURE}"
  echo "REF_CONFIDENCE_BAND_KD_CHECKPOINT=${REF_CONFIDENCE_BAND_KD_CHECKPOINT}"
  echo "LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT=${LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT}"
  echo "LOCAL_REF_CONFIDENCE_BAND_KD_LOW=${LOCAL_REF_CONFIDENCE_BAND_KD_LOW}"
  echo "LOCAL_REF_CONFIDENCE_BAND_KD_HIGH=${LOCAL_REF_CONFIDENCE_BAND_KD_HIGH}"
  echo "LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE=${LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE}"
  echo "LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT=${LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT}"
  echo "SELECTIVE_BIN_ANCHOR_WEIGHT=${SELECTIVE_BIN_ANCHOR_WEIGHT}"
  echo "SELECTIVE_BIN_ANCHOR_LAYERS=${SELECTIVE_BIN_ANCHOR_LAYERS}"
  echo "SELECTIVE_BIN_ANCHOR_CAPTURE_UPDATE=${SELECTIVE_BIN_ANCHOR_CAPTURE_UPDATE}"
  echo "SELECTIVE_BIN_ANCHOR_END_UPDATE=${SELECTIVE_BIN_ANCHOR_END_UPDATE}"
  echo "SELECTIVE_BIN_ANCHOR_MARGIN=${SELECTIVE_BIN_ANCHOR_MARGIN}"
  echo "CANDIDATE_BIN_ANCHOR_WEIGHT=${CANDIDATE_BIN_ANCHOR_WEIGHT}"
  echo "CANDIDATE_BIN_ANCHOR_LAYERS=${CANDIDATE_BIN_ANCHOR_LAYERS}"
  echo "CANDIDATE_BIN_ANCHOR_CAPTURE_UPDATE=${CANDIDATE_BIN_ANCHOR_CAPTURE_UPDATE}"
  echo "CANDIDATE_BIN_ANCHOR_END_UPDATE=${CANDIDATE_BIN_ANCHOR_END_UPDATE}"
  echo "CANDIDATE_BIN_ANCHOR_SOURCE_CHECKPOINT=${CANDIDATE_BIN_ANCHOR_SOURCE_CHECKPOINT}"
  echo "MAX_TRAIN_UPDATES=${MAX_TRAIN_UPDATES}"
  echo "SKIP_VALIDATE=${SKIP_VALIDATE}"
  ls -lh "${RESUME}"
  python3 - <<'PY'
from pathlib import Path
root = Path('/tmp/imagenet1k_full_parquet/data')
print('train_shards=' + str(len(list(root.glob('train-*.parquet')))))
print('validation_shards=' + str(len(list(root.glob('validation-*.parquet')))))
PY
  git -C "${QATS}" rev-parse --short HEAD || true
  test -e /dev/nvidia0 && echo gpu-device-present || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  printf 'command='
  printf '%q ' "${cmd[@]}"
  echo
} | tee "${LOG}"

"${cmd[@]}" 2>&1 | tee -a "${LOG}"

echo "wall_seconds=${SECONDS}" | tee -a "${LOG}"
echo "train_log=${LOG}" | tee -a "${LOG}"
echo "output=${OUT}/${EXP}" | tee -a "${LOG}"
