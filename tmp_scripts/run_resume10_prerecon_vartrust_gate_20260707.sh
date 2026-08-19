#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
EXP="${EXP:-recipe_resume10_prerecon_vartrust_gate_20260707}"
DATA="${DATA:-/tmp/imagenet1k_full_parquet}"
TEACHER="${TEACHER:-/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth}"
RESUME="${RESUME:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe10ep_e_featnorm_aug_fixed_qkr_softkd_t275_rebuild_20260706/checkpoint-10.pth.tar}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30597}"
START_EPOCH="${START_EPOCH:-0}"
EPOCHS="${EPOCHS:-3}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-3}"
LR="${LR:-1.5e-5}"
MIN_LR="${MIN_LR:-5e-6}"
QUANT_LR_MULTIPLIER="${QUANT_LR_MULTIPLIER:-2}"
PRE_QAT_FEATURE_RECON_UPDATES="${PRE_QAT_FEATURE_RECON_UPDATES:-100}"
PRE_QAT_FEATURE_RECON_LAYERS="${PRE_QAT_FEATURE_RECON_LAYERS:-features.5.5,features.7.1}"
PRE_QAT_FEATURE_RECON_POLICY="${PRE_QAT_FEATURE_RECON_POLICY:-quant}"
PRE_QAT_ACT_MSE_CALIB_BATCHES="${PRE_QAT_ACT_MSE_CALIB_BATCHES:-0}"
PRE_QAT_ACT_MSE_CALIB_LAYERS="${PRE_QAT_ACT_MSE_CALIB_LAYERS:-}"
PRE_QAT_ACT_MSE_CALIB_QUANTIZERS="${PRE_QAT_ACT_MSE_CALIB_QUANTIZERS:-}"
PRE_QAT_ACT_MSE_CALIB_GRID="${PRE_QAT_ACT_MSE_CALIB_GRID:-0.85,1.15,13}"
PRE_QAT_ACT_MSE_CALIB_BLEND="${PRE_QAT_ACT_MSE_CALIB_BLEND:-0.35}"
VARIATION_TRUST_WEIGHT="${VARIATION_TRUST_WEIGHT:-0.001}"
VARIATION_TRUST_LAYERS="${VARIATION_TRUST_LAYERS:-}"
VARIATION_TRUST_LATE_LAYERS="${VARIATION_TRUST_LATE_LAYERS:-features.5.5,features.7.1}"
VARIATION_TRUST_LATE_MULTIPLIER="${VARIATION_TRUST_LATE_MULTIPLIER:-0.25}"
VARIATION_TRUST_EARLY_LAYERS="${VARIATION_TRUST_EARLY_LAYERS:-features.0.0,features.1.0,features.1.1}"
VARIATION_TRUST_EARLY_MULTIPLIER="${VARIATION_TRUST_EARLY_MULTIPLIER:-2.0}"
VARIATION_TRUST_SOFTMAX_MULTIPLIER="${VARIATION_TRUST_SOFTMAX_MULTIPLIER:-2.0}"
VARIATION_TRUST_MOVE_V_MULTIPLIER="${VARIATION_TRUST_MOVE_V_MULTIPLIER:-1.5}"
VARIATION_TRUST_PROJ_MOVE_MULTIPLIER="${VARIATION_TRUST_PROJ_MOVE_MULTIPLIER:-1.25}"
VARIATION_TRUST_START_UPDATE="${VARIATION_TRUST_START_UPDATE:-0}"
DELTA_DIRECTION_ANCHOR_WEIGHT="${DELTA_DIRECTION_ANCHOR_WEIGHT:-0}"
DELTA_DIRECTION_ANCHOR_BASE_CHECKPOINT="${DELTA_DIRECTION_ANCHOR_BASE_CHECKPOINT:-}"
DELTA_DIRECTION_ANCHOR_TARGET_CHECKPOINT="${DELTA_DIRECTION_ANCHOR_TARGET_CHECKPOINT:-}"
DELTA_DIRECTION_ANCHOR_PARAMS="${DELTA_DIRECTION_ANCHOR_PARAMS:-}"
DELTA_DIRECTION_ANCHOR_START_UPDATE="${DELTA_DIRECTION_ANCHOR_START_UPDATE:-0}"
BIN_REG_WEIGHT="${BIN_REG_WEIGHT:-0}"
BIN_REG_VARIANCE_WEIGHT="${BIN_REG_VARIANCE_WEIGHT:-1.0}"
BIN_REG_LAYERS="${BIN_REG_LAYERS:-}"
BIN_REG_ATTN_ONLY="${BIN_REG_ATTN_ONLY:-0}"
BIN_REG_START_UPDATE="${BIN_REG_START_UPDATE:-}"
BIN_REG_END_UPDATE="${BIN_REG_END_UPDATE:-}"
AOQ_EXPLORE_SCALE_RATIO="${AOQ_EXPLORE_SCALE_RATIO:-1.0}"
AOQ_EXPLORE_LAYERS="${AOQ_EXPLORE_LAYERS:-}"
AOQ_EXPLORE_START_UPDATE="${AOQ_EXPLORE_START_UPDATE:-0}"
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-0}"
TEACHER_FEATURE_OUTPUT_WEIGHT="${TEACHER_FEATURE_OUTPUT_WEIGHT:-0.003}"
TEACHER_ATTN_OUTPUT_WEIGHT="${TEACHER_ATTN_OUTPUT_WEIGHT:-0}"
TEACHER_ATTN_OUTPUT_LAYERS="${TEACHER_ATTN_OUTPUT_LAYERS:-10,11}"
TEACHER_ATTN_OUTPUT_WARMUP_EPOCHS="${TEACHER_ATTN_OUTPUT_WARMUP_EPOCHS:-0}"
TEACHER_CONFIDENCE_BAND_KD_WEIGHT="${TEACHER_CONFIDENCE_BAND_KD_WEIGHT:-0}"
TEACHER_CONFIDENCE_BAND_KD_LOW="${TEACHER_CONFIDENCE_BAND_KD_LOW:-0.2}"
TEACHER_CONFIDENCE_BAND_KD_HIGH="${TEACHER_CONFIDENCE_BAND_KD_HIGH:-0.6}"
TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE="${TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE:-2.75}"
REF_CONFIDENCE_BAND_KD_WEIGHT="${REF_CONFIDENCE_BAND_KD_WEIGHT:-0}"
REF_CONFIDENCE_BAND_KD_LOW="${REF_CONFIDENCE_BAND_KD_LOW:-0.2}"
REF_CONFIDENCE_BAND_KD_HIGH="${REF_CONFIDENCE_BAND_KD_HIGH:-0.6}"
REF_CONFIDENCE_BAND_KD_TEMPERATURE="${REF_CONFIDENCE_BAND_KD_TEMPERATURE:-2.75}"
REF_CONFIDENCE_BAND_KD_CHECKPOINT="${REF_CONFIDENCE_BAND_KD_CHECKPOINT:-}"
LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT="${LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT:-0}"
LOCAL_REF_CONFIDENCE_BAND_KD_LOW="${LOCAL_REF_CONFIDENCE_BAND_KD_LOW:-0.2}"
LOCAL_REF_CONFIDENCE_BAND_KD_HIGH="${LOCAL_REF_CONFIDENCE_BAND_KD_HIGH:-0.4}"
LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE="${LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE:-2.75}"
LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT="${LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT:-}"
CLASS_PROTECT_REF_KL_WEIGHT="${CLASS_PROTECT_REF_KL_WEIGHT:-0}"
CLASS_PROTECT_REF_KL_CLASSES="${CLASS_PROTECT_REF_KL_CLASSES:-}"
CLASS_PROTECT_REF_KL_TEMPERATURE="${CLASS_PROTECT_REF_KL_TEMPERATURE:-2.75}"
CLASS_PROTECT_REF_KL_CHECKPOINT="${CLASS_PROTECT_REF_KL_CHECKPOINT:-}"
QUANT_ONLY_START_EPOCH="${QUANT_ONLY_START_EPOCH:-}"
TRAINABLE_POLICY="${TRAINABLE_POLICY:-}"
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS="${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS:-}"
TRAINABLE_POLICY_UPDATE_OVERRIDES="${TRAINABLE_POLICY_UPDATE_OVERRIDES:-}"
TRAINABLE_POLICY_UPDATE_MODE="${TRAINABLE_POLICY_UPDATE_MODE:-}"
TRAINABLE_POLICY_GRAD_DAMP="${TRAINABLE_POLICY_GRAD_DAMP:-}"
SAVE_STEP_CHECKPOINTS="${SAVE_STEP_CHECKPOINTS:-0}"
SAVE_INITIAL_STEP_CHECKPOINT="${SAVE_INITIAL_STEP_CHECKPOINT:-0}"
STEP_CHECKPOINT_INTERVAL="${STEP_CHECKPOINT_INTERVAL:-}"
STEP_CHECKPOINT_WARMUP_UPDATES="${STEP_CHECKPOINT_WARMUP_UPDATES:-}"
MAX_STEP_CHECKPOINTS_TO_SAVE="${MAX_STEP_CHECKPOINTS_TO_SAVE:-}"
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-}"
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
  --epochs "${EPOCHS}" --scheduler-epochs "${SCHEDULER_EPOCHS}" --batch-size 64 --workers 8
  --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay 0.0
  --quant-lr-multiplier "${QUANT_LR_MULTIPLIER}"
  --pre-qat-feature-recon-updates "${PRE_QAT_FEATURE_RECON_UPDATES}"
  --pre-qat-feature-recon-layers "${PRE_QAT_FEATURE_RECON_LAYERS}"
  --pre-qat-feature-recon-policy "${PRE_QAT_FEATURE_RECON_POLICY}"
  --pre-qat-act-mse-calib-batches "${PRE_QAT_ACT_MSE_CALIB_BATCHES}"
  --variation-trust-weight "${VARIATION_TRUST_WEIGHT}"
  --variation-trust-late-layers "${VARIATION_TRUST_LATE_LAYERS}"
  --variation-trust-late-multiplier "${VARIATION_TRUST_LATE_MULTIPLIER}"
  --variation-trust-early-layers "${VARIATION_TRUST_EARLY_LAYERS}"
  --variation-trust-early-multiplier "${VARIATION_TRUST_EARLY_MULTIPLIER}"
  --variation-trust-softmax-multiplier "${VARIATION_TRUST_SOFTMAX_MULTIPLIER}"
  --variation-trust-move-v-multiplier "${VARIATION_TRUST_MOVE_V_MULTIPLIER}"
  --variation-trust-proj-move-multiplier "${VARIATION_TRUST_PROJ_MOVE_MULTIPLIER}"
  --variation-trust-start-update "${VARIATION_TRUST_START_UPDATE}"
  --epoch-checkpoint-interval 1 --checkpoint-hist 10
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75
  --quantized --qk-reparam --qk-reparam-type 0
  --amp --amp-dtype bf16
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

if [[ -n "${VARIATION_TRUST_LAYERS}" ]]; then
  cmd+=(--variation-trust-layers "${VARIATION_TRUST_LAYERS}")
fi
if [[ "${PRE_QAT_ACT_MSE_CALIB_BATCHES}" != "0" ]]; then
  if [[ -n "${PRE_QAT_ACT_MSE_CALIB_LAYERS}" ]]; then
    cmd+=(--pre-qat-act-mse-calib-layers "${PRE_QAT_ACT_MSE_CALIB_LAYERS}")
  fi
  if [[ -n "${PRE_QAT_ACT_MSE_CALIB_QUANTIZERS}" ]]; then
    cmd+=(--pre-qat-act-mse-calib-quantizers "${PRE_QAT_ACT_MSE_CALIB_QUANTIZERS}")
  fi
  cmd+=(--pre-qat-act-mse-calib-grid "${PRE_QAT_ACT_MSE_CALIB_GRID}")
  cmd+=(--pre-qat-act-mse-calib-blend "${PRE_QAT_ACT_MSE_CALIB_BLEND}")
fi
if [[ -n "${DELTA_DIRECTION_ANCHOR_WEIGHT}" && "${DELTA_DIRECTION_ANCHOR_WEIGHT}" != "0" ]]; then
  cmd+=(--delta-direction-anchor-weight "${DELTA_DIRECTION_ANCHOR_WEIGHT}")
  cmd+=(--delta-direction-anchor-base-checkpoint "${DELTA_DIRECTION_ANCHOR_BASE_CHECKPOINT}")
  cmd+=(--delta-direction-anchor-target-checkpoint "${DELTA_DIRECTION_ANCHOR_TARGET_CHECKPOINT}")
  cmd+=(--delta-direction-anchor-params "${DELTA_DIRECTION_ANCHOR_PARAMS}")
  cmd+=(--delta-direction-anchor-start-update "${DELTA_DIRECTION_ANCHOR_START_UPDATE}")
fi
if [[ -n "${BIN_REG_WEIGHT}" && "${BIN_REG_WEIGHT}" != "0" ]]; then
  cmd+=(--bin-reg-weight "${BIN_REG_WEIGHT}")
  cmd+=(--bin-reg-variance-weight "${BIN_REG_VARIANCE_WEIGHT}")
  if [[ -n "${BIN_REG_LAYERS}" ]]; then
    cmd+=(--bin-reg-layers "${BIN_REG_LAYERS}")
  fi
  if [[ "${BIN_REG_ATTN_ONLY}" == "1" ]]; then
    cmd+=(--bin-reg-attn-only)
  fi
  if [[ -n "${BIN_REG_START_UPDATE}" ]]; then
    cmd+=(--bin-reg-start-update "${BIN_REG_START_UPDATE}")
  fi
  if [[ -n "${BIN_REG_END_UPDATE}" ]]; then
    cmd+=(--bin-reg-end-update "${BIN_REG_END_UPDATE}")
  fi
fi
if [[ -n "${AOQ_EXPLORE_SCALE_RATIO}" && "${AOQ_EXPLORE_SCALE_RATIO}" != "1.0" && "${AOQ_EXPLORE_SCALE_RATIO}" != "1" ]]; then
  cmd+=(--aoq-explore-scale-ratio "${AOQ_EXPLORE_SCALE_RATIO}")
  cmd+=(--aoq-explore-start-update "${AOQ_EXPLORE_START_UPDATE}")
  cmd+=(--aoq-explore-end-update "${AOQ_EXPLORE_END_UPDATE}")
  if [[ -n "${AOQ_EXPLORE_LAYERS}" ]]; then
    cmd+=(--aoq-explore-layers "${AOQ_EXPLORE_LAYERS}")
  fi
fi

if [[ -n "${MAX_TRAIN_UPDATES}" && "${MAX_TRAIN_UPDATES}" != "0" ]]; then
  cmd+=(--extra-arg=--max_train_updates --extra-arg="${MAX_TRAIN_UPDATES}" --extra-arg=--skip_validate)
fi

if [[ -n "${TEACHER_FEATURE_OUTPUT_WEIGHT}" && "${TEACHER_FEATURE_OUTPUT_WEIGHT}" != "0" ]]; then
  cmd+=(
    --teacher-feature-output-weight "${TEACHER_FEATURE_OUTPUT_WEIGHT}"
    --teacher-feature-output-layers features.5.5,features.7.1
    --teacher-feature-output-loss norm_mse
  )
fi
if [[ -n "${TEACHER_ATTN_OUTPUT_WEIGHT}" && "${TEACHER_ATTN_OUTPUT_WEIGHT}" != "0" ]]; then
  cmd+=(
    --teacher-attn-output-weight "${TEACHER_ATTN_OUTPUT_WEIGHT}"
    --teacher-attn-output-layers "${TEACHER_ATTN_OUTPUT_LAYERS}"
    --teacher-attn-output-warmup-epochs "${TEACHER_ATTN_OUTPUT_WARMUP_EPOCHS}"
  )
fi
if [[ -n "${TEACHER_CONFIDENCE_BAND_KD_WEIGHT}" && "${TEACHER_CONFIDENCE_BAND_KD_WEIGHT}" != "0" ]]; then
  cmd+=(
    --teacher-confidence-band-kd-weight "${TEACHER_CONFIDENCE_BAND_KD_WEIGHT}"
    --teacher-confidence-band-kd-low "${TEACHER_CONFIDENCE_BAND_KD_LOW}"
    --teacher-confidence-band-kd-high "${TEACHER_CONFIDENCE_BAND_KD_HIGH}"
    --teacher-confidence-band-kd-temperature "${TEACHER_CONFIDENCE_BAND_KD_TEMPERATURE}"
  )
fi
if [[ -n "${REF_CONFIDENCE_BAND_KD_WEIGHT}" && "${REF_CONFIDENCE_BAND_KD_WEIGHT}" != "0" ]]; then
  cmd+=(
    --ref-confidence-band-kd-weight "${REF_CONFIDENCE_BAND_KD_WEIGHT}"
    --ref-confidence-band-kd-low "${REF_CONFIDENCE_BAND_KD_LOW}"
    --ref-confidence-band-kd-high "${REF_CONFIDENCE_BAND_KD_HIGH}"
    --ref-confidence-band-kd-temperature "${REF_CONFIDENCE_BAND_KD_TEMPERATURE}"
  )
  if [[ -n "${REF_CONFIDENCE_BAND_KD_CHECKPOINT}" ]]; then
    cmd+=(--ref-confidence-band-kd-checkpoint "${REF_CONFIDENCE_BAND_KD_CHECKPOINT}")
  fi
fi
if [[ -n "${LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT}" && "${LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT}" != "0" ]]; then
  cmd+=(
    --local-ref-confidence-band-kd-weight "${LOCAL_REF_CONFIDENCE_BAND_KD_WEIGHT}"
    --local-ref-confidence-band-kd-low "${LOCAL_REF_CONFIDENCE_BAND_KD_LOW}"
    --local-ref-confidence-band-kd-high "${LOCAL_REF_CONFIDENCE_BAND_KD_HIGH}"
    --local-ref-confidence-band-kd-temperature "${LOCAL_REF_CONFIDENCE_BAND_KD_TEMPERATURE}"
  )
  if [[ -n "${LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT}" ]]; then
    cmd+=(--local-ref-confidence-band-kd-checkpoint "${LOCAL_REF_CONFIDENCE_BAND_KD_CHECKPOINT}")
  fi
fi
if [[ -n "${CLASS_PROTECT_REF_KL_WEIGHT}" && "${CLASS_PROTECT_REF_KL_WEIGHT}" != "0" ]]; then
  cmd+=(
    --class-protect-ref-kl-weight "${CLASS_PROTECT_REF_KL_WEIGHT}"
    --class-protect-ref-kl-classes "${CLASS_PROTECT_REF_KL_CLASSES}"
    --class-protect-ref-kl-temperature "${CLASS_PROTECT_REF_KL_TEMPERATURE}"
  )
  if [[ -n "${CLASS_PROTECT_REF_KL_CHECKPOINT}" ]]; then
    cmd+=(--class-protect-ref-kl-checkpoint "${CLASS_PROTECT_REF_KL_CHECKPOINT}")
  fi
fi
if [[ -n "${QUANT_ONLY_START_EPOCH}" ]]; then
  cmd+=(--quant-only-start-epoch "${QUANT_ONLY_START_EPOCH}")
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
if [[ "${SAVE_STEP_CHECKPOINTS}" == "1" ]]; then
  cmd+=(--extra-arg=--save_step_checkpoints)
fi
if [[ "${SAVE_INITIAL_STEP_CHECKPOINT}" == "1" ]]; then
  cmd+=(--extra-arg=--save_initial_step_checkpoint)
fi
if [[ -n "${STEP_CHECKPOINT_INTERVAL}" ]]; then
  cmd+=(--extra-arg=--step_checkpoint_interval --extra-arg="${STEP_CHECKPOINT_INTERVAL}")
fi
if [[ -n "${STEP_CHECKPOINT_WARMUP_UPDATES}" ]]; then
  cmd+=(--extra-arg=--step_checkpoint_warmup_updates --extra-arg="${STEP_CHECKPOINT_WARMUP_UPDATES}")
fi
if [[ -n "${MAX_STEP_CHECKPOINTS_TO_SAVE}" ]]; then
  cmd+=(--extra-arg=--max_step_checkpoints_to_save --extra-arg="${MAX_STEP_CHECKPOINTS_TO_SAVE}")
fi

{
  echo "===== Resume10 pre-QAT feature recon + variation trust gate $(date '+%F %T') ====="
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
  echo "LR=${LR}"
  echo "MIN_LR=${MIN_LR}"
  echo "QUANT_LR_MULTIPLIER=${QUANT_LR_MULTIPLIER}"
  echo "PRE_QAT_FEATURE_RECON_UPDATES=${PRE_QAT_FEATURE_RECON_UPDATES}"
  echo "PRE_QAT_FEATURE_RECON_LAYERS=${PRE_QAT_FEATURE_RECON_LAYERS}"
  echo "PRE_QAT_FEATURE_RECON_POLICY=${PRE_QAT_FEATURE_RECON_POLICY}"
  echo "PRE_QAT_ACT_MSE_CALIB_BATCHES=${PRE_QAT_ACT_MSE_CALIB_BATCHES}"
  echo "PRE_QAT_ACT_MSE_CALIB_LAYERS=${PRE_QAT_ACT_MSE_CALIB_LAYERS}"
  echo "PRE_QAT_ACT_MSE_CALIB_QUANTIZERS=${PRE_QAT_ACT_MSE_CALIB_QUANTIZERS}"
  echo "PRE_QAT_ACT_MSE_CALIB_GRID=${PRE_QAT_ACT_MSE_CALIB_GRID}"
  echo "PRE_QAT_ACT_MSE_CALIB_BLEND=${PRE_QAT_ACT_MSE_CALIB_BLEND}"
  echo "VARIATION_TRUST_WEIGHT=${VARIATION_TRUST_WEIGHT}"
  echo "VARIATION_TRUST_LAYERS=${VARIATION_TRUST_LAYERS}"
  echo "VARIATION_TRUST_LATE_LAYERS=${VARIATION_TRUST_LATE_LAYERS}"
  echo "VARIATION_TRUST_LATE_MULTIPLIER=${VARIATION_TRUST_LATE_MULTIPLIER}"
  echo "VARIATION_TRUST_EARLY_LAYERS=${VARIATION_TRUST_EARLY_LAYERS}"
  echo "VARIATION_TRUST_EARLY_MULTIPLIER=${VARIATION_TRUST_EARLY_MULTIPLIER}"
  echo "VARIATION_TRUST_SOFTMAX_MULTIPLIER=${VARIATION_TRUST_SOFTMAX_MULTIPLIER}"
  echo "VARIATION_TRUST_MOVE_V_MULTIPLIER=${VARIATION_TRUST_MOVE_V_MULTIPLIER}"
  echo "VARIATION_TRUST_PROJ_MOVE_MULTIPLIER=${VARIATION_TRUST_PROJ_MOVE_MULTIPLIER}"
  echo "VARIATION_TRUST_START_UPDATE=${VARIATION_TRUST_START_UPDATE}"
  echo "DELTA_DIRECTION_ANCHOR_WEIGHT=${DELTA_DIRECTION_ANCHOR_WEIGHT}"
  echo "DELTA_DIRECTION_ANCHOR_BASE_CHECKPOINT=${DELTA_DIRECTION_ANCHOR_BASE_CHECKPOINT}"
  echo "DELTA_DIRECTION_ANCHOR_TARGET_CHECKPOINT=${DELTA_DIRECTION_ANCHOR_TARGET_CHECKPOINT}"
  echo "DELTA_DIRECTION_ANCHOR_PARAMS=${DELTA_DIRECTION_ANCHOR_PARAMS}"
  echo "DELTA_DIRECTION_ANCHOR_START_UPDATE=${DELTA_DIRECTION_ANCHOR_START_UPDATE}"
  echo "BIN_REG_WEIGHT=${BIN_REG_WEIGHT}"
  echo "BIN_REG_VARIANCE_WEIGHT=${BIN_REG_VARIANCE_WEIGHT}"
  echo "BIN_REG_LAYERS=${BIN_REG_LAYERS}"
  echo "BIN_REG_ATTN_ONLY=${BIN_REG_ATTN_ONLY}"
  echo "BIN_REG_START_UPDATE=${BIN_REG_START_UPDATE}"
  echo "BIN_REG_END_UPDATE=${BIN_REG_END_UPDATE}"
  echo "AOQ_EXPLORE_SCALE_RATIO=${AOQ_EXPLORE_SCALE_RATIO}"
  echo "AOQ_EXPLORE_LAYERS=${AOQ_EXPLORE_LAYERS}"
  echo "AOQ_EXPLORE_START_UPDATE=${AOQ_EXPLORE_START_UPDATE}"
  echo "AOQ_EXPLORE_END_UPDATE=${AOQ_EXPLORE_END_UPDATE}"
  echo "TEACHER_FEATURE_OUTPUT_WEIGHT=${TEACHER_FEATURE_OUTPUT_WEIGHT}"
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
  echo "CLASS_PROTECT_REF_KL_WEIGHT=${CLASS_PROTECT_REF_KL_WEIGHT}"
  echo "CLASS_PROTECT_REF_KL_CLASSES=${CLASS_PROTECT_REF_KL_CLASSES}"
  echo "CLASS_PROTECT_REF_KL_TEMPERATURE=${CLASS_PROTECT_REF_KL_TEMPERATURE}"
  echo "CLASS_PROTECT_REF_KL_CHECKPOINT=${CLASS_PROTECT_REF_KL_CHECKPOINT}"
  echo "QUANT_ONLY_START_EPOCH=${QUANT_ONLY_START_EPOCH}"
  echo "TRAINABLE_POLICY=${TRAINABLE_POLICY}"
  echo "TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS=${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS}"
  echo "TRAINABLE_POLICY_UPDATE_OVERRIDES=${TRAINABLE_POLICY_UPDATE_OVERRIDES}"
  echo "TRAINABLE_POLICY_UPDATE_MODE=${TRAINABLE_POLICY_UPDATE_MODE}"
  echo "TRAINABLE_POLICY_GRAD_DAMP=${TRAINABLE_POLICY_GRAD_DAMP}"
  echo "SAVE_STEP_CHECKPOINTS=${SAVE_STEP_CHECKPOINTS}"
  echo "SAVE_INITIAL_STEP_CHECKPOINT=${SAVE_INITIAL_STEP_CHECKPOINT}"
  echo "STEP_CHECKPOINT_INTERVAL=${STEP_CHECKPOINT_INTERVAL}"
  echo "STEP_CHECKPOINT_WARMUP_UPDATES=${STEP_CHECKPOINT_WARMUP_UPDATES}"
  echo "MAX_STEP_CHECKPOINTS_TO_SAVE=${MAX_STEP_CHECKPOINTS_TO_SAVE}"
  echo "MAX_TRAIN_UPDATES=${MAX_TRAIN_UPDATES}"
  echo "TEACHER_ATTN_OUTPUT_WEIGHT=${TEACHER_ATTN_OUTPUT_WEIGHT}"
  echo "TEACHER_ATTN_OUTPUT_LAYERS=${TEACHER_ATTN_OUTPUT_LAYERS}"
  echo "TEACHER_ATTN_OUTPUT_WARMUP_EPOCHS=${TEACHER_ATTN_OUTPUT_WARMUP_EPOCHS}"
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
