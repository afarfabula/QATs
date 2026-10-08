#!/usr/bin/env bash
# Swin-T W4A4 QAT + attention-relation ranking, reference = prev-step self model.
#
# Design (agreed 2026-10-08):
#   * constrain ALL heads of global blocks 8, 9, 10, 11  (no prior/oscillation-analysis head picking)
#   * reference is the *prev-step copy of the student itself* (self-consistency), not the FP teacher
#   * ranking on visual-token pairs per query row: teacher/ref top-5 keys vs all strictly lower keys,
#     pairs where either side is below --attn-rank-min-attn are dropped (SW-MSA mask / padding)
#   * the constraint runs for the WHOLE schedule at a constant weight (no on/off pulse controller)
#
# Usage:
#   RANK_W=0.0  MAX_UPDATES=100 EXP=... bash <this>      # control / A-B
#   RANK_W=<l>  MAX_UPDATES=25030 EXP=... bash <this>    # 10-epoch gate
#   RANK_W=<l>  MAX_UPDATES=0 EXP=... bash <this>        # full 100 epochs
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/swin_t-704ceda3.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-4,5,6,7}"
NPROC="${NPROC:-$(echo "${DEVICES}" | tr ',' '\n' | grep -c .)}"
MASTER_PORT="${MASTER_PORT:-30710}"
EXP="${EXP:-ofq_swin_t_w4a4_attnrank_prevstep_l8to11_20261008}"
LOG="${LOG:-${OUT}/${EXP}.log}"

RANK_W="${RANK_W:-0.3}"          # attention-relation ranking 权重；0 表示对照
RANK_TOPK="${RANK_TOPK:-5}"
RANK_MIN_ATTN="${RANK_MIN_ATTN:-1e-4}"
REF_INTERVAL="${REF_INTERVAL:-50}"   # prev-step ref 每多少个 optimizer step 同步一次
MAX_UPDATES="${MAX_UPDATES:-0}"      # >0 时提前停止（按 optimizer step 计数，2503/epoch）
PROBE="${PROBE:-0}"                  # 1 = 只跑一次梯度范数探针，用来定权重
LOG_INTERVAL="${LOG_INTERVAL:-50}"
RANK_TARGET="${RANK_TARGET:-post_quant}"   # post_quant / post_quant_detached_scale / pre_quant
FREEZE_S="${FREEZE_S:-0}"                  # 1 = 冻结注意力 softmax 的 LSQ 步长 s
FREEZE_SUFFIX="${FREEZE_SUFFIX:-}"         # 逗号分隔的参数名子串,粘性冻结
HINGE="${HINGE:-0}"                        # 1 = 用 relu(margin-Delta) 替代 softplus(-Delta)
MARGIN="${MARGIN:-0}"                      # hinge 的目标 margin
SAVE_STEPS="${SAVE_STEPS:-0}"              # 1 = 按 optimizer step 存 checkpoint（离线评估用）
STEP_INTERVAL="${STEP_INTERVAL:-50}"

# all heads of global blocks 8,9,10,11 (Swin-T: stage2 = 12 heads, stage3 = 24 heads)
build_heads() {
  local spec="$1" out=""
  for pair in ${spec}; do
    local layer="${pair%%:*}" nheads="${pair##*:}"
    for ((h = 0; h < nheads; h++)); do out="${out}${layer}:${h},"; done
  done
  echo "custom_subset:${out%,}"
}
HEADS="$(build_heads '8:12 9:12 10:24 11:24')"

mkdir -p "${OUT}"
rm -f "${LOG}"

EXTRA=()
if [[ "${MAX_UPDATES}" != "0" ]]; then
  EXTRA+=(--extra-arg=--max_train_updates --extra-arg="${MAX_UPDATES}")
fi
if [[ "${PROBE}" == "1" ]]; then
  EXTRA+=(--attn-rank-probe)
fi
if [[ "${FREEZE_S}" == "1" ]]; then
  EXTRA+=(--freeze-attn-softmax-scale)
fi
if [[ -n "${FREEZE_SUFFIX}" ]]; then
  EXTRA+=(--freeze-param-suffix "${FREEZE_SUFFIX}")
fi
if [[ "${HINGE}" == "1" ]]; then
  EXTRA+=(--attn-rank-hinge --attn-rank-margin "${MARGIN}")
fi
if [[ "${SAVE_STEPS}" == "1" ]]; then
  EXTRA+=(--extra-arg=--save_step_checkpoints --extra-arg=--step_checkpoint_interval --extra-arg="${STEP_INTERVAL}")
fi

{
  echo "===== Swin-T W4A4 + attn-relation ranking (ref=prev-step), $(date '+%F %T') ====="
  echo "QATS=${QATS}  git_head=$(git -C "${QATS}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "DATA=${DATA} (folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}  nproc=${NPROC}  batch=32/GPU  accum=ceil(16/${NPROC})=?  -> 512 images/optimizer step"
  echo "layers=8,9,10,11 (all heads)  n_head_slots=$(echo -n "${HEADS}" | tr ',' '\n' | wc -l)"
  echo "attn_rank_weight=${RANK_W}  topk=${RANK_TOPK}  min_attn=${RANK_MIN_ATTN}  source=ref"
  echo "attn_rank_target=${RANK_TARGET}  freeze_attn_softmax_scale=${FREEZE_S}  freeze_suffix=${FREEZE_SUFFIX}"
  echo "hinge=${HINGE}  margin=${MARGIN}"
  echo "ref_update=prev_step  ref_update_interval=${REF_INTERVAL}"
  echo "max_train_updates=${MAX_UPDATES}  probe=${PROBE}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo "start_epoch=$(date +%s)"
  echo
} | tee "${LOG}"

set +e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
"${PY}" "${QATS}/qat_launch.py" \
  --method ofq --stage train \
  --config "${QATS}/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --data "${DATA}" --dataset-format folder \
  --output "${OUT}" --experiment "${EXP}" \
  --devices "${DEVICES}" --nproc-per-node "${NPROC}" --master-port "${MASTER_PORT}" \
  --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-pretrained \
  --teacher-checkpoint "${TEACHER}" \
  --epochs 100 --scheduler-epochs 100 \
  --batch-size 32 --workers 8 --lr 2e-4 --min-lr 5e-6 --weight-decay 0.0 \
  --grad-accum-steps 16 \
  --epoch-checkpoint-interval 1 --checkpoint-hist 3 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --train-scheme ema_ref_attn_kl --ref-update prev_step --ref-update-interval "${REF_INTERVAL}" \
  --ref-attn-kl-weight 0.0 --ref-logit-kl-weight 0.0 \
  --ref-head-mode "${HEADS}" \
  --attn-rank-weight "${RANK_W}" --attn-rank-source ref \
  --attn-rank-topk "${RANK_TOPK}" --attn-rank-min-attn "${RANK_MIN_ATTN}" \
  --attn-rank-target "${RANK_TARGET}" \
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1 \
  --extra-arg=--color-jitter --extra-arg=0.4 \
  --extra-arg=--reprob --extra-arg=0.25 \
  --extra-arg=--log-interval --extra-arg="${LOG_INTERVAL}" \
  --extra-arg=--seed --extra-arg=42 \
  "${EXTRA[@]}" \
  >> "${LOG}" 2>&1
status=$?
set -e

{
  echo
  echo "exit_status=${status}"
  echo "wall_seconds=${SECONDS}"
  echo "train_log=${LOG}"
  echo "output=${OUT}/${EXP}"
} | tee -a "${LOG}"
