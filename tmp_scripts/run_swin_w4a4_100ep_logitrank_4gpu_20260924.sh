#!/usr/bin/env bash
# Swin-T W4A4 OFQ/QAT, 100 epochs on 4x RTX3090, with the classification-logit
# ranking loss on top of the usual KD (same batch arithmetic as the historical
# 8xH100 control: 32 per GPU x 4 ranks x accum 4 = 512 images / optimizer step).
#
# Usage:  RANK_W=1e-2 DEVICES=4,5,6,7 bash tmp_scripts/run_swin_w4a4_100ep_logitrank_4gpu_20260924.sh
#         (start it under setsid nohup; the script itself runs in the foreground)
#
# Expected cost with an idle node: ~0.37 s per micro-step -> ~1.0-1.1 h/epoch
# -> 100 epochs ≈ 4.2-4.5 days train + ~1.7 h full validation. On a contended node
# multiply by up to ~2x (see docs/deit_tiny_w4a4_smoke_and_speed_20260923.md §8).
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-ofq_swin_t_w4a4_100ep_logitrank_20260924}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/swin_t-704ceda3.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30690}"
RANK_W="${RANK_W:-1e-2}"        # 分类 logits ranking 权重；先看短跑再定
RANK_TOPK="${RANK_TOPK:-5}"
SECONDS=0

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== Swin-T W4A4 100ep + logits ranking, 4x3090, $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA} (folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES}  batch=32/GPU  accum=ceil(16/4)=4  -> 512 images/optimizer step"
  echo "logit_rank_weight=${RANK_W}  topk=${RANK_TOPK}"
  echo "PY=${PY}"
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
  --devices "${DEVICES}" --nproc-per-node 4 --master-port "${MASTER_PORT}" \
  --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-pretrained \
  --teacher-checkpoint "${TEACHER}" \
  --epochs 100 --scheduler-epochs 100 \
  --batch-size 32 --workers 8 --lr 2e-4 --min-lr 5e-6 --weight-decay 0.0 \
  --grad-accum-steps 16 \
  --epoch-checkpoint-interval 1 --checkpoint-hist 2 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
  --logit-rank-weight "${RANK_W}" --logit-rank-topk "${RANK_TOPK}" \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1 \
  --extra-arg=--color-jitter --extra-arg=0.4 \
  --extra-arg=--reprob --extra-arg=0.25 \
  --extra-arg=--log-interval --extra-arg=50 \
  --extra-arg=--seed --extra-arg=42 \
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
