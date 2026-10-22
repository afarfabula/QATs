#!/usr/bin/env bash
# Bisect helper: run the *vendored old trainer* (third_party/OFQ/train.py, i.e. the
# subprocess/OFQ entry point) for exactly 1 epoch on the same folder data / same 4 GPUs /
# same effective batch / same seed as the qat_launch unified-trainer runs.
#
# Purpose: decide whether the trainer implementation itself moves epoch-0 val accuracy.
#   - if ep0 lands near 77.7 (historical box level) -> trainer implementation is the cause
#   - if ep0 lands near 78.4 (qat_launch level on this box) -> trainer is not the cause
#
# Deviations forced by this machine (same as the 9/20 repro notes):
#   * no --static-graph (torch 2.5.1 DDP reducer asserts)
#   * torch 2.5.1 (historical: 2.9.1)
#   * batch 32/GPU + grad-accum 4 instead of batch 64/GPU, so the optimizer-step batch
#     stays 512 like the historical 8xH100 run and like the qat_launch runs.
#   * --max_train_updates 2503 stops right after epoch 0 while keeping the 100-epoch cosine.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-oldtrain_path_1ep_repro_20260929}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/swin_t-704ceda3.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30701}"
SECONDS=0

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== old train.py path, 1 epoch, $(date '+%F %T') ====="
  echo "QATS=${QATS}  git_head=$(git -C "${QATS}" rev-parse --short HEAD 2>/dev/null)"
  echo "DATA=${DATA} (torch/image_folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "DEVICES=${DEVICES}  batch=32/GPU x 4 ranks x accum 4 = 512 / optimizer step"
  echo
} | tee "${LOG}"

set +e
cd "${QATS}/third_party/OFQ"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
"${PY}" train.py \
  "${DATA}" --dataset torch/image_folder --val-split val \
  -c "${QATS}/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
  --model swin_t --model_type swin \
  --output "${OUT}" --experiment "${EXP}" \
  --visible_gpu "${DEVICES}" --world_size 4 --tcp_port "${MASTER_PORT}" \
  --teacher swin_t --teacher_type swin --teacher_pretrained --teacher-checkpoint "${TEACHER}" \
  --epochs 100 --scheduler-epochs 100 --max_train_updates 2503 \
  --batch-size 32 --grad-accum-steps 4 --workers 8 \
  --lr 2e-4 --weight-decay 0.0 --min-lr 5e-6 \
  --checkpoint-hist 2 --epoch-checkpoint-interval 1 \
  --wq-bitw 4 --wq-enable --aq-bitw 4 --aq-enable \
  --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq_clip_learnable \
  --pretrained --pretrained_initialized \
  --use-kd --kd_hard_and_soft 0 \
  --quantized --qk_reparam --qk_reparam_type 0 \
  --amp --amp-dtype bf16 \
  --smoothing 0.1 --mixup 0.0 --cutmix 0.0 \
  --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --reprob 0.25 \
  --log-interval 50 --seed 42 \
  >> "${LOG}" 2>&1
status=$?
set -e

{
  echo
  echo "exit_status=${status}"
  echo "wall_seconds=${SECONDS}"
  echo "log=${LOG}"
} | tee -a "${LOG}"
