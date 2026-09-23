#!/usr/bin/env bash
# 2-epoch reproduction run on 2x RTX3090 (this box), mirroring the historical
# 100ep from-scratch no-KL control as closely as the local hardware allows.
#
# Differences forced by this machine (documented in
# docs/how_to_run_qat_on_this_3090_machine_20260920.md):
#   - folder-format ImageNet instead of the historical parquet shards
#   - batch 32/GPU x 2 GPU x grad-accum 8 = optimizer-step batch 512 (same as historical)
#   - --static-graph dropped: it trips DDP's reducer assert on torch 2.5.1
#     (the historical H100 runs used torch 2.9.1, where it is fine)
#   - teacher checkpoint converted from the HF-mirror release (see
#     tmp_scripts/prepare_swin_t_teacher_ckpt_20260920.py)
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-repro_2ep_2gpu_3090_20260920}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/swin_t-704ceda3.pth}"
# persistent env copy; /tmp/qat_env is the legacy volatile overlay
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-6,7}"
MASTER_PORT="${MASTER_PORT:-30667}"
SECONDS=0

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== 2ep repro on 2x3090 $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA} (dataset-format=folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES} batch=32/GPU accum=8 -> optimizer-step batch 512"
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
  --devices "${DEVICES}" --nproc-per-node 2 --master-port "${MASTER_PORT}" \
  --model-type swin \
  --teacher swin_t --teacher-type swin --teacher-pretrained \
  --teacher-checkpoint "${TEACHER}" \
  --epochs 100 --scheduler-epochs 100 \
  --batch-size 32 --workers 8 --lr 2e-4 --min-lr 5e-6 --weight-decay 0.0 \
  --grad-accum-steps 8 \
  --epoch-checkpoint-interval 1 --checkpoint-hist 2 \
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
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
  --extra-arg=--max_train_updates --extra-arg=5004 \
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
