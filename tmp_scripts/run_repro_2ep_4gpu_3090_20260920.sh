#!/usr/bin/env bash
# 2-epoch reproduction run on 4x RTX3090 (this box), batch-matched to the historical
# 8xH100 from-scratch no-KL control: optimizer-step batch = 512.
#
# Batch arithmetic (important, see docs/how_to_run_qat_on_this_3090_machine_20260920.md):
#   qat_launch.py divides --grad-accum-steps by world_size:  accum = ceil(passed / world_size)
#   we want accum = 4 with 4 ranks, so pass 16.
#   => 32 (per GPU) x 4 (ranks) x 4 (accum) = 512 images per optimizer step
#   => 1,281,167 / 128 = 10,009 micro-steps/epoch; /4 = 2,502 optimizer steps/epoch
#   => --max_train_updates 5004 == exactly 2 epochs
#
# Known deviations from the historical run: folder-format ImageNet instead of parquet,
# no --static-graph (torch 2.5.1 DDP reducer assert), torch 2.5.1 vs 2.9.1.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
EXP="${EXP:-repro_2ep_4gpu_3090_20260920}"
LOG="${LOG:-${OUT}/${EXP}.log}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/swin_t-704ceda3.pth}"
# persistent env copy; /tmp/qat_env is the legacy volatile overlay
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
DEVICES="${DEVICES:-2,5,6,7}"
MASTER_PORT="${MASTER_PORT:-30678}"
SECONDS=0

mkdir -p "${OUT}"
rm -f "${LOG}"

{
  echo "===== 2ep repro on 4x3090 $(date '+%F %T') ====="
  echo "QATS=${QATS}"
  echo "DATA=${DATA} (dataset-format=folder)"
  echo "OUT=${OUT}/${EXP}"
  echo "LOG=${LOG}"
  echo "TEACHER=${TEACHER}"
  echo "DEVICES=${DEVICES} batch=32/GPU accum=ceil(16/4)=4 -> optimizer-step batch 512"
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
