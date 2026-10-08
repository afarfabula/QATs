#!/usr/bin/env bash
# Weight sweep for the prev-step attention-relation ranking loss (see
# docs/attn_relation_ranking_qat_prevstep_l8to11_failure_20261008.md).
#
# Runs one lambda per GPU, single-process each, identical recipe / seed / data order.
# 100 optimizer steps = 1600 micro-steps = 2% of epoch 0, then a full 50k validation.
# The sweep is what exposed the lambda cliff (0.3 -> 3 collapses Top-1).
#
# Usage: WEIGHTS="0 3 10 30" DEVICES=4,5,6,7 bash <this>
set -euo pipefail

QATS="${QATS:-/home/quyanyi/tiger/resume_repos/QATs}"
DEVICES="${DEVICES:-4,5,6,7}"
WEIGHTS="${WEIGHTS:-0 3 10 30}"
MAX_UPDATES="${MAX_UPDATES:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
TAG_PREFIX="${TAG_PREFIX:-attnrank_sweep_20261008}"
RUN_SCRIPT="${QATS}/tmp_scripts/run_swin_w4a4_attnrank_prevstep_l8to11_4gpu_20261008.sh"

read -r -a GPUS <<< "$(echo "${DEVICES}" | tr ',' ' ')"
read -r -a WS <<< "${WEIGHTS}"

if [[ "${#WS[@]}" -gt "${#GPUS[@]}" ]]; then
  echo "需要至少和权重数量一样多的 GPU" >&2
  exit 2
fi

for i in "${!WS[@]}"; do
  w="${WS[$i]}"
  gpu="${GPUS[$i]}"
  tag="${TAG_PREFIX}_w${w}"
  RANK_W="${w}" \
  MAX_UPDATES="${MAX_UPDATES}" \
  LOG_INTERVAL="${LOG_INTERVAL}" \
  DEVICES="${gpu}" \
  NPROC=1 \
  MASTER_PORT="$((30721 + gpu))" \
  EXP="${tag}" \
  LOG="/datadisk2/quyanyi/qat_runs/${tag}.log" \
  setsid nohup bash "${RUN_SCRIPT}" > "/tmp/${tag}.out" 2>&1 < /dev/null &
  echo "launched lambda=${w} on gpu=${gpu} (experiment ${tag})"
done

wait
