#!/usr/bin/env bash
# Leave-one-layer-in-full-precision diagnostic for post-softmax attention.
# Every run changes exactly one attention quantizer from A4 to FP and evaluates
# the full 50k ImageNet validation set.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
CKPT="${CKPT:-/datadisk2/quyanyi/qat_checkpoint_scales_20261107/deit_baseline_ep16_top1_66p984.pth.tar}"
RUNS="${RUNS:-/datadisk2/quyanyi/qat_runs}"
DEVICES="${DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-30861}"
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11}"

for layer in ${LAYERS}; do
  exp="deit_tiny_w4a4_ep16_attn_fp_layer${layer}_eval_20261107"
  log="${RUNS}/${exp}.log"
  if [[ -f "${log}" ]] && grep -q 'Eval-only metrics:' "${log}"; then
    printf 'layer %s: ' "${layer}"
    grep 'Eval-only metrics:' "${log}" | tail -1
    continue
  fi

  EXP="${exp}" \
  LOG="${log}" \
  CKPT="${CKPT}" \
  EVAL_ONLY=1 \
  MAX_UPDATES=0 \
  DEVICES="${DEVICES}" \
  MASTER_PORT="${MASTER_PORT}" \
  ATTENTION_QUANT_WEIGHT=0 \
  ATTENTION_SOFTMAX_FP=1 \
  ATTENTION_SOFTMAX_FP_LAYERS="blocks.${layer}.attn" \
    bash "${QATS}/tmp_scripts/run_deit_tiny_w4a4_attnquant_gate_2gpu_20261107.sh"

  printf 'layer %s: ' "${layer}"
  grep 'Eval-only metrics:' "${log}" | tail -1
done
