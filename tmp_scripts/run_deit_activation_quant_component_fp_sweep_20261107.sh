#!/usr/bin/env bash
# Activation-quantizer component sensitivity sweep on a fixed W4A4 checkpoint.
# Each case restores exactly one activation family to FP and uses full 50k val.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
CKPT="${CKPT:-/datadisk2/quyanyi/qat_checkpoint_scales_20261107/deit_baseline_ep16_top1_66p984.pth.tar}"
RUNS="${RUNS:-/datadisk2/quyanyi/qat_runs}"
DEVICES="${DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-30881}"

cases=(
  "qkx:quan_a_qkx_fn"
  "v:quan_a_v_fn"
  "attn_proj:attn.proj.input_quant_fn"
  "mlp_fc1:mlp.fc1.input_quant_fn"
  "mlp_fc2:mlp.fc2.input_quant_fn"
)

for item in "${cases[@]}"; do
  tag="${item%%:*}"
  suffix="${item#*:}"
  exp="deit_tiny_w4a4_ep16_activation_fp_${tag}_eval_20261107"
  log="${RUNS}/${exp}.log"
  if [[ -f "${log}" ]] && grep -q 'Eval-only metrics:' "${log}"; then
    printf '%s: ' "${tag}"
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
  ACTIVATION_QUANT_FP=1 \
  ACTIVATION_QUANT_FP_SUFFIXES="${suffix}" \
    bash "${QATS}/tmp_scripts/run_deit_tiny_w4a4_attnquant_gate_2gpu_20261107.sh"

  printf '%s: ' "${tag}"
  grep 'Eval-only metrics:' "${log}" | tail -1
done
