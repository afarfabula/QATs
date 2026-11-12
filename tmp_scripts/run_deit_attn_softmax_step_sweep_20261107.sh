#!/usr/bin/env bash
# Evaluate post-training multipliers for the learned post-softmax LSQ step.
#
# This is an inference-time calibration check, not a new training objective:
# every result still executes the real W4A4 forward pass. A useful factor is
# evidence that the attention activation quantizer lands on a poor local step
# and should become an explicit calibration/training axis.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
CKPT="${CKPT:-/datadisk2/quyanyi/qat_runs/deit_tiny_w4a4_baseline_100ep_2gpu_20261107/checkpoint-9.pth.tar}"
DEVICES="${DEVICES:-0,1}"
OUT="${OUT:-/datadisk2/quyanyi/qat_checkpoint_scales_20261107}"
RUNS="${RUNS:-/datadisk2/quyanyi/qat_runs}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
FACTORS="${FACTORS:-1.0 0.75 1.25 0.5 1.5 0.9 1.1 2.0}"

mkdir -p "${OUT}"

for factor in ${FACTORS}; do
  tag="${factor//./p}"
  scaled_ckpt="${OUT}/deit_baseline_ep8_attn_softmax_step_x${tag}.pth.tar"
  exp="deit_tiny_w4a4_attn_softmax_step_x${tag}_eval_20261107"

  "${PY}" "${QATS}/tmp_scripts/scale_attention_softmax_checkpoint_20261107.py" \
    --checkpoint "${CKPT}" \
    --output "${scaled_ckpt}" \
    --factor "${factor}"

  EXP="${exp}" \
  LOG="${RUNS}/${exp}.log" \
  CKPT="${scaled_ckpt}" \
  EVAL_ONLY=1 \
  MAX_UPDATES=0 \
  DEVICES="${DEVICES}" \
  MASTER_PORT="$((30800 + RANDOM % 400))" \
    bash "${QATS}/tmp_scripts/run_deit_tiny_w4a4_attnquant_gate_2gpu_20261107.sh"

  grep -E 'Eval-only metrics:|Test: \[distributed-summary\]' "${RUNS}/${exp}.log" | tail -1
done
