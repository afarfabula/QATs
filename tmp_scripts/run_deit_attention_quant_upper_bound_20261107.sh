#!/usr/bin/env bash
# Full-val upper-bound checks for two quantization-specific failure modes:
#   1. non-positive LSQ scales trapped in the clip gradient dead zone;
#   2. row mass removed by finite-step post-softmax quantization.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
CKPT="${CKPT:-/datadisk2/quyanyi/qat_checkpoint_scales_20261107/deit_baseline_ep16_top1_66p984.pth.tar}"
OUT="${OUT:-/datadisk2/quyanyi/qat_checkpoint_scales_20261107}"
RUNS="${RUNS:-/datadisk2/quyanyi/qat_runs}"
DEVICES="${DEVICES:-0,1}"

revived="${OUT}/deit_baseline_ep16_deadscale_revived_median.pth.tar"
combined="${OUT}/deit_baseline_ep16_deadscale_revived_median_renorm.pth.tar"

"${PY}" "${QATS}/tmp_scripts/revive_nonpositive_attention_softmax_scales_20261107.py" \
  --checkpoint "${CKPT}" \
  --output "${revived}" \
  --stat median
cp -p "${revived}" "${combined}"

run_eval() {
  local exp="$1"
  local ckpt="$2"
  local renorm="$3"
  local fp="$4"
  EXP="${exp}" \
  LOG="${RUNS}/${exp}.log" \
  CKPT="${ckpt}" \
  EVAL_ONLY=1 \
  MAX_UPDATES=0 \
  DEVICES="${DEVICES}" \
  MASTER_PORT="${MASTER_PORT:-30821}" \
  ATTENTION_QUANT_WEIGHT=0 \
  ATTENTION_SOFTMAX_POST_QUANT_RENORM="${renorm}" \
  ATTENTION_SOFTMAX_FP="${fp}" \
    bash "${QATS}/tmp_scripts/run_deit_tiny_w4a4_attnquant_gate_2gpu_20261107.sh"
  grep -E 'Eval-only metrics:|Test: \[distributed-summary\]' "${RUNS}/${exp}.log" | tail -1
}

run_eval deit_tiny_w4a4_ep16_baseline_eval_20261107 "${CKPT}" 0 0
run_eval deit_tiny_w4a4_ep16_attn_softmax_fp_eval_20261107 "${CKPT}" 0 1
run_eval deit_tiny_w4a4_ep16_deadscale_revived_eval_20261107 "${revived}" 0 0
run_eval deit_tiny_w4a4_ep16_attn_postquant_renorm_eval_20261107 "${CKPT}" 1 0
run_eval deit_tiny_w4a4_ep16_deadscale_revived_plus_renorm_eval_20261107 "${combined}" 1 0

old_ckpt="${OUT}/deit_baseline_ep10_top1_65p52.pth.tar"
old_revived="${OUT}/deit_baseline_ep10_deadscale_revived_median.pth.tar"
"${PY}" "${QATS}/tmp_scripts/revive_nonpositive_attention_softmax_scales_20261107.py" \
  --checkpoint "${old_ckpt}" \
  --output "${old_revived}" \
  --stat median
run_eval deit_tiny_w4a4_ep10_deadscale_revived_eval_20261107 "${old_revived}" 0 0
