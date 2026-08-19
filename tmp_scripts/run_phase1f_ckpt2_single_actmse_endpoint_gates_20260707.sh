#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
SAVE_SCRIPT="${SAVE_SCRIPT:-${QATS}/tmp_scripts/run_phase1f_ckpt2_selective_actmse_endpoint_save_20260707.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${QATS}/tmp_scripts/eval_strict_w4a4_checkpoint_fullval_20260707.sh}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30670}"
ACT_MSE_BATCHES="${ACT_MSE_BATCHES:-8}"
ACT_MSE_BLEND="${ACT_MSE_BLEND:-0.35}"
ACT_MSE_GRID="${ACT_MSE_GRID:-0.85,1.25,17}"

names=(
  qkx
  qkv_input
  v
)

quantizers=(
  features.5.5.attn.quan_a_qkx_fn
  features.7.1.attn.quant_x_4_qkv.input_quant_fn
  features.5.5.attn.quan_a_v_fn
)

layers=(
  features.5.5
  features.7.1
  features.5.5
)

for i in "${!names[@]}"; do
  name="${names[$i]}"
  quantizer="${quantizers[$i]}"
  layer="${layers[$i]}"
  save_port="$((MASTER_PORT_BASE + i * 2))"
  eval_port="$((save_port + 1))"
  save_exp="recipe_phase1f_ckpt2_single_actmse_${name}_endpoint_20260707"
  eval_exp="eval_phase1f_ckpt2_single_actmse_${name}_endpoint_20260707"
  ckpt="${OUT}/${save_exp}/step_checkpoints/step_0000.pth.tar"

  echo "===== single activation-MSE endpoint save: ${name} (${quantizer}) ====="
  EXP="${save_exp}" \
  MASTER_PORT="${save_port}" \
  ACT_MSE_BATCHES="${ACT_MSE_BATCHES}" \
  ACT_MSE_LAYERS="${layer}" \
  ACT_MSE_QUANTIZERS="${quantizer}" \
  ACT_MSE_GRID="${ACT_MSE_GRID}" \
  ACT_MSE_BLEND="${ACT_MSE_BLEND}" \
  bash "${SAVE_SCRIPT}"

  echo "===== strict W4A4 full-val: ${name} (${ckpt}) ====="
  CKPT="${ckpt}" \
  EXP="${eval_exp}" \
  MASTER_PORT="${eval_port}" \
  bash "${EVAL_SCRIPT}"
done
