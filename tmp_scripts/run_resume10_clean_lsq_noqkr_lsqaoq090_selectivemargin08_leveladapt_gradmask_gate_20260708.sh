#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"

EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_gate_20260708}" \
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_leveladapt_gradmask_gate_20260708.log}" \
MASTER_PORT="${MASTER_PORT:-31351}" \
AOQ_EXPLORE_SCALE_RATIO="${AOQ_EXPLORE_SCALE_RATIO:-0.90}" \
AOQ_EXPLORE_SELECTIVE_MARGIN="${AOQ_EXPLORE_SELECTIVE_MARGIN:-0.08}" \
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-1800}" \
TRAINABLE_POLICY="${TRAINABLE_POLICY:-all}" \
TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS="${TRAINABLE_POLICY_FREEZE_ACT_EXCEPT_LAYERS:-features.5.5,features.7.1}" \
TRAINABLE_POLICY_UPDATE_OVERRIDES="${TRAINABLE_POLICY_UPDATE_OVERRIDES:-1800:quant_in_layers}" \
TRAINABLE_POLICY_UPDATE_MODE="${TRAINABLE_POLICY_UPDATE_MODE:-grad_mask}" \
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-0}" \
SKIP_VALIDATE="${SKIP_VALIDATE:-0}" \
"${QATS}/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh"
