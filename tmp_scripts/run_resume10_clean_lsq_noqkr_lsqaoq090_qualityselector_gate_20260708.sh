#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"

EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_gate_20260708}" \
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_qualityselector_gate_20260708.log}" \
MASTER_PORT="${MASTER_PORT:-31113}" \
AOQ_EXPLORE_SELECTIVE_MARGIN="${AOQ_EXPLORE_SELECTIVE_MARGIN:-0.08}" \
AOQ_EXPLORE_QUALITY_MODE="${AOQ_EXPLORE_QUALITY_MODE:-grad_cross}" \
AOQ_EXPLORE_QUALITY_MIN_FRAC="${AOQ_EXPLORE_QUALITY_MIN_FRAC:-0.10}" \
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-1800}" \
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-0}" \
SKIP_VALIDATE="${SKIP_VALIDATE:-0}" \
"${QATS}/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh"
