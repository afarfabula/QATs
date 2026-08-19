#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"

EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708}" \
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_pulse3_gate_20260708.log}" \
MASTER_PORT="${MASTER_PORT:-31354}" \
AOQ_EXPLORE_SCALE_RATIO="${AOQ_EXPLORE_SCALE_RATIO:-1.0}" \
AOQ_EXPLORE_SELECTIVE_MARGIN="${AOQ_EXPLORE_SELECTIVE_MARGIN:-0.0}" \
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-0}" \
AOQ_EXPLORE_UPDATE_SCHEDULE="${AOQ_EXPLORE_UPDATE_SCHEDULE:-0:0.90:0:0.08,300:1.0:0:0,600:0.90:0:0.08,900:1.0:0:0,1200:0.90:0:0.08,1500:1.0:0:0}" \
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-0}" \
SKIP_VALIDATE="${SKIP_VALIDATE:-0}" \
"${QATS}/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh"
