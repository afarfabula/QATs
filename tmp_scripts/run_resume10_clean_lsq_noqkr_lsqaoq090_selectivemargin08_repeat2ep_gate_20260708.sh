#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"

EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_repeat2ep_gate_20260708}" \
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_repeat2ep_gate_20260708.log}" \
MASTER_PORT="${MASTER_PORT:-31213}" \
START_EPOCH="${START_EPOCH:-3}" \
EPOCHS="${EPOCHS:-5}" \
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-5}" \
AOQ_EXPLORE_SCALE_RATIO="${AOQ_EXPLORE_SCALE_RATIO:-0.90}" \
AOQ_EXPLORE_SELECTIVE_MARGIN="${AOQ_EXPLORE_SELECTIVE_MARGIN:-0.08}" \
AOQ_EXPLORE_QUALITY_MODE="${AOQ_EXPLORE_QUALITY_MODE:-none}" \
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-1800}" \
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-0}" \
SKIP_VALIDATE="${SKIP_VALIDATE:-0}" \
"${QATS}/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh"
