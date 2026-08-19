#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"

EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_attnanchor_gate_20260708}" \
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_attnanchor_gate_20260708.log}" \
MASTER_PORT="${MASTER_PORT:-31153}" \
AOQ_EXPLORE_SELECTIVE_MARGIN="${AOQ_EXPLORE_SELECTIVE_MARGIN:-0.08}" \
AOQ_EXPLORE_QUALITY_MODE="${AOQ_EXPLORE_QUALITY_MODE:-none}" \
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-1800}" \
SELECTIVE_BIN_ANCHOR_WEIGHT="${SELECTIVE_BIN_ANCHOR_WEIGHT:-0.00005}" \
SELECTIVE_BIN_ANCHOR_LAYERS="${SELECTIVE_BIN_ANCHOR_LAYERS:-features.5.5.attn.qkv,features.5.5.attn.proj,features.7.1.attn.qkv,features.7.1.attn.proj}" \
SELECTIVE_BIN_ANCHOR_CAPTURE_UPDATE="${SELECTIVE_BIN_ANCHOR_CAPTURE_UPDATE:-1800}" \
SELECTIVE_BIN_ANCHOR_END_UPDATE="${SELECTIVE_BIN_ANCHOR_END_UPDATE:-0}" \
SELECTIVE_BIN_ANCHOR_MARGIN="${SELECTIVE_BIN_ANCHOR_MARGIN:-0.05}" \
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-0}" \
SKIP_VALIDATE="${SKIP_VALIDATE:-0}" \
"${QATS}/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh"
