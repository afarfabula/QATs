#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
SOURCE="${SOURCE:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar}"

EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_gate_20260708}" \
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_source_hybridcore71_gate_20260708.log}" \
MASTER_PORT="${MASTER_PORT:-31347}" \
RESUME="${RESUME:-${SOURCE}}" \
START_EPOCH="${START_EPOCH:-3}" \
EPOCHS="${EPOCHS:-4}" \
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-4}" \
AOQ_EXPLORE_SCALE_RATIO="${AOQ_EXPLORE_SCALE_RATIO:-0.90}" \
AOQ_EXPLORE_SELECTIVE_MARGIN="${AOQ_EXPLORE_SELECTIVE_MARGIN:-0.08}" \
AOQ_EXPLORE_QUALITY_MODE="${AOQ_EXPLORE_QUALITY_MODE:-anchor_unmoved}" \
AOQ_EXPLORE_ANCHOR_CHECKPOINT="${AOQ_EXPLORE_ANCHOR_CHECKPOINT:-${SOURCE}}" \
AOQ_EXPLORE_LAYERS="${AOQ_EXPLORE_LAYERS:-features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv}" \
AOQ_EXPLORE_LAYER_RATIOS="${AOQ_EXPLORE_LAYER_RATIOS:-features.7.1.attn.proj:0.90,features.7.1.mlp.fc2:0.90}" \
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-1800}" \
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-0}" \
SKIP_VALIDATE="${SKIP_VALIDATE:-0}" \
"${QATS}/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh"
