#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/mlx_devbox/users/quyanyi/playground/QATs}"
SOURCE="${SOURCE:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/recipe_resume10_clean_lsq_noqkr_native_warmstart_300logit300feat_cont1to3_20260708/checkpoint-3.pth.tar}"
LATE5571="${LATE5571:-features.5.5.attn.qkv,features.5.5.attn.proj,features.5.5.mlp.fc2,features.7.1.attn.qkv,features.7.1.attn.proj,features.7.1.mlp.fc2}"

EXP="${EXP:-recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708}" \
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_recipe_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_candidateanchor_gate_20260708.log}" \
MASTER_PORT="${MASTER_PORT:-31356}" \
AOQ_EXPLORE_SCALE_RATIO="${AOQ_EXPLORE_SCALE_RATIO:-0.90}" \
AOQ_EXPLORE_SELECTIVE_MARGIN="${AOQ_EXPLORE_SELECTIVE_MARGIN:-0.08}" \
AOQ_EXPLORE_LAYERS="${AOQ_EXPLORE_LAYERS:-${LATE5571}}" \
AOQ_EXPLORE_END_UPDATE="${AOQ_EXPLORE_END_UPDATE:-1800}" \
CANDIDATE_BIN_ANCHOR_WEIGHT="${CANDIDATE_BIN_ANCHOR_WEIGHT:-0.0001}" \
CANDIDATE_BIN_ANCHOR_LAYERS="${CANDIDATE_BIN_ANCHOR_LAYERS:-${LATE5571}}" \
CANDIDATE_BIN_ANCHOR_CAPTURE_UPDATE="${CANDIDATE_BIN_ANCHOR_CAPTURE_UPDATE:-1800}" \
CANDIDATE_BIN_ANCHOR_END_UPDATE="${CANDIDATE_BIN_ANCHOR_END_UPDATE:-0}" \
CANDIDATE_BIN_ANCHOR_SOURCE_CHECKPOINT="${CANDIDATE_BIN_ANCHOR_SOURCE_CHECKPOINT:-${SOURCE}}" \
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-0}" \
SKIP_VALIDATE="${SKIP_VALIDATE:-0}" \
"${QATS}/tmp_scripts/run_resume10_clean_lsq_noqkr_lsqaoq090_selectivemargin08_gate_20260708.sh"
