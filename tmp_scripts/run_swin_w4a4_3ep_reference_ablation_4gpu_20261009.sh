#!/usr/bin/env bash
# 3-epoch follow-up of the reference ablation (see
# docs/attn_relation_ranking_qat_anticheat_experiments_20261008.md sections 6/7).
#
# Three runs, strictly sequential on the same 4 GPUs, same seed / data order / effective
# batch (32 per GPU x 4 ranks x accum 4 = 512). The ONLY difference between them is the
# attention constraint that is switched on, so the numbers are directly comparable:
#
#   baseline     : nothing on (lambda = 0)
#   teacher-kl   : FP teacher attention KL, weight 0.01   (best point of the 2%-epoch sweep)
#   teacher-rank : FP teacher rank/softplus, lambda 0.3   (best point of the 2%-epoch sweep)
#
# All three: global blocks 8/9/10/11, ALL heads, warmup 0, no clip, no drop, always on.
# 3 epochs = 3 x 2503 = 7509 optimizer steps, full 50k validation after every epoch.
#
# Usage: bash tmp_scripts/run_swin_w4a4_3ep_reference_ablation_4gpu_20261009.sh
set -euo pipefail

QATS="${QATS:-/home/quyanyi/tiger/resume_repos/QATs}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
RUN_SCRIPT="${QATS}/tmp_scripts/run_swin_w4a4_attnrank_prevstep_l8to11_4gpu_20261008.sh"

EPOCH_UPDATES="${EPOCH_UPDATES:-2503}"          # 10009 micro-steps / accum 4, 4 ranks
EPOCHS="${EPOCHS:-3}"
MAX_UPDATES=$((EPOCH_UPDATES * EPOCHS))
DEVICES="${DEVICES:-4,5,6,7}"
TAG_PREFIX="${TAG_PREFIX:-swin3ep_20261009}"
MASTER_PORT="${MASTER_PORT:-30830}"
SECONDS=0

run_one() {
  local name="$1" rank_w="$2" rank_source="$3" teacher_kl_w="$4" port="$5"
  local exp="${TAG_PREFIX}_${name}"
  local log="${OUT}/${exp}.log"
  echo "=================================================================="
  echo "[driver] start ${name}  rank_w=${rank_w} source=${rank_source} teacher_kl_w=${teacher_kl_w}"
  echo "[driver] exp=${exp}  log=${log}  max_train_updates=${MAX_UPDATES} (${EPOCHS} epochs)"
  echo "[driver] elapsed=${SECONDS}s"
  echo "=================================================================="
  RANK_W="${rank_w}" \
  RANK_SOURCE="${rank_source}" \
  TEACHER_KL_W="${teacher_kl_w}" \
  REF_KL_W=0 \
  REF_KL_CLIP=0 \
  MAX_UPDATES="${MAX_UPDATES}" \
  LOG_INTERVAL=50 \
  DEVICES="${DEVICES}" \
  MASTER_PORT="${port}" \
  EXP="${exp}" \
  LOG="${log}" \
  bash "${RUN_SCRIPT}"
  echo "[driver] finished ${name}; log=${log}"
}

run_one baseline      0    ref     0     "${MASTER_PORT}"
run_one teacherkl     0    ref     0.01  "$((MASTER_PORT + 1))"
run_one teacherrank   0.3  teacher 0     "$((MASTER_PORT + 2))"

echo
echo "[driver] all done, wall_seconds=${SECONDS}"
for name in baseline teacherkl teacherrank; do
  log="${OUT}/${TAG_PREFIX}_${name}.log"
  echo "--- ${name} (${log}) ---"
  grep -E "distributed-summary|Best metric" "${log}" | tail -5 || true
done
