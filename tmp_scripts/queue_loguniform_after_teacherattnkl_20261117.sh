#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
WAIT_LOG="${WAIT_LOG:-/datadisk2/quyanyi/qat_runs/deit_tiny_w2a2_teacherattnkl_t2_5pct_20ep_4gpu_20261117.log}"
DEVICES="${DEVICES:-1,4,5,7}"
POLL_SECONDS="${POLL_SECONDS:-60}"

echo "[queue] waiting for ${WAIT_LOG} to finish at $(date '+%F %T')"
while ! grep -q '^exit_status=' "${WAIT_LOG}"; do
  sleep "${POLL_SECONDS}"
done

status="$(grep '^exit_status=' "${WAIT_LOG}" | tail -1 | cut -d= -f2)"
if [[ "${status}" != "0" ]]; then
  echo "[queue] prerequisite run failed with exit_status=${status}; not launching" >&2
  exit 1
fi

echo "[queue] prerequisite finished at $(date '+%F %T'); launching non-uniform run"
exec env \
  EXP=deit_tiny_w2a2_attn_loguniform_retry_20ep_4gpu_20261117 \
  DEVICES="${DEVICES}" \
  MASTER_PORT=30741 \
  bash "${QATS}/tmp_scripts/run_deit_tiny_w2a2_attn_loguniform_20ep_4gpu_20261117.sh"
