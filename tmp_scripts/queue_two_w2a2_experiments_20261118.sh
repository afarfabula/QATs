#!/usr/bin/env bash
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
OUT="${OUT:-/datadisk2/quyanyi/qat_runs}"
PREREQ_LOG="${PREREQ_LOG:-${OUT}/deit_tiny_w2a2_attn_loguniform_retry_20ep_4gpu_20261117.log}"
DEVICES="${DEVICES:-1,4,5,7}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CALIBRATOR="${QATS}/tmp_scripts/calibrate_aux_loss_weight_20261118.py"

wait_for_log_exit() {
  local log="$1"
  while ! grep -q '^exit_status=' "${log}"; do
    sleep "${POLL_SECONDS}"
  done
  local status
  status="$(grep '^exit_status=' "${log}" | tail -1 | cut -d= -f2)"
  if [[ "${status}" != "0" ]]; then
    return 1
  fi
}

echo "[queue] waiting for ${PREREQ_LOG} at $(date '+%F %T')"
wait_for_log_exit "${PREREQ_LOG}"
echo "[queue] prerequisite complete at $(date '+%F %T')"

P1_PROBE_EXP="probe_deit_tiny_w2a2_preqattnkl_20261118"
P1_PROBE_LOG="${OUT}/${P1_PROBE_EXP}.log"
P1_WEIGHT=0.01
if TEACHER_ATTN_KL_WEIGHT=0.01 \
  MAX_UPDATES=50 SKIP_VALIDATE=1 \
  EXP="${P1_PROBE_EXP}" DEVICES="${DEVICES}" MASTER_PORT=30750 \
    bash "${QATS}/tmp_scripts/run_deit_tiny_w2a2_preqattnkl_19ep_4gpu_20261118.sh"; then
  P1_WEIGHT="$(python3 "${CALIBRATOR}" --log "${P1_PROBE_LOG}" --aux-field TeacherAttnKL --target-share 0.05)"
else
  echo "[queue] P1 calibration failed; using fallback weight=${P1_WEIGHT}" >&2
fi
echo "[queue] pre-quant attention KL weight=${P1_WEIGHT}"

if ! TEACHER_ATTN_KL_WEIGHT="${P1_WEIGHT}" \
  EXP=deit_tiny_w2a2_preqattnkl_t2_19ep_4gpu_20261118 \
  DEVICES="${DEVICES}" MASTER_PORT=30751 \
    bash "${QATS}/tmp_scripts/run_deit_tiny_w2a2_preqattnkl_19ep_4gpu_20261118.sh"; then
  echo "[queue] P1 full run failed; continuing to P2" >&2
fi

P2_PROBE_EXP="probe_deit_tiny_w2a2_attn_rowmass_20261118"
P2_PROBE_LOG="${OUT}/${P2_PROBE_EXP}.log"
P2_WEIGHT=1.0
if ATTENTION_QUANT_WEIGHT=1.0 \
  MAX_UPDATES=50 SKIP_VALIDATE=1 \
  EXP="${P2_PROBE_EXP}" DEVICES="${DEVICES}" MASTER_PORT=30752 \
    bash "${QATS}/tmp_scripts/run_deit_tiny_w2a2_rowmass_19ep_4gpu_20261118.sh"; then
  P2_WEIGHT="$(python3 "${CALIBRATOR}" --log "${P2_PROBE_LOG}" --aux-field AttnQuant --target-share 0.05)"
else
  echo "[queue] P2 calibration failed; using fallback weight=${P2_WEIGHT}" >&2
fi
echo "[queue] row-mass weight=${P2_WEIGHT}"

if ! ATTENTION_QUANT_WEIGHT="${P2_WEIGHT}" \
  EXP=deit_tiny_w2a2_attn_rowmass_19ep_4gpu_20261118 \
  DEVICES="${DEVICES}" MASTER_PORT=30753 \
    bash "${QATS}/tmp_scripts/run_deit_tiny_w2a2_rowmass_19ep_4gpu_20261118.sh"; then
  echo "[queue] P2 full run failed" >&2
fi
