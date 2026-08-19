#!/usr/bin/env bash
set -euo pipefail
EXP=swin_t_w4a4_stage2_prevstep_custom10_from50_to100_bsz192_nokd_20260630
ps -eo pid,ppid,pgid,cmd | grep -E "${EXP}|qat_launch.py" | grep -v grep || true
PGIDS=$(ps -eo pgid=,cmd= | grep -E "${EXP}|qat_launch.py" | grep -v grep | awk '{print $1}' | sort -u || true)
if [ -n "${PGIDS}" ]; then
  echo "Killing PGIDs: ${PGIDS}"
  for pg in ${PGIDS}; do kill -TERM -"${pg}" 2>/dev/null || true; done
  sleep 5
  for pg in ${PGIDS}; do kill -KILL -"${pg}" 2>/dev/null || true; done
else
  echo "No matching running process."
fi
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
