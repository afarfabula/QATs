#!/usr/bin/env bash
set -euo pipefail
for p in $(pgrep -f 'swin_t_w4a4_stage2_prevstep_customhead.*from10_to20|qat_launch.py' || true); do
  pg=$(ps -o pgid= -p "$p" | tr -d ' ' || true)
  if [ -n "$pg" ]; then echo "kill pid=$p pgid=$pg"; kill -TERM -"$pg" || true; fi
done
sleep 8
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
