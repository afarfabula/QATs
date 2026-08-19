#!/usr/bin/env bash
set -euo pipefail
echo PROCS_BEFORE
ps -eo pid,pgid,cmd | grep -E 'swin_t_w4a4_stage1_strict_resume50_to100_lr1e5_tmp_20260701|qat_launch.py' | grep -v grep || true
pg=$(ps -eo pgid,cmd | awk '/swin_t_w4a4_stage1_strict_resume50_to100_lr1e5_tmp_20260701/ && !/awk/ {print $1; exit}')
if [ -n "${pg:-}" ]; then echo killing_pgid=$pg; kill -TERM -$pg || true; sleep 8; fi
echo GPU
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
