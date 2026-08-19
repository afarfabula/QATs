#!/usr/bin/env bash
set -euo pipefail
pg=1374012
echo killing_pgid=$pg
kill -TERM -$pg || true
sleep 8
ps -eo pid,pgid,cmd | grep -E 'swin_t_w4a4_stage1_continue60_to100_tmp_20260701|qat_launch.py' | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
