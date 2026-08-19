#!/usr/bin/env bash
set -euo pipefail
echo PROCS
ps -eo pid,pgid,cmd | grep -E 'qat_launch.py|train.py|swin_t_w4a4|torch' | grep -v grep || true
echo GPU_APPS
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
echo GPU_SUM
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
