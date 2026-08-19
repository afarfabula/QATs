#!/usr/bin/env bash
set -euo pipefail
ps -eo pid,ppid,pgid,cmd | grep -E 'qat_launch.py|bench_stage2|torchrun|train.py' | grep -v grep || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
