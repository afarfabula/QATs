#!/usr/bin/env bash
set -euo pipefail
LOG=/tmp/train_swin_t_w4a4_stage2_prevstep_customhead_from10_to20_fastval_20260630.log

echo '---LOG TAIL---'
tail -n 240 "$LOG" 2>/dev/null || echo missing

echo '---PROC TREE---'
ps -o pid,ppid,pgid,stat,etime,%cpu,%mem,cmd -p 727409,727411,727412 2>/dev/null || true
pgrep -P 727411 -a || true
ps -o pid,ppid,pgid,stat,etime,%cpu,%mem,cmd --ppid 727411 2>/dev/null | head -40 || true

echo '---NVIDIA---'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

echo '---RECENT FILES---'
OUT=/tmp/qats_stage1_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage2_prevstep_customhead_from10_to20_fastval_20260630
find "$OUT" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -40 || true
