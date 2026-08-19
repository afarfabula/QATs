#!/usr/bin/env bash
set -euo pipefail
EXP=swin_t_w4a4_stage2_prevstep_custom10_from50_to100_bsz128_kd_ema999_20260630
LOG=/tmp/${EXP}.log
OUT=/tmp/qats_stage2_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/${EXP}
echo PROCS
ps -eo pid,ppid,pgid,cmd | grep -E "${EXP}|qat_launch.py" | grep -v grep || true
echo METRICS
[ -f "${LOG}" ] && grep -E "TrainSummary|Test: \[distributed-summary\]|Enabled EMA refmodel|Enabled student weight EMA|Effective batch|RefAttnKL|RuntimeError|CUDA out|Traceback|Killed" "${LOG}" | tail -140 || true
echo CKPTS
ls -lh "${OUT}" 2>/dev/null | tail -40 || true
echo DISK
df -h / /tmp
du -sh "${OUT}" 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
