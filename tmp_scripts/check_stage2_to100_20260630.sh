#!/usr/bin/env bash
set -euo pipefail
EXP=swin_t_w4a4_stage2_prevstep_custom10_from50_to100_bsz192_nokd_20260630
LOG=/tmp/${EXP}.log
OUT=/tmp/qats_stage2_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/${EXP}
echo PROCS
ps -eo pid,ppid,pgid,cmd | grep -E "${EXP}|qat_launch.py" | grep -v grep || true
echo METRICS
[ -f "${LOG}" ] && grep -E "Loaded checkpoint|Enabled EMA refmodel|Effective batch|TrainSummary|Test: \[distributed-summary\]|Stopped early|RefAttnKL|RuntimeError|CUDA out|Traceback|Killed" "${LOG}" | tail -120 || true
echo CKPTS
ls -lh "${OUT}" 2>/dev/null | tail -30 || true
echo DISK
df -h / /tmp
du -sh "${OUT}" 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
