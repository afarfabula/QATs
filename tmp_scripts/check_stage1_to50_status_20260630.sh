#!/usr/bin/env bash
set -euo pipefail
EXP=swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630
LOG=/tmp/train_${EXP}.log
OUT=/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/$EXP

echo PROCS
pgrep -af "$EXP|qat_launch.py" || true

echo METRICS
python3 - <<'PY'
from pathlib import Path
log=Path('/tmp/train_swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630.log')
print('LOG', log.exists(), log.stat().st_size if log.exists() else 0)
if log.exists():
    keys=['TrainSummary','epoch:','distributed-summary','wall_seconds','Traceback','RuntimeError','ProcessRaised','terminate called','Loaded checkpoint']
    lines=[l for l in log.read_text(errors='ignore').splitlines() if any(k in l for k in keys)]
    for line in lines[-220:]: print(line)
PY

echo CKPTS
ls -lh "$OUT"/checkpoint-*.pth.tar "$OUT"/checkpoint-*.ema.pth.tar 2>/dev/null | tail -40 || true

echo DISK
df -h / /tmp | sed -n '1,5p'
du -sh "$OUT" 2>/dev/null || true

echo GPU
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
