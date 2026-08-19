#!/usr/bin/env bash
set -euo pipefail
EXP=swin_t_w4a4_stage2_prevstep_customhead_nokd_from10_to20_fastval_20260630
LOG=/tmp/train_${EXP}.log
OUT=/tmp/qats_stage1_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/$EXP

echo PROCS
pgrep -af "$EXP|qat_launch.py" || true

echo METRICS
python3 - <<'PY'
from pathlib import Path
log=Path('/tmp/train_swin_t_w4a4_stage2_prevstep_customhead_nokd_from10_to20_fastval_20260630.log')
if not log.exists():
    print('missing log')
else:
    keys=['Train:','TrainSummary','epoch:','distributed-summary','RefAttnKL','Test:','wall_seconds','Traceback','RuntimeError','ProcessRaised','terminate called']
    lines=[l for l in log.read_text(errors='ignore').splitlines() if any(k in l for k in keys)]
    for line in lines[-220:]:
        print(line)
PY

echo CKPTS
ls -lh "$OUT"/checkpoint-*.pth.tar "$OUT"/checkpoint-*.ema.pth.tar 2>/dev/null | tail -30 || true

echo GPU
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

echo DF
df -h / /tmp | sed -n '1,5p'
