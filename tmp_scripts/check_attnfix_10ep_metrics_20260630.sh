#!/usr/bin/env bash
set -euo pipefail
LOG=/tmp/train_swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep_fastval_20260630.log
OUT=/tmp/qats_stage1_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep_fastval_20260630
echo PROCS
pgrep -af 'swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep|qat_launch.py' || true
echo METRICS
python3 - <<'PY'
from pathlib import Path
log=Path('/tmp/train_swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep_fastval_20260630.log')
if not log.exists():
    print('missing log')
else:
    keys=['TrainSummary','epoch:','distributed-summary','wall_seconds','Traceback','RuntimeError']
    for line in [l for l in log.read_text(errors='ignore').splitlines() if any(k in l for k in keys)][-120:]:
        print(line)
PY
echo CKPTS
ls -lh "$OUT"/checkpoint-*.pth.tar 2>/dev/null | tail -20 || true
echo DF
df -h / /tmp | sed -n '1,5p'
