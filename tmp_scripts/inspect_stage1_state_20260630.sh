#!/usr/bin/env bash
set -euo pipefail

echo 'EMA90 files:'
ls -lh /tmp/qats_stage1_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_ema90_1ep_fastval_20260630 2>/dev/null || true

echo 'EMA90 metrics:'
python3 - <<'PY'
from pathlib import Path
for log in [
    Path('/tmp/train_swin_t_w4a4_stage1_bs256_kd_noaug_ema90_1ep_fastval_20260630.log'),
    Path('/tmp/train_swin_t_w4a4_stage1_setupalpha16_100steps_20260630.log'),
]:
    print('====', log)
    if not log.exists():
        print('missing')
        continue
    keys = ['TrainSummary', 'Train: 0', 'distributed-summary', 'wall_seconds', 'Traceback', 'ProcessRaised', 'RuntimeError', 'terminate called', 'Stopped early']
    lines = [line for line in log.read_text(errors='ignore').splitlines() if any(k in line for k in keys)]
    for line in lines[-80:]:
        print(line)
PY
