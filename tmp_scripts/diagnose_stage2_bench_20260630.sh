#!/usr/bin/env bash
set -euo pipefail
EXP=swin_t_w4a4_stage2_prevstep_custom10_from50_bsz64_100step_bench_20260630
LOG=/tmp/${EXP}.log
ARGS=/tmp/qats_stage2_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/${EXP}/args.yaml
bash /mlx_devbox/users/quyanyi/playground/QATs/tmp_scripts/check_stage2_bench_20260630.sh || true
echo LOG_HEAD
sed -n '1,140p' "$LOG" 2>/dev/null || true
echo LOG_TAIL
tail -180 "$LOG" 2>/dev/null || true
echo ARGS
python3 - <<PY
import yaml, pathlib
p=pathlib.Path('$ARGS')
if p.exists():
    d=yaml.safe_load(p.read_text())
    for k in ['start_epoch','epochs','max_train_updates','skip_validate','resume','batch_size','world_size','train_scheme','ref_update','ref_warmup_epochs','ref_attn_kl_weight','ref_head_mode','use_kd','model_ema','static_graph']:
        print(k, d.get(k))
else:
    print('missing args', p)
PY
