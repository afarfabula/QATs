#!/usr/bin/env bash
set -euo pipefail
QATS=/mlx_devbox/users/quyanyi/playground/QATs
OUT=/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_fromscratch_bs256_kd_noaug_strictckpt_100ep_20260702
CKPT=${OUT}/checkpoint-100.pth.tar
python3 - <<'PY'
from pathlib import Path
import torch
ckpt_path = Path('/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_fromscratch_bs256_kd_noaug_strictckpt_100ep_20260702/checkpoint-100.pth.tar')
if not ckpt_path.exists():
    raise SystemExit(f'MISSING {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
required = ['epoch', 'arch', 'state_dict', 'optimizer', 'args', 'rng_state', 'lr_scheduler']
missing = [k for k in required if k not in ckpt]
print('checkpoint:', ckpt_path)
print('keys:', sorted(ckpt.keys()))
print('missing_required:', missing)
print('epoch:', ckpt.get('epoch'))
print('state_dict_entries:', len(ckpt.get('state_dict', {})))
opt = ckpt.get('optimizer', {})
print('optimizer_state_entries:', len(opt.get('state', {})) if isinstance(opt, dict) else 'not_dict')
print('optimizer_param_groups:', len(opt.get('param_groups', [])) if isinstance(opt, dict) else 'not_dict')
sched = ckpt.get('lr_scheduler', {})
print('lr_scheduler_keys:', sorted(sched.keys()) if isinstance(sched, dict) else type(sched))
rng = ckpt.get('rng_state', {})
print('rng_state_keys:', sorted(rng.keys()) if isinstance(rng, dict) else type(rng))
if missing:
    raise SystemExit(2)
if ckpt.get('epoch') != 100:
    raise SystemExit(f'bad epoch: {ckpt.get("epoch")}')
PY
