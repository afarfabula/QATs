import torch
from pathlib import Path
p=Path('/tmp/qats_stage2_outputs/stage2_50to51_teacherref_20260701/swin_t_w4a4_stage2_c35_teacher_softonly_partial_ema999_from50_to51_20260701/checkpoint-51.ema.pth.tar')
print('exists', p.exists(), 'size', p.stat().st_size if p.exists() else None)
ckpt=torch.load(p, map_location='cpu', weights_only=False)
print('type', type(ckpt))
if isinstance(ckpt, dict):
    print('keys', ckpt.keys())
    for k,v in ckpt.items():
        if isinstance(v, dict):
            print(k, 'dict len', len(v), 'sample keys', list(v.keys())[:10])
        else:
            print(k, type(v), v if isinstance(v,(int,float,str)) else '')
    sd=ckpt.get('state_dict') or ckpt.get('model') or ckpt
    if isinstance(sd, dict):
        keys=list(sd.keys())[:30]
        print('sd sample', keys)
