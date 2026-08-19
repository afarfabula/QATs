from pathlib import Path
patterns=["/tmp/*quantflags*20260629*.log","/tmp/*quantflags*20260629*.runner.log","/tmp/train_swin_t_w4a4_imagenet1k_8gpu_scratch_stage1_no_kl*20260629.log"]
files=[]
for pat in patterns:
    files += list(Path('/').glob(pat.lstrip('/')))
for p in sorted(set(files)):
    print('====', p, p.stat().st_size)
    txt=p.read_text(errors='ignore')
    for line in txt.splitlines():
        if ('[QATs] command=' in line or 'TrainSummary' in line or 'epoch:' in line or 'Test: [  97/97]' in line or 'Test: [  24/24]' in line):
            print(line)
