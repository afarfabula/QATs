#!/usr/bin/env bash
HOST='fdbd:dccd:cdc2:12c8:0:138::'
PORT=9680
EXP='qssv1_best_100ep_fromscratch_20260703'
LOCAL_STATUS='/mlx_devbox/users/quyanyi/playground/QATs/docs/qssv1_best_100ep_fromscratch_20260703_status.tsv'
LOCAL_RAW='/mlx_devbox/users/quyanyi/playground/QATs/docs/qssv1_best_100ep_fromscratch_20260703_status.raw.json'
LOCAL_LOG='/tmp/monitor_qssv1_100ep_20260703.out'
mkdir -p "$(dirname "$LOCAL_STATUS")"
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  raw=$(ssh -p "$PORT" "$HOST" "python3 - <<'PY'
from pathlib import Path
import re, os, json, time
log=Path('/tmp/train_qssv1_best_100ep_fromscratch_20260703.log')
out=Path('/tmp/qat_recipe1_runs/qssv1_best_100ep_fromscratch_20260703')
text=log.read_text(errors='ignore') if log.exists() else ''
acc=[]
for m in re.finditer(r'\*\s+Acc@1\s+([0-9.]+)\s+Acc@5\s+([0-9.]+)', text):
    acc.append((float(m.group(1)), float(m.group(2))))
ckpts=[]
if out.exists():
    for p in sorted(out.glob('checkpoint-*.pth.tar')):
        m=re.fullmatch(r'checkpoint-(\d+)\.pth\.tar', p.name)
        if m:
            ckpts.append((int(m.group(1)), str(p), p.stat().st_size))
train=[]
for m in re.finditer(r'Train:\s+(\d+)\s+\[\s*(\d+)/(\d+)', text):
    train.append(tuple(map(int,m.groups())))
summ=[]
for m in re.finditer(r'TrainSummary:\s+epoch=(\d+).*?avg_step_time=([0-9.]+)s.*?samples_per_sec=([0-9.]+)', text):
    summ.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
proc_alive=(os.system('ps -p 2596676 >/dev/null 2>&1')==0)
print(json.dumps({'acc':acc,'ckpts':ckpts,'latest_train':train[-1] if train else None,'latest_summary':summ[-1] if summ else None,'proc_alive':proc_alive}, ensure_ascii=False))
PY" 2>&1)
  printf '%s\n' "$raw" > "$LOCAL_RAW"
  python3 - "$LOCAL_STATUS" "$LOCAL_RAW" "$ts" <<'PY'
import json, sys
from pathlib import Path
status=Path(sys.argv[1]); rawp=Path(sys.argv[2]); ts=sys.argv[3]
try:
    obj=json.loads(rawp.read_text().strip())
except Exception as e:
    obj={'acc':[],'ckpts':[],'latest_train':None,'latest_summary':None,'proc_alive':False,'error':str(e),'raw':rawp.read_text(errors='ignore')[:500]}
acc=[tuple(x) for x in obj.get('acc', [])]
ckpts={int(e):(path,size) for e,path,size in obj.get('ckpts', [])}
lines=['timestamp\tepoch\tacc1\tacc5\tcheckpoint\tckpt_bytes\tstate\n']
state=f"alive={obj.get('proc_alive')} latest_train={obj.get('latest_train')} latest_summary={obj.get('latest_summary')}"
if 'error' in obj:
    state += f" error={obj.get('error')}"
for i, epoch in enumerate(range(10,101,10), start=1):
    a=acc[i-1] if i <= len(acc) else ('NA','NA')
    ckpt=ckpts.get(epoch, ('NA','NA'))
    lines.append(f"{ts}\t{epoch}\t{a[0]}\t{a[1]}\t{ckpt[0]}\t{ckpt[1]}\t{state}\n")
status.write_text(''.join(lines))
PY
  echo "[$ts] refreshed status" >> "$LOCAL_LOG"
  if grep -q $'\t100\t[0-9]' "$LOCAL_STATUS"; then break; fi
  if grep -q 'alive=False' "$LOCAL_STATUS"; then break; fi
  sleep 300
done
