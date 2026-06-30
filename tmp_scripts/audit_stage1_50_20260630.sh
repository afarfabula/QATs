set -euo pipefail
cd /mlx_devbox/users/quyanyi/playground/QATs
CKPT=/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630/checkpoint-50.pth.tar
LOG=/tmp/train_swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630.log
OUT=/mlx_devbox/users/quyanyi/playground/QATs/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630
printf 'CKPT_STAT\n'
ls -lh "$CKPT"
stat -c '%n %s bytes %y' "$CKPT"
printf '\nCKPT_LOAD\n'
CKPT_PATH="$CKPT" python3 - <<'PY'
import os, torch
p = os.environ['CKPT_PATH']
ckpt = torch.load(p, map_location='cpu', weights_only=False)
print(type(ckpt))
print(sorted(list(ckpt.keys()))[:50])
print('epoch', ckpt.get('epoch'))
print('state_dict_keys', len(ckpt.get('state_dict', {})))
print('optimizer', type(ckpt.get('optimizer', None)).__name__, ckpt.get('optimizer', None) is not None)
PY
printf '\nMETRIC_SUMMARY\n'
LOG_PATH="$LOG" python3 - <<'PY'
import os, re, json
log = os.environ['LOG_PATH']
cur_epoch = None
rows = []
train = {}
for line in open(log, errors='ignore'):
    m = re.search(r'TrainSummary: epoch=(\d+) updates=(\d+) avg_step_time=([0-9.]+)s samples_per_step=(\d+) samples_per_sec=([0-9.]+)', line)
    if m:
        e=int(m.group(1)); train[e]={'updates':int(m.group(2)),'avg_step_time':float(m.group(3)),'samples_per_step':int(m.group(4)),'samples_per_sec':float(m.group(5))}; cur_epoch=e
    m = re.search(r'Test: \[distributed-summary\]\s+Time: ([0-9.]+)s\s+Loss: ([0-9.]+)\s+Acc@1: ([0-9.]+)\s+Acc@5: ([0-9.]+)\s+Samples: (\d+)\s+RankSamples: \[([^\]]+)\]', line)
    if m:
        rows.append({'epoch':cur_epoch,'val_time':float(m.group(1)),'loss':float(m.group(2)),'top1':float(m.group(3)),'top5':float(m.group(4)),'samples':int(m.group(5)),'rank_samples':'['+m.group(6)+']', **train.get(cur_epoch,{})})
print('n_rows', len(rows))
if rows:
    best=max(rows, key=lambda r: r['top1'])
    final=rows[-1]
    print('best', json.dumps(best, ensure_ascii=False))
    print('final', json.dumps(final, ensure_ascii=False))
    print('last10')
    for r in rows[-10:]: print(json.dumps(r, ensure_ascii=False))
PY
printf '\nDISK\n'
df -h / /tmp
du -sh "$OUT"
printf '\nGPU\n'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
