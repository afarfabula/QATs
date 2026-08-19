#!/usr/bin/env bash
set -euo pipefail
QATS=/mlx_devbox/users/quyanyi/playground/QATs
BASE=/tmp/qats_stage1_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624
STAGE1_EXP=swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep_fastval_20260630
STAGE1_DIR=$BASE/$STAGE1_EXP
STAGE1_LOG=/tmp/train_${STAGE1_EXP}.log

echo '---PROCS---'
pgrep -af 'swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep|swin_t_w4a4_stage2|qat_launch.py' || true

echo '---STAGE1 METRICS---'
python3 - <<'PY'
from pathlib import Path
log=Path('/tmp/train_swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep_fastval_20260630.log')
if not log.exists():
    print('missing log')
else:
    keys=['TrainSummary','epoch:','distributed-summary','wall_seconds','Traceback','RuntimeError']
    for line in [l for l in log.read_text(errors='ignore').splitlines() if any(k in l for k in keys)][-160:]:
        print(line)
PY

echo '---STAGE1 CKPTS---'
ls -lh "$STAGE1_DIR"/checkpoint-*.pth.tar 2>/dev/null | tail -20 || true

if pgrep -f 'swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep' >/dev/null; then
  echo 'STAGE1_STILL_RUNNING'
  exit 0
fi

LAST_CKPT=$(ls "$STAGE1_DIR"/checkpoint-*.pth.tar 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$LAST_CKPT" ]; then
  echo 'NO_STAGE1_CKPT'
  exit 1
fi

echo "LAST_CKPT=$LAST_CKPT"
python3 - <<PY
import torch
p='$LAST_CKPT'
ck=torch.load(p,map_location='cpu',weights_only=False)
print('LAST_CKPT_OK epoch=', ck.get('epoch'), 'state_keys=', len(ck.get('state_dict',{})))
PY

STAGE2_EXP=swin_t_w4a4_stage2_prevstep_customhead_from10_to20_fastval_20260630
STAGE2_LOG=/tmp/train_${STAGE2_EXP}.log
if pgrep -f "$STAGE2_EXP" >/dev/null; then
  echo 'STAGE2_ALREADY_RUNNING'
  exit 0
fi
if [ -d "$BASE/$STAGE2_EXP" ] && ls "$BASE/$STAGE2_EXP"/checkpoint-*.pth.tar >/dev/null 2>&1; then
  echo 'STAGE2_OUTPUT_EXISTS_WITH_CKPTS_NOT_RESTARTING'
  exit 0
fi

cat > /tmp/run_${STAGE2_EXP}.sh <<STAGE2EOF
#!/usr/bin/env bash
set -euo pipefail
QATS=/mlx_devbox/users/quyanyi/playground/QATs
OUT=/tmp/qats_stage1_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624
TEACHER=/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth
RESUME=$LAST_CKPT
EXP=$STAGE2_EXP
LOG=/tmp/train_\${EXP}.log
SECONDS=0
PYTHONUNBUFFERED=1 python3 "\$QATS/qat_launch.py" \\
  --method ofq --stage train \\
  --config "\$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \\
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \\
  --output "\$OUT" --experiment "\$EXP" \\
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 29989 --model-type swin \\
  --teacher swin_t --teacher-type swin --teacher-checkpoint "\$TEACHER" --teacher-pretrained \\
  --resume "\$RESUME" --no-resume-opt --epochs 20 --scheduler-epochs 50 \\
  --batch-size 256 --workers 8 --lr 5e-5 --min-lr 1e-5 --weight-decay 0.0 --epoch-checkpoint-interval 1 \\
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \\
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 \\
  --train-scheme ema_ref_attn_kl --ref-update prev_step --ref-attn-kl-weight 0.001 \\
  --ref-head-mode custom:5:2,10:14,5:1,4:1,9:10,6:1,8:4,8:9,11:18,11:4 --ref-warmup-epochs 0 \\
  --model-ema --model-ema-decay 0.999 \\
  --quantized --amp --amp-dtype bf16 \\
  --extra-arg=--static-graph \\
  --extra-arg=--smoothing --extra-arg=0.0 --extra-arg=--mixup --extra-arg=0.0 --extra-arg=--cutmix --extra-arg=0.0 --extra-arg=--aa --extra-arg=none --extra-arg=--color-jitter --extra-arg=0.0 --extra-arg=--reprob --extra-arg=0.0 \\
  --extra-arg=--log-interval --extra-arg=50 --extra-arg=--seed --extra-arg=42 \\
  2>&1 | tee "\$LOG"
echo "wall_seconds=\$SECONDS"
echo "train log: \$LOG"
STAGE2EOF
chmod +x /tmp/run_${STAGE2_EXP}.sh
nohup bash /tmp/run_${STAGE2_EXP}.sh > /tmp/${STAGE2_EXP}.runner.log 2>&1 &
echo "STAGE2_LAUNCHED exp=$STAGE2_EXP"
sleep 5
pgrep -af "$STAGE2_EXP" || true
tail -n 60 /tmp/${STAGE2_EXP}.runner.log || true
