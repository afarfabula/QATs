#!/usr/bin/env bash
set -euo pipefail
# Stop any running stage2 jobs first.
for p in $(pgrep -f 'swin_t_w4a4_stage2|qat_launch.py' || true); do
  cmd=$(ps -o cmd= -p "$p" || true)
  if echo "$cmd" | grep -q 'swin_t_w4a4_stage2'; then
    pg=$(ps -o pgid= -p "$p" | tr -d ' ' || true)
    if [ -n "$pg" ]; then echo "Killing stage2 pid=$p pgid=$pg"; kill -TERM -"$pg" || true; fi
  fi
done
sleep 8

echo GPU_AFTER_STAGE2_KILL
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

QATS=/mlx_devbox/users/quyanyi/playground/QATs
# user requested system/workspace disk, not /tmp
OUT=$QATS/outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624
TEACHER=/home/tiger/.cache/torch/hub/checkpoints/swin_t-704ceda3.pth
SRC=/tmp/qats_stage1_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep_fastval_20260630/checkpoint-10.pth.tar
EXP=swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_continue10_to50_sysdisk_20260630
LOG=/tmp/train_${EXP}.log
mkdir -p "$OUT"

cat > /tmp/run_${EXP}.sh <<STAGE1EOF
#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
PYTHONUNBUFFERED=1 python3 "$QATS/qat_launch.py" \\
  --method ofq --stage train \\
  --config "$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \\
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \\
  --output "$OUT" --experiment "$EXP" \\
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 29993 --model-type swin \\
  --teacher swin_t --teacher-type swin --teacher-checkpoint "$TEACHER" --teacher-pretrained \\
  --resume "$SRC" --no-resume-opt --epochs 50 --scheduler-epochs 50 \\
  --batch-size 256 --workers 8 --lr 2e-4 --min-lr 1e-5 --weight-decay 0.0 --epoch-checkpoint-interval 1 \\
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \\
  --pretrained --pretrained-initialized --use-kd --kd-hard-and-soft 1 \\
  --quantized --amp --amp-dtype bf16 \\
  --extra-arg=--static-graph \\
  --extra-arg=--smoothing --extra-arg=0.0 --extra-arg=--mixup --extra-arg=0.0 --extra-arg=--cutmix --extra-arg=0.0 --extra-arg=--aa --extra-arg=none --extra-arg=--color-jitter --extra-arg=0.0 --extra-arg=--reprob --extra-arg=0.0 \\
  --extra-arg=--log-interval --extra-arg=50 --extra-arg=--seed --extra-arg=42 \\
  2>&1 | tee "$LOG"
echo "wall_seconds=\$SECONDS"
echo "train log: $LOG"
STAGE1EOF
chmod +x /tmp/run_${EXP}.sh
nohup bash /tmp/run_${EXP}.sh > /tmp/${EXP}.runner.log 2>&1 &
echo "LAUNCHED_STAGE1_TO50 exp=$EXP"
sleep 10
pgrep -af "$EXP|qat_launch.py" || true
tail -n 120 /tmp/${EXP}.runner.log || true
echo DF
 df -h / /tmp | sed -n '1,5p'
