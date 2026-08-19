#!/usr/bin/env bash
set -euo pipefail
for p in $(pgrep -f 'swin_t_w4a4_stage2_prevstep_customhead.*from10_to20|qat_launch.py' || true); do
  pg=$(ps -o pgid= -p "$p" | tr -d ' ' || true)
  if [ -n "$pg" ]; then
    echo "Killing stage2 pid=$p pgid=$pg"
    kill -TERM -"$pg" || true
  fi
done
sleep 8

echo GPU_AFTER_KILL
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

QATS=/mlx_devbox/users/quyanyi/playground/QATs
OUT=/tmp/qats_stage1_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624
RESUME=$OUT/swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep_fastval_20260630/checkpoint-10.pth.tar
EXP=swin_t_w4a4_stage2_prevstep_customhead_refonly_from10_to20_fastval_20260630
LOG=/tmp/train_${EXP}.log

cat > /tmp/run_${EXP}.sh <<STAGE2EOF
#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
PYTHONUNBUFFERED=1 python3 "$QATS/qat_launch.py" \\
  --method ofq --stage train \\
  --config "$QATS/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \\
  --model swin_t --data /tmp/imagenet1k_full_parquet --dataset-format parquet \\
  --output "$OUT" --experiment "$EXP" \\
  --devices 0,1,2,3,4,5,6,7 --nproc-per-node 8 --master-port 29991 --model-type swin \\
  --resume "$RESUME" --no-resume-opt --epochs 20 --scheduler-epochs 50 \\
  --batch-size 256 --workers 8 --lr 5e-5 --min-lr 1e-5 --weight-decay 0.0 --epoch-checkpoint-interval 1 \\
  --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq --wq-per-channel --aq-per-channel --aq-clip-learnable \\
  --pretrained --pretrained-initialized --quantized --amp --amp-dtype bf16 \\
  --train-scheme ema_ref_attn_kl --ref-update prev_step --ref-attn-kl-weight 0.001 \\
  --ref-head-mode custom:5:2,10:14,5:1,4:1,9:10,6:1,8:4,8:9,11:18,11:4 --ref-warmup-epochs 0 \\
  --extra-arg=--smoothing --extra-arg=0.0 --extra-arg=--mixup --extra-arg=0.0 --extra-arg=--cutmix --extra-arg=0.0 --extra-arg=--aa --extra-arg=none --extra-arg=--color-jitter --extra-arg=0.0 --extra-arg=--reprob --extra-arg=0.0 \\
  --extra-arg=--log-interval --extra-arg=50 --extra-arg=--seed --extra-arg=42 \\
  2>&1 | tee "$LOG"
echo "wall_seconds=\$SECONDS"
echo "train log: $LOG"
STAGE2EOF
chmod +x /tmp/run_${EXP}.sh
nohup bash /tmp/run_${EXP}.sh > /tmp/${EXP}.runner.log 2>&1 &
echo "LAUNCHED $EXP"
sleep 15
pgrep -af "$EXP|qat_launch.py" || true
echo LOG_TAIL
tail -n 120 /tmp/${EXP}.runner.log || true
echo GPU_AFTER_LAUNCH
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
