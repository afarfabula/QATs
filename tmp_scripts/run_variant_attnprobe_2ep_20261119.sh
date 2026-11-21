#!/usr/bin/env bash
# 2-epoch probe of an aux-attention objective, keeping the 20-epoch LR schedule.
set -uo pipefail
QATS=/home/quyanyi/tiger/resume_repos/QATs
DATA=/datadisk2/linyichen/OFQ/ImageNet-1K
OUT=/datadisk2/quyanyi/qat_runs
TEACHER=/home_ext/quyanyi/qat_weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth
PY=/datadisk2/quyanyi/envs/qat_env/bin/python
EXP="${EXP:?EXP required}"
LOG="$OUT/$EXP.log"
DEVICES="${DEVICES:-1,4,5,7}"
PORT="${PORT:-30801}"
HEAD="custom_subset:8:0,8:1,8:2,9:0,9:1,9:2,10:0,10:1,10:2,11:0,11:1,11:2"

ARGS=()
if [ "${KL_WEIGHT:-0}" != "0" ]; then
  ARGS+=(--teacher-attn-kl-weight "$KL_WEIGHT" --teacher-attn-kl-temperature 2.0
         --teacher-attn-kl-warmup-epochs 0 --ref-head-mode "$HEAD")
fi
if [ -n "${AQ_WEIGHT:-}" ]; then
  ARGS+=(--extra-arg=--attention-quant-weight --extra-arg="$AQ_WEIGHT"
         --extra-arg=--attention-quant-jsd-weight --extra-arg=0
         --extra-arg=--attention-quant-mass-weight --extra-arg=1
         --extra-arg=--attention-quant-topk-mass-weight --extra-arg=0)
fi
if [ -n "${MAX_UPDATES:-}" ]; then
  ARGS+=(--extra-arg=--max_train_updates --extra-arg="$MAX_UPDATES")
fi
if [ -n "${LSQ_FLOOR:-}" ]; then
  ARGS+=(--lsq-scale-floor "$LSQ_FLOOR" --lsq-scale-floor-target attention_softmax)
fi

rm -f "$LOG"
{
  echo "===== $EXP start $(date '+%F %T') ====="
  echo "KL_WEIGHT=${KL_WEIGHT:-0} AQ_WEIGHT=${AQ_WEIGHT:-none} LSQ_FLOOR=${LSQ_FLOOR:-none}"
  echo "DSH_TEACHER_KL_TARGET=${DSH_TEACHER_KL_TARGET:-default} DSH_FREEZE_ATTN_S=${DSH_FREEZE_ATTN_S:-0} DSH_TOPK_RECALL_WEIGHT=${DSH_TOPK_RECALL_WEIGHT:-0}"
  echo "epochs=${EPOCHS:-2} scheduler_epochs=${SCHED:-20} devices=$DEVICES"
} | tee -a "$LOG"

TORCH_HOME=/datadisk2/quyanyi/cache/torch \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
"$PY" "$QATS/qat_launch.py" \
  --method ofq --stage train \
  --config "$QATS/third_party/OFQ/configs/deit_tiny_qat_baseline_100ep.yml" \
  --model deit_tiny_distilled_patch16_224 --model-type deit \
  --teacher deit_tiny_distilled_patch16_224 --teacher-type deit \
  --teacher-checkpoint "$TEACHER" --teacher-pretrained \
  --data "$DATA" --dataset-format folder \
  --output "$OUT" --experiment "$EXP" \
  --devices "$DEVICES" --nproc-per-node 4 --master-port "$PORT" \
  --epochs "${EPOCHS:-2}" --scheduler-epochs "${SCHED:-20}" \
  --batch-size 64 --workers 8 \
  --lr 5e-4 --min-lr 1e-5 --weight-decay 5e-2 --warmup-epochs 5 \
  --grad-accum-steps 8 \
  --epoch-checkpoint-interval 1 --checkpoint-hist 1 \
  --num-aug-repeats 0 \
  --wbits 2 --abits 2 --wq-mode statsq --aq-mode lsq \
  --wq-per-channel --aq-per-channel --aq-clip-learnable \
  --pretrained --pretrained-initialized \
  --use-kd --kd-hard-and-soft 1 --ofq-official-kd-heads \
  --teacher-soft-temperature 1.0 \
  --logit-rank-weight 0 --attn-rank-weight 0 \
  --attn-rank-target post_quant \
  --train-scheme ema_ref_attn_kl \
  --ref-update prev_step --ref-update-interval 50 \
  --ref-attn-kl-weight 0 --ref-logit-kl-weight 0 \
  --quantized --qk-reparam --qk-reparam-type 0 \
  --amp --amp-dtype bf16 \
  "${ARGS[@]}" \
  --extra-arg=--smoothing --extra-arg=0.1 \
  --extra-arg=--mixup --extra-arg=0.0 \
  --extra-arg=--cutmix --extra-arg=0.0 \
  --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1 \
  --extra-arg=--color-jitter --extra-arg=0.4 \
  --extra-arg=--reprob --extra-arg=0.25 \
  >> "$LOG" 2>&1
echo "exit_status=$?" | tee -a "$LOG"
