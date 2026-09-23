#!/usr/bin/env bash
# Smoke + speed comparison: DeiT-Tiny vs Swin-T under the same QAT settings
# (W4A4 + statsq/lsq + QK-reparam + KD + bf16, folder ImageNet, 1 GPU).
#
# Purpose: verify the DeiT-Tiny OFQ/QAT path runs end to end, and measure whether
# it is actually cheaper per optimizer step than Swin-T before investing in it.
#
# Usage:  GPU=4 STEPS=64 bash tmp_scripts/smoke_deit_vs_swin_speed_20260923.sh
#         RUNS=deit bash ...     # only one of: deit | swin

set -uo pipefail

REPO=/home_ext/quyanyi/tiger/resume_repos/QATs
PY=/datadisk2/quyanyi/envs/qat_env/bin/python
DATA=/datadisk2/linyichen/OFQ/ImageNet-1K
OUT=/tmp/qat_runs_smoke_20260923
LOGDIR=${LOGDIR:-/tmp/qat_smoke_20260923}

GPU=${GPU:-4}
STEPS=${STEPS:-64}
BATCH=${BATCH:-32}
WORKERS=${WORKERS:-8}
PORT=${PORT:-30640}
RUNS=${RUNS:-deit,swin}
LOGINTERVAL=${LOGINTERVAL:-8}
TAG=${TAG:-smoke}

mkdir -p "$LOGDIR" "$OUT"

run_one () {
  local name=$1 model=$2 mtype=$3 cfg=$4 lr=$5 wd=$6 port=$7 log=$8
  echo "[smoke] start $name  model=$model lr=$lr wd=$wd gpu=$GPU port=$port log=$log"
  cd "$REPO"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  "$PY" qat_launch.py \
    --method ofq --stage train \
    --config "$cfg" \
    --model "$model" --model-type "$mtype" \
    --teacher "$model" --teacher-type "$mtype" \
    --data "$DATA" --dataset-format folder \
    --output "$OUT" --experiment "${name}_${TAG}_w4a4_bs${BATCH}_${STEPS}step" \
    --devices "$GPU" --nproc-per-node 1 --master-port "$port" \
    --epochs 1 --scheduler-epochs 1 --batch-size "$BATCH" --workers "$WORKERS" \
    --grad-accum-steps 1 --lr "$lr" --min-lr 1e-5 --weight-decay "$wd" \
    --epoch-checkpoint-interval 10000 --checkpoint-hist 0 \
    --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
    --wq-per-channel --aq-per-channel --aq-clip-learnable \
    --use-kd --kd-hard-and-soft 1 --quantized --qk-reparam --qk-reparam-type 0 \
    --amp --amp-dtype bf16 \
    --extra-arg=--skip_validate \
    --extra-arg=--max_train_updates --extra-arg="$STEPS" \
    --extra-arg=--log-interval --extra-arg="$LOGINTERVAL" \
    --extra-arg=--seed --extra-arg=42 \
    > "$log" 2>&1
  echo "[smoke] $name exit=$? log=$log"
}

case ",$RUNS," in
  *,deit,*) run_one deit deit_tiny_distilled_patch16_224 deit \
              third_party/OFQ/configs/deit_default_imagent.attn_q.yml 5e-4 5e-2 "$PORT" \
              "$LOGDIR/deit_${TAG}_bs${BATCH}_${STEPS}step_gpu${GPU}.log" ;;
esac

case ",$RUNS," in
  *,swin,*) run_one swin swin_t swin \
              third_party/OFQ/configs/swin_t_imagenet.attn_q.yml 2e-4 0.0 "$((PORT + 1))" \
              "$LOGDIR/swin_${TAG}_bs${BATCH}_${STEPS}step_gpu${GPU}.log" ;;
esac

echo "[smoke] done; logs in $LOGDIR"
