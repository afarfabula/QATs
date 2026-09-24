#!/usr/bin/env bash
# Sweep the classification-logits ranking weight for Swin-T W4A4 OFQ/QAT.
# Each config runs on its own GPU in parallel, same recipe as the planned 100-epoch run
# (pretrained student + FP teacher checkpoint, KD soft temperature 2.75, effective batch 512).
#
# Usage: STEPS=1500 CONFIGS="0:2 0.04:4 0.12:5 0.4:6" bash tmp_scripts/sweep_swin_w4a4_logitrank_weight_4gpu_20260924.sh
set -uo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
DATA="${DATA:-/datadisk2/linyichen/OFQ/ImageNet-1K}"
OUT="${OUT:-/tmp/qat_sweep_20260924/out}"
LOGDIR="${LOGDIR:-/tmp/qat_sweep_20260924}"
TEACHER="${TEACHER:-/home_ext/quyanyi/qat_weights/swin_t-704ceda3.pth}"
PY="${PY:-/datadisk2/quyanyi/envs/qat_env/bin/python}"
STEPS="${STEPS:-1500}"        # micro-steps per config
CONFIGS="${CONFIGS:-0:2 0.04:4 0.12:5 0.4:6}"   # weight:gpu
MASTER_PORT="${MASTER_PORT:-30700}"

mkdir -p "${OUT}" "${LOGDIR}"
pids=()

for cfg in ${CONFIGS}; do
  w="${cfg%%:*}"; gpu="${cfg##*:}"
  tag="w$(echo "${w}" | tr -d '.')"
  exp="sweep_swin_w4a4_rank${tag}_$(date '+%m%d')"
  log="${LOGDIR}/${tag}.log"
  port=$((MASTER_PORT + gpu))
  echo "[sweep] weight=${w} gpu=${gpu} steps=${STEPS} log=${log}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  "${PY}" "${QATS}/qat_launch.py" \
    --method ofq --stage train \
    --config "${QATS}/third_party/OFQ/configs/swin_t_imagenet.attn_q.yml" \
    --model swin_t --data "${DATA}" --dataset-format folder \
    --output "${OUT}" --experiment "${exp}" \
    --devices "${gpu}" --nproc-per-node 1 --master-port "${port}" \
    --model-type swin \
    --teacher swin_t --teacher-type swin --teacher-pretrained \
    --teacher-checkpoint "${TEACHER}" \
    --epochs 1 --scheduler-epochs 1 --batch-size 32 --workers 6 \
    --lr 2e-4 --min-lr 2e-4 --weight-decay 0.0 --grad-accum-steps 16 \
    --epoch-checkpoint-interval 10000 --checkpoint-hist 0 \
    --wbits 4 --abits 4 --wq-mode statsq --aq-mode lsq \
    --wq-per-channel --aq-per-channel --aq-clip-learnable \
    --pretrained --pretrained-initialized \
    --use-kd --kd-hard-and-soft 0 --teacher-soft-temperature 2.75 \
    --logit-rank-weight "${w}" --logit-rank-topk 5 \
    --quantized --qk-reparam --qk-reparam-type 0 \
    --amp --amp-dtype bf16 \
    --extra-arg=--smoothing --extra-arg=0.1 \
    --extra-arg=--mixup --extra-arg=0.0 \
    --extra-arg=--cutmix --extra-arg=0.0 \
    --extra-arg=--aa --extra-arg=rand-m9-mstd0.5-inc1 \
    --extra-arg=--color-jitter --extra-arg=0.4 \
    --extra-arg=--reprob --extra-arg=0.25 \
    --extra-arg=--skip_validate \
    --extra-arg=--max_train_updates --extra-arg="${STEPS}" \
    --extra-arg=--log-interval --extra-arg=100 \
    --extra-arg=--seed --extra-arg=42 \
    > "${log}" 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do wait "${pid}"; done

echo
echo "===== sweep summary (${LOGDIR}) ====="
python3 - "${LOGDIR}" <<'PY'
import re, statistics, sys, pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob("w*.log")):
    txt = p.read_text(errors="ignore")
    base = [float(x) for x in re.findall(r"BaseLoss: [\d.]+ \(([\d.]+)\)", txt)]
    rank = [float(x) for x in re.findall(r"LogitRank: [\d.eE+-]+ \(([\d.eE+-]+)\)", txt)]
    t = [float(x) for x in re.findall(r"Time: ([\d.]+)s,", txt)]
    if not base:
        print(f"{p.name:12s} no data (check log)")
        continue
    st = t[2:] if len(t) > 2 else t
    print(f"{p.name:12s} BaseLoss {base[0]:8.4f} -> {base[-1]:8.4f} | LogitRank {rank[0]:.4f} -> {rank[-1]:.4f} "
          f"| step median {statistics.median(st):.3f}s")
PY
