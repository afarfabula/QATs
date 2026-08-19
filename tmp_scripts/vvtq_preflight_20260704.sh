#!/usr/bin/env bash
set -euo pipefail

VVTQ_DIR="${VVTQ_DIR:-/mlx_devbox/users/quyanyi/playground/Quantization-Variation}"
DATA_DIR="${DATA_DIR:-/tmp/qats/imagenet1k/imagefolder}"
SOFTLABEL_DIR="${SOFTLABEL_DIR:-${VVTQ_DIR}/FKD_soft_label_500_crops_marginal_smoothing_k_5}"
OUT_DIR="${OUT_DIR:-/tmp/qat_public_repro/vvtq_preflight_20260704}"
LOG="${LOG:-/tmp/vvtq_preflight_20260704.log}"

mkdir -p "${OUT_DIR}"
export VVTQ_DIR DATA_DIR SOFTLABEL_DIR OUT_DIR

{
  echo "===== VVTQ preflight $(date '+%F %T') ====="
  echo "VVTQ_DIR=${VVTQ_DIR}"
  echo "DATA_DIR=${DATA_DIR}"
  echo "SOFTLABEL_DIR=${SOFTLABEL_DIR}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo

  if [[ ! -d "${VVTQ_DIR}" ]]; then
    echo "FAIL: missing VVTQ_DIR"
    exit 2
  fi

  git -C "${VVTQ_DIR}" rev-parse --short HEAD || true

  python3 - <<'PY'
import os
import sys
import torch
import timm
import torchvision

vvtq_dir = os.environ["VVTQ_DIR"]
data_dir = os.environ["DATA_DIR"]
softlabel_dir = os.environ["SOFTLABEL_DIR"]
sys.path.insert(0, vvtq_dir)

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("timm", timm.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
print("data_train_exists", os.path.isdir(os.path.join(data_dir, "train")))
print("data_val_exists", os.path.isdir(os.path.join(data_dir, "val")))
print("softlabel_exists", os.path.isdir(softlabel_dir))

from timm.models import create_model
import quantization.Swin_quant  # registers model with timm

model = create_model(
    "swin_tiny_patch4_window7_224_quant",
    pretrained=False,
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    wbits=4,
    abits=4,
    act_layer=torch.nn.GELU,
    offset=False,
    learned=True,
    mixpre=False,
    headwise=False,
)
print("model_built", type(model).__name__)
print("patch_embed_proj_nbits", getattr(model.patch_embed.proj, "nbits", None).detach().cpu().tolist())
print("head_nbits", getattr(model.head, "nbits", None).detach().cpu().tolist())
print("first_block_attn_q_nbits", getattr(model.layers[0].blocks[0].attn.proj_q, "nbits", None).detach().cpu().tolist())

if not os.path.isdir(os.path.join(data_dir, "train")) or not os.path.isdir(os.path.join(data_dir, "val")):
    print("BLOCKER: VVTQ requires ImageFolder train/ and val/ directories.")
if not os.path.isdir(softlabel_dir):
    print("BLOCKER: VVTQ official training requires FKD soft-label directory.")
PY
} 2>&1 | tee "${LOG}"

echo "preflight_log=${LOG}"
