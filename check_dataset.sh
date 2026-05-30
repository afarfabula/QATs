#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-/tmp/qats/imagenet1k}"
IMG_ROOT="${IMG_ROOT:-${DATA_ROOT}/imagefolder}"
PARQUET_ROOT="${PARQUET_ROOT:-${DATA_ROOT}/parquet}"
USE_PARQUET_EXPORT="${USE_PARQUET_EXPORT:-1}"
HF_REPO_ID="${HF_REPO_ID:-ILSVRC/imagenet-1k}"
SMOKE_DATASET="${SMOKE_DATASET:-0}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HF_HUB_TOKEN:-}}}"

mkdir -p "${IMG_ROOT}" "${PARQUET_ROOT}"
export ROOT_DIR PYTHON_BIN DATA_ROOT IMG_ROOT PARQUET_ROOT USE_PARQUET_EXPORT HF_REPO_ID SMOKE_DATASET

mkdir -p "${PARQUET_ROOT}/data"

if compgen -G "${PARQUET_ROOT}/data/train-*.parquet" > /dev/null && compgen -G "${PARQUET_ROOT}/data/validation-*.parquet" > /dev/null; then
  echo "[QATs] parquet dataset already exists at ${PARQUET_ROOT}"
else
  if [[ "${SMOKE_DATASET}" == "1" ]]; then
    echo "[QATs] smoke mode enabled, will generate local smoke imagefolder and parquet dataset"
  else
    if [[ -z "${HF_TOKEN}" ]]; then
      echo "[QATs] missing HF token. Please export HF_TOKEN (or HF_HUB_TOKEN / HUGGINGFACE_HUB_TOKEN) after accepting ImageNet terms on Hugging Face." >&2
      exit 1
    fi
    echo "[QATs] parquet dataset not found, downloading ImageNet-1k parquet shards from Hugging Face..."
    export HF_TOKEN
    "${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["HF_REPO_ID"]
parquet_root = os.environ["PARQUET_ROOT"]
token = os.environ["HF_TOKEN"]

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    token=token,
    local_dir=parquet_root,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["data/train-*.parquet", "data/validation-*.parquet", "*.json", "README*"],
)
print(f"[QATs] downloaded parquet dataset snapshot to {parquet_root}")
PY
  fi
fi

if [[ -d "${IMG_ROOT}/train" && -d "${IMG_ROOT}/val" ]]; then
  echo "[QATs] ImageNet imagefolder dataset already exists at ${IMG_ROOT}"
else
  if [[ "${SMOKE_DATASET}" == "1" ]]; then
    echo "[QATs] generating smoke imagefolder dataset at ${IMG_ROOT}"
    "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import numpy as np
from PIL import Image

img_root = Path(os.environ["IMG_ROOT"])
rng = np.random.default_rng(0)

train_num_classes = int(os.environ.get("SMOKE_TRAIN_CLASSES", "32"))
train_samples_per_class = int(os.environ.get("SMOKE_TRAIN_SAMPLES_PER_CLASS", "256"))
val_num_classes = int(os.environ.get("SMOKE_VAL_CLASSES", str(train_num_classes)))
val_samples_per_class = int(os.environ.get("SMOKE_VAL_SAMPLES_PER_CLASS", "32"))

spec = {
    "train": (train_num_classes, train_samples_per_class),
    "val": (val_num_classes, val_samples_per_class),
}

for split, (num_classes, samples_per_class) in spec.items():
    for class_id in range(num_classes):
        class_dir = img_root / split / f"class_{class_id:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in range(samples_per_class):
            array = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
            Image.fromarray(array).save(class_dir / f"{sample_id:05d}.jpg", quality=90)

print(f"[QATs] smoke dataset ready at {img_root}")
PY
  fi
fi

if [[ "${USE_PARQUET_EXPORT}" == "1" ]]; then
  if compgen -G "${PARQUET_ROOT}/data/train-*.parquet" > /dev/null && compgen -G "${PARQUET_ROOT}/data/validation-*.parquet" > /dev/null; then
    echo "[QATs] parquet dataset already exists at ${PARQUET_ROOT}"
  else
    echo "[QATs] exporting parquet shards for Q-ViT/OFQ compatibility..."
    mkdir -p "${PARQUET_ROOT}/data"
    "${PYTHON_BIN}" - <<'PY'
import io
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from torchvision.datasets import ImageFolder

img_root = Path(os.environ["IMG_ROOT"])
parquet_root = Path(os.environ["PARQUET_ROOT"]) / "data"
rows_per_file = int(os.environ.get("ROWS_PER_FILE", "2048"))

def export_split(split_name: str, target_name: str) -> None:
    dataset = ImageFolder(str(img_root / split_name))
    rows = []
    shard_id = 0
    for idx, (path, label) in enumerate(dataset.samples):
        with open(path, "rb") as f:
            img_bytes = f.read()
        rows.append({"image": {"bytes": img_bytes}, "label": int(label)})
        if len(rows) >= rows_per_file:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, parquet_root / f"{target_name}-{shard_id:05d}.parquet")
            rows.clear()
            shard_id += 1
    if rows:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, parquet_root / f"{target_name}-{shard_id:05d}.parquet")

export_split("train", "train")
export_split("val", "validation")
print(f"[QATs] parquet export finished at {parquet_root}")
PY
  fi
fi

echo "[QATs] dataset check finished"
