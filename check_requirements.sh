#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[QATs] root=${ROOT_DIR}"
echo "[QATs] python=${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import importlib
import subprocess
import sys

required = {
    "numpy": "numpy",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "torchvision": "torchvision",
    "timm": "timm==0.9.16",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "requests": "requests",
    "huggingface_hub": "huggingface_hub",
    "tqdm": "tqdm",
    "google.protobuf": "protobuf<3.21",
}

missing = []
for module_name, package_name in required.items():
    try:
        importlib.import_module(module_name)
        print(f"[QATs] ok: {module_name}")
    except Exception:
        print(f"[QATs] missing: {module_name} -> {package_name}")
        missing.append(package_name)

if missing:
    cmd = [sys.executable, "-m", "pip", "install", *missing]
    print("[QATs] installing:", " ".join(missing))
    subprocess.check_call(cmd)
else:
    print("[QATs] all python dependencies are already installed")
PY

echo "[QATs] dependency check finished"
