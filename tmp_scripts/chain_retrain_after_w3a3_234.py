#!/usr/bin/env python3
"""Wait for W3A3 {2,3,4} gate search completion, extract frozen profile, and launch static 20ep retrain."""

import datetime
import os
import re
import subprocess
import sys
import time
from typing import List, Tuple

QLOG = "/datadisk2/quyanyi/qat_runs/chain_retrain_w3a3_234_20261127.nohup.log"
SEARCH_EXP = "deit_tiny_mixed_bit_gate_w3a3_234_20ep_4gpu_20261127"
SEARCH_LOG = f"/datadisk2/quyanyi/qat_runs/{SEARCH_EXP}.log"
TEMPLATE_SH = "/home_ext/quyanyi/tiger/resume_repos/QATs/tmp_scripts/run_deit_tiny_qat_sub24_v2shape_20ep_4gpu_20261126.sh"
RETRAIN_EXP = "deit_tiny_qat_sub24_w3a3_234shape_20ep_4gpu_20261128"
RETRAIN_SH = f"/home_ext/quyanyi/tiger/resume_repos/QATs/tmp_scripts/run_{RETRAIN_EXP}.sh"
RETRAIN_NOHUP_LOG = "/datadisk2/quyanyi/qat_runs/qat_sub24_w3a3_234shape_20261128.nohup.log"
WATCH_PIDS = (2003830, 2003841)
ALLOWED_TOKENS = {"T", "2", "3", "4"}


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%F %T")
    with open(QLOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def extract_frozen_profile(text: str) -> Tuple[str, List[str]]:
    val_matches = re.findall(r"Test:\s*\[distributed-summary\]", text)
    if len(val_matches) < 13:
        raise RuntimeError(
            f"Expected >= 13 validation epochs in search log before chaining, found {len(val_matches)}"
        )
    frozen_matches = re.findall(r"MixedBitGateFrozen:.*?\bprofile=([A-Za-z0-9:,]+)", text)
    if not frozen_matches:
        raise RuntimeError("Could not find MixedBitGateFrozen profile in search log")
    raw_profile = frozen_matches[-1]
    bits: List[str] = []
    for item in raw_profile.split(","):
        if ":" not in item:
            raise ValueError(f"Unexpected profile item {item!r} in {raw_profile!r}")
        _name, tok = item.split(":", 1)
        tok = tok.strip().upper()
        if tok not in ALLOWED_TOKENS:
            raise ValueError(f"Unsupported bit token {tok!r} in {raw_profile!r}")
        bits.append(tok)
    if len(bits) != 24:
        raise ValueError(f"Expected 24 submodule tokens, got {len(bits)}: {bits}")
    return ",".join(bits), bits


def render_retrain_script(template_text: str, profile_csv: str) -> str:
    replacements = [
        (
            "DeiT-Tiny static 24-submodule retrain of v2-discovered profile",
            "DeiT-Tiny static 24-submodule retrain of w3a3_234-discovered profile",
        ),
        (
            'EXP="${EXP:-deit_tiny_qat_sub24_v2shape_20ep_4gpu_20261126}"',
            f'EXP="${{EXP:-{RETRAIN_EXP}}}"',
        ),
        (
            'MASTER_PORT="${MASTER_PORT:-30985}"',
            'MASTER_PORT="${MASTER_PORT:-30988}"',
        ),
        (
            'DEPTH_BIT_PROFILE="${DEPTH_BIT_PROFILE:-T,3,2,3,T,T,3,2,3,T,2,T,3,T,3,T,T,T,T,T,2,T,3,T}"',
            f'DEPTH_BIT_PROFILE="${{DEPTH_BIT_PROFILE:-{profile_csv}}}"',
        ),
        (
            "--wbits 2 --abits 2",
            "--wbits 3 --abits 3",
        ),
    ]
    out = template_text
    for old, new in replacements:
        if out.count(old) != 1:
            raise ValueError(f"Template replacement mismatch for: {old!r} (count={out.count(old)})")
        out = out.replace(old, new, 1)
    return out


def is_search_alive() -> bool:
    for pid in WATCH_PIDS:
        if os.path.exists(f"/proc/{pid}"):
            return True
    self_pids = {os.getpid(), os.getppid()}
    r = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=True)
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("PID"):
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid in self_pids:
            continue
        if SEARCH_EXP in parts[1]:
            return True
    return False


def gpus_0_to_3_free() -> bool:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    mems = {}
    for line in r.stdout.strip().splitlines():
        if not line.strip():
            continue
        idx_s, mem_s = line.split(",", 1)
        mems[int(idx_s.strip())] = int(mem_s.strip())
    return all(mems.get(i, 99999) < 500 for i in (0, 1, 2, 3))


def main() -> int:
    log(f"Auto-chain watcher started for {SEARCH_EXP} (watch_pids={WATCH_PIDS})")
    while True:
        alive = is_search_alive()
        free = gpus_0_to_3_free()
        if (not alive) and free:
            break
        time.sleep(15)

    log("Search process exited and GPUs 0-3 < 500 MiB; sleeping 10s before parsing log")
    time.sleep(10)

    with open(SEARCH_LOG, "r", errors="ignore") as f:
        text = f.read()
    profile_csv, bits = extract_frozen_profile(text)
    log(f"Extracted frozen 24-submodule profile: {profile_csv} (counts={ {k: bits.count(k) for k in ('2', '3', '4', 'T') if bits.count(k)} })")

    with open(TEMPLATE_SH, "r") as f:
        tpl = f.read()
    rendered = render_retrain_script(tpl, profile_csv)
    with open(RETRAIN_SH, "w") as f:
        f.write(rendered)
    os.chmod(RETRAIN_SH, 0o755)
    log(f"Wrote static retrain script: {RETRAIN_SH}")

    if not gpus_0_to_3_free():
        raise RuntimeError("GPUs 0-3 became occupied before launch")

    with open(RETRAIN_NOHUP_LOG, "ab") as lf:
        p = subprocess.Popen(
            ["bash", RETRAIN_SH],
            cwd="/home_ext/quyanyi/tiger/resume_repos/QATs",
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    log(f"Launched static retrain {RETRAIN_EXP} PID={p.pid}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ERROR: {exc!r}")
        raise
