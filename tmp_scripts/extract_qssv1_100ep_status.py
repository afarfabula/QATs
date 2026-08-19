#!/usr/bin/env python3
from __future__ import annotations
import re
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("usage: extract_qssv1_100ep_status.py <log> <out_dir>")
log = Path(sys.argv[1])
out = Path(sys.argv[2])
text = log.read_text(errors="ignore") if log.exists() else ""
acc = [(float(a), float(b)) for a, b in re.findall(r"\*\s+Acc@1\s+([0-9.]+)\s+Acc@5\s+([0-9.]+)", text)]
train_summaries = [(int(e), float(t), float(sps)) for e, t, sps in re.findall(r"TrainSummary:\s+epoch=(\d+).*?avg_step_time=([0-9.]+)s.*?samples_per_sec=([0-9.]+)", text)]
ckpts = {}
if out.exists():
    for p in out.glob("checkpoint-*.pth.tar"):
        m = re.fullmatch(r"checkpoint-(\d+)\.pth\.tar", p.name)
        if m:
            ckpts[int(m.group(1))] = p
print("epoch\tacc1\tacc5\tcheckpoint\tckpt_bytes")
for i, epoch in enumerate(range(10, 101, 10), start=1):
    a = acc[i - 1] if i <= len(acc) else (None, None)
    p = ckpts.get(epoch)
    print(f"{epoch}\t{a[0]}\t{a[1]}\t{p if p else None}\t{p.stat().st_size if p else None}")
print("\nlatest_train_summary:")
if train_summaries:
    e, t, sps = train_summaries[-1]
    print(f"epoch={e} avg_step_time={t} samples_per_sec={sps}")
else:
    print("None")
