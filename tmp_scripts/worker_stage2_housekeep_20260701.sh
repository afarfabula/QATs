#!/usr/bin/env bash
set -euo pipefail
echo "WORKER=$(hostname)"
echo '---procs---'
ps -eo pid,ppid,pgid,cmd | grep -E 'qat_launch|run_stage2|torchrun|python.*train' | grep -v grep || true
echo '---outputs---'
du -sh /tmp/qats_stage2_outputs 2>/dev/null || true
find /tmp/qats_stage2_outputs -maxdepth 3 -type d -name '*to100*' -print 2>/dev/null | head -20 || true
echo '---logs---'
ls -lh /tmp/*stage2*to100*.log 2>/dev/null || true
