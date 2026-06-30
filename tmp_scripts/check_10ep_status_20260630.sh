#!/usr/bin/env bash
set -euo pipefail
echo PROCS
pgrep -af 'stage1_bs256_kd_noaug_attncopyfix_10ep|qat_launch.py' || true
echo RUNNER
if [ -f /tmp/stage1_10ep_kd_noaug_attnfix_fastval_20260630.runner.log ]; then tail -n 120 /tmp/stage1_10ep_kd_noaug_attnfix_fastval_20260630.runner.log; else echo missing; fi
echo TRAINLOG
if [ -f /tmp/train_swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep_fastval_20260630.log ]; then tail -n 100 /tmp/train_swin_t_w4a4_stage1_bs256_kd_noaug_attncopyfix_10ep_fastval_20260630.log; else echo missing; fi
