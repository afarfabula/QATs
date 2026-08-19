#!/usr/bin/env bash
set -euo pipefail
for exp in \
  swin_t_w4a4_stage2_prevstep_custom10_from50_to51_bsz128_kd_20260630 \
  swin_t_w4a4_stage2_prevstep_custom10_from50_to51_bsz128_kd_ref1e4_20260630 \
  swin_t_w4a4_stage2_prevstep_custom10_from50_to51_bsz128_kd_ref1e5_lr1e5_20260630 \
  swin_t_w4a4_stage2_prevstep_custom10_from50_to51_bsz128_kd_ema999_20260630; do
  echo "--- ${exp}"
  grep -E 'TrainSummary|Test: \[distributed-summary\]|Enabled EMA refmodel|Enabled student weight EMA|Effective batch' "/tmp/${exp}.log" 2>/dev/null | tail -20 || true
  ls -lh "/tmp/qats_stage2_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/${exp}" 2>/dev/null | tail -10 || true
done
