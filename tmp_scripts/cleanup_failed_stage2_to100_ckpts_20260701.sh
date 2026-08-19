#!/usr/bin/env bash
set -euo pipefail
for d in \
  /tmp/qats_stage2_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage2_prevstep_custom10_from50_to100_bsz128_kd_ema999_20260630 \
  /tmp/qats_stage2_outputs/ofq_prevstep_attn_kl_headtop5_303opt_20260624/swin_t_w4a4_stage2_prevstep_custom10_from50_to100_bsz192_nokd_20260630
 do
  if [[ -d "$d" ]]; then
    echo "DIR $d"
    echo "before: $(du -sh "$d" | awk '{print $1}')"
    find "$d" -maxdepth 1 -type f \( -name 'checkpoint*.pth.tar' -o -name 'model_best*.pth.tar' \) -print -delete
    echo "after: $(du -sh "$d" | awk '{print $1}')"
  fi
done
echo "total: $(du -sh /tmp/qats_stage2_outputs | awk '{print $1}')"
