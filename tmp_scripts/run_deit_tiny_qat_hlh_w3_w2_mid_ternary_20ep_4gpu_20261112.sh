#!/usr/bin/env bash
# H-L-H precision-profile QAT:
#   blocks 0,11      W3A3
#   blocks 1,10      W2A2
#   blocks 2-9       W-T/A-T
#
# Effective entropy-based block mean:
#   (2*3 + 2*2 + 8*log2(3)) / 12 = 1.8900 bit
#
# The local uniform W2A2 baseline already exists, so it does not need rerunning.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
export QATS
export DEPTH_BIT_PROFILE="${DEPTH_BIT_PROFILE:-3,2,T,T,T,T,T,T,T,T,2,3}"
export EXP="${EXP:-deit_tiny_qat_hlh_w3_w2_mid_ternary_20ep_4gpu_20261112}"

exec bash "${QATS}/tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh"
