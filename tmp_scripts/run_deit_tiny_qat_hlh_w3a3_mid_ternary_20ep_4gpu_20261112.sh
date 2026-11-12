#!/usr/bin/env bash
# H-L-H precision-profile QAT using 3-bit at the ends and true ternary
# {-1,0,1} in the middle:
#   blocks 0-1    W3A3
#   blocks 2-9    W-T/A-T
#   blocks 10-11  W3A3
#
# Effective entropy-based block mean:
#   (4*3 + 8*log2(3)) / 12 = 2.0566 bit
#
# This is close to, but not exactly equal to, uniform W2A2. The exact realized
# storage and bit-op costs must be included in the comparison with the current
# uniform W2A2 baseline.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
export QATS
export DEPTH_BIT_PROFILE="${DEPTH_BIT_PROFILE:-3,3,T,T,T,T,T,T,T,T,3,3}"
export EXP="${EXP:-deit_tiny_qat_hlh_w3a3_mid_ternary_20ep_4gpu_20261112}"

exec bash "${QATS}/tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh"
