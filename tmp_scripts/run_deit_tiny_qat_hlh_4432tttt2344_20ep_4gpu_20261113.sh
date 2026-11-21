#!/usr/bin/env bash
# Asymmetric-to-ternary H-L-H QAT profile:
#   blocks 0,1,10,11  W4A4
#   blocks 2,9        W3A3
#   blocks 3,8        W2A2
#   blocks 4-7        W-T/A-T
#
# Effective entropy-based block mean:
#   (8 non-ternary values + 4*log2(3)) / 12 = 2.6950 bit
# This is 10.17% below uniform W3A3.
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
export QATS
export DEPTH_BIT_PROFILE="${DEPTH_BIT_PROFILE:-4,4,3,2,T,T,T,T,2,3,4,4}"
export EXP="${EXP:-deit_tiny_qat_hlh_4432tttt2344_20ep_4gpu_20261113}"

exec bash "${QATS}/tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh"
