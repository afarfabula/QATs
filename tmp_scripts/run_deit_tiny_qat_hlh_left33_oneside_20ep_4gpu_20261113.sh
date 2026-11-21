#!/usr/bin/env bash
# Only the left boundary shoulder is promoted from 2-bit to 3-bit:
#   blocks 0,1    W3A3
#   blocks 2-9    W-T/A-T
#   blocks 10     W2A2
#   block 11      W3A3
#
# Effective entropy-based block mean:
#   (3+3+2+3 + 8*log2(3)) / 12 = 1.9733 bit
set -euo pipefail

QATS="${QATS:-/home_ext/quyanyi/tiger/resume_repos/QATs}"
export QATS
export DEPTH_BIT_PROFILE="${DEPTH_BIT_PROFILE:-3,3,T,T,T,T,T,T,T,T,2,3}"
export EXP="${EXP:-deit_tiny_qat_hlh_left33_oneside_20ep_4gpu_20261113}"

exec bash "${QATS}/tmp_scripts/run_deit_tiny_qat_depth_profile_hlh_20ep_4gpu_20261112.sh"
