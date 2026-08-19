#!/usr/bin/env bash
set -euo pipefail

EXP="${EXP:-ofq_resume10_to60_sparse_pulse_prevstep_refkl_c_20260709}"
BASELINE="${BASELINE:-80.5980}"
TARGET="${TARGET:-81.0}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
OUT="${OUT:-/mlx_devbox/users/quyanyi/playground/qat_public_repro/${EXP}}"
REMOTE_HOST="${REMOTE_HOST:-fdbd:dccd:cdc2:1234:0:b8::}"
REMOTE_PORT="${REMOTE_PORT:-9801}"
REMOTE_OUT="${REMOTE_OUT:-/tmp/qat_public_repro/${EXP}}"
DOC_DIR="${DOC_DIR:-/mlx_devbox/users/quyanyi/playground/QATs/docs}"
STATUS="${STATUS:-${DOC_DIR}/ofq_sparse_pulse_prevstep_refkl_c_50epoch_status_20260709.tsv}"
REF_STATUS="${REF_STATUS:-${DOC_DIR}/ofq_sparse_pulse_prevstep_refkl_c_50epoch_refw_20260709.tsv}"
SUMMARY="${SUMMARY:-${DOC_DIR}/ofq_sparse_pulse_prevstep_refkl_c_50epoch_monitor_summary_20260709.txt}"

mkdir -p "${DOC_DIR}"
ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

{
  printf 'timestamp\tcheckpoint\tloss\tacc1\tacc5\tsamples\tdelta_vs_80.5980\tstatus\n'
  if [[ -f "${LOG}" ]]; then
    awk -v ts="${ts}" -v baseline="${BASELINE}" '
      /Test: \[distributed-summary\]/ {
        loss="NA"; acc1="NA"; acc5="NA"; samples="NA";
        for (i=1; i<=NF; i++) {
          if ($i == "Loss:") loss=$(i+1);
          if ($i == "Acc@1:") acc1=$(i+1);
          if ($i == "Acc@5:") acc5=$(i+1);
          if ($i == "Samples:") samples=$(i+1);
        }
        vals[++n] = loss "\t" acc1 "\t" acc5 "\t" samples;
      }
      END {
        for (i=1; i<=n; i++) {
          split(vals[i], a, "\t");
          ckpt = 10 + i;
          delta = (a[2] == "NA") ? "NA" : sprintf("%.4f", a[2] - baseline);
          status = "below_baseline";
          if (a[2] != "NA" && a[2] > baseline) status = "above_baseline";
          if (a[2] != "NA" && a[2] >= 81.0) status = "target_81_reached";
          printf "%s\tcheckpoint-%d\t%s\t%s\t%s\t%s\t%s\t%s\n", ts, ckpt, a[1], a[2], a[3], a[4], delta, status;
        }
      }
    ' "${LOG}"
  fi
} > "${STATUS}"

{
  printf 'timestamp\tepoch\trefw\trefattnkl\tline\n'
  if [[ -f "${LOG}" ]]; then
    awk -v ts="${ts}" '
      /Train:/ && /RefW:/ {
        epoch="NA"; refw="NA"; refkl="NA";
        for (i=1; i<=NF; i++) {
          if ($i == "Train:") epoch=$(i+1);
          if ($i == "RefAttnKL:") refkl=$(i+1);
          if ($i == "RefW:") refw=$(i+1);
        }
        gsub(/\r/, "", $0);
        printf "%s\t%s\t%s\t%s\t%s\n", ts, epoch, refw, refkl, $0;
      }
    ' "${LOG}"
  fi
} > "${REF_STATUS}"

ckpt_count=0
latest_ckpt="NA"
output_exists="no"
if [[ -n "${REMOTE_HOST}" ]] && ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "test -d '${REMOTE_OUT}'" >/dev/null 2>&1; then
  output_exists="remote:${REMOTE_OUT}"
  ckpt_count="$(ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "find '${REMOTE_OUT}' -maxdepth 1 -type f -name 'checkpoint-*.pth.tar' | wc -l" | tr -d ' ')"
  latest_ckpt="$(ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "find '${REMOTE_OUT}' -maxdepth 1 -type f -name 'checkpoint-*.pth.tar' -printf '%f\n' | sort -V | tail -n 1")"
  [[ -n "${latest_ckpt}" ]] || latest_ckpt="NA"
elif [[ -d "${OUT}" ]]; then
  output_exists="local:${OUT}"
  ckpt_count="$(find "${OUT}" -maxdepth 1 -type f -name 'checkpoint-*.pth.tar' | wc -l | tr -d ' ')"
  latest_ckpt="$(find "${OUT}" -maxdepth 1 -type f -name 'checkpoint-*.pth.tar' -printf '%f\n' | sort -V | tail -n 1)"
  [[ -n "${latest_ckpt}" ]] || latest_ckpt="NA"
fi

best_line="NA"
if [[ -s "${STATUS}" ]]; then
  best_line="$(awk -F '\t' 'NR > 1 && $4 != "NA" { if (!seen || $4+0 > best+0) { best=$4; line=$0; seen=1 } } END { if (seen) print line; else print "NA" }' "${STATUS}")"
fi

pulse_refw="NA"
if [[ -s "${REF_STATUS}" ]]; then
  pulse_refw="$(awk -F '\t' 'NR > 1 && ($2==28 || $2==29 || $2==36 || $2==37 || $2==44 || $2==45 || $2==52 || $2==53) && $3 != "0.000e+00" { c++ } END { print c+0 }' "${REF_STATUS}")"
fi

{
  printf 'timestamp=%s\n' "${ts}"
  printf 'log=%s\n' "${LOG}"
  printf 'output=%s\n' "${OUT}"
  printf 'remote_output=%s:%s\n' "${REMOTE_HOST}" "${REMOTE_OUT}"
  printf 'log_exists=%s\n' "$([[ -f "${LOG}" ]] && echo yes || echo no)"
  printf 'output_exists=%s\n' "${output_exists}"
  printf 'checkpoint_count=%s\n' "${ckpt_count}"
  printf 'latest_checkpoint=%s\n' "${latest_ckpt}"
  printf 'status_tsv=%s\n' "${STATUS}"
  printf 'refw_tsv=%s\n' "${REF_STATUS}"
  printf 'best_fullval_line=%s\n' "${best_line}"
  printf 'pulse_refw_nonzero_lines=%s\n' "${pulse_refw}"
} > "${SUMMARY}"

printf '%s\n' "${SUMMARY}"
