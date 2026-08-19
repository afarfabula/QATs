#!/usr/bin/env bash
set -euo pipefail

EXP="${EXP:-ofq_100ep_fromscratch_original_ofq_public_control_20260714}"
BASELINE="${BASELINE:-80.5980}"
SCHEME_C_BEST="${SCHEME_C_BEST:-80.6820}"
ORIGINAL_10TO60_BEST="${ORIGINAL_10TO60_BEST:-80.7240}"
ORIGINAL_10TO110_BEST="${ORIGINAL_10TO110_BEST:-80.7520}"
DYNAMIC_10TO110_BEST="${DYNAMIC_10TO110_BEST:-80.7600}"
KL_100EP_BEST="${KL_100EP_BEST:-80.7720}"
LATE_10TO210_BEST="${LATE_10TO210_BEST:-80.8280}"
LOG="${LOG:-/mlx_devbox/users/quyanyi/playground/train_${EXP}.log}"
REMOTE_HOST="${REMOTE_HOST:-fdbd:dccd:cdc2:1234:0:b8::}"
REMOTE_PORT="${REMOTE_PORT:-9801}"
REMOTE_OUT="${REMOTE_OUT:-/tmp/qat_public_repro/${EXP}}"
DOC_DIR="${DOC_DIR:-/mlx_devbox/users/quyanyi/playground/QATs/docs}"
STATUS="${STATUS:-${DOC_DIR}/ofq_100ep_fromscratch_original_ofq_public_control_status_20260714.tsv}"
REF_STATUS="${REF_STATUS:-${DOC_DIR}/ofq_100ep_fromscratch_original_ofq_public_control_refw_20260714.tsv}"
SUMMARY="${SUMMARY:-${DOC_DIR}/ofq_100ep_fromscratch_original_ofq_public_control_monitor_summary_20260714.txt}"

mkdir -p "${DOC_DIR}"
ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

{
  printf 'timestamp\tcheckpoint\tloss\tacc1\tacc5\tsamples\tdelta_vs_80.5980\tdelta_vs_scheme_c_80.6820\tdelta_vs_original10to60_80.7240\tdelta_vs_original10to110_80.7520\tdelta_vs_dynamic10to110_80.7600\tdelta_vs_kl100ep_80.7720\tdelta_vs_late10to210_80.8280\tstatus\n'
  if [[ -f "${LOG}" ]]; then
    awk -v ts="${ts}" -v baseline="${BASELINE}" -v scheme_c="${SCHEME_C_BEST}" -v original60="${ORIGINAL_10TO60_BEST}" -v original110="${ORIGINAL_10TO110_BEST}" -v dynamic110="${DYNAMIC_10TO110_BEST}" -v kl100="${KL_100EP_BEST}" -v late210="${LATE_10TO210_BEST}" '
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
          ckpt = i;
          delta_base = (a[2] == "NA") ? "NA" : sprintf("%.4f", a[2] - baseline);
          delta_scheme = (a[2] == "NA") ? "NA" : sprintf("%.4f", a[2] - scheme_c);
          delta_original60 = (a[2] == "NA") ? "NA" : sprintf("%.4f", a[2] - original60);
          delta_original110 = (a[2] == "NA") ? "NA" : sprintf("%.4f", a[2] - original110);
          delta_dynamic110 = (a[2] == "NA") ? "NA" : sprintf("%.4f", a[2] - dynamic110);
          delta_kl100 = (a[2] == "NA") ? "NA" : sprintf("%.4f", a[2] - kl100);
          delta_late210 = (a[2] == "NA") ? "NA" : sprintf("%.4f", a[2] - late210);
          status = "below_baseline";
          if (a[2] != "NA" && a[2] > baseline) status = "above_baseline";
          if (a[2] != "NA" && a[2] > scheme_c) status = "above_scheme_c";
          if (a[2] != "NA" && a[2] > original60) status = "above_original10to60_best";
          if (a[2] != "NA" && a[2] > original110) status = "above_original10to110_best";
          if (a[2] != "NA" && a[2] > dynamic110) status = "above_dynamic10to110_best";
          if (a[2] != "NA" && a[2] > kl100) status = "above_kl100ep_best";
          if (a[2] != "NA" && a[2] > late210) status = "above_late10to210_best";
          if (a[2] != "NA" && a[2] >= 81.0) status = "target_81_reached";
          printf "%s\tcheckpoint-%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ts, ckpt, a[1], a[2], a[3], a[4], delta_base, delta_scheme, delta_original60, delta_original110, delta_dynamic110, delta_kl100, delta_late210, status;
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
args_yaml="missing"
controller_artifact="absent"
if [[ -n "${REMOTE_HOST}" ]] && ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "test -d '${REMOTE_OUT}'" >/dev/null 2>&1; then
  output_exists="remote:${REMOTE_OUT}"
  ckpt_count="$(ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "find '${REMOTE_OUT}' -maxdepth 1 -type f -name 'checkpoint-*.pth.tar' | wc -l" | tr -d ' ')"
  latest_ckpt="$(ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "find '${REMOTE_OUT}' -maxdepth 1 -type f -name 'checkpoint-*.pth.tar' -printf '%f\n' | sort -V | tail -n 1")"
  [[ -n "${latest_ckpt}" ]] || latest_ckpt="NA"
  if ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "test -f '${REMOTE_OUT}/args.yaml'" >/dev/null 2>&1; then
    args_yaml="present"
  fi
  if ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "find '${REMOTE_OUT}' -maxdepth 1 -type f -name '*controller*.tsv' | grep -q ." >/dev/null 2>&1; then
    controller_artifact="present"
  fi
fi

best_line="NA"
fullval_rows=0
bad_sample_rows=0
last20_avg="NA"
last10_avg="NA"
above_baseline=0
above_scheme_c=0
above_original60=0
above_original110=0
above_dynamic110=0
above_kl100=0
above_late210=0
target_81=0
if [[ -s "${STATUS}" ]]; then
  fullval_rows="$(awk -F '\t' 'NR > 1 { c++ } END { print c+0 }' "${STATUS}")"
  bad_sample_rows="$(awk -F '\t' 'NR > 1 && $6 != "50000" { c++ } END { print c+0 }' "${STATUS}")"
  best_line="$(awk -F '\t' 'NR > 1 && $4 != "NA" { if (!seen || $4+0 > best+0) { best=$4; line=$0; seen=1 } } END { if (seen) print line; else print "NA" }' "${STATUS}")"
  above_baseline="$(awk -F '\t' 'NR > 1 && $4+0 > 80.5980 { c++ } END { print c+0 }' "${STATUS}")"
  above_scheme_c="$(awk -F '\t' 'NR > 1 && $4+0 > 80.6820 { c++ } END { print c+0 }' "${STATUS}")"
  above_original60="$(awk -F '\t' 'NR > 1 && $4+0 > 80.7240 { c++ } END { print c+0 }' "${STATUS}")"
  above_original110="$(awk -F '\t' 'NR > 1 && $4+0 > 80.7520 { c++ } END { print c+0 }' "${STATUS}")"
  above_dynamic110="$(awk -F '\t' 'NR > 1 && $4+0 > 80.7600 { c++ } END { print c+0 }' "${STATUS}")"
  above_kl100="$(awk -F '\t' 'NR > 1 && $4+0 > 80.7720 { c++ } END { print c+0 }' "${STATUS}")"
  above_late210="$(awk -F '\t' 'NR > 1 && $4+0 > 80.8280 { c++ } END { print c+0 }' "${STATUS}")"
  target_81="$(awk -F '\t' 'NR > 1 && $4+0 >= 81.0 { c++ } END { print c+0 }' "${STATUS}")"
  last20_avg="$(awk -F '\t' 'NR > 1 && $4 != "NA" { vals[++n]=$4 } END { if (n==0) { print "NA"; exit } start=n-19; if (start<1) start=1; for (i=start; i<=n; i++) { sum+=vals[i]; c++ } printf "%.4f", sum/c }' "${STATUS}")"
  last10_avg="$(awk -F '\t' 'NR > 1 && $4 != "NA" { vals[++n]=$4 } END { if (n==0) { print "NA"; exit } start=n-9; if (start<1) start=1; for (i=start; i<=n; i++) { sum+=vals[i]; c++ } printf "%.4f", sum/c }' "${STATUS}")"
fi

nonzero_refw=0
nonzero_refw_epochs="NA"
if [[ -s "${REF_STATUS}" ]]; then
  nonzero_refw="$(awk -F '\t' 'NR > 1 && $3 != "0.000e+00" && $3+0 != 0 { c++ } END { print c+0 }' "${REF_STATUS}")"
  nonzero_refw_epochs="$(awk -F '\t' 'NR > 1 && $3 != "0.000e+00" && $3+0 != 0 { seen[$2]=1 } END { out=""; for (e in seen) out=(out==""?e:out "," e); print (out==""?"NA":out) }' "${REF_STATUS}")"
fi

{
  printf 'timestamp=%s\n' "${ts}"
  printf 'log=%s\n' "${LOG}"
  printf 'remote_output=%s:%s\n' "${REMOTE_HOST}" "${REMOTE_OUT}"
  printf 'log_exists=%s\n' "$([[ -f "${LOG}" ]] && echo yes || echo no)"
  printf 'output_exists=%s\n' "${output_exists}"
  printf 'checkpoint_count=%s\n' "${ckpt_count}"
  printf 'latest_checkpoint=%s\n' "${latest_ckpt}"
  printf 'args_yaml=%s\n' "${args_yaml}"
  printf 'controller_artifact=%s\n' "${controller_artifact}"
  printf 'status_tsv=%s\n' "${STATUS}"
  printf 'refw_tsv=%s\n' "${REF_STATUS}"
  printf 'fullval_rows=%s\n' "${fullval_rows}"
  printf 'bad_sample_rows=%s\n' "${bad_sample_rows}"
  printf 'best_fullval_line=%s\n' "${best_line}"
  printf 'above_baseline_lines=%s\n' "${above_baseline}"
  printf 'above_scheme_c_lines=%s\n' "${above_scheme_c}"
  printf 'above_original10to60_lines=%s\n' "${above_original60}"
  printf 'above_original10to110_lines=%s\n' "${above_original110}"
  printf 'above_dynamic10to110_lines=%s\n' "${above_dynamic110}"
  printf 'above_kl100ep_lines=%s\n' "${above_kl100}"
  printf 'above_late10to210_lines=%s\n' "${above_late210}"
  printf 'target_81_lines=%s\n' "${target_81}"
  printf 'last20_avg=%s\n' "${last20_avg}"
  printf 'last10_avg=%s\n' "${last10_avg}"
  printf 'nonzero_refw_lines=%s\n' "${nonzero_refw}"
  printf 'nonzero_refw_epochs=%s\n' "${nonzero_refw_epochs}"
} > "${SUMMARY}"

printf '%s\n' "${SUMMARY}"
