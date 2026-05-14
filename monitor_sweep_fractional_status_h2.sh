#!/usr/bin/env bash
set -euo pipefail

# Run this from /u/scratch/n/nicjia/order-burst-analysis on Hoffman.
cd "$(dirname "$0")"

# Check for both legacy sweep_frac logs and new hawkes_sweep logs
if ls logs/sweep_frac_*.out >/dev/null 2>&1; then
  LOG_PREFIX="sweep_frac"
elif ls logs/hawkes_sweep_*.out >/dev/null 2>&1; then
  LOG_PREFIX="hawkes_sweep"
else
  echo "No sweep out logs found under logs/."
  exit 2
fi

latest_file=$(ls -1t logs/${LOG_PREFIX}_*.out | head -n 1)
latest_base=$(basename "$latest_file")
latest_job_id=$(echo "$latest_base" | awk -F'_' '{print $3}')

log_glob="logs/${LOG_PREFIX}_${latest_job_id}_*.out"
log_count=$(ls -1 $log_glob 2>/dev/null | wc -l | awk '{print $1}')

echo "Latest ${LOG_PREFIX} job id: $latest_job_id"
echo "Log files matched: $log_count"

running_jobs=$(qstat -u "${USER}" 2>/dev/null | grep -Ei 'sweep_frac|hawkes_sweep|silence_optimized_sweep' || true)
if [[ -n "$running_jobs" ]]; then
  echo ""
  echo "Job appears to still be in queue/running:"
  echo "$running_jobs"
else
  echo "No sweep job currently in qstat."
fi

fail_count=0
if grep -Eqi 'Traceback|ModuleNotFoundError|CalledProcessError|ImportError' $log_glob; then
  echo ""
  echo "Errors detected in latest sweep logs:"
  grep -Ein 'Traceback|ModuleNotFoundError|CalledProcessError|ImportError' $log_glob || true
  fail_count=1
fi

declare -a tickers=(JPM MS NVDA TSLA)
missing=0
for t in "${tickers[@]}"; do
  if grep -q "Fractional.*sweep task complete for ${t}." $log_glob; then
    echo "${t}: complete"
  else
    echo "${t}: NOT marked complete in latest logs"
    missing=1
  fi
done

if [[ $fail_count -eq 0 && $missing -eq 0 && -z "$running_jobs" ]]; then
  echo ""
  echo "STATUS: FINISHED OK"
  exit 0
fi

if [[ -n "$running_jobs" ]]; then
  echo ""
  echo "STATUS: STILL RUNNING"
  exit 3
fi

echo ""
echo "STATUS: FINISHED WITH ERRORS OR MISSING TICKERS"
exit 1
