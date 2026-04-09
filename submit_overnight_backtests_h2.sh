#!/bin/bash
set -Eeo pipefail

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

TICKERS=${1:-"NVDA TSLA JPM MS"}
read -r -a ARR <<< "${TICKERS}"
N=${#ARR[@]}
if [ "${N}" -eq 0 ]; then
  echo "ERROR: no tickers provided" >&2
  exit 1
fi

mkdir -p logs

# Remove stale fault logs from prior failed overnight runs so fresh submits are clean.
FAULT_PATTERN='Input y contains NaN|^ERROR: line '
mapfile -t STALE_FAULT_LOGS < <(grep -El "${FAULT_PATTERN}" logs/overnight_bt_*.out 2>/dev/null || true)
if [ "${#STALE_FAULT_LOGS[@]}" -gt 0 ]; then
  echo "Removing ${#STALE_FAULT_LOGS[@]} stale fault log(s) from prior runs..."
  rm -f "${STALE_FAULT_LOGS[@]}"
fi

echo "Submitting overnight backtests for tickers: ${TICKERS}"
QSUB_OUT=$(qsub -t 1-"${N}" -v TICKERS="${TICKERS}" run_overnight_backtests_h2.sh)
echo "${QSUB_OUT}"

echo "Use: qstat -u nicjia"