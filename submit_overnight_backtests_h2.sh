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

echo "Submitting overnight backtests for tickers: ${TICKERS}"
QSUB_OUT=$(qsub -t 1-"${N}" -v TICKERS="${TICKERS}" run_overnight_backtests_h2.sh)
echo "${QSUB_OUT}"

echo "Use: qstat -u nicjia"