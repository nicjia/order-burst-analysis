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

echo "Submitting burst-level 1m direction backtests for tickers: ${TICKERS}"
QSUB_OUT=$(qsub -t 1-"${N}" \
  -v TICKERS="${TICKERS}",TARGETS="reg_1m",EXECUTION_MODE=burst_stream,SIGNAL_MODE=direction,POSITION_MODE=shares,SHARES_PER_TRADE=1,SPREAD_COL=NO_SPREAD_COL \
  run_overnight_backtests_h2.sh)

echo "${QSUB_OUT}"
echo "Mode: burst_stream (reg_1m), direction signals, shares=1, no spread costs"
echo "Use: qstat -u nicjia"
