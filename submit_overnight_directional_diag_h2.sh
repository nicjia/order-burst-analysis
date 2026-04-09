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

echo "Submitting direction-only diagnostic overnight backtests for tickers: ${TICKERS}"
QSUB_OUT=$(qsub -t 1-"${N}" \
  -v TICKERS="${TICKERS}",SIGNAL_MODE=direction,POSITION_MODE=shares,SHARES_PER_TRADE=1,SPREAD_COL=NO_SPREAD_COL,COST_BUFFER_MULT=0 \
  run_overnight_backtests_h2.sh)

echo "${QSUB_OUT}"
echo "Mode: signal=direction, position=shares(1), spread-cost disabled via SPREAD_COL=NO_SPREAD_COL"
echo "Use: qstat -u nicjia"
