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

echo "Submitting Phase-III CLCL flow backtests for tickers: ${TICKERS}"
QSUB_OUT=$(qsub -t 1-"${N}" \
  -v TICKERS="${TICKERS}",TARGETS="reg_clcl",EXECUTION_MODE=phase3_flow,SIGNAL_MODE=direction,POSITION_MODE=shares,SHARES_PER_TRADE=1,SPREAD_COL=NO_SPREAD_COL,DAILY_CLOSE_CSV=close_all.csv,DAILY_OPEN_CSV=open_all.csv,PHASE3_THRESH=0.0,PHASE3_MIN_LAG_MINUTES=10.0,PHASE3_FLOW_COL=signed_volume \
  run_overnight_backtests_h2.sh)

echo "${QSUB_OUT}"
echo "Mode: phase3_flow (CLCL), theta=0, lag=10m, signed-volume aggregation, shares=1, no spread costs"
echo "Use: qstat -u nicjia"
