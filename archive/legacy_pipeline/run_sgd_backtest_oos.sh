#!/bin/bash

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"

TARGET_TO_RUN="reg_clop"
HAWKES_TAG="b1p0_i0p3"
START_DATE="2019-01-01"
END_DATE="2021-12-31"
TICKERS="AAPL LLY SPY"

VOL_FRAC=0.002077762
DIR_THRESH=0.634886
VOL_RATIO=0.44239
KAPPA=1.039886

mkdir -p results/sgd_backtests_fixed_aum_oos

for ticker in $TICKERS; do
  data_path="results/bursts_${ticker}_baseline_unfiltered.csv"
  out_prefix="results/sgd_backtests_fixed_aum_oos/${ticker}_${TARGET_TO_RUN}_regression_oos"
  
  echo "================================================"
  echo "OOS Ticker: ${ticker}"
  echo "================================================"
  
  python3 src_py/online_sgd_backtest.py \
    --data "${data_path}" \
    --target "${TARGET_TO_RUN}" \
    --hawkes-tag "${HAWKES_TAG}" \
    --vol-frac "${VOL_FRAC}" \
    --dir-thresh "${DIR_THRESH}" \
    --vol-ratio "${VOL_RATIO}" \
    --kappa "${KAPPA}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --ticker "${ticker}" \
    --execution-mode phase3_flow \
    --signal-mode direction \
    --position-mode fixed_aum \
    --fixed-aum 10000000 \
    --round-trip-bps-cost 1.0 \
    --daily-open-csv open_all.csv \
    --daily-close-csv close_all.csv \
    --debug-trades-out "${out_prefix}_debug_trades.csv" \
    --debug-signals-out "${out_prefix}_debug_signals.csv" \
    | tee "${out_prefix}.log"
done
