#!/bin/bash
set -e

if [ -f /etc/profile ]; then
    . /etc/profile
fi
if [ -f /u/local/Modules/default/init/bash ]; then
    . /u/local/Modules/default/init/bash
    module load gcc/11.3.0 python/3.9.6
fi

ROOT="/u/scratch/n/nicjia/order-burst-analysis"
cd $ROOT
source .venv/bin/activate
export PYTHONNOUSERSITE=1

echo "=============================="
echo "Phase 1: DATA (C++ Extraction)"
echo "=============================="
rm -f results/bursts_NVDA_baseline*.csv results/NVDA_debug_*.csv results/NVDA_backtest.log
make clean && make
time ./data_processor data/NVDA results/bursts_NVDA_baseline.csv \
    -H 1.0 -I 0.5 -w 0.050 -v 0.0001 -d 0.8 -r 0.3 -k 0 -t 10.0 -j 16 -b 34200 -e 57600

echo "=============================="
echo "Phase 2: PERM (Permanence)"
echo "=============================="
time python3 src_py/compute_permanence.py results/bursts_NVDA_baseline.csv open_all.csv close_all.csv --kappa 0 --ticker NVDA
mv results/bursts_NVDA_baseline_filtered.csv results/bursts_NVDA_baseline_unfiltered.csv

cp results/bursts_NVDA_baseline.csv results/bursts_NVDA_baseline_longseed.csv
time python3 src_py/compute_permanence.py results/bursts_NVDA_baseline_longseed.csv open_all.csv close_all.csv --kappa 0.5 --ticker NVDA
mv results/bursts_NVDA_baseline_longseed_filtered.csv results/bursts_NVDA_baseline_filtered.csv
rm results/bursts_NVDA_baseline_longseed.csv

echo "=============================="
echo "Phase 4: BACKTEST (SGD Walk-forward)"
echo "=============================="
time python3 src_py/online_sgd_backtest.py \
    --data results/bursts_NVDA_baseline_unfiltered.csv \
    --target reg_clop \
    --hawkes-tag b1p0_i0p5 \
    --vol-frac 0.0001 \
    --dir-thresh 0.8 \
    --vol-ratio 0.3 \
    --kappa 0.5 \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --ticker NVDA \
    --execution-mode phase3_flow \
    --signal-mode direction \
    --position-mode fixed_aum \
    --fixed-aum 10000000 \
    --round-trip-bps-cost 1.0 \
    --daily-open-csv open_all.csv \
    --daily-close-csv close_all.csv \
    --debug-trades-out results/NVDA_debug_trades.csv \
    --debug-signals-out results/NVDA_debug_signals.csv | tee results/NVDA_backtest.log

echo "=============================="
echo "Pipeline finished!"
echo "=============================="
