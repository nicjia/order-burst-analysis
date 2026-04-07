#!/bin/bash
#$ -cwd
#$ -N SGD_backtest
#$ -l h_data=8G,h_rt=2:00:00,highp
#$ -M nicjia@ucla.edu
#$ -m bea
#$ -o logs/backtest_$JOB_ID.out
#$ -e logs/backtest_$JOB_ID.err

# run_sgd_backtest.sh

# Stop on errors
set -e

echo "=========================================================="
echo "  VECTORIZED COMPUTE PERMANENCE & SGD BACKTEST PIPELINE"
echo "=========================================================="

echo "-> [1/2] Injecting explicitly mapped targets into s2p0 dataset..."
# Overrides C++ "archive" Ticker string with strict NVDA
python3 src_py/compute_permanence.py results/nvda_raw_bursts_s2.csv open_all.csv close_all.csv --kappa 0 --ticker NVDA
echo "Done mapping targets format for s2p0."

echo "-> [2/2] Firing SGD Walk-Forward Engine..."
python3 src_py/online_sgd_backtest.py --data results/nvda_raw_bursts_s2_filtered.csv --target reg_10m --silence-tag s2p0 --vol-frac 0.0027 --dir-thresh 0.68

echo "=========================================================="
echo "  BACKTEST SECURELY COMPLETED!"
echo "=========================================================="
