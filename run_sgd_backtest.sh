#!/bin/bash
#$ -cwd
#$ -N SGD_backtest
#$ -l h_data=8G,h_rt=24:00:00,highp
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

echo "-> [1/4] Injecting targets into s2p0 dataset (cls_10m)..."
# Overrides C++ "archive" Ticker string with strict NVDA
python3 src_py/compute_permanence.py results/nvda_raw_bursts_s2.csv open_all.csv close_all.csv --kappa 0 --ticker NVDA
echo "Done mapping targets format for s2p0."

echo "-> [2/4] Generating s0p5 raw dataset natively using data_processor for cls_close..."
./data_processor data/NVDA/archive_2019_2022 results/nvda_raw_bursts_s0p5.csv -s 0.5 -v 5 -d 0.5 -k 0 -t 10.0 -j 8
echo "Done extracting raw s0p5 bursts."

echo "-> [3/4] Injecting targets into s0p5 dataset (cls_close)..."
python3 src_py/compute_permanence.py results/nvda_raw_bursts_s0p5.csv open_all.csv close_all.csv --kappa 0 --ticker NVDA
echo "Done mapping targets format for s0p5."

echo "-> [4/4] Firing Parallel SGD Walk-Forward Backtesters..."
python3 src_py/online_sgd_backtest.py --data results/nvda_raw_bursts_s2_filtered.csv --target reg_10m --silence-tag s2p0 --vol-frac 0.0027 --dir-thresh 0.68 > logs/pnl_sim_10m.out &
PID1=$!

python3 src_py/online_sgd_backtest.py --data results/nvda_raw_bursts_s0p5_filtered.csv --target reg_close --silence-tag s0p5 --vol-frac 0.0019 --dir-thresh 0.64 > logs/pnl_sim_close.out &
PID2=$!

echo "Simulations are executing efficiently in RAM. Waiting for completions..."
wait $PID1
wait $PID2

echo "\n=========================================================="
echo "  ALL BACKTESTS SECURELY COMPLETED!"
echo "  Inspect logs/pnl_sim_10m.out and logs/pnl_sim_close.out"
echo "=========================================================="
