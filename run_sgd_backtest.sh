#!/bin/bash
#$ -cwd
#$ -N SGD_backtest
#$ -l h_data=8G,h_rt=6:00:00
#$ -o logs/backtest_$JOB_ID.out
#$ -e logs/backtest_$JOB_ID.err

# run_sgd_backtest.sh

# Stop on errors
set -e

source /etc/profile.d/modules.sh 2>/dev/null || . /u/local/Modules/default/init/modules.sh

# ==========================================================
# FIX: Load a newer GCC compiler BEFORE loading Python.
# Hoffman2 usually has gcc/10.2.0 or gcc/9.3.0 available.
# ==========================================================
module load gcc/10.2.0
module load python/3.9.6

source /u/scratch/n/nicjia/order-burst-analysis/.venv/bin/activate

python3 src_py/online_sgd_backtest.py --data results/nvda_raw_bursts_s2_filtered.csv --target reg_10m --silence-tag s2p0 --vol-frac 0.0027 --dir-thresh 0.68