#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -o logs/passive_optuna_$JOB_ID.out
#$ -l h_data=16G,h_rt=12:00:00

# ─────────────────────────────────────────────────────────────
# run_passive_optuna_h2.sh
# ─────────────────────────────────────────────────────────────

# Navigate to project root assuming this script is in passive/
cd "$(dirname "$0")/.."

# Initialize Hoffman2 environment
. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6

# Activate Python virtual environment
source .venv/bin/activate

# Ensure required directories exist
mkdir -p results/optuna_passive
mkdir -p logs

CORE_TICKERS="NVDA,TSLA,JPM,MS"
TARGETS=("reg_clop")

for target in "${TARGETS[@]}"; do
    echo "========================================================="
    echo "Running passive Optuna sweep across Core Universe ($CORE_TICKERS) -> $target"
    echo "Job ID: $JOB_ID"
    echo "========================================================="
    
    python3 passive/src_py/passive_optuna_sweep.py \
        --core-tickers "$CORE_TICKERS" \
        --target "$target" \
        --trials 50
        
    echo "---------------------------------------------------------"
    echo "Sweep for $target complete."
done