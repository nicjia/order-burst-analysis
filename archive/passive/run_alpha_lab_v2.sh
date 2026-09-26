#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -o logs/alpha_lab_v2_$JOB_ID.out
#$ -l h_data=16G,h_rt=03:00:00
#$ -pe shared 1

# ─────────────────────────────────────────────
# Alpha Lab v2: Look-Ahead-Safe Strategy Tests
# Runs on a compute node (16GB) for large tickers
# ─────────────────────────────────────────────
ROOT=${ROOT:-/u/scratch/n/nicjia/order-burst-analysis}
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
set -Eeo pipefail

echo "Starting Alpha Lab v2 at $(date)"
echo "Host: $(hostname)"
echo "Memory: $(free -h | head -2)"

python3 passive/src_py/passive_alpha_lab_v2.py

echo "Completed Alpha Lab v2 at $(date)"
