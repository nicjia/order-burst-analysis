#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/optuna_physical_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=04:00:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source /u/scratch/n/nicjia/order-burst-analysis/.venv/bin/activate

set -Eeo pipefail

TICKERS=(NVDA TSLA JPM MS)
TARGETS=("cls_1m" "cls_3m" "cls_5m" "cls_10m" "cls_close" "cls_clop" "cls_clcl")
TRIALS=100

echo "========== OPTUNA PHYSICAL PARAMETER SEARCH =========="
echo "Targets: ${TARGETS[*]}"
echo "Trials: ${TRIALS}"
echo "======================================================"

if [ -n "${SGE_TASK_ID:-}" ]; then
    IDX=$((SGE_TASK_ID - 1))
    if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${#TICKERS[@]}" ]; then
        echo "INFO: Task ${SGE_TASK_ID} out of range; exiting."
        exit 0
    fi
    TICKER=${TICKERS[$IDX]}
else
    # Fallback to local loop for testing
    TICKER=${TICKERS[0]}
fi

echo "Running ticker=${TICKER}"

for TARGET in "${TARGETS[@]}"; do
    echo "[Target: ${TARGET}]"
    python3 src_py/optuna_physical_sweep.py \
        --ticker "${TICKER}" \
        --target "${TARGET}" \
        --trials "${TRIALS}"
done

echo "Completed: ${TICKER} at $(date)"
exit 0
