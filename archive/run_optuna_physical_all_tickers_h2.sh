#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/optuna_physical_$JOB_ID_$TASK_ID.out
#$ -l h_data=16G,h_rt=06:59:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/11.3.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
set -Eeo pipefail

TICKERS=(NVDA TSLA JPM MS)
TARGETS=("cls_close" "cls_clop" "cls_clcl")
TRIALS=100
START_DATE=${START_DATE:-2023-01-01}
END_DATE=${END_DATE:-2024-12-31}

echo "========== OPTUNA PHYSICAL PARAMETER SEARCH =========="
echo "Targets: ${TARGETS[*]}"
echo "Trials: ${TRIALS}"
echo "Date window: ${START_DATE} -> ${END_DATE}"
echo "Cache dir: results/<TICKER>_params/shared_cache/"
echo "======================================================"

if [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ]; then
    IDX=$((SGE_TASK_ID - 1))
    if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${#TICKERS[@]}" ]; then
        echo "INFO: Task ${SGE_TASK_ID} out of range; exiting."
        exit 0
    fi
    TICKER=${TICKERS[$IDX]}
else
    # Fallback: default to first ticker when run outside SGE array
    TICKER=${TICKERS[0]}
    echo "INFO: SGE_TASK_ID not set or undefined; defaulting to ${TICKER}"
fi

echo "Running ticker=${TICKER}"

for TARGET in "${TARGETS[@]}"; do
    echo "[Target: ${TARGET}]"
    python3 src_py/optuna_physical_sweep.py \
        --ticker "${TICKER}" \
        --target "${TARGET}" \
        --trials "${TRIALS}" \
        --start-date "${START_DATE}" \
        --end-date "${END_DATE}"
done

echo "Completed: ${TICKER} at $(date)"
exit 0
