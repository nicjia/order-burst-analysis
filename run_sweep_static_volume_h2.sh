#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/sweep_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=24:00:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source /u/scratch/n/nicjia/order-burst-analysis/.venv/bin/activate

set -Eeo pipefail

# NOTE:
# Keep comma-delimited grids defined inside this script.
# Passing comma lists via qsub -v can be split incorrectly by SGE.
# If you need to change these, edit this file directly.
TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
MODELS="logreg_l2"

# Target split avoids mixing short+long in one call.
# Reduced target set for fast sweep — expand to all targets on winning configs.
SHORT_TARGETS=${SHORT_TARGETS:-"cls_1m,cls_10m"}
LONG_TARGETS=${LONG_TARGETS:-"cls_close"}

SILENCE_VALUES="0.5,1.0,2.0"
MIN_VOL_VALUES="50,100,200,1000"
DIR_THRESH_VALUES="0.7,0.8,0.9"
VOL_RATIO_VALUES="0.1,0.3,0.5"

# Kappa = 0 only for sweep phase (expand on winning configs later).
KAPPA_SHORT="0.0"
KAPPA_LONG="0.0"
MIN_ROWS=100
REQUIRE_DIRECTIONAL=0

echo "========== PARAMETER SWEEP =========="
echo "Tickers: ${TICKERS}"
echo "Models: ${MODELS}"
echo "Short targets: ${SHORT_TARGETS}"
echo "Long targets: ${LONG_TARGETS}"
echo "Min rows: ${MIN_ROWS}"
echo "Require directional: ${REQUIRE_DIRECTIONAL}"
echo "====================================="

IFS=',' read -ra MODEL_LIST <<< "${MODELS}"
EXTRA_FLAGS=()
if [ "${REQUIRE_DIRECTIONAL}" = "1" ]; then
    EXTRA_FLAGS+=(--require-directional)
fi

run_one_phase() {
    local ticker="$1"
    local model="$2"
    local phase="$3"

    local target_arg
    local kappa_arg
    local outdir
    local cache_dir="${ROOT}/results/silence_sweep_${ticker}/${model}/shared_cache"

    if [ "${phase}" = "short" ]; then
        target_arg="${SHORT_TARGETS}"
        kappa_arg="${KAPPA_SHORT}"
        outdir="${ROOT}/results/silence_sweep_${ticker}/${model}/short"
    else
        target_arg="${LONG_TARGETS}"
        kappa_arg="${KAPPA_LONG}"
        outdir="${ROOT}/results/silence_sweep_${ticker}/${model}/long"
    fi

    echo "Running ticker=${ticker} model=${model} phase=${phase}"
    python3 src_py/silence_optimized_sweep.py \
        --stock-folder "${ROOT}/data/${ticker}" \
        --ticker "${ticker}" \
        --open "${ROOT}/open_all.csv" \
        --close "${ROOT}/close_all.csv" \
        --data-processor "${ROOT}/data_processor" \
        --outdir "${outdir}" \
        --precompute-dir "${cache_dir}" \
        --silence-values "${SILENCE_VALUES}" \
        --min-vol-values "${MIN_VOL_VALUES}" \
        --dir-thresh-values "${DIR_THRESH_VALUES}" \
        --vol-ratio-values "${VOL_RATIO_VALUES}" \
        --kappa-values "${kappa_arg}" \
        --model "${model}" \
        --target "${target_arg}" \
        --workers "${NSLOTS:-8}" \
        --min-rows "${MIN_ROWS}" \
        --skip-existing \
        "${EXTRA_FLAGS[@]}"
}

if [ -n "${SGE_TASK_ID:-}" ]; then
    read -r -a TICKER_ARR <<< "${TICKERS}"
    N_TICKERS=${#TICKER_ARR[@]}
    IDX=$((SGE_TASK_ID - 1))

    if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${N_TICKERS}" ]; then
        echo "INFO: Task ${SGE_TASK_ID} out of range; exiting."
        exit 0
    fi

    TICKER=${TICKER_ARR[$IDX]}
    echo "Assigning node to ticker=${TICKER}"

    for MODEL in "${MODEL_LIST[@]}"; do
        M=$(echo "${MODEL}" | xargs)
        [ -z "${M}" ] && continue
        # Run phases sequentially on the same node so long reuses short cache.
        run_one_phase "${TICKER}" "${M}" "short"
        run_one_phase "${TICKER}" "${M}" "long"
    done

    echo "Sweep task complete for ${TICKER}."
    exit 0
fi

for TICKER in ${TICKERS}; do
    for MODEL in "${MODEL_LIST[@]}"; do
        M=$(echo "${MODEL}" | xargs)
        [ -z "${M}" ] && continue
        run_one_phase "${TICKER}" "${M}" "short"
        run_one_phase "${TICKER}" "${M}" "long"
    done
done

echo "Sweep complete."