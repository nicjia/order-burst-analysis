#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/sweep_$JOB_ID.out
#$ -l h_data=32G,h_rt=12:00:00
#$ -pe shared 8

set -Eeo pipefail

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source /u/scratch/n/nicjia/order-burst-analysis/.venv/bin/activate

# NOTE:
# Keep comma-delimited grids defined inside this script.
# Passing comma lists via qsub -v can be split incorrectly by SGE.
# If you need to change these, edit this file directly.
TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
MODELS="logreg_l2"

# Target split avoids mixing short+long in one call.
SHORT_TARGETS=${SHORT_TARGETS:-"cls_1m,cls_3m,cls_5m,cls_10m"}
LONG_TARGETS=${LONG_TARGETS:-"cls_close,cls_clop,cls_clcl"}

SILENCE_VALUES="0.5,1.0,2.0"
MIN_VOL_VALUES="50,100,200"
DIR_THRESH_VALUES="0.7,0.8,0.9"
VOL_RATIO_VALUES="0.1,0.3,0.5"

# Short horizons: kappa must be 0.
KAPPA_SHORT="0.0"
# Long horizons: evaluate several kappa choices.
KAPPA_LONG="0.0,0.2,0.5"
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

for TICKER in ${TICKERS}; do
    for MODEL in "${MODEL_LIST[@]}"; do
        M=$(echo "${MODEL}" | xargs)
        [ -z "${M}" ] && continue

        # Phase 1: short horizons (unfiltered / kappa=0)
        python3 src_py/silence_optimized_sweep.py \
            --stock-folder "${ROOT}/data/${TICKER}" \
            --ticker "${TICKER}" \
            --open "${ROOT}/open_all.csv" \
            --close "${ROOT}/close_all.csv" \
            --data-processor "${ROOT}/data_processor" \
            --outdir "${ROOT}/results/silence_sweep_${TICKER}/${M}/short" \
            --silence-values "${SILENCE_VALUES}" \
            --min-vol-values "${MIN_VOL_VALUES}" \
            --dir-thresh-values "${DIR_THRESH_VALUES}" \
            --vol-ratio-values "${VOL_RATIO_VALUES}" \
            --kappa-values "${KAPPA_SHORT}" \
            --model "${M}" \
            --target "${SHORT_TARGETS}" \
            --workers "${NSLOTS:-8}" \
            --min-rows "${MIN_ROWS}" \
            "${EXTRA_FLAGS[@]}"

        # Phase 2: long horizons (default kappa grid)
        python3 src_py/silence_optimized_sweep.py \
            --stock-folder "${ROOT}/data/${TICKER}" \
            --ticker "${TICKER}" \
            --open "${ROOT}/open_all.csv" \
            --close "${ROOT}/close_all.csv" \
            --data-processor "${ROOT}/data_processor" \
            --outdir "${ROOT}/results/silence_sweep_${TICKER}/${M}/long" \
            --silence-values "${SILENCE_VALUES}" \
            --min-vol-values "${MIN_VOL_VALUES}" \
            --dir-thresh-values "${DIR_THRESH_VALUES}" \
            --vol-ratio-values "${VOL_RATIO_VALUES}" \
            --kappa-values "${KAPPA_LONG}" \
            --model "${M}" \
            --target "${LONG_TARGETS}" \
            --workers "${NSLOTS:-8}" \
            --min-rows "${MIN_ROWS}" \
            "${EXTRA_FLAGS[@]}"
    done
done

echo "Sweep complete."