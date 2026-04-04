#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/sweep_frac_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=24:00:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source /u/scratch/n/nicjia/order-burst-analysis/.venv/bin/activate

set -Eeo pipefail

# ────────────────────────────────────────────────────────────────
# Fractional-ADV Volume Sweep
#
# Instead of flat volume thresholds (v=50, 100, ...), this sweep
# uses fractions of the trailing 14-day Average Daily Volume.
# This ensures burst significance scales automatically with each
# stock's liquidity regime.
# ────────────────────────────────────────────────────────────────

TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
MODELS="logreg_l2"

# Reduced target set for fast sweep.
SHORT_TARGETS=${SHORT_TARGETS:-"cls_1m,cls_10m"}
LONG_TARGETS=${LONG_TARGETS:-"cls_close"}

SILENCE_VALUES="0.5,1.0,2.0"

# Fractional volume grid: fraction of trailing 14-day ADV.
# For NVDA (~50M daily): 0.00001 ≈ 5 shares, 0.0001 ≈ 50, 0.001 ≈ 500
# For MS   (~15M daily): 0.00001 ≈ 1.5 shares, 0.0001 ≈ 15, 0.001 ≈ 150
VOL_FRAC_VALUES="0.00001,0.0001,0.001"

DIR_THRESH_VALUES="0.7,0.8,0.9"
VOL_RATIO_VALUES="0.1,0.3,0.5"

KAPPA_SHORT="0.0"
KAPPA_LONG="0.0"
MIN_ROWS=100
REQUIRE_DIRECTIONAL=0

echo "========== FRACTIONAL ADV VOLUME SWEEP =========="
echo "Tickers:     ${TICKERS}"
echo "Models:      ${MODELS}"
echo "Short tgts:  ${SHORT_TARGETS}"
echo "Long tgts:   ${LONG_TARGETS}"
echo "Vol fracs:   ${VOL_FRAC_VALUES}"
echo "Min rows:    ${MIN_ROWS}"
echo "================================================="

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
        outdir="${ROOT}/results/silence_sweep_frac_${ticker}/${model}/short"
    else
        target_arg="${LONG_TARGETS}"
        kappa_arg="${KAPPA_LONG}"
        outdir="${ROOT}/results/silence_sweep_frac_${ticker}/${model}/long"
    fi

    echo "Running ticker=${ticker} model=${model} phase=${phase} (fractional ADV)"
    python3 src_py/silence_optimized_sweep.py \
        --stock-folder "${ROOT}/data/${ticker}" \
        --ticker "${ticker}" \
        --open "${ROOT}/open_all.csv" \
        --close "${ROOT}/close_all.csv" \
        --data-processor "${ROOT}/data_processor" \
        --outdir "${outdir}" \
        --precompute-dir "${cache_dir}" \
        --silence-values "${SILENCE_VALUES}" \
        --vol-frac-values "${VOL_FRAC_VALUES}" \
        --adv-window 14 \
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
        run_one_phase "${TICKER}" "${M}" "short"
        run_one_phase "${TICKER}" "${M}" "long"
    done

    echo "Fractional sweep task complete for ${TICKER}."
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

echo "Fractional sweep complete."
