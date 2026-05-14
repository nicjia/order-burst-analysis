#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/sweep_frac_$JOB_ID_$TASK_ID.out
#$ -l h_data=4G,h_rt=02:59:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"

set -Eeo pipefail

# ────────────────────────────────────────────────────────────────
# Fractional-ADV Volume Sweep (Hawkes-based)
#
# Replaces silence-threshold sweep with Hawkes trigger_intensity
# sweep. Beta is FIXED at 1.0 to avoid parameter explosion.
# ────────────────────────────────────────────────────────────────

TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
MODELS="logreg_l2"

# Reduced target set for fast sweep.
SHORT_TARGETS=${SHORT_TARGETS:-"cls_1m,cls_10m"}
LONG_TARGETS=${LONG_TARGETS:-"cls_close"}

# Hawkes parameters (β fixed, sweep trigger_intensity only)
HAWKES_BETA="1.0"
TRIGGER_INTENSITY_VALUES="0.3,0.5,0.8"
CANCEL_WINDOW="0.050"

# Fractional volume grid: fraction of trailing 14-day ADV.
VOL_FRAC_VALUES="0.00001,0.0001,0.001"

DIR_THRESH_VALUES="0.7,0.8,0.9"
VOL_RATIO_VALUES="0.1,0.3,0.5"

KAPPA_SHORT="0.0"
KAPPA_LONG="0.0"
MIN_ROWS=100
REQUIRE_DIRECTIONAL=0

echo "========== FRACTIONAL ADV VOLUME SWEEP (HAWKES) =========="
echo "Tickers:     ${TICKERS}"
echo "Models:      ${MODELS}"
echo "Short tgts:  ${SHORT_TARGETS}"
echo "Long tgts:   ${LONG_TARGETS}"
echo "Hawkes beta: ${HAWKES_BETA} (fixed)"
echo "Trigger vals:${TRIGGER_INTENSITY_VALUES}"
echo "Cancel win:  ${CANCEL_WINDOW}s"
echo "Vol fracs:   ${VOL_FRAC_VALUES}"
echo "Min rows:    ${MIN_ROWS}"
echo "========================================================="

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
    local cache_dir="${ROOT}/results/hawkes_sweep_${ticker}/${model}/shared_cache"

    if [ "${phase}" = "short" ]; then
        target_arg="${SHORT_TARGETS}"
        kappa_arg="${KAPPA_SHORT}"
        outdir="${ROOT}/results/hawkes_sweep_frac_${ticker}/${model}/short"
    else
        target_arg="${LONG_TARGETS}"
        kappa_arg="${KAPPA_LONG}"
        outdir="${ROOT}/results/hawkes_sweep_frac_${ticker}/${model}/long"
    fi

    echo "Running ticker=${ticker} model=${model} phase=${phase} (Hawkes + fractional ADV)"
    python3 src_py/silence_optimized_sweep.py \
        --stock-folder "${ROOT}/data/${ticker}" \
        --ticker "${ticker}" \
        --open "${ROOT}/open_all.csv" \
        --close "${ROOT}/close_all.csv" \
        --data-processor "${ROOT}/data_processor" \
        --outdir "${outdir}" \
        --precompute-dir "${cache_dir}" \
        --hawkes-beta "${HAWKES_BETA}" \
        --trigger-intensity-values "${TRIGGER_INTENSITY_VALUES}" \
        --cancel-window "${CANCEL_WINDOW}" \
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

    echo "Fractional Hawkes sweep task complete for ${TICKER}."
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

echo "Fractional Hawkes sweep complete."
