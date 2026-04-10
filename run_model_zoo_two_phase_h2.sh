#!/bin/bash
#===================================================================
# run_model_zoo_two_phase_h2.sh — SGE Job Array: Two-Phase Model Zoo
#
# Phase 1 (Tasks 1-84):  SHORT horizons on UNFILTERED data
#   → cls_1m, cls_3m, cls_5m, cls_10m  (D_b features auto-dropped)
#
# Phase 2 (Tasks 85-147): LONG horizons on FILTERED data
#   → cls_close, cls_clop, cls_clcl
#
# Usage:
#   qsub run_model_zoo_two_phase_h2.sh
#===================================================================

#$ -cwd
#$ -j y
#$ -o logs/zoo_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=4:00:00
#$ -pe shared 2
#$ -t 1-147

# ── 1. Initialize cluster profile (enables `module` command) ─
. /etc/profile

# ── 2. Load Python environment ──────────────────────────────
. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source /u/scratch/n/nicjia/order-burst-analysis/.venv/bin/activate

set -Eeo pipefail

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

# Build binary only when running prep mode.
echo "Compiler: $(g++ --version | head -n 1)"

# Tickers to evaluate for cross-stock stability.
TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}

# Input filename suffixes for short/long phases.
# Model-zoo should benchmark models on a single baseline extraction.
# Default behavior:
#   short phase -> bursts_${TICKER}_baseline_unfiltered.csv   (kappa=0 permanence-enriched)
#   long phase  -> bursts_${TICKER}_baseline_filtered.csv     (kappa=BASE_KAPPA_LONG)
#
# If you really need legacy names, set:
#   UNFILTERED_SUFFIX=unfiltered FILTERED_SUFFIX=filtered
UNFILTERED_SUFFIX=${UNFILTERED_SUFFIX:-baseline_unfiltered}
FILTERED_SUFFIX=${FILTERED_SUFFIX:-baseline_filtered}

# Default burst extraction params for model-selection baseline.
# These should stay fixed while choosing the best model family.
BASE_SILENCE=${BASE_SILENCE:-0.5}
BASE_VOL_FRAC=${BASE_VOL_FRAC:-0.0001}
BASE_DIR_THRESH=${BASE_DIR_THRESH:-0.8}
BASE_VOL_RATIO=${BASE_VOL_RATIO:-0.3}
BASE_TAU_MAX=${BASE_TAU_MAX:-10.0}
BASE_KAPPA_LONG=${BASE_KAPPA_LONG:-0.5}

# Set FORCE_REBUILD_BASELINE=1 to delete and rebuild baseline CSVs.
FORCE_REBUILD_BASELINE=${FORCE_REBUILD_BASELINE:-0}

# PREP_ONLY=1 runs a sequential precompute stage and exits.
# This avoids all array-task races during data preparation.
PREP_ONLY=${PREP_ONLY:-0}
PREP_WORKERS=${PREP_WORKERS:-1}

prepare_one_ticker() {
    local ticker="$1"
    local raw_csv="results/bursts_${ticker}_baseline.csv"
    local short_csv="results/bursts_${ticker}_baseline_unfiltered.csv"
    local filtered_csv="results/bursts_${ticker}_baseline_filtered.csv"
    local long_seed_csv="results/bursts_${ticker}_baseline_longseed.csv"
    local long_seed_out_csv="results/bursts_${ticker}_baseline_longseed_filtered.csv"
    local stock_dir="${ROOT}/data/${ticker}"

    # If source data is missing (e.g., scratch purge), skip this ticker.
    if [ ! -d "${stock_dir}" ]; then
        echo "WARN: Missing stock folder ${stock_dir}; skipping ${ticker}"
        return 1
    fi
    if ! ls "${stock_dir}"/*_message_*.csv >/dev/null 2>&1; then
        echo "WARN: No *_message_*.csv found in ${stock_dir}; skipping ${ticker}"
        return 1
    fi

    if [ "${FORCE_REBUILD_BASELINE}" = "1" ]; then
        echo "INFO: FORCE_REBUILD_BASELINE=1 -> removing baseline artifacts for ${ticker}"
        rm -f "${raw_csv}" "${short_csv}" "${filtered_csv}" \
              "${long_seed_csv}" "${long_seed_out_csv}"
    fi

    # If prior job crashed and left an empty/corrupt file, rebuild it.
    if [ -f "${raw_csv}" ] && [ ! -s "${raw_csv}" ]; then
        echo "WARN: Found empty baseline file ${raw_csv}; rebuilding"
        rm -f "${raw_csv}"
    fi
    if [ -f "${short_csv}" ] && [ ! -s "${short_csv}" ]; then
        echo "WARN: Found empty short baseline file ${short_csv}; rebuilding"
        rm -f "${short_csv}"
    fi
    if [ -f "${filtered_csv}" ] && [ ! -s "${filtered_csv}" ]; then
        echo "WARN: Found empty filtered baseline file ${filtered_csv}; rebuilding"
        rm -f "${filtered_csv}"
    fi

    if [ ! -f "${raw_csv}" ]; then
        echo "INFO: Building baseline bursts for ${ticker} -> ${raw_csv}"
        ./data_processor "${ROOT}/data/${ticker}" "${raw_csv}" \
            -s "${BASE_SILENCE}" \
            -v "${BASE_VOL_FRAC}" \
            -d "${BASE_DIR_THRESH}" \
            -r "${BASE_VOL_RATIO}" \
            -k 0 \
            -t "${BASE_TAU_MAX}" \
            -j "${PREP_WORKERS}" \
            -b 34200 -e 57600

        if [ ! -s "${raw_csv}" ]; then
            echo "ERROR: data_processor produced empty output: ${raw_csv}" >&2
            exit 1
        fi
    fi

    if [ ! -f "${short_csv}" ]; then
        echo "INFO: Building SHORT permanence dataset for ${ticker} (kappa=0) -> ${short_csv}"
        python3 src_py/compute_permanence.py \
            "${raw_csv}" \
            "${ROOT}/open_all.csv" \
            "${ROOT}/close_all.csv" \
            --kappa 0

        # compute_permanence always writes *_filtered.csv; with kappa=0 this is unfiltered permanence.
        if [ -f "${filtered_csv}" ] && [ ! -f "${short_csv}" ]; then
            mv "${filtered_csv}" "${short_csv}"
        fi

        if [ ! -s "${short_csv}" ]; then
            echo "ERROR: compute_permanence produced empty output: ${short_csv}" >&2
            exit 1
        fi
    fi

    if [ ! -f "${filtered_csv}" ]; then
        echo "INFO: Building LONG permanence dataset for ${ticker} (kappa=${BASE_KAPPA_LONG}) -> ${filtered_csv}"
        cp "${raw_csv}" "${long_seed_csv}"

        python3 src_py/compute_permanence.py \
            "${long_seed_csv}" \
            "${ROOT}/open_all.csv" \
            "${ROOT}/close_all.csv" \
            --kappa "${BASE_KAPPA_LONG}"

        if [ -f "${long_seed_out_csv}" ]; then
            mv "${long_seed_out_csv}" "${filtered_csv}"
        fi
        rm -f "${long_seed_csv}"

        if [ ! -s "${filtered_csv}" ]; then
            echo "ERROR: compute_permanence produced empty output: ${filtered_csv}" >&2
            exit 1
        fi
    fi

}

prepare_all_baselines() {
    mkdir -p logs results

    if [ ! -x "${ROOT}/data_processor" ]; then
        echo "INFO: data_processor missing; building with make"
        make clean && make
    fi

    echo "=========================================="
    echo "Baseline Preparation (Sequential)"
    echo "  Tickers:      ${TICKERS}"
    echo "  Params:       s=${BASE_SILENCE} v_frac=${BASE_VOL_FRAC} d=${BASE_DIR_THRESH} r=${BASE_VOL_RATIO}"
    echo "  Kappa long:   ${BASE_KAPPA_LONG}"
    echo "  Workers:      ${PREP_WORKERS}"
    echo "  Force rebuild:${FORCE_REBUILD_BASELINE}"
    echo "=========================================="

    for TICKER in ${TICKERS}; do
        prepare_one_ticker "${TICKER}" || true
    done

    echo "Baseline preparation finished at $(date)"
}

if [ "${PREP_ONLY}" = "1" ]; then
    # If submitted as an array, only task 1 performs prep; others exit.
    if [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "1" ]; then
        echo "PREP_ONLY=1 and SGE_TASK_ID=${SGE_TASK_ID}; exiting non-task-1 shard"
        exit 0
    fi
    prepare_all_baselines
    exit 0
fi

# ── 3. Determine phase from task ID ─────────────────────────
PHASE1_JOBS=84   # short horizons (unfiltered): cls models × 4 targets

if [ ${SGE_TASK_ID} -le ${PHASE1_JOBS} ]; then
    # Phase 1: Short-horizon prediction on unfiltered bursts
    INDEX=$((SGE_TASK_ID - 1))
    TARGET="short"
    PHASE_TAG="unfiltered"
else
    # Phase 2: Long-horizon prediction on filtered bursts
    INDEX=$((SGE_TASK_ID - PHASE1_JOBS - 1))
    TARGET="long"
    PHASE_TAG="filtered"
fi

echo "=========================================="
echo "SGE Job Array: Model Zoo (Two-Phase)"
echo "  Job ID:       ${JOB_ID}"
echo "  Task ID:      ${SGE_TASK_ID}  (0-based index: ${INDEX})"
echo "  Phase:        ${TARGET}"
echo "  Input suffix (short/long): ${UNFILTERED_SUFFIX} / ${FILTERED_SUFFIX}"
echo "  Baseline params: s=${BASE_SILENCE} v_frac=${BASE_VOL_FRAC} d=${BASE_DIR_THRESH} r=${BASE_VOL_RATIO} k_long=${BASE_KAPPA_LONG}"
echo "  Force rebuild baseline: ${FORCE_REBUILD_BASELINE}"
echo "  Tickers:      ${TICKERS}"
echo "  Hostname:     $(hostname)"
echo "  Date:         $(date)"
echo "=========================================="

mkdir -p logs

echo "INFO: Training mode (no baseline generation in array tasks)."
echo "INFO: Expecting precomputed files:"
echo "      short -> results/bursts_<TICKER>_${UNFILTERED_SUFFIX}.csv"
echo "      long  -> results/bursts_<TICKER>_${FILTERED_SUFFIX}.csv"

for TICKER in ${TICKERS}; do
    if [ "${PHASE_TAG}" = "unfiltered" ]; then
        BURSTS_CSV="results/bursts_${TICKER}_${UNFILTERED_SUFFIX}.csv"
        LEGACY_CSV="results/bursts_${TICKER}_unfiltered.csv"
    else
        BURSTS_CSV="results/bursts_${TICKER}_${FILTERED_SUFFIX}.csv"
        LEGACY_CSV="results/bursts_${TICKER}_filtered.csv"
    fi
    OUTDIR="results/zoo_bursts_${TICKER}_${PHASE_TAG}/"

    if [ ! -f "${BURSTS_CSV}" ]; then
        if [ -f "${LEGACY_CSV}" ]; then
            echo "INFO: Missing ${BURSTS_CSV}; falling back to ${LEGACY_CSV}"
            BURSTS_CSV="${LEGACY_CSV}"
        else
            echo "WARNING: Missing input ${BURSTS_CSV} and fallback ${LEGACY_CSV}; skipping ${TICKER}"
            continue
        fi
    fi

    echo "Running ${TICKER} | ${TARGET} | ${BURSTS_CSV}"
    mkdir -p "${OUTDIR}"

    python3 src_py/train_model_zoo.py "${BURSTS_CSV}" \
        --model all \
        --target "${TARGET}" \
        --features extended \
        --outdir "${OUTDIR}" \
        --slurm-index ${INDEX}
done

echo "Task ${SGE_TASK_ID} (index ${INDEX}, phase=${TARGET}) finished at $(date)"