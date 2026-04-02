#!/bin/bash
#===================================================================
# hoffman2_model_zoo.sh — SGE Job Array: Two-Phase Model Zoo
#
# Phase 1 (Tasks 1-84):  SHORT horizons on UNFILTERED data
#   → cls_1m, cls_3m, cls_5m, cls_10m  (D_b features auto-dropped)
#
# Phase 2 (Tasks 85-147): LONG horizons on FILTERED data
#   → cls_close, cls_clop, cls_clcl
#
# Usage:
#   qsub hoffman2_model_zoo.sh
#===================================================================

#$ -cwd
#$ -j y
#$ -o logs/zoo_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=4:00:00
#$ -pe shared 4
#$ -t 1-147

# ── 1. Initialize cluster profile (enables `module` command) ─
. /etc/profile

# ── 2. Load Python environment ──────────────────────────────
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source /u/scratch/n/nicjia/order-burst-analysis/.venv/bin/activate

set -Eeo pipefail

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

# Tickers to evaluate for cross-stock stability.
TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}

# Input filename suffixes for short/long phases.
# Model-zoo should benchmark models on a single baseline extraction.
# Default behavior:
#   short phase -> bursts_${TICKER}_baseline.csv
#   long phase  -> bursts_${TICKER}_baseline_filtered.csv
#
# If you really need legacy names, set:
#   UNFILTERED_SUFFIX=unfiltered FILTERED_SUFFIX=filtered
UNFILTERED_SUFFIX=${UNFILTERED_SUFFIX:-baseline}
FILTERED_SUFFIX=${FILTERED_SUFFIX:-baseline_filtered}

# Default burst extraction params for model-selection baseline.
# These should stay fixed while choosing the best model family.
BASE_SILENCE=${BASE_SILENCE:-0.5}
BASE_MIN_VOL=${BASE_MIN_VOL:-100}
BASE_DIR_THRESH=${BASE_DIR_THRESH:-0.8}
BASE_VOL_RATIO=${BASE_VOL_RATIO:-0.3}
BASE_TAU_MAX=${BASE_TAU_MAX:-10.0}
BASE_KAPPA_LONG=${BASE_KAPPA_LONG:-0.5}

# Set FORCE_REBUILD_BASELINE=1 to delete and rebuild baseline CSVs.
FORCE_REBUILD_BASELINE=${FORCE_REBUILD_BASELINE:-0}

ensure_baseline_inputs() {
    local ticker="$1"
    local raw_csv="results/bursts_${ticker}_baseline.csv"
    local filtered_csv="results/bursts_${ticker}_baseline_filtered.csv"

    if [ "${FORCE_REBUILD_BASELINE}" = "1" ]; then
        echo "INFO: FORCE_REBUILD_BASELINE=1 -> removing ${raw_csv} and ${filtered_csv}"
        rm -f "${raw_csv}" "${filtered_csv}"
    fi

    if [ ! -f "${raw_csv}" ]; then
        echo "INFO: Building baseline bursts for ${ticker} -> ${raw_csv}"
        ./data_processor "${ROOT}/data/${ticker}" "${raw_csv}" \
            -s "${BASE_SILENCE}" \
            -v "${BASE_MIN_VOL}" \
            -d "${BASE_DIR_THRESH}" \
            -r "${BASE_VOL_RATIO}" \
            -k 0 \
            -t "${BASE_TAU_MAX}" \
            -j "${NSLOTS:-4}" \
            -b 34200 -e 57600
    fi

    if [ ! -f "${filtered_csv}" ]; then
        echo "INFO: Building baseline permanence for ${ticker} (kappa=${BASE_KAPPA_LONG}) -> ${filtered_csv}"
        python3 src_py/compute_permanence.py \
            "${raw_csv}" \
            "${ROOT}/open_all.csv" \
            "${ROOT}/close_all.csv" \
            --kappa "${BASE_KAPPA_LONG}"
    fi
}

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
echo "  Baseline params: s=${BASE_SILENCE} v=${BASE_MIN_VOL} d=${BASE_DIR_THRESH} r=${BASE_VOL_RATIO} k_long=${BASE_KAPPA_LONG}"
echo "  Force rebuild baseline: ${FORCE_REBUILD_BASELINE}"
echo "  Tickers:      ${TICKERS}"
echo "  Hostname:     $(hostname)"
echo "  Date:         $(date)"
echo "=========================================="

mkdir -p logs

for TICKER in ${TICKERS}; do
    ensure_baseline_inputs "${TICKER}"

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