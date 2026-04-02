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
#$ -pe shared 2
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

# Build binary on demand inside the proper toolchain environment.
if [ ! -x "${ROOT}/data_processor" ]; then
    echo "INFO: data_processor missing; building with make"
    make clean && make
fi

echo "Compiler: $(g++ --version | head -n 1)"

# Tickers to evaluate for cross-stock stability.
TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}

# Input filename suffixes for short/long phases.
# Model-zoo should benchmark models on a single baseline extraction.
# Default behavior:
#   short phase -> bursts_${TICKER}_baseline_unfiltered.csv (Perm_* present, kappa=0)
#   long phase  -> bursts_${TICKER}_baseline_filtered.csv
#
# If you really need legacy names, set:
#   UNFILTERED_SUFFIX=unfiltered FILTERED_SUFFIX=filtered
UNFILTERED_SUFFIX=${UNFILTERED_SUFFIX:-baseline_unfiltered}
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
    local unfiltered_csv="results/bursts_${ticker}_baseline_unfiltered.csv"
    local filtered_csv="results/bursts_${ticker}_baseline_filtered.csv"
    local lock_dir="results/.baseline_lock_${ticker}"
    local wait_s=0
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

    # One writer per ticker to avoid SGE array races creating partial/empty CSVs.
    while ! mkdir "${lock_dir}" 2>/dev/null; do
        sleep 2
        wait_s=$((wait_s + 2))
        if [ $((wait_s % 30)) -eq 0 ]; then
            echo "INFO: waiting for baseline lock ${lock_dir} (${wait_s}s)"
        fi
    done
    trap 'rmdir "${lock_dir}" 2>/dev/null || true' RETURN

    if [ "${FORCE_REBUILD_BASELINE}" = "1" ]; then
        echo "INFO: FORCE_REBUILD_BASELINE=1 -> removing ${raw_csv}, ${unfiltered_csv}, ${filtered_csv}"
        rm -f "${raw_csv}" "${unfiltered_csv}" "${filtered_csv}"
    fi

    # If prior job crashed and left an empty/corrupt file, rebuild it.
    if [ -f "${raw_csv}" ] && [ ! -s "${raw_csv}" ]; then
        echo "WARN: Found empty baseline file ${raw_csv}; rebuilding"
        rm -f "${raw_csv}"
    fi
    if [ -f "${unfiltered_csv}" ] && [ ! -s "${unfiltered_csv}" ]; then
        echo "WARN: Found empty unfiltered baseline file ${unfiltered_csv}; rebuilding"
        rm -f "${unfiltered_csv}"
    fi
    if [ -f "${filtered_csv}" ] && [ ! -s "${filtered_csv}" ]; then
        echo "WARN: Found empty filtered baseline file ${filtered_csv}; rebuilding"
        rm -f "${filtered_csv}"
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

        if [ ! -s "${raw_csv}" ]; then
            echo "ERROR: data_processor produced empty output: ${raw_csv}" >&2
            exit 1
        fi
    fi

    # Short-phase input must include Perm_* columns with NO kappa filtering.
    if [ ! -f "${unfiltered_csv}" ]; then
        local tmp_raw="results/bursts_${ticker}_baseline_k0tmp.csv"
        local tmp_out="results/bursts_${ticker}_baseline_k0tmp_filtered.csv"

        cp "${raw_csv}" "${tmp_raw}"
        echo "INFO: Building baseline permanence for ${ticker} (kappa=0) -> ${unfiltered_csv}"
        python3 src_py/compute_permanence.py \
            "${tmp_raw}" \
            "${ROOT}/open_all.csv" \
            "${ROOT}/close_all.csv" \
            --kappa 0
        mv -f "${tmp_out}" "${unfiltered_csv}"
        rm -f "${tmp_raw}"

        if [ ! -s "${unfiltered_csv}" ]; then
            echo "ERROR: kappa=0 permanence output empty: ${unfiltered_csv}" >&2
            exit 1
        fi
    fi

    if [ ! -f "${filtered_csv}" ]; then
        echo "INFO: Building baseline permanence for ${ticker} (kappa=${BASE_KAPPA_LONG}) -> ${filtered_csv}"
        python3 src_py/compute_permanence.py \
            "${raw_csv}" \
            "${ROOT}/open_all.csv" \
            "${ROOT}/close_all.csv" \
            --kappa "${BASE_KAPPA_LONG}"

        if [ ! -s "${filtered_csv}" ]; then
            echo "ERROR: compute_permanence produced empty output: ${filtered_csv}" >&2
            exit 1
        fi
    fi

    # Release lock for waiting array tasks.
    rmdir "${lock_dir}" 2>/dev/null || true
    trap - RETURN
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
    if ! ensure_baseline_inputs "${TICKER}"; then
        continue
    fi

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