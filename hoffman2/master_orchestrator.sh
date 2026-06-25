#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# master_orchestrator.sh — HPC Pipeline Controller (runs on DTN in tmux)
# ─────────────────────────────────────────────────────────────────────────
#
# Iterates through the 500-ticker universe in batches of BATCH_SIZE.
# For each batch:
#   1. rsync the .7z files from lobster2 to $SCRATCH/lobster_staging/
#   2. Submit an SGE job array (qsub -sync y) and wait for completion
#   3. Verify outputs, clean up staging, and advance to next batch
#
# Supports resumption: skips tickers whose output CSV already exists.
#
# Usage:
#   # In a tmux session on the DTN:
#   bash hoffman2/master_orchestrator.sh
#
#   # Override defaults:
#   bash hoffman2/master_orchestrator.sh \
#       --universe universes/full_500.txt \
#       --batch-size 20 \
#       --years "2022 2023 2024 2025 2026"
#
# ─────────────────────────────────────────────────────────────────────────
set -Eeo pipefail
trap 'echo "[ORCHESTRATOR] ERROR at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# ── Configuration (env-overridable for testing/portability) ──────────────
SCRATCH="${SCRATCH:-/u/scratch/n/nicjia}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/order-burst-analysis}"
STAGING_DIR="${STAGING_DIR:-${SCRATCH}/lobster_staging}"
LOBSTER_HOST="${LOBSTER_HOST:-nicjia@lobster2.math.ucla.edu}"
LOBSTER_ROOT="/lobster"

# Defaults
UNIVERSE_FILE="${PROJECT_DIR}/universes/full_500.txt"
BATCH_SIZE=20
YEARS="2022 2023 2024 2025 2026"
SGE_WALLTIME="04:00:00"
SGE_MEMORY="8G"
DRY_RUN=0

# Hawkes / filter params passed to the C++ parser
HAWKES_BETA="${HAWKES_BETA:-1.0}"
TRIGGER_INTENSITY="${TRIGGER_INTENSITY:-0.5}"
CANCEL_WINDOW="${CANCEL_WINDOW:-0.050}"
VOL_FRAC="${VOL_FRAC:-0.0001}"
DIR_THRESH="${DIR_THRESH:-0.8}"
VOL_RATIO="${VOL_RATIO:-0.3}"
TAU_MAX="${TAU_MAX:-10.0}"

# ── Parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --universe)     UNIVERSE_FILE="$2";  shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";     shift 2 ;;
        --years)        YEARS="$2";          shift 2 ;;
        --walltime)     SGE_WALLTIME="$2";   shift 2 ;;
        --memory)       SGE_MEMORY="$2";     shift 2 ;;
        --dry-run)      DRY_RUN=1;           shift   ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Load environment ─────────────────────────────────────────────────────
if [ -f /u/local/Modules/default/init/bash ]; then
    . /u/local/Modules/default/init/bash
    module load gcc/11.3.0 2>/dev/null || true
    module load python/3.9.6 2>/dev/null || true
fi
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi
export PATH="${HOME}/bin:${PATH}"
export PYTHONNOUSERSITE=1

# ── Validate ─────────────────────────────────────────────────────────────
if [ ! -f "${UNIVERSE_FILE}" ]; then
    echo "ERROR: Universe file not found: ${UNIVERSE_FILE}"
    echo "  Create it with one ticker per line, or use --universe <path>"
    exit 1
fi

if [ ! -x "${PROJECT_DIR}/data_processor" ]; then
    echo "ERROR: data_processor binary not found. Run setup_hoffman2.sh first."
    exit 1
fi

# ── Read universe (skip comments and blanks) ─────────────────────────────
readarray -t ALL_TICKERS < <(grep -v '^\s*#' "${UNIVERSE_FILE}" | grep -v '^\s*$' | awk '{print $1}')
TOTAL_TICKERS=${#ALL_TICKERS[@]}

if [ "${TOTAL_TICKERS}" -eq 0 ]; then
    echo "ERROR: No tickers found in ${UNIVERSE_FILE}"
    exit 1
fi

# ── Filter out already-completed tickers ─────────────────────────────────
PENDING_TICKERS=()
SKIPPED=0
for TICKER in "${ALL_TICKERS[@]}"; do
    OUTPUT_CSV="${PROJECT_DIR}/results/bursts_${TICKER}_baseline.csv"
    if [ -s "${OUTPUT_CSV}" ]; then
        SKIPPED=$((SKIPPED + 1))
    else
        PENDING_TICKERS+=("${TICKER}")
    fi
done

PENDING_COUNT=${#PENDING_TICKERS[@]}
TOTAL_BATCHES=$(( (PENDING_COUNT + BATCH_SIZE - 1) / BATCH_SIZE ))

# ── Banner ───────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  MASTER ORCHESTRATOR — 500-Ticker HPC Burst Detection Pipeline    ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Universe:       ${UNIVERSE_FILE}"
echo "║  Total tickers:  ${TOTAL_TICKERS}"
echo "║  Already done:   ${SKIPPED}"
echo "║  Pending:        ${PENDING_COUNT}"
echo "║  Batch size:     ${BATCH_SIZE}"
echo "║  Total batches:  ${TOTAL_BATCHES}"
echo "║  Years:          ${YEARS}"
echo "║  SGE walltime:   ${SGE_WALLTIME}"
echo "║  SGE memory:     ${SGE_MEMORY}"
echo "║  Lobster host:   ${LOBSTER_HOST}"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

if [ "${PENDING_COUNT}" -eq 0 ]; then
    echo "All tickers already processed. Nothing to do."
    exit 0
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY RUN] Would process ${PENDING_COUNT} tickers in ${TOTAL_BATCHES} batches."
    echo "[DRY RUN] First batch: ${PENDING_TICKERS[@]:0:${BATCH_SIZE}}"
    exit 0
fi

# ── Create log directory ─────────────────────────────────────────────────
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"
}

log "Pipeline started. Log: ${MASTER_LOG}"

# ── Main batch loop ──────────────────────────────────────────────────────
BATCH_NUM=0
GLOBAL_SUCCESS=0
GLOBAL_FAIL=0

for ((START=0; START<PENDING_COUNT; START+=BATCH_SIZE)); do
    BATCH_NUM=$((BATCH_NUM + 1))
    END=$((START + BATCH_SIZE))
    if [ ${END} -gt ${PENDING_COUNT} ]; then
        END=${PENDING_COUNT}
    fi
    BATCH_ACTUAL_SIZE=$((END - START))

    # Extract batch tickers
    BATCH_TICKERS=("${PENDING_TICKERS[@]:${START}:${BATCH_ACTUAL_SIZE}}")

    log ""
    log "════════════════════════════════════════════════════════════════════"
    log "  BATCH ${BATCH_NUM}/${TOTAL_BATCHES} — ${BATCH_ACTUAL_SIZE} tickers"
    log "  Tickers: ${BATCH_TICKERS[*]}"
    log "════════════════════════════════════════════════════════════════════"

    # ── Step 1: Write batch manifest ─────────────────────────────────────
    BATCH_MANIFEST="${PROJECT_DIR}/hoffman2/current_batch.txt"
    printf '%s\n' "${BATCH_TICKERS[@]}" > "${BATCH_MANIFEST}"
    log "  Wrote batch manifest: ${BATCH_MANIFEST}"

    # ── Step 2: rsync .7z files from lobster2 ────────────────────────────
    log "  Syncing .7z files from lobster2..."
    BATCH_START_TIME=$(date +%s)

    # Build the find pattern for all tickers in this batch
    FIND_ARGS=""
    for TICKER in "${BATCH_TICKERS[@]}"; do
        if [ -n "${FIND_ARGS}" ]; then
            FIND_ARGS="${FIND_ARGS} -o"
        fi
        FIND_ARGS="${FIND_ARGS} -name ${TICKER}.7z"
    done

    # Build find paths for specified years
    FIND_PATHS=""
    for YEAR in ${YEARS}; do
        FIND_PATHS="${FIND_PATHS} ${LOBSTER_ROOT}/${YEAR}"
    done

    # Generate file list on lobster2 and rsync
    FILE_LIST="${PROJECT_DIR}/hoffman2/batch_file_list.txt"
    ssh "${LOBSTER_HOST}" "find ${FIND_PATHS} \( ${FIND_ARGS} \) 2>/dev/null" \
        > "${FILE_LIST}" 2>/dev/null || true

    FILE_COUNT=$(wc -l < "${FILE_LIST}" | xargs)
    log "  Found ${FILE_COUNT} .7z files on lobster2 for this batch"

    if [ "${FILE_COUNT}" -eq 0 ]; then
        log "  WARNING: No .7z files found for batch ${BATCH_NUM}. Skipping."
        continue
    fi

    # Ensure staging directory exists
    mkdir -p "${STAGING_DIR}"

    # rsync using the file list (preserves directory structure).
    # Don't let a partial-transfer non-zero abort the run; the STAGED_COUNT
    # check below decides whether the batch can proceed.
    set +e
    rsync -av --progress \
        --files-from="${FILE_LIST}" \
        "${LOBSTER_HOST}:/" \
        "${STAGING_DIR}/" \
        2>&1 | tail -5 | while read -r line; do log "  rsync: ${line}"; done
    RSYNC_EXIT=${PIPESTATUS[0]}
    set -e
    if [ "${RSYNC_EXIT}" -ne 0 ]; then
        log "  NOTE: rsync exited ${RSYNC_EXIT}; checking staged files anyway."
    fi

    RSYNC_ELAPSED=$(( $(date +%s) - BATCH_START_TIME ))
    log "  rsync completed in ${RSYNC_ELAPSED}s"

    # Verify staging
    STAGED_COUNT=$(find "${STAGING_DIR}" -name "*.7z" 2>/dev/null | wc -l)
    log "  Staged ${STAGED_COUNT} .7z files to ${STAGING_DIR}"

    # ── Step 3: Submit SGE job array ─────────────────────────────────────
    log "  Submitting SGE job array (${BATCH_ACTUAL_SIZE} tasks)..."

    # Write the SGE parameters to an env file the worker can source
    BATCH_ENV="${PROJECT_DIR}/hoffman2/batch_env.sh"
    cat > "${BATCH_ENV}" << ENVEOF
# Auto-generated by master_orchestrator.sh for batch ${BATCH_NUM}
export STAGING_DIR="${STAGING_DIR}"
export PROJECT_DIR="${PROJECT_DIR}"
export HAWKES_BETA="${HAWKES_BETA}"
export TRIGGER_INTENSITY="${TRIGGER_INTENSITY}"
export CANCEL_WINDOW="${CANCEL_WINDOW}"
export VOL_FRAC="${VOL_FRAC}"
export DIR_THRESH="${DIR_THRESH}"
export VOL_RATIO="${VOL_RATIO}"
export TAU_MAX="${TAU_MAX}"
export YEARS="${YEARS}"
ENVEOF

    # Submit and wait (-sync y blocks until all tasks complete).
    # IMPORTANT: qsub -sync y exits non-zero if ANY array task failed. We must
    # NOT let that abort the whole run (set -e / pipefail) — the per-ticker
    # verification in Step 4 is authoritative and lets healthy tickers proceed.
    # Capture qsub's true exit via PIPESTATUS (not the trailing while-loop's).
    SUBMIT_START=$(date +%s)
    set +e
    qsub -sync y \
        -S /bin/bash \
        -t "1-${BATCH_ACTUAL_SIZE}" \
        -l "highp" \
        -l "h_data=${SGE_MEMORY},h_rt=${SGE_WALLTIME}" \
        -pe shared 1 \
        -cwd \
        -o "${LOG_DIR}/worker_\$JOB_ID.\$TASK_ID.out" \
        -e "${LOG_DIR}/worker_\$JOB_ID.\$TASK_ID.err" \
        -N "burst_b${BATCH_NUM}" \
        "${PROJECT_DIR}/hoffman2/sge_compute_worker.sh" \
        2>&1 | while read -r line; do log "  qsub: ${line}"; done
    QSUB_EXIT=${PIPESTATUS[0]}
    set -e

    SUBMIT_ELAPSED=$(( $(date +%s) - SUBMIT_START ))
    log "  SGE array finished in ${SUBMIT_ELAPSED}s (qsub -sync exit: ${QSUB_EXIT})"
    if [ "${QSUB_EXIT}" -ne 0 ]; then
        log "  NOTE: qsub -sync reported non-zero (≥1 task may have failed);"
        log "        per-ticker verification below is authoritative."
    fi

    # ── Step 4: Verify outputs ───────────────────────────────────────────
    BATCH_SUCCESS=0
    BATCH_FAIL=0
    for TICKER in "${BATCH_TICKERS[@]}"; do
        OUTPUT_CSV="${PROJECT_DIR}/results/bursts_${TICKER}_baseline.csv"
        if [ -s "${OUTPUT_CSV}" ]; then
            ROWS=$(wc -l < "${OUTPUT_CSV}")
            log "  ✓ ${TICKER}: ${ROWS} rows"
            BATCH_SUCCESS=$((BATCH_SUCCESS + 1))
            GLOBAL_SUCCESS=$((GLOBAL_SUCCESS + 1))
        else
            log "  ✗ ${TICKER}: OUTPUT MISSING OR EMPTY"
            BATCH_FAIL=$((BATCH_FAIL + 1))
            GLOBAL_FAIL=$((GLOBAL_FAIL + 1))
        fi
    done

    log "  Batch ${BATCH_NUM} results: ${BATCH_SUCCESS} success, ${BATCH_FAIL} failed"

    # ── Step 5: Clean up staging area ────────────────────────────────────
    log "  Cleaning up staging area..."
    # Only remove the .7z files for tickers in this batch
    for TICKER in "${BATCH_TICKERS[@]}"; do
        find "${STAGING_DIR}" -name "${TICKER}.7z" -delete 2>/dev/null || true
    done

    # If staging is now empty, remove the directory structure
    find "${STAGING_DIR}" -type d -empty -delete 2>/dev/null || true

    BATCH_TOTAL_TIME=$(( $(date +%s) - BATCH_START_TIME ))
    log "  Batch ${BATCH_NUM} total time: ${BATCH_TOTAL_TIME}s"
done

# ── Final Summary ────────────────────────────────────────────────────────
log ""
log "╔══════════════════════════════════════════════════════════════════════╗"
log "║  PIPELINE COMPLETE                                                ║"
log "╠══════════════════════════════════════════════════════════════════════╣"
log "║  Total batches:    ${TOTAL_BATCHES}"
log "║  Successful:       ${GLOBAL_SUCCESS}"
log "║  Failed:           ${GLOBAL_FAIL}"
log "║  Previously done:  ${SKIPPED}"
log "║  Results in:       ${PROJECT_DIR}/results/"
log "╚══════════════════════════════════════════════════════════════════════╝"

# ── Trigger Phase 2: Permanence computation ──────────────────────────────
if [ "${GLOBAL_FAIL}" -eq 0 ]; then
    log ""
    log "All tickers processed successfully."
    log "To compute permanence, run:"
    log "  bash hoffman2/master_orchestrator.sh --phase perm"
    log "Or run the full downstream pipeline:"
    log "  bash run_pipeline.sh --phase perm --cluster"
else
    log ""
    log "WARNING: ${GLOBAL_FAIL} tickers failed. Check logs in ${LOG_DIR}/"
    log "Re-run the orchestrator to retry failed tickers (successful ones will be skipped)."
fi
