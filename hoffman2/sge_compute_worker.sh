#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# sge_compute_worker.sh — SGE Job Array Worker (runs on highp compute nodes)
# ─────────────────────────────────────────────────────────────────────────
#
# Each task ($SGE_TASK_ID) processes ONE ticker from the batch manifest.
#
# Workflow:
#   1. Read ticker name from current_batch.txt
#   2. Find all staged .7z files for this ticker
#   3. Extract message CSVs into a temp folder (structured for the parser)
#   4. Run the C++ data_processor (single-threaded, kappa=0)
#   5. Run compute_permanence.py to add overnight targets
#   6. rm -rf all extracted CSVs immediately
#   7. Exit with status code
#
# SGE directives (set by master_orchestrator.sh via qsub flags):
#   -t 1-BATCH_SIZE
#   -l highp
#   -l h_data=8G,h_rt=04:00:00
#   -pe shared 1
#
# ─────────────────────────────────────────────────────────────────────────
set -Eeo pipefail

# ── Load environment ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the batch environment variables
if [ -f "${SCRIPT_DIR}/batch_env.sh" ]; then
    source "${SCRIPT_DIR}/batch_env.sh"
fi

# Fallback defaults
SCRATCH="${SCRATCH:-/u/scratch/n/nicjia}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/order-burst-analysis}"
STAGING_DIR="${STAGING_DIR:-${SCRATCH}/lobster_staging}"
YEARS="${YEARS:-2022 2023 2024 2025 2026}"

# Load modules
if [ -f /u/local/Modules/default/init/bash ]; then
    . /u/local/Modules/default/init/bash
    module load gcc/11.3.0 2>/dev/null || true
    module load python/3.9.6 2>/dev/null || true
fi

# Activate venv
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi
export PATH="${HOME}/bin:${PATH}"
export PYTHONNOUSERSITE=1

# ── Resolve ticker from batch manifest ───────────────────────────────────
BATCH_MANIFEST="${PROJECT_DIR}/hoffman2/current_batch.txt"
if [ ! -f "${BATCH_MANIFEST}" ]; then
    echo "ERROR: Batch manifest not found: ${BATCH_MANIFEST}"
    exit 1
fi

TICKER=$(sed -n "${SGE_TASK_ID}p" "${BATCH_MANIFEST}")
if [ -z "${TICKER}" ]; then
    echo "ERROR: No ticker found at line ${SGE_TASK_ID} in ${BATCH_MANIFEST}"
    exit 1
fi

# ── Logging setup ────────────────────────────────────────────────────────
WORKER_START=$(date +%s)
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  SGE WORKER — Task ${SGE_TASK_ID} — Ticker: ${TICKER}"
echo "║  Job ID: ${JOB_ID:-local}  |  Host: $(hostname)"
echo "║  Started: $(date)"
echo "╚════════════════════════════════════════════════════════════╝"

# ── Output paths ─────────────────────────────────────────────────────────
OUTPUT_DIR="${PROJECT_DIR}/results"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_CSV="${OUTPUT_DIR}/bursts_${TICKER}_baseline.csv"

# Skip if already computed
if [ -s "${OUTPUT_CSV}" ]; then
    echo "[${TICKER}] Output already exists: ${OUTPUT_CSV} ($(wc -l < "${OUTPUT_CSV}") rows)"
    echo "[${TICKER}] Skipping. Delete the file to reprocess."
    exit 0
fi

# ── Create temporary work directory ──────────────────────────────────────
# Structure: $WORK_DIR/$TICKER/ contains all message CSVs for the parser
WORK_DIR=$(mktemp -d "${SCRATCH}/tmp_burst_${TICKER}_XXXXXX")
EXTRACT_DIR="${WORK_DIR}/${TICKER}"
mkdir -p "${EXTRACT_DIR}"

echo "[${TICKER}] Work directory: ${WORK_DIR}"

# ── Cleanup trap (ensures temp files are removed on ANY exit) ────────────
cleanup() {
    echo "[${TICKER}] Cleaning up ${WORK_DIR}..."
    rm -rf "${WORK_DIR}"
    echo "[${TICKER}] Cleanup complete."
}
trap cleanup EXIT

# ── Step 1: Find and extract all .7z files for this ticker ───────────────
echo "[${TICKER}] Scanning staging area for .7z files..."
ARCHIVE_COUNT=0
EXTRACT_ERRORS=0

# Pick an extractor ONCE. Native 7z/7za are preferred; py7zr (pure-Python,
# installed into the venv by setup_hoffman2.sh) is the fallback when no
# native binary exists on the compute nodes.
# NOTE: this runs in the MAIN shell (not a `find | while` subshell) so that a
# missing-extractor `exit 1` actually terminates the worker instead of being
# swallowed and silently producing an empty "no data" output.
EXTRACTOR=""
if command -v 7z &>/dev/null; then
    EXTRACTOR="7z"
elif command -v 7za &>/dev/null; then
    EXTRACTOR="7za"
elif python3 -c "import py7zr" 2>/dev/null; then
    EXTRACTOR="py7zr"
else
    echo "[${TICKER}] ERROR: no 7z/7za binary and no py7zr module available."
    echo "[${TICKER}]        Run hoffman2/setup_hoffman2.sh first."
    exit 1
fi
echo "[${TICKER}] Extractor: ${EXTRACTOR}"

extract_one() {
    # $1 = archive path, $2 = destination dir
    case "${EXTRACTOR}" in
        7z)    7z  x "$1" -o"$2" -y >/dev/null 2>&1 ;;
        7za)   7za x "$1" -o"$2" -y >/dev/null 2>&1 ;;
        py7zr) python3 -m py7zr x "$1" "$2/" >/dev/null 2>&1 ;;
    esac
}

# Extract every archive for this ticker (process substitution → main shell).
while IFS= read -r ARCHIVE_PATH; do
    if ! extract_one "${ARCHIVE_PATH}" "${EXTRACT_DIR}"; then
        echo "[${TICKER}] WARNING: Failed to extract ${ARCHIVE_PATH}"
        EXTRACT_ERRORS=$((EXTRACT_ERRORS + 1))
    fi
    ARCHIVE_COUNT=$((ARCHIVE_COUNT + 1))
done < <(find "${STAGING_DIR}" -name "${TICKER}.7z" 2>/dev/null | sort)

echo "[${TICKER}] Processed ${ARCHIVE_COUNT} archive(s), ${EXTRACT_ERRORS} extraction error(s)"

# Count extracted message files
MSG_FILE_COUNT=$(find "${EXTRACT_DIR}" -name "*message*" -name "*.csv" 2>/dev/null | wc -l)
echo "[${TICKER}] Extracted ${MSG_FILE_COUNT} message files from archives"

if [ "${MSG_FILE_COUNT}" -eq 0 ]; then
    echo "[${TICKER}] ERROR: No message files found after extraction!"
    echo "[${TICKER}] This ticker may not exist in LOBSTER for years ${YEARS}"
    # Create an empty output to mark it as attempted
    echo "# No data found for ${TICKER} in years ${YEARS}" > "${OUTPUT_CSV}"
    exit 0
fi

# ── Step 2: Run C++ data_processor ───────────────────────────────────────
echo ""
echo "[${TICKER}] ── Running C++ burst detector ──"
echo "[${TICKER}] Input:  ${EXTRACT_DIR} (${MSG_FILE_COUNT} day files)"
echo "[${TICKER}] Output: ${OUTPUT_CSV}"

# Parser parameters from batch_env.sh (with fallback defaults)
PARSE_START=$(date +%s)
"${PROJECT_DIR}/data_processor" \
    "${EXTRACT_DIR}" \
    "${OUTPUT_CSV}" \
    -H "${HAWKES_BETA:-1.0}" \
    -I "${TRIGGER_INTENSITY:-0.5}" \
    -w "${CANCEL_WINDOW:-0.050}" \
    -v "${VOL_FRAC:-0.0001}" \
    -d "${DIR_THRESH:-0.8}" \
    -r "${VOL_RATIO:-0.3}" \
    -k 0 \
    -t "${TAU_MAX:-10.0}" \
    -j 1 \
    -b 34200 \
    -e 57600

PARSE_EXIT=$?
PARSE_ELAPSED=$(( $(date +%s) - PARSE_START ))

if [ ${PARSE_EXIT} -ne 0 ]; then
    echo "[${TICKER}] ERROR: data_processor exited with code ${PARSE_EXIT}"
    exit ${PARSE_EXIT}
fi

if [ ! -s "${OUTPUT_CSV}" ]; then
    echo "[${TICKER}] ERROR: Output CSV is empty after data_processor"
    exit 1
fi

BURST_COUNT=$(wc -l < "${OUTPUT_CSV}")
echo "[${TICKER}] C++ parser completed in ${PARSE_ELAPSED}s — ${BURST_COUNT} rows (including header)"

# ── Step 3: IMMEDIATELY delete all extracted CSVs ────────────────────────
echo "[${TICKER}] Deleting extracted CSVs from ${EXTRACT_DIR}..."
rm -rf "${EXTRACT_DIR}"
echo "[${TICKER}] ✓ Extracted data deleted"

# ── Step 4: Compute permanence (overnight targets) ───────────────────────
# Check if CRSP price matrices exist
OPEN_CSV="${PROJECT_DIR}/open_all.csv"
CLOSE_CSV="${PROJECT_DIR}/close_all.csv"

if [ -f "${OPEN_CSV}" ] && [ -f "${CLOSE_CSV}" ]; then
    echo ""
    echo "[${TICKER}] ── Computing permanence (CLOP/CLCL targets) ──"

    PERM_START=$(date +%s)

    # Unfiltered (kappa=0) — this is the primary output
    python3 "${PROJECT_DIR}/src_py/compute_permanence.py" \
        "${OUTPUT_CSV}" \
        "${OPEN_CSV}" \
        "${CLOSE_CSV}" \
        --kappa 0 \
        --ticker "${TICKER}"

    # The compute_permanence.py script outputs:
    #   bursts_TICKER_baseline_filtered.csv (despite kappa=0, this is the naming convention)
    # Rename to the expected unfiltered name
    PERM_OUTPUT="${OUTPUT_CSV%.csv}_filtered.csv"
    UNFILTERED_OUTPUT="${OUTPUT_DIR}/bursts_${TICKER}_baseline_unfiltered.csv"
    if [ -f "${PERM_OUTPUT}" ]; then
        mv -f "${PERM_OUTPUT}" "${UNFILTERED_OUTPUT}"
        echo "[${TICKER}] Permanence output: ${UNFILTERED_OUTPUT}"
    fi

    PERM_ELAPSED=$(( $(date +%s) - PERM_START ))
    echo "[${TICKER}] Permanence completed in ${PERM_ELAPSED}s"
else
    echo "[${TICKER}] WARNING: CRSP price matrices not found. Skipping permanence."
    echo "[${TICKER}]   Expected: ${OPEN_CSV} and ${CLOSE_CSV}"
    echo "[${TICKER}]   Run 'python3 src_py/pivot_returns.py yearly/' to generate."
fi

# ── Final summary ────────────────────────────────────────────────────────
WORKER_ELAPSED=$(( $(date +%s) - WORKER_START ))
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  WORKER COMPLETE — ${TICKER}"
echo "║  Total time:    ${WORKER_ELAPSED}s"
echo "║  Burst CSV:     ${OUTPUT_CSV}"
echo "║  Rows:          ${BURST_COUNT}"
echo "╚════════════════════════════════════════════════════════════╝"
exit 0
