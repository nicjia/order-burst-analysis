#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -t 1-7
#$ -o logs/passive_detection_$JOB_ID.$TASK_ID.out
#$ -l h_data=4G,h_rt=03:59:00
#$ -pe shared 4

# ─────────────────────────────────────────────────────────────
# run_passive_detection_h2.sh — Passive Limit Order Burst Detection
# ─────────────────────────────────────────────────────────────
# Array job: Task 1=NVDA, 2=TSLA, 3=JPM, 4=MS, 5=LLY, 6=AAPL, 7=SPY
#
# Step 1: Compile passive_data_processor (if not present)
# Step 2: Run passive detector on LOBSTER message files
# Step 3: Compute Transformed Price Drift targets
# ─────────────────────────────────────────────────────────────

ROOT=${ROOT:-/u/scratch/n/nicjia/order-burst-analysis}
TICKERS_ARRAY=("NVDA" "TSLA" "JPM" "MS" "LLY" "AAPL" "SPY")

# ── Hawkes parameters ────────────────────────────────────────
HAWKES_BETA=${HAWKES_BETA:-1.0}
TRIGGER_INTENSITY=${TRIGGER_INTENSITY:-0.5}
MAX_BBO_LEVELS=${MAX_BBO_LEVELS:-3}

# ── Physical filters (permissive defaults for initial exploration) ──
VOL_FRAC=${VOL_FRAC:-0.00000001}
DIR_THRESH=${DIR_THRESH:-0.6}
VOL_RATIO=${VOL_RATIO:-0.5}
KAPPA=${KAPPA:-0}

RTH_START=${RTH_START:-34200}
RTH_END=${RTH_END:-57600}
WORKERS=${WORKERS:-${NSLOTS:-4}}

cd "${ROOT}"
mkdir -p logs
mkdir -p results/passive

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# ── Determine ticker for this array task ─────────────────────
if [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ]; then
    idx=$((SGE_TASK_ID - 1))
    if [ "${idx}" -lt 0 ] || [ "${idx}" -ge "${#TICKERS_ARRAY[@]}" ]; then
        echo "ERROR: Task ${SGE_TASK_ID} out of range" >&2
        exit 1
    fi
    TICKER="${TICKERS_ARRAY[$idx]}"
else
    TICKER="${TICKERS_ARRAY[0]}"
    echo "WARNING: Running outside qsub. Defaulting to ${TICKER}."
fi

echo "==============================================="
echo "PASSIVE BURST DETECTION: ${TICKER}"
echo "Root: ${ROOT}"
echo "Hawkes: beta=${HAWKES_BETA} trigger=${TRIGGER_INTENSITY}"
echo "BBO levels: ${MAX_BBO_LEVELS}"
echo "Filters: vol_frac=${VOL_FRAC} dir=${DIR_THRESH} vol_ratio=${VOL_RATIO} kappa=${KAPPA}"
echo "Workers: ${WORKERS}"
echo "==============================================="

# ── Step 1: Compile if needed ────────────────────────────────
if [ ! -x "${ROOT}/passive/passive_data_processor" ]; then
    echo "INFO: Compiling passive_data_processor..."
    cd "${ROOT}/passive"
    make clean && make
    cd "${ROOT}"
fi

# ── Step 2: Run passive detector ─────────────────────────────
STOCK_DIR="${ROOT}/data/${TICKER}"
RAW_CSV="results/passive/passive_bursts_${TICKER}_raw.csv"

if [ ! -d "${STOCK_DIR}" ]; then
    echo "ERROR: Missing ${STOCK_DIR}" >&2
    exit 1
fi

echo "[${TICKER}] Running passive_data_processor -> ${RAW_CSV}"
./passive/passive_data_processor "${STOCK_DIR}" "${RAW_CSV}" \
    -H "${HAWKES_BETA}" \
    -I "${TRIGGER_INTENSITY}" \
    -L "${MAX_BBO_LEVELS}" \
    -v "${VOL_FRAC}" \
    -d "${DIR_THRESH}" \
    -r "${VOL_RATIO}" \
    -k "${KAPPA}" \
    -j "${WORKERS}" \
    -b "${RTH_START}" -e "${RTH_END}"

# ── Step 3: Compute Transformed Price Drift targets ──────────
FILTERED_CSV="results/passive/passive_bursts_${TICKER}_raw_filtered.csv"

echo "[${TICKER}] Computing Transformed Price Drift targets..."
python3 passive/src_py/passive_compute_permanence.py \
    "${RAW_CSV}" open_all.csv close_all.csv \
    --kappa "${KAPPA}" --ticker "${TICKER}"

echo ""
echo "DONE: Passive burst detection complete for ${TICKER}."
echo "Raw output:      ${RAW_CSV}"
echo "Filtered output: ${FILTERED_CSV}"
