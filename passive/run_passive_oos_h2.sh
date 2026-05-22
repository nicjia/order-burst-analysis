#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -t 1-3
#$ -o logs/passive_oos_$JOB_ID.$TASK_ID.out
#$ -l h_data=4G,h_rt=02:59:00
#$ -pe shared 4

# ─────────────────────────────────────────────────────────────
# run_passive_oos_h2.sh
# OOS-only passive detection + evaluation on LLY, SPY, AAPL
# Uses frozen params from Core Optuna sweep (NVDA/TSLA/JPM/MS).
# Task 1=LLY, 2=SPY, 3=AAPL
# ─────────────────────────────────────────────────────────────

ROOT=${ROOT:-/u/scratch/n/nicjia/order-burst-analysis}
TICKERS_ARRAY=("LLY" "SPY" "AAPL")

# Fixed Hawkes (same as training)
HAWKES_BETA=1.0
TRIGGER_INTENSITY=0.5
MAX_BBO_LEVELS=3
RTH_START=34200
RTH_END=57600
WORKERS=${NSLOTS:-4}

cd "${ROOT}"
mkdir -p logs results/passive results/oos_passive

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

if [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ]; then
    idx=$((SGE_TASK_ID - 1))
    TICKER="${TICKERS_ARRAY[$idx]}"
else
    TICKER="${TICKERS_ARRAY[0]}"
    echo "WARNING: No task ID; defaulting to ${TICKER}"
fi

echo "==============================================="
echo "PASSIVE OOS: ${TICKER}"
echo "Using frozen core params"
echo "==============================================="

STOCK_DIR="${ROOT}/data/${TICKER}"
RAW_CSV="results/passive/passive_bursts_${TICKER}_raw.csv"
FILTERED_CSV="results/passive/passive_bursts_${TICKER}_raw_filtered.csv"

if [ ! -d "${STOCK_DIR}" ]; then
    echo "ERROR: ${STOCK_DIR} not found" >&2; exit 1
fi

# ── Step 1: Compile if needed ─────────────────────────────────
if [ ! -x "${ROOT}/passive/passive_data_processor" ]; then
    echo "Compiling passive_data_processor..."
    cd "${ROOT}/passive" && make clean && make && cd "${ROOT}"
fi

# ── Step 2: Run passive detector ──────────────────────────────
echo "[${TICKER}] Running passive_data_processor..."
./passive/passive_data_processor "${STOCK_DIR}" "${RAW_CSV}" \
    -H "${HAWKES_BETA}" -I "${TRIGGER_INTENSITY}" \
    -L "${MAX_BBO_LEVELS}" -v 0 -d 0.5 -r 1.0 -k 0 \
    -j "${WORKERS}" -b "${RTH_START}" -e "${RTH_END}"

# ── Step 3: Compute Transformed Price Drift targets ───────────
echo "[${TICKER}] Computing Transformed Price Drift targets..."
python3 passive/src_py/passive_compute_permanence.py \
    "${RAW_CSV}" open_all.csv close_all.csv \
    --kappa 0 --ticker "${TICKER}"

# ── Step 4: Run OOS eval with frozen Core params ─────────────────
echo "[${TICKER}] Running OOS evaluation with frozen Core params..."
python3 passive/src_py/passive_oos_eval.py \
    --ticker "${TICKER}" \
    --target reg_clop \
    --use-core-params "results/optuna_passive/best_params_core_reg_clop.json" \
    --out-json "results/oos_passive/oos_${TICKER}_reg_clop.json"

echo "DONE: ${TICKER} OOS complete."
