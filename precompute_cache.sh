#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/precompute_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=2:00:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

set -Eeo pipefail

mkdir -p logs

# ────────────────────────────────────────────────────────────────
# Precompute burst cache files ONLY (no model training)
#
# This is the FAST part: C++ data_processor + compute_permanence.py
# For each ticker, generates raw burst CSVs and _filtered CSVs
# for all silence thresholds. Skips files that already exist.
#
# Array: 1 task per ticker (4 total)
# Expected runtime: ~10-20 minutes per ticker
# ────────────────────────────────────────────────────────────────

TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
SILENCE_VALUES="0.5 1.0 2.0"

read -r -a TICKER_ARR <<< "${TICKERS}"

if [ -n "${SGE_TASK_ID:-}" ]; then
  IDX=$((SGE_TASK_ID - 1))
  N=${#TICKER_ARR[@]}
  if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${N}" ]; then
    echo "INFO: Task ${SGE_TASK_ID} out of range; exiting."
    exit 0
  fi
  TICKER_LIST="${TICKER_ARR[$IDX]}"
else
  TICKER_LIST="${TICKERS}"
fi

for TICKER in ${TICKER_LIST}; do
  CACHE_DIR="${ROOT}/results/silence_sweep_${TICKER}/logreg_l2/shared_cache"
  mkdir -p "${CACHE_DIR}"

  for S in ${SILENCE_VALUES}; do
    S_TAG=$(echo "${S}" | tr '.' 'p')
    RAW_CSV="${CACHE_DIR}/bursts_${TICKER}_s${S_TAG}.csv"
    FILT_CSV="${CACHE_DIR}/bursts_${TICKER}_s${S_TAG}_filtered.csv"

    # Step 1: Run C++ data_processor (if raw CSV missing)
    if [ -s "${RAW_CSV}" ]; then
      echo "[SKIP] Raw CSV exists: ${RAW_CSV} ($(wc -l < "${RAW_CSV}") lines)"
    else
      echo "[RUN]  data_processor: ${TICKER} s=${S}"
      ./data_processor \
        "${ROOT}/data/${TICKER}" \
        "${RAW_CSV}" \
        -s "${S}" \
        -v 1 \
        -d 0.5 \
        -r 1.0 \
        -k 0 \
        -t 10.0 \
        -j "${NSLOTS:-4}" \
        -b 34200.0 \
        -e 57600.0
      echo "  -> $(wc -l < "${RAW_CSV}") lines"
    fi

    # Step 2: Run compute_permanence.py (if filtered CSV missing)
    if [ -s "${FILT_CSV}" ]; then
      echo "[SKIP] Filtered CSV exists: ${FILT_CSV} ($(wc -l < "${FILT_CSV}") lines)"
    else
      echo "[RUN]  compute_permanence: ${TICKER} s=${S}"
      python3 src_py/compute_permanence.py \
        "${RAW_CSV}" \
        "${ROOT}/open_all.csv" \
        "${ROOT}/close_all.csv" \
        --kappa 0
      echo "  -> $(wc -l < "${FILT_CSV}") lines"
    fi

    echo ""
  done

  echo "=== ${TICKER} precompute complete ==="
  echo ""
done

echo "All precompute tasks done at $(date)"
