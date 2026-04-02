#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/prep_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=6:00:00
#$ -pe shared 4



ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6

set -Eeo pipefail

source "${ROOT}/.venv/bin/activate"

mkdir -p logs results

echo "=========================================="
echo "Prep Data Job"
echo "  Job ID:      ${JOB_ID:-N/A}"
echo "  Task ID:     ${SGE_TASK_ID:-N/A}"
echo "  Hostname:    $(hostname)"
echo "  Date:        $(date)"
echo "=========================================="

TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
BASE_SILENCE=${BASE_SILENCE:-0.5}
BASE_MIN_VOL=${BASE_MIN_VOL:-100}
BASE_DIR_THRESH=${BASE_DIR_THRESH:-0.8}
BASE_VOL_RATIO=${BASE_VOL_RATIO:-0.3}
BASE_TAU_MAX=${BASE_TAU_MAX:-10.0}
BASE_KAPPA_LONG=${BASE_KAPPA_LONG:-0.5}
FORCE_REBUILD_BASELINE=${FORCE_REBUILD_BASELINE:-0}
WORKERS=${WORKERS:-${NSLOTS:-1}}

if [ -z "${SGE_TASK_ID:-}" ]; then
  echo "ERROR: prep_data.sh must run as an SGE array task (SGE_TASK_ID missing)." >&2
  exit 1
fi

read -r -a TICKER_ARR <<< "${TICKERS}"
N_TICKERS=${#TICKER_ARR[@]}
if [ "${N_TICKERS}" -eq 0 ]; then
  echo "ERROR: TICKERS is empty." >&2
  exit 1
fi

TASK_INDEX=$((SGE_TASK_ID - 1))
if [ "${TASK_INDEX}" -lt 0 ] || [ "${TASK_INDEX}" -ge "${N_TICKERS}" ]; then
  echo "INFO: Task ${SGE_TASK_ID} has no ticker assignment (N=${N_TICKERS}); exiting."
  exit 0
fi

TICKER=${TICKER_ARR[$TASK_INDEX]}
STOCK_DIR="${ROOT}/data/${TICKER}"
RAW_CSV="results/bursts_${TICKER}_baseline.csv"
SHORT_CSV="results/bursts_${TICKER}_baseline_unfiltered.csv"
LONG_CSV="results/bursts_${TICKER}_baseline_filtered.csv"
LONG_SEED="results/bursts_${TICKER}_baseline_longseed.csv"
LONG_SEED_OUT="results/bursts_${TICKER}_baseline_longseed_filtered.csv"

echo "Assigned ticker: ${TICKER}"
echo "Data folder: ${STOCK_DIR}"

if [ ! -d "${STOCK_DIR}" ]; then
  echo "ERROR: Missing stock folder ${STOCK_DIR}" >&2
  exit 1
fi
if ! ls "${STOCK_DIR}"/*_message_*.csv >/dev/null 2>&1; then
  echo "ERROR: No *_message_*.csv files found in ${STOCK_DIR}" >&2
  exit 1
fi

if [ ! -x "${ROOT}/data_processor" ]; then
  echo "INFO: data_processor missing; rebuilding"
  make clean && make
fi

echo "Compiler: $(g++ --version | head -n 1)"

if [ "${FORCE_REBUILD_BASELINE}" = "1" ]; then
  echo "INFO: FORCE_REBUILD_BASELINE=1 -> removing prior outputs for ${TICKER}"
  rm -f "${RAW_CSV}" "${SHORT_CSV}" "${LONG_CSV}" "${LONG_SEED}" "${LONG_SEED_OUT}"
fi

if [ -f "${RAW_CSV}" ] && [ ! -s "${RAW_CSV}" ]; then rm -f "${RAW_CSV}"; fi
if [ -f "${SHORT_CSV}" ] && [ ! -s "${SHORT_CSV}" ]; then rm -f "${SHORT_CSV}"; fi
if [ -f "${LONG_CSV}" ] && [ ! -s "${LONG_CSV}" ]; then rm -f "${LONG_CSV}"; fi

if [ ! -f "${RAW_CSV}" ]; then
  echo "INFO: Running data_processor for ${TICKER}"
  ./data_processor "${STOCK_DIR}" "${RAW_CSV}" \
    -s "${BASE_SILENCE}" \
    -v "${BASE_MIN_VOL}" \
    -d "${BASE_DIR_THRESH}" \
    -r "${BASE_VOL_RATIO}" \
    -k 0 \
    -t "${BASE_TAU_MAX}" \
    -j "${WORKERS}" \
    -b 34200 -e 57600
fi

if [ ! -s "${RAW_CSV}" ]; then
  echo "ERROR: Missing/empty raw baseline CSV: ${RAW_CSV}" >&2
  exit 1
fi

if [ ! -f "${SHORT_CSV}" ]; then
  echo "INFO: Building short-horizon permanence CSV (kappa=0) for ${TICKER}"
  python3 src_py/compute_permanence.py \
    "${RAW_CSV}" \
    "${ROOT}/open_all.csv" \
    "${ROOT}/close_all.csv" \
    --kappa 0

  # compute_permanence writes *_filtered.csv next to input file.
  if [ -f "results/bursts_${TICKER}_baseline_filtered.csv" ]; then
    mv "results/bursts_${TICKER}_baseline_filtered.csv" "${SHORT_CSV}"
  fi
fi

if [ ! -s "${SHORT_CSV}" ]; then
  echo "ERROR: Missing/empty short permanence CSV: ${SHORT_CSV}" >&2
  exit 1
fi

if [ ! -f "${LONG_CSV}" ]; then
  echo "INFO: Building long-horizon permanence CSV (kappa=${BASE_KAPPA_LONG}) for ${TICKER}"
  cp "${RAW_CSV}" "${LONG_SEED}"
  python3 src_py/compute_permanence.py \
    "${LONG_SEED}" \
    "${ROOT}/open_all.csv" \
    "${ROOT}/close_all.csv" \
    --kappa "${BASE_KAPPA_LONG}"

  if [ -f "${LONG_SEED_OUT}" ]; then
    mv "${LONG_SEED_OUT}" "${LONG_CSV}"
  fi
  rm -f "${LONG_SEED}"
fi

if [ ! -s "${LONG_CSV}" ]; then
  echo "ERROR: Missing/empty long permanence CSV: ${LONG_CSV}" >&2
  exit 1
fi

echo "INFO: Prep complete for ${TICKER}"
echo "  Raw:   ${RAW_CSV}"
echo "  Short: ${SHORT_CSV}"
echo "  Long:  ${LONG_CSV}"