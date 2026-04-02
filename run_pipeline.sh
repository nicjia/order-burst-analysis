#!/bin/bash
# Master submitter: prep -> train (hold_jid dependency)

set -Eeo pipefail

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 \"NVDA TSLA JPM MS\""
  exit 1
fi

TICKERS="$1"
read -r -a TICKER_ARR <<< "${TICKERS}"
N_TICKERS=${#TICKER_ARR[@]}
if [ "${N_TICKERS}" -eq 0 ]; then
  echo "ERROR: no tickers provided"
  exit 1
fi

if [ ! -f "prep_data.sh" ] || [ ! -f "train_zoo.sh" ]; then
  echo "ERROR: prep_data.sh or train_zoo.sh not found in ${ROOT}" >&2
  exit 1
fi

mkdir -p logs

echo "Submitting prep_data.sh for ${N_TICKERS} tickers: ${TICKERS}"
PREP_OUT=$(qsub -t 1-"${N_TICKERS}" -v TICKERS="${TICKERS}",FORCE_REBUILD_BASELINE=1 prep_data.sh)
echo "${PREP_OUT}"

PREP_JOB_ID=$(echo "${PREP_OUT}" | sed -n 's/.*job[- ]array \([0-9][0-9]*\).*/\1/p')
if [ -z "${PREP_JOB_ID}" ]; then
  PREP_JOB_ID=$(echo "${PREP_OUT}" | sed -n 's/.*job \([0-9][0-9]*\).*/\1/p')
fi

if [ -z "${PREP_JOB_ID}" ]; then
  echo "ERROR: could not parse prep job id from qsub output" >&2
  exit 1
fi

echo "Submitting train_zoo.sh with hold_jid=${PREP_JOB_ID}"
TRAIN_OUT=$(qsub -hold_jid "${PREP_JOB_ID}" -v TICKERS="${TICKERS}" train_zoo.sh)
echo "${TRAIN_OUT}"

echo "Pipeline submitted successfully."
echo "  Prep job id:  ${PREP_JOB_ID}"
echo "  Tickers:      ${TICKERS}"
echo "Use: qstat -u nicjia"