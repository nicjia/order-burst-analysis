#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/zoo_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=4:00:00
#$ -pe shared 2
#$ -t 1-147


ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6

set -Eeo pipefail

source "${ROOT}/.venv/bin/activate"

mkdir -p logs

TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
UNFILTERED_SUFFIX=${UNFILTERED_SUFFIX:-baseline_unfiltered}
FILTERED_SUFFIX=${FILTERED_SUFFIX:-baseline_filtered}

PHASE1_JOBS=84
if [ "${SGE_TASK_ID}" -le "${PHASE1_JOBS}" ]; then
  INDEX=$((SGE_TASK_ID - 1))
  TARGET="short"
  PHASE_TAG="unfiltered"
else
  INDEX=$((SGE_TASK_ID - PHASE1_JOBS - 1))
  TARGET="long"
  PHASE_TAG="filtered"
fi

echo "=========================================="
echo "Train Zoo Job"
echo "  Job ID:      ${JOB_ID:-N/A}"
echo "  Task ID:     ${SGE_TASK_ID:-N/A}"
echo "  Index:       ${INDEX}"
echo "  Target mode: ${TARGET}"
echo "  Tickers:     ${TICKERS}"
echo "  Hostname:    $(hostname)"
echo "  Date:        $(date)"
echo "=========================================="

for TICKER in ${TICKERS}; do
  if [ "${PHASE_TAG}" = "unfiltered" ]; then
    BURSTS_CSV="results/bursts_${TICKER}_${UNFILTERED_SUFFIX}.csv"
  else
    BURSTS_CSV="results/bursts_${TICKER}_${FILTERED_SUFFIX}.csv"
  fi

  if [ ! -s "${BURSTS_CSV}" ]; then
    echo "WARN: Missing or empty input ${BURSTS_CSV}; skipping ${TICKER}"
    continue
  fi

  OUTDIR="results/zoo_bursts_${TICKER}_${PHASE_TAG}/"
  mkdir -p "${OUTDIR}"

  echo "Running ${TICKER} | ${TARGET} | ${BURSTS_CSV}"
  python3 src_py/train_model_zoo.py "${BURSTS_CSV}" \
    --model all \
    --target "${TARGET}" \
    --features extended \
    --outdir "${OUTDIR}" \
    --slurm-index "${INDEX}"
done

echo "Task ${SGE_TASK_ID} finished at $(date)"