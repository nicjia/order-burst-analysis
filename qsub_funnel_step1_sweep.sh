#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/funnel_step1_sweep_$JOB_ID_$TASK_ID.out
#$ -l h_data=16G,h_rt=24:00:00
#$ -pe shared 8
#$ -t 1-4
#$ -N burst_sweep

set -Eeo pipefail
trap 'echo "ERROR line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
TICKERS=(NVDA TSLA JPM MS)
IDX=$((SGE_TASK_ID - 1))
TICKER=${TICKERS[$IDX]}

if [ -z "${TICKER}" ]; then
  echo "Invalid SGE_TASK_ID=${SGE_TASK_ID}" >&2
  exit 2
fi

cd "${ROOT}"
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

mkdir -p "${ROOT}/logs" "${ROOT}/results"

SILENCE_VALUES=${SILENCE_VALUES:-0.5,1.0,2.0}
MIN_VOL_VALUES=${MIN_VOL_VALUES:-50,100,200,500}
DIR_THRESH_VALUES=${DIR_THRESH_VALUES:-0.7,0.8,0.9}
VOL_RATIO_VALUES=${VOL_RATIO_VALUES:-0.1,0.3,0.5}
KAPPA_VALUES=${KAPPA_VALUES:-0.2,0.5,1.0}
SWEEP_MODEL=${SWEEP_MODEL:-logreg_l2}
SWEEP_TARGETS=${SWEEP_TARGETS:-cls_1m,cls_5m,cls_10m,cls_close}
OUTDIR="${ROOT}/results/silence_sweep_${TICKER}"

python3 src_py/silence_optimized_sweep.py \
  --stock-folder "${ROOT}/data/${TICKER}" \
  --ticker "${TICKER}" \
  --open "${ROOT}/open_all.csv" \
  --close "${ROOT}/close_all.csv" \
  --data-processor "${ROOT}/data_processor" \
  --outdir "${OUTDIR}" \
  --silence-values "${SILENCE_VALUES}" \
  --min-vol-values "${MIN_VOL_VALUES}" \
  --dir-thresh-values "${DIR_THRESH_VALUES}" \
  --vol-ratio-values "${VOL_RATIO_VALUES}" \
  --kappa-values "${KAPPA_VALUES}" \
  --tau-max 10.0 \
  --workers "${NSLOTS:-1}" \
  --model "${SWEEP_MODEL}" \
  --target "${SWEEP_TARGETS}" \
  --features extended \
  --min-train-months 3 \
  --require-directional \
  --min-rows 500

echo "Step1 sweep done for ${TICKER} at $(date)"