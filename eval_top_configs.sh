#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/eval_topcfg_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=4:00:00
#$ -pe shared 2

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

set -Eeo pipefail


mkdir -p logs

if [ -z "${SGE_TASK_ID:-}" ]; then
  echo "ERROR: eval_top_configs.sh must run as an SGE array task." >&2
  exit 1
fi

TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
MODELS=${MODELS:-"et rf stacking lgb_tuned adaboost"}
TOP_CONFIGS_FILE=${TOP_CONFIGS_FILE:-"results/sweep_rankings/top5_configs.txt"}

# Reduced target set — expand on final winning configs.
SHORT_TARGETS="cls_1m,cls_10m"
LONG_TARGETS="cls_close"

if [ ! -f "${TOP_CONFIGS_FILE}" ]; then
  echo "ERROR: Missing TOP_CONFIGS_FILE=${TOP_CONFIGS_FILE}" >&2
  exit 1
fi

read -r -a TICKER_ARR <<< "${TICKERS}"
if [[ "${MODELS}" == *,* ]]; then
  IFS=',' read -r -a MODEL_ARR <<< "${MODELS}"
else
  read -r -a MODEL_ARR <<< "${MODELS}"
fi

N_TICKERS=${#TICKER_ARR[@]}
N_MODELS=${#MODEL_ARR[@]}
TOTAL=$((N_TICKERS * N_MODELS))

IDX=$((SGE_TASK_ID - 1))
if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${TOTAL}" ]; then
  echo "INFO: Task ${SGE_TASK_ID} out of range TOTAL=${TOTAL}; exiting."
  exit 0
fi

TIDX=$((IDX / N_MODELS))
MIDX=$((IDX % N_MODELS))
TICKER=${TICKER_ARR[$TIDX]}
MODEL=$(echo "${MODEL_ARR[$MIDX]}" | xargs)

echo "=========================================="
echo "Top-Config Model Eval"
echo "  Job ID:   ${JOB_ID:-N/A}"
echo "  Task ID:  ${SGE_TASK_ID} / ${TOTAL}"
echo "  Ticker:   ${TICKER}"
echo "  Model:    ${MODEL}"
echo "  Host:     $(hostname)"
echo "  Date:     $(date)"
echo "=========================================="

mapfile -t CONFIGS < "${TOP_CONFIGS_FILE}"
if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "ERROR: No configs listed in ${TOP_CONFIGS_FILE}" >&2
  exit 1
fi

for cfg in "${CONFIGS[@]}"; do
  cfg=$(echo "${cfg}" | xargs)
  [ -z "${cfg}" ] && continue

  for phase in short long; do
    CAND="results/silence_sweep_${TICKER}/logreg_l2/${phase}/candidates/${TICKER}_${cfg}.csv"
    if [ ! -s "${CAND}" ]; then
      echo "WARN: Missing candidate ${CAND}; skipping"
      continue
    fi

    OUTDIR="results/topcfg_eval/${TICKER}/${MODEL}/${cfg}/${phase}"
    mkdir -p "${OUTDIR}"

    echo "Running cfg=${cfg} phase=${phase}"
    if [ "${phase}" = "short" ]; then
      TARGET_ARG="${SHORT_TARGETS}"
    else
      TARGET_ARG="${LONG_TARGETS}"
    fi
    python3 src_py/train_model_zoo.py "${CAND}" \
      --model "${MODEL}" \
      --target "${TARGET_ARG}" \
      --features extended \
      --outdir "${OUTDIR}" \
      --min-train-months 3
  done
done

echo "Completed ${TICKER} ${MODEL} at $(date)"
