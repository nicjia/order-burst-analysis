#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/eval_optuna_$JOB_ID_$TASK_ID.out
#$ -l h_data=12G,h_rt=8:00:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

set -Eeo pipefail

mkdir -p logs

# ────────────────────────────────────────────────────────────────
# Optuna-Tuned Model Evaluation on Top Sweep Configurations
#
# This script evaluates Optuna-tuned models (lgb_tuned, xgb_tuned)
# on the top-ranked parameter configurations from the sweep.
# Unlike eval_top_configs.sh which runs 5 models × 4 tickers = 20 tasks,
# this focuses specifically on Bayesian-optimized model variants.
#
# Array layout: (N_TICKERS × N_MODELS) tasks
#   e.g. 4 tickers × 2 models = 8 total tasks
# ────────────────────────────────────────────────────────────────

if [ -z "${SGE_TASK_ID:-}" ]; then
  echo "ERROR: Must run as an SGE array task." >&2
  exit 1
fi

TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
# Optuna-tuned models only
MODELS=${MODELS:-"lgb_tuned xgb_tuned"}
SWEEP_PREFIX=${SWEEP_PREFIX:-"silence_sweep"}
TOP_CONFIGS_FILE=${TOP_CONFIGS_FILE:-"results/sweep_rankings/top5_configs.txt"}

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

echo "==========================================="
echo "Optuna-Tuned Model Evaluation"
echo "  Job ID:       ${JOB_ID:-N/A}"
echo "  Task ID:      ${SGE_TASK_ID} / ${TOTAL}"
echo "  Ticker:       ${TICKER}"
echo "  Model:        ${MODEL}"
echo "  Sweep prefix: ${SWEEP_PREFIX}"
echo "  Host:         $(hostname)"
echo "  Date:         $(date)"
echo "==========================================="

mapfile -t CONFIGS < "${TOP_CONFIGS_FILE}"
if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "ERROR: No configs in ${TOP_CONFIGS_FILE}" >&2
  exit 1
fi

PASSED=0
FAILED=0

for cfg in "${CONFIGS[@]}"; do
  cfg=$(echo "${cfg}" | xargs)
  [ -z "${cfg}" ] && continue

  for phase in short long; do
    # Look for candidate CSV from the logreg_l2 sweep
    CAND="results/${SWEEP_PREFIX}_${TICKER}/logreg_l2/${phase}/candidates/${TICKER}_${cfg}.csv"
    if [ ! -s "${CAND}" ]; then
      echo "WARN: Missing candidate ${CAND}; skipping"
      ((FAILED++)) || true
      continue
    fi

    OUTDIR="results/optuna_eval/${TICKER}/${MODEL}/${cfg}/${phase}"
    mkdir -p "${OUTDIR}"

    # Check if already completed (skip-existing logic)
    if [ "${phase}" = "short" ]; then
      EXPECTED_TARGETS="cls_1m cls_3m cls_5m cls_10m"
    else
      EXPECTED_TARGETS="cls_close cls_clop cls_clcl"
    fi

    ALL_EXIST=true
    for tgt in ${EXPECTED_TARGETS}; do
      if [ ! -f "${OUTDIR}/${MODEL}__${tgt}.json" ]; then
        ALL_EXIST=false
        break
      fi
    done

    if [ "${ALL_EXIST}" = true ]; then
      echo "SKIP (already done): cfg=${cfg} phase=${phase}"
      ((PASSED++)) || true
      continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: ${TICKER} / ${MODEL} / ${cfg} / ${phase}"
    echo "  Candidate: ${CAND} ($(wc -l < "${CAND}") lines)"
    echo "  Output:    ${OUTDIR}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    python3 src_py/train_model_zoo.py "${CAND}" \
      --model "${MODEL}" \
      --target "${phase}" \
      --features extended \
      --outdir "${OUTDIR}" \
      --min-train-months 3

    ((PASSED++)) || true
  done
done

echo ""
echo "==========================================="
echo "Completed: ${TICKER} / ${MODEL}"
echo "  Passed: ${PASSED}"
echo "  Failed/Missing: ${FAILED}"
echo "  Finished at: $(date)"
echo "==========================================="
