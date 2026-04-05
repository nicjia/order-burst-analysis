#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/optuna_direct_$JOB_ID_$TASK_ID.out
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
# Direct Optuna Evaluation on Precomputed Cache Files
#
# Runs lgb_tuned and xgb_tuned directly on the full precomputed
# burst CSVs (bursts_<TICKER>_s<X>_filtered.csv) without needing
# the sweep to finish first. Uses the subsample limit built into
# train_model_zoo.py so even 6M-row files are fast.
#
# Array layout: N_TICKERS × N_SILENCE = 4 × 3 = 12 tasks
# ────────────────────────────────────────────────────────────────

if [ -z "${SGE_TASK_ID:-}" ]; then
  echo "ERROR: Must run as an SGE array task." >&2
  exit 1
fi

TICKERS=(NVDA TSLA JPM MS)
SILENCE_TAGS=(s0p5 s1p0 s2p0)
MODELS="lgb_tuned xgb_tuned"
TARGETS="cls_1m,cls_10m,cls_close"

N_TICKERS=${#TICKERS[@]}
N_SILENCE=${#SILENCE_TAGS[@]}
TOTAL=$((N_TICKERS * N_SILENCE))

IDX=$((SGE_TASK_ID - 1))
if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${TOTAL}" ]; then
  echo "INFO: Task ${SGE_TASK_ID} out of range TOTAL=${TOTAL}; exiting."
  exit 0
fi

TIDX=$((IDX / N_SILENCE))
SIDX=$((IDX % N_SILENCE))
TICKER=${TICKERS[$TIDX]}
STAG=${SILENCE_TAGS[$SIDX]}

# The precomputed filtered burst CSV
BURST_CSV="results/silence_sweep_${TICKER}/logreg_l2/shared_cache/bursts_${TICKER}_${STAG}_filtered.csv"

if [ ! -s "${BURST_CSV}" ]; then
  echo "ERROR: Missing burst CSV: ${BURST_CSV}" >&2
  exit 1
fi

echo "==========================================="
echo "Direct Optuna Evaluation"
echo "  Job ID:     ${JOB_ID:-N/A}"
echo "  Task ID:    ${SGE_TASK_ID} / ${TOTAL}"
echo "  Ticker:     ${TICKER}"
echo "  Silence:    ${STAG}"
echo "  Burst CSV:  ${BURST_CSV} ($(wc -l < "${BURST_CSV}") lines)"
echo "  Models:     ${MODELS}"
echo "  Targets:    ${TARGETS}"
echo "  Host:       $(hostname)"
echo "  Date:       $(date)"
echo "==========================================="

IFS=' ' read -ra MODEL_ARR <<< "${MODELS}"

for MODEL in "${MODEL_ARR[@]}"; do
  OUTDIR="results/optuna_direct/${TICKER}/${MODEL}/${STAG}"
  mkdir -p "${OUTDIR}"

  # Skip if all target JSONs already exist
  ALL_EXIST=true
  IFS=',' read -ra TGT_ARR <<< "${TARGETS}"
  for tgt in "${TGT_ARR[@]}"; do
    if [ ! -f "${OUTDIR}/${MODEL}__${tgt}.json" ]; then
      ALL_EXIST=false
      break
    fi
  done

  if [ "${ALL_EXIST}" = true ]; then
    echo "SKIP (already done): ${TICKER} / ${MODEL} / ${STAG}"
    continue
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Running: ${TICKER} / ${MODEL} / ${STAG}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  python3 src_py/train_model_zoo.py "${BURST_CSV}" \
    --model "${MODEL}" \
    --target "${TARGETS}" \
    --features extended \
    --outdir "${OUTDIR}" \
    --min-train-months 3
done

echo ""
echo "==========================================="
echo "Completed: ${TICKER} / ${STAG} at $(date)"
echo "==========================================="
