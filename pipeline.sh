#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/selection_pipeline_$JOB_ID.out
#$ -l h_data=8G,h_rt=12:00:00
#$ -pe shared 8
#$ -N tsla_sweep_phase

set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
TICKER=${TICKER:-TSLA}
WORKERS=${NSLOTS:-1}

# --- FILES ---
BURSTS_CSV="${ROOT}/results/bursts_${TICKER}_baseline.csv"
MASTER_FILTERED_CSV="${ROOT}/results/bursts_${TICKER}_baseline_filtered.csv"
ZOO_INPUT_CSV="${ROOT}/results/bursts_${TICKER}_zoo_baseline_input.csv"
ZOO_TEMP_CSV="${ROOT}/results/bursts_${TICKER}_zoo_baseline_input_filtered.csv"

ZOO_OUTDIR="${ROOT}/results/model_select_${TICKER}"

# --- PHASE I: ZOO PARAMETERS ---
KAPPA_ZOO=${KAPPA_ZOO:-0.5}
MODEL_KEY=${MODEL_KEY:-all_cls}
TARGET_KEY=${TARGET_KEY:-all}

# --- PHASE II: SWEEP PARAMETERS ---
RUN_SWEEP=1
SWEEP_MODEL="xgb" 
SWEEP_TARGETS=("cls_1m" "cls_5m" "cls_10m" "cls_close") 
SILENCE_VALUES=${SILENCE_VALUES:-0.5,1.0,2.0}
MIN_VOL_VALUES=${MIN_VOL_VALUES:-50,100,200,2000}
DIR_THRESH_VALUES=${DIR_THRESH_VALUES:-0.7,0.8,0.9}
VOL_RATIO_VALUES=${VOL_RATIO_VALUES:-0.1,0.3,0.5}
KAPPA_VALUES=${KAPPA_VALUES:-0.2,0.5,1.0}

mkdir -p "${ROOT}/logs" "${ROOT}/results"
cd "${ROOT}"

# --- Environment ---
. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"

stage_time() {
  local stage_name="$1"
  shift
  local t0 t1 dt
  echo -e "\n---- ${stage_name} START: $(date '+%F %T') ----"
  t0=$(date +%s)
  "$@"
  t1=$(date +%s)
  dt=$((t1 - t0))
  echo "---- ${stage_name} END:   $(date '+%F %T') (${dt}s) ----"
}

echo "========== SELECTION PIPELINE START =========="
echo "Job ID: ${JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date '+%F %T')"
echo "NSLOTS: ${NSLOTS:-N/A}"
echo "Ticker: ${TICKER}"
echo "Workers: ${WORKERS}"
echo "Run sweep: ${RUN_SWEEP}"
echo "Sweep model: ${SWEEP_MODEL}"
echo "Sweep targets: ${SWEEP_TARGETS[*]}"
echo "=============================================="

if [ ! -f "${BURSTS_CSV}" ]; then
  stage_time "DATA_PROCESSOR_BASELINE" \
    ./data_processor "${ROOT}/data/${TICKER}" "${BURSTS_CSV}" \
    -s 1.0 -v 1 -d 0.5 -r 1.0 -k 0 -t 10.0 -j "${WORKERS}" -b 34200 -e 57600
fi

if [ ! -f "${MASTER_FILTERED_CSV}" ]; then
  stage_time "MASTER_PERMANENCE_K0" \
    python "${ROOT}/src_py/compute_permanence.py" \
    "${BURSTS_CSV}" "${ROOT}/open_all.csv" "${ROOT}/close_all.csv" --kappa 0
fi

# We are skipping the ZOO rebuild here because you already have your best model (xgb)

# --- STEP 3: THE MULTI-TARGET SWEEP ---
if [ "${RUN_SWEEP}" = "1" ] && [ "${SWEEP_MODEL}" != "PENDING" ]; then
  SWEEP_TARGET_ARG=$(IFS=, ; echo "${SWEEP_TARGETS[*]}")
  SWEEP_OUTDIR="${ROOT}/results/silence_sweep_${TICKER}"

  stage_time "SILENCE_SWEEP_MULTI_TARGET" \
    python "${ROOT}/src_py/silence_optimized_sweep.py" \
    --stock-folder "${ROOT}/data/${TICKER}" \
    --ticker "${TICKER}" \
    --open "${ROOT}/open_all.csv" \
    --close "${ROOT}/close_all.csv" \
    --data-processor "${ROOT}/data_processor" \
    --outdir "${SWEEP_OUTDIR}" \
    --silence-values "${SILENCE_VALUES}" \
    --min-vol-values "${MIN_VOL_VALUES}" \
    --dir-thresh-values "${DIR_THRESH_VALUES}" \
    --vol-ratio-values "${VOL_RATIO_VALUES}" \
    --kappa-values "${KAPPA_VALUES}" \
    --tau-max 10.0 \
    --workers "${WORKERS}" \
    --model "${SWEEP_MODEL}" \
    --target "${SWEEP_TARGET_ARG}" \
    --features extended \
    --min-train-months 3 \
    --require-directional \
    --min-rows 500
else
  echo -e "\nPHASE I COMPLETE. Zoo data used kappa=${KAPPA_ZOO}."
  echo "Set RUN_SWEEP=1 and provide SWEEP_MODEL to start sweep phase."
fi

echo -e "\n========== SELECTION PIPELINE DONE =========="
echo "End: $(date '+%F %T')"