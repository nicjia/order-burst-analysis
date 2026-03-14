#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/selection_pipeline_$JOB_ID.out
#$ -l h_data=8G,h_rt=12:00:00
#$ -pe shared 8
#$ -N nvda_selection_resume

set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
TICKER=${TICKER:-NVDA}
WORKERS=${NSLOTS:-1}

# --- FILES ---
BURSTS_CSV="${ROOT}/results/bursts_${TICKER}_baseline.csv"
MASTER_FILTERED_CSV="${ROOT}/results/bursts_${TICKER}_baseline_filtered.csv"
ZOO_INPUT_CSV="${ROOT}/results/bursts_${TICKER}_zoo_baseline_input.csv"
ZOO_TEMP_CSV="${ROOT}/results/bursts_${TICKER}_zoo_baseline_input_filtered.csv"

ZOO_OUTDIR="${ROOT}/results/model_select_${TICKER}"
SWEEP_OUTDIR="${ROOT}/results/silence_sweep_${TICKER}"

# --- PHASE I: ZOO PARAMETERS ---
KAPPA_ZOO=${KAPPA_ZOO:-0.5}
MODEL_KEY=${MODEL_KEY:-all_cls}
TARGET_KEY=${TARGET_KEY:-all}

# --- PHASE II: SWEEP PARAMETERS (Update after Phase I) ---
RUN_SWEEP=${RUN_SWEEP:-0}
SWEEP_MODEL=${SWEEP_MODEL:-PENDING}
SWEEP_TARGET=${SWEEP_TARGET:-PENDING}
SILENCE_VALUES=${SILENCE_VALUES:-0.5,1.0,2.0}
MIN_VOL_VALUES=${MIN_VOL_VALUES:-50,100,200,2000}
DIR_THRESH_VALUES=${DIR_THRESH_VALUES:-0.7,0.8,0.9}
VOL_RATIO_VALUES=${VOL_RATIO_VALUES:-0.1,0.3,0.5}
KAPPA_VALUES=${KAPPA_VALUES:-0.2,0.5,1.0}

mkdir -p "${ROOT}/logs" "${ROOT}/results"
cd "${ROOT}"

# --- TEMPORARILY DISABLE STRICT ERROR HANDLING FOR SYSTEM SCRIPTS ---
set +Eeo pipefail
trap - ERR

if [ -f /u/local/Modules/default/init/bash ]; then
  . /u/local/Modules/default/init/bash
elif [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
fi

module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

# --- RE-ENABLE STRICT ERROR HANDLING ---
set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

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

echo "========== SELECTION PIPELINE RESUME START =========="
echo "Job ID: ${JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date '+%F %T')"
echo "NSLOTS: ${NSLOTS:-N/A}"
echo "Ticker: ${TICKER}"
echo "Workers: ${WORKERS}"
echo "Model: ${MODEL_KEY}"
echo "Target: ${TARGET_KEY}"
echo "Run sweep: ${RUN_SWEEP}"
echo "Sweep model: ${SWEEP_MODEL}"
echo "Sweep target: ${SWEEP_TARGET}"
echo "Sweep silence: ${SILENCE_VALUES}"
echo "=============================================="

# ==========================================================
# SKIPPING COMPLETED STAGES
# ==========================================================
# if [ ! -f "${BURSTS_CSV}" ]; then
#   stage_time "DATA_PROCESSOR_BASELINE" \
#     ./data_processor "${ROOT}/data/${TICKER}" "${BURSTS_CSV}" \
#     -s 1.0 -v 1 -d 0.5 -r 1.0 -k 0 -t 10.0 -j "${WORKERS}" -b 34200 -e 57600
# fi
# 
# if [ ! -f "${MASTER_FILTERED_CSV}" ]; then
#   stage_time "MASTER_PERMANENCE_K0" \
#     python "${ROOT}/src_py/compute_permanence.py" \
#     "${BURSTS_CSV}" "${ROOT}/open_all.csv" "${ROOT}/close_all.csv" --kappa 0
# fi
# 
# stage_time "PREPARE_ZOO_INPUT_COPY" cp "${BURSTS_CSV}" "${ZOO_INPUT_CSV}"
# 
# stage_time "PREPARE_ZOO_DATA" \
#   python "${ROOT}/src_py/compute_permanence.py" \
#   "${ZOO_INPUT_CSV}" "${ROOT}/open_all.csv" "${ROOT}/close_all.csv" \
#   --kappa "${KAPPA_ZOO}"
# ==========================================================

echo -e "\nSkipped Data Processor and Permanence steps. Resuming at MODEL_ZOO..."

if [ ! -f "${ZOO_TEMP_CSV}" ]; then
  echo "ERROR: expected zoo input file not found: ${ZOO_TEMP_CSV}" >&2
  exit 1
fi

# --- STEP 2: THE MODEL ZOO ---
stage_time "MODEL_ZOO" \
  python -u "${ROOT}/src_py/train_model_zoo.py" \
  "${ZOO_TEMP_CSV}" --model "${MODEL_KEY}" --target "${TARGET_KEY}" \
  --features extended --outdir "${ZOO_OUTDIR}"

# --- STEP 3: THE SWEEP ---
if [ "${RUN_SWEEP}" = "1" ] && [ "${SWEEP_MODEL}" != "PENDING" ] && [ "${SWEEP_TARGET}" != "PENDING" ]; then
  stage_time "SILENCE_SWEEP" \
    python -u "${ROOT}/src_py/silence_optimized_sweep.py" \
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
    --target "${SWEEP_TARGET}" \
    --features extended \
    --min-train-months 3 \
    --require-directional \
    --min-rows 500
else
  echo -e "\nPHASE I COMPLETE. Zoo data used kappa=${KAPPA_ZOO}."
  echo "Set RUN_SWEEP=1 and provide SWEEP_MODEL/SWEEP_TARGET to start sweep phase."
fi

echo -e "\n========== SELECTION PIPELINE DONE =========="
echo "End: $(date '+%F %T')"
echo "Master bursts: ${BURSTS_CSV}"
echo "Zoo input filtered (kappa=${KAPPA_ZOO}): ${ZOO_TEMP_CSV}"
echo "Zoo output: ${ZOO_OUTDIR}"
echo "Sweep output: ${SWEEP_OUTDIR}"