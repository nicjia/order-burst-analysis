#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/baseline_pipeline_$JOB_ID.out
#$ -l h_data=64G,h_rt=12:00:00
#$ -pe shared 8
#$ -N nvda_baseline_resume

set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
TICKER=${TICKER:-NVDA}
SILENCE=${SILENCE:-1.0}
WORKERS=${NSLOTS:-1}
MODEL_KEY=${MODEL_KEY:-all_cls}
TARGET_KEY=${TARGET_KEY:-cls_close}
SWEEP_MODEL=${SWEEP_MODEL:-logreg_l2}
SWEEP_TARGET=${SWEEP_TARGET:-cls_close}
SILENCE_VALUES=${SILENCE_VALUES:-0.5,1.0,2.0}
MIN_VOL_VALUES=${MIN_VOL_VALUES:-50,100,200}
DIR_THRESH_VALUES=${DIR_THRESH_VALUES:-0.8,0.9}
VOL_RATIO_VALUES=${VOL_RATIO_VALUES:-0.3,0.5}
KAPPA_VALUES=${KAPPA_VALUES:-0.0,0.1,0.2}

BURSTS_CSV="${ROOT}/results/bursts_${TICKER}_baseline.csv"
FILTERED_CSV="${ROOT}/results/bursts_${TICKER}_baseline_filtered.csv"
ZOO_OUTDIR="${ROOT}/results/model_select_${TICKER}"
SWEEP_OUTDIR="${ROOT}/results/silence_sweep_${TICKER}"

mkdir -p "${ROOT}/logs" "${ROOT}/results"

echo "========== BASELINE PIPELINE RESUME START =========="
echo "Job ID: ${JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date '+%F %T')"
echo "NSLOTS: ${NSLOTS:-N/A}"
echo "Ticker: ${TICKER}"
echo "Workers: ${WORKERS}"
echo "Model: ${MODEL_KEY}"
echo "Target: ${TARGET_KEY}"
echo "Sweep model: ${SWEEP_MODEL}"
echo "Sweep target: ${SWEEP_TARGET}"
echo "============================================="

cd "${ROOT}"

# --- TEMPORARILY DISABLE STRICT ERROR HANDLING FOR SYSTEM SCRIPTS ---
set +Eeo pipefail
trap - ERR

if [ -f /u/local/Modules/default/init/bash ]; then
  . /u/local/Modules/default/init/bash
elif [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
fi

module load gcc/11.3.0
module load python/3.9.6

source "${ROOT}/.venv/bin/activate"

# --- RE-ENABLE STRICT ERROR HANDLING ---
set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

TOTAL_START=$(date +%s)

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

# ==========================================================
# SKIPPING COMPLETED STAGES
# ==========================================================
# make clean || true
# stage_time "BUILD" make
#
# stage_time "DATA_PROCESSOR" \
#   ./data_processor "${ROOT}/data/${TICKER}" "${BURSTS_CSV}" \
#   -s "${SILENCE}" -v 1 -d 0.5 -r 1.0 -k 0 -t 10.0 -j "${WORKERS}" -b 34200 -e 57600
#
# stage_time "COMPUTE_PERMANENCE" \
#   python "${ROOT}/src_py/compute_permanence.py" \
#   "${BURSTS_CSV}" "${ROOT}/open_all.csv" "${ROOT}/close_all.csv" --kappa 0
# ==========================================================

echo -e "\nSkipped Build, Data Processor, and Permanence steps. Resuming at MODEL_ZOO..."

stage_time "MODEL_ZOO" \
  python "${ROOT}/src_py/train_model_zoo.py" \
  "${FILTERED_CSV}" --model "${MODEL_KEY}" --target "${TARGET_KEY}" \
  --features extended --outdir "${ZOO_OUTDIR}"

stage_time "SILENCE_SWEEP" \
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
  --rth-start 34200 \
  --rth-end 57600 \
  --model "${SWEEP_MODEL}" \
  --target "${SWEEP_TARGET}" \
  --features extended \
  --min-train-months 3 \
  --require-directional \
  --min-rows 500

TOTAL_END=$(date +%s)
TOTAL_DT=$((TOTAL_END - TOTAL_START))

echo -e "\n========== BASELINE PIPELINE DONE =========="
echo "End:   $(date '+%F %T')"
echo "Total runtime: ${TOTAL_DT}s"
echo "Bursts file: ${BURSTS_CSV} (From previous run)"
echo "Filtered file: ${FILTERED_CSV} (From previous run)"
echo "Zoo output: ${ZOO_OUTDIR}"
echo "Sweep output: ${SWEEP_OUTDIR}"
echo "==========================================="