#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/selection_finish_$JOB_ID.out
#$ -l h_data=8G,h_rt=24:00:00
#$ -pe shared 8
#$ -N nvda_finish

set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
TICKER=${TICKER:-NVDA}
WORKERS=${NSLOTS:-1}

# --- FILES ---
ZOO_TEMP_CSV="${ROOT}/results/bursts_${TICKER}_zoo_baseline_input_filtered.csv"
ZOO_OUTDIR="${ROOT}/results/model_select_${TICKER}"
SWEEP_OUTDIR="${ROOT}/results/silence_sweep_${TICKER}"

# --- PHASE I: REMAINING ZOO PARAMETERS ---
KAPPA_ZOO=0.5
MODEL_KEY="logreg_en,sgd_hinge,ridge_cls,knn,svm_rbf,mlp_small,mlp_large,stacking,voting,naive_bayes,lgb_calibrated"
TARGET_KEY="all_cls"

# --- PHASE II: SWEEP PARAMETERS ---
RUN_SWEEP=1
SWEEP_TARGET="cls_1m" # The target the sweep will actually optimize for
SILENCE_VALUES="0.5,1.0,2.0"
MIN_VOL_VALUES="50,100,200,2000"
DIR_THRESH_VALUES="0.7,0.8,0.9"
VOL_RATIO_VALUES="0.1,0.3,0.5"
KAPPA_VALUES="0.2,0.5,1.0"

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

if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: 'module' command unavailable" >&2
  exit 1
fi

module load gcc/11.3.0 python/3.9.6

if [ ! -f "${ROOT}/.venv/bin/activate" ]; then
  echo "ERROR: missing venv activate script at ${ROOT}/.venv/bin/activate" >&2
  exit 1
fi
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

echo "========== SELECTION PIPELINE FINISH START =========="
echo "Job ID: ${JOB_ID:-N/A}"
echo "Models Remaining: ${MODEL_KEY}"
echo "Sweep Target: ${SWEEP_TARGET}"
echo "====================================================="

if [ ! -f "${ZOO_TEMP_CSV}" ]; then
  echo "ERROR: expected zoo input file not found: ${ZOO_TEMP_CSV}" >&2
  exit 1
fi

# --- STEP 2: THE MODEL ZOO (FINISH REMAINING) ---
stage_time "MODEL_ZOO_FINISH" \
  python -u "${ROOT}/src_py/train_model_zoo.py" \
  "${ZOO_TEMP_CSV}" --model "${MODEL_KEY}" --target "${TARGET_KEY}" \
  --features extended --outdir "${ZOO_OUTDIR}"

# --- DYNAMIC SWEEP MODEL SELECTION ---
echo -e "\n---- DYNAMIC MODEL SELECTION START ----"
DYNAMIC_MODEL=$(python3 -c "
import json, glob, sys
from collections import defaultdict

files = glob.glob('${ZOO_OUTDIR}/*.json')
target = '${SWEEP_TARGET}'
models = defaultdict(lambda: {'aucs': [], 'times': []})

for f in files:
    try:
        with open(f) as j:
            d = json.load(j)
        # Correctly indented Python code
        if d.get('task_type') == 'binary' and d.get('target') == target:
            m_key = d['model_key']
            auc = d.get('pooled', {}).get('AUC', 0)
            sec = d.get('elapsed_sec', 0)
            models[m_key]['aucs'].append(auc)
            models[m_key]['times'].append(sec)
    except Exception:
        pass

data = []
for m_key, metrics in models.items():
    if not metrics['aucs']: continue
    avg_auc = sum(metrics['aucs']) / len(metrics['aucs'])
    avg_time = sum(metrics['times']) / len(metrics['times'])
    data.append((m_key, avg_auc, avg_time, len(metrics['aucs'])))

# Sort by Average AUC descending
data.sort(key=lambda x: x[1], reverse=True)

# Take top 5 models for the selected sweep target
top_5 = data[:5]
print(f'Top 5 for target={target} by AUC:', file=sys.stderr)
for i, m in enumerate(top_5):
    print(f'  {i+1}. {m[0]}: Avg AUC = {m[1]:.4f} | Avg Time = {m[2]:.0f}s | Repeats = {m[3]}', file=sys.stderr)

# Re-sort the top 5 by Avg Time ascending and take fastest among high-AUC models
top_5.sort(key=lambda x: x[2])
print(top_5[0][0] if top_5 else 'logreg_l2')
")

echo "Winner selected for sweep: ${DYNAMIC_MODEL}"
echo "---- DYNAMIC MODEL SELECTION END ----"


# --- STEP 3: THE SWEEP ---
if [ "${RUN_SWEEP}" = "1" ]; then
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
    --model "${DYNAMIC_MODEL}" \
    --target "${SWEEP_TARGET}" \
    --features extended \
    --min-train-months 3 \
    --require-directional \
    --min-rows 500
fi

echo -e "\n========== SELECTION PIPELINE COMPLETELY DONE =========="
echo "End: $(date '+%F %T')"