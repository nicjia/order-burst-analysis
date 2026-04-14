#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -t 1-4
#$ -o logs/split_logic_$JOB_ID.$TASK_ID.out
#$ -l h_data=12G,h_rt=12:00:00
#$ -pe shared 4

ROOT=${ROOT:-/u/scratch/n/nicjia/order-burst-analysis}
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"
set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

mkdir -p logs results

TICKERS_ARRAY=("NVDA" "TSLA" "JPM" "MS")
TARGET="reg_clop"
START_DATE="2023-01-01"
END_DATE="2024-12-31"
SILENCE_TAG="s0p5"

RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="results/split_logic_${RUN_TAG}"
mkdir -p "${OUT_ROOT}"

resolve_optuna_json() {
  local ticker="$1"
  local preferred="results/optuna_physical/${ticker}/best_physical_params_cls_10m_${SILENCE_TAG}.json"
  local fallback="results/optuna_physical/${ticker}/best_physical_params_cls_10m.json"

  if [ -f "${preferred}" ]; then echo "${preferred}"; return 0; fi
  if [ -f "${fallback}" ]; then echo "${fallback}"; return 0; fi
  echo "ERROR: Missing optuna params for ${ticker}" >&2
  return 1
}

parse_optuna_params() {
  local json_file="$1"
  python3 - "$json_file" "$SILENCE_TAG" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1]))
print(obj.get("best_params", {}).get("vol_frac", 0.0001))
print(obj.get("best_params", {}).get("dir_thresh", 0.5))
print(obj.get("best_params", {}).get("vol_ratio", 1.0))
print(obj.get("best_params", {}).get("kappa", 0.0))
PY
}

main() {
  if [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ]; then
    idx=$((SGE_TASK_ID - 1))
    TARGET_TICKER="${TICKERS_ARRAY[$idx]}"
  else
    TARGET_TICKER="${TICKERS_ARRAY[0]}" 
  fi

  echo "==============================================="
  echo "Running Split Logic For: ${TARGET_TICKER}"
  echo "==============================================="

  local data_path="results/bursts_${TARGET_TICKER}_baseline_unfiltered.csv"
  local json_file
  json_file=$(resolve_optuna_json "${TARGET_TICKER}")

  mapfile -t P < <(parse_optuna_params "${json_file}")
  local opt_vf="${P[0]}"
  local opt_d="${P[1]}"
  local opt_r="${P[2]}"
  local opt_k="${P[3]}"

  local FINAL_VF FINAL_D FINAL_R FINAL_K METHOD_NAME

  # THE SPLIT LOGIC
  if [[ "${TARGET_TICKER}" == "MS" || "${TARGET_TICKER}" == "JPM" ]]; then
    # Institutional: Use Strict Optuna Filters
    FINAL_VF=$opt_vf
    FINAL_D=$opt_d
    FINAL_R=$opt_r
    FINAL_K=$opt_k
    METHOD_NAME="optuna_strict"
  else
    # Retail: Use permissive dust filters (feature learning)
    FINAL_VF="0.0001"
    FINAL_D="0.5"
    FINAL_R="1.0"
    FINAL_K="0.0"
    METHOD_NAME="dust_feature"
  fi

  local out_prefix="${OUT_ROOT}/${TARGET_TICKER}_${METHOD_NAME}_vf${FINAL_VF}"
  echo "Applying ${METHOD_NAME} (vf=${FINAL_VF}, k=${FINAL_K})"

  python3 src_py/online_sgd_backtest.py \
    --data "${data_path}" \
    --target "${TARGET}" \
    --silence-tag "${SILENCE_TAG}" \
    --vol-frac "${FINAL_VF}" \
    --dir-thresh "${FINAL_D}" \
    --vol-ratio "${FINAL_R}" \
    --kappa "${FINAL_K}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --ticker "${TARGET_TICKER}" \
    --execution-mode "phase3_flow" \
    --phase3-percentile 90.0 \
    --signal-mode "cost_aware" \
    --cost-buffer-mult 1.0 \
    --position-mode "shares" \
    --shares-per-trade 1.0 \
    --daily-open-csv open_all.csv \
    --daily-close-csv close_all.csv \
    --debug-trades-out "${out_prefix}_debug_trades.csv" \
    --debug-signals-out "${out_prefix}_debug_signals.csv" \
    | tee "${out_prefix}.log"
}

main "$@"