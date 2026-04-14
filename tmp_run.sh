#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -t 1-4
#$ -o logs/eval_10m_$JOB_ID.$TASK_ID.out
#$ -l h_data=12G,h_rt=4:00:00
#$ -pe shared 4

# eval_10m_horizon_h2.sh
#
# Evaluates the 10-minute intraday horizon (reg_10m).
# Execution Mode: burst_stream (Enter at burst end, exit 10 mins later).
# Runs a side-by-side comparison for all 4 tickers:
# 1) Method 1: Strict Optuna Geometric Filters
# 2) Method 2: Permissive Dust Filters (Feature Learning)

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
TARGET="reg_10m"
EXECUTION_MODE="burst_stream"

START_DATE=${START_DATE:-2023-01-01}
END_DATE=${END_DATE:-2024-12-31}
SILENCE_TAG=${SILENCE_TAG:-s0p5}
DATA_TEMPLATE="results/bursts_%s_baseline_unfiltered.csv"

# Fixed Sizing (1 Share)
POSITION_MODE="shares"
SHARES_PER_TRADE="1.0"

RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="results/eval_10m_${RUN_TAG}"
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

run_backtest() {
  local ticker="$1"
  local method_name="$2"
  local out_prefix="$3"
  local vol_frac="$4"
  local dir_thresh="$5"
  local vol_ratio="$6"
  local kappa="$7"
  local data_path="$8"

  python3 src_py/online_sgd_backtest.py \
    --data "${data_path}" \
    --target "${TARGET}" \
    --silence-tag "${SILENCE_TAG}" \
    --vol-frac "${vol_frac}" \
    --dir-thresh "${dir_thresh}" \
    --vol-ratio "${vol_ratio}" \
    --kappa "${kappa}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --ticker "${ticker}" \
    --execution-mode "${EXECUTION_MODE}" \
    --signal-mode "cost_aware" \
    --cost-buffer-mult 1.0 \
    --position-mode "${POSITION_MODE}" \
    --shares-per-trade "${SHARES_PER_TRADE}" \
    --daily-open-csv open_all.csv \
    --daily-close-csv close_all.csv \
    --debug-trades-out "${out_prefix}_debug_trades.csv" \
    --debug-signals-out "${out_prefix}_debug_signals.csv" \
    | tee "${out_prefix}.log"
}

score_trades_csv() {
  local trades_csv="$1"
  python3 - "${trades_csv}" <<'PY'
import csv
import sys
path = sys.argv[1]
trades = 0
net_sum = 0.0
try:
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            trades += 1
            try:
                net_sum += float(row.get('net_raw', '0') or 0)
            except Exception:
                pass
except FileNotFoundError:
    print('0,0.0')
    sys.exit(0)

mean_net = (net_sum / trades) if trades > 0 else 0.0
print(f"{trades},{mean_net}")
PY
}

main() {
  local summary_csv="${OUT_ROOT}/10m_comparison_summary.csv"

  if [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ]; then
    idx=$((SGE_TASK_ID - 1))
    TARGET_TICKER="${TICKERS_ARRAY[$idx]}"
  else
    TARGET_TICKER="${TICKERS_ARRAY[0]}" 
  fi

  if [ "${SGE_TASK_ID:-1}" -eq 1 ]; then
    echo "ticker,target,method,vol_frac,trades,mean_net_1share,out_prefix" > "${summary_csv}"
  fi

  echo "==============================================="
  echo "Running 10m Intraday Eval For: ${TARGET_TICKER}"
  echo "==============================================="

  local data_path
  printf -v data_path "${DATA_TEMPLATE}" "${TARGET_TICKER}"
  
  local json_file
  json_file=$(resolve_optuna_json "${TARGET_TICKER}")

  mapfile -t P < <(parse_optuna_params "${json_file}")
  local opt_vf="${P[0]}"
  local opt_d="${P[1]}"
  local opt_r="${P[2]}"
  local opt_k="${P[3]}"

  # === METHOD 1: STRICT OPTUNA FILTER ===
  local out_prefix_m1="${OUT_ROOT}/${TARGET_TICKER}_${TARGET}_optuna_strict_vf${opt_vf}"
  echo "[${TARGET_TICKER}] METHOD 1: Optuna Strict (vf=${opt_vf}, k=${opt_k})"
  run_backtest "${TARGET_TICKER}" "optuna_strict" "${out_prefix_m1}" "${opt_vf}" "${opt_d}" "${opt_r}" "${opt_k}" "${data_path}"
  
  local metrics_m1
  metrics_m1=$(score_trades_csv "${out_prefix_m1}_debug_trades.csv")
  IFS=',' read -r trades1 net1 <<< "${metrics_m1}"
  echo "${TARGET_TICKER},${TARGET},optuna_strict,${opt_vf},${trades1},${net1},${out_prefix_m1}" >> "${summary_csv}"

  # === METHOD 2: DUST FILTER (FEATURE LEARNING) ===
  local dust_vf="0.0001"
  local dust_d="0.5"
  local dust_r="1.0"
  local dust_k="0.0"
  
  local out_prefix_m2="${OUT_ROOT}/${TARGET_TICKER}_${TARGET}_dust_feature_vf${dust_vf}"
  echo "[${TARGET_TICKER}] METHOD 2: Dust Feature (vf=${dust_vf}, k=${dust_k})"
  run_backtest "${TARGET_TICKER}" "dust_feature" "${out_prefix_m2}" "${dust_vf}" "${dust_d}" "${dust_r}" "${dust_k}" "${data_path}"

  local metrics_m2
  metrics_m2=$(score_trades_csv "${out_prefix_m2}_debug_trades.csv")
  IFS=',' read -r trades2 net2 <<< "${metrics_m2}"
  echo "${TARGET_TICKER},${TARGET},dust_feature,${dust_vf},${trades2},${net2},${out_prefix_m2}" >> "${summary_csv}"
}

main "$@"