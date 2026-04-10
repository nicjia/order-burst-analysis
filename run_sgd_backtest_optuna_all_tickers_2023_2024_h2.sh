#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -o logs/sgd_optuna_2023_2024_$JOB_ID_$TASK_ID.out
#$ -l h_data=10G,h_rt=08:00:00
#$ -pe shared 4

set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"

mkdir -p logs results/sgd_backtests_optuna_2023_2024

# Edit these defaults directly for quick reruns.
TARGET_TO_RUN=${TARGET_TO_RUN:-reg_clop}
SILENCE_TAG=${SILENCE_TAG:-s2p0}
START_DATE=${START_DATE:-2023-01-01}
END_DATE=${END_DATE:-2024-12-31}

TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
DATA_TEMPLATE=${DATA_TEMPLATE:-"results/bursts_%s_baseline_unfiltered.csv"}

cls_key_for_target() {
  local target="$1"
  case "${target}" in
    reg_1m) echo "cls_1m" ;;
    reg_3m) echo "cls_3m" ;;
    reg_5m) echo "cls_5m" ;;
    reg_10m) echo "cls_10m" ;;
    reg_close) echo "cls_close" ;;
    reg_clop) echo "cls_clop" ;;
    reg_clcl) echo "cls_clcl" ;;
    *)
      echo "ERROR: Unsupported TARGET_TO_RUN='${target}'" >&2
      return 1
      ;;
  esac
}

resolve_optuna_json() {
  local ticker="$1"
  local cls_key="$2"
  local preferred="results/optuna_physical/${ticker}/best_physical_params_${cls_key}_${SILENCE_TAG}.json"
  local fallback="results/optuna_physical/${ticker}/best_physical_params_${cls_key}.json"

  if [ -f "${preferred}" ]; then
    echo "${preferred}"
    return 0
  fi
  if [ -f "${fallback}" ]; then
    echo "${fallback}"
    return 0
  fi

  echo "ERROR: Missing optuna params for ${ticker} ${cls_key}. Tried:" >&2
  echo "  ${preferred}" >&2
  echo "  ${fallback}" >&2
  return 1
}

parse_optuna_params() {
  local json_file="$1"
  python3 - "$json_file" "$SILENCE_TAG" <<'PY'
import json
import sys

json_file = sys.argv[1]
default_silence = sys.argv[2]
obj = json.load(open(json_file))
bp = obj.get("best_params", {})
print(bp.get("silence_tag", default_silence))
print(bp["vol_frac"])
print(bp["dir_thresh"])
print(bp["vol_ratio"])
print(bp.get("kappa", 0.0))
PY
}

run_one_ticker() {
  local ticker="$1"
  local cls_key
  cls_key=$(cls_key_for_target "${TARGET_TO_RUN}")

  local data_path
  printf -v data_path "${DATA_TEMPLATE}" "${ticker}"
  if [ ! -s "${data_path}" ]; then
    echo "ERROR: Missing input dataset for ${ticker}: ${data_path}" >&2
    return 1
  fi

  local json_file
  json_file=$(resolve_optuna_json "${ticker}" "${cls_key}")

  mapfile -t P < <(parse_optuna_params "${json_file}")
  local use_silence="${P[0]}"
  local vol_frac="${P[1]}"
  local dir_thresh="${P[2]}"
  local vol_ratio="${P[3]}"
  local kappa="${P[4]}"

  local out_prefix="results/sgd_backtests_optuna_2023_2024/${ticker}_${TARGET_TO_RUN}_${use_silence}_vf${vol_frac}_d${dir_thresh}_r${vol_ratio}_k${kappa}"

  echo "================================================"
  echo "Ticker: ${ticker}"
  echo "Target: ${TARGET_TO_RUN}"
  echo "Date window: ${START_DATE} -> ${END_DATE}"
  echo "Params file: ${json_file}"
  echo "Data: ${data_path}"
  echo "================================================"

  python3 src_py/online_sgd_backtest.py \
    --data "${data_path}" \
    --target "${TARGET_TO_RUN}" \
    --silence-tag "${use_silence}" \
    --vol-frac "${vol_frac}" \
    --dir-thresh "${dir_thresh}" \
    --vol-ratio "${vol_ratio}" \
    --kappa "${kappa}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --ticker "${ticker}" \
    --execution-mode burst_stream \
    --signal-mode cost_aware \
    --position-mode fraction \
    --position-size-mult 1.0 \
    --shares-per-trade 1.0 \
    --spread-col Spread \
    --spread-multiplier 0.5 \
    --spread-exit-multiplier 0.5 \
    --daily-open-csv open_all.csv \
    --daily-close-csv close_all.csv \
    --debug-trades-out "${out_prefix}_debug_trades.csv" \
    --debug-signals-out "${out_prefix}_debug_signals.csv" \
    | tee "${out_prefix}.log"
}

if [ -n "${SGE_TASK_ID:-}" ]; then
  read -r -a ticker_arr <<< "${TICKERS}"
  idx=$((SGE_TASK_ID - 1))
  if [ "${idx}" -lt 0 ] || [ "${idx}" -ge "${#ticker_arr[@]}" ]; then
    echo "INFO: Task ${SGE_TASK_ID} out of range for tickers='${TICKERS}', exiting."
    exit 0
  fi
  run_one_ticker "${ticker_arr[$idx]}"
else
  for ticker in ${TICKERS}; do
    run_one_ticker "${ticker}"
  done
fi

echo "All requested SGD backtests complete."
