#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -t 1-4
#$ -o logs/sgd_optuna_regression_2023_2024_$JOB_ID.$TASK_ID.out
#$ -l h_data=4G,h_rt=02:59:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

mkdir -p logs results/sgd_backtests_fixed_aum_2023_2024

TARGET_TO_RUN=${TARGET_TO_RUN:-reg_clop}
HAWKES_TAG=${HAWKES_TAG:-b1p0_i0p3}
START_DATE=${START_DATE:-2023-01-01}
END_DATE=${END_DATE:-2024-12-31}

TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
DATA_TEMPLATE=${DATA_TEMPLATE:-"results/bursts_%s_baseline_unfiltered.csv"}

resolve_optuna_json() {
  local ticker="$1"
  local preferred="results/optuna_regression/${ticker}/best_regression_params_${TARGET_TO_RUN}_${HAWKES_TAG}.json"
  
  if [ -f "${preferred}" ]; then
    echo "${preferred}"
    return 0
  fi

  echo "ERROR: Missing optuna regression params for ${ticker} ${TARGET_TO_RUN}. Tried:" >&2
  echo "  ${preferred}" >&2
  return 1
}

parse_optuna_params() {
  local json_file="$1"
  python3 - "$json_file" "$HAWKES_TAG" <<'PY'
import json
import sys

json_file = sys.argv[1]
default_hawkes_tag = sys.argv[2]
obj = json.load(open(json_file))
print(obj.get("hawkes_tag", default_hawkes_tag))
print(obj["vol_frac"])
print(obj["dir_thresh"])
print(obj["vol_ratio"])
print(obj.get("kappa", 0.0))
PY
}

run_one_ticker() {
  local ticker="$1"
  
  local data_path
  printf -v data_path "${DATA_TEMPLATE}" "${ticker}"
  
  local json_file
  json_file=$(resolve_optuna_json "${ticker}") || return 1

  mapfile -t P < <(parse_optuna_params "${json_file}")
  local use_hawkes_tag="${P[0]}"
  local vol_frac="${P[1]}"
  local dir_thresh="${P[2]}"
  local vol_ratio="${P[3]}"
  local kappa="${P[4]}"

  local out_prefix="results/sgd_backtests_fixed_aum_2023_2024/${ticker}_${TARGET_TO_RUN}_regression_${use_hawkes_tag}"
  echo "================================================"
  echo "Ticker: ${ticker}"
  echo "Target: ${TARGET_TO_RUN}"
  echo "Date window: ${START_DATE} -> ${END_DATE}"
  echo "Params file: ${json_file}"
  echo "Data: ${data_path}"
  echo "Hawkes tag: ${use_hawkes_tag}"
  echo "================================================"

  python3 src_py/online_sgd_backtest.py \
    --data "${data_path}" \
    --target "${TARGET_TO_RUN}" \
    --hawkes-tag "${use_hawkes_tag}" \
    --vol-frac "${vol_frac}" \
    --dir-thresh "${dir_thresh}" \
    --vol-ratio "${vol_ratio}" \
    --kappa "${kappa}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --ticker "${ticker}" \
    --execution-mode phase3_flow \
    --signal-mode direction \
    --position-mode fixed_aum \
    --fixed-aum 10000000 \
    --round-trip-bps-cost 1.0 \
    --daily-open-csv open_all.csv \
    --daily-close-csv close_all.csv \
    --debug-trades-out "${out_prefix}_debug_trades.csv" \
    --debug-signals-out "${out_prefix}_debug_signals.csv" \
    | tee "${out_prefix}.log"
}

if [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ]; then
  read -r -a ticker_arr <<< "${TICKERS}"
  idx=$((SGE_TASK_ID - 1))
  if [ "${idx}" -lt 0 ] || [ "${idx}" -ge "${#ticker_arr[@]}" ]; then
    exit 0
  fi
  run_one_ticker "${ticker_arr[$idx]}"
else
  for ticker in ${TICKERS}; do
    run_one_ticker "${ticker}"
  done
fi
