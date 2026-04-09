#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -o logs/overnight_bt_$JOB_ID_$TASK_ID.out
#$ -l h_data=10G,h_rt=12:00:00
#$ -pe shared 4

# Do not enable errexit before module init scripts; some Hoffman init snippets
# intentionally return non-zero during probing.
set -o pipefail

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

echo "========================================"
echo "Overnight Backtest Job"
echo "  JOB_ID:      ${JOB_ID:-N/A}"
echo "  SGE_TASK_ID: ${SGE_TASK_ID:-N/A}"
echo "  Hostname:    $(hostname)"
echo "  Date:        $(date)"
echo "  PWD:         $(pwd)"
echo "========================================"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6

source "${ROOT}/.venv/bin/activate"

# Enable strict mode only after environment setup.
set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

echo "Python: $(which python3)"
python3 --version

mkdir -p logs results/overnight_backtests

TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
DEFAULT_DATA_TEMPLATE=${DEFAULT_DATA_TEMPLATE:-"results/bursts_%s_baseline_unfiltered.csv"}
COST_BUFFER_MULT=${COST_BUFFER_MULT:-1.0}
EXECUTION_MODE=${EXECUTION_MODE:-burst_stream}
SIGNAL_MODE=${SIGNAL_MODE:-cost_aware}
POSITION_MODE=${POSITION_MODE:-fraction}
POSITION_SIZE_MULT=${POSITION_SIZE_MULT:-1.0}
SHARES_PER_TRADE=${SHARES_PER_TRADE:-1.0}
SPREAD_COL=${SPREAD_COL:-Spread}
SPREAD_MULTIPLIER=${SPREAD_MULTIPLIER:-0.5}
SPREAD_EXIT_MULTIPLIER=${SPREAD_EXIT_MULTIPLIER:-0.5}
DAILY_OPEN_CSV=${DAILY_OPEN_CSV:-open_all.csv}
DAILY_CLOSE_CSV=${DAILY_CLOSE_CSV:-close_all.csv}
PHASE3_THRESH=${PHASE3_THRESH:-0.0}
PHASE3_MIN_LAG_MINUTES=${PHASE3_MIN_LAG_MINUTES:-10.0}
PHASE3_FLOW_COL=${PHASE3_FLOW_COL:-signed_volume}
TICKER_OVERRIDE=${TICKER_OVERRIDE:-}
TARGETS=${TARGETS:-"reg_clop reg_clcl"}

# Optional date window overrides for online_sgd_backtest.py defaults.
# Keep empty to use script defaults.
START_DATE=${START_DATE:-}
END_DATE=${END_DATE:-}

if [ -n "${SGE_TASK_ID:-}" ]; then
  read -r -a TICKER_ARR <<< "${TICKERS}"
  idx=$((SGE_TASK_ID - 1))
  if [ "${idx}" -lt 0 ] || [ "${idx}" -ge "${#TICKER_ARR[@]}" ]; then
    echo "INFO: Task ${SGE_TASK_ID} out of range for tickers='${TICKERS}', exiting."
    exit 0
  fi
  TICKER="${TICKER_ARR[$idx]}"
else
  if [ -z "${TICKER:-}" ]; then
    echo "ERROR: Set TICKER when not running as array task." >&2
    exit 1
  fi
fi

echo "Resolved ticker: ${TICKER}"

printf -v DATA_PATH "${DEFAULT_DATA_TEMPLATE}" "${TICKER}"
if [ ! -s "${DATA_PATH}" ]; then
  echo "ERROR: Missing input data '${DATA_PATH}'." >&2
  echo "Run prep first (Hoffman): qsub -t 1-4 -v TICKERS=\"NVDA TSLA JPM MS\",FORCE_REBUILD_BASELINE=1 prep_data.sh" >&2
  exit 1
fi

echo "Resolved data path: ${DATA_PATH}"

parse_optuna_params() {
  local json_file="$1"
  python3 - "$json_file" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1]))["best_params"]
print(p["silence_tag"])
print(p["vol_frac"])
print(p["dir_thresh"])
print(p["vol_ratio"])
print(p.get("kappa", 0.0))
PY
}

cls_key_for_target() {
  local target="$1"
  case "${target}" in
    reg_clop) echo "cls_clop" ;;
    reg_clcl) echo "cls_clcl" ;;
    reg_close) echo "cls_close" ;;
    reg_1m) echo "cls_1m" ;;
    reg_3m) echo "cls_3m" ;;
    reg_5m) echo "cls_5m" ;;
    reg_10m) echo "cls_10m" ;;
    *) return 1 ;;
  esac
}

run_one_target() {
  local target="$1"
  local cls_key
  cls_key=$(cls_key_for_target "${target}") || {
    echo "ERROR: Unsupported target '${target}'" >&2
    return 1
  }
  local json_file="results/optuna_physical/${TICKER}/best_physical_params_${cls_key}.json"

  if [ ! -f "${json_file}" ]; then
    echo "ERROR: Missing optuna params '${json_file}' for ${TICKER} ${target}" >&2
    return 1
  fi

  mapfile -t P < <(parse_optuna_params "${json_file}")
  local SILENCE_TAG="${P[0]}"
  local VOL_FRAC="${P[1]}"
  local DIR_THRESH="${P[2]}"
  local VOL_RATIO="${P[3]}"
  local KAPPA="${P[4]}"

  local out_prefix="results/overnight_backtests/${TICKER}_${target}_${SILENCE_TAG}_vf${VOL_FRAC}_d${DIR_THRESH}_r${VOL_RATIO}_k${KAPPA}"

  echo "================================================"
  echo "Ticker: ${TICKER}"
  echo "Target: ${target}"
  echo "Params from: ${json_file}"
  echo "  silence=${SILENCE_TAG} vf=${VOL_FRAC} d=${DIR_THRESH} r=${VOL_RATIO} k=${KAPPA}"
  echo "Data: ${DATA_PATH}"
  echo "Output prefix: ${out_prefix}"
  echo "================================================"

  cmd=(
    python3 src_py/online_sgd_backtest.py
    --data "${DATA_PATH}"
    --target "${target}"
    --silence-tag "${SILENCE_TAG}"
    --vol-frac "${VOL_FRAC}"
    --dir-thresh "${DIR_THRESH}"
    --vol-ratio "${VOL_RATIO}"
    --kappa "${KAPPA}"
    --execution-mode "${EXECUTION_MODE}"
    --signal-mode "${SIGNAL_MODE}"
    --cost-buffer-mult "${COST_BUFFER_MULT}"
    --position-mode "${POSITION_MODE}"
    --position-size-mult "${POSITION_SIZE_MULT}"
    --shares-per-trade "${SHARES_PER_TRADE}"
    --spread-col "${SPREAD_COL}"
    --spread-multiplier "${SPREAD_MULTIPLIER}"
    --spread-exit-multiplier "${SPREAD_EXIT_MULTIPLIER}"
    --daily-open-csv "${DAILY_OPEN_CSV}"
    --daily-close-csv "${DAILY_CLOSE_CSV}"
    --phase3-thresh "${PHASE3_THRESH}"
    --phase3-min-lag-minutes "${PHASE3_MIN_LAG_MINUTES}"
    --phase3-flow-col "${PHASE3_FLOW_COL}"
    --debug-trades-out "${out_prefix}_debug_trades.csv"
    --debug-signals-out "${out_prefix}_debug_signals.csv"
  )

  if [ -n "${TICKER_OVERRIDE}" ]; then
    cmd+=(--ticker "${TICKER_OVERRIDE}")
  else
    cmd+=(--ticker "${TICKER}")
  fi

  if [ -n "${START_DATE}" ]; then
    cmd+=(--start-date "${START_DATE}")
  fi
  if [ -n "${END_DATE}" ]; then
    cmd+=(--end-date "${END_DATE}")
  fi

  "${cmd[@]}" | tee "${out_prefix}.log"
}

for target in ${TARGETS}; do
  run_one_target "${target}"
done

echo "Overnight backtests complete for ${TICKER}"