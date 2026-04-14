#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -t 1-4
#$ -o logs/tmp_dual_volfrac_$JOB_ID.$TASK_ID.out
#$ -l h_data=12G,h_rt=12:00:00
#$ -pe shared 4

# tmp_dual_volfrac_overnight_h2.sh
#
# Runs BOTH approaches overnight:
# 1) Financial objective scan over vol_frac (score = mean_net_per_trade * log(trade_count + 1))
# 2) Feature-not-filter run with dust-level filtering

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

# ----- User-tunable knobs -----
TICKERS_ARRAY=("NVDA" "TSLA" "JPM" "MS")
TARGET=${TARGET:-reg_clop}
START_DATE=${START_DATE:-2023-01-01}
END_DATE=${END_DATE:-2024-12-31}
SILENCE_TAG=${SILENCE_TAG:-s0p5}
DATA_TEMPLATE=${DATA_TEMPLATE:-"results/bursts_%s_baseline_unfiltered.csv"}

# Method 1: financial scan over vol_frac while keeping other params per ticker from optuna json.
VOL_FRAC_GRID=${VOL_FRAC_GRID:-"0.0001 0.0003 0.0007 0.0015 0.0030"}
METHOD1_COST_BUFFER=${METHOD1_COST_BUFFER:-1.0}
METHOD1_SIGNAL_MODE=${METHOD1_SIGNAL_MODE:-cost_aware}

# Method 2: low dust filter + let model learn volume via features.
DUST_VOL_FRAC=${DUST_VOL_FRAC:-0.0001}
DUST_KAPPA=${DUST_KAPPA:-0.0}
DUST_DIR_THRESH=${DUST_DIR_THRESH:-0.5}
DUST_VOL_RATIO=${DUST_VOL_RATIO:-1.0}
METHOD2_COST_BUFFER=${METHOD2_COST_BUFFER:-1.0}
METHOD2_SIGNAL_MODE=${METHOD2_SIGNAL_MODE:-cost_aware}

# Shared execution settings (FIXED for Overnight Targets)
EXECUTION_MODE=${EXECUTION_MODE:-phase3_flow}
PHASE3_THRESH=${PHASE3_THRESH:-5.0} # <-- Required for Phase 3!
POSITION_MODE=${POSITION_MODE:-fraction}
POSITION_SIZE_MULT=${POSITION_SIZE_MULT:-1.0}
SHARES_PER_TRADE=${SHARES_PER_TRADE:-1.0}

RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="results/tmp_dual_volfrac_${RUN_TAG}"
mkdir -p "${OUT_ROOT}"

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
      echo "ERROR: Unsupported TARGET='${target}'" >&2
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

  echo "ERROR: Missing optuna params for ${ticker} ${cls_key}" >&2
  return 1
}

parse_optuna_params() {
  local json_file="$1"
  python3 - "$json_file" "$SILENCE_TAG" <<'PY'
import json
import sys

obj = json.load(open(sys.argv[1]))
default_s = sys.argv[2]
bp = obj.get("best_params", {})
print(bp.get("silence_tag", default_s))
print(bp["vol_frac"])
print(bp["dir_thresh"])
print(bp["vol_ratio"])
print(bp.get("kappa", 0.0))
PY
}

validate_dataset() {
  local path="$1"
  local expected_ticker="$2"

  if [ ! -s "${path}" ]; then
    echo "ERROR: Missing dataset ${path}" >&2
    return 1
  fi

  python3 - "${path}" "${expected_ticker}" <<'PY'
import csv
import sys

path = sys.argv[1]
expected = sys.argv[2]
rows = 0
bad_ticker = 0
with open(path, newline='') as f:
    r = csv.DictReader(f)
    if not r.fieldnames or 'Ticker' not in r.fieldnames:
        print('ERROR: Missing Ticker column')
        sys.exit(2)
    for row in r:
        rows += 1
        if row.get('Ticker', '') != expected:
            bad_ticker += 1
if rows == 0:
    print('ERROR: CSV has zero rows')
    sys.exit(3)
if bad_ticker > 0:
    sys.exit(4)
PY
}

run_backtest() {
  local ticker="$1"
  local data_path="$2"
  local out_prefix="$3"
  local vol_frac="$4"
  local dir_thresh="$5"
  local vol_ratio="$6"
  local kappa="$7"
  local signal_mode="$8"
  local cost_buffer="$9"

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
    --phase3-thresh "${PHASE3_THRESH}" \
    --signal-mode "${signal_mode}" \
    --cost-buffer-mult "${cost_buffer}" \
    --position-mode "${POSITION_MODE}" \
    --position-size-mult "${POSITION_SIZE_MULT}" \
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
import math
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
    print('0,0,0')
    sys.exit(0)

mean_net = (net_sum / trades) if trades > 0 else 0.0
score = mean_net * math.log(trades + 1.0)
print(f"{trades},{mean_net},{score}")
PY
}

main() {
  local cls_key
  # Force cls_10m so Optuna's base parameters don't starve the model with crazy Kappas
  cls_key="cls_10m"

  local summary_csv="${OUT_ROOT}/financial_scan_summary.csv"

  if [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ]; then
    idx=$((SGE_TASK_ID - 1))
    TARGET_TICKER="${TICKERS_ARRAY[$idx]}"
  else
    TARGET_TICKER="${TICKERS_ARRAY[0]}" 
  fi

  # Only print the header if we are Task 1 so we don't jumble the CSV
  if [ "${SGE_TASK_ID:-1}" -eq 1 ]; then
    echo "ticker,method,vol_frac,dir_thresh,vol_ratio,kappa,trades,mean_net_raw,score_mean_logtrades,out_prefix" > "${summary_csv}"
  fi

  echo "==============================================="
  echo "Running dual overnight test for: ${TARGET_TICKER}"
  echo "Target: ${TARGET}"
  echo "==============================================="

  local data_path
  printf -v data_path "${DATA_TEMPLATE}" "${TARGET_TICKER}"
  validate_dataset "${data_path}" "${TARGET_TICKER}"

  local json_file
  json_file=$(resolve_optuna_json "${TARGET_TICKER}" "${cls_key}")

  mapfile -t P < <(parse_optuna_params "${json_file}")
  local opt_silence="${P[0]}"
  local opt_d="${P[2]}"
  local opt_r="${P[3]}"
  local opt_k="${P[4]}"

  # Method 1: financial objective scan over vol_frac.
  for vf in ${VOL_FRAC_GRID}; do
    local out_prefix="${OUT_ROOT}/${TARGET_TICKER}_method1_vf${vf}_d${opt_d}_r${opt_r}_k${opt_k}"
    echo "[${TARGET_TICKER}] METHOD1 vf=${vf}"
    run_backtest "${TARGET_TICKER}" "${data_path}" "${out_prefix}" "${vf}" "${opt_d}" "${opt_r}" "${opt_k}" "${METHOD1_SIGNAL_MODE}" "${METHOD1_COST_BUFFER}"

    local metrics
    metrics=$(score_trades_csv "${out_prefix}_debug_trades.csv")
    IFS=',' read -r trades mean_net score <<< "${metrics}"
    echo "${TARGET_TICKER},method1,${vf},${opt_d},${opt_r},${opt_k},${trades},${mean_net},${score},${out_prefix}" >> "${summary_csv}"
  done

  # Method 2: dust filter + feature learning.
  local out_prefix2="${OUT_ROOT}/${TARGET_TICKER}_method2_dust_vf${DUST_VOL_FRAC}_d${DUST_DIR_THRESH}_r${DUST_VOL_RATIO}_k${DUST_KAPPA}"
  echo "[${TARGET_TICKER}] METHOD2 dust vf=${DUST_VOL_FRAC}"
  run_backtest "${TARGET_TICKER}" "${data_path}" "${out_prefix2}" "${DUST_VOL_FRAC}" "${DUST_DIR_THRESH}" "${DUST_VOL_RATIO}" "${DUST_KAPPA}" "${METHOD2_SIGNAL_MODE}" "${METHOD2_COST_BUFFER}"

  local metrics2
  metrics2=$(score_trades_csv "${out_prefix2}_debug_trades.csv")
  IFS=',' read -r trades2 mean_net2 score2 <<< "${metrics2}"
  echo "${TARGET_TICKER},method2,${DUST_VOL_FRAC},${DUST_DIR_THRESH},${DUST_VOL_RATIO},${DUST_KAPPA},${trades2},${mean_net2},${score2},${out_prefix2}" >> "${summary_csv}"
}

main "$@"