#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -o logs/sgd_optuna_nvda_archive_$JOB_ID.out
#$ -l h_data=12G,h_rt=08:00:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"

set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

mkdir -p logs results/sgd_backtests_archive_nvda

TICKER=NVDA
TARGET_TO_RUN=${TARGET_TO_RUN:-reg_clop}
SILENCE_TAG=${SILENCE_TAG:-s2p0}
START_DATE=${START_DATE:-2019-01-01}
END_DATE=${END_DATE:-2022-12-31}

# POINT THIS EXACTLY TO YOUR PRE-BUILT RESULTS CSV
ARCHIVE_PERM_CSV=${ARCHIVE_PERM_CSV:-"${ROOT}/results/nvda_archive_2019_2022_raw_s2p0_filtered.csv"}

BASE_SILENCE=${BASE_SILENCE:-0.5}
BASE_VOL_FRAC=${BASE_VOL_FRAC:-0.0001}
BASE_DIR_THRESH=${BASE_DIR_THRESH:-0.8}
BASE_VOL_RATIO=${BASE_VOL_RATIO:-0.3}
BASE_TAU_MAX=${BASE_TAU_MAX:-10.0}

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
  local cls_key="$1"
  local preferred="results/optuna_physical/${TICKER}/best_physical_params_${cls_key}_${SILENCE_TAG}.json"
  local fallback="results/optuna_physical/${TICKER}/best_physical_params_${cls_key}.json"

  if [ -f "${preferred}" ]; then
    echo "${preferred}"
    return 0
  fi
  if [ -f "${fallback}" ]; then
    echo "${fallback}"
    return 0
  fi

  echo "ERROR: Missing optuna params for ${TICKER} ${cls_key}. Tried:" >&2
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

build_archive_dataset_if_missing() {
  if [ ! -s "${ARCHIVE_PERM_CSV}" ]; then
    echo "ERROR: Cannot find ${ARCHIVE_PERM_CSV}. Please ensure it is in the results/ folder." >&2
    return 1
  fi

  echo "Validating archive permanence CSV: ${ARCHIVE_PERM_CSV}"
  if ! python3 - "${ARCHIVE_PERM_CSV}" "${TICKER}" <<'PY'
import csv
import math
import sys

path = sys.argv[1]
expected_ticker = sys.argv[2]

rows = 0
bad_ticker = 0
finite_clop = 0
finite_clcl = 0

with open(path, newline='') as f:
    r = csv.DictReader(f)
    required = {'Ticker', 'Perm_CLOP', 'Perm_CLCL'}
    missing = [c for c in required if c not in (r.fieldnames or [])]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(2)

    for row in r:
        rows += 1
        if row.get('Ticker', '') != expected_ticker:
            bad_ticker += 1

        for key, acc in (('Perm_CLOP', 'clop'), ('Perm_CLCL', 'clcl')):
            v = (row.get(key) or '').strip()
            if not v:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                if acc == 'clop':
                    finite_clop += 1
                else:
                    finite_clcl += 1

if rows == 0:
    print("ERROR: Archive CSV has zero rows.")
    sys.exit(3)

print(f"rows={rows} bad_ticker_rows={bad_ticker} finite_clop={finite_clop} finite_clcl={finite_clcl}")

if bad_ticker > 0:
    print("ERROR: Ticker column does not match expected ticker; this usually means data_processor extracted folder name instead of stock symbol.")
    sys.exit(4)

if finite_clop == 0 and finite_clcl == 0:
    print("ERROR: Overnight permanence targets are all non-finite/empty.")
    sys.exit(5)
PY
  then
    echo "ERROR: Archive permanence CSV failed validation." >&2
    echo "Recompute it with forced ticker to avoid archive-folder ticker bug:" >&2
    echo "  python3 src_py/compute_permanence.py ${ARCHIVE_PERM_CSV%_filtered.csv}.csv open_all.csv close_all.csv --kappa 0 --ticker ${TICKER}" >&2
    return 1
  fi

  echo "Archive CSV validation passed."
  return 0
}

main() {
  local cls_key
  cls_key=$(cls_key_for_target "${TARGET_TO_RUN}")

  build_archive_dataset_if_missing

  local json_file
  json_file=$(resolve_optuna_json "${cls_key}")

  mapfile -t P < <(parse_optuna_params "${json_file}")
  local use_silence="${P[0]}"
  local vol_frac="${P[1]}"
  local dir_thresh="${P[2]}"
  local vol_ratio="${P[3]}"
  local kappa="${P[4]}"

  local out_prefix="results/sgd_backtests_archive_nvda/${TICKER}_${TARGET_TO_RUN}_${use_silence}_vf${vol_frac}_d${dir_thresh}_r${vol_ratio}_k${kappa}"

  echo "================================================"
  echo "Ticker: ${TICKER}"
  echo "Target: ${TARGET_TO_RUN}"
  echo "Archive date window: ${START_DATE} -> ${END_DATE}"
  echo "Params file: ${json_file}"
  echo "Data: ${ARCHIVE_PERM_CSV}"
  echo "================================================"

  python3 src_py/online_sgd_backtest.py \
    --data "${ARCHIVE_PERM_CSV}" \
    --target "${TARGET_TO_RUN}" \
    --silence-tag "${use_silence}" \
    --vol-frac "${vol_frac}" \
    --dir-thresh "${dir_thresh}" \
    --vol-ratio "${vol_ratio}" \
    --kappa "${kappa}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --ticker "${TICKER}" \
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

main