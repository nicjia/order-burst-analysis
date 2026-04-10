#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -o logs/sgd_optuna_nvda_archive_$JOB_ID.out
#$ -l h_data=12G,h_rt=08:00:00
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

mkdir -p logs results/sgd_backtests_archive_nvda

TICKER=NVDA
TARGET_TO_RUN=${TARGET_TO_RUN:-reg_clop}
SILENCE_TAG=${SILENCE_TAG:-s2p0}
START_DATE=${START_DATE:-2019-01-01}
END_DATE=${END_DATE:-2022-12-31}

ARCHIVE_STOCK_FOLDER=${ARCHIVE_STOCK_FOLDER:-"${ROOT}/data/archive_2019_2022/NVDA"}
ARCHIVE_RAW_CSV=${ARCHIVE_RAW_CSV:-"${ROOT}/results/bursts_NVDA_archive_2019_2022_raw.csv"}
ARCHIVE_PERM_CSV=${ARCHIVE_PERM_CSV:-"${ROOT}/results/bursts_NVDA_archive_2019_2022_permanence.csv"}

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
  if [ -s "${ARCHIVE_PERM_CSV}" ]; then
    echo "Using existing archive permanence CSV: ${ARCHIVE_PERM_CSV}"
    return 0
  fi

  if [ ! -x "${ROOT}/data_processor" ]; then
    echo "ERROR: data_processor is missing or not executable at ${ROOT}/data_processor" >&2
    return 1
  fi

  if [ ! -d "${ARCHIVE_STOCK_FOLDER}" ]; then
    echo "ERROR: Missing archive stock folder ${ARCHIVE_STOCK_FOLDER}" >&2
    return 1
  fi

  echo "Building archive raw bursts from ${ARCHIVE_STOCK_FOLDER}"
  ./data_processor "${ARCHIVE_STOCK_FOLDER}" "${ARCHIVE_RAW_CSV}" \
    -s "${BASE_SILENCE}" \
    -v "${BASE_VOL_FRAC}" \
    -d "${BASE_DIR_THRESH}" \
    -r "${BASE_VOL_RATIO}" \
    -k 0 \
    -t "${BASE_TAU_MAX}" \
    -j "${NSLOTS:-4}" \
    -b 34200 -e 57600

  if [ ! -s "${ARCHIVE_RAW_CSV}" ]; then
    echo "ERROR: data_processor did not produce ${ARCHIVE_RAW_CSV}" >&2
    return 1
  fi

  echo "Computing permanence columns for archive dataset"
  python3 src_py/compute_permanence.py \
    "${ARCHIVE_RAW_CSV}" \
    "${ROOT}/open_all.csv" \
    "${ROOT}/close_all.csv" \
    --kappa 0

  local produced="${ARCHIVE_RAW_CSV%.csv}_filtered.csv"
  if [ ! -s "${produced}" ]; then
    echo "ERROR: compute_permanence did not produce ${produced}" >&2
    return 1
  fi

  mv "${produced}" "${ARCHIVE_PERM_CSV}"
  echo "Archive permanence dataset ready: ${ARCHIVE_PERM_CSV}"
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
