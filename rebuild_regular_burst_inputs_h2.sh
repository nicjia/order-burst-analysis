#!/bin/bash
# Rebuild non-archive burst inputs with current data_processor + permanence logic.
# Covers:
# 1) Baseline files used by model-zoo / SGD runners
# 2) shared_cache files used by sweep_frac + optuna_physical

set -Eeo pipefail

ROOT=${ROOT:-/u/scratch/n/nicjia/order-burst-analysis}
TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}
REBUILD_BASELINE=${REBUILD_BASELINE:-1}
REBUILD_SHARED_CACHE=${REBUILD_SHARED_CACHE:-1}

BASE_SILENCE=${BASE_SILENCE:-0.5}
BASE_VOL_FRAC=${BASE_VOL_FRAC:-0.0001}
BASE_DIR_THRESH=${BASE_DIR_THRESH:-0.8}
BASE_VOL_RATIO=${BASE_VOL_RATIO:-0.3}
BASE_TAU_MAX=${BASE_TAU_MAX:-10.0}
BASE_KAPPA_LONG=${BASE_KAPPA_LONG:-0.5}

SILENCE_VALUES=${SILENCE_VALUES:-"0.5 1.0 2.0"}
RTH_START=${RTH_START:-34200}
RTH_END=${RTH_END:-57600}
WORKERS=${WORKERS:-${NSLOTS:-4}}

cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"

silence_tag() {
  local s="$1"
  echo "${s}" | sed 's/\./p/g'
}

ensure_data_processor() {
  if [ ! -x "${ROOT}/data_processor" ]; then
    echo "INFO: data_processor missing; building with make"
    make clean && make
  fi
}

build_baseline_for_ticker() {
  local ticker="$1"
  local stock_dir="${ROOT}/data/${ticker}"
  local raw_csv="results/bursts_${ticker}_baseline.csv"
  local unfiltered_csv="results/bursts_${ticker}_baseline_unfiltered.csv"
  local filtered_csv="results/bursts_${ticker}_baseline_filtered.csv"
  local long_seed_csv="results/bursts_${ticker}_baseline_longseed.csv"
  local long_seed_out_csv="results/bursts_${ticker}_baseline_longseed_filtered.csv"

  if [ ! -d "${stock_dir}" ]; then
    echo "WARN: Missing ${stock_dir}; skipping baseline for ${ticker}"
    return 0
  fi

  if [ "${REBUILD_BASELINE}" = "1" ]; then
    rm -f "${raw_csv}" "${unfiltered_csv}" "${filtered_csv}" "${long_seed_csv}" "${long_seed_out_csv}"
  fi

  echo "[${ticker}] baseline data_processor -> ${raw_csv}"
  ./data_processor "${stock_dir}" "${raw_csv}" \
    -s "${BASE_SILENCE}" \
    -v "${BASE_VOL_FRAC}" \
    -d "${BASE_DIR_THRESH}" \
    -r "${BASE_VOL_RATIO}" \
    -k 0 \
    -t "${BASE_TAU_MAX}" \
    -j "${WORKERS}" \
    -b "${RTH_START}" -e "${RTH_END}"

  echo "[${ticker}] permanence kappa=0 -> ${unfiltered_csv}"
  python3 src_py/compute_permanence.py "${raw_csv}" open_all.csv close_all.csv --kappa 0 --ticker "${ticker}"
  if [ -f "${raw_csv%*.csv}_filtered.csv" ]; then
    mv -f "${raw_csv%*.csv}_filtered.csv" "${unfiltered_csv}"
  fi

  echo "[${ticker}] permanence kappa=${BASE_KAPPA_LONG} -> ${filtered_csv}"
  cp -f "${raw_csv}" "${long_seed_csv}"
  python3 src_py/compute_permanence.py "${long_seed_csv}" open_all.csv close_all.csv --kappa "${BASE_KAPPA_LONG}" --ticker "${ticker}"
  if [ -f "${long_seed_out_csv}" ]; then
    mv -f "${long_seed_out_csv}" "${filtered_csv}"
  fi
  rm -f "${long_seed_csv}"
}

build_shared_cache_for_ticker() {
  local ticker="$1"
  local stock_dir="${ROOT}/data/${ticker}"
  local precompute_dir="results/silence_sweep_${ticker}/logreg_l2/shared_cache"

  if [ ! -d "${stock_dir}" ]; then
    echo "WARN: Missing ${stock_dir}; skipping shared_cache for ${ticker}"
    return 0
  fi

  mkdir -p "${precompute_dir}"

  for s in ${SILENCE_VALUES}; do
    local s_tag
    s_tag=$(silence_tag "${s}")
    local raw_csv="${precompute_dir}/bursts_${ticker}_s${s_tag}.csv"
    local perm_csv="${precompute_dir}/bursts_${ticker}_s${s_tag}_filtered.csv"

    if [ "${REBUILD_SHARED_CACHE}" = "1" ]; then
      rm -f "${raw_csv}" "${perm_csv}"
    fi

    echo "[${ticker}] shared_cache s=${s} data_processor -> ${raw_csv}"
    ./data_processor "${stock_dir}" "${raw_csv}" \
      -s "${s}" \
      -v 0 \
      -d 0.5 \
      -r 1.0 \
      -k 0 \
      -t "${BASE_TAU_MAX}" \
      -j "${WORKERS}" \
      -b "${RTH_START}" -e "${RTH_END}"

    echo "[${ticker}] shared_cache s=${s} permanence kappa=0 -> ${perm_csv}"
    python3 src_py/compute_permanence.py "${raw_csv}" open_all.csv close_all.csv --kappa 0 --ticker "${ticker}"
  done
}

main() {
  ensure_data_processor

  echo "==============================================="
  echo "Rebuild regular burst inputs"
  echo "Root: ${ROOT}"
  echo "Tickers: ${TICKERS}"
  echo "Rebuild baseline: ${REBUILD_BASELINE}"
  echo "Rebuild shared_cache: ${REBUILD_SHARED_CACHE}"
  echo "Workers: ${WORKERS}"
  echo "==============================================="

  for ticker in ${TICKERS}; do
    echo ""
    echo "========== ${ticker} =========="
    if [ "${REBUILD_BASELINE}" = "1" ]; then
      build_baseline_for_ticker "${ticker}"
    fi
    if [ "${REBUILD_SHARED_CACHE}" = "1" ]; then
      build_shared_cache_for_ticker "${ticker}"
    fi
  done

  echo ""
  echo "DONE: rebuilt regular burst/permanence inputs."
}

main "$@"
