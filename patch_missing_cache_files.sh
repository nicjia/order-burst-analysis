#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/patch_missing_cache_$JOB_ID.out
#$ -l h_data=4G,h_rt=04:00:00
#$ -pe shared 4
#
# patch_missing_cache_files.sh
#
# Targeted repair script for missing shared_cache files.
#
# What it does:
#   NVDA b1p0_i0p5 : raw exists → only runs compute_permanence.py
#   NVDA b1p0_i0p8 : both missing → runs data_processor + compute_permanence.py
#   TSLA b1p0_i0p8 : both missing → runs data_processor + compute_permanence.py
#   Then flattens results/TICKER_params/shared_cache/ → results/TICKER_params/
#

ROOT=${ROOT:-/u/scratch/n/nicjia/order-burst-analysis}
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
set -Eeo pipefail
trap 'echo "ERROR at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# ── Hawkes parameters (must match rebuild_regular_burst_inputs_h2.sh) ──
BETA=1.0
CANCEL_WINDOW=0.050
TAU_MAX=10.0
RTH_START=34200
RTH_END=57600
WORKERS=${WORKERS:-4}

# ── Helper: run data_processor for one ticker + intensity ──
run_data_processor() {
  local ticker="$1"
  local intensity="$2"
  local outdir="$3"

  local tag="b${BETA}_i${intensity}"
  tag="${tag//./'p'}"  # replace . with p
  # Use sed for reliable substitution
  tag=$(echo "${tag}" | sed 's/\./p/g')

  local raw_csv="${outdir}/bursts_${ticker}_${tag}.csv"
  local perm_csv="${outdir}/bursts_${ticker}_${tag}_filtered.csv"

  if [ -s "${perm_csv}" ]; then
    echo "[${ticker}/${tag}] Already complete — skipping."
    return 0
  fi

  if [ ! -s "${raw_csv}" ]; then
    echo "[${ticker}/${tag}] Running data_processor -> ${raw_csv}"
    ./data_processor "${ROOT}/data/${ticker}" "${raw_csv}" \
      -H "${BETA}" \
      -I "${intensity}" \
      -w "${CANCEL_WINDOW}" \
      -v 0 \
      -d 0.5 \
      -r 1.0 \
      -k 0 \
      -t "${TAU_MAX}" \
      -j "${WORKERS}" \
      -b "${RTH_START}" -e "${RTH_END}"
  else
    echo "[${ticker}/${tag}] Raw CSV exists — skipping data_processor."
  fi

  echo "[${ticker}/${tag}] Running compute_permanence.py -> ${perm_csv}"
  python3 src_py/compute_permanence.py \
    "${raw_csv}" open_all.csv close_all.csv \
    --kappa 0 --ticker "${ticker}"

  echo "[${ticker}/${tag}] Done."
}

# ── Flatten: move files out of shared_cache and remove the now-empty dir ──
flatten_params_dir() {
  local ticker="$1"
  local params_dir="results/${ticker}_params"
  local cache_dir="${params_dir}/shared_cache"

  if [ ! -d "${cache_dir}" ]; then
    echo "[${ticker}] No shared_cache dir found — already flat."
    return 0
  fi

  echo "[${ticker}] Flattening ${cache_dir} -> ${params_dir}/"
  mv -f "${cache_dir}"/* "${params_dir}/"
  rmdir "${cache_dir}"
  echo "[${ticker}] Flattened."
}

echo "============================================"
echo "  Targeted Cache Patch + Flatten"
echo "============================================"

# ── NVDA ──
echo ""
echo "--- NVDA ---"
run_data_processor "NVDA" "0.5" "results/NVDA_params/shared_cache"
run_data_processor "NVDA" "0.8" "results/NVDA_params/shared_cache"

# ── TSLA ──
echo ""
echo "--- TSLA ---"
run_data_processor "TSLA" "0.8" "results/TSLA_params/shared_cache"

# ── Flatten all four tickers ──
echo ""
echo "--- Flattening directory structure ---"
for TICKER in NVDA TSLA JPM MS; do
  flatten_params_dir "${TICKER}"
done

echo ""
echo "============================================"
echo "  All done! New layout: results/TICKER_params/*.csv"
echo "============================================"
