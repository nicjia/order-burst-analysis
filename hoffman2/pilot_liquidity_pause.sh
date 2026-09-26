#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
mkdir -p results/liquidity_pause_pilot
pilot_dir=$(mktemp -d results/liquidity_pause_pilot/raw.XXXXXX)
cleanup(){ rm -rf "$pilot_dir"; }
trap cleanup EXIT
dd=$(head -1 results/strict_cont_train/dates.txt)
rsync -a --timeout=120 \
  "nicjia@lobster2.math.ucla.edu:/lobster/2023/$dd/AAPL.7z" "$pilot_dir/AAPL.7z"
~/bin/7z x "$pilot_dir/AAPL.7z" -o"$pilot_dir/extracted" -y >/dev/null
msg=$(find "$pilot_dir/extracted" -name '*message*.csv' | head -1)
python3 src_py/liquidity_pause_extract.py --msg "$msg" --ticker AAPL \
  --simulation-model config/metaorder_simulation_model.json \
  --strict-frozen config/strict_continuation_2023.json --header \
  > "results/liquidity_pause_pilot/AAPL_${dd}.csv"
python3 src_py/validate_liquidity_pause_rows.py \
  --input "results/liquidity_pause_pilot/AAPL_${dd}.csv" --expected-year 2023
