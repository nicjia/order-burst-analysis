#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
test -s config/strict_continuation_2023.json
mkdir -p results/strict_cont_oos/pilot
pilot_dir=$(mktemp -d results/strict_cont_oos/pilot/raw.XXXXXX)
cleanup(){ rm -rf "$pilot_dir"; }
trap cleanup EXIT
rsync -a --timeout=120 \
  nicjia@lobster2.math.ucla.edu:/lobster/2025/20250102/AAPL.7z "$pilot_dir/AAPL.7z"
~/bin/7z x "$pilot_dir/AAPL.7z" -o"$pilot_dir/extracted" -y >/dev/null
msg=$(find "$pilot_dir/extracted" -name '*message*.csv' | head -1)
python3 src_py/strict_continuation_oos_day.py --msg "$msg" --ticker AAPL \
  --simulation-model config/metaorder_simulation_model.json \
  --frozen config/strict_continuation_2023.json --header \
  > results/strict_cont_oos/pilot/AAPL_20250102.csv
python3 src_py/validate_strict_continuation_output.py \
  --input results/strict_cont_oos/pilot/AAPL_20250102.csv --expected-year 2025
