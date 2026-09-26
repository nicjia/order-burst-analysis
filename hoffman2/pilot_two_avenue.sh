#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
test -f src_py/burst_alt.py
test -f src_py/execution_packets.py
mkdir -p results/two_avenue/pilot
pilot_dir=$(mktemp -d results/two_avenue/pilot/raw.XXXXXX)
cleanup(){ rm -rf "$pilot_dir"; }
trap cleanup EXIT
ssh nicjia@lobster2.math.ucla.edu "cat /lobster/2024/20240103/AAPL.7z" > "$pilot_dir/AAPL.7z"
~/bin/7z x "$pilot_dir/AAPL.7z" -o"$pilot_dir/extracted" -y >/dev/null
msg=$(find "$pilot_dir/extracted" -name '*message*.csv' | head -1)
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
python3 src_py/two_avenue_extract.py --msg "$msg" --ticker AAPL \
  --model config/metaorder_simulation_model.json --header \
  > results/two_avenue/pilot/AAPL_20240103.csv
python3 src_py/burst_quality_v2.py --msg "$msg" --ticker AAPL --draws 10 \
  > results/two_avenue/pilot/AAPL_20240103_bq2.csv
wc -l results/two_avenue/pilot/AAPL_20240103.csv \
  results/two_avenue/pilot/AAPL_20240103_bq2.csv
