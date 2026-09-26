#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
python3 src_py/aggregate_hidden_packet_bounds.py \
  --input 'results/hidden_packet_bounds/out/*.csv' \
  --out results/hidden_packet_bounds/summary.json \
  > results/hidden_packet_bounds/aggregate.log
test -s results/hidden_packet_bounds/summary.json
