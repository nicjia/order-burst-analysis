#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
python3 src_py/aggregate_hidden_packet_spread_scaling.py \
  --input 'results/hidden_packet_spread_scaling/out/*.csv' \
  --out results/hidden_packet_spread_scaling/summary.json \
  > results/hidden_packet_spread_scaling/aggregate.log
