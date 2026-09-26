#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
python3 src_py/aggregate_strict_continuation.py \
  --input 'results/strict_cont_oos/out/*.csv' \
  --out results/strict_cont_oos/summary.json \
  > results/strict_cont_oos/aggregate.log
test -s results/strict_cont_oos/summary.json
