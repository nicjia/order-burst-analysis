#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
python3 src_py/aggregate_burst_quality_v2.py \
  --input 'results/bq2/out/*.csv' --out results/bq2/summary.csv \
  > results/bq2/aggregate.log
test -s results/bq2/summary.csv

