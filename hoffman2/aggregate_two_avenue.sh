#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
python3 src_py/aggregate_two_avenue_oos.py \
  --input 'results/two_avenue_oos/out/*.csv' \
  --out results/two_avenue_oos/summary.json \
  > results/two_avenue_oos/aggregate.log
test -s results/two_avenue_oos/summary.json

