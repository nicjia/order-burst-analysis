#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
python3 src_py/fit_strict_continuation.py \
  --input 'results/strict_cont_train/out/*.csv' \
  --out config/strict_continuation_2023.json \
  > results/strict_cont_train/freeze.log
test -s config/strict_continuation_2023.json
