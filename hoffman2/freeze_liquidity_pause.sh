#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
python3 src_py/fit_liquidity_pause.py \
  --input 'results/liquidity_pause_train/out/*.csv' \
  --out config/liquidity_pause_2023.json \
  > results/liquidity_pause_train/freeze.log
test -s config/liquidity_pause_2023.json
