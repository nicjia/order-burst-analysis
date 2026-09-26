#!/bin/bash
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
test -f src_py/fit_frozen_models.py
python3 src_py/fit_frozen_models.py \
  --input 'results/two_avenue/out/*.csv' \
  --out config/informed_models_2023.json \
  > results/two_avenue/freeze.log
test -s config/informed_models_2023.json

