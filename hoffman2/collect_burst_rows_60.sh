#!/bin/bash
# Collect usable run/60 burst rows (stage 3b input) for both fingerprint panels.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
for g in explore_2024 confirm_2021; do
  G=results/fingerprint_v1/$g
  python3 results/fingerprint_v1/code_v2/collect_burst_rows.py --dir $G/burst_rows_run_60 --out $G/burst_rows_run_60_usable.csv.gz
done
