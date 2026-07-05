#!/bin/bash
cd /u/scratch/n/nicjia/order-burst-analysis || exit 1
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source .venv/bin/activate
export OMP_NUM_THREADS=1
export OB_DROP_DB=1
mkdir -p results/sgd_nodb
read VF DT VR KP < <(python3 -c "import json;d=json.load(open('results/optuna_regression/universal_median_params.json'));print(d.get('vol_frac',0.00197),d.get('dir_thresh',0.763),d.get('vol_ratio',0.280),d.get('kappa',1.085))")
echo "params: $VF $DT $VR $KP"
NAMES="AAPL ADBE COST CRM DIS HD KO MA MCD MSFT ORCL PEP PG SBUX T TGT V VZ CAT GE"
for t in $NAMES; do
  data="results/bursts_${t}_baseline_unfiltered.csv"; [ -s "$data" ] || { echo "skip $t"; continue; }
  python3 src_py/online_sgd_backtest.py --data "$data" --target reg_clop --hawkes-tag b1p0_i0p5 \
     --vol-frac "$VF" --dir-thresh "$DT" --vol-ratio "$VR" --kappa "$KP" \
     --start-date 2019-01-01 --end-date 2026-12-31 --ticker "$t" \
     --execution-mode phase3_flow --signal-mode direction --position-mode fixed_aum \
     --fixed-aum 10000000 --round-trip-bps-cost 1.0 \
     --daily-open-csv open_all.csv --daily-close-csv close_all.csv \
     --debug-trades-out "results/sgd_nodb/${t}_debug.csv" > "results/sgd_nodb/${t}.log" 2>&1 || echo "FAIL $t"
  s=$(grep -i "Annualized Sharpe Ratio" "results/sgd_nodb/${t}.log" | head -1 | grep -oE '[-0-9.]+$')
  echo "DONE $t nodb_sharpe=$s"
done
echo "ALL_M3_DONE"
