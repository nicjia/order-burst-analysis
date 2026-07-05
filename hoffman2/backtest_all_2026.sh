#!/bin/bash
#$ -S /bin/bash
# Full adaptive per-name SGD backtest across the whole non-TRAIN universe, 2026 data.
# Permanence is already 2026-extended (do NOT redo). ADV table now covers all 483 names
# (keystone fix), and the kappa-empty burn-in now falls back instead of crashing — so this
# should complete for the full universe, not just the ~65 that ran before.
#   qsub -l h_data=8G,h_rt=10:00:00 -pe shared 16 -cwd \
#        -o logs/backtest_all_2026.out -e logs/backtest_all_2026.err \
#        -N bt26 hoffman2/backtest_all_2026.sh
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
source .venv/bin/activate
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
JOBS=${JOBS:-14}
export TARGET=reg_clop HTAG=b1p0_i0p5
export OOS_START=2019-01-01 OOS_END=2026-12-31
mkdir -p results/sgd_backtests_oos

NADV=$(python3 -c "import pandas as pd;print(pd.read_csv('results/true_adv_daily.csv')['Ticker'].nunique())")
echo "=== BACKTEST-ALL 2026 — ADV tickers=$NADV — $(date) host=$(hostname) JOBS=$JOBS ==="
[ "$NADV" -gt 400 ] || { echo "ABORT: ADV table only $NADV tickers"; exit 1; }

mapfile -t TRAIN < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' universes/train_50.txt | awk '{print $1}')
declare -A IS_TRAIN; for t in "${TRAIN[@]}"; do IS_TRAIN[$t]=1; done
OOS=()
for t in $(grep -vE '^[[:space:]]*#|^[[:space:]]*$' universes/full_500.txt | awk '{print $1}'); do
  b="results/bursts_${t}_baseline_unfiltered.csv"
  [ -s "$b" ] && [ "$(wc -l < "$b")" -gt 1 ] || continue
  [ -n "${IS_TRAIN[$t]:-}" ] && continue
  OOS+=("$t")
done
echo "OOS universe: ${#OOS[@]} non-train tickers"

read VF DT VR KP < <(python3 -c "import json;d=json.load(open('results/optuna_regression/universal_median_params.json'));print(d['vol_frac'],d['dir_thresh'],d['vol_ratio'],d['kappa'])")
export VF DT VR KP
echo "params: vf=$VF dt=$DT vr=$VR kappa=$KP"

bt_one() {
  t=$1; data="results/bursts_${t}_baseline_unfiltered.csv"; [ -s "$data" ] || return 0
  pre="results/sgd_backtests_oos/${t}_${TARGET}_${HTAG}"
  python3 src_py/online_sgd_backtest.py --data "$data" --target "$TARGET" --hawkes-tag "$HTAG" \
     --vol-frac "$VF" --dir-thresh "$DT" --vol-ratio "$VR" --kappa "$KP" \
     --start-date "$OOS_START" --end-date "$OOS_END" --ticker "$t" \
     --execution-mode phase3_flow --signal-mode direction --position-mode fixed_aum \
     --fixed-aum 10000000 --round-trip-bps-cost 1.0 \
     --daily-open-csv open_all.csv --daily-close-csv close_all.csv \
     --debug-trades-out "${pre}_debug_trades.csv" > "${pre}.log" 2>&1 || echo "BACKTEST FAIL: $t"
}
export -f bt_one
echo "--- BACKTEST ($(date)) ---"
printf '%s\n' "${OOS[@]}" | xargs -P "$JOBS" -I{} bash -c 'bt_one "$1"' _ {}

echo "=== BACKTEST-ALL DONE $(date) ==="
N=$(grep -lE 'Annualized Sharpe Ratio:' results/sgd_backtests_oos/*_${HTAG}.log 2>/dev/null | wc -l)
F=$(grep -lE 'minimum of 1 is required|eliminated 100%' results/sgd_backtests_oos/*_${HTAG}.log 2>/dev/null | wc -l)
echo "completed (reached Sharpe): $N   still-failing: $F"
grep -hE 'Annualized Sharpe Ratio:' results/sgd_backtests_oos/*_${HTAG}.log 2>/dev/null | grep -oE '[-0-9.]+$' | \
  python3 -c 'import sys;v=[float(x) for x in sys.stdin if x.strip()];import statistics as s;print(f"N={len(v)} mean={s.mean(v):.3f} median={s.median(v):.3f} pos%={100*sum(x>0 for x in v)/len(v):.0f} min={min(v):.2f} max={max(v):.2f}") if v else print("none")'
