#!/bin/bash
#$ -S /bin/bash
# Re-run on the CRSP-through-2026 matrices: regenerate permanence (overnight
# labels now extend to 2026), backtest every non-TRAIN stock with the already
# tuned params, and rebuild the cross-sectional panel. Optuna is intentionally
# skipped — physical params were tuned on TRAIN 2023-2024 and are unchanged.
#   qsub -S /bin/bash -l highp -l h_data=8G,h_rt=10:00:00 -pe shared 16 -cwd \
#        -o logs/rerun2026.out -e logs/rerun2026.err -N rerun26 hoffman2/rerun_2026.sh
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
source .venv/bin/activate
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
JOBS=${JOBS:-14}
export TARGET=reg_clop HTAG=b1p0_i0p5
export OOS_START=2019-01-01 OOS_END=2026-12-31
mkdir -p results/sgd_backtests_oos results/research

# Sanity: prices must extend past 2024.
PXMAX=$(python3 -c "import pandas as pd;print(int(pd.read_csv('close_all.csv',index_col=0).index.max()))")
echo "=== RERUN 2026 — close_all max=$PXMAX — $(date) host=$(hostname) JOBS=$JOBS ==="
[ "$PXMAX" -gt 20241231 ] || { echo "ABORT: close_all does not extend past 2024 ($PXMAX)"; exit 1; }

# Ticker sets.
mapfile -t TRAIN < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' universes/train_50.txt | awk '{print $1}')
declare -A IS_TRAIN; for t in "${TRAIN[@]}"; do IS_TRAIN[$t]=1; done
DATA=(); for t in $(grep -vE '^[[:space:]]*#|^[[:space:]]*$' universes/full_500.txt | awk '{print $1}'); do
  b="results/bursts_${t}_baseline.csv"; [ -s "$b" ] && [ "$(wc -l < "$b")" -gt 1 ] && DATA+=("$t"); done
OOS=(); for t in "${DATA[@]}"; do [ -n "${IS_TRAIN[$t]:-}" ] || OOS+=("$t"); done
echo "data tickers=${#DATA[@]}  train=${#TRAIN[@]}  OOS-eligible=${#OOS[@]}"

# ── 1. PERM (parallel): unfiltered (k=0) + filtered (k=0.5) with 2026 prices ──
perm_one() {
  t=$1; base="results/bursts_${t}_baseline.csv"
  [ -s "$base" ] && [ "$(wc -l < "$base")" -gt 1 ] || return 0
  if python3 src_py/compute_permanence.py "$base" open_all.csv close_all.csv --kappa 0 --ticker "$t" >/dev/null 2>&1; then
    mv -f "${base%.csv}_filtered.csv" "results/bursts_${t}_baseline_unfiltered.csv"
  fi
  seed="results/bursts_${t}_baseline_longseed_$$.csv"; cp -f "$base" "$seed"
  if python3 src_py/compute_permanence.py "$seed" open_all.csv close_all.csv --kappa 0.5 --ticker "$t" >/dev/null 2>&1; then
    mv -f "${seed%.csv}_filtered.csv" "results/bursts_${t}_baseline_filtered.csv"
  fi
  rm -f "$seed"
}
export -f perm_one
echo "--- [1/3] PERM regen (482, 2026 labels) $(date) ---"
printf '%s\n' "${DATA[@]}" | xargs -P "$JOBS" -I{} bash -c 'perm_one "$1"' _ {}

# ── universal median params (already tuned on TRAIN 2023-2024) ──
read VF DT VR KP < <(python3 -c "import json;d=json.load(open('results/optuna_regression/universal_median_params.json'));print(d['vol_frac'],d['dir_thresh'],d['vol_ratio'],d['kappa'])")
export VF DT VR KP
echo "universal params: vf=$VF dt=$DT vr=$VR kappa=$KP"

# ── 2. BACKTEST (parallel, every non-TRAIN stock, OOS through 2026) ──
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
echo "--- [2/3] BACKTEST (${#OOS[@]} non-train, OOS_END=$OOS_END) $(date) ---"
printf '%s\n' "${OOS[@]}" | xargs -P "$JOBS" -I{} bash -c 'bt_one "$1"' _ {}

# ── 3. Cross-sectional panel (full universe, 2022-2026) ──
echo "--- [3/3] panel regression (full universe, 2026) $(date) ---"
OOS_CSV=$(IFS=,; echo "${DATA[*]}")
REGIME_ARG=""; [ -f results/regime/regime_classifications.csv ] && REGIME_ARG="--regime-csv results/regime/regime_classifications.csv"
python3 src_py/panel_regression.py --burst-dir results/ --tickers "$OOS_CSV" \
   --open-csv open_all.csv --close-csv close_all.csv $REGIME_ARG \
   --start-date 2022-01-01 --end-date 2026-12-31 \
   --output-csv results/research/coi_panel_2026.csv > results/research/panel_2026.log 2>&1 \
   && echo "panel OK" || echo "panel FAILED"

echo "=== RERUN 2026 DONE $(date) ==="
echo "backtests: $(ls results/sgd_backtests_oos/*_${HTAG}.log 2>/dev/null | wc -l)"
N=$(ls results/sgd_backtests_oos/*_${HTAG}.log 2>/dev/null | wc -l)
echo "mean Sharpe: $(grep -hE 'Annualized Sharpe Ratio:' results/sgd_backtests_oos/*_${HTAG}.log 2>/dev/null | grep -oE '[-0-9.]+$' | python3 -c 'import sys;v=[float(x) for x in sys.stdin if x.strip()];print(f"N={len(v)} mean={sum(v)/len(v):.3f} pos={100*sum(x>0 for x in v)/len(v):.0f}%" if v else "none")')"
sed -n '/FAMA-MACBETH/,$p' results/research/panel_2026.log 2>/dev/null | grep -iE 'COI |Long-Short' | head
