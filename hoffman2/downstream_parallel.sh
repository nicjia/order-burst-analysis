#!/bin/bash
#$ -S /bin/bash
# ─────────────────────────────────────────────────────────────────────────
# downstream_parallel.sh — Parallel downstream analysis for Hoffman2
# ─────────────────────────────────────────────────────────────────────────
# Cluster-parallel counterpart to run_pipeline.sh's sequential perm/optuna/
# backtest/research phases. Per-ticker work is embarrassingly parallel, so we
# fan out with `xargs -P JOBS`. BLAS threads are pinned to 1 to avoid
# oversubscription when many single-core python jobs run at once.
#
# Strict TRAIN/OOS firewall: Optuna runs on TRAIN tickers only; the median of
# the TRAIN-derived physical params is then applied UNCHANGED to OOS backtests
# and research (no leakage).
#
# Submit (compute node):
#   qsub -S /bin/bash -l highp -l h_data=8G,h_rt=06:00:00 -pe shared 16 -cwd \
#        -o logs/dsp.out -e logs/dsp.err -N dsp hoffman2/downstream_parallel.sh
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
source .venv/bin/activate
export PYTHONNOUSERSITE=1
# Pin BLAS/OpenMP to 1 thread/process — we get parallelism from xargs -P.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOBS=${JOBS:-${NSLOTS:-12}}
export TARGET=${TARGET:-reg_clop} HTAG=${HTAG:-b1p0_i0p5}
export TRAIN_START=${TRAIN_START:-2023-01-01} TRAIN_END=${TRAIN_END:-2024-12-31}
export OOS_START=${OOS_START:-2019-01-01}     OOS_END=${OOS_END:-2024-12-31}
export TRIALS=${TRIALS:-100}

mapfile -t TRAIN < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' universes/train_50.txt | awk '{print $1}')
mapfile -t OOS   < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' universes/oos_100.txt  | awk '{print $1}')
mkdir -p logs results/sgd_backtests_oos results/research

echo "=== PARALLEL DOWNSTREAM START $(date) host=$(hostname) JOBS=$JOBS ==="
echo "TRAIN=${#TRAIN[@]}  OOS=${#OOS[@]}  TARGET=$TARGET  HTAG=$HTAG  TRIALS=$TRIALS"

# ── 1. PERM filtered (kappa) in parallel — unfiltered already built upstream ──
perm_filt() {
  t=$1; base=results/bursts_${t}_baseline.csv
  [ -s "$base" ] && [ "$(wc -l < "$base")" -gt 1 ] || return 0
  [ -s results/bursts_${t}_baseline_filtered.csv ] && return 0
  seed=results/bursts_${t}_baseline_longseed_$$.csv; cp -f "$base" "$seed"
  if python3 src_py/compute_permanence.py "$seed" open_all.csv close_all.csv \
        --kappa 0.5 --ticker "$t" >/dev/null 2>&1; then
    mv -f "${seed%.csv}_filtered.csv" results/bursts_${t}_baseline_filtered.csv 2>/dev/null
  fi
  rm -f "$seed"
}
export -f perm_filt
echo "--- [1/5] PERM filtered (parallel) $(date) ---"
printf '%s\n' "${TRAIN[@]}" "${OOS[@]}" | xargs -P "$JOBS" -I{} bash -c 'perm_filt "$1"' _ {}

# ── 2. OPTUNA (TRAIN only, parallel) ──
opt_one() {
  t=$1; [ -s "results/bursts_${t}_baseline_unfiltered.csv" ] || return 0
  if ! python3 src_py/optuna_regression_sweep.py --ticker "$t" --target "$TARGET" \
        --hawkes-tag "$HTAG" --trials "$TRIALS" \
        --start-date "$TRAIN_START" --end-date "$TRAIN_END" > "logs/optuna_${t}.log" 2>&1; then
    echo "OPTUNA FAIL: $t"
  fi
}
export -f opt_one
echo "--- [2/5] OPTUNA TRAIN x$TRIALS (parallel) $(date) ---"
printf '%s\n' "${TRAIN[@]}" | xargs -P "$JOBS" -I{} bash -c 'opt_one "$1"' _ {}

# ── 3. Median universal params from TRAIN (applied unchanged to OOS) ──
python3 - <<'PY'
import glob, json, numpy as np
keys = ['vol_frac', 'dir_thresh', 'vol_ratio', 'kappa']
acc = {k: [] for k in keys}
n = 0
for f in glob.glob('results/optuna_regression/*/best_regression_params_reg_clop_b1p0_i0p5.json'):
    d = json.load(open(f)); n += 1
    for k in keys:
        if d.get(k) is not None: acc[k].append(float(d[k]))
med = {k: float(np.median(v)) for k, v in acc.items() if v}
json.dump({'n_train': n, **med},
          open('results/optuna_regression/universal_median_params.json', 'w'), indent=2)
print(f'Universal median params from {n} TRAIN tickers: {med}')
PY
read VF DT VR KP < <(python3 -c "import json;d=json.load(open('results/optuna_regression/universal_median_params.json'));print(d['vol_frac'],d['dir_thresh'],d['vol_ratio'],d['kappa'])")
export VF DT VR KP
echo "Universal params: vf=$VF dt=$DT vr=$VR kappa=$KP"

# ── 4. BACKTEST (OOS only, parallel) using TRAIN-derived params ──
bt_one() {
  t=$1; data=results/bursts_${t}_baseline_unfiltered.csv; [ -s "$data" ] || return 0
  pre=results/sgd_backtests_oos/${t}_${TARGET}_${HTAG}
  if ! python3 src_py/online_sgd_backtest.py --data "$data" --target "$TARGET" --hawkes-tag "$HTAG" \
        --vol-frac "$VF" --dir-thresh "$DT" --vol-ratio "$VR" --kappa "$KP" \
        --start-date "$OOS_START" --end-date "$OOS_END" --ticker "$t" \
        --execution-mode phase3_flow --signal-mode direction --position-mode fixed_aum \
        --fixed-aum 10000000 --round-trip-bps-cost 1.0 \
        --daily-open-csv open_all.csv --daily-close-csv close_all.csv \
        --debug-trades-out "${pre}_debug_trades.csv" --debug-signals-out "${pre}_debug_signals.csv" \
        > "${pre}.log" 2>&1; then
    echo "BACKTEST FAIL: $t"
  fi
}
export -f bt_one
echo "--- [3/5] BACKTEST OOS (parallel) $(date) ---"
printf '%s\n' "${OOS[@]}" | xargs -P "$JOBS" -I{} bash -c 'bt_one "$1"' _ {}

# ── 5. RESEARCH (OOS only, parallel) ──
res_one() {
  t=$1; unf=results/bursts_${t}_baseline_unfiltered.csv; [ -s "$unf" ] || return 0
  filt=results/bursts_${t}_baseline_filtered.csv; [ -f "$filt" ] || filt=""
  python3 src_py/ablation_study.py "$unf" --ticker "$t" --target "$TARGET" \
     --start-date "$OOS_START" --end-date "$OOS_END" > "results/research/${t}_ablation.log" 2>&1 || true
  python3 src_py/transaction_cost_grid.py "$unf" --ticker "$t" --close-csv close_all.csv \
     --start-date "$OOS_START" --end-date "$OOS_END" \
     --output-csv "results/research/${t}_tc_grid.csv" > "results/research/${t}_tc_grid.log" 2>&1 || true
  python3 src_py/time_of_day_analysis.py "$unf" --ticker "$t" \
     --start-date "$OOS_START" --end-date "$OOS_END" > "results/research/${t}_time_of_day.log" 2>&1 || true
  python3 src_py/poisson_baseline_test.py "$unf" ${filt:+--filtered "$filt"} --ticker "$t" \
     --start-date "$OOS_START" --end-date "$OOS_END" > "results/research/${t}_poisson.log" 2>&1 || true
  python3 src_py/naive_baseline_markout.py "$unf" ${filt:+--filtered "$filt"} --ticker "$t" \
     --start-date "$OOS_START" --end-date "$OOS_END" > "results/research/${t}_naive.log" 2>&1 || true
}
export -f res_one
echo "--- [4/5] RESEARCH OOS (parallel) $(date) ---"
printf '%s\n' "${OOS[@]}" | xargs -P "$JOBS" -I{} bash -c 'res_one "$1"' _ {}

# ── Cross-ticker (sequential, cheap) ──
echo "--- [5/5] cross-ticker $(date) ---"
python3 src_py/multiple_testing_correction.py results/optuna_regression/ --all-tickers \
   --n-trials "$TRIALS" > results/research/multiple_testing_correction.log 2>&1 || true
OOS_CSV=$(IFS=,; echo "${OOS[*]}")
REGIME_ARG=""; [ -f results/regime/regime_classifications.csv ] && REGIME_ARG="--regime-csv results/regime/regime_classifications.csv"
python3 src_py/panel_regression.py --burst-dir results/ --tickers "$OOS_CSV" \
   --open-csv open_all.csv --close-csv close_all.csv $REGIME_ARG \
   --start-date "$OOS_START" --end-date "$OOS_END" \
   --output-csv results/research/coi_panel_oos.csv > results/research/panel_regression_oos.log 2>&1 || true

echo "=== PARALLEL DOWNSTREAM DONE $(date) ==="
echo "optuna JSONs:   $(find results/optuna_regression -name 'best_regression_params_*.json' | wc -l)"
echo "backtest logs:  $(ls results/sgd_backtests_oos/*.log 2>/dev/null | wc -l)"
echo "research files: $(ls results/research/ 2>/dev/null | wc -l)"
