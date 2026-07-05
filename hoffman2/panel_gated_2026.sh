#!/bin/bash
#$ -S /bin/bash
# Make-or-break test (Reviewer B3/B4/R6): does the GATED informational-burst COI
# predict cross-sectionally where the UNGATED signed-flow baseline (M7) does not?
# Runs both panels on the SAME 2026 data + full ~483-name universe so the only
# difference is the gate. Permanence/prices are already 2026-extended; this only
# reads burst CSVs + price matrices (no re-extraction).
#   qsub -l h_data=8G,h_rt=4:00:00 -pe shared 16 -cwd \
#        -o logs/panel_gated_2026.out -e logs/panel_gated_2026.err \
#        -N panel26 hoffman2/panel_gated_2026.sh
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
source .venv/bin/activate
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/research

# Sanity: ADV table must cover the full universe (the keystone fix).
NADV=$(python3 -c "import pandas as pd;print(pd.read_csv('results/true_adv_daily.csv')['Ticker'].nunique())")
echo "=== PANEL 2026 (gated vs ungated) — ADV tickers=$NADV — $(date) host=$(hostname) ==="
[ "$NADV" -gt 400 ] || { echo "ABORT: true_adv_daily.csv covers only $NADV tickers"; exit 1; }

# Full universe = full_500 names that actually have non-empty burst files.
DATA=()
for t in $(grep -vE '^[[:space:]]*#|^[[:space:]]*$' universes/full_500.txt | awk '{print $1}'); do
  b="results/bursts_${t}_baseline_unfiltered.csv"
  [ -s "$b" ] && [ "$(wc -l < "$b")" -gt 1 ] && DATA+=("$t")
done
TICKERS=$(IFS=,; echo "${DATA[*]}")
echo "universe: ${#DATA[@]} tickers"

read VF DT VR KP < <(python3 -c "import json;d=json.load(open('results/optuna_regression/universal_median_params.json'));print(d['vol_frac'],d['dir_thresh'],d['vol_ratio'],d['kappa'])")
echo "gate params: vf=$VF dt=$DT vr=$VR"

REGIME_ARG=""; [ -f results/regime/regime_classifications.csv ] && REGIME_ARG="--regime-csv results/regime/regime_classifications.csv"
COMMON="--burst-dir results/ --tickers $TICKERS --open-csv open_all.csv --close-csv close_all.csv \
  $REGIME_ARG --start-date 2022-01-01 --end-date 2026-12-31"

echo "--- [1/2] UNGATED baseline (M7: all directional bursts) $(date) ---"
python3 src_py/panel_regression.py $COMMON \
  --output-csv results/research/coi_panel_ungated_2026.csv \
  > results/research/panel_ungated_2026.log 2>&1 && echo "ungated OK" || echo "ungated FAILED"

echo "--- [2/2] GATED COI^info (informational bursts) $(date) ---"
python3 src_py/panel_regression.py $COMMON --gated \
  --vol-frac "$VF" --dir-thresh "$DT" --vol-ratio "$VR" \
  --output-csv results/research/coi_panel_gated_2026.csv \
  > results/research/panel_gated_2026.log 2>&1 && echo "gated OK" || echo "gated FAILED"

echo "=== PANEL 2026 DONE $(date) ==="
for tag in ungated gated; do
  echo "----- $tag headline -----"
  grep -E 'COI mode|Tickers present|Merged panel|^  COI |Long-Short' results/research/panel_${tag}_2026.log 2>/dev/null
  sed -n '/FAMA-MACBETH REGRESSION RESULTS/,/Cross-sections/p' results/research/panel_${tag}_2026.log 2>/dev/null
done
