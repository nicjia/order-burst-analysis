#!/bin/bash
#$ -S /bin/bash
# Full-universe cross-sectional panel (Reviewer R6/B3-B6): regime classification
# + COI Fama-MacBeth + quintile sorts over every ticker that has burst data.
# Window is 2022-2024 (CRSP overnight labels end 2024-12-30).
#   qsub -S /bin/bash -l highp -l h_data=16G,h_rt=02:00:00 -pe shared 8 -cwd \
#        -o logs/panelfull.out -e logs/panelfull.err -N panelfull hoffman2/run_panel_full.sh
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
source .venv/bin/activate
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p results/research results/regime

# Every full_500 ticker that actually has permanence data.
TK=$(for t in $(grep -vE '^[[:space:]]*#|^[[:space:]]*$' universes/full_500.txt | awk '{print $1}'); do
       [ -s "results/bursts_${t}_baseline_unfiltered.csv" ] && echo "$t"; done | paste -sd, -)
N=$(echo "$TK" | tr ',' '\n' | grep -c .)
echo "=== FULL-UNIVERSE PANEL — $N tickers — $(date) host=$(hostname) ==="

echo "--- regime classification (universe-wide R3 sign-flip) ---"
python3 src_py/regime_classifier.py --burst-dir results/ --close-csv close_all.csv \
   --tickers "$TK" --output-dir results/regime > results/research/regime_full.log 2>&1 \
   && echo "regime OK" || echo "regime FAILED (see results/research/regime_full.log)"

echo "--- panel regression (COI Fama-MacBeth + quintile + FF if available) ---"
python3 src_py/panel_regression.py --burst-dir results/ --tickers "$TK" \
   --open-csv open_all.csv --close-csv close_all.csv \
   --regime-csv results/regime/regime_classifications.csv \
   --start-date 2022-01-01 --end-date 2024-12-31 \
   --output-csv results/research/coi_panel_full.csv > results/research/panel_full.log 2>&1 \
   && echo "panel OK" || echo "panel FAILED (see results/research/panel_full.log)"

echo "=== DONE $(date) ==="
echo "----- panel_full.log (results) -----"
sed -n '/FAMA-MACBETH/,$p' results/research/panel_full.log 2>/dev/null
