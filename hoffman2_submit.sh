#!/bin/bash
# ─────────────────────────────────────────────────────────────
# hoffman2_submit.sh — UGE/SGE job script for UCLA Hoffman2
# ─────────────────────────────────────────────────────────────
#
# Full pipeline: compile C++, detect bursts for every stock
# folder under data/, then run permanence calculations.
#
# Submit:  qsub hoffman2_submit.sh
# ─────────────────────────────────────────────────────────────

#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#$ -l h_data=16G,h_rt=12:00:00
#$ -pe shared 1
#$ -N burst_pipeline

# ── Environment ──────────────────────────────────────────────
. /u/local/Modules/default/init/bash
module load gcc       # C++17 support
module load python/3.9.6     # pandas, numpy

mkdir -p logs results

# ── 1. Compile C++ engine ────────────────────────────────────
echo "=== Compiling C++ ==="
make clean && make
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed." >&2
    exit 1
fi
echo "Compilation successful."

# ── 2. CRSP price matrices (skip if already built) ──────────
# Expects yearly/ folder with subfolders 2016/, 2017/, …
# Produces *_all.csv merged files for cross-year lookups.
OPEN_ALL="open_all.csv"
CLOSE_ALL="close_all.csv"

if [ ! -f "$OPEN_ALL" ] || [ ! -f "$CLOSE_ALL" ]; then
    echo ""
    echo "=== Building CRSP price matrices ==="
    python3 src_py/pivot_returns.py yearly/
    if [ $? -ne 0 ]; then
        echo "ERROR: pivot_returns.py failed." >&2
        exit 1
    fi
else
    echo "CRSP matrices already exist — skipping pivot_returns.py"
fi

# ── 3. Burst detection for each stock folder ────────────────
echo ""
echo "=== Running burst detection ==="
for stock_dir in data/*/; do
    ticker=$(basename "$stock_dir" | cut -d'_' -f1)
    outfile="results/bursts_${ticker}.csv"

    echo "  Processing ${ticker} → ${outfile}"
    ./data_processor "$stock_dir" "$outfile" -s 1.0 -d 0.9 -t 10.0 -b 34200 -e 57600

    if [ $? -ne 0 ]; then
        echo "  WARNING: data_processor failed for ${ticker}" >&2
        continue
    fi

    # ── 4. Compute permanence + D_b decay filter ──────────────
    echo "  Computing permanence for ${ticker}…"
    python3 src_py/compute_permanence.py "$outfile" "$OPEN_ALL" "$CLOSE_ALL" --kappa 0.10

    if [ $? -ne 0 ]; then
        echo "  WARNING: compute_permanence.py failed for ${ticker}" >&2
    fi

    # ── 5. Phase I EDA plots ─────────────────────────────────
    filtered="${outfile%.csv}_filtered.csv"
    if [ -f "$filtered" ]; then
        echo "  Generating EDA plots for ${ticker}…"
        python3 src_py/eda_phase1.py "$filtered" --outdir "results/plots_${ticker}"
    fi

    # ── 6. Phase II model training ────────────────────────────
    if [ -f "$filtered" ]; then
        echo "  Training Phase II model for ${ticker}…"
        python3 src_py/train_model.py "$filtered" --outdir "results/model_${ticker}"
    fi
done

echo ""
echo "=== All done ==="
echo "Unfiltered + filtered burst CSVs in results/"
echo "EDA plots in results/plots_<TICKER>/"
echo "Phase II model results in results/model_<TICKER>/"