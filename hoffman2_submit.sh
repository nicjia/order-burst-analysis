#!/bin/bash
# ─────────────────────────────────────────────────────────────
# hoffman2_submit.sh — UGE/SGE job script for UCLA Hoffman2
# ─────────────────────────────────────────────────────────────
#
# Submit with:
#   qsub hoffman2_submit.sh
#
# Or compile first, then submit just the run:
#   make clean && make
#   qsub hoffman2_submit.sh
# ─────────────────────────────────────────────────────────────

#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#$ -l h_data=16G,h_rt=8:00:00
#$ -pe shared 1
#$ -N burst_detect

# ── Environment ──────────────────────────────────────────────
module load gcc/11.3.0       # need C++17 support (gcc >= 7)

mkdir -p logs

# ── Compile ──────────────────────────────────────────────────
make clean && make
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed." >&2
    exit 1
fi

# ── Run ──────────────────────────────────────────────────────
# Process every stock folder under data/
for stock_dir in data/*/; do
    ticker=$(basename "$stock_dir" | cut -d'_' -f1)
    outfile="bursts_${ticker}.csv"
    echo "Processing ${ticker} → ${outfile}"
    ./data_processor "$stock_dir" "$outfile" -s 1.0 -d 0.9
done

echo "All done."
