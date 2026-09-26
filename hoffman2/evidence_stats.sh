#!/bin/bash
# program-evidence-v1 modules A, B, I from fingerprint-v1 cached packets: one task per name.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${FP_DIR:?set FP_DIR to the fingerprint-v1 group directory}
OUT=${EV_OUT:?set EV_OUT to the output directory}
CODE=results/program_evidence_v1/code
test -s "$CODE/evidence_stats.py"
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
test -n "$TK"
mkdir -p "$OUT"
if [ -s "$OUT/$TK.npz" ]; then exit 0; fi
if ! ls "$GD/packets/$TK/"*.npz > /dev/null 2>&1; then echo "no packets for $TK" > "$OUT/$TK.absent"; exit 0; fi
python3 "$CODE/evidence_stats.py" --packets "$GD/packets/$TK" --pairs "$GD/pairs.txt" --ticker "$TK" \
  --out "$OUT/$TK.npz" > "$OUT/$TK.json" 2> "$OUT/$TK.stderr"
