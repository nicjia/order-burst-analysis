#!/bin/bash
# program-evidence-v1 module J: per-ticker statistics for a years_* or tsp_* group, after packet
# extraction (contig_packets.sh with the group's jobs). Runs fingerprint-v1 stage 2 (code_v2) and
# tsp_stats.py. EV_PACKETS optionally points at another packet root (reused fingerprint-v1 caches).
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${EV_GROUP:?}; PK=${EV_PACKETS:-$GD/packets}; UNI=${EV_UNIVERSE:-$GD/universe.txt}
CODE=results/program_evidence_v1/code
TK=$(sed -n "${SGE_TASK_ID}p" "$UNI")
test -n "$TK"
mkdir -p "$GD/stats_v2" "$GD/tape"
if [ -f "$GD/jobs/$TK.txt" ] && [ -d "$GD/status/$TK" ]; then
  while read -r dd tk; do
    s=$(cat "$GD/status/$TK/$dd.txt" 2>/dev/null || echo absent)
    if [ "$s" != ok ] && [ "$s" != missing ]; then echo "incomplete receipt $TK $dd: $s" >&2; exit 7; fi
  done < "$GD/jobs/$TK.txt"
fi
if ! ls "$PK/$TK/"*.npz > /dev/null 2>&1; then echo absent > "$GD/tape/$TK.absent"; exit 0; fi
if [ ! -s "$GD/stats_v2/$TK.npz" ]; then
  python3 results/fingerprint_v1/code_v2/fingerprint_stats.py --packets "$PK/$TK" --pairs "$GD/pairs.txt" --ticker "$TK" \
    --out "$GD/stats_v2/$TK.npz" > "$GD/stats_v2/$TK.json" 2> "$GD/stats_v2/$TK.stderr"
fi
if [ ! -s "$GD/tape/$TK.csv" ]; then
  python3 "$CODE/tsp_stats.py" --packets "$PK/$TK" --ticker "$TK" --model results/program_evidence_v1/program_model_run60.json \
    --out "$GD/tape/$TK.csv" > /dev/null 2> "$GD/tape/$TK.stderr"
fi
