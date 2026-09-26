#!/bin/bash
# fingerprint-v1 stage 2 only (v2 code: depth-matched null, size-similar control), from cached packets.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${FP_DIR:?set FP_DIR to the group directory}
CODE=results/fingerprint_v1/code_v2
test -s "$CODE/fingerprint_stats.py"
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
test -n "$TK"
mkdir -p "$GD/stats_v2"
if [ -s "$GD/status/$TK/stats_v2.txt" ] && [ "$(cat "$GD/status/$TK/stats_v2.txt")" = ok ] && [ -s "$GD/stats_v2/$TK.npz" ]; then exit 0; fi
while read -r dd; do
  s=$(cat "$GD/status/$TK/$dd.txt" 2>/dev/null || echo absent)
  if [ "$s" != ok ] && [ "$s" != missing ]; then echo "incomplete receipt $TK $dd: $s" >&2; exit 7; fi
done < "$GD/dates.txt"
python3 "$CODE/fingerprint_stats.py" --packets "$GD/packets/$TK" --pairs "$GD/pairs.txt" \
  --ticker "$TK" --out "$GD/stats_v2/$TK.npz" > "$GD/status/$TK/stats_v2.json" 2> "$GD/status/$TK/stats_v2.stderr"
printf 'ok\n' > "$GD/status/$TK/stats_v2.txt"
