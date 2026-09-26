#!/bin/bash
# fingerprint-v1 stage 3 input: one row per burst for a fixed definition, from cached packets.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${FP_DIR:?}; RULE=${FP_RULE:?}; GAP=${FP_GAP:?}
CODE=results/fingerprint_v1/code_v2
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
test -n "$TK"
OUT="$GD/burst_rows_${RULE}_${GAP}"
mkdir -p "$OUT"
if [ -s "$OUT/$TK.csv.gz" ]; then exit 0; fi
while read -r dd; do
  s=$(cat "$GD/status/$TK/$dd.txt" 2>/dev/null || echo absent)
  if [ "$s" != ok ] && [ "$s" != missing ]; then echo "incomplete receipt $TK $dd: $s" >&2; exit 7; fi
done < "$GD/dates.txt"
python3 "$CODE/fingerprint_burst_rows.py" --packets "$GD/packets/$TK" --pairs "$GD/pairs.txt" \
  --ticker "$TK" --rule "$RULE" --gap "$GAP" --out "$OUT/$TK.part.csv" > "$OUT/$TK.log" 2>&1
gzip -c "$OUT/$TK.part.csv" > "$OUT/$TK.part.csv.gz" && mv "$OUT/$TK.part.csv.gz" "$OUT/$TK.csv.gz" && rm -f "$OUT/$TK.part.csv"
