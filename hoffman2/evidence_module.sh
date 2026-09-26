#!/bin/bash
# program-evidence-v1 per-name modules on fingerprint-v1 cached packets: EV_MODULE=campaigns|markouts|price_null.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${FP_DIR:?}; OUT=${EV_OUT:?}; MOD=${EV_MODULE:?}
CODE=results/program_evidence_v1/code
MODEL=results/program_evidence_v1/program_model_run60.json
test -s "$CODE/evidence_$MOD.py"; test -s "$MODEL"
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
test -n "$TK"
mkdir -p "$OUT"
if ! ls "$GD/packets/$TK/"*.npz > /dev/null 2>&1; then echo "no packets" > "$OUT/$TK.absent"; exit 0; fi
case "$MOD" in
  campaigns)
    [ -s "$OUT/$TK.csv.gz" ] && exit 0
    python3 "$CODE/evidence_campaigns.py" --packets "$GD/packets/$TK" --pairs "$GD/pairs.txt" --ticker "$TK" \
      --out "$OUT/$TK.csv.gz" > "$OUT/$TK.json" 2> "$OUT/$TK.stderr" ;;
  markouts)
    [ -s "$OUT/$TK.npz" ] && exit 0
    python3 "$CODE/evidence_markouts.py" --packets "$GD/packets/$TK" --pairs "$GD/pairs.txt" --ticker "$TK" \
      --model "$MODEL" --out "$OUT/$TK.npz" > "$OUT/$TK.json" 2> "$OUT/$TK.stderr" ;;
  markouts_trunc)
    [ -s "$OUT/$TK.npz" ] && exit 0
    python3 "$CODE/evidence_markouts_trunc.py" --packets "$GD/packets/$TK" --pairs "$GD/pairs.txt" --ticker "$TK" \
      --model "$MODEL" --out "$OUT/$TK.npz" > "$OUT/$TK.json" 2> "$OUT/$TK.stderr" ;;
  price_null)
    [ -s "$OUT/$TK.npz" ] && exit 0
    python3 "$CODE/evidence_price_null.py" --packets "$GD/packets/$TK" --pairs "$GD/pairs.txt" --ticker "$TK" \
      --out "$OUT/$TK.npz" > "$OUT/$TK.json" 2> "$OUT/$TK.stderr" ;;
  *) echo "unknown module $MOD" >&2; exit 2 ;;
esac
