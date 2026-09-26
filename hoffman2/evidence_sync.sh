#!/bin/bash
# program-evidence-v1 module G: one task per fingerprint date (all names of the group together).
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${FP_DIR:?}; OUT=${EV_OUT:?}
CODE=results/program_evidence_v1/code
DD=$(sed -n "${SGE_TASK_ID}p" "$GD/dates.txt")
test -n "$DD"
mkdir -p "$OUT"
[ -s "$OUT/$DD.npz" ] && exit 0
python3 "$CODE/evidence_sync.py" --group "$GD" --date "$DD" --model results/program_evidence_v1/program_model_run60.json \
  --out "$OUT/$DD.npz" > "$OUT/$DD.json" 2> "$OUT/$DD.stderr"
