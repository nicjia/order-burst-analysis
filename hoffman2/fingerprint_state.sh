#!/bin/bash
# fingerprint-v1 corrected E3 (lag-binned state similarity), from cached packets.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${FP_DIR:?}
CODE=results/fingerprint_v1/code_v2
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
test -n "$TK"
mkdir -p "$GD/state_v3"
if [ -s "$GD/state_v3/$TK.npz" ]; then exit 0; fi
python3 "$CODE/fingerprint_state.py" --packets "$GD/packets/$TK" --pairs "$GD/pairs.txt" \
  --ticker "$TK" --out "$GD/state_v3/$TK.npz" > "$GD/state_v3/$TK.log" 2>&1
