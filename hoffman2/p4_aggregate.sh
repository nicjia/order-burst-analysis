#!/bin/bash
#$ -cwd
#$ -o results/p4_revisit_v1/log/
#$ -e results/p4_revisit_v1/log/
# P4 revisit v1 stage 2: aggregate one cell (P4_REVISIT_DESIGN.md sections 4-6).
# Submit: qsub -N p4aggDEV -l highp,h_rt=12:00:00,h_data=3G -pe shared 24 -q bertozzi_pod.q \
#           -v CELL=DEV,PROCS=24[,MODELS=results/p4_revisit_v1/phase2] hoffman2/p4_aggregate.sh
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${CELL:?}"
B=results/p4_revisit_v1
OUT=$B/agg/$CELL
mkdir -p "$OUT"
EXTRA=""
[ -n "${MODELS:-}" ] && EXTRA="--models $MODELS"
python3 $B/code/p4_aggregate.py --cell "$CELL" --jobs $B/jobs/$CELL.txt --npz-dir $B/out/$CELL --crsp-dir $B/crsp \
  --out "$OUT" --per-nameday 10 --procs "${PROCS:-24}" $EXTRA || { echo "aggregation failed $(date)" >&2; exit 1; }
sha256sum "$OUT"/*_"$CELL".csv.gz $B/code/p4_aggregate.py $B/code/p4_phase2.py > "$OUT/checksums.txt"
echo "aggregate $CELL done $(date)"
