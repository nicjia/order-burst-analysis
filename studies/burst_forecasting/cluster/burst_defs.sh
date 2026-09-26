#!/bin/bash
#$ -cwd
#$ -o results/fp_multiday_v1/log/
#$ -e results/fp_multiday_v1/log/
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cp results/p4_revisit_v1/code/p4_aggregate.py results/fp_multiday_v1/code/ 2>/dev/null
python3 results/fp_multiday_v1/code/burst_defs_extract.py results/burst_defs_v1 results/p4_revisit_v1/out/$CELL || exit 1
echo "done $(date)"
