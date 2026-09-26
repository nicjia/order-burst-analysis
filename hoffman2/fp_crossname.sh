#!/bin/bash
#$ -cwd
#$ -o results/fp_multiday_v1/log/
#$ -e results/fp_multiday_v1/log/
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 results/fp_multiday_v1/code/fp_crossname.py results/fp_multiday_v1/crossname_$CELL results/p4_revisit_v1/out/$CELL || exit 1
echo "done $(date)"
