#!/bin/bash
#$ -cwd
#$ -o results/fp_multiday_v1/log/
#$ -e results/fp_multiday_v1/log/
# fingerprint-multiday-v1 stage 1: compact fingerprint tables from the p4-revisit-v1 npz files (no outcomes read).
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
B=results/p4_revisit_v1/out
python3 results/fp_multiday_v1/code/fp_multiday_extract.py results/fp_multiday_v1/tables $B/${CELLS// / $B/} || exit 1
echo "done $(date)"
