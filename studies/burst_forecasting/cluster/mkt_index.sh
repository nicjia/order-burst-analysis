#!/bin/bash
#$ -cwd
#$ -o results/fp_multiday_v1/log/
#$ -e results/fp_multiday_v1/log/
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1
for c in DEV VAL TEST; do python3 results/fp_multiday_v1/code/market_minute_index.py results/market_index results/p4_revisit_v1/out/$c; done
echo done
