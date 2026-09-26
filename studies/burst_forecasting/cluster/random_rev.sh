#!/bin/bash
#$ -cwd
#$ -o results/fp_multiday_v1/log/
#$ -e results/fp_multiday_v1/log/
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
for c in VAL TEST; do python3 results/fp_multiday_v1/code/random_time_reversal.py results/random_rev results/p4_revisit_v1/out/$c results/market_index/${c}_mkt.npz; done
echo done
