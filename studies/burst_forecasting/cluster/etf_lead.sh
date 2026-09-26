#!/bin/bash
#$ -cwd
#$ -o results/etf_bursts/log/
#$ -e results/etf_bursts/log/
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1
python3 results/p4_revisit_v1/code/etf_lead.py results/etf_bursts/lead results/p4_revisit_v1/out/TEST results/etf_bursts/out/ETF || exit 1
echo done
