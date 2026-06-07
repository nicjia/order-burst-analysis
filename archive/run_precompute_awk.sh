#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/precompute.out
#$ -l h_data=4G,h_rt=00:20:00
#$ -pe shared 8

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

python3 src_py/precompute_lob_volume_awk.py
