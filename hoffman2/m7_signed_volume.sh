#!/bin/bash
# qsub driver for M7 (ii) plain un-gated signed-volume reversal baseline.
# qsub -l h_data=16G,h_rt=2:00:00 -pe shared 1 -cwd -o logs/m7sv.out -e logs/m7sv.err -N m7sv hoffman2/m7_signed_volume.sh
cd /u/scratch/n/nicjia/order-burst-analysis || exit 1
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source .venv/bin/activate
export OMP_NUM_THREADS=1
python src_py/m7_signed_volume.py
echo "DONE_M7SV rc=$?"
