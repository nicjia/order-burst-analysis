#!/bin/bash
cd /u/scratch/n/nicjia/order-burst-analysis || exit 1
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source .venv/bin/activate
export OMP_NUM_THREADS=1
python src_py/tod_coi_test.py shard ${SGE_TASK_ID} 10
echo "DONE_TODCOI ${SGE_TASK_ID} rc=$?"
