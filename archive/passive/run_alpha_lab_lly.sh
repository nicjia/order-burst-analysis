#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -j y
#$ -o logs/alpha_lab_lly_$JOB_ID.out
#$ -l h_data=16G,h_rt=02:00:00
#$ -pe shared 1

# Alpha Lab: LLY on a compute node (needs 16GB for 1.86M bursts)
ROOT=${ROOT:-/u/scratch/n/nicjia/order-burst-analysis}
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

python3 -c "
import sys; sys.path.insert(0, 'passive/src_py')
from passive_alpha_lab import run_all_strategies, generate_report

results = []
run_all_strategies('LLY', results)

# Write just LLY results
from passive_alpha_lab import generate_report
report = generate_report(results)
with open('passive/ALPHA_LAB_LLY.md', 'w') as f:
    f.write(report)
print('DONE! LLY results written.')
"
