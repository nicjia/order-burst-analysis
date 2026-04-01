#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/funnel_step3_zoo_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=8:00:00
#$ -pe shared 4
#$ -t 1-147
#$ -N model_zoo

set -Eeo pipefail
trap 'echo "ERROR line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

# Delegate to multi-ticker zoo runner (expects bursts_*_{unfiltered,filtered}.csv).
bash "${ROOT}/hoffman2_model_zoo.sh"

echo "Step3 model zoo task ${SGE_TASK_ID} done at $(date)"