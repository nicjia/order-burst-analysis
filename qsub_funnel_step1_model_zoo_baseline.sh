#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/funnel_step1_zoo_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=8:00:00
#$ -pe shared 4
#$ -t 1-147
#$ -N zoo_baseline

set -Eeo pipefail
trap 'echo "ERROR line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

# Run zoo directly against baseline artifacts to avoid shared symlink races.
UNFILTERED_SUFFIX=baseline FILTERED_SUFFIX=baseline_filtered bash "${ROOT}/hoffman2_model_zoo.sh"

echo "Baseline model-zoo task ${SGE_TASK_ID} done at $(date)"