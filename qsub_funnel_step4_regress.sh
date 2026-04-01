#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/funnel_step4_regress_$JOB_ID.out
#$ -l h_data=16G,h_rt=12:00:00
#$ -pe shared 8
#$ -N regress_eval

set -Eeo pipefail
trap 'echo "ERROR line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

bash "${ROOT}/regress.sh"

echo "Step4 regression eval done at $(date)"