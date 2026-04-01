#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/funnel_step2_zoo_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=8:00:00
#$ -pe shared 4
#$ -t 1-147
#$ -N model_zoo

set -Eeo pipefail
trap 'echo "ERROR line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

# Provide expected inputs for hoffman2_model_zoo.sh if only baseline files exist.
for T in NVDA TSLA JPM MS; do
  if [ -f "results/bursts_${T}_baseline.csv" ] && [ ! -f "results/bursts_${T}_unfiltered.csv" ]; then
    ln -sf "results/bursts_${T}_baseline.csv" "results/bursts_${T}_unfiltered.csv"
  fi
  if [ -f "results/bursts_${T}_baseline_filtered.csv" ] && [ ! -f "results/bursts_${T}_filtered.csv" ]; then
    ln -sf "results/bursts_${T}_baseline_filtered.csv" "results/bursts_${T}_filtered.csv"
  fi
  if [ ! -f "results/bursts_${T}_unfiltered.csv" ] || [ ! -f "results/bursts_${T}_filtered.csv" ]; then
    echo "Missing required zoo input for ${T}. Need results/bursts_${T}_unfiltered.csv and results/bursts_${T}_filtered.csv" >&2
    exit 3
  fi
done

# Delegate per-task model execution.
bash "${ROOT}/hoffman2_model_zoo.sh"

echo "Step2 model zoo task ${SGE_TASK_ID} done at $(date)"