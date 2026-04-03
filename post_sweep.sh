#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/post_sweep_$JOB_ID.out
#$ -l h_data=4G,h_rt=1:00:00

set -Eeo pipefail

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

echo "1. Ranking parameter sets..."
python3 src_py/rank_sweep_params.py \
  --tickers "NVDA,TSLA,JPM,MS" \
  --models "logreg_l2" \
  --results-root results \
  --min-coverage 4 \
  --top-k 5

echo "2. Extracting top 5 configs..."
cut -d, -f2 results/sweep_rankings/global_top_configs.csv | tail -n +2 | head -n 5 > results/sweep_rankings/top5_configs.txt

echo "3. Submitting final evaluation array..."
# Submitting the 20-task array (4 tickers x 5 models)
qsub -t 1-20 -v TICKERS="NVDA TSLA JPM MS",MODELS="et,rf,stacking,lgb_tuned,adaboost" eval_top_configs.sh

echo "Handoff complete."