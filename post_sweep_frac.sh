#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/post_sweep_frac_$JOB_ID.out
#$ -l h_data=4G,h_rt=1:00:00

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

set -Eeo pipefail

echo "1. Ranking fractional-ADV parameter sets..."
python3 src_py/rank_sweep_params.py \
  --tickers "NVDA,TSLA,JPM,MS" \
  --models "logreg_l2" \
  --results-root results \
  --sweep-prefix "silence_sweep_frac" \
  --min-coverage 4 \
  --rank-objective variance_first \
  --top-k 5

echo "2. Extracting top 5 fractional configs..."
mkdir -p results/sweep_rankings_frac
cut -d, -f2 results/sweep_rankings/global_top_configs.csv | tail -n +2 | head -n 5 > results/sweep_rankings_frac/top5_configs.txt

echo "Done. Top configs:"
cat results/sweep_rankings_frac/top5_configs.txt
