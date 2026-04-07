#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/post_sweep_all_$JOB_ID.out
#$ -l h_data=4G,h_rt=1:00:00

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source "${ROOT}/.venv/bin/activate"

set -Eeo pipefail

mkdir -p results/sweep_rankings

echo "=============================================="
echo "Post-Sweep Analysis (Reduced: cls_1m, cls_10m, cls_close)"
echo "=============================================="

# ── 1. Rank static (flat volume) sweep ──────────────────────
echo ""
echo "1. Ranking STATIC sweep configs..."
python3 src_py/rank_sweep_params.py \
  --tickers "NVDA,TSLA,JPM,MS" \
  --models "logreg_l2" \
  --results-root results \
  --sweep-prefix "silence_sweep" \
  --min-coverage 4 \
  --rank-objective variance_first \
  --top-k 5

echo ""
echo "  Extracting top 5 static configs..."
cut -d, -f2 results/sweep_rankings/global_top_configs.csv \
  | tail -n +2 | head -n 5 \
  > results/sweep_rankings/top5_static_configs.txt
cat results/sweep_rankings/top5_static_configs.txt

# ── 2. Rank fractional ADV sweep ────────────────────────────
echo ""
echo "2. Ranking FRACTIONAL ADV sweep configs..."
python3 src_py/rank_sweep_params.py \
  --tickers "NVDA,TSLA,JPM,MS" \
  --models "logreg_l2" \
  --results-root results \
  --sweep-prefix "silence_sweep_frac" \
  --min-coverage 4 \
  --rank-objective variance_first \
  --top-k 5

# Rename output to avoid overwriting static results
mv results/sweep_rankings/global_top_configs.csv \
   results/sweep_rankings/global_top_frac_configs.csv
mv results/sweep_rankings/logreg_l2_config_overall.csv \
   results/sweep_rankings/logreg_l2_frac_config_overall.csv 2>/dev/null || true
mv results/sweep_rankings/logreg_l2_config_target_stats.csv \
   results/sweep_rankings/logreg_l2_frac_config_target_stats.csv 2>/dev/null || true

echo ""
echo "  Extracting top 5 fractional configs..."
cut -d, -f2 results/sweep_rankings/global_top_frac_configs.csv \
  | tail -n +2 | head -n 5 \
  > results/sweep_rankings/top5_frac_configs.txt
cat results/sweep_rankings/top5_frac_configs.txt

# ── 3. Merge top configs from both sweeps ───────────────────
echo ""
echo "3. Merging top configs from both sweeps..."
cat results/sweep_rankings/top5_static_configs.txt \
    results/sweep_rankings/top5_frac_configs.txt \
  | sort -u \
  > results/sweep_rankings/top5_configs.txt

echo "  Combined unique configs:"
cat results/sweep_rankings/top5_configs.txt
N_CONFIGS=$(wc -l < results/sweep_rankings/top5_configs.txt)
echo "  Total: ${N_CONFIGS} configs"

# ── 4. Summary comparison ──────────────────────────────────
echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo ""
echo "=== STATIC (flat volume) TOP 5 ==="
head -6 results/sweep_rankings/global_top_configs.csv 2>/dev/null || echo "(not found)"
echo ""
echo "=== FRACTIONAL ADV TOP 5 ==="
head -6 results/sweep_rankings/global_top_frac_configs.csv 2>/dev/null || echo "(not found)"
echo ""
echo "Results saved to: results/sweep_rankings/"
echo ""
echo "Next step: submit full model evaluation on the merged top configs:"
echo "  qsub -t 1-20 eval_top_configs.sh"
echo "  qsub -t 1-8 eval_optuna.sh"
echo ""
echo "Done at $(date)"
