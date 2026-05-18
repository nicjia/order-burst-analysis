#!/bin/bash
# Run regression-based Optuna sweep for all tickers
. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source /u/scratch/n/nicjia/order-burst-analysis/.venv/bin/activate
cd /u/scratch/n/nicjia/order-burst-analysis

mkdir -p results/optuna_regression

echo "================================================================"
echo "  REGRESSION OPTUNA SWEEP (SGDRegressor Mirror)"
echo "  Metric: Spearman rho x confidence(n_test/500)"
echo "  Anti-Sparsity: min 200 bursts, confidence-scaled scoring"
echo "================================================================"

# Run reg_clop (overnight) for all 2023-2024 tickers with all 3 hawkes tags
for ticker in NVDA TSLA JPM MS; do
    for tag in b1p0_i0p3 b1p0_i0p5 b1p0_i0p8; do
        echo ""
        echo ">>> ${ticker} reg_clop ${tag} <<<"
        python3 src_py/optuna_regression_sweep.py \
            --ticker "${ticker}" \
            --target reg_clop \
            --hawkes-tag "${tag}" \
            --trials 100 \
            --start-date 2023-01-01 \
            --end-date 2024-12-31
    done
done

echo ""
echo "All regression Optuna sweeps complete."
echo "Results saved in results/optuna_regression/<TICKER>/"
