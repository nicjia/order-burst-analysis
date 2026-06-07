#!/bin/bash

# Submit Phase 1: Rebuild regular burst inputs
PHASE1_JOB=$(qsub -terse rebuild_regular_burst_inputs_h2.sh)
echo "Phase 1 submitted: $PHASE1_JOB"

# Create Phase 2 script for regression optuna sweep
cat << 'EOF' > run_optuna_regression_all_tickers_h2.sh
#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/optuna_regression_$JOB_ID_$TASK_ID.out
#$ -l h_data=16G,h_rt=06:59:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
set -Eeo pipefail

TICKERS=(NVDA TSLA JPM MS)
TARGET="reg_clop"
TRIALS=100
START_DATE=${START_DATE:-2023-01-01}
END_DATE=${END_DATE:-2024-12-31}

if [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ]; then
    IDX=$((SGE_TASK_ID - 1))
    if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${#TICKERS[@]}" ]; then
        exit 0
    fi
    TICKER=${TICKERS[$IDX]}
else
    TICKER=${TICKERS[0]}
fi

echo "Running ticker=${TICKER}"
python3 src_py/optuna_regression_sweep.py \
    --ticker "${TICKER}" \
    --target "${TARGET}" \
    --trials "${TRIALS}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}"
EOF

chmod +x run_optuna_regression_all_tickers_h2.sh

# Submit Phase 2: Optuna Regression Sweep (depends on Phase 1)
PHASE2_JOB=$(qsub -terse -hold_jid ${PHASE1_JOB} -t 1-4 run_optuna_regression_all_tickers_h2.sh)
echo "Phase 2 submitted: $PHASE2_JOB"

# Submit Phase 3: SGD Backtest on 2023-2024 (depends on Phase 2)
PHASE3_JOB=$(qsub -terse -hold_jid ${PHASE2_JOB} -t 1-4 run_sgd_backtest_optuna_regression_all_tickers_2023_2024_h2.sh)
echo "Phase 3 (Train Tickers) submitted: $PHASE3_JOB"

# Modify Phase 3 script to handle OOS tickers
cat << 'EOF' > run_sgd_backtest_optuna_regression_oos_h2.sh
#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/sgd_optuna_regression_oos_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=02:59:00
#$ -pe shared 4

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /etc/profile
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
set -Eeo pipefail

OOS_TICKERS=(AAPL LLY SPY)
# Use JPM's optimal params as the representative institutional footprint
JSON_FILE="results/optuna_regression/JPM/best_regression_params_reg_clop_b1p0_i0p3.json"

if [ -n "${SGE_TASK_ID:-}" ] && [ "${SGE_TASK_ID}" != "undefined" ]; then
    IDX=$((SGE_TASK_ID - 1))
    if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${#OOS_TICKERS[@]}" ]; then
        exit 0
    fi
    TICKER=${OOS_TICKERS[$IDX]}
else
    TICKER=${OOS_TICKERS[0]}
fi

mapfile -t P < <(python3 - "$JSON_FILE" "b1p0_i0p3" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1]))
print(obj.get("hawkes_tag", sys.argv[2]))
print(obj["vol_frac"])
print(obj["dir_thresh"])
print(obj["vol_ratio"])
print(obj.get("kappa", 0.0))
PY
)

USE_HAWKES_TAG="${P[0]}"
VOL_FRAC="${P[1]}"
DIR_THRESH="${P[2]}"
VOL_RATIO="${P[3]}"
KAPPA="${P[4]}"

DATA_PATH="results/bursts_${TICKER}_baseline_unfiltered.csv"
OUT_PREFIX="results/sgd_backtests_fixed_aum_oos/${TICKER}_reg_clop_regression_${USE_HAWKES_TAG}"
mkdir -p results/sgd_backtests_fixed_aum_oos

python3 src_py/online_sgd_backtest.py \
    --data "${DATA_PATH}" \
    --target "reg_clop" \
    --hawkes-tag "${USE_HAWKES_TAG}" \
    --vol-frac "${VOL_FRAC}" \
    --dir-thresh "${DIR_THRESH}" \
    --vol-ratio "${VOL_RATIO}" \
    --kappa "${KAPPA}" \
    --start-date "2019-01-01" \
    --end-date "2021-12-31" \
    --ticker "${TICKER}" \
    --execution-mode phase3_flow \
    --signal-mode direction \
    --position-mode fixed_aum \
    --fixed-aum 10000000 \
    --round-trip-bps-cost 1.0 \
    --daily-open-csv open_all.csv \
    --daily-close-csv close_all.csv \
    --debug-trades-out "${OUT_PREFIX}_debug_trades.csv" \
    --debug-signals-out "${OUT_PREFIX}_debug_signals.csv" \
    | tee "${OUT_PREFIX}.log"
EOF

chmod +x run_sgd_backtest_optuna_regression_oos_h2.sh

PHASE3_OOS_JOB=$(qsub -terse -hold_jid ${PHASE2_JOB} -t 1-3 run_sgd_backtest_optuna_regression_oos_h2.sh)
echo "Phase 3 (OOS Tickers) submitted: $PHASE3_OOS_JOB"
