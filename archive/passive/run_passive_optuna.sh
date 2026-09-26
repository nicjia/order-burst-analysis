#!/bin/bash
# ─────────────────────────────────────────────────────────────
# run_passive_optuna.sh
# ─────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."

mkdir -p results/optuna_passive

CORE_TICKERS="NVDA,TSLA,JPM,MS"
TARGETS=("reg_clop")

for target in "${TARGETS[@]}"; do
    echo "Running passive Optuna sweep across Core Universe ($CORE_TICKERS) -> $target"
    .venv/bin/python3 passive/src_py/passive_optuna_sweep.py --core-tickers "$CORE_TICKERS" --target "$target" --trials 50
    echo "---------------------------------------------------------"
done
