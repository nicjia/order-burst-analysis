# Order Burst Analysis & Informational Price Discovery Pipeline

**Note to LLM Agents**: This repository contains a complex, multi-stage pipeline combining C++ data extraction on HPC clusters and Python-based machine learning evaluation. It is designed to rigorously backtest predictive models on high-frequency limit order book (LOBSTER) data without look-ahead bias. Read this document carefully to understand the pipeline architecture, execution order, and strict anti-bias rules.

## 1. Context & Objective
The goal of this project is to identify "informed trading bursts" from intraday limit order book data, and predict whether those bursts contain permanent informational value that will affect the next day's opening price. The pipeline scales to a 500-ticker universe (spanning 2016–2026), runs natively on UCLA's Hoffman2 cluster using SGE job arrays, and evaluates strategy performance using rigorous institutional statistical standards to satisfy academic referee mandates.

---

## 2. System Architecture & File Manifest

### A. HPC Infrastructure (`hoffman2/`)
- `setup_hoffman2.sh`: One-time setup script (loads gcc/11.3.0, python/3.9.6 modules, creates `venv`, compiles C++ parser).
- `master_orchestrator.sh`: The DTN (Data Transfer Node) controller run in `tmux`. Partitions the 500-ticker universe into batches of 20, rsyncs `.7z` LOBSTER files from the raw data server, and submits SGE job arrays.
- `sge_compute_worker.sh`: The compute node worker. Extracts `.7z` files, runs the C++ parser, runs Python permanence calculations, and deletes the raw data to stay under the 2TB scratch quota.
- `rerun_2026.sh`: The specific driver used to extend the 2024 analysis through 2026 for the final referee response.

### B. C++ Parser Engine (`src_cpp/`)
- `main.cpp`, `burst.cpp`, `orderbook.cpp`: High-speed C++ engine that consumes raw `*message_0.csv` and `*orderbook_0.csv` files. It reconstructs the BBO and deterministically clusters sequences of orders into directional "bursts" using a recursive Hawkes process.

### C. Python Evaluation Suite (`src_py/`)
- **Data Layers**: `compute_permanence.py` (calculates target labels like `CLOP` and regularized directional impact $D_b$), `pivot_returns.py` (merges CRSP open/close daily prices into fast lookup tables).
- **ML & Backtesting**: `train_model_zoo.py` (Optuna parameter tuning on isolated `TRAIN` sets), `online_sgd_backtest.py` (Stochastic Gradient Descent online walk-forward prediction on `OOS` sets).
- **Strategy Execution**: `regime_classifier.py` (K-means rolling beta for sign flipping), `transaction_cost_grid.py` (Almgren-Chriss square-root impact slippage simulation).
- **Statistics**: `multiple_testing_correction.py` (Deflated Sharpe, block bootstraps, Lo 2002 autocorrelation SE), `panel_regression.py` (Fama-MacBeth HAC cross-sectional analysis), `ablation_study.py` (permutation importance).
- **Aggregation**: `aggregate_results.py` (rolls up all 500 tickers into a final summary).

---

## 3. End-to-End Pipeline Execution

The pipeline is driven sequentially to prevent data leakage between `TRAIN` and `OOS` (Out-of-Sample) universes.

1. **HPC Data Phase**: `bash hoffman2/master_orchestrator.sh`
   - Extracts all bursts for all 500 tickers using the C++ parser.
2. **Permanence**: `python src_py/compute_permanence.py`
   - Attaches forward-looking target labels (like next day's open price) to the burst CSVs.
3. **Aggregation**: `python src_py/aggregate_results.py`
   - Concatenates the raw results into master panels.
4. **Optuna Tuning (TRAIN ONLY)**: `run_pipeline.sh --phase optuna`
   - Bayesian search over the C++ physical parameters on 40 strictly in-sample tickers.
5. **Backtest (OOS ONLY)**: `run_pipeline.sh --phase backtest`
   - Evaluates the strategy on the remaining ~460 unseen tickers using the tuned physical parameters.
6. **Research & Statistics**: `run_pipeline.sh --phase research`
   - Executes Fama-MacBeth, transaction cost grids, and ablation studies.

---

## 4. Parameter Definitions & Anti-Bias Rules

### C++ Physical Parameters (Optuna Tuned)
Instead of arbitrary burst definitions, the C++ engine isolates bursts using parameters discovered by Optuna:
- **`-v` (Volume Fraction)**: Minimum fractional ADV (Average Daily Volume) required to trigger a burst.
- **`-d` (Direction Threshold)**: The threshold for directional consistency within the burst.
- **`-r` (Volume Ratio)**: The ratio of volume required to maintain the burst state.
- **`-H` (Hawkes Decay $\beta$)**: Note: This is *fixed* to 1.0 to prevent overfitting and is *not* tuned by Optuna.

### The $\kappa$ (Kappa) Firewall (Look-Ahead Bias Prevention)
$\kappa$ is the threshold for minimum directional price impact ($D_b$).
- **Rule**: The C++ engine MUST extract all bursts unconditionally (`-k 0`).
- **Rule**: In `online_sgd_backtest.py`, the $\kappa$ filter is applied *strictly to the trailing training window* during burn-in and daily model updates. It is NEVER applied to the current day's prediction candidates, ensuring the model blindly evaluates all unseen bursts without peeking at their future price impact.

---

## 5. Machine Learning & Trading Logic

**The Model**: An online Stochastic Gradient Descent (SGD) classifier/regressor.
- **Features**: Microstructure metrics recorded precisely when the burst terminates (duration, intensity, orderbook imbalance).
- **Target**: Inverse hyperbolic sine (arcsinh) transformed Volume-Weighted Permanence. Specifically `CLOP` (Close-to-Open overnight return).
- **Walk-Forward**: The model trains on $T-60$ to $T-1$, predicting day $T$. The feature `StandardScaler` is strictly fit on $T-1$ data.

**Strategy Execution**:
1. **Entry Rule**: Compute the 75th percentile prediction score using the *training set only*. If day $T$'s burst scores above this static threshold, execute a trade in the burst's direction.
2. **Regime Rule**: `regime_classifier.py` checks the stock's rolling 60-day beta. If the stock is in a "Mean-Reverting" microstructural regime, the strategy automatically flips the direction of the trade (shorts a buy burst).
3. **Holding Rule**: Enter at burst execution price, hold overnight, exit at the next morning's `Open`.
4. **Transaction Costs**: Apply a base 1.0 bps round-trip friction, plus Almgren-Chriss square-root impact slippage based on empirical participation rates.

---

## 6. Statistical Robustness Framework

To prove the alpha is not a statistical artifact or p-hacked, the pipeline outputs:
- **Fama-MacBeth Panel Regressions**: Cross-sectional analysis of all 500 stocks with Newey-West HAC standard errors and FF5+MOM+UMD risk factor controls.
- **Deflated Sharpe Ratio (DSR)**: Haircuts the final Sharpe based on the number of Optuna trials evaluated.
- **Lo (2002) SE**: Corrects Sharpe standard errors for daily return autocorrelation.
- **Ablation Studies**: Uses Permutation Importance to prove complex ML features add value over a simple "Direction-Only" baseline.
