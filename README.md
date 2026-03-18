

# Burst Detection and Analysis Pipeline

This repository contains a full C++ and Python pipeline to identify informed trading activity from high-frequency limit order book (LOBSTER) data and evaluate long-horizon price permanence. It systematically distinguishes between temporary mechanical price impact and permanent informational price discovery.

## Data Requirements

1. **LOBSTER Intraday Data:** A folder containing one message CSV per trading day (e.g., `TSLA_2026-01-02_message_0.csv`). No orderbook.csv is needed. The C++ engine dynamically reconstructs the top-of-book from the message events.
2. **CRSP Daily Data:** Yearly folders containing gzipped CSVs of daily open and close prices. These are required to compute overnight permanence targets (Perm_CLOP and Perm_CLCL).

---

## Pipeline Execution Steps

### Step 1: Generate CRSP Price Pivots

Before calculating overnight permanence, you must aggregate the raw CRSP daily files into unified pivot tables.

```shell
python src_py/pivot_returns.py yearly/
```

This script scans the `yearly/` directory for subfolders (e.g., 2023, 2024), extracts the open and close prices, and generates `open_all.csv`, `close_all.csv`, `pvCLCL_all.csv`, and `OPCL_all.csv` in the root directory. These matrices allow O(1) lookups for next-day prices when evaluating bursts that span multiple years.

### Step 2: Parse Messages and Extract Bursts (C++ Engine)

Compile the C++ engine:

```shell
make clean && make
```

Run the processor:

```shell
./data_processor data/TSLA_folder/ results/bursts_TSLA.csv -s 0.5 -v 100 -d 0.8 -k 0 -t 10.0 -j 8
```

The engine scans chronological order submissions and groups them into discrete bursts based on temporal proximity and directional consistency. It simultaneously tracks the Best Bid and Best Ask to record market state features. To avoid look-ahead bias in the machine learning phase, ensure you pass `-k 0` (kappa = 0) so the engine outputs all bursts regardless of their future 10-minute decay. The Db filtering rule will be applied dynamically in Python.

### Step 3: Calculate Permanence

```shell
python src_py/compute_permanence.py results/bursts_TSLA.csv open_all.csv close_all.csv --kappa 0
```

This script calculates the Transformed Volume-Weighted Permanence for each burst across multiple horizons (1m, 3m, 5m, 10m, tCLOSE, CLOP, CLCL). It applies the inverse hyperbolic sine (arcsinh) transformation to safely compress extreme outlier events into a stable regression target while preserving directional sign. It also computes the short-horizon decay measure (Db) representing the average mark-to-market directional price movement over minute-level horizons. Keep `--kappa 0` here to maintain the full unfiltered dataset for evaluation.

### Step 4: Machine Learning Evaluation

```shell
python src_py/regression_eval.py --ticker TSLA --data results/bursts_TSLA_filtered.csv --config s0p5_v100_d0p8 --kappa 0.2
```

This script maps early-time microstructure features to long-horizon persistence using an XGBoost regression model. It utilizes a strictly chronological, out-of-sample walk-forward validation scheme to prevent future data from leaking into the training set.

Crucially, it handles target horizons dynamically:

- For intraday targets (Perm_t1m through Perm_t10m), it trains on all bursts and strictly excludes Db from the feature set to prevent look-ahead bias.
- For long horizons (Perm_tCLOSE, Perm_CLOP, Perm_CLCL), it safely applies the Db >= kappa condition to filter the datasets before training.

To simulate a trading strategy without temporal leakage, the script establishes a 75th percentile entry threshold strictly from the training set predictions and blindly applies that static threshold to the unseen test data.

---

## Core File Manifest

- **main.cpp:** Entry point for the C++ parser. Handles multithreading, day-file discovery, and rolling market state snapshots.
- **burst.cpp / burst.h:** Implements the deterministic clustering algorithm to group order submissions into discrete, directional blocks of trading intent.
- **orderbook.cpp / orderbook.h:** Rebuilds the limit order book state from raw LOBSTER messages (Submission, Cancel, Deletion, Execution).
- **pivot_returns.py:** Aggregates yearly CRSP files into fast-lookup pivot tables for overnight target variables.
- **compute_permanence.py:** Calculates the arcsinh permanence regression targets and the Db short-horizon decay metric.
- **regression_eval.py:** Runs the walk-forward XGBoost evaluation, separating intraday from long-horizon logic to ensure strictly ex-ante predictions.
- **train_model_zoo.py:** A larger benchmarking suite containing roughly 20 ML algorithms (LightGBM, Random Forest, Ridge, etc.) for broader model exploration.
- **regress.sh:** The grid engine submission script that orchestrates the C++, permanence calculation, and regression pipeline across multiple tickers and configurations.
