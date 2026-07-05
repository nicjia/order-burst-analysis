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

---
---

# PART II — HANDOFF & CURRENT STATE (2026-07)

> **Read this before continuing `corrections.md`.** Sections 1–6 above describe the *originally intended* pipeline. Several of its promises — Deflated Sharpe, the Direction-only ablation, Almgren–Chriss costs, the Poisson null, FF5+MOM controls — were **never actually delivered in the paper**. That gap is precisely what the referee report targets. This part is the ground truth for continuing the corrections. **Handoff package = this README + `corrections.md` + `FINDINGS_LOG.md` + the referee report + the codebase.**

## 7. Canonical documents & where results live
- **`FINDINGS_LOG.md`** — the authoritative record of every experiment, number, and verdict (§§0–5). **Read first.**
- **`corrections.md`** — the referee report as a 26-item tracker (12 Major P1/P2 + 14 Minor + process notes + resubmission path + triage). This is the work queue.
- **`BACKFILL_PLAN_2017_2021.md`** — scoped plan for the decisive statistical-power upgrade (NOT yet executed; data still populating on lobster2).
- **`main.tex` / `main.pdf`** — the paper (24 pp; compiles clean). Key `\label`s: `sec:markout` (D_b look-ahead), `sec:breadth` (full-universe null + loss decomposition), `sec:reversion` (tick-constrained reversal), `sec:reconstruction` (Hawkes grid + 4 burst defs + hidden footprint). Abstract + these four sections carry the honest results; §§4–7 (Empirical Results etc.) still contain the stale optimistic framing the referee wants rewritten (M1).
- **`results/summary/`** — consolidated CSVs + `KEY_RESULTS.md` index: `sgd_backtest_summary.csv` (438 OOS names: Sharpe/trades/DSR/Lo-SE), `research_{ablation,poisson,markout,timeofday}_summary.csv` (stale subset), `tc_grid_all.csv`.
- **`results/research/`** — canonical panels: `coi_panel_{gated,ungated}_2026.csv`, `markout_panel_2026.csv`, `intraday_{unfiltered,filtered}_2026_daily.csv`, `reversion_sweep.csv`, `name_relspread_bps.csv`, `true_adv_daily.csv`; + aggregate logs (`panel_full`, `multiple_testing_correction`, `regime_full`, …).
- **Burst data (full 2022–2026, 482 names):** `results/bursts_<T>_baseline_{unfiltered,filtered,adv}.csv`.
  - `_unfiltered` = **master** (all bursts + permanence labels). Use this.
  - `_baseline` = raw C++ output (redundant subset of `_unfiltered`, kept per user).
  - `_filtered` = D_b≥0.5 gated — **LOOK-AHEAD for intraday markout; do not use for intraday tests** (see traps).
  - `_adv` = per-ticker daily traded volume (feeds `true_adv_daily.csv`).
- **Per-name backtest outputs:** `results/sgd_backtests_oos/<T>_reg_clop_b1p0_i0p5{.log,_debug_trades.csv}`. The **`_debug_trades.csv`** (daily rows: `day, side, flow_signal, pred, net_raw, gross_raw`) drive **all** reversion/attribution analyses.
- **FF factors:** `data_factors/` (Ken French FF5 + Momentum daily, already fetched).
- **Prices:** `open_all.csv` / `close_all.csv` — date-indexed, ticker columns, **2016-01-04 → 2026** (so no price backfill needed for 2017–2021). `*_pre2025.csv` = pre-2025-swap backups. Source = Massive/Polygon REST (`polygon_fetch_prices.py`); **adjustment status UNVERIFIED — this is referee M8**.

## 8. Infrastructure & how to run
- **Hoffman2 cluster**, project dir `/u/scratch/n/nicjia/order-burst-analysis` (a git clone of this repo).
- **SSH quirk:** the `hoff`/`h2` aliases carry a RemoteCommand and reject extra commands. Use the full host + heredoc:
  `ssh -o BatchMode=yes hoffman2.idre.ucla.edu 'bash -ls' <<'EOF' … EOF`
- **Per-session env:** `. /u/local/Modules/default/init/bash; module load gcc/11.3.0 python/3.9.6; source .venv/bin/activate` (gcc is REQUIRED — pandas needs its libstdc++). Set `export OMP_NUM_THREADS=1`.
- **Jobs:** `qsub -l h_data=8G,h_rt=Nh -pe shared N -cwd -o logs/x.out -e logs/x.err -N name driver.sh`. Heavy/long work MUST be `qsub` (login-node kills long processes; `nohup` does not survive dropped SSH). SSH drops frequently — prefer qsub + poll a log file.
- **Sync:** `scp file hoffman2.idre.ucla.edu:/u/scratch/n/nicjia/order-burst-analysis/path`. **This local agent env CANNOT push to GitHub; the Hoffman clone CAN git push/pull.** Edit locally → scp (files stay byte-identical).
- **Compile paper (cluster):** `module load texlive; pdflatex -interaction=nonstopmode main.tex; bibtex main; pdflatex ×2`.
- **Raw LOBSTER:** `nicjia@lobster2.math.ucla.edu:/lobster/YEAR/YYYYMMDD/TICKER.7z` — 2017–2026, **message-only** (C++ / `burst_alt.py` reconstruct the book), ~30–46 MB/ticker-day. `data/<ticker>` folders were DELETED for quota → re-fetch from lobster2 to reprocess. **2017–2021 present but STILL POPULATING** (AAPL absent as of 2026-07-04). Templates for scoped re-fetch: `hoffman2/hidden_2023.sh`, `hoffman2/alt_validate.sh`.

## 9. Current honest state (ground truth the corrections must preserve)
- **Bursts are real** structures; **BUT no burst definition yields tradable overnight alpha.**
- Single-name flagships (NVDA/TSLA…) are **in-sample** (TRAIN set) and their intraday markouts were **inflated by D_b look-ahead**.
- Full 482-name universe (2022–2026): cross-sectional COI **null** (Fama-MacBeth t≈−0.6 ungated / −1.4 gated); per-name overnight SGD strategy **mean Sharpe −0.28**; loss decomposes to **~62% incidental short-market beta + residual edge ≈0**.
- **One conditional positive:** dollar-neutral **reversal in tick-constrained (low-price) names**, 20-day hold — **full-sample Sharpe 1.48 (t=2.96) but walk-forward OOS Sharpe ≈0.81 (t=1.40), NOT significant, front-loaded** (2023 +1.97, 2024 +0.50). FF5+MOM α on the *full-period* series = +6 bps/day (t=3.11) — **but that's in-sample (referee M6)**. DSR-z ≈ +0.02 (marginal). Mechanism (tick-regime asymmetry, monotone across price & spread) is the durable part.
- **Alternative burst defs (`burst_alt.py`):** Hawkes/OFI/Book-Resilience ≈0 at 3-min; only **hidden-execution** bursts have a real footprint (day-clustered t≈6–9) — but **sub-spread, decays by 30 min, null overnight** (n=2 names, referee M12).

## 10. Critical gotchas / traps (re-learn before touching anything)
- **`D_b` IS the forward markout** (`¼·Σ Direction×(Mid_τ − M_end)`, τ∈{1,5,10}m). Gating on `D_b ≥ κ` is **circular look-ahead** at any intraday horizon. `_filtered` files are D_b-gated. (Referee **M3**: D_b is ALSO a prediction-time feature but is realized 10 min post-burst = *after* the 3:50/3:55 pm MOC cutoff → the stated MOC execution is infeasible; must drop D_b or window features to the cutoff and re-run.)
- **LOBSTER sign conventions:** `Direction` = side of the **resting** order (execution against the bid = a market **SELL**; aggressor sign = **−Direction**). **Type-5 (hidden) `Direction` is uniformly +1** (side undisclosed) → sign hidden trades by **Lee-Ready** (exec price vs prevailing mid). Bears on **M10** (81% net-short audit) and **M12**.
- **Pseudo-replication:** per-burst t-stats/p-values treat ~10⁵ bursts as independent; **effective N = number of days**. Date-cluster every per-burst statistic (referee **M5**).
- **In-sample stars:** NVDA, TSLA, JPM, MS, **LLY** are all in `universes/train_50.txt` → flagship results are in-sample; the paper mislabels LLY "OOS" (**M2**). The 438 OOS names are everything else.
- **Universal median params** (used for the breadth run): `results/optuna_regression/universal_median_params.json` → vol_frac≈0.00197, dir_thresh≈0.763, vol_ratio≈0.280, kappa≈1.085.
- **The tick-constrained reversal signal = the SGD daily `flow_signal`** (from `_debug_trades.csv`), z-scored across names per day; reversal = short high-flow, dollar-neutral, 20-day overlapping hold. Reproduced by `src_py/reversion_sweep.py` / `reversion_walkforward.py`.

## 11. `corrections.md` → where to fix (code/data pointers)
| Item | Where to fix | Effort |
|---|---|---|
| **M1** rewrite §§4–7 as diagnostic; **M2** LLY-OOS mislabel + captions; **m1–m14** | `main.tex` (+ `universes/train_50.txt` to confirm TRAIN membership) | paper edits |
| **M5** DSR + date-clustered inference | `reversion_sweep.py` (DSR-z already there); re-cluster per-burst stats; SGD logs have Lo-SE/DSR per name | cheap, existing data |
| **M6** OOS factor alpha | run FF regression on the **walk-forward OOS** return series from `reversion_walkforward.py` × `data_factors/` | cheap |
| **M7** plain-reversal baseline + Direction-only ablation | reversion machinery: swap `flow_signal` for lagged returns (`close_all`) and plain signed volume (burst files); Direction-only via `ablation_study.py` / SGD feature subset | cheap, existing data |
| **M4** close-mid target | `compute_permanence.py` (VSI target measured from burst-start mid → recompute from close mid); recompute Table 8 ρ | moderate (re-permanence) |
| **M10** sign-convention audit | quick script: per-name corr(daily signed flow, same-day return) from `_debug_trades.csv` × `close_all`; disclose the panel sign-flip (`panel_regression.py --regime-csv`) | cheap |
| **M3** drop D_b feature / MOC-cutoff window + re-run | `online_sgd_backtest.py` (`get_features`), `compute_permanence.py` (D_b def) | moderate, re-run |
| **M8** price source/splits/delistings/costs | `polygon_fetch_prices.py` (adjustment?), `open_all/close_all`, `universes/full_500.txt` (as-of/survivorship), spread costs via `name_relspread_bps.csv` | moderate |
| **M9** NASDAQ-venue coverage | `true_adv_daily.csv` (burst ADV vs consolidated); add coverage statement | moderate |
| **M11** Poisson test (DELETE or build), define `cb` | no test exists — build Hawkes-vs-inhomogeneous-Poisson KS, or delete the claim; `cb` buffer undefined in `online_sgd_backtest.py` | build / edit |
| **M12** hidden cross-section + midpoint sensitivity | `src_py/{burst_alt,hidden_full}.py`; needs lobster2 re-fetch (use `hoffman2/hidden_2023.sh` as template) | needs re-fetch |
| **decisive** 2017–2021 backfill (powers M6) | `BACKFILL_PLAN_2017_2021.md` — gated on lobster2 population | data-gated |

**Recommended start:** M7 (plain-reversal baseline + Direction-only ablation) + M5/M6 (honest inference) — all cheap, use data on hand, and together decide whether the paper has a real positive result before investing in the backfill.
