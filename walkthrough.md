# Walkthrough — 500-Ticker HPC Refactor & Referee-Mandate Upgrade

This document maps every change made in this revamp to the referee mandates it
satisfies, then gives the **single-ticker smoke-test** procedure to run before
launching the full 500-ticker Hoffman2 job, plus the automated validation tests.

Reports referenced: `referee_report_bursts_claude.pdf` (issues **M1–M10**) and
`referee_report_addendum.pdf` (reframes **R1–R6**, borrowable elements **B1–B12**).

---

## 1. Referee-mandate traceability

| Mandate | Where addressed | Status |
|---|---|---|
| **M1** true walk-forward | Structural TRAIN/OOS firewall in `run_pipeline.sh` (`resolve_tickers`); 2019–2021 reframed as robustness probe. Per-year expanding-window re-fit is documented as the next step. | Partial |
| **M2** Direction dominance | `src_py/ablation_study.py`: adds **Direction-only** single-feature model + **permutation importance** (Spearman scorer); reports % of full-model ρ recovered by Direction alone. | ✓ |
| **M3** Sharpe inference | `online_sgd_backtest.py`: Lo (2002) SE + 95% CI + Deflated Sharpe (units-corrected). `multiple_testing_correction.py --pnl-csv`: Lo SE, Deflated Sharpe, circular block-bootstrap CI. | ✓ |
| **M4** multiple testing | `multiple_testing_correction.py`: Bonferroni / Holm / BH-FDR + Harvey-Liu-Zhu haircut + Deflated Sharpe. | ✓ |
| **M5** buried buy-and-hold / alpha | `src_py/beta_hedged_markout.py` (PnL vs stock/SPY/FF5+MOM, NW HAC); `panel_regression.py` long-short FF5+MOM+UMD α. | ✓ |
| **M6** transaction-cost realism | `transaction_cost_grid.py`: **Almgren–Chriss** square-root impact by participation rate + breakeven participation. | ✓ |
| **M7** microstructure baselines | `poisson_baseline_test.py`, `naive_baseline_markout.py`, and the unconditional-COI sort in `panel_regression.py`. | ✓ |
| **M8** Hawkes (α,β) pinned | `HAWKES_BETA` / `TRIGGER_INTENSITY` are parameterized in `run_pipeline.sh`; an (α,β) sensitivity sweep is documented as future work. | Partial |
| **M9** universe too small | `universes/full_500.txt` + the Hoffman2 pipeline + `aggregate_results.py` cross-sectional IC distribution. | ✓ |
| **M10** D_b ratio stability | `compute_permanence.py`: ε-regularized PeakImpact denominator (`clip(lower=1e-4)`). | ✓ |
| **R1** markout panel | `naive_baseline_markout.py` multi-horizon markouts (gross bps). | Partial |
| **R2** PPT in bps | `transaction_cost_grid.py` reports gross/net bps per trade with t-stats. | ✓ |
| **R3** sign-conditional flip | `regime_classifier.py` (K-means regime + `FlipSign`) → consumed by `panel_regression.py --regime-csv` and `aggregate_results.py`. | ✓ |
| **R4** tmrwOPCL horizons | Permanence targets `Perm_CLOP` / `Perm_CLCL` / `Perm_tCLOSE`. Finer OPCL split documented. | Partial |
| **R5** gross-of-cost default | `transaction_cost_grid.py` reports gross first, then the cost grid. | ✓ |
| **R6** cross-sectional breadth | `full_500.txt` + `aggregate_results.py` (IC distribution + Grinold–Kahn IR proxy). | ✓ |
| **B1** 457/500-name universe | `universes/full_500.txt`; cluster manifest scan in `hoffman2/master_orchestrator.sh`. | ✓ |
| **B2** Poisson null model | `poisson_baseline_test.py`. | ✓ |
| **B3** COI framework | `panel_regression.compute_daily_coi`. | ✓ |
| **B4** panel reg + FF + NW | `panel_regression.py`: `linearmodels.FamaMacBeth` (with hand-rolled NW fallback). | ✓ |
| **B5** single/double sorts | `panel_regression.py` quintile single-sort; double-sort documented. | Partial |
| **B6** long-short risk-adj α | `panel_regression.factor_adjust_long_short` (FF5+MOM+UMD, NW). | ✓ |
| **B7** benchmark suite | `naive_baseline_markout.py` + unconditional-COI sort. | Partial |
| **B8** TC grid in bps | `transaction_cost_grid.py` flat 0–5 bps grid. | ✓ |
| **B9** time-of-day | `time_of_day_analysis.py`. | ✓ |
| **B10–B11** robustness / count-COI | Documented; ADV-window and count-based COI are follow-ups. | Partial |
| **B12** literature | Paper-side (`main.tex` / `references.bib`); out of code scope. | N/A |

---

## 2. Architecture recap

- **HPC data build** (per-ticker SGE array): `hoffman2/setup_hoffman2.sh` →
  `hoffman2/master_orchestrator.sh` (DTN, tmux) → `hoffman2/sge_compute_worker.sh`
  (compute nodes: rsync `.7z` from lobster2 → extract → C++ `data_processor` → permanence → cleanup).
- **C++ engine**: `make` (local) or `make hoffman2` (loads `gcc/11.3.0`). Parser now uses a
  fast allocation-free integer scan; ADV pre-compute is parallelized across the worker's threads.
- **Evaluation suite** (`src_py/`): permanence → Optuna (TRAIN) → SGD backtest (OOS) →
  research scripts → `aggregate_results.py` cross-sectional rollup.
- **Pipeline driver**: `run_pipeline.sh --phase {data|hpc-data|perm|optuna|backtest|research|aggregate|all|hpc-all}`.

---

## 3. Single-ticker (NVDA) smoke test

Run this end-to-end on **one** ticker before submitting the 500-ticker array.

### 3a. On Hoffman2 (cluster)
```bash
ssh hoff                       # or: ssh h2   (user nicjia)
cd /u/scratch/n/nicjia/order-burst-analysis

# One-time environment + build
bash hoffman2/setup_hoffman2.sh          # venv, pip deps, make, verify 7z

# Single-ticker manifest, then submit a 1-task array
printf "NVDA\n" > hoffman2/current_batch.txt
# (master_orchestrator.sh drives rsync+qsub for a real batch; for the smoke
#  test you can run the worker body interactively on a compute node via qrsh.)
qrsh -l h_data=8G,h_rt=02:00:00,highp -pe shared 1
#   inside the interactive node:
SGE_TASK_ID=1 bash hoffman2/sge_compute_worker.sh
```

### 3b. Locally (if NVDA day-files exist under `data/NVDA/`)
```bash
source .venv/bin/activate
make                                            # build data_processor

ROOT=$(pwd) OOS_START=2023-01-01 OOS_END=2024-12-31 \
  ./run_pipeline.sh --phase data   --oos-file <(printf "NVDA\n") --train-file <(printf "TSLA\n")
./run_pipeline.sh --phase perm     --oos-file <(printf "NVDA\n") --train-file <(printf "TSLA\n")
```

Then exercise each new analysis on the produced CSVs:
```bash
U=results/bursts_NVDA_baseline_unfiltered.csv

# M2 — Direction-only ablation + permutation importance
python3 src_py/ablation_study.py "$U" --ticker NVDA --target reg_clop

# M6 — Almgren–Chriss participation grid (empirical vol from close_all.csv)
python3 src_py/transaction_cost_grid.py "$U" --ticker NVDA --close-csv close_all.csv

# M3/M4 — realized-PnL Sharpe inference on a debug-trades CSV
python3 src_py/multiple_testing_correction.py --pnl-csv results/sgd_backtests_oos/NVDA_reg_clop_b1p0_i0p3_debug_trades.csv --n-trials 100

# R3 — regime classification, then panel regression consuming it
python3 src_py/regime_classifier.py --burst-dir results/ --close-csv close_all.csv --tickers NVDA,JPM,MS,SPY
python3 src_py/panel_regression.py --burst-dir results/ --tickers NVDA,JPM,MS,SPY \
    --open-csv open_all.csv --close-csv close_all.csv \
    --regime-csv results/regime/regime_classifications.csv

# R6/M9 — cross-sectional aggregation
python3 src_py/aggregate_results.py --results-dir results/ --universe-file universes/full_500.txt \
    --open-csv open_all.csv --close-csv close_all.csv --run-panel-regression
```

---

## 4. Automated validation tests

1. **Build sanity** — `make clean && make` must succeed; `make hoffman2` on the cluster.
2. **Python import/compile** — `python3 -m py_compile src_py/*.py` and
   `python3 -c "import ablation_study, multiple_testing_correction, panel_regression, transaction_cost_grid, aggregate_results"`
   (must succeed even without `linearmodels`/`statsmodels` — the graceful-fallback path).
3. **Shell syntax** — `bash -n run_pipeline.sh hoffman2/*.sh`.
4. **Output validation** — each `bursts_<T>_baseline.csv` has the expected header and a
   non-zero row count; `aggregate_results.py` prints a coverage table of found/missing tickers.
5. **ADV sanity check** — the C++ side-output `min_vol` (per `[start ...]` log line) should match
   `compute_trailing_adv(...) * vol_frac` from the Python side for the same ticker/day.
6. **Kappa-firewall test** — confirm `online_sgd_backtest.py` loads bursts with κ=0 and applies the κ
   filter to the **training window only** (search the run log for
   `Applying Grand Universal geometry structural rules (kappa=0 for OOS integrity)` and
   `Training kappa filter:`). Pre-filtering the whole dataset by κ must produce different results
   than the training-only filter — that difference is the look-ahead bias the referee (M3) flagged.

---

## 5. Open items requiring user input (from prompt.md)

- **Full 500 list**: `universes/full_500.txt` currently holds ~128 names (train ∪ OOS).
  The orchestrator materializes the remaining names from `lobster2:/lobster/manifest.csv`; append
  them to the file to pin a reproducible universe.
- **Fama-French factors**: `run_pipeline.sh` expects `data/ff5_mom_daily.csv` (env `FACTOR_CSV`).
  When absent, FF5+MOM adjustment is skipped with a NOTE (download from Ken French's library).
- **CRSP price matrices**: `open_all.csv` / `close_all.csv` are assumed to cover the full universe.
