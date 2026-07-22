# Results Provenance Ledger

Purpose: for every quantitative result in `main.tex`, show the **code**, the
**data**, and the **saved output** that produced it, so a reviewer can trace (and
re-run) each number. This is a verifiable ledger, not a claim of infallibility —
scope, subsets, and the one failed side-experiment are stated honestly at the end.

All artifacts live on Hoffman2 at `/u/scratch/n/nicjia/order-burst-analysis`
(reachable via `ssh hoff`); logs are under `results/research/`.

---

## 1. Per-stock data coverage (proof the pipeline ran on all names)

| Artifact | Count | What it proves |
|---|---|---|
| `results/sgd_backtests_oos/*_reg_clop_b1p0_i0p5_debug_trades.csv` | **438 files** | daily burst-flow signal computed for every OOS name |
| `results/hidden_xsec/out/*.csv` | **483 files** | hidden-execution footprint processed per name |
| `results/research/hidden_xsec_daily.csv` | 22 MB (~221k ticker-days) | the full hidden cross-section (474 names, 2023–24) |
| `close_all.csv` price panel | 5,960 names × 2,632 days (2016–2026) | daily prices for returns/gaps |
| COI panel ticker list (in `panel_gated_2026.log`) | ~460 names enumerated | the Fama–MacBeth panel ran name-by-name over the universe |

The 438 signal files and 483 hidden files are the direct evidence that the
pipeline executed on the **full universe**, not a hand-picked few.

---

## 2. Result → code → data → output (freshly verified against saved logs)

Each row below was cross-checked this session by reading the saved log on the
cluster; the "log value" column is copied from that log and matches `main.tex`.

| main.tex result | code | log (`results/research/`) | log value (= main.tex) |
|---|---|---|---|
| Hidden 3-min footprint (Tab. hidden_xsec: +1.62, t=3.38, 79%) | `hidden_full.py` + agg | `hidden_xsec_daily.csv` (data) | +1.62 bps, pooled t=3.38 |
| **Intraday term structure** (Tab. hidden_term) | `hidden_term.py`, `agg_term.py` | `term_structure.log` | 3m +1.90 (t 9.4) … close +2.44 (t 6.3) |
| **R3 day-level gradient** (§10.1, t=−1.6) | `referee_hardening.py` | `referee_hardening_v2.log` | slope −1.83, NW t=−1.57; skip-day −1.99/−1.69; ortho −2.22/−2.01 |
| R3 naive HC1 (disclosed as overstated) | `referee_hardening.py` | `referee_hardening_v2.log` | −2.06, t=−3.81 (n=434) |
| Momentum in large-tick (§10.1, no continuation) | `referee_hardening.py` | `referee_hardening_R3R5R9R11.log` | Sharpe −0.51, t=−0.87 |
| **ST_Rev factor control** (Tab. ff_alpha) | `st_rev_regression.py` | `st_rev_regression.log` | α full +5.94 (t 3.08), OOS +6.25 (t 1.41); ST_Rev loading ~0 |
| **Borrow-cost sensitivity** (§10.5) | `borrow_costs.py` | `borrow_costs.log` | OOS 0.79→0.77→0.71→0.66→0.52→0.25 (0–20%) |
| Overnight COI panel + sign-flip (Tab. coi_panel) | `panel_regression.py` | `panel_gated_2026.log` | ~460-name FM panel; 111,657 stock-days inverted for 116 names |
| Intensity→vol lead (Conclusion) | `next_directions.py` | `next_directions.log` | +28.2 bps t=8.6; incremental +15.8 t=6.9 |
| Campaign reversion (Conclusion) | `next_directions.py` | `next_directions.log` | −7.45 bps/day, t=−4.07 |
| Cross-asset / short-tilt asymmetry (this round's leads) | `other_signals.py` | `other_signals.log` | sell-campaign −11.4, t=−3.14 |

---

## 3. Reproduce any row

```bash
ssh hoff
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash && module load gcc/11.3.0 python/3.9.6 && source .venv/bin/activate
python src_py/referee_hardening.py     # R3 (naive + day-level + skip-day), R5/R9
python src_py/st_rev_regression.py     # ST_Rev factor control
python src_py/borrow_costs.py          # borrow-fee sensitivity
python src_py/next_directions.py       # intensity->vol, campaigns
python src_py/r2_reversal.py           # reversal Sharpe/t, baselines, DSR, overnight panel
# LaTeX: module load texlive; pdflatex/bibtex x3 main.tex  -> 34 pp, 0 undefined refs
```

---

## 4. Honest scope — what this ledger does NOT claim

- **Subsets, clearly labelled in the paper.** The term structure is a 48-name /
  2023 sample; the drift placebo a 39-name subsample; R3 covers the 434 names with
  a valid per-name estimate. These are stated as such in `main.tex`, not passed off
  as the full universe.
- **Pre-existing pipeline outputs consumed, not re-derived here.** The 438 signal
  files and the 474-name hidden run were produced by the C++/SGD pipeline in prior
  sessions; the analyses above consume them. Their provenance is the file counts in
  §1, not a fresh re-execution of the raw-message reconstruction.
- **One failed side-experiment, NOT in `main.tex`.** The 2019–2021 NYSE historical
  probe (`hist_flow.py`/`hist_test_local.py`) was exploratory. Its 2019 slice ran
  clean (reversal fails OOS, sell-campaign reversion replicates at t=−2.24); its
  **2020 slice is corrupted** — ~82% of rows are failed rsync pulls (VPN dropped mid-
  extraction), so 2020 is discarded and nothing from this probe is in the paper.
- **Not independently re-implemented.** The code is the author's; this ledger shows
  internal consistency (same harness → the paper's §10 numbers), not third-party
  replication.
