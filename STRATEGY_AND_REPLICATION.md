# Strategy & Replication Guide

What the trading actually is, how it is formulated, and exactly what someone must do
to reproduce every backtest number in the paper.

This supersedes the trading-related parts of `METHODS_AND_REPRODUCIBILITY.md`, whose
headline markout figures (+1.6 bps, t = 3.4) predate the outlier rule described in §6.

---

## 1. The idea in one paragraph

Institutions do not execute a large order at once. They split it into a sequence of
same-side child orders, which arrive clustered in time. If we can detect those
clusters — "order-submission bursts" — from raw exchange messages, the *side* of a
burst tells us the direction of an institutional intention, and the question is
whether that intention carries information about future price, and for how long.
That is the term structure of price impact: at what horizon does the burst's
directional content show up in price, and does it stay there or come back?

The answer we measure: for **concealed** (hidden / iceberg) executions, the impact
is **permanent** — about +2 bps, fully impounded within ~3 minutes, then flat for the
rest of the session, with no reversion at any horizon. That is the adverse-selection
component of the bid-ask spread, and it is *smaller than the spread* (4–12 bps), so
it cannot be crossed profitably. For **aggressive** (visible marketable) flow, the
daily signal is essentially noise.

---

## 2. The signal

One number per name per day, from the raw LOBSTER message file.

For each execution message (type 4 = visible, type 5 = hidden) inside regular trading
hours, assign an aggressor sign:

- **Type 4:** `sign = -Direction`. LOBSTER's `Direction` is the *resting* side, so an
  execution against a resting bid is a market **sell**.
- **Type 5:** `Direction` is uniformly +1 and carries no information. Sign by the
  **Lee–Ready** rule (execution price vs. prevailing mid). The 48.9% of hidden prints
  that execute exactly at the mid are dropped in the headline; signing them by tick
  rule instead changes the magnitude by ~3x (see §7).

Then per name-day:

```
netflow  = Σ sign_j × size_j          # net signed share volume  (buy - sell)
buy      = Σ size_j  over sign > 0
sell     = Σ size_j  over sign < 0
n_bursts = count of same-sign runs with inter-trade gap < 1s and length >= 3
```

Reference implementation: [`src_py/hist_flow.py`](src_py/hist_flow.py) (46 lines — no
book reconstruction, no ML; the paper's own baselines show the deployable content is
the sign of net flow).

**Known issue:** `RTH1 = 57600.0` (16:00:00), so flow in the 15:50–16:00 window is
included, while the body pipeline enforces a 15:50 dead-zone. Re-running truncated at
15:50 is outstanding.

---

## 3. The strategy (the paper's headline backtest)

A dollar-neutral cross-sectional reversal, restricted to tick-constrained names.

| Step | Definition |
|---|---|
| Universe | the **100 lowest-priced** names each day (tick-constrained subset) |
| Signal | `z[i,t]` = cross-sectional z-score of `netflow`, clipped to [-4, 4], computed **within the subset** |
| Raw position | `p[i,t] = -sign(z[i,t])` — fade the flow |
| Traded weight | `w[i,t] = mean(p[i, t-1 .. t-20])` — trailing 20-day mean, strictly past |
| Neutralize | demean `w` cross-sectionally each day (dollar-neutral) |
| Normalize | scale so `Σ|w| = 1` (unit gross) |
| Return | close-to-close, `Σ w[i,t] × r[i,t+1]` |
| Cost | 1 bp per side on turnover `Σ|w[i,t] - w[i,t-1]|` |

Turnover is ~0.058/day (≈15x/year) because of the overlapping 20-day hold — this is
why the result survives realistic spread costs.

**The economic bet:** aggressive burst flow in cheap, tick-constrained names pushes
price beyond fair value (the inside queue is a binding constraint, so aggressive flow
overshoots); liquidity providers absorb it and the overshoot reverts over the
following weeks.

### Session variant (Section 12)

Identical weight construction, but instead of close-to-close the return is split:

```
overnight  =  Open[t+1] / Close[t]   - 1     # position formed at close t
intraday   =  Close[t+1] / Open[t+1] - 1
close-close = Close[t+1] / Close[t]  - 1     # = the compounding of the two
```

Requires daily **opening** prices, which the Polygon feed we use does not supply;
Section 12 uses Yahoo Finance adjusted bars. This is load-bearing — the whole P&L of
the overnight leg *is* the close-to-open segment.

---

## 4. Data pipeline

```
lobster2 archive (.7z per ticker-day)
   │   rsync from a Hoffman2 compute node (lobster2 is not reachable directly)
   ▼
7z x  →  {ticker}_{date}_..._message_10.csv
   │   src_py/hist_flow.py   (one row per ticker-day)
   ▼
results/hist_flow/rows/{TICKER}/{date}.row
   │   concatenate
   ▼
all_rows.csv        ticker,date,netflow,n_bursts,buy,sell
   │   + close_all.csv (Polygon daily closes)   + opens.parquet (Yahoo, session tests only)
   ▼
backtest
```

**MISSING handling is critical.** A failed rsync, an empty archive, or a pre-IPO /
post-delisting date writes the literal token `MISSING` in all four numeric fields —
*not* zero. A genuine zero (file present, fewer than 10 trades) is emitted normally.
Conflating the two silently corrupted an earlier 2020 run: dropped VPN connections
wrote zeros that were indistinguishable from real no-flow days. Every downstream
loader coerces `MISSING → NaN` and drops it, while keeping genuine zeros.

---

## 5. Exact commands

### Extraction (cluster; the expensive step)

```bash
# on Hoffman2, from /u/scratch/n/nicjia/order-burst-analysis
qsub -t 1-580 -tc 40 hoffman2/hist_flow.sh      # one array task per ticker
# consolidate
python3 -c "import glob,pandas as pd; \
  pd.concat([pd.read_csv(f) for f in glob.glob('results/hist_flow/out/*.csv')]) \
    .to_csv('results/hist_flow/all_rows.csv', index=False, header=False)"
```

### Backtests (local; all take seconds)

```bash
python3 src_py/hist_test_2017.py          # reversal + DLRET toggle + campaigns
python3 src_py/overnight_reversal.py      # session decomposition, Sample A
python3 src_py/tugofwar_2023.py           # Sample B (hidden, 2023-24)
python3 src_py/tugofwar_2022_2026.py      # Sample C (primary 2022-26) - the decisive null
python3 src_py/term_structure_clean.py    # Table 18 under the outlier rule
python3 src_py/hidden_audit_and_lead.py   # markout outlier audit + jump concentration
python3 src_py/single_stock_explore.py    # per-stock time-series test
python3 src_py/fig_tugofwar.py            # figures/fig_tugofwar.pdf
```

### Data-quality gates (run before trusting any panel)

```bash
python3 src_py/missing_audit.py results/hist_flow/all_rows.csv   # MISSING classification
python3 src_py/data_quality.py                                    # PIT universe, volume checks
```

---

## 6. Runtime

Measured on the 493-name × 1,257-day 2017–2021 panel (540,788 name-days):

| Stage | Time |
|---|---|
| Extraction — 580 tickers × ~1,250 days, SGE array `-tc 40` | **~4.1 hours** |
| Parse `all_rows.csv` (726k rows) | 3.5 s |
| Load + align prices | 2.7 s |
| **The backtest itself (all 493 names)** | **0.10 s** |
| **Total, end-to-end from CSV to Sharpe** | **~7.4 s** |

The backtest is free; essentially all wall-clock cost is the one-time extraction, and
that is dominated by network transfer from lobster2, not computation. Once
`all_rows.csv` exists, a full parameter sweep over hundreds of configurations is a
matter of minutes.

---

## 7. What the results actually are

### Real
- **Hidden-execution permanent footprint.** +2.00 bps at 3 min, flat to +2.06 at 30
  min, placebo-netted +2.08 to the close. 474 names, 2023–2024. Under the outlier rule
  in §6 below. This is the paper's result.
- **A well-powered null catalogue.** The overnight relation, the cross-sectional COI
  panel, the passive-burst control, Hawkes parameter insensitivity.

### Not tradeable (each tested and rejected)
| Candidate | Verdict |
|---|---|
| Tick-constrained reversal | OOS Sharpe ≈ 0.8, fails deflation (DSR 0.70) |
| Daily flow reversal | P&L corr **0.997** with plain price short-term reversal — redundant |
| Sell-campaign reversion | market beta; dollar-neutral it collapses to the same 0.45 |
| Intensity → volatility | subsumed by trailing realized vol; sign flips across specs |
| Overnight continuation, aggressive | confined to 2020–21; **−0.26 (t=−0.67)** on the primary 2022–26 panel |
| Overnight continuation, hidden | 606 name-days (0.26%) carry **89%** of P&L — a jump bet, not a signal |
| Per-stock time-series | first-half/second-half Sharpe rank corr **+0.10**; top-50 → +0.03 OOS |

### Two data rules that must be applied
1. **Outlier rule.** Drop any name-day whose markout exceeds 1,000 bps at any horizon
   (56 of 221,261; one reaches −114,627 bps = −1146%). They cluster at short horizons
   and were deflating the 3- and 15-minute t-statistics specifically.
2. **MISSING ≠ 0.** See §4.

### Open
- Classifier sensitivity: +2.0 bps (drop at-mid) vs +0.6 bps (tick-sign) — a 3x range
  on the headline magnitude. Needs EMO/CLNV + SIP NBBO. **Most important remaining run.**
- 15:50 truncation and the 15:55 → 09:35 buffer test (both need LOBSTER intraday).

---

## 8. Minimum path to reproduce the headline

1. Get LOBSTER access; stage the 2023–2024 message + orderbook files for the 474-name
   universe (`universes/full_500.txt`).
2. Extract hidden (type-5) executions, sign by Lee–Ready against the prevailing mid,
   drop at-mid prints.
3. Cluster into bursts (same-sign, gap < 1s, length ≥ 3).
4. Compute directional markouts at 3/15/30 min from the **burst-termination mid**
   (not the burst-start mid — that conflates the prediction with the burst's own impact).
5. Apply the 1,000 bps outlier rule.
6. Average to name-day, then to daily means; Newey–West (10 lags) on the day series.

Expected: +2.00 bps at 3 minutes, flat thereafter, significant in ~79% of names.
