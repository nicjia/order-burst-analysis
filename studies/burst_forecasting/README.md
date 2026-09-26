# burst_forecasting — do order bursts forecast prices?

Started 2026-09-24 as a leak-free rebuild of the MATH 279 idea (detect a burst → predict whether its price impact is
permanent → trade). Since the supervisor meeting and the 2026-09-25 instruction, the question is **forecasting
power only**: how well information available the moment a burst is known forecasts the mid-quote from then on
(seconds to the close, the next open and the next close), raw and market-excess. No fills, no spread-crossing, no
costs, no strategy P&L — where an order is filled is a separate problem, and a passive fill could even earn the spread.

## Layout

| path | what |
|---|---|
| `docs/FORECASTING_TEST_LIST.md` | the full list of tests with status and every result table (the running log) |
| `docs/IDEAS_BACKLOG.md` | ~500 untested / tested ideas: definitions, decision timing, features, targets, models, data |
| `docs/BURST279_RESULTS.md` | first write-up of the leak-free 279 model (10-minute decisions) |
| `docs/CONFIRMATION_2025.md` | the pre-registered 2025 check and its result |
| `code/` | every script (below) |
| `cluster/` | Hoffman2 job scripts (copies deployed next to the p4 code, `results/p4_revisit_v1/code/`) |
| `results/burst_forecasting/` (repo root, **gitignored**) | every output, one folder per pass |

Code imports the shared project modules from `src_py/` through a two-line path shim. Data never goes in this folder:
the repo is public, and all outputs stay under the gitignored `results/`.

## Protocol

- **Decision time = the moment the burst is known.** A run with gap G is known to have ended at its last trade + G;
  decisions are taken 0.1 s after that (or at the k-th child for "act during the burst"). Every feature uses data up
  to the decision only. (The first pass used the P4 rule, 10 minutes after the burst began; that discards everything
  that plays out in seconds to minutes and is kept only as a record.)
- **Targets:** the signed mid move from the decision (sign = burst side) to +1 s … +60 min, to the close, next open,
  next close; from 30 minutes on, market-excess against an equal-weight minute index of ~500 stocks.
- **Split:** train on 56 stocks in 2022–23; test on 56 *different* stocks in 2024, and on all 96 still-listed stocks
  in 2025 (never used for any choice). Gradient boosting with fixed hyperparameters (depth 3, 200 iterations).
- **Metrics:** daily rank IC (Spearman across that day's events) with Newey–West(10) t; paired IC gain of the burst
  features over controls + order book; out-of-sample R²; hit rate; top-minus-bottom decile of the realized mid move.
- **Placebo:** a pseudo event at a random time in 10:00–15:30 with the same duration and side for every sampled burst.

## Scripts

| pass | extractor (cluster) | analysis (local) | output folder |
|---|---|---|---|
| 279 rebuild (10-min decisions) | p4 per-burst files | `burst279_model.py`, `burst279_grid.py`, `burst279_daily.py`, `burst279_oracle.py`, `forecast_eval.py`, `forecast_eval_daily.py` | `burst279_v1/` |
| definitions from p4 files | `burst_defs_extract.py` | `burst_defs_model.py` | `burst_defs_v1/` |
| raw v1 (7 definitions, book) | `burst_defs_raw.py` | `burst_defs_raw_model.py` | `burst_defs_raw/` |
| raw v2 (real-time, 13 definitions) | `burst_defs_raw2.py` | `burst_defs_raw2_model.py`, `burst_defs_raw2_vsbook.py` | `burst_defs_raw2/` |
| placebo | `burst_pseudo_raw.py` | `burst_pseudo_model.py` | `burst_pseudo/` |
| raw v3 (19 more definitions, act during the burst, latency) | `burst_defs_raw3.py` | `burst_defs_raw3_model.py`, `forecast_power.py`, `confirm_2025.py` | `burst_defs_raw3/` |
| **raw v4 (forecasting products)** | `burst_defs_raw4.py` | `burst_defs_raw4_model.py` | `burst_defs_raw4/` |
| market index | `market_minute_index.py` | — | `market_index/` |
| random-time reversal | `random_time_reversal.py` | — | `random_rev/` |
| ETF → constituents | `etf_bursts_extract.py`, `etf_lead.py` | — | (cluster) |
| ClusterLOB replication | `clusterlob_extract.py` | `clusterlob_fit_eval.py` | `clusterlob/` |
| trading evaluations (superseded) | — | `burst_trading_eval_v2.py`, `burst_trading_eval_v3.py`, `burst279_oracle2.py` | — |

Terms: a **run** is consecutive same-side trades (marketable-order packets), broken by any opposite-side trade or by a
gap longer than G; a **stream** follows each side separately and ignores the other side's trades in between.

## Forecasting results

1. **Burst features forecast the next seconds to minutes, well beyond the order book.** Deciding the moment a short
   run is known (v2/v3, 2024 test stocks): IC 0.20–0.23 at +10 s, 0.16–0.19 at +60 s, 0.08–0.09 at +5 min,
   0.05 at +30 min (market-excess). Over controls plus queue imbalance, quote OFI, trade-flow imbalance, microprice and
   touch cancellations, the burst features add +0.05 to +0.11 IC at 10 s (t 16–33), +0.06 at 60 s, +0.03 at 5 min,
   about +0.01 at 30 min (t ≈ 3) and +0.005 to the close (t ≈ 2–3). Acting at the 5th child is the best rule.
2. **It is about bursts, not any recent move.** Matched random-time events forecast half as well (run 0.5 s:
   IC 0.185 vs 0.108 at 10 s; 0.053 vs 0.017 at 30 min; 0.038 vs 0.018 to the close; every gap t > 4). The burst's
   own push reverts (its move's IC −0.10 at 10 s), random windows' do not (−0.003).
3. **It replicates out of time.** 2025, all stocks, same models: IC ≈ 0.20 / 0.15 / 0.08 / 0.045 / 0.03–0.04 at
   10 s / 60 s / 5 min / 30 min / close, definition by definition.
4. **Long gaps kill it** (60 s runs ≈ 0), regularity-defined bursts (timers, TWAP-like) forecast worst, and the
   forecast decays fast with latency (10-s IC 0.21 at the decision, 0.15 one second later, 0.06 ten seconds later).
5. **To the close the forecast is mostly intraday reversal**, which bursts mark (reversal IC −0.045 to −0.058 at
   burst times vs −0.021 at random times of the same hours); the burst characteristics add little beyond the move
   since the open. **Close-to-close (next day) was not forecastable** in the 10-minute-decision pass; re-tested in v4.
6. **Cross-stock:** market-wide burst flow continues into the next minute; SPY/QQQ-specific burst flow reverses in
   the constituents (t −2.8 to −10). ClusterLOB's cluster-OFI result did not replicate under our implementation.

7. **What carries it** (drop-one feature-group ablation, `code/forecast_power.py`): at +10 s the order book and the
   burst's own price path (≈ 0.03–0.04 IC each), at 1–5 minutes the price path, from 30 minutes to the close the
   controls (reversal). **The burst's size structure and the fingerprint regularities (modal / non-round clip, size
   CV, inter-arrival CV, timer phase) add 0.000 IC at every horizon** — they identify algorithms, not price moves.
8. **Stable across stocks:** positive IC in 99–100% of the 96 stocks in 2025 at 10–60 s, 83–99% at 5 min, 73–80% to
   the close; twice as strong in wide-spread stocks (tick-constrained mids rarely move within seconds).
9. **The information is local to each burst and does not aggregate** (v4): signed burst imbalance summed into
   5-minute bins adds +0.001 to +0.002 IC to next-bin cross-sectional forecasts (base 0.028: past returns, all-trade
   flow, quote OFI) and nothing to the rest of the day; cross-stock burst imbalance does not time the market (OOS
   R² ≤ 0.0003); the day's burst imbalance does not forecast the overnight, next-day or next-5-day return. The
   overnight return does reverse the day's move (rank-linear model IC 0.032–0.035, t 2.5–2.7), bursts or not;
   close-to-close is not forecastable.

10. **The full term structure** (v4, 2025, act-at-5th-child; figure `results/burst_forecasting/burst_defs_raw4/fig_term_structure.png`):
    IC 0.18 at +1 s, peaking at 0.20 at +5–10 s, 0.155 at +1 min, 0.09 at +5 min, 0.05 at +30 min, 0.04 to the
    close, 0.03 to the next open, 0.01 to the next close. Out-of-sample R² peaks at 6.4% (+10 s). The direction of the
    next mid change is called right 59–60% of the time (AUC 0.63) vs 55% at random times. Real bursts beat
    random-time events 2–4.5× at every intraday horizon; the burst features' gain over controls + book is t > 20
    through +1 min, t ≥ 3 at +30 min, not reliably significant to the close, and zero overnight.
11. **Mostly transient impact** (`code/burst_defs_raw4_mechanical.py`): the burst features' gain over the order book is
    +0.10 IC at 10 s when the burst widened the spread, ≈ +0.01 when the spread was unchanged or one tick, and ≤ 0 when
    the mid did not move — the book relaxing after the burst displaced it. On average the mid keeps drifting
    +1.0–1.3 bps in the burst's direction over the next minutes (0 at random times); bursts that pushed harder give
    back more.
12. **v5 on 676 stocks (the three checks, `code/burst_defs_raw5*.py`; figure `burst_defs_raw5/fig_v5_placebo.png`):
    the short-horizon forecast belongs to aggressive orders, not to bursts — one isolated marketable order is as
    forecastable as a burst on the mid at every horizon (2025, 10 s: 0.195 vs 0.20; random times 0.13); bursts keep a
    small edge on the far quote in the first seconds and a larger drift per unit of volume. The far-side quote and the
    microprice are more forecastable than the mid (0.30 vs 0.20 at 10 s). Against a burst-blind baseline with multi-window
    returns, flows, OFI and a 5-level book rebuilt from messages, the event's own features add +0.02 IC at 1 s-1 min and
    nothing from 30 min on. Robust to dropping the top-10 stocks, stock-clustered bootstrap, time of day; weaker at
    one-tick spreads. Aggregates (5-minute, daily, market) remain proxies for total order flow.
13. **Volatility:** after controlling for pre-event realized variance (IC 0.65–0.88 on its own), burst features add
    ≤ 0.008 IC; bursts mark a modest volatility rise (+8–14% log RV over the next 5 minutes vs −2% at random times).

## Superseded: trading framing

Earlier passes also scored fill-based strategies (cross the spread at the decision, exit at the close). Those numbers
are in `docs/FORECASTING_TEST_LIST.md`, `docs/BURST279_RESULTS.md` and `docs/CONFIRMATION_2025.md` for the record; they
are no longer a criterion, because any assumed fill (and crossing in particular) is arbitrary.
