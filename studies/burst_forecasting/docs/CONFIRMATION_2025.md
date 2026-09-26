# Pre-registered confirmation on 2025 (written 2026-09-25, before any 2025 v3 data was extracted)

**Rule (fixed).** v3 extractor `code/burst_defs_raw3.py` unchanged. Model: gradient boosting (depth 3, 250 iterations,
learning rate 0.05, min leaf 200) on every v3 feature (controls, book, burst), target = market-excess mid move from
the decision to the close, trained **only on the train stocks in 2022–23** (the same model as in the 2024 analysis).
Trade: at the burst's decision time, when |prediction| ≥ the training 80th percentile of |prediction|, take the side
sign(prediction), exit at the closing mid. Cost: half the quoted spread at the decision + 1 bp. Daily equal-weight
P&L, Newey–West(10).

**Sample.** All 112 stocks of the raw passes (56 train + 56 test), every available 2025 trading day — unseen in time
by every model and every selection made so far.

**Primary cells (gate: net > 0 with t > 2 in 2025, each):**
1. level-clearing runs, spread ≤ 5 bps (2024: +6.3 bps, t 3.08)
2. runs with a 10 ms gap, spread ≤ 5 bps (2024: +6.0, t 2.55)
3. confirmed-decision Hawkes, spread 5–10 bps (2024: +7.2, t 3.86)

**Reported without selection:** every other v3 definition in the ≤ 5 and 5–10 bps buckets and overall; the mid P&L
(forecast) at every horizon.

If none of the three passes, the after-cost candidate is recorded as not confirmed.

---

## Result (read once, 2026-09-25)

`code/confirm_2025.py` on 22,784 stock-days of 2025 (96 of the 112 stocks still trading; jobs 14911940/41).

| primary cell | 2024 (selection) | 2025 | gate |
|---|---|---|---|
| level-clearing runs, spread ≤ 5 bps | +6.30 bps, t 3.08 | **+4.54 bps, t 3.60** (4,450 trades) | **pass** |
| 10 ms runs, spread ≤ 5 bps | +6.03, t 2.55 | +3.32, t 1.83 (6,838) | fail (t < 2) |
| confirmed Hawkes, spread 5–10 bps | +7.15, t 3.86 | +2.98, t 1.07 (7,216) | fail |

**One of three pre-registered cells passes.** Reported without selection: in the ≤ 5 bps bucket the 2025 net is
positive for 19 of 20 definitions (timer chains are the exception); cells with t ≥ 2 include confirmed Hawkes
+6.83 (t 4.20), act-at-5th-child +5.36 (t 2.59), hidden-heavy +6.80 (t 2.18), cancellation bursts +3.77 (t 2.10),
absorbed +3.24 (t 2.08), side-specific Hawkes +3.46 (t 2.07), run 0.1 s +2.77 (t 2.06), tolerant runs +3.22
(t 2.03), run 0.05 s +4.75 (t 2.02). Across all spreads the trade still loses (the 5–10 bps bucket is mostly
insignificant).

**Forecasting at the mid replicates in 2025** — IC at +10 s ≈ 0.20, +60 s ≈ 0.15, +5 min ≈ 0.08, +30 min
market-excess ≈ 0.045, close ≈ 0.03–0.04, matching 2024 definition by definition.

**Caveats before anything is called a strategy.** The P&L is market-excess, so a tradable version needs a hedge
(e.g. SPY) whose cost is not included; each stock can carry several overlapping positions to the close; entry
assumes crossing half the quoted spread at the decision and exit at the closing mid plus 1 bp; capacity and
queue/impact of our own orders are not modelled.
