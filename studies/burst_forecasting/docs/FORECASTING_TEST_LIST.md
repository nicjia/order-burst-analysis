# Burst forecasting — the full test list (2026-09-24)

Framing from the supervisor meeting: **forecasting power is the objective**, not net P&L. Targets are
decision-time-to-close (mid to close) and close-to-close, in **market-excess** terms; spread crossing is ignored;
a 3-year window is enough. Every test keeps the leak-free protocol: features known at the decision time, train
2017–19 (one third of names), evaluate 2020–21 and 2022–25 (the other two thirds of names). Metrics: daily rank
IC (Newey–West t), out-of-sample R² against a zero forecast, sign hit rate, and the gross mid-to-mid long–short
decile spread. Status: ✅ done · 🔄 running · ⬜ queued.

## A. Targets (what is forecast)

| # | target | measured from → to | status |
|---|---|---|---|
| A1 | t→close, raw | mid at T_dec → closing mid (per burst) | ✅ IC +0.032–0.039 (t 6–7) |
| A2 | t→close, **market excess** | minus the equal-weight intraday market over the same interval | ✅ IC +0.040 (t 14.5) — all from non-burst predictors |
| A3 | real-time t→close | first minute after the burst ends → close (4 burst definitions) | ✅ AUC 0.510–0.517 |
| A4 | burst → +5 / +30 / +60 min | same, intraday | ✅ AUC up to 0.558 (merge 30 min, 5 min) |
| A5 | close→next close, raw | per burst and per name-day | ✅ burst AUC 0.504; daily books null |
| A6 | close→next close, **market excess** | CRSP return − value-weighted universe | ✅ null: every set IC ≈ 0 |
| A7 | close→next open, 15:30→close | per name-day | ✅ null / sign-unstable |
| A8 | permanence from burst start (the 279 label) | φ at close / next open / next close | ✅ AUC 0.64 / 0.59 / 0.55 — overlaps the observed move |

## B. Event definitions (what a "burst" is)

| # | definition | status |
|---|---|---|
| B1 | trade bursts: run of ≥ 3 same-side economic packets, gaps ≤ 60 s | ✅ |
| B2 | submission bursts: runs of qualifying adds at or inside the touch | ✅ |
| B3 | merged runs, same side, gaps ≤ 5 min / ≤ 30 min | ✅ |
| B4 | fingerprint chains: same side + same repeated clip within 30 min | ✅ |
| B5 | sub-populations: ≥ 5 / ≥ 10 children, fast ≤ 5 s, slow ≥ 60 s, large, program-like, linked, multi-day, hidden-heavy, tight-spread | ✅ |
| B6 | alternative gaps 1 s / 5 s / 300 s and minimum sizes 5 / 10 packets from raw data | ✅ raw pass: no definition adds to the close forecast |
| B7 | Hawkes-intensity bursts (the original C++ detector, corrected signs) | ✅ raw pass: best at 30 min (+0.006 IC over controls, t 3.0) |
| B8 | side-only 2–5 s streams (tied with runs in fingerprint-v1) | ✅ raw pass: no gain |
| B9 | random-time pseudo-bursts (placebo event) | ✅ in P4; ⬜ in this forecasting grid |

## C. Predictors

| # | group | contents | status |
|---|---|---|---|
| C1 | known non-burst predictors | move since the open (intraday reversal), 30-min pre-move, time of day, spread | ✅ carries the whole t→close forecast |
| C2 | burst impact path | peak impact, displacement at 1 and 10 min, mean displacement, retained ratio | ✅ alone IC +0.012 (t 10), adds 0 over C1 |
| C3 | burst structure | size/ADV, children, duration | 🔄 |
| C4 | fingerprint | modal-clip share, non-round clip | 🔄 |
| C5 | program score | inter-arrival regularity, book imbalance, execution depth | 🔄 |
| C6 | linkage | same-side same-size bursts in the previous 30 min | 🔄 |
| C7 | multi-day | yesterday's repeated-clip programs, same and opposite side | 🔄 |
| C8 | liquidity type | hidden share, truncated share | 🔄 |
| C9 | order-book outcomes (submission bursts) | executed / cancelled share by T_dec | ✅ no gain |
| C10 | cross-name flow | peer burst flow in the same minute | ✅ predicts next minute (t 16); ⬜ at t→close |
| C11 | market and sector state | market return so far today, sector flow | ⬜ |
| C12 | order-flow imbalance around the burst (raw book) | touch depth imbalance, OFI 60 s before and during | ✅ adds nothing in any definition (|t| < 1.9) |

## D. Models and aggregation

| # | item | status |
|---|---|---|
| D1 | gradient boosting (depth 3, fixed) | ✅ |
| D2 | logistic / ridge baselines | ✅ / 🔄 |
| D3 | nested models: C1 → C1+C2 → C1+C2+C3–C8 (incremental forecasting power of bursts) | ✅ bursts add 0 (t −1.0 to +0.5) |
| D4 | per burst vs per name-day (once per day) | ✅ both |
| D5 | Fama–MacBeth with controls at the name-day level | ✅ in P4 (null) |
| D6 | per-name vs pooled models | ⬜ |

## E. Robustness for anything that forecasts

| # | check | status |
|---|---|---|
| E1 | market-excess vs raw | ✅ excess strengthens the reversal forecast |
| E2 | incremental over known predictors (C1) | ✅ none |
| E3 | by year, by listing exchange, by size tercile | ⬜ |
| E4 | placebo (labels shuffled within day; random-time events) | ✅ for AUC |
| E5 | multiple-testing count vs chance | ✅ trading grid |
| E6 | third, untouched period (2012–16) for any single named candidate | ⬜ |

## Results so far (2026-09-24)

- **The setup is coherent (oracle).** A perfect model trading every burst at T_dec earns +65–79 bps per trade
  net at the close (86–94% winners); a noisy model breaks even at AUC ≈ 0.53. Ours reaches 0.511–0.515.
- **Decision-to-close is forecastable — but not by bursts.** Market-excess IC +0.038 (t 8.0) in 2020–21 and
  +0.040 (t 14.5) in 2022–25 from the move since the open, the 30-minute pre-move, time of day and spread alone
  (intraday reversal); the gross top-minus-bottom decile is +12 to +13 bps. Adding the burst's impact path and
  every burst-structure feature changes the IC by +0.0002 (t 0.5) and −0.0005 (t −1.0).
- **Close-to-close (next day) is not forecastable** by anything tested: controls, total order-flow imbalance
  or burst flow, raw or market-excess (all |IC| < 0.005 in 2022–25).
- **Why the earlier "positive" trading numbers existed:** they were this intraday reversal, visible gross and
  consumed by the half-spread.

## Raw pass: seven burst definitions with order-book features (2026-09-24)

`studies/burst_forecasting/code/burst_defs_raw.py` on 35,778 stock-days (2.97 M sampled bursts). Three-year window:
train 56 stocks in 2022–23, test 56 different stocks in 2024. Target: market-excess move from T_dec.

| definition | to close: controls-only IC | bursts + book add | 30 min: controls-only IC | bursts + book add |
|---|---|---|---|---|
| hawkes (original detector) | +0.040 (t 7.1) | +0.0014 (t 0.7) | +0.020 (t 4.9) | **+0.0064 (t 3.0)** |
| run1 | +0.028 (t 4.4) | −0.0011 (t −0.4) | +0.016 (t 4.5) | +0.0030 (t 1.1) |
| run5 | +0.029 (t 4.9) | +0.0007 (t 0.2) | +0.017 (t 4.4) | +0.0043 (t 1.5) |
| run60 | +0.029 (t 4.4) | +0.0010 (t 0.4) | +0.017 (t 5.0) | +0.0062 (t 1.7) |
| run300 | +0.028 (t 5.2) | −0.0003 (t −0.1) | +0.015 (t 4.6) | +0.0057 (t 2.0) |
| stream2 | +0.030 (t 5.0) | −0.0006 (t −0.2) | +0.019 (t 5.0) | −0.0002 (t −0.1) |
| stream5 | +0.025 (t 4.0) | +0.0032 (t 1.2) | +0.014 (t 4.2) | −0.0022 (t −0.7) |

Order-book features on top of everything else: −0.0038 to +0.0024, no |t| above 1.9.

**Bursts mark moments of stronger reversal.** Rank correlation of the market-excess move since the open with
the market-excess move to the close, same 56 test stocks, 2024: **−0.045 to −0.058 at burst decision times**
(t −5.9 to −8.3) against −0.021 to −0.023 at random times re-weighted to the same hours of the day. The event
matters; the burst's characteristics do not add once the move since the open is known.

## Raw pass v2: real-time decisions (2026-09-25) — the first burst-specific forecasting result

`code/burst_defs_raw2.py` (jobs 14910710/11, concat 14910716): 36,141 stock-days, 4.4 M sampled bursts, 13
definitions. Decision the moment a burst is confirmed over; mid-to-mid; train 56 stocks 2022–23, test 56
different stocks 2024. `code/burst_defs_raw2_model.py`, `code/burst_defs_raw2_vsbook.py`.

**Burst features add far beyond the order-book state at short horizons.** IC gained by burst price path,
structure and regularity over non-burst controls **plus** queue imbalance, quote OFI and trade-flow imbalance:

| definition | +10 s | +60 s | +5 min | +30 min (mkt-excess) | close (mkt-excess) |
|---|---|---|---|---|---|
| run 0.1 s | +0.109 (t 26.2) | +0.067 (t 17.9) | +0.035 (t 12.1) | +0.011 (t 3.5) | +0.006 (t 2.8) |
| run 0.5 s | +0.094 (t 32.7) | +0.061 (t 17.0) | +0.031 (t 8.6) | +0.014 (t 3.6) | +0.005 (t 2.0) |
| run 1 s | +0.087 (t 25.0) | +0.061 (t 16.6) | +0.035 (t 12.3) | +0.010 (t 3.6) | +0.002 (t 1.0) |
| stream 0.5 s | +0.098 (t 30.1) | +0.057 (t 16.4) | +0.028 (t 8.7) | +0.009 (t 4.2) | 0.000 |
| clip chain 5 s | +0.021 (t 5.9) | +0.022 (t 4.3) | +0.011 (t 2.1) | +0.001 | +0.004 |
| run 60 s | −0.001 | +0.005 | +0.001 | +0.003 | +0.002 |

Total IC with every feature: +0.20 at 10 s, +0.16 at 60 s, +0.08 at 5 min for run 0.1 s; top-minus-bottom decile
+3.9 / +6.5 / +7.4 bps at the mid. Almost all of the gain is the burst's own price path; structure and
regularity add +0.001 to +0.005; the order book adds +0.006 to +0.017 at 10 s on top of the burst features.

- **Long gaps destroy the signal** (run 60 s ≈ 0): the supervisor's and the user's suspicion was right.
- **30-minute buckets** of all burst flow add nothing (next bucket: burst adds −0.002, t −0.8; to close +0.001).
- **Hawkes caveat.** Its decision time t_e + 1.3 s can precede the moment its cluster is confirmed (up to
  ln(λ/0.3) s after the last trade), so its short-horizon numbers are excluded until re-run.
- **Open question — placebo** (`code/burst_pseudo_raw.py`, running): does a matched random-time event with the
  same duration and side forecast as well? If yes, this is generic high-frequency mean reversion, not bursts.

## Placebo: is the real-time forecast about bursts? (2026-09-25) — yes

`code/burst_pseudo_raw.py` (jobs 14911651/52/53) + `code/burst_pseudo_model.py`. For each sampled run burst, a
pseudo event at a uniformly random time in [10:00, 15:30] the same day with the same duration and side; identical
features (controls, window move, 60-s pre-move, duration, queue imbalance, OFI, trade-flow imbalance) and outcomes.
2.14 M events; train 56 stocks 2022–23, test 56 different stocks 2024.

| run 0.5 s | real bursts (model trained on real) | pseudo events (model trained on pseudo) | real − pseudo |
|---|---|---|---|
| +10 s | IC 0.185 (t 64) | 0.108 (t 38) | **+0.078 (t 22.6)** |
| +60 s | 0.155 | 0.079 | +0.076 (t 20.2) |
| +5 min | 0.083 | 0.031 | +0.052 (t 14.8) |
| +30 min, market-excess | 0.053 | 0.017 | +0.036 (t 10.1) |
| close, market-excess | 0.038 | 0.018 | +0.020 (t 4.4) |

Run 1 s is the same. The window's own move forecasts the next 10 s at real bursts (IC −0.097 — the burst's push
reverts) and not at pseudo windows (−0.003). Decile spreads at the mid are 2–3.5× larger at bursts. The burst model
applied to pseudo events scores only 0.029 at 10 s. **Bursts are moments of unusually forecastable prices.**

## Trading ideas T1.1 / T1.2 / T2.1 / T3.1 at the mid (2026-09-25)

`code/burst_trading_eval_v2.py`. Trade the top 20% most confident forecasts in the predicted direction, test stocks
2024. Mid P&L per trade is positive in all 45 short-gap definition × horizon cells (t 5–42); e.g. run 0.1 s: +2.3 bps
at 10 s, **+3.7 bps at 60 s (hit 60%)**, +3.8 at 5 min, +4.9 at 30 min, +7.3 to the close. Crossing the spread loses
everywhere at 10 s – 30 min: the confident trades sit in wide-spread stocks (mean round trip ≈ 17 bps), and even
2–5 bps names fall short (60 s: +3.1 mid vs ~4 cost). **To the close in stocks with spreads ≤ 10 bps** the net is
mostly positive: run 0.5 s +3.9 / +2.9 / +4.1 bps (spread 0–2 / 2–5 / 5–10 bps; t up to 2.0) — post-hoc bucket
selection, to be confirmed on another period.

## T2.3 / T3.2: reversal to the close only at burst moments (2026-09-25)

Position −sign(market-excess move since the open), exit at the close, mid, test stocks 2024:

| events | all | largest 20% of moves since the open |
|---|---|---|
| random times (same stocks) | +1.2 bps (t 1.2) | +2.9 bps (t 1.4) |
| run 0.5 s bursts | +3.1 (t 3.6) | **+6.8 (t 4.5)** |
| stream 0.5 s bursts | +2.8 (t 3.6) | +5.9 (t 5.2) |
| clip chains 5 s | +2.8 (t 2.5) | +7.8 (t 3.8) |
| run 60 s | +2.5 (t 2.9) | +5.7 (t 4.0) |

Intraday reversal is only significant as a trade when it is triggered by a burst.

## Raw pass v3: 19 more definitions, deciding during the burst, latency, continuation (2026-09-25)

`code/burst_defs_raw3.py` (jobs 14911744/45/46), 5.28 M events; `code/burst_defs_raw3_model.py`. The book baseline
now also includes the microprice gap and touch cancellations, so it is stronger than in v2 (IC 0.15–0.18 at 10 s).

| definition (best first) | +10 s IC | gain over controls + book | +60 s IC | +5 min IC | +30 min (mkt-ex) IC | gain at 30 min |
|---|---|---|---|---|---|---|
| **early5** (act at the 5th child) | **0.230** | +0.051 (t 15.8) | **0.187** | **0.094** | **0.055** | +0.009 (t 3.3) |
| **early3** (act at the 3rd child) | 0.221 | +0.058 (t 19.6) | 0.172 | 0.089 | 0.048 | +0.008 (t 3.1) |
| run 10 ms | 0.223 | +0.055 (t 18.5) | 0.173 | 0.084 | 0.052 | +0.008 (t 2.2) |
| run 25 ms | 0.220 | +0.052 (t 18.6) | 0.168 | 0.085 | 0.048 | +0.009 (t 3.4) |
| run 0.1 s | 0.225 | +0.050 (t 18.5) | 0.175 | 0.084 | 0.051 | +0.003 |
| adaptive gap (2 × median IAT) | 0.217 | +0.046 (t 20.9) | 0.164 | 0.085 | 0.048 | +0.004 |
| level-clearing runs | 0.218 | +0.045 (t 16.9) | 0.164 | 0.082 | 0.042 | +0.001 |
| cancellation bursts | 0.192 | +0.031 (t 14.2) | 0.142 | 0.062 | 0.034 | +0.002 |
| sweeps | 0.182 | +0.031 (t 14.6) | 0.143 | 0.070 | 0.043 | +0.002 |
| Hawkes, confirmed decision | 0.167 | +0.024 (t 11.5) | 0.133 | 0.069 | 0.037 | +0.007 (t 2.7) |
| hidden-heavy runs | 0.172 | +0.014 (t 4.3) | 0.132 | 0.069 | 0.038 | +0.011 (t 2.5) |
| timer / phase / TWAP-like | 0.05–0.13 | ≈ 0 | — | — | — | ≈ 0 |

- **Deciding during the burst is the best rule**, and very short gaps (10–50 ms) match or beat 0.1 s.
- **Regularity-defined bursts forecast worst**: the most fingerprintable algorithms are not the forecastable ones.
- **Latency**: IC at 10 s falls from ~0.21 at the decision to ~0.15 one second later and ~0.06 ten seconds later.
- **Continuation**: "another burst within 60 s" has AUC 0.70–0.87, but opposite-side continuation is predicted as
  well, so it is mostly activity; "the run grows past its 3rd child" has AUC 0.745.
- Hawkes with the confirmed decision time is still positive but weaker than short runs (the v2 leak inflated it).

## ETF -> constituent lead (T1.10 / B11.3 / G4, 2026-09-25)

`code/etf_bursts_extract.py` (SPY + QQQ, 1,506 ETF-days, job 14911894) + `code/etf_lead.py` (job 14911900): every
TEST-cell stock's next-minute mid return on its own burst flow, other stocks' burst flow, SPY and QQQ burst flow,
its own lagged return and other stocks' same-minute return (stale-price control); stock-day demeaned,
date-clustered.

| coefficient | 2022–23 (498 days) | 2024 (251 days) |
|---|---|---|
| own burst flow | +115 (t 51.6) | +111 (t 7.7) |
| other stocks' burst flow | +1748 (t 33.2) | +949 (t 21.4) |
| **SPY burst flow** | **−108 (t −8.6)** | **−27 (t −2.8)** |
| **QQQ burst flow** | **−117 (t −10.1)** | **−50 (t −3.3)** |

Market-wide burst flow continues into the next minute; the ETF-specific part, holding it fixed, reverses in the
constituents — consistent with ETF-arbitrage price pressure that unwinds. Same sign in both periods, weaker in
2024. ETF and market flow are highly collinear, so the split is a partial effect, not a clean decomposition.

## ClusterLOB replication (T2.11 / B8.1 / F11, 2026-09-25)

`code/clusterlob_extract.py` (book rebuilt from messages — the archives hold no depth files), 21 NASDAQ-traded stocks
in three tick-size groups (7 each), 10,182 stock-days; K-means (k = 3) on 3.04 M sampled 2023 events
(`code/clusterlob_fit_eval.py`); clusters labelled on 2023 by return correlations as in the paper; 2024 is the test.

- Clusters are behaviourally distinct: fast activity at existing levels right after mid changes (labelled
  directional), orders opening new price levels (market-making), slow activity at old, deep levels (opportunistic).
- **2024 test: cluster OFI does not beat plain OFI.** Plain size OFI → next bucket +0.40 bps per trade, t 2.50,
  Sharpe 2.59; directional cluster t 1.94; market-making t 0.13; opportunistic t −0.08. In small-tick stocks the
  paper's headline cell (opportunistic OFI → next bucket, Sharpe 1.34 vs 0.60 plain) is Sharpe −0.28 here vs 1.52
  for plain OFI.
- Differences from the paper that could matter: message-rebuilt book (files start ~7 am, so early level volumes are
  understated), 2023–24 instead of 2021, different stocks, log-standardised features. Recorded as **not replicated
  under our implementation**.

## Forecasting power: how much, from what, how stable (2026-09-25) — forecasting only from here on

Instruction 2026-09-25: no fills, no spread-crossing, no strategy P&L; measure forecasting power. `code/forecast_power.py`
on the v3 panel (6 definitions × 5 horizons), gradient boosting on every feature, trained on the train stocks in
2022–23, tested on the 56 test stocks in 2024 and on all 96 stocks in 2025 (never used for any choice).

**How much** (best definition, early5 = decide at the 5th child; 2024 test stocks / 2025):

| horizon | IC (NW t) | out-of-sample R² vs zero | top − bottom decile of the realized mid move |
|---|---|---|---|
| +10 s | 0.230 (49) / 0.198 (34) | 8.3% / 6.4% | 5.3 / 4.5 bps |
| +60 s | 0.187 (37) / 0.152 (32) | 5.9% / 4.5% | 8.8 / 7.6 bps |
| +5 min | 0.094 (21) / 0.079 (17) | 1.4% / 1.1% | 8.3 / 7.5 bps |
| +30 min, market-excess | 0.055 (15) / 0.048 (16) | 0.45% / 0.27% | 10.7 / 8.4 bps |
| close, market-excess | 0.042 (8.4) / 0.043 (6.5) | 0.20% / 0.16% | 16.6 / 15.4 bps |

The other definitions are close behind (run 10 ms 0.223 / 0.206 at 10 s; level-clearing 0.218 / 0.196; cancellation
bursts 0.192 / 0.180; hidden-heavy 0.172 / 0.136; confirmed Hawkes 0.167 / 0.140). Every cell is t > 4.

**What carries it** — IC lost when one feature group is removed and the model refitted (mean over the six
definitions, 2024 / 2025):

| horizon | order book | burst price path | controls (move since open, 30-min pre-move, time, spread) | burst size structure | size / timing regularity |
|---|---|---|---|---|---|
| +10 s | 0.039 / 0.025 | 0.036 / 0.039 | 0.018 / 0.020 | 0.000 | 0.000 |
| +60 s | 0.020 / 0.008 | 0.024 / 0.027 | 0.021 / 0.024 | 0.000 | 0.000 |
| +5 min | 0.007 / 0.003 | 0.012 / 0.015 | 0.010 / 0.012 | 0.000 | 0.000 |
| +30 min | 0.001 / −0.001 | 0.005 / 0.006 | 0.012 / 0.015 | 0.000 | −0.001 |
| close | 0.000 / 0.000 | 0.003 / 0.003 | 0.024 / 0.022 | 0.001 | −0.001 |

- Seconds: the order book and the burst's own price path (its push, the 60-s pre-move, how much of the opposite
  touch it consumed, the spread change) carry it jointly. Minutes: the burst's path. Thirty minutes to the close:
  the controls — intraday reversal of the day's move — with the burst path adding t ≈ 1–5.
- **Who is trading adds nothing to forecasting**: the burst's size structure (children, duration, volume / ADV) and
  the fingerprint regularities (modal clip, non-round clip, size CV, inter-arrival CV, timer phase) lose 0.000 IC at
  every horizon. The fingerprint identifies algorithms (see the fingerprint studies) but not their price effect.

**Size of the next move** (|signed move|): IC 0.30–0.43, almost all from controls + book; burst features add
+0.005 to +0.020 (2025: +0.007 to +0.017 at 60 s). Re-tested with pre-event realized variance as a control in v4.

**Stability (2025, 96 stocks):** the IC is positive for 99–100% of stocks at 10 s and 60 s, 83–99% at 5 min,
83–92% at 30 min and 73–80% to the close. It is twice as strong in wide-spread stocks (10 s: 0.22 wide, 0.15 middle,
0.09 tight tercile — tick-constrained mids rarely move within seconds) and similar across the day, except that the
to-close IC is lower after 14:52 (0.023 vs 0.044 in the morning). Hit rates at +10 s understate skill because 26–41%
of 10-s outcomes are exactly zero (10–14% at 60 s, 3–5% at 5 min; fixed in v4: hit rate on non-zero moves).

## Raw pass v4: forecasting products (2026-09-25)

`code/burst_defs_raw4.py` (cell V4B: jobs 14913058/59, 14913104 for a slow shard's remainder, concat 14913060):
58,257 stock-days 2022–25 (early-close days excluded), 4.39 M sampled events (10 per definition per stock-day, plus a
matched random-time event per sampled run 0.1 s burst) and 4.54 M five-minute bins. Analysis
`code/burst_defs_raw4_model.py`; train = 56 train stocks 2022–23, tests = 56 test stocks 2024 and all stocks 2025.

**Event level — the term structure** (`results/burst_forecasting/burst_defs_raw4/fig_term_structure.png`). Gradient
boosting on controls + book + burst features; 2024 test stocks / 2025 all stocks; * = market-excess:

| horizon | early5 IC | early5 gain over controls + book (t), 2025 | run 0.1 s in 10:00–15:30: real vs random-time IC, 2025 | direction AUC on non-zero moves, early5 2025 | R² vs zero, early5 2025 |
|---|---|---|---|---|---|
| +1 s | 0.189 / 0.180 | +0.069 (22.4) | 0.168 vs 0.075 | 0.644 | 4.0% |
| +10 s | 0.226 / 0.198 | +0.059 (22.5) | 0.209 vs 0.097 | 0.626 | 6.4% |
| +60 s | 0.179 / 0.155 | +0.042 (23.3) | 0.168 vs 0.062 | 0.585 | 4.6% |
| +5 min | 0.093 / 0.087 | +0.026 (14.4) | 0.086 vs 0.027 | 0.544 | 1.2% |
| +30 min* | 0.061 / 0.054 | +0.008 (3.3) | 0.040 vs 0.009 | 0.527 | 0.39% |
| +60 min* | 0.054 / 0.045 | +0.006 (2.4) | 0.034 vs 0.009 | 0.521 | 0.23% |
| close* | 0.043 / 0.040 | +0.006 (2.7) | 0.031 vs 0.010 | 0.520 | 0.18% |
| next open* | 0.041 / 0.033 | +0.002 (1.1) | 0.021 vs 0.010 | 0.517 | 0.13% |
| next close* | 0.010 / 0.007 | 0.000 (0.0) | −0.002 vs −0.003 | 0.503 | < 0 |
| direction of the first mid change | 0.236 / 0.216 | +0.065 (28.3) | 0.233 vs 0.120 | 0.628 | — |

- **Predictability peaks 5–10 s after the decision**, then falls to ~60% of the peak by 2 minutes and ~45% by
  5 minutes (early5, 2025): IC 0.18–0.20 at +1 to +10 s, 0.155 at +60 s, 0.09 at +5 min, 0.05 at +30 min, 0.04 to the
  close, 0.03 to the next open, and zero to the next close.
- **Burst-specific at every intraday horizon**: in the same hours, real bursts forecast 2–4.5× better than
  random-time events with the same duration and side, and the same path features add at most 0.005 at random times.
  Through 30 minutes the burst gain is t ≥ 3 for early5 and the short-gap runs in both years; to the close it is
  between −0.003 and +0.006 (t ≤ 2.7, not consistently significant); to the next open and next close it is zero.
- **The core signal is the reversal of the burst's own push**: the burst's move alone has IC −0.13 at +1 s, −0.11 at
  +10 s, −0.06 at +60 s, −0.03 at +5 min, −0.016 to the close (early5, 2025); at random times −0.003 at +10 s.
- **Direction of the next mid change** (no zero outcomes): hit rate 59–60% (early5, runs, level-clearing) vs 55% at
  random times; AUC 0.62–0.64 vs 0.57. At fixed horizons, 26–74% of +1 to +10 s outcomes are exactly zero (57–90% at
  random times).
- Ranking of definitions is unchanged: act at the 5th child ≈ 10 ms / 0.1 s runs > level-clearing > cancellation >
  hidden-heavy >> 60 s runs (gain ≈ 0 at every horizon).
- Decile spreads of the realized mid move (early5, 2025): 4.5 bps at +10 s, 7.7 at +60 s and +5 min, 8.8 at +30 min,
  13.2 to the close.

**Volatility, controlled for the pre-event level.** log(1 + realized variance) of 1-s mid returns over the next
1 / 5 / 30 minutes: pre-event realized variance (1 / 5 / 30 min) + time + spread gives IC ≈ 0.65 / 0.82 / 0.88; the
book adds ≤ 0.009 and the burst features +0.000 to +0.008 (2025 early5: +0.004 / +0.001 / +0.000). Bursts do mark a
modest volatility increase: in 10:00–15:30, 5-minute log RV after minus before is +0.14 / +0.08 (2024 / 2025) at run
0.1 s bursts and −0.03 / −0.02 at random times (all-day groups: +0.04 to +0.18 for runs and early5, ≈ 0 for 60 s
runs, −0.07 for cancellation bursts).

**5-minute panel — aggregating bursts into bins loses the information.** Cross-sectional rank IC per date × bin
(averaged within the day, NW over days), market-excess targets:

| target | base (past 5/15/30-min returns, move since open, all-trade flow and quote OFI over 1 / 3 bins and since open, relative volume, bin) | + signed burst imbalance and burst counts, 7 definitions × 3 windows | gain |
|---|---|---|---|
| next 5 min | 0.028 (t 10.4) / 0.028 (t 7.7) | 0.029 / 0.030 | +0.001 (t 1.5) / +0.002 (t 3.9) |
| next 15 min | 0.023 (t 6.7) / 0.023 (t 5.3) | 0.023 / 0.024 | −0.001 / +0.001 (t 2.2) |
| bin end → close | 0.017 (t 2.9) / 0.012 (t 1.7) | 0.014 / 0.010 | −0.004 / −0.002 |

Robustness — the same comparison with models linear in per-(date × bin) ranks (the rest-of-day targets overlap within
a day, so boosting could overfit): base 0.029 (t 8.2) / 0.029 (t 6.6) next bin, 0.022 / 0.024 next 15 min, 0.016
(t 3.0) / 0.014 (t 2.5) to the close; adding bursts changes these by −0.002 (t −3.8 / −5.3), −0.002 and −0.003 / +0.000.

Univariate: a bin's signed burst imbalance forecasts the next bin with |IC| ≤ 0.003 for every definition; the
since-open burst imbalance forecasts the rest of the day at 0.015–0.017 (t ≈ 3) in 2025 but 0.004–0.006 (t < 1) in
2024 — and the since-open *all-trade* imbalance does the same (0.018, t 3.0 / 0.004), so it is not burst-specific.
What forecasts the next bin is short-term reversal: the last bin's excess return (IC −0.029, t −8.3 / −0.027, t −5.8)
and its quote OFI (−0.017 / −0.014). Next-bin realized volatility: HAR terms + trade counts IC 0.726; burst counts
add +0.000 / −0.007. By time of day (rest-of-day target), the burst model is below the base before 13:30
(e.g. 10:00–11:30: 0.016 vs 0.025 / 0.010 vs 0.017) and slightly above it after 13:30 (+0.002 to +0.006, not
significant on its own: every with-burst IC after 13:30 has t ≤ 2.5).

**Market timing — nothing.** Cross-stock mean burst imbalance (per date × bin) against the equal-weight market's next
5 / 15 minutes and rest of day: correlations within ±0.03, signs flip between 2022–23 and 2024–25 (run bursts →
rest of day −0.007 in-sample, +0.011 / +0.014 out-of-sample), and the out-of-sample R² of an OLS fitted in 2022–23 is
≤ 0.0003 with or without the burst aggregates. All-trade flow is weakly positive for the next 5 minutes (0.011–0.022,
t 1.9–3.1).

**Daily horizon — nothing from bursts.** The day's (and last hour's) signed burst imbalance per definition against the
overnight, next open-to-close, next close-to-close and next-5-day mid returns, cross-sectional per date: every burst
IC within ±0.02 and |t| < 2.1 in 2024. Models linear in per-date ranks (gradient boosting overfits the 22k training
stock-days and scored ≈ 0 even with the base features), base = day and last-hour return, realized variance, all-trade
flow and quote OFI (day, last hour), relative volume:

| target | base IC (t), 2024 / 2025 | + burst imbalances and burst shares | gain |
|---|---|---|---|
| overnight (16:00 mid → 9:31 mid) | 0.032 (2.5) / 0.035 (2.7) | 0.022 / 0.033 | −0.010 / −0.002 |
| next open-to-close | 0.025 (2.3) / 0.003 (0.2) | 0.024 / 0.006 | −0.001 / +0.003 |
| next close-to-close | 0.007 (0.5) / −0.005 | 0.017 / −0.003 | +0.010 (t 1.3) / +0.002 |
| next 5 days close-to-close | −0.006 / −0.003 | −0.003 / −0.009 | +0.003 / −0.006 |

The overnight return reverses the day (last-hour return IC −0.031, t −2.9 in 2022–23; −0.037, t −3.0 in 2024;
−0.041, t −3.6 in 2025); close-to-close is not forecastable, and bursts add nothing at any daily horizon.

## Is the short-horizon burst forecast mechanical? (2026-09-25)

`code/burst_defs_raw4_mechanical.py`: v4 models fitted once on the train stocks 2022–23; the 2025 IC split by what the
burst did to the book (spread at the decision vs just before the burst, in ticks; the mid's move during the burst).
Share of events (early5): spread unchanged 34%, widened 45%, narrowed 22%; mid did not move 9%; one-tick spread at the
decision 27%.

| early5, 2025 | IC +10 s: full / controls + book | burst gain +10 s | +60 s | +5 min | +30 min* |
|---|---|---|---|---|---|
| all | 0.197 / 0.139 | +0.058 | +0.043 | +0.026 | +0.010 |
| burst widened the spread | 0.235 / 0.136 | **+0.099** | +0.081 | +0.048 | +0.017 |
| spread unchanged | 0.175 / 0.164 | +0.011 | +0.013 | +0.008 | +0.006 |
| one-tick spread at the decision | 0.162 / 0.153 | +0.010 | +0.013 | +0.006 | +0.008 |
| mid did not move during the burst | 0.177 / 0.187 | −0.011 | +0.013 | +0.008 | −0.001 |

(run 0.1 s is the same: +0.092 at 10 s when the spread widened, +0.016 when unchanged, −0.003 when the mid did not move.)

- **What bursts add beyond the order book is mostly the book relaxing after the burst displaced it**: the gain lives
  in bursts that widened the spread or moved the mid, and is ≈ 0.01 when the spread is unchanged or one tick wide and
  ≤ 0 when the mid did not move. This is transient impact / order-book resilience — the mid-quote reverting as the
  cleared level refills and the temporary part of the push decays.
- **Predictability after bursts stays high even without displacement** (IC 0.16–0.19 at 10 s), but there the
  controls + book model already has it (0.15–0.19) — and that book-only forecast is itself stronger at burst times
  than at random times (≈ 0.15 vs ≈ 0.10 at 10 s).
- **On average prices continue in the burst's direction**: the mid moves +4.8 to +5.9 bps during short-gap bursts and
  a further +0.6 to +0.7 bps by +10 s and +1.0 to +1.3 bps by +1 to +5 min after the decision (random times: ≈ 0; 60 s
  runs: ≈ 0; cancellation bursts: −0.6 to −1.1). The cross-sectional forecast is the other margin: bursts whose push
  was larger, or that left a wider spread, give back more.
- Implication for any write-up: the burst-specific forecasting result must be framed as transient impact and
  resilience, measured against that literature, not as information about future fundamentals. Open checks: isolated
  single aggressive orders as a placebo (are bursts different from any marketable order?); targets that exclude spread
  reversion (the far-side quote, the microprice); a multi-level book baseline rebuilt from messages; an ex-ante universe.

## Raw pass v5: the three checks on a larger universe (2026-09-25)

`code/burst_defs_raw5.py` (verified on synthetic LOBSTER days by `tests/test_burst_defs_raw5.py`: no look-ahead in any
decision-time feature or in event selection, a positive control that catches the known whole-day OFI leak, the rebuilt
depth ladder equals an order-by-order replay, targets equal a recomputation from the quote path). Cluster cell V5
(jobs 14922969 / 14923074, concat 14922972): 73,315 stock-days, 676 stocks + SPY / QQQ, 144 sampled dates (3 a month,
2022–25), 9.19 M events of 13 types. Analysis `code/burst_defs_raw5_model.py`; figure
`results/burst_forecasting/burst_defs_raw5/fig_v5_placebo.png`. Train: 2022–23 on the train names (the 60 v2–v4 train
names + a fixed hash-half of the new names); tests: the other names in 2024 and 2025, and the train names in 2025.
Models: GENERIC (burst-blind: mid return, trade-flow imbalance, trade count and quote OFI over the last 1–300 s,
relative volume, realized variance, spread, queue imbalance, microprice), + DEPTH (5-level book rebuilt from messages),
+ EVENT (the event's own features), + BHIST (bursts in the last 1 / 5 / 30 min).

**Book reconstruction (A39).** At 9.19 M real decisions the rebuilt level-1 size equals the C++ quote path's at
99.9994% (both sides); every decision matches on 99.86% of stock-days.

**Check 1 — bursts vs single aggressive orders vs random times** (2025, names never trained on; IC of GENERIC + DEPTH
+ EVENT; * market-excess):

| event | mid +10 s | mid +1 min | mid +5 min | mid +30 min* | mid close* | far quote +10 s | microprice +1 min |
|---|---|---|---|---|---|---|---|
| burst: act at 5th order | 0.201 | 0.153 | 0.087 | 0.052 | 0.037 | 0.295 | 0.252 |
| burst: run 0.1 s | 0.197 | 0.158 | 0.085 | 0.053 | 0.031 | 0.288 | 0.262 |
| two orders | 0.201 | 0.151 | 0.088 | 0.043 | 0.036 | 0.263 | 0.258 |
| one isolated order | 0.195 | 0.161 | 0.077 | 0.034 | 0.025 | 0.246 | 0.253 |
| one large isolated order | 0.193 | 0.142 | 0.075 | 0.045 | 0.029 | 0.253 | 0.251 |
| any marketable order | 0.203 | 0.152 | 0.083 | 0.042 | 0.023 | 0.268 | 0.245 |
| random time (same duration, side) | 0.128 | 0.086 | 0.031 | 0.005 | 0.018 | 0.152 | 0.261 |

- **The short-horizon forecast belongs to aggressive orders, not to bursts.** A single isolated marketable order is as
  forecastable as a burst at every horizon on the mid; bursts keep a small edge only on the far quote in the first
  seconds (0.29 vs 0.25 at 10 s). Random times are far weaker (half at 10 s, a third at 5 min, a tenth at 30 min) —
  except for the microprice beyond 10 s, where book dynamics alone forecast as well at random times.
- Dose-response at the same confirmation lag (one pooled model): mid IC at 10 s is flat in the number of orders (0.204
  for 1 order, 0.21 for 2–9, 0.195 for 10+); the far quote and longer horizons rise with it (far quote +60 s: 0.217 →
  0.267 at 5–9 orders; +30 min*: 0.042 → 0.058 at 5–9 → 0.084 at 10+).
- Size-matched (quintiles of volume / ADV): after a burst the mid drifts further in its direction than after one order
  of the same total size (top quintile, +60 s: 1.55 vs 0.99 bps; bottom: 0.81 vs 0.20 bps), and the far-quote IC is
  higher in every quintile (0.23–0.31 vs 0.22–0.27 at 10 s).
- Mean signed moves after the decision (bps): act-at-5th-order bursts +0.81 at 10 s, +1.37 at 1 min, +1.46 at 5 min, the
  far quote following (+0.54 / +1.05 / +1.22); isolated orders +0.46 / +0.89 / +1.02; random times ≈ 0. On average the
  whole quote keeps moving in the aggressor's direction.

**Check 2 — targets the near-side refill cannot move.** The far-side quote (the side the event did not trade against) is MORE
forecastable than the mid (0.30 vs 0.20 at 10 s; 0.23 vs 0.15 at 1 min, 2025), and so is the microprice. The event's own
features add the most on the near quote (+0.04 to +0.05, the refill), +0.01 to +0.02 on the far quote and the mid, and
+0.006 on the microprice. The first change of each quote is called right at IC 0.23–0.29, and the log waiting time to
the next change is forecast at IC 0.36–0.45.

**Check 3 — a deeper, burst-blind baseline.** Against GENERIC + DEPTH the event's features add +0.02 IC at 1 s–1 min
(t 6–11), +0.01 at 5 min, 0 at 30 min and the close — a third of what they added over the v4 book baseline (+0.06 at
10 s). DEPTH adds +0.003 to +0.010 at seconds and nothing beyond; BHIST (recent burst history) adds nothing (|t| < 2).

**Robustness (early5 and run 0.1 s).** Unseen names 2024 / 2025 / train names 2025: mid 10 s 0.237 / 0.204 / 0.195. Stock-
block bootstrap 95% CIs exclude zero through the market-excess close ([0.018, 0.038] in 2025); the raw (not
market-adjusted) close IC is ≈ 0. Without the 10 most active stocks: unchanged. Share of stocks with IC > 0: 97–99.7% at
10–60 s, 70–80% at 30 min, 50–69% at the close. By time of day: 0.19 (9:30–10) to 0.24 (14:00–15:30) at 10 s. One-tick
spreads: 0.171 vs 0.224 (2+ ticks) at 10 s; far quote at 1 min 0.110 vs 0.273. Dropping the pre-burst drift features
changes nothing. The mid moves −0.2 bps in the 60 s before a burst, +5.8 during, +1.4 in the minute after.

**Cross-stock (A35).** Adding other stocks' and SPY / QQQ same-side bursts in the last 1 / 5 / 30 s to the full model:
|gain| ≤ 0.003 for bursts and single orders (+0.012 to +0.015 only at random times). Bursts that coincide with many other
stocks' same-side bursts drift about half as far afterwards (30 min*: 0.7 vs 1.6 bps; close*: 0.6 vs 1.7 bps).

**Leak check.** The v4 feature set with the whole-day OFI scale and with the leak-free scale give identical ICs.

**5-minute panel on 676 stocks.** Burst aggregates add +0.0009 (t 3.4) / +0.0005 IC to next-bin forecasts and nothing
to the rest of the day; the since-open burst imbalance forecasts the rest of the day univariately (0.019, t 6.8 in 2024;
0.010 in 2025) but the since-open all-trade imbalance does at least as well (0.027 / 0.013) and bursts add nothing to it.
