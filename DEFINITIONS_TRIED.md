# Every Burst Definition and Parameterisation Tried

Complete ledger. The count matters because it sets the multiple-testing hurdle: with this many
configurations searched, a Sharpe or t-statistic must clear a deflated bar, not a conventional
one. Harvey–Liu–Zhu put the single-test hurdle at t ≈ 3.0; a search this wide puts it higher.

Phases 1–3 predate the current session and are reconstructed from `main.tex` and `src_cpp/`.
Phases 4–10 were run in-session with array IDs recorded.

---

## Phase 1 — original C++ detector (`src_cpp/burst.cpp`)

| # | Definition | Parameters swept |
|---|---|---|
| 1 | Strict silence threshold — burst ends after a quiet gap | Δt ∈ {0.5s, 1.0s, 2.0s} |
| 2 | Self-exciting decaying counter ("Hawkes") — intensity decays exponentially, +1 per arrival, burst lives while λ > λ_min | β ∈ {0.5, 1, 2, 5}; λ_min ∈ {0.3, 0.5, 0.8} |
| 3 | Minimum cluster size | {3, 5, 10} |
| 4 | Fractional sweep volume — burst volume as a share of 14-day trailing ADV | Optuna-tuned per name (NVDA 6.45e-5 … TSLA 0.0033) |
| 5 | Directional consistency ratio | thresholds ≈ 0.5–0.75 |
| 6 | Volume ratio — cap on opposing-side volume | ~0.5 |
| 7 | Decay filter κ on D_b (forward markout gating) | κ ∈ {0, …, 1.28}; forced to 0 for short horizons |
| 8 | Passive bursts — type-1 limit submissions at levels 1–3 | same gates, ADV from executed volume |

Note on #2: the excitation increment is hardcoded at unity, so α is a normalisation, not a
free parameter. Only β and λ_min are identified. `main.tex` previously described "α = 0.5",
which was the *threshold* mislabelled.

## Phase 2 — alternative reconstructions (pilot, AAPL/TSLA)

| # | Definition |
|---|---|
| 9 | Order-flow imbalance (Cont–Kukanov–Stoikov top-of-book queue dynamics) |
| 10 | Book resilience — aggressive sweeps whose consumed depth fails to replenish |
| 11 | Hidden-execution clustering (type-5 prints) |

## Phase 3 — trade-sign conventions for hidden prints

| # | Rule |
|---|---|
| 12 | Quote rule, abstaining at the midpoint |
| 13 | Tick rule |
| 14 | EMO (Ellis–Michaely–O'Hara) |
| 15 | CLNV (Chakrabarty–Li–Nguyen–Van Ness) |
| 16 | Quote rule vs a deliberately staler mid (1s earlier) |
| 17 | Quote rule vs a forward mid (1s later, look-ahead diagnostic) |
| 18 | Outside-the-quote prints only |

## Phase 4 — burst zoo, 10 definitions (arrays 14368098, 14368227)

| # | Definition | Signing |
|---|---|---|
| 19 | Hidden prints, time-clustered | outside pre-quote |
| 20 | Hidden clusters with elevated local arrival rate (>3× median) | outside pre-quote |
| 21 | Visible executions, time-clustered (min run 5) | native ITCH Direction |
| 22 | Visible clusters with volume > 2× median cluster volume | native |
| 23 | One-sided cancellation clusters (type 2/3) | native, inverted |
| 24 | One-sided submission clusters (type 1) | native |
| 25 | Mixed clusters containing both visible and hidden | native, visible leg |
| 26 | Accelerating visible clusters (second half faster than first) | native |
| 27 | Odd-lot clusters (size not a multiple of 100) | native |
| 28 | Single block prints > 5× median trade size | native |

## Phase 5 — block-print variants, 6 (array 14368227)

| # | Variant |
|---|---|
| 29–32 | Size threshold k × median trade size, k ∈ {3, 5, 10, 20} |
| 33 | Blocks executing at or through the pre-print touch |
| 34 | Blocks followed by another same-direction block within 30s |

## Phase 6 — idea zoo, 15 ideas × 3 parameters = 45 (array 14368630)

| # | Idea | Parameters |
|---|---|---|
| 35 | Metaorder detection — sustained one-sided visible flow | window 60 / 300 / 900 s |
| 36 | Iceberg replenishment — repeated hidden prints at one price | 5 / 30 / 120 s |
| 37 | Algorithmic schedule fingerprint — repeated identical sizes | 3 / 5 / 10 repeats |
| 38 | **Volume-clock bursts** — clustering in volume time | 0.2% / 0.5% / 1% of daily volume |
| 39 | **Directional-change events** (intrinsic time) | δ = 5 / 10 / 25 bps |
| 40 | Quote-update bursts (maker side) | min run 10 / 25 / 50 |
| 41 | Burst intensity → future \|return\| | 60 / 300 / 900 s |
| 42 | Vol-of-vol — dispersion of arrival rate | 60 / 300 / 900 s |
| 43 | Spread-widening prediction | 60 / 300 s |
| 44 | Queue-depletion prediction | 60 / 300 s |
| 45 | Own-flow persistence | 60 / 300 / 900 s |

## Phase 7 — idea zoo 2, 7 (array 14418087)

| # | Idea | Status |
|---|---|---|
| 46 | HAR incrementality — counts vs lagged realized volatility | the one that passed |
| 47 | Jump arrival (>3σ move) | IC 0.23–0.28 |
| 48 | Adverse-selection avoidance — when *not* to quote | IC 0.35–0.38 |
| 49 | Time-to-fill proxy | IC 0.54, but near-trivial activity persistence |
| 50 | Volume-profile deviation | **broken** — mechanically complementary |
| 51 | Closing-auction imbalance | **broken** — contemporaneous window |
| 52 | Entropy of the message-type mix | IC 0.11–0.14 |

## Phase 8 — price-free formation arms, 3 (array 14314701)

| # | Formation | Signing |
|---|---|---|
| 53 | Runs of same-side prints | contemporaneous mid |
| 54 | Time clusters only | mid 1 ms before the print |
| 55 | Time clusters only | outside the pre-print quote |

## Phase 9 — per-print, no clustering, 3 (array 14367935)

| # | Signing |
|---|---|
| 56 | Contemporaneous midpoint |
| 57 | Midpoint 1 ms before the print |
| 58 | Outside the pre-print quote |

## Phase 10 — point-in-time daily signals, 3 (array 14489997)

| # | Signal |
|---|---|
| 59 | Natively-signed visible flow |
| 60 | Natively-signed visible flow / volume |
| 61 | Cleanly-signed hidden flow |

## Phase 11 — hidden-vs-visible nesting, 5 models (array 14482647)

| # | Model |
|---|---|
| 62–66 | HAR; +visible; +visible+hidden; +hidden alone; +visible+volume+hidden |

---

## Count

**66 distinct definitions/models**, and **~110 parameter configurations** once sweeps are
counted individually. Every directional one lands on the same line: markout ≈ 0.709 ×
half-spread, intercept zero, 0 of 40 names clearing a round-trip cost.

## Never tried

Blocked on data: index-rebalance and expiry calendars, ETF baskets, options/variance data,
consolidated NBBO, multi-venue routing.

## 2026-09-13 addition: incremental burst-information screen

See `studies/burst_information/BURST_INFORMATION_DESIGN.md` and `results/burst_information_v1/design.json`.
Five fixed forecast configurations (linear state, nonlinear state, state + online regime,
state/regime + burst, state/regime/burst + simulation score), three prospective landmarks,
five flow/price/execution targets: **75 fits**, **120 paired comparisons** across two cohorts.
Three fixed reconstruction definitions across nine controlled mechanisms add **27 diagnostic
cells**. These are exploratory comparisons, not independent alpha discoveries. No evaluation
outcome is used to select parameters. The 66/~110 count above is a historical subtotal and
does not include this addition or all intervening packet/continuation phases.

Blocked on nothing — genuinely untested: cross-sectional lead–lag across names, cross-impact
networks, clustering of names by flow structure, forced/uninformed flow windows, passive
execution simulation, and the SEC Tick Size Pilot as an instrument.


The same session added a **paired join-training bug diagnostic**: two fits on 30 new simulated
training days (legacy pooled-day candidates versus session-scoped candidates), evaluated on
18 new simulated days with threshold 0.5 fixed. It is a software/design diagnosis, not two
new real-market strategy searches. Source: `diagnose_join_sessions.py`; independent CSV audit:
`audit_join_sessions.py`. The old fit's failure cannot close the reconstruction question.

Completed: all 75 fits / 120 comparisons plus 90 execution statistics. Post-result descriptive
loss levels and leave-one-name-out checks of the secondary third-packet one-minute return
result add no fitted specifications or inferential tests; they are explicitly exploratory.

## 2026-09-13 addition: fingerprint-v1 validation grid — not an alpha search

`studies/fingerprint/BURST_FINGERPRINT_DESIGN.md`. **66 burst definitions** (3 rules × 11 gaps × minimum 2 or 3
packets) are *scored* against real-data evidence of common origin (identical untruncated
same-side child sizes in excess of chance), in 3 size classes under 2 nulls. No price outcome is
used, and no definition is traded. The single pre-declared selection (Youden J; class
`u_nonround`, minimum 3, within-day long-lag null) is confirmed once on 2021 with disjoint names.
These cells add to the definition ledger for completeness; they do not enter a Sharpe or
deflated-Sharpe hurdle.

Corrections to the addition above: the burst-information matrix's relative-MSE contrasts
are uninformative (every return/wait model loses to a zero forecast; flow contrasts are
dominated by the two most active names). Its 75 fits count as tried configurations, but they
tested nothing about burst information in either direction (`VERIFIED_RESULTS.md` §1.24).

Fingerprint-v1 outcome (2026-09-13): uninterrupted same-side runs (30–300 s cap) and side-only
2–5 s streams tie on mean per-name Youden J in both years. The pre-declared single winner (run,
300 s) ranked 7th of 33 in 2021; the rank correlation of all 33 definitions across years was
0.95. Working definition: run, 60 s cap, ≥ 3 packets. See `studies/fingerprint/BURST_FINGERPRINT_RESULTS.md`.


## 2026-09-14 addition: program-evidence-v1 and metaorder-v1

- **Burst rules re-scored, not new definitions.** The 33 fingerprint-v1 definitions (min 3) were
  scored against a timer fingerprint (module A3). Two candidates, run60 and stream5, were compared
  on a combined size + phase J in 2013, 2016, 2019 (gate), 2021 and 2024. Result: a tie; run60 kept.
- **Tried signals (count toward the multiple-testing hurdle):**
  - daily program imbalance → next-day close-to-close, next-day open-to-close and next-5-day
    returns, in 2024 and 2021 (6 cells, all null, |t| ≤ 1.5);
  - passive 100-share postings triggered by program-like, middle and low-score bursts against
    random times, 4 markout horizons, in 2024 and 2021 (every per-fill markout negative).
- **Linkage and scoring constructs (not traded):**
  - rare-size campaigns (C);
  - forward identical-size link evidence per burst and a positive-unlabeled linkage score (M2);
  - a first-three-packet real-time program score (M4);
  - synthetic-parent injection (M3).

## 2026-09-15 addition: p4-revisit-v1 (`studies/p4_revisit/P4_REVISIT_DESIGN.md`, amendments A1–A6)

The original MATH 279 P4 pipeline, run with native signs, point-in-time universes and leakage firewalls.
Every cell below counts toward the multiple-testing hurdle.

- **Event families (2).**
  - Trade bursts: run60 on economic packets.
  - Submission bursts: run rule on non-fleeting, non-replace, non-remainder adds at or inside the touch.
- **Filter.** P4 eq. 3.3, D_b ≥ κ·PeakImpact.
  - κ ∈ {0.25, 0.5 primary, 0.75}.
  - Large: top quintile of |Q_b| over the name's trailing 20 days.
  - Pseudo-burst placebo at random times (A4 rules).
- **Q1:** 2 primary statistics (info − pseudo per family) plus secondary contrasts at 3 horizons and 3 κ.
- **Q2:** (a) linkage; (b) 13F; (c) mutual-fund holdings and flow-induced trading; (d) index events; (e) BJZZ retail. That is 2 families × 6 statistics.
- **Q3:** 2 models (ridge, boosting) × 2 families × 3 targets.
- **Q4:** 6 primary cells (2 families × tCLOSE, CLOP, CLCL).
  - Secondary: placebo signal and κ grid, 3 × 6.
  - Decile portfolios at 2 cost levels, 2 × 6.
  - Deflated-Sharpe trial count: 311.
- **Q5:** 5 split dimensions, descriptive.
- **Q0 audit (not new signals):** the legacy C++ rule on three trade streams; overnight long-only benchmarks for 4 names; the legacy κ gate at 2 levels.

**Outcome, 2026-09-15** (`VERIFIED_RESULTS.md` §1.31, tables in `studies/p4_revisit/P4_REVISIT_RESULTS.md`). All cells were run in
three periods (2012–16, 2020–21, 2022–25). Nothing in Q1 or Q4 survived: the Q1 placebo contrast is negative or
zero everywhere, and of the six Q4 cells exactly one passed validation (submission CLOP, reversal-signed) and
failed confirmation. Q2(a) separates the families — trade-burst linkage holds its sign in all three periods,
submission linkage reverses — and Q2(b)/(c) on 13F and mutual-fund holdings replicate in all three for both
families. Q3 replicates. **Do not re-enter any of these definitions as a fresh signal search:** the 311-trial
count above is already spent, and a new configuration of the same idea needs a new deflated-Sharpe budget.

*Post hoc, 2026-09-20:* one further specification — the Q2(b)/(c) regressions with the same-quarter return
added — run on VAL, TEST and ERA2 for both families (18 regressions). It is a robustness control on the
pre-registered tests, not a signal search; see `studies/p4_revisit/P4_REVISIT_RESULTS.md` "Post hoc: is the institutional
association a return channel?".
