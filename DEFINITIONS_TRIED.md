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

Blocked on nothing — genuinely untested: cross-sectional lead–lag across names, cross-impact
networks, clustering of names by flow structure, forced/uninformed flow windows, passive
execution simulation, and the SEC Tick Size Pilot as an instrument.
