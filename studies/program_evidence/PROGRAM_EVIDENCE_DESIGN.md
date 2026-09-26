# Program-evidence-v1: are fingerprint bursts real execution programs, and what do they show?

Pre-registered 2026-09-13, before any statistic below was computed or read. Builds on
fingerprint-v1 (`studies/fingerprint/BURST_FINGERPRINT_RESULTS.md`, `VERIFIED_RESULTS.md` §1.27). Results go to
`studies/program_evidence/PROGRAM_EVIDENCE_RESULTS.md`; code hashes are frozen in `results/program_evidence_v1/freeze*.json`
before each stage's outputs are read.

## Why

Fingerprint-v1 showed that same-side aggressive orders with identical untruncated non-round sizes
recur about twice as often as chance, and that simple burst definitions keep about half of that
evidence together. The user asked (2026-09-13):

1. how to show these are real execution programs rather than an imposed pattern;
2. what the definition says about the market, including the original daily buy-minus-sell burst
   idea;
3. what else is worth exploring.

Every direction proposed in that answer is tested here. Identical sizes identify *algorithms*. A
market maker's algorithm can repeat sizes as well as an institution's execution algorithm, so the
central question is whether "program" can honestly mean *directional execution*.

## Samples, splits and data rules

| panel | names | days | use |
|---|---|---|---|
| fingerprint cache, exploration | 174 (2024, hash 0) | 10 adjacent pairs | modules A, B, C, F, G, H, I |
| fingerprint cache, confirmation | 291 of 300 (2021, hash 1–2) | 10 adjacent pairs | same, read once |
| contiguous exploration | point-in-time 2024 universe, hash 0 | all 2024 trading days | modules D, E |
| contiguous confirmation | point-in-time 2021 universe, hash 1–2 | all 2021 trading days | modules D, E |
| index events | every S&P 500 and Nasdaq-100 change effective in 2021 or 2024 | event windows | module D4, one shot |
| earlier years | fingerprint names present in 2013, 2016, 2019 | 10 adjacent pairs per year | module J1 |
| Tick Size Pilot | pilot and control names present in lobster2 | before and during the pilot | module J2, exploratory |

- **Name split.** `sha256("fingerprint-v1|" + ticker) mod 3`: 0 is exploration, 1–2
  confirmation, as in fingerprint-v1.
- **Point-in-time universe.** For year Y: CRSP common stocks (share codes 10, 11), top 500 by
  average daily dollar volume over October–December of Y−1, kept on dates where lobster2 has the
  ticker. Names that delist during Y stay until the archive ends. This removes the ex-post
  universe defect that sank the reversal result.
- **Confirmation discipline.** No parameter, threshold, class, null or lag bin changes after
  exploration output is read. Confirmation outputs are read once.
- **Licensed data.** LOBSTER derivatives stay on Hoffman2 or under gitignored `data/` and
  `results/`. WRDS extracts go to gitignored `data/wrds/`. Credentials are read from gitignored
  `.env` by `src_py/wrds_access.py` and never printed. Nothing licensed is committed.
- **Inference.** Unless stated otherwise: equal weight per name, medians of per-name ratios or
  means of per-name statistics, 95% intervals from 1,000 bootstraps over names. Daily panels use
  Fama–MacBeth slopes with Newey–West (10 lags) on the daily series.
- **Multiple testing.** This design adds 20 gated tests. Gates use pre-declared directions.
  Exploratory return tests use |t| > 3 (Harvey–Liu–Zhu); confirmation needs the same sign and
  t > 2.

## Common definitions

- **Packets and classes** are fingerprint-v1's, from `execution_packets.reconstruct_packets`:
  untruncated non-round (`u_nonround`), rare non-round (`u_rare`), and all signed packets.
- **Depth-matched cross-day null** (fingerprint-v1 primary): pairs across two adjacent trading days
  at the same clock lag, with both packets in the same executed-side depth quartile (thresholds
  pooled over the two days).
- **Burst:** run rule, 60 s cap, ≥ 3 own-side packets (the fingerprint-v1 working definition).
- **Program score for run/60 (stage 3b).** `fingerprint_burst_rows.py` with rule run and gap 60,
  then `program_score.py` unchanged: fit on 2024 exploration bursts, evaluated once on 2021.
  - P1b: top-minus-bottom decile excess repeats per 1,000 pairs has lower bound > 0.
  - P2b: Spearman(decile, excess) > 0.7.
  - The fitted model (standardization and coefficients) is saved. **Program bursts** are those
    whose linear score is at or above the 80th percentile of 2024 training bursts. If P1b fails,
    downstream modules use the frozen run/300 score and bursts instead, and say so.

## Module A — an independent timing fingerprint

**Question.** Does a size-free signature of common origin agree with the size fingerprint? A
timer-driven algorithm fires its children at whole-second multiples, so lags between its children
sit near integers regardless of size.

**Measurement** (`src_py/evidence_stats.py`). Same-side pairs with lag ℓ in [0.5, 60.5) s,
k = round(ℓ), phase φ = ℓ − k, counted in signed φ bins with edges ±{0, 1, 2, 5, 10, 20, 50,
100, 250, 500} ms, separately for each k:

- within-day pairs among `u_nonround` packets: all, identical-size, and opposite-side;
- within-day pairs among all signed packets;
- cross-day pairs (adjacent day, same clock lag) for `u_nonround` and all signed packets.

Also recorded: the distribution of absolute timestamp phase (t mod 1 s, 1 ms bins), and, for all
33 run/stream/timing × gap definitions (min 3 packets), within-burst `u_nonround` pairs split
into locked (|φ| < 10 ms) and unlocked, all and identical-size. Depth-matched pairs and matches
over [0.5, 60.5) s are recorded for the same definitions.

**Statistics.** Concentration c = (pairs with |φ| < δ) / (all pairs in the k-windows) / (2δ / 1 s),
with δ = 10 ms primary (1, 2, 5, 20, 50 ms reported).

- **A1.** PLR = c(within-day `u_nonround`) / c(cross-day `u_nonround`). Names with ≥ 1,000
  within-window pairs.
- **A2.** c(identical-size pairs) / c(different-size pairs), within day. Names with ≥ 500
  identical-size window pairs.
- **A3.** For each definition, phase TPR = excess locked pairs inside bursts / all excess locked
  pairs, with excess = locked − window pairs × 2δ × c(cross-day); FPR = window pairs inside bursts
  / all window pairs; J = TPR − FPR, averaged over names with ≥ 20 expected and ≥ 10 excess
  locked pairs. Size J is recomputed over the same lags [0.5, 60.5) s with the depth-matched null.

**Gates** (2024 and again 2021):

| gate | test |
|---|---|
| **A1** | name-median PLR, lower 95% bound > 1 |
| **A2** | name-median identical/different concentration ratio, lower bound > 1 |
| **A3** | Spearman(phase J, size J) across the 33 definitions > 0.7 |

A2 is the agreement test: identical-size pairs are enriched for same-origin pairs, so if timers
mark common origin they must be more phase-locked. Phase locking among *different*-size pairs
(A1 minus A2's share) estimates programs that randomize sizes, which identical-size matching
misses.

**Known direction of bias (from `tests/test_evidence.py`, before any real output):**
- *A1 is conservative for wall-clock-anchored timers.* An algorithm that fires at the same
  sub-second phase every day (for example on each whole second) locks cross-day pairs as well, so
  its locking cancels in A1.
- *Unrelated traders sharing a whole-second clock* give A1 ≈ 1 and A2 ≈ 1 in synthetic tapes, so
  a shared clock cannot fake agreement.

A2 is robust to both. The absolute-phase histogram shows how much anchoring there is.

*All cross-day nulls in this design are conservative for programs that repeat the same child size
at the same clock times on adjacent days.* The synthetic one-sided and two-sided algorithms were
invisible until their sizes differed across days. B3 measures that contamination directly.

## Module B — are same-size algorithms directional?

**Question.** An execution program buying a parent keeps buying. A market-making or two-sided
algorithm that reuses a child size trades both sides. Opposite-side identical-size repeats beyond
chance measure the second kind.

**Measurement.** Identical-size pairs, same side and opposite side, depth-matched, lag bins with
edges 0, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 14400, 23400 s, classes
`u_nonround` and `u_rare`. The cross-day null is computed for each side relation and lag bin
(clock-time seasonality matters at long lags). Unmatched versions are kept as sensitivity.

**Statistics and gates** (2024 and 2021):

- **B1.** Name-median of ratio_same − ratio_opp (observed/expected identical-size matches), in
  aggregated lag ranges 2–10 s, 10–60 s, 60–600 s and 600–3,600 s: lower 95% bound > 0 in all
  four ranges.
- **B2 (long-lived programs).** Same-side name-median ratio, lower bound > 1 at 600–1,800 s and at
  1,800–3,600 s.
- **One-sidedness index** (descriptive): X_same / (X_same + X_opp) per lag bin, with
  X = observed − expected pooled over names, and a name bootstrap.
- **B3 (multi-day campaigns).** Adjacent days share more same-side sizes than days a month apart if
  parents span days with a fixed child size. Size distributions also drift over a month, which
  affects both sides equally, so the statistic is a difference in differences: pooled (adjacent
  same-side rate / distant same-side rate) ÷ (adjacent opposite-side rate / distant opposite-side
  rate), for lags < 3,600 s. Distant pairs join the first day of each date pair with the first day
  of the next pair. **Gate:** ratio > 1, lower bound > 1, in 2024 and 2021.
- **Chains** (descriptive): rare-size chains linked sign-blind (gap ≤ 300 s, ≥ 3 packets) give
  lifetime and one-sidedness distributions. The comparison chains are built from rare-size
  packets with sizes permuted within 30-minute windows, which keeps the local sign mix. This
  comparison is conservative, because a program that dominates its window survives the
  permutation.

## Module C — metaorder physics of fingerprint-linked campaigns

**Campaigns** (`src_py/evidence_campaigns.py`). Same-side `u_rare` packets with one identical size,
consecutive gaps ≤ 60 s, ≥ 4 packets. The size's base rate λ on the name's other sampled days
(same side) must satisfy λ × 60 s < 0.05, so a chance link has probability below 5%. The number of
chance campaigns implied by λ is reported.

**Per campaign:**
- side, start t0, end t1, n, size;
- child volume Q, same-side and opposite-side packet volume in [t0, t1], and daily volume;
- daily volatility from 5-minute mid returns;
- mids before t0, after t1, and at t1 + 1, 5, 15, 30, 60 minutes (the pre-trade mid of the first
  packet after each time);
- the mid at t0 − 5 minutes;
- same-side packets of other sizes in the 5 minutes before and after t1.

**Placebos:** windows of the same duration on the same name-day and side, starting at a random
same-side packet and not overlapping any campaign. They carry the same measurements.

**Pre-declared expectations** (descriptive, no gate; each reported as holds or does not hold):
- **C1.** Impact during the campaign, in daily-volatility units, rises concavely with Q / daily
  volume. The exponent, fit to binned means, lies in [0.3, 0.7].
- **C2.** Impact at t1 + 30 min divided by impact at t1 lies in [0.4, 0.9]: partial reversion
  after completion.
- **C3.** Campaigns and placebos are compared in the same bins of duration and same-side volume
  share, for C2's ratio and for the drop in other-size same-side flow after t1. Two-sided; both
  are reported.

## Module D — external validation with WRDS

WRDS tables used: CRSP daily stock file, CRSP S&P 500 membership, Compustat index constituents
(Nasdaq-100), CRSP mutual-fund holdings and monthly TNA, ETF Global constituents and fund flows,
the TAQ-derived WRDS Intraday Indicators (retail and large-trade buy and sell volume), and the TAQ
master file.

Thomson 13F (`tfn.s34`) and FactSet ownership are not licensed to this account. Mutual-fund
holdings substitute for 13F.

**Daily flow variables** (contiguous panels, `src_py/daily_flow.py`):
- PI = (program-burst buy volume − program-burst sell volume) / total signed packet volume;
- NPI = the same for all other signed volume;
- shares-outstanding-scaled versions for quarterly sums.

**Tests.** D1 and D3 are two-sided and exploratory in 2024, then replicated in 2021. D2 is
descriptive. D4 is one shot.

- **D1 (mutual-fund trading).** Cross-section by quarter: ΔMF_q (net shares bought by funds
  reporting at both quarter ends, split-adjusted, / shares outstanding) regressed on NP_q and NNP_q
  (quarter sums of program and other net buying / shares outstanding), with quarter fixed effects
  and name clusters.
  - Prediction: β_NP > 0 and β_NP > β_NNP.
  - Second form: flow-induced trading (Lou 2012, from each fund's prior holdings and quarterly
    flow) as the regressor for NP_q and NNP_q.
- **D2 (retail and large-trade flow).** Daily cross-sectional Spearman correlations of PI and of
  NPI with BJZZ retail imbalance and with ≥ $50k-trade imbalance, averaged over days (NW t).
  - Descriptive, two-sided. Wholesalers hedging retail flow on lit venues could make program flow
    retail-aligned, so no direction is assumed.
- **D3 (ETF basket demand).** Implied demand for stock i on day t: Σ over ETFs of prior-day
  weight × dollar fund flow, divided by market cap. ETFs are the 40 largest US equity ETFs by
  assets at the start of each year.
  - Fama–MacBeth regression of PI_t and of NPI_t on implied demand at t and at t + 1 (flows are
    reported a day late).
  - Prediction: the PI slope exceeds the NPI slope at t or t + 1.
- **D4 (index events, one shot).** Every S&P 500 and Nasdaq-100 addition and deletion effective in
  2021 or 2024, regardless of name split.
  - PI and NPI are z-scored against the name's own days E−40 to E−11 (E is the first membership
    day). Take the mean z over E−5 to E−1.
  - **Gate:** additions minus deletions > 0 for PI (Welch t > 2).
  - Reported beside it: the same for NPI, and the PI − NPI contrast.

## Module E — the original idea: daily program imbalance

On the contiguous panels:

- **E1 (persistence).** Fama–MacBeth regression of PI_{t+1} and NPI_{t+1} on PI_t and NPI_t.
  **Gate:** the PI→PI slope exceeds the NPI→NPI slope, t > 3 in 2024 and t > 2 in 2021.
  Multi-day parents predict this.
- **E2 (returns).** Fama–MacBeth regression of r_{t+1} on PI_t, NPI_t and r_t, plus log market cap.
  r_{t+1} is CRSP close-to-close and open-to-close, and the regression is repeated for r over
  t+1..t+5.
  - Exploration gate: |t(PI)| > 3 at any horizon (two-sided).
  - Confirmation: same sign and t > 2.
  - If exploration fails, 2021 is still reported but nothing is claimed.
- **E3 (only if E2 confirms).** Daily quintile long–short on PI_t, net of half-spread costs times
  turnover.

## Module F — who pays for program flow: liquidity-provider markouts

Untruncated single-level packets fill at the touch, so the liquidity provider's markout is exact:
π_h = half-spread − side × (mid_{t+h} − mid_t) / mid_t, in bps, for h = 1, 10, 60 and 300 s. Mids
are the pre-trade mid of the first packet after t + h.

- **F1.** Packets in program bursts minus packets in bottom-quintile bursts, within spread
  deciles, equal weight per name-day, Newey–West over days.
  - Exploration (2024): two-sided, |t| > 3 at h = 60 s.
  - Confirmation (2021): same sign, t > 2.
- The same contrast for packets not in any burst is descriptive. Burst scores use whole-burst
  features, so F describes toxicity. It is not a real-time signal.

## Module G — basket programs across names

Per date, all names together (`src_py/evidence_sync.py`), plus QQQ and SPY extracted for the same
dates.

- **Synchrony ratio.** Same-side `u_nonround` packet pairs from two different names with
  |Δt| < 1 ms (100 µs and 10 ms reported), divided by the mean count at offsets ±0.731, ±2.371 and
  ±7.129 s (non-integer, so clock effects do not enter the baseline). Opposite-side pairs are
  reported separately.
- **G1.** Name-pair-bootstrap difference between the synchrony ratio of program-burst packets and
  of other packets. **Gate:** > 0, lower bound > 0, in 2024 and 2021.
- **G2.** Pair-level log synchrony ratio regressed on ETF weight overlap (Σ min weights over the
  40 ETFs) and same two-digit SIC. **Gate:** overlap slope t > 3 in 2024, same sign and t > 2 in
  2021.
- Synchrony with QQQ and SPY trades is descriptive.

## Module H — the passive side

New extraction from raw messages on the fingerprint dates (`src_py/passive_stats.py`).

**Adds** are type-1 messages with non-round sizes. Adds that are the add half of a same-timestamp,
same-side delete + add (an ITCH replace) are marked replace-adds and excluded from H1.

- **H1.** Identical-size same-side add pairs at 0.5–10 s against the cross-day null (same side,
  same clock lag, same price-position class: inside the spread, at the touch, or behind it).
  **Gate:** name-median ratio lower bound > 1, both years.
- **H2 (one program, both styles).** Pairs of an untruncated non-round aggressive packet and a
  non-replace add of the identical size on the same economic side (aggressive buy with bid add),
  at |lag| in [0.5, 10) s in either order, against the cross-day null. **Gate:** name-median ratio
  lower bound > 1, both years.
- Replace-chain lengths, size retention and fill outcomes are descriptive.

## Module I — dollar-sized children

If an algorithm fixes child notional, shares ≈ N / price, so when the mid rises the next child has
fewer shares. Take same-side `u_nonround` pairs at lags 0.5–10 s in the same depth quartile, with
Δs = s_j − s_i and Δm the mid change. Pairs are near (1 ≤ |Δs| ≤ 3) or far (10 ≤ |Δs| ≤ 30) and
are stratified by expected share change e = s_i |Δm| / m_i in [0, 0.1), [0.1, 0.5), [0.5, ∞).

- D = P(Δs < 0 | Δm > 0) − P(Δs < 0 | Δm < 0).
- **I1.** D_near − D_far at e ≥ 0.5 is > 0 for buys **and** for sells: pooled, name-bootstrap lower
  bound > 0, in 2024 and 2021.
- A depth-censoring artifact would give opposite signs for buys and sells, so the gate requires
  both.

## Module J — change over time, and the Tick Size Pilot

- **J1 (descriptive).** Fingerprint-v1's C1 ratio, run/60 J and program-burst volume share for
  2013, 2016, 2019, 2021 and 2024, on fingerprint names present on at least 16 of 20 sampled dates
  in every year. There are 10 adjacent date pairs per year in the fingerprint-v1 months. The
  program score is transported from 2024, which is descriptive only.
- **J2 (exploratory, no gate).** lobster2 carries only ~140 pilot and ~130 control names, on some
  dates.
  - Design: difference in differences between pilot groups and control, April–September 2016
    versus November 2016–June 2017.
  - Outcomes: fingerprint ratio, untruncated share, burst J, and the ratio of the 3-minute
    run-burst markout to the half-spread.
  - The last outcome is the instrumented spread-scaling slope from the pilot's exogenous tick
    change, which `LLM_README.md` §4 lists as the missing causal test.

## Amendments before any daily-flow output existed (2026-09-14)

1. **Comparable flow units for "program exceeds other" contrasts.** PI and NPI are shares of one
   total, so any demand spread evenly over all volume loads on NPI more, simply because other flow
   is most of the volume. Contrasts in D2, D3 and E1 therefore also use within-type imbalances:
   - PIR = (program buy − program sell) / program volume;
   - NPIR = the same for other volume.

   E2 keeps PI and NPI as declared. D1 is in shares per share outstanding, which is already
   comparable per share traded.
2. **D1 fund universe.** A fund enters a quarter only if it filed holdings at both quarter ends.
   Filers are taken from all of its holdings, not only positions in sample names, so full exits
   count as sales.
3. **D1 flow-induced trading.** Fund flow = (TNA_Q − TNA_P × (1 + R_q)) / TNA_P, with share classes
   summed to portfolios and flows winsorized at the 1st and 99th percentile each quarter. FIT_i =
   Σ_f shares held at P × flow / shares outstanding.
4. **Coverage.** A name-quarter needs flow data on ≥ 80% of its trading days, and quarter sums are
   scaled up by trading days / observed days. Coverage of the point-in-time universe by lobster2 is
   reported, and returns of covered and uncovered names are compared.
5. **G2 ETF weights.** Only ETFs with constituent weights on the first trading day of the year
   enter the overlap: 29 of 40 in 2021 and 23 of 40 in 2024. D3 uses same-day weights with the
   day's dollar fund flow. Duplicate constituent rows are removed before summing.
6. **Name-split correction.** The first point-in-time job files hashed the full sha256 digest.
   Fingerprint-v1 uses the first 8 hex digits, so names used to fit the program score could have
   landed in the 2021 confirmation panel. Fixed after only packet extraction had run; no flow output
   existed. Contiguous groups are now 179 names (2024 exploration) and 319 names (2021
   confirmation); see `freeze_v2c.json`.

## Post-hoc module B4, added after reading the 2021 module B output (2026-09-14)

**What the read showed.** In 2021, opposite-side identical-size matches exceed the depth-matched
cross-day null by 1.35–1.42× at 0.5–10 s, decaying to 1.13× at 30–60 minutes. In 2024 the excess is
1.03–1.11× at ≤ 5 s and about 1.00 beyond. That is not what two-sided algorithms alone would
produce, and it has a competing explanation: **dollar-sized orders from unrelated traders.** Share
counts coincide only when prices are close, and the cross-day null compares different price levels.
A common size state inflates matches on both sides, and the same-side fingerprint too.

**Test** (`src_py/evidence_price_null.py`; synthetic checks in `tests/test_evidence.py`). Within-day
and cross-day pairs must share a price bucket (10 and 50 bps) as well as a depth quartile. Two nulls:
- the same day's price-matched match rate at lags ≥ 1 hour;
- the price-matched cross-day rate, where adjacent days' prices overlap.

Statistic: name-median observed/null for same-side and opposite-side `u_nonround` pairs at 2–10 s,
10–60 s and 60–600 s, both years.

**Predictions, written before any B4 output:**
- If a dollar-sizing state explains the 2021 opposite-side excess, the 10 bps within-day
  price-matched opposite-side ratio at 2–60 s falls below 1.10 in 2021, while the same-side ratio
  stays above 1.5.
- If two-sided algorithms explain it, the opposite-side ratio stays at or above 1.2.
- Anything in between is reported as mixed.

This test is post hoc and labelled so wherever it is cited. The long-lag null absorbs programs
lasting hours, so it is conservative for the same side.

## Order of execution

1. Stage 3b (run/60 score) and modules A, B, I on the cached panels. Freeze, then read 2024.
2. Modules C, F, G, H. Freeze, then read 2024.
3. Read the 2021 confirmation for 1–2 once.
4. Point-in-time universes and contiguous extraction, then daily flows, then D and E (2024 then
   2021), then D4 once.
5. J1 and J2.

## What this cannot establish

- None of these tests labels an individual burst as institutional. Mutual-fund holdings, ETF
  flows and index events are *known institutional demand*, so a positive association shows
  program flow carries it; it does not show that every program is an institution.
- Hedging flow from wholesalers and market makers can be directional over minutes. One-sidedness
  (B) rules out two-sided quoting algorithms, not every intermediary.
- NASDAQ is a minority of US volume; flow measured here is a venue sample of each parent.
