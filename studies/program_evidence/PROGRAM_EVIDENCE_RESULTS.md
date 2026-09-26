# Program-evidence-v1 results

Completed 2026-09-14. Design, predictions and every amendment: `studies/program_evidence/PROGRAM_EVIDENCE_DESIGN.md`. Code
hashes: `results/program_evidence_v1/freeze_v1*.json`–`freeze_v4_flows.json`. Follow-on work
(child-cluster rule, linkage score, calibration, passive simulation): `studies/metaorder/METAORDER_RESULTS.md`.

## Bottom line

1. **Fingerprint bursts are real algorithm activity.**
   - Sub-second timer phase and identical child sizes pick out the same pairs (A1, A2 pass in both
     years).
   - Same-size flow is directional at every horizon to an hour and persists for hours (B1, B2).
   - The fingerprint survives depth and price matching (B4).
2. **The algorithms are mostly intermediaries, not institutional parents.**
   - In 2021, two-sided fixed-clip algorithms were common (B4).
   - Program-like bursts are cheap for resting liquidity to absorb (F1; +0.8 bps within truncation
     strata).
   - They synchronize across stocks within a millisecond (G1).
   - Program flow on full-year point-in-time panels persists less than other flow (E1 reversed) and
     ignores mutual-fund trading (D1). It sells into index-addition demand (D4), leans slightly
     toward retail and away from block flow (D2), and tracks same-day ETF creation and redemption
     demand (D3, replicated).

   That profile is ETF and basket arbitrage, retail hedging and liquidity-taking market making.
3. **The original trading idea is null.** Daily program buy-minus-sell imbalance predicts no
   returns (E2) on a clean point-in-time design, in 2024 or 2021.
4. **Detection is limited by construction.**
   - One best boundary does not exist (A3 fails).
   - Dollar-sized children are untestable here (I1).
   - Linked 4–5-child chains are fragments without measurable impact physics (C).
   - Passive and aggressive children share sizes only slightly (H2).
5. **Two results beyond bursts.**
   - The identical-size fingerprint roughly halved between 2016 and 2024 (J1).
   - The Tick Size Pilot widened spreads by 9 bps without moving burst markouts (J2): the
     spread-scaling law is not causal compensation (exploratory).

## Samples

| panel | names | name-days | signed packets |
|---|---|---|---|
| exploration, 2024 (fingerprint-v1 cache) | 174 | 3,480 | 18.5M |
| confirmation, 2021 (fingerprint-v1 cache) | 291 | 5,702 | 36.1M |

Order note: the 2021 read for modules A, B and I happened after all stage-2 code
(modules C, F, G, H, D, E, J) was frozen, rather than after the stage-2 exploration read the design
listed. No frozen choice could be affected.

## Stage 3b — program score for the working definition (run, 60 s cap)

Refit of the fingerprint-v1 score on 2024 run/60 bursts (836,762), evaluated once on 2021
(1,554,369 bursts, 291 names).

| gate | result |
|---|---|
| **P1b** top − bottom decile excess repeats per 1,000 pairs, lower bound > 0 | **PASS**: +190 [151, 234] (51 → 241) |
| **P2b** Spearman(decile, excess) > 0.7 | **PASS**: 1.00 |

The largest standardized coefficients are:
- truncated share (−0.80; programs rarely exhaust the touch);
- time of day (+0.73 linear, −0.81 quadratic);
- log executed-side depth (−0.38);
- packet count (+0.26);
- median inter-arrival gap (+0.21);
- spread (+0.23).

"Program bursts" (score ≥ 2024 80th percentile) and "bottom bursts" (≤ 20th) use this model
(`results/program_evidence_v1/program_model_run60.json`).

## Module A — an independent timing fingerprint

Same-side pairs at lags 0.5–60.5 s. Concentration = share of pairs within ±δ of a whole-second lag,
over the uniform share. Entries are name medians with 95% name-bootstrap intervals.

| statistic (δ = 10 ms unless noted) | 2024 | 2021 |
|---|---|---|
| **A1** same-day vs cross-day concentration, non-round untruncated | **1.27 [1.22, 1.31]** | **1.14 [1.12, 1.16]** |
| A1 at δ = 1 ms | 1.68 [1.55, 1.79] | 1.39 [1.33, 1.48] |
| **A2** identical-size vs different-size concentration, same day | **1.62 [1.52, 1.71]** | **1.19 [1.15, 1.23]** |
| A2 at δ = 1 ms | 2.26 [2.06, 2.49] | 1.50 [1.32, 1.61] |
| A1, all signed packets | 1.15 [1.14, 1.18] | 1.13 [1.12, 1.15] |
| A1, opposite-side pairs | 1.03 [1.00, 1.05] | 1.03 [1.01, 1.05] |
| **A3** Spearman(phase J, size J), 33 definitions | **0.60 [0.40, 0.78]** | **0.76 [0.59, 0.85]** |

Gates: **A1 PASS** and **A2 PASS** in both years. **A3 FAILS**: 0.60 in 2024 is below 0.7, even
though 2021 passes.

**Findings.**
1. **Timers mark common origin, and they agree with sizes.** Same-side lags cluster within
   milliseconds of whole seconds more on the same day than across days. Identical-size pairs,
   which the size fingerprint says are enriched for one origin, are phase-locked 1.2–1.6× more
   than different-size pairs at the same lags. Opposite-side pairs are barely locked (1.03). Two
   signatures built from unrelated information (sizes and sub-second timing) pick out the same
   pairs.
2. **Locking is sharp.** The excess grows as the window narrows from 50 ms to 1 ms, so it is
   millisecond-scale timer structure, not a smooth lag preference.
3. **The two fingerprints disagree about burst boundaries.**
   - Size evidence ranks uninterrupted same-side runs (30–300 s caps) first in both years; run/60
     ranks 3rd (2024) and 1st (2021).
   - Timing evidence ranks side-only streams and sign-blind timing clusters with 2–10 s gaps first;
     run/60 ranks 9th and 8th.

   Phase-locked evidence is concentrated at short lags and includes programs that randomize sizes,
   which dense short-gap clusters capture. No single definition is best by both fingerprints.
4. **Wall-clock anchoring is modest.** Packet timestamps within 10 ms after a whole second are 1.2×
   the uniform share in both years. The most common offset moved from 13–19 ms after the second
   (2021) to 3–10 ms (2024), consistent with lower latency in 2024.

## Module B — are same-size algorithms directional?

Identical-size matches over the depth-matched cross-day null with lag-specific rates. Name medians,
non-round untruncated packets.

| lag | 2024 same side | 2024 opposite | 2021 same side | 2021 opposite |
|---|---|---|---|---|
| 0.5–1 s | 2.10 | 1.11 | 2.47 | 1.37 |
| 2–5 s | 2.18 | 1.08 | 2.15 | 1.42 |
| 10–30 s | 1.69 | 1.03 | 1.83 | 1.33 |
| 60–120 s | 1.43 | 1.00 | 1.60 | 1.26 |
| 5–10 min | 1.22 | 1.01 | 1.46 | 1.18 |
| 10–20 min | 1.17 | 1.00 | 1.39 | 1.17 |
| 30–60 min | 1.11 | 1.00 | 1.31 | 1.13 |
| 1–2 h | 1.08 | 1.01 | 1.26 | 1.13 |
| 4–6.5 h | 1.03 | 1.01 | 1.10 | 1.06 |

| gate | 2024 | 2021 |
|---|---|---|
| **B1** same − opposite ratio > 0 at 2–10 s, 10–60 s, 60–600 s, 600–3,600 s | **PASS**: 1.05 [0.87, 1.23]; 0.55 [0.47, 0.70]; 0.29 [0.24, 0.34]; 0.13 [0.10, 0.16] | **PASS**: 0.55 [0.43, 0.66]; 0.33 [0.28, 0.39]; 0.22 [0.16, 0.26]; 0.11 [0.08, 0.14] |
| **B2** same-side ratio > 1 at 10–30 min and 30–60 min | **PASS**: 1.15 [1.13, 1.18]; 1.11 [1.10, 1.14] | **PASS**: 1.37 [1.32, 1.44]; 1.31 [1.26, 1.36] |
| **B3** multi-day DiD > 1 | **PASS**: 1.014 [1.006, 1.025] | **FAIL**: 1.006 [0.995, 1.018] |

B3 fails overall (not replicated).

**Findings.**
1. **Same-size flow is directional at every horizon, in both years.** Same-side excess exceeds
   opposite-side excess from half a second to an hour. The same-side excess persists for hours:
   1.11–1.31 at 30–60 minutes, fading toward 1 by the end of the day. That is the lifetime profile
   of a persistent one-sided program, not of a seconds-long quoting loop. It fits a working parent
   order, but equally a persistent intermediary such as a retail-flow hedger; modules D and E show
   the high-scoring programs are mostly the latter.
2. **2024 is almost purely one-sided; 2021 is not.** The one-sidedness index is 0.83–0.98 in 2024
   and 0.55–0.75 in 2021. The 2021 opposite-side excess (1.3–1.4 at seconds) is large, but a
   two-sided algorithm is not its only explanation. Unrelated dollar-sized orders on both sides
   also share share counts when prices are close. Post-hoc module B4 tests which it is (see below).
3. **Rare sizes are partly reused on both sides** (one-sidedness 0.6–0.7 at minutes, 2024). A
   distinctive size is a weaker directional label than a typical non-round size.
4. **Multi-day campaigns with a fixed child size are not detectable.** Adjacent days share
   same-side sizes 1.4% more than days a month apart in 2024, which did not replicate.
5. **Chains.** Rare-size chains with ≥ 5 packets, linked sign-blind within 300 s:

   | | real chains | size-permuted chains |
   |---|---|---|
   | ≥ 90% one-sided, 2024 | 21% | 6% |
   | ≥ 90% one-sided, 2021 | 17% | 6% |
   | lasting under a minute, 2024 | 889 | 54 |

## Module I — dollar-sized children

**I1 FAILS in both years.** Near-minus-far D at e ≥ 0.5 shares:

| side | 2024 | 2021 |
|---|---|---|
| buys | −0.09 [−0.33, 0.06] | +0.01 [−0.23, 0.23] |
| sells | +0.10 [−0.13, 0.26] | −0.07 [−0.31, 0.19] |

The test has no power where it matters. Only 108–209 near pairs per side have an expected
share change of half a share or more. Untruncated child sizes are small and ten-second price
moves are a few basis points, so a fixed-notional child almost never changes its share count
within ten seconds.

## Module B4 (post hoc) — is the 2021 opposite-side excess a price-level artifact?

Written after the 2021 module B read; predictions fixed in the design before any output. Pairs
must share a 10 bps price bucket (50 bps reported in `b4_*.json`) as well as a depth quartile.

| name-median observed/null, 10 bps | 2024 same | 2024 opposite | 2021 same | 2021 opposite |
|---|---|---|---|---|
| 2–10 s vs same-day price-matched ≥ 1 h | 1.95 [1.78, 2.11] | 0.85 [0.79, 0.89] | 1.79 [1.73, 1.93] | **1.29 [1.22, 1.39]** |
| 10–60 s vs same-day price-matched ≥ 1 h | 1.49 [1.39, 1.74] | 0.86 [0.81, 0.90] | 1.61 [1.50, 1.72] | **1.23 [1.20, 1.30]** |
| 2–10 s vs cross-day price-matched | 1.93 [1.52, 2.26] | 1.04 [0.82, 1.24] | 2.03 [1.67, 2.32] | 1.24 [1.07, 1.33] |
| 10–60 s vs cross-day price-matched | 1.95 [1.71, 2.41] | 0.96 [0.88, 1.03] | 1.71 [1.60, 1.85] | 1.37 [1.22, 1.47] |
| 60–600 s vs cross-day price-matched | 1.38 [1.29, 1.54] | 0.98 [0.95, 1.04] | 1.70 [1.60, 1.79] | 1.25 [1.20, 1.31] |

**Outcome against the written rule: two-sided algorithms, not a price-level artifact.** In 2021 the
opposite-side excess stays at or above 1.2 when prices match within 10 bps. Rare sizes, which a
common size state cannot explain, show the same pattern (module B).

The same-side fingerprint survives price matching in both years (1.4–2.3 at 2–60 s). In 2024,
opposite-side matches are at or below chance.

**What this means for "program = directional execution".**
- **2024.** Same-size algorithms on NASDAQ were almost entirely one-sided.
- **2021.** A large population of algorithms reused one size on both sides within seconds to
  minutes, the signature of two-sided liquidity-taking strategies (for example market makers or
  arbitrageurs hedging with fixed clips).

Either way, the directional (same-side minus opposite-side) excess is present at every horizon in
both years.

## Module C — metaorder physics of fingerprint-linked campaigns

| | 2024 | 2021 |
|---|---|---|
| campaigns (names) | 3,239 (150) | 5,653 (245) |
| chance campaigns implied by base rates | 3.3 | 3.3 |
| median children / duration | 4 / 24 s | 5 / 47 s |
| median child volume / daily volume | 0.020% | 0.034% |
| price move in the campaign's direction over the 5 minutes before it | +4.4 bps | +2.1 bps |

- **C1 does not hold.** Impact during a campaign (in daily-volatility units) against child volume
  / daily volume gives exponents of −0.23 [−0.48, 0.12] (2024) and 0.19 [−0.02, 0.46] (2021),
  outside [0.3, 0.7]. Against all same-side volume in the window: 0.71 [0.46, 1.09] and
  0.90 [0.63, 1.32], also outside.
- **C2 is uninformative.** The ratio of impact 30 minutes after completion to impact at completion
  is 0.74 (2024) and 0.61 (2021), inside [0.4, 0.9]. But the name-bootstrap intervals are
  [−2.3, 4.5] and [−0.3, 1.4]. Impact keeps growing for 1–5 minutes after the last linked child
  (ratio 1.4–1.9 at 1–5 min), then decays.
- **C3 (matched duration × same-side-share cells).**
  - Other-size same-side flow falls *less* after a campaign ends than after a placebo window ends
    (+0.03 to +0.04 on a [−1, 1] scale, both years). The end of an identical-size chain is
    therefore not the end of the parent: the program keeps trading in other sizes.
  - Campaigns start after a move in their direction (+2.8 bps more than placebos, both years).
  - Completion-to-30-minute impact differences are unstable in sign and size.

The campaign linking is not chance (3 expected against thousands observed). But linked chains of
four to five identical children are too small a slice of a parent for square-root impact or
completion reversion to be measurable. This matches the old corroboration: burst impact exponent
0.26, "a fragment of a parent".

## Module F — who pays for program flow

Liquidity-provider markouts on untruncated (touch-filled) packets, by the run/60 program score of
the burst each packet belongs to, within spread deciles. Mean over days, Newey–West (2 lags, 20
days) and name bootstrap.

| contrast (bps) | 2024, 60 s | 2021, 60 s |
|---|---|---|
| **F1** program bursts − bottom-quintile bursts | **+2.51, t = 31.7, [2.24, 2.84]** | **+2.28, t = 25.7, [2.10, 2.61]** |
| program bursts − packets outside bursts | −0.77 [−0.88, −0.66] | −0.97 [−1.19, −0.90] |
| bottom bursts − packets outside bursts | −3.23 [−3.57, −2.90] | −3.22 [−3.61, −3.06] |

Mean markout levels at 60 s:

| | program bursts | middle | bottom bursts | outside bursts |
|---|---|---|---|---|
| 2024 | +0.70 | −1.02 | −1.72 | +1.58 |
| 2021 | +0.58 | −1.04 | −1.64 | +1.75 |

**Gate: F1 PASS** (2024 |t| > 3; 2021 same sign, t > 2). The same ordering holds at 1, 10 and 300 s.

**Reading.** Liquidity providers make money against program-like bursts and lose against low-score
bursts. That is the same direction as the surprise-flow result (§1.26): anticipated flow is cheap
to absorb.

**Caveat.** The score's largest coefficient is negative truncation share, and low-score bursts are
the ones that exhaust the touch. Part of this contrast is therefore the mechanical cost of
book-sweeping bursts. Volume shares: program bursts 5–6% of signed volume, bottom 33–35%, outside
bursts 34–37%.

**Post-hoc F2 (added after the F1 read): the contrast within burst truncation-share strata.**
Program-minus-bottom markout at 60 s, averaged over the strata where both groups occur, equal
weight per name-day:

| | 2024 | 2021 |
|---|---|---|
| all strata | **+0.82 bps**, t = 5.2, [0.46, 1.20] | **+0.82 bps**, t = 7.8, [0.72, 1.38] |
| truncation share 0.25–0.5 | +0.88 [0.43, 1.68] | +0.60 [0.23, 1.49] |
| truncation share 0.5–0.75 | +0.72 [0.43, 1.97] | +0.82 [0.55, 1.81] |

About two-thirds of the headline gap is book-sweeping. A residual of about 0.8 bps survives in
both years among bursts that exhaust the touch equally often. The groups barely overlap:
- 70% of program-burst packets sit in bursts with truncation share below 0.25;
- 87–90% of bottom-burst packets sit above 0.5.

Only the middle strata compare like with like.

## Module G — basket programs across names

Pairs of untruncated non-round packets from different names within ±1 ms, over the mean count at
±0.731, ±2.371 and ±7.129 s offsets.

| synchrony ratio | 2024 (174 names) | 2021 (291 names) |
|---|---|---|
| same side, 1 ms | 18.7 | 28.1 |
| same side, 0.1 ms | 32.3 | 48.9 |
| opposite side, 1 ms | 4.4 | 5.1 |
| first packet in a program burst | 31.9 | 34.4 |
| first packet not in a program burst | 15.8 | 26.8 |
| both packets in program bursts | 76.0 | 50.8 |

- **G1 PASS** in both years. Program minus other synchrony: 16.1 [10.7, 22.4] (2024) and
  7.6 [4.3, 12.1] (2021).
- **G2 FAILS.** The ETF-overlap slope is positive but below the exploration hurdle: t = 2.81 in
  2024 against the pre-declared 3; 2021 has t = 2.26. Same two-digit SIC is positive in both
  years (t = 2.94, 3.42).

**Reading.** Aggressive orders in different stocks arrive within a millisecond of each other 19–28×
more often than chance, and mostly on the same side. Program-like bursts synchronize about twice as
much as other flow (2024). This is the footprint of basket and index-arbitrage execution: one
decision sent to many names at once. The basket structure lines up with shared industry more
clearly than with shared ETF membership. Descriptive synchrony with ETFs at 1 ms:
- SPY: 13.8 (2024);
- QQQ: 33.8, TQQQ: 34.8, SMH: 49.9 (2021);
- SQQQ, the inverse ETF: 4.5 (2021).

## Module H — the passive side

New extraction of non-round limit-order submissions (type-1 adds) on the fingerprint dates. Replace
halves (same-timestamp delete + add) and remainders posted with an execution are excluded.
Observed identical-size matches over the cross-day null at 0.5–10 s; name medians.

| statistic | 2024 (174 names) | 2021 (291 names) |
|---|---|---|
| **H1** same-side adds with identical sizes (position class matched) | **1.75 [1.60, 1.90]**, 99% of names > 1 | **2.43 [2.25, 2.78]**, 98% > 1 |
| **H2** aggressive child and same-side add of identical size, either order | **1.10 [1.08, 1.11]** | **1.13 [1.11, 1.15]** |
| H2 control: opposite-side add | 1.05 [1.04, 1.05] | 1.08 [1.06, 1.11] |
| H2 same side ÷ opposite side | 1.05 [1.03, 1.06] | 1.03 [1.02, 1.04] |

**Gates: H1 PASS and H2 PASS in both years.**

**Reading.**
- Passive order flow carries a strong same-origin size fingerprint, as expected: quoting and
  passive execution algorithms repost fixed clips.
- The link between styles is real but small. An aggressive child shares its exact size with a
  same-side passive order posted within ten seconds 10–13% more often than chance. Part of that is
  a common size state, which the opposite-side control shows at 5–8%. The same-side increment
  beyond it is 3–5%.

Some programs mix passive and aggressive children of the same size, but most aggressive bursts
detected here are not visibly paired with passive clips of the same size. The passive side of a
parent cannot be recovered through size matching alone.

## Modules D and E — what program flow is, on full-year point-in-time panels

**Panels.**
- 2024 exploration: 163 of 179 point-in-time names had lobster2 data (39,121 name-days).
- 2021 confirmation: 223 of 321 (55,426 name-days).
- Program bursts carry a median 5.5–5.7% of signed volume.

**Coverage bias.** Names with archive data returned 16.5% (2024) and 26.4% (2021) over the year,
against 4.3% and 1.8% for names without. lobster2's download list is tilted toward names that did
well, which is a caveat for any return test on it.

Fama–MacBeth slopes, Newey–West (10 lags) t-statistics.

| test | 2024 (exploration) | 2021 (confirmation) | verdict |
|---|---|---|---|
| **E1** persistence: PI→PI minus NPI→NPI | −0.119, t = −8.7 (0.21 vs 0.33) | −0.067, t = −7.1 (0.18 vs 0.25) | **FAIL**, reversed in both years |
| E1 within type: PIR→PIR minus NPIR→NPIR | −0.139, t = −11.2 | −0.088, t = −8.4 | same |
| **E2** next-day close-to-close return on PI | t = 0.15 | t = 0.19 | **FAIL** (null) |
| E2 next-day open-to-close / next 5 days | t = 0.02 / −0.17 | t = −1.41 / −1.49 | null |
| E2 per 1 s.d. of PI, next day | −0.5 bps (t = −0.4) | +0.3 bps (t = 0.3) | null |
| **D1** Δ mutual-fund holdings on NP, NNP (shares/shares out) | NP −0.74 (t = −0.6); NNP +1.17 (t = 3.4) | NP +2.07 (t = 1.9); NNP +0.70 (t = 2.4) | **not supported** (difference t = −1.3, +1.1) |
| D1 flow-induced trading → NP / NNP | 0.005 (t = 0.3) / −0.019 (t = −0.3) | 0.032 (t = 2.9) / −0.037 (t = −1.1) | not replicated |
| **D2** corr(PIR, retail) − corr(NPIR, retail) | +0.028, t = 3.9 | +0.012, t = 2.4 | program flow slightly **more** retail-aligned |
| D2 corr(PIR, ≥ $50k trades) − corr(NPIR, ≥ $50k) | −0.071, t = −10.5 | −0.019, t = −3.1 | program flow **less** large-trade-aligned |
| **D3** PIR − NPIR slope on same-day ETF basket demand (z) | +0.020, t = 7.3 | +0.017, t = 5.5 | **replicates**: program flow tilts with ETF creations and redemptions |
| D3 on next-day demand | +0.003, t = 0.9 | +0.009, t = 4.2 | 2021 only |

E1 exploration's |t| > 3 is in the wrong direction and E2 exploration never reached |t| > 3, so
neither gate was met; 2021 is reported, nothing is claimed. D1 and D3 were two-sided (explore,
then replicate).

**D4 — index changes (one shot, 68 of 103 events usable).**
- Unusable events: 25 had no archive data, 10 had too few days.
- Mean z-score over E−5 to E−1 against each name's own E−40 to E−11:

| | additions (39) | deletions (29) | additions − deletions |
|---|---|---|---|
| program imbalance PI | −0.24 (t = −2.0) | +0.01 | **−0.25, Welch t = −1.45** |
| other imbalance NPI | +0.24 (t = 2.5) | −0.10 | +0.34, t = 2.07 |
| PI − NPI | −0.48 (t = −3.7) | +0.11 | −0.59, t = −2.94 |

**Gate D4 FAILS**: wrong sign. Before an index addition, non-program flow buys, as expected. Program
flow leans the other way, selling into that demand.

**What D and E establish.** "Program-like" bursts are not a proxy for fundamental institutional
parents. Program flow:
- persists less from day to day than other flow;
- carries no return information;
- does not track mutual-fund trading;
- leans against index-addition demand;
- aligns slightly with retail flow and less with large trades.

What it does track, in both years, is same-day ETF creation and redemption demand. Together with
the millisecond cross-stock synchrony (module G) and the two-sided clips of 2021 (module B4), this
points to **intermediation and arbitrage programs**:
- ETF and basket arbitrage;
- hedging of internalized retail flow;
- liquidity-taking market-making algorithms.

The fingerprint and score find algorithms well; the algorithms they find mostly belong to
intermediaries. The original trading idea (daily program buy-minus-sell imbalance predicts the move)
is null on a clean point-in-time design in both years.

## Post hoc (2026-09-14, after D1 and D4 were read): which flow component carries institutional demand?

Exploratory, not pre-registered (`src_py/posthoc_components.py` → `posthoc_components.json`). Daily
signed volume is split into four components. Median shares of signed volume:

| component | share of signed volume |
|---|---|
| program bursts (top-quintile score) | 5.5% |
| middle bursts | 27–29% |
| bottom bursts (bottom quintile: larger children, 76% truncated) | 32–34% |
| trades outside any burst | 32% |

| test | program | middle | bottom | outside bursts |
|---|---|---|---|---|
| quarterly Δ mutual-fund holdings, joint regression, 2024 | 0.83 (t = 0.3) | 0.07 (t = 0.1) | 2.03 (t = 1.7) | 0.73 (t = 0.6) |
| same, 2021 | 0.75 (t = 0.4) | 1.82 (t = 2.1) | 0.54 (t = 0.7) | −0.35 (t = −0.4) |
| index changes, additions − deletions z, E−5..E−1 (68 events) | −0.25 (t = −1.45) | +0.06 (t = 0.3) | **+0.49 (t = 2.82)** | +0.24 (t = 1.4) |

**Reading.**
- *Mutual-fund trading.* No component tracks it consistently; the components are collinear and there
  are fewer than 900 name-quarters.
- *Index additions.* The one canonical institutional demand event shows up in the **bottom** bursts
  (larger, book-sweeping children) and leans against the small, regular program-like bursts.

This points opposite to "small, regular slices are institutional". It is a single post-hoc sample
and needs a pre-registered, replicated test before it is cited.

## Module J — change over time, and the Tick Size Pilot

**J1 (descriptive).** The same 474 tickers are sampled on 10 calendar-matched adjacent date pairs
per year. The balanced panel has the 130 names with ≥ 16 usable days in every year. The
fingerprint ratio is a name median (names with ≥ 20 expected matches).

| year | names eligible | ratio 0.5–2 s | ratio 2–10 s | run60 J | untruncated share | program-burst volume share | half-spread (bps) |
|---|---|---|---|---|---|---|---|
| 2013 | 18 | 4.25 [2.98, 7.74] | 3.59 [2.66, 4.65] | 0.32 | 0.45 | 8.4% | 1.70 |
| 2016 | 92 | 5.64 [5.12, 6.15] | 4.11 [3.75, 4.57] | 0.31 | 0.43 | 8.0% | 1.58 |
| 2019 | 91 | 3.36 [2.86, 3.75] | 2.44 [2.03, 2.68] | 0.37 | 0.45 | 7.2% | 1.37 |
| 2021 | 118 | 2.82 [2.59, 3.03] | 2.04 [1.91, 2.24] | 0.27 | 0.40 | 5.6% | 1.57 |
| 2024 | 127 | 2.58 [2.22, 2.94] | 2.33 [2.19, 2.48] | 0.35 | 0.36 | 5.3% | 1.62 |

**Reading.**
- **The fingerprint has weakened.** Fixed-clip repeats were about twice as strong relative to chance
  in 2016 as in 2021–2024.
- **Burst quality has not.** How well run60 keeps same-origin pairs together (J 0.27–0.37) shows no
  trend.
- **Programs have become harder to see by sizes, not less common by timing.** More algorithms now
  randomize clips, odd-lot flow has grown, and untruncated packets have fallen from 45% to 36%.
- **Program-burst share of volume** falls from about 8% to 5%. The score is transported from 2024,
  so this is descriptive only.

**J2 (exploratory, no gate) — the Tick Size Pilot as an instrument.** 144 pilot names (groups 1–3)
and 125 controls present in lobster2 with ≥ 8 usable days in both periods: April–September 2016
against November 2016–June 2017. Name-level differences in differences, name bootstrap.

| outcome | pilot change | control change | DiD [95% CI] |
|---|---|---|---|
| mean half-spread, all signed packets (bps) | +8.24 | −1.17 | **+9.41 [6.63, 12.50]** |
| half-spread at run60 burst ends (bps) | +5.84 | −0.58 | +6.43 [1.14, 11.36] |
| untruncated share | +0.114 | +0.019 | +0.096 [0.082, 0.109] |
| fingerprint ratio 2–10 s | +8.4 | −6.9 | +15.3 [2.4, 32.8] |
| run60 J | −0.026 | +0.004 | −0.030 [−0.093, 0.037] |
| 3-minute run60 burst markout (bps) | −1.55 | −0.73 | **−0.83 [−3.15, 1.39]** |

**Instrumented slope** of burst markout on half-spread (pilot minus control, then ratio of DiDs):
**−0.13 [−1.09, 0.29]**. The cross-sectional slope in the same small-cap sample is 0.08.

**Reading.**
- *What the pilot did.* It widened spreads by about 9 bps in treated names, thickened queues (more
  untruncated packets) and made fixed-clip repeats easier to see. It did not change how well bursts
  group same-origin pairs.
- *What it did not do.* The burst's 3-minute directional markout did not rise. If the
  spread-scaling law (`VERIFIED_RESULTS.md` §1.21: signal markout ≈ 0.71 × half-spread) were causal
  compensation, a 6.4 bp rise in the half-spread at burst ends should have raised the markout by
  about 4.5 bps. The 95% interval excludes it.
- *Verdict.* The first causal test of the spread-scaling law points to a common factor or price-grid
  explanation, not informed-flow compensation.
- *Caveats.* Small caps rather than the large-cap universe where the law was measured. A simpler
  markout (run60, from the first packet after the burst) than the 66-definition mk3. Uneven archive
  coverage of pilot names. Exploratory by design.
