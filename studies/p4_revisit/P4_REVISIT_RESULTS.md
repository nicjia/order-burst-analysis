# P4 revisit v1 — results

Design: `studies/p4_revisit/P4_REVISIT_DESIGN.md` (frozen 2026-09-15, amendments A1–A6). Prior work and its defects:
`studies/p4_revisit/PRIOR_WORK_INVENTORY.md`. Jobs and checksums: `RESULTS_PROVENANCE.md`.

**Status (2026-09-15). Complete.**
- Q0 (audit of the legacy pipeline) is complete.
- DEV, VAL, TEST and ERA2 are extracted (1,267,489 name-days present of 2,435,582 requested) and aggregated
  (1,255,983 after the early-close and stub-quote exclusions), spanning 2012–2025 under the corrected
  pipeline (amendment A5).
- VAL, TEST and ERA2 have each been read once, in that order. Verdicts are in **Conclusions across the three
  periods** at the end of this document.

### Amendment A5 and the re-run (read this before comparing to an earlier draft)

The first TEST read produced a −334 bps/day CLOP decile portfolio. The cause was the CRSP split adjustment in
`p4_aggregate.crsp_frame`, applied with the factor ratio inverted, so a split day looked like a −99% gap:
53 of 511,727 TEST name-days had |CLOP| > 0.5 (NVDA 2024-06-07 read −0.999 against a true −0.0043). All four
cells were re-aggregated with the corrected line and re-analysed in protocol order (VAL, then TEST, then ERA2).
Pre-fix outputs are kept as `analysis/*_preA5.json`.

**What the fix touches.** Only the next-day targets: `d_open`, `d_cc`, `phi_open`, `phi_cc`. The Phase II
feature matrix and the `d_close` target are bit-identical before and after, verified directly, so the
DEV-frozen models were not refit and the freeze still holds.

**Statistics that changed** (VAL, corrected vs pre-A5):

| statistic | pre-A5 | corrected |
|---|---|---|
| Q1 trade, info − non, to next open | −1.75 (t −1.48) | −1.83 (t −1.54) |
| Q1 trade, info − non, close-to-close | −3.97 (t −2.68) | −4.04 (t −2.73) |
| Q1 submission, info − non, next open / close-to-close | −0.15 (t −0.13) / −2.32 (t −1.88) | −0.23 (t −0.21) / −2.41 (t −1.94) |
| Q4 trade CLOP, t on S_info | −1.15 | −1.50 |
| **Q4 submission CLOP, t on S_info** | **−2.76**, Holm p 0.0058, t > 3 no | **−3.16**, Holm p 0.0016, **t > 3 yes** |
| Q4 trade CLOP portfolio, gross | +8.25 bps/day (SR 2.49) | +7.83 bps/day (SR 2.46) |

Everything keyed to `d_close` — the Q1 close displacement, the Q2(a) linkage, the Q3 IC on `d_close`, the
tCLOSE cells — is unchanged to float precision. The one consequential change is the last row but one:
**under the corrected VAL, submission CLOP clears the pre-registered Q4 gate**, so Q4 acquired a confirmatory
cell that the pre-A5 draft said did not exist. It is carried to TEST under the pre-registered rule (same sign,
t > 2) and fails there; see the TEST section.

---

## Q0 — audit of the legacy pipeline (complete)

Sample: 2023, NVDA, TSLA, JPM, MS plus 36 random names × 20 random dates. 680 of 800 name-days are present
in the archive. Code: `src_py/p4_q0_legacy.py` replicates `src_cpp/burst.cpp`:
- Hawkes β = 1, trigger 0.3;
- direction by count share ≥ 0.763 and minority volume ≤ 0.28 × majority;
- volume ≥ 0.00197 × the day's executed volume.

It runs on three trade streams. Outputs: `results/p4_revisit_v1/q0/q0_summary.json` (jobs 14752792, 14752986).

### Q0a — hidden prints signed as sells

All regular-hours type-5 messages carry Direction +1 (100.0%); they are 25.5% of executed volume.

| stream | sells among directional bursts (count / volume) | net-short name-days | 1-min hit rate, end mid / start mid | corr(net burst flow, open→close), pooled / within name |
|---|---|---|---|---|
| legacy (type 5 = sell) | 78.5% / 86.4% | **97.6%** | 51.6% / 74.9% | +0.023 / +0.006 |
| type 4 only | 51.5% / 51.1% | 52.5% | 54.8% / 79.1% | +0.017 / +0.049 |
| native packets | 51.0% / 50.1% | 49.0% | 55.4% / 78.6% | −0.003 / +0.019 |

- **Explained.** The legacy signing produces the old 81% net-short tilt, which was never explained at the time.
- **Not explained.** It does not produce the old 37.8% hit rate or the +0.02 same-day correlation.

### Q0b — were the flagship overnight Sharpes market exposure?

Long-only close-to-open Sharpe, 2023–2024, CRSP official prices:

| name | long-only overnight | net of 1 bp | reported legacy strategy |
|---|---|---|---|
| NVDA | 0.45 | 0.42 | 1.58 |
| TSLA | 0.93 | 0.86 | 1.57 |
| JPM | 0.67 | 0.47 | −0.03 |
| MS | 0.97 | 0.82 | −0.47 |

Overnight beta alone does not produce the NVDA and TSLA numbers. Their standing rests on in-sample tuning:
the same names and years were used for the Optuna search (`studies/p4_revisit/PRIOR_WORK_INVENTORY.md` §4 row 4), and they did
not generalize (438 names, mean −0.28).

### Q0c — the κ gate re-measures its own outcome

Three-minute markout from the burst-end mid (bps) for directional legacy bursts:

| stream | all | gated D_b ≥ 0.5 (share kept) | gated D_b ≥ 1.085 (share kept) |
|---|---|---|---|
| legacy | +1.09 | **+11.88** (66%) | +12.00 (65%) |
| type 4 only | +1.54 | +11.61 (68%) | +11.72 (68%) |
| native packets | +1.46 | +11.82 (68%) | +11.94 (67%) |

- **The gate is circular.** Legacy D_b is in share-dollar units, so κ ≈ 1 is effectively "D_b > 0". Keeping bursts whose price went up over the next 1–10 minutes raises the 3-minute markout 8–11×, regardless of signing.
- **It reproduces the April artifact:** +8.76 gated against +0.53 honest.

---

## Coverage

Share of requested point-in-time name-days present in the archive after the renamed-ticker retry, which
recovered none (`src_py/p4_coverage.py`):

| cell | name-days present | names with any data | NYSE-listed | NASDAQ-listed | prior-year-return quintile 1 vs 2–5 |
|---|---|---|---|---|---|
| ERA2 2012–16 | 34.4% | 461 / 1,429 | 40% | 23% | 26.1% vs 37–41% |
| DEV 2017–19 | 55.8% | 208 / 419 | 71% | 27% | 38.5% vs 61–68% |
| VAL 2020–21 | 59.7% | 417 / 779 | 76% | 34% | 58.7% vs 62–75% |
| TEST 2022–25 | 80.2% | 676 / 876 | 80% | 81% | 75.9% vs 82–85% |

The archive under-represents NASDAQ-listed names and prior-year losers, severely in 2012–16 and mildly by
2022–25. Every result below is conditional on that coverage. Two consequences worth stating plainly:

- **The confirmatory cell has the best coverage.** TEST is 80% complete and balanced across listing venues,
  so the replications reported there are not a thin-archive artefact.
- **ERA2 is the weakest cell on coverage** (a third of requested name-days, NASDAQ names at 23%, prior-year
  losers at 26%). Its results are read as a second-era check, not as evidence of equal standing.

---

## DEV 2017–2019 (exploratory; group-0 names; not a test)

Inputs: `agg/DEV/nameday_DEV.csv.gz` sha256 `5e13b1096927` (137,275 name-days, 208 names), after amendments
A3 (early closes, stub quotes), A4 (placebo leak) and A5 (split adjustment). Output: `analysis/DEV_primary.json`
(job 14761321). DEV is where the first two implementation defects surfaced; the numbers before each fix are kept
(`DEV_primary_preA3.json`, `DEV_primary_preA4.json`). The A5 fix moves only the next-day rows below: to the next
open, trade bursts read −2.13 (t −6.7) before and −2.12 (t −6.63) after; close-to-close −1.57 (t −2.9) and
−1.56 (t −2.89).

**Filter selectivity.**
- Share of large eligible bursts passing D_b ≥ 0.5·PeakImpact: trade bursts 58%, submission bursts 53%.
- Pseudo-bursts at random times pass at nearly the same rate.

**Q1: post-decision displacement to the close** (bps; daily mean of name-day means, NW t):

| family | informative | pseudo | non-informative | info − pseudo | info − non |
|---|---|---|---|---|---|
| trade | −1.44 | −0.96 | +0.83 | −0.28 (t −2.4) | −2.25 (t −13.2) |
| submission | −0.68 | −0.63 | +0.92 | −0.05 (t −0.6) | −1.61 (t −8.3) |

Informative minus non-informative to the next open and the next close:
- trade bursts: −2.12 (t −6.63) and −1.56 (t −2.89);
- submission bursts: −1.64 (t −5.30) and −1.41 (t −2.43).

Bursts whose impact persisted over the first ten minutes give part of it back afterwards, relative to
bursts that did not persist. Relative to random-time price paths, which pass the same filter, they are no
better (trade bursts slightly worse).

**Q2(a): directional linkage** (informative − other, same-side minus opposite-side link rate):
- trade bursts: +0.0084 [0.0035, 0.0132] (199 names);
- submission bursts: −0.0073 [−0.0115, −0.0032].

**Q4: daily signals.**
- No informative-flow coefficient reaches |t| = 1.6 in any of the six cells (the largest is submission CLOP at −1.57); the placebo and κ-grid versions are also flat.
- Decile portfolios lose after costs in every cell. Gross decile spreads: −4 to +4.5 bps per day.
- Controls: the stock's own open-to-15:30 return predicts the last half-hour with t = −8.9 / −8.2 (intraday reversal). It is a control, not a finding of this study.

**Phase II fit** (DEV, in-sample). The largest ridge loadings on d_close are negative:
- own open-to-decision return (−1.7 bps per s.d., trade bursts);
- D-profile slope (−0.8);
- D_b itself (−0.5).

This matches the Q1 pattern. The models are frozen by hash in `results/p4_revisit_v1/freeze_before_VAL.json`.

---

## VAL 2020–2021 (single read, 2026-09-15)

**Inputs** (sha256): `nameday_VAL` `3f32af47f02f`, `sample_VAL` `b3dce8245883`, `strata_VAL` `a2afd6600b60`.
192,179 name-days, 417 names; groups 1–2. The aggregator hash `106d9198bd1b` and the Phase II models match
`freeze_before_VAL.json`.

**Outputs:** `analysis/VAL_primary.json`, `VAL_q2ext.json`, `VAL_q3.json`, `VAL_q5.json`, `VAL_phase1.json`. Every
read is logged in `access_log.txt`.

### Q1 (H1): does informative impact persist after the decision time?

Displacement from the decision mid to the close, bps; daily mean of name-day means, NW(10):

| family | informative | pseudo | non-informative | **info − pseudo** (G1) | info − non |
|---|---|---|---|---|---|
| trade | −1.66 | −0.71 | +1.71 | **−0.74 (t −2.19)**, name-bootstrap CI [−0.97, −0.49] | −3.36 (t −5.24) |
| submission | −1.08 | −0.84 | +1.09 | **−0.20 (t −0.94)**, CI [−0.32, −0.08] | −2.17 (t −3.40) |

- **κ grid.** Trade bursts: −0.51 (t −1.67) at 0.25 and −1.00 (t −2.69) at 0.75. Submission bursts: −0.16 and −0.31.
- **Other horizons, informative − non-informative.** Trade: −1.83 (t −1.54) to the next open and −4.04 (t −2.73) to the next close. Submission: −0.23 (t −0.21) and −2.41 (t −1.94).
- **G1 fails for both families, with the wrong sign.**

**Phase I** (per-burst sample). Displacement to the close by decile of D_b/PeakImpact, lowest to highest:
- trade bursts: +2.58, +0.49, +0.78, +0.97, +0.30, −0.27, −0.65, −1.23, −2.01, −3.53;
- submission bursts: +1.15 … −1.49.

It is monotone and inverted. The more a burst's impact held over its first ten minutes, the more it gives back
by the close. Burst size shows no pattern.

### Q2 (H2): are persistent-impact bursts parent-order children?

| item | trade | submission | pass rule |
|---|---|---|---|
| (a) directional linkage, informative − other | **+0.0035 [0.0021, 0.0051]**, 414 names | −0.0014 [−0.0024, −0.0004] | > 0, CI excludes 0 |
| (b) 13F ΔIO on quarter sums, informative − other large | **+0.067 (t 4.20)**; informative +0.034 (t 4.79), other −0.033 (t −3.41) | **+0.027 (t 4.77)** | > 0, t > 2 |
| (c) CRSP mutual-fund Δholdings | **+0.040 (t 4.10)** | **+0.020 (t 6.43)** | > 0, t > 2 |
| (c) flow-induced trading (Lou 2012) | +0.003 (t 1.40) | **+0.004 (t 5.48)** | > 0, t > 2 |
| (d) index events, 20 additions / 12 deletions, z(info) − z(other) | −0.16 (t −0.48) | −0.43 (t −2.23) | > 0, t > 2 |
| (e) retail: corr(info, BJZZ) − corr(other, BJZZ) | −0.013 (t −4.87) | −0.011 (t −3.82) | fails only if > 0, t > 2 |

Samples for (b)–(c): 3,050–3,062 name-quarters, 414–415 names, 8 quarters, quarter fixed effects, PERMNO-clustered.

**Verdict rule:** (a) and at least two of (b)–(d), and (e) not failing.
- **Submission bursts:** not met, because (a) fails.
- **Trade bursts:** depends on a reading the design left open, whether (c) passes on holdings changes alone or needs flow-induced trading as well.
  - Holdings alone: met — (a), (b), (c) pass and (e) does not fail.
  - Both required: not met — only (b) of (b)–(d).

  Both readings are carried to TEST unchanged. No choice was made after seeing these numbers.

What the pattern says:
- Quarter by quarter, informative burst flow co-moves with institutional ownership and mutual-fund holdings changes in the same direction. Other large burst flow co-moves in the opposite direction.
- Informative flow is less aligned with retail flow than other large flow.
- These are contemporaneous associations with slow-moving holdings, not predictions.

### Q3 (Phase II): is post-decision persistence predictable from information at T_dec?

Daily Spearman IC on the VAL sample (DEV-frozen models):

| family | ridge d_close | boosting d_close | ridge d_open | ridge d_cc | gate (t > 3 after ×2) |
|---|---|---|---|---|---|
| trade | **+0.040 (t 5.97)** | +0.035 (t 5.30) | +0.005 (t 0.52) | +0.024 (t 2.70) | pass |
| submission | **+0.022 (t 3.44)** | +0.013 (t 2.41) | −0.006 (t −0.72) | +0.014 (t 1.73) | pass |

- **What is predicted.** The DEV loadings are negative on the own open-to-decision move, the D slope and D_b. So what is predictable is an intraday give-back to the close.
- **What is not.** There is no predictability past the next open.

### Q4 (H4): daily informative flow and returns

FM coefficient on S_info/ADV20 with S_large, S_all and controls:

| cell | t | per s.d. (bps) | Holm (6 cells) | t > 3 | gate |
|---|---|---|---|---|---|
| trade CLOP | −1.50 | −1.53 | no | no | fail |
| trade CLCL | −0.90 | −1.62 | no | no | fail |
| trade tCLOSE | +0.63 | +0.23 | no | no | fail |
| **submission CLOP** | **−3.16** | **−5.48** | **yes (p 0.0016)** | **yes** | **pass** |
| submission CLCL | −2.41 | −7.51 | no (p 0.0159) | no | fail |
| submission tCLOSE | +0.29 | +0.12 | no | no | fail |

- **One cell passes VAL: submission CLOP, and it is reversal-signed** (−5.48 bps per s.d. of informative
  submission flow). A day's informative submission flow is followed by the *opposite* move into the next open.
  It is the confirmatory cell carried to TEST.
- **It is not an artefact of the filter.** The placebo signal is flat (t −1.15) and the κ-grid versions match
  the primary (−3.61 at κ 0.25, −3.19 at κ 0.75).
- **It is not tradable in VAL either.** The submission CLOP decile book is −0.18 bps a day *gross* before any
  cost. The coefficient is a regression slope with controls; the sortable version of it earns nothing.
- **Decile portfolios.** The trade-burst CLOP book earns +7.83 bps a day gross (NW t 3.88, annualized Sharpe
  2.46), but nets SR −0.05 at 2 bps per side and +1.21 at 1 bp. Deflated Sharpe probability given 311 trials:
  0.74 gross, 0.0015 net. Every other cell is negative net.

### Q5: where effects live

The Q1 difference is negative or zero in every split: listing exchange, year, coverage, size, tick constraint. The
most negative is trade bursts in NASDAQ-listed names, −1.65 (t −4.01). No split shows a positive or confirming Q4
coefficient.

### Family comparison (pre-registered scoreboard, VAL)

| item | trade | submission | higher |
|---|---|---|---|
| Q1 info − pseudo t | −2.19 | −0.94 | submission |
| Q2(a) linkage | +0.0035 (CI > 0) | −0.0014 (CI < 0) | trade |
| best Q4 primary t | +0.63 | +0.29 | trade |

**Trade bursts are "more promising" (2 of 3).** Both families go to TEST.

*Reading of row 3.* "Best Q4 primary t" is read as the highest signed t, which is how the row was filled
before TEST was read and is kept here. Under the other available reading — largest |t|, i.e. the strongest
cell of any sign — submission bursts win the row (−3.16 against −1.50) and the scoreboard would call
submission "more promising" 2 of 3. The design did not disambiguate this, so it is recorded rather than
resolved. TEST settles it independently of the reading: the trade family replicates Q2(a) and the submission
family's linkage reverses sign, and the submission cell that won the row under the second reading is the cell
that fails to replicate.

### What TEST decides (fixed now, before TEST is read)

- **Q2 verdict (trade bursts).** Replication in TEST of (a), and of (b) with (c) under each reading, with (e) not failing. Submission bursts are reported only.
- **Q3.** The same DEV-frozen ridge models for both families: IC > 0 with t > 2.
- **Q1 and Q4.** No confirmatory cell; TEST values are reported descriptively.
- **TEST period.** TEST is 2022–2025, but 13F runs only to 2025Q3 and CRSP returns end 2025-12-31.

**Amendment A5 changed the Q4 line above, after TEST had been read once.** The list was fixed against the
pre-A5 VAL, in which no Q4 cell passed. On the corrected VAL, submission CLOP passes the gate, so under the
design's own rule ("TEST: only VAL-passing cells, same sign with t > 2") Q4 does have a confirmatory cell.
The change adds a hurdle rather than removing one, and TEST fails it; no Q4 claim can be manufactured either
way. The Q2, Q3 and family-comparison rules are untouched — they key on `d_close`, which the fix does not move.

---

## TEST 2022–2025 (single read, 2026-09-15, corrected pipeline)

**Inputs** (sha256): `nameday_TEST` `563b2eb7215d`, `strata_TEST` `edb29be587cc`, `sample_TEST` `dd6bb7d54c48`.
511,727 name-days, 676 names; groups 1–2, the same two groups as VAL over a disjoint period. Code and models
match `freeze_before_VAL.json`. Outputs: `analysis/TEST_primary.json`, `TEST_q2ext.json`, `TEST_q3.json`,
`TEST_q5.json`, `TEST_phase1.json`. The pre-A5 read is kept as `TEST_*_preA5.json`.

### Q1 (H1): persistence after the decision time — descriptive, no confirmatory cell

| family | informative | pseudo | non-informative | info − pseudo | info − non |
|---|---|---|---|---|---|
| trade | −1.22 | −0.80 | +1.12 | −0.19 (t −1.52), CI [−0.34, −0.02] | −2.34 (t −8.41) |
| submission | −0.66 | −0.54 | +0.78 | −0.08 (t −0.69), CI [−0.18, +0.03] | −1.43 (t −5.00) |

- **κ grid.** Trade bursts −0.14 (t −1.21) at 0.25 and −0.32 (t −2.27) at 0.75; submission −0.06 and −0.17.
- **Other horizons, informative − non-informative.** Trade −2.17 (t −4.40) to the next open and −1.95 (t −2.06)
  close-to-close; submission −1.36 (t −2.36) and −1.53 (t −1.66).
- **Phase I deciles of D_b/PeakImpact**, lowest to highest: trade +1.15, +0.37, +0.17, −0.18, −0.17, −0.75,
  −0.31, −0.87, −0.87, −0.60; submission +0.15 … −0.62.

The VAL pattern repeats with the same signs and about half the magnitude: bursts whose impact held over the
first ten minutes give more of it back by the close than bursts that did not, and they do no better than
random-time windows that pass the same filter. In TEST the informative-minus-pseudo gap is not distinguishable
from zero (t −1.52), so the honest statement is **no persistence, not negative persistence**: the filter selects
bursts that look like the placebo, while the *unfiltered* contrast against non-informative bursts stays sharply
negative because non-informative bursts drift the other way.

### Q2 (H2): are persistent-impact bursts parent-order children?

**(a) Directional linkage** (informative − other, same-side minus opposite-side link rate):

| family | VAL | TEST | replicates |
|---|---|---|---|
| trade | +0.0035 [0.0021, 0.0051] | **+0.0046 [0.0034, 0.0060]**, 675 names | **yes** |
| submission | −0.0014 [−0.0024, −0.0004] | +0.0012 [0.0008, 0.0017] | no — the sign flips between cells |

Informative trade bursts are more often followed by a same-side burst in the same name than other large bursts
are, in both cells, with CIs excluding zero. For submission bursts the effect is significantly negative in VAL
and significantly positive in TEST; two confident opposite answers are not evidence, and the submission family
fails (a) under the pre-registered rule.

**(b)–(e) External institutional labels** (`TEST_q2ext.json`; 7,533–8,666 name-quarters, 674–676 names,
quarter fixed effects, PERMNO-clustered; 13F covers 2022Q1–2025Q2 after amendment A6 dropped 2025Q3 at a
median of 63 filers):

| item | trade (VAL → TEST) | submission (VAL → TEST) | pass rule | replicates |
|---|---|---|---|---|
| (b) 13F ΔIO, informative − other large | +0.067 (t 4.20) → **+0.045 (t 3.79)** | +0.027 (t 4.77) → **+0.014 (t 3.31)** | > 0, t > 2 | **both** |
| (c) CRSP mutual-fund Δholdings | +0.040 (t 4.10) → **+0.054 (t 7.16)** | +0.020 (t 6.43) → **+0.012 (t 4.75)** | > 0, t > 2 | **both** |
| (c) flow-induced trading (Lou 2012) | +0.003 (t 1.40) → +0.002 (t 1.69) | +0.004 (t 5.48) → **+0.002 (t 3.49)** | > 0, t > 2 | submission only |
| (d) index events (49 adds / 38 deletes) | −0.16 (t −0.48) → −0.04 (t −0.18) | −0.43 (t −2.23) → −0.03 (t −0.19) | > 0, t > 2 | neither |
| (e) retail alignment, corr(info) − corr(other) | −0.013 (t −4.87) → −0.002 (t −1.04) | −0.011 (t −3.82) → −0.002 (t −0.92) | fails only if > 0 | does not fail |

In TEST as in VAL, the decomposition is two-sided: informative flow loads **positively** on institutional
ownership change (13F +0.027, t 5.06; mutual funds +0.033, t 9.57 for trade bursts) while *other* large burst
flow loads **negatively** (−0.018, t −2.49; −0.021, t −4.51). Two independently constructed institutional
measures agree in both cells and for both families — but both also co-move with the stock's same-quarter
return, which the pre-registered regression does not control. Read **Post hoc: is the institutional
association a return channel?** below before treating this as institutional trading.

**Q2 verdict.**

| reading of (c) | trade bursts | submission bursts |
|---|---|---|
| holdings changes alone | (a), (b), (c) pass and (e) does not fail in **both VAL and TEST** — **met** | not met: (a) reverses sign between cells |
| flow-induced trading also required | not met in either cell — only (b) of (b)–(d) | not met: (a) |

The design left this reading open and both branches were carried to TEST unchanged, so the outcome is a
replication under one reading and a consistent failure under the other, not a choice made after the fact.
The honest statement is: **flow from persistent-impact trade bursts is associated, quarter by quarter, with
institutional ownership and mutual-fund holdings changes in the same direction, out of sample, and other large
burst flow is associated with the opposite direction; the Lou (2012) flow-induced-trading instrument does not
corroborate this for trade bursts, and index events are uninformative at 87 events.**

### Q3 (Phase II): is post-decision persistence predictable from information at T_dec?

Daily Spearman IC, DEV-frozen models, TEST rule "same model, t > 2":

| family | ridge d_close | boosting d_close | ridge d_open | ridge d_cc | replicates |
|---|---|---|---|---|---|
| trade | **+0.039 (t 7.73)** | +0.030 (t 6.51) | +0.037 (t 5.67) | +0.021 (t 3.38) | **yes** |
| submission | **+0.023 (t 4.86)** | +0.013 (t 3.33) | +0.025 (t 4.28) | +0.013 (t 2.28) | **yes** |

- **Q3 replicates for both families.** A model fit on 2017–2019 still ranks 2022–2025 bursts by how much of
  their impact they will give back by the close.
- **One difference from VAL.** In VAL the prediction stopped at the close (d_open IC +0.005, t 0.52); in TEST
  it carries overnight (+0.037, t 5.67). This is a between-cell difference, not an effect of the A5 fix —
  the corrected VAL d_open IC is unchanged at +0.005. It is descriptive: nothing in the design nominated the
  overnight horizon, and the magnitude is a rank correlation of 0.04.

### Q4 (H4): daily informative flow and returns

| cell | t | per s.d. (bps) | Holm (6 cells) | verdict |
|---|---|---|---|---|
| trade CLOP | −0.74 | −0.64 | no (p 0.4590) | — |
| trade CLCL | −0.34 | −0.73 | no (p 0.7358) | — |
| trade tCLOSE | +0.85 | +0.16 | no (p 0.3962) | — |
| **submission CLOP** (confirmatory) | **−0.37** | −0.37 | no (p 0.7138) | **fails to replicate** |
| submission CLCL | +0.96 | +3.26 | no (p 0.3380) | — |
| submission tCLOSE | +0.15 | +0.04 | no (p 0.8803) | — |

- **The one VAL-passing cell does not replicate.** Submission CLOP is t −3.16 in VAL and t −0.37 in TEST,
  against a pre-registered hurdle of t > 2 with the same sign. The sign agrees; the magnitude is a tenth.
- **No other cell is close.** The largest |t| among the six is 0.96.
- **Decile portfolios.** Every cell loses after costs, and four of six lose before costs. The best gross book
  is submission CLCL at +3.58 bps/day (t 1.49), which nets −4.42 at 2 bps per side. The trade CLOP book that
  earned +7.83 gross in VAL earns +0.93 (t 0.67) in TEST and nets −7.07.

**Q4 conclusion: there is no tradable daily signal in either burst family.** This is the same verdict the
pre-A5 draft reached, now reached against a cell that VAL had actually nominated.

### Q5: where effects live (TEST)

The Q1 informative-minus-pseudo difference by split, trade bursts: NASDAQ-listed −0.64 (t −3.6) against
NYSE +0.20 (t +1.2); high-coverage tercile −0.60 (t −3.3) against low +0.09 (t +0.5); tick-constrained
−0.43 (t −2.3) against unconstrained −0.09 (t −0.7); by year −0.56, −0.18, −0.37, +0.36 for 2022–2025.
Submission bursts are flat everywhere (|t| ≤ 1.3 except AMEX, 14 names).

The concentration matches VAL: whatever the filter selects, it shows up in NASDAQ-listed, well-covered,
tick-constrained names — the names where LOBSTER sees the largest share of consolidated volume. That is the
signature of a measurement effect, not of an economic one.

### TEST verdict and family comparison

| pre-registered question | trade bursts | submission bursts |
|---|---|---|
| Q1 persistence past T_dec (G1) | fails — no persistence over placebo (−0.19, t −1.52) | fails (−0.08, t −0.69) |
| Q2(a) within-tape linkage | **replicates** (+0.0046, CI > 0) | fails — sign reverses between cells |
| Q2(b)–(e) institutional labels | **replicates** on 13F and mutual-fund holdings; FIT does not | **replicates** on 13F, holdings *and* FIT |
| Q2 verdict | **met** under the holdings reading, in both cells | not met — (a) |
| Q3 predictability of give-back | **replicates** (IC +0.039, t 7.73) | replicates (+0.023, t 4.86) |
| Q4 daily return signal | nothing passes | the one VAL-passing cell fails to replicate |

**Trade bursts remain the more promising family**, now on out-of-sample evidence rather than on the VAL
scoreboard: they are the only family whose within-tape linkage holds its sign across both cells, and they
carry the institutional association as well. A burst family whose own definition of "informative" predicts
same-side continuation in one period and opposite-side continuation in the next cannot support a claim. (An
earlier draft called the submission family's external labels "if anything cleaner" because they also pass
flow-induced trading. The post-hoc return control below shows those passes were largely a same-quarter
return channel; that remark is withdrawn.)

**What this study establishes, and what it does not.**

1. **Establishes (replicated; narrowed by a post-hoc control):** persistent-impact trade bursts carry flow that
   is aligned with institutional ownership and mutual-fund holdings changes, while other large burst flow is
   aligned the opposite way, in 2020–21 and again in 2022–25. In 2022–25 both associations survive a control
   for the stock's same-quarter return (t 2.80 and 5.21); in 2020–21 only the 13F one does. The retail
   comparison is not evidence — see the post-hoc section.
2. **Establishes (replicated):** how much of a burst's impact will be given back by the close is predictable at
   the decision time, in both families and both cells, from features frozen on 2017–19.
3. **Refutes:** the original P4 premise. Informative impact does not persist past the decision point — after the
   firewall, "informative" bursts do no better than same-day random windows that pass the same filter, and
   worse than the bursts the filter rejected.
4. **Refutes:** any tradable version of it. No Q4 cell survives, the one VAL-nominated cell fails out of sample,
   and every decile book loses after realistic costs.

---

## ERA2 2012–2016 (second-era replication, single read, 2026-09-15)

**Inputs** (sha256): `nameday_ERA2` `5275893a4f48`, `strata_ERA2` `fdd6d21e5214`, `sample_ERA2` `ea90fad93b8b`.
414,802 name-days, 461 names, all three name groups. Read after the TEST section above was written; the
protocol lock in `p4_analyze.check_protocol` required `TEST_primary.json` to exist first.

**Caveat on names.** ERA2 uses all groups, so it overlaps the DEV names the Phase II models were fit on. Its
Q3 numbers are out of sample in time (2012–16 against a 2017–19 fit) but not in names; VAL and TEST are the
name-disjoint evidence for Q3.

### Q1: persistence, 2012–2016

| family | informative | pseudo | non-informative | info − pseudo | info − non |
|---|---|---|---|---|---|
| trade | −1.49 | −0.65 | +0.58 | **−0.60 (t −7.80)**, CI [−0.74, −0.47] | −2.04 (t −14.12) |
| submission | −0.57 | −0.38 | +0.75 | **−0.18 (t −3.18)**, CI [−0.24, −0.12] | −1.31 (t −8.46) |

κ grid, trade bursts: −0.55 (t −7.34) at 0.25 and −0.76 (t −9.26) at 0.75. To the next open, informative minus
non-informative is −2.52 (t −11.05); close-to-close −2.29 (t −5.73). Phase I deciles run +1.46 to −1.97,
monotone.

**The earlier era refutes H1 more sharply than the later ones.** In 2012–16 informative bursts do measurably
*worse* than same-day random windows passing the same filter (t −7.80), where in TEST the two were
indistinguishable. Across the three periods the ordering is monotone in time: −0.60 (2012–16), −0.74 (2020–21),
−0.19 (2022–25) — the give-back is largest early and fades, consistent with the 2016→2024 halving of the size
fingerprint reported in `studies/fingerprint/BURST_FINGERPRINT_RESULTS.md`, and with impact being impounded faster over time.

### Q2: linkage and institutional labels, 2012–2016

| item | trade | submission |
|---|---|---|
| (a) directional linkage | **+0.0068 [0.0026, 0.0108]**, 407 names | −0.0002 [−0.0023, +0.0019], 451 names |
| (b) 13F ΔIO, informative − other | **+0.069 (t 3.40)**; info +0.042 (t 4.82), other −0.027 (t −2.10) | **+0.019 (t 3.28)** |
| (c) mutual-fund Δholdings | **+0.038 (t 7.09)** | **+0.011 (t 5.30)** |
| (c) flow-induced trading | +0.002 (t 1.82) | **+0.002 (t 5.28)** |
| (d) index events (40 adds / 16 deletes) | +0.12 (t 0.50) | −0.24 (t −1.12) |
| (e) retail alignment | −0.014 (t −6.64) — does not fail | −0.012 (t −5.66) — does not fail |

13F covers 2013Q2–2016Q4 (14 quarters); amendment A6 dropped 2012Q4 and 2013Q1 at 8 and 17 median filers.

**Every part of the Q2 pattern reproduces a decade earlier**, including the exact asymmetry seen in the later
cells: flow-induced trading corroborates for submission bursts and not for trade bursts, and index events are
uninformative at 56 events. The trade-burst directional linkage here comes from *fewer opposite-side* links
(same +0.0001, opposite −0.0067) rather than more same-side ones — the mechanism differs from the later eras,
where same-side links rose.

### Q3 and Q4, 2012–2016

- **Q3.** Daily IC from the DEV-frozen models: trade +0.052 (t 11.46), submission +0.029 (t 7.15); overnight
  d_open +0.056 (t 9.30) and +0.036 (t 6.64). Strongest of the three cells — again consistent with impact
  being given back more slowly in the earlier era. Name overlap with DEV applies.
- **Q4.** Nothing passes. The largest cell is trade tCLOSE at t +2.65 (Holm p 0.0082 but below the t > 3
  hurdle), and its κ 0.25 version is +4.62 while its κ 0.75 version is +1.14 — unstable in the parameter the
  design varied precisely to detect this. Every decile book loses after costs; four of six lose gross.

---

## Post hoc: is the institutional association a return channel? (2026-09-20, not pre-registered)

**Why.** A burst is classified informative because the price moved its way over ten minutes, so informative
flow summed over a quarter tracks the stock's own return in that quarter. Institutional ownership changes
co-move with contemporaneous returns (Sias, Starks and Titman 2006). The pre-registered Q2(b)/(c) regressions
control only for the *previous* quarter's return, and "informative flow loads positive, other large flow loads
negative" is exactly what a pure return channel would produce. So the pre-registered pass cannot, by itself,
distinguish institutional trading from flow that is merely aligned with price.

**Check.** `src_py/p4_q2_return_control.py` re-runs (b) and both (c) tests with `ret_q`, the stock's
compounded return over the same calendar quarter, added to the pre-registered controls. Inputs, code path,
winsorization and clustering are otherwise unchanged. It can remove a result; it cannot create one. Outputs:
`analysis/{VAL,TEST,ERA2}_q2ext_retctrl.json`.

**The channel exists.** Spearman correlation of quarterly flow with the same-quarter return: informative
+0.15 to +0.21 across cells and families; other large −0.05 to −0.12.

**Informative − other, pre-registered → with `ret_q`** (t in parentheses; bold = still > 0 with t > 2):

| test | family | 2012–16 | 2020–21 | 2022–25 |
|---|---|---|---|---|
| (b) 13F ΔIO | trade | +0.069 (3.40) → **+0.071 (3.30)** | +0.067 (4.20) → **+0.039 (2.45)** | +0.045 (3.79) → **+0.034 (2.80)** |
| (c) MF Δholdings | trade | +0.038 (7.09) → **+0.025 (4.44)** | +0.040 (4.10) → +0.012 (1.19) | +0.054 (7.16) → **+0.041 (5.21)** |
| (c) FIT | trade | +0.002 (1.82) → −0.001 (−1.13) | +0.003 (1.40) → −0.005 (−2.45) | +0.002 (1.69) → −0.001 (−0.65) |
| (b) 13F ΔIO | submission | +0.019 (3.28) → **+0.019 (3.04)** | +0.027 (4.77) → **+0.014 (2.30)** | +0.014 (3.31) → +0.007 (1.57) |
| (c) MF Δholdings | submission | +0.011 (5.30) → **+0.005 (2.60)** | +0.020 (6.43) → **+0.008 (2.37)** | +0.012 (4.75) → +0.004 (1.52) |
| (c) FIT | submission | +0.002 (5.28) → **+0.001 (2.14)** | +0.004 (5.48) → +0.000 (0.58) | +0.002 (3.49) → +0.000 (0.32) |

**Reading.**
- Part of the pre-registered association was the return channel. How much varies by cell: none for trade
  bursts and 13F in 2012–16, about 70% for trade bursts and mutual funds in 2020–21.
- **What survives is trade-burst flow and 13F ownership changes, in all three periods** (t 2.45–3.30), and
  trade-burst flow and mutual-fund holdings in two of three, strongest in the confirmatory cell (t 5.21).
- **Submission bursts' institutional evidence does not survive in the confirmatory cell.** In 2022–25 all
  three of its tests fall below t 2 once the same-quarter return is controlled.
- **Flow-induced trading supports neither family** once the return is controlled (1 of 6 cells above t 2).
  FIT is itself a source of contemporaneous price pressure (Lou 2012), so its pre-registered passes were
  exposed to the same channel.
- **The retail comparison (e) was not re-run.** Retail flow is contrarian to returns, so "informative flow is
  less retail-aligned" is equally consistent with the return channel and must not be cited as evidence of
  institutional participation.

This check was run on cells already read. It does not change the pre-registered verdicts above, which are
reported as they fell; it narrows what those verdicts can be taken to mean.

---

## Conclusions across the three periods

| pre-registered question | 2012–16 | 2020–21 | 2022–25 | verdict |
|---|---|---|---|---|
| H1 impact persists past T_dec (trade) | −0.60 (t −7.80) | −0.74 (t −2.19) | −0.19 (t −1.52) | **refuted**: negative or zero in all three |
| Q2(a) directional linkage (trade) | +0.0068 | +0.0035 | +0.0046 | **holds**, CI > 0 in all three |
| Q2(a) directional linkage (submission) | −0.0002 | −0.0014 | +0.0012 | inconsistent |
| Q2(b) 13F ΔIO (trade), pre-registered | +0.069 (t 3.40) | +0.067 (t 4.20) | +0.045 (t 3.79) | passes |
| … with same-quarter return control (post hoc) | +0.071 (t 3.30) | +0.039 (t 2.45) | +0.034 (t 2.80) | **holds** |
| Q2(c) mutual-fund holdings (trade), pre-registered | +0.038 (t 7.09) | +0.040 (t 4.10) | +0.054 (t 7.16) | passes |
| … with same-quarter return control (post hoc) | +0.025 (t 4.44) | +0.012 (t 1.19) | +0.041 (t 5.21) | 2 of 3 |
| Q2(e) retail alignment (trade) | −0.014 | −0.013 | −0.002 | not evidence: same return channel, untested |
| Q3 give-back predictable (trade) | +0.052 (t 11.5) | +0.040 (t 5.97) | +0.039 (t 7.73) | **holds** |
| Q4 daily return signal | none | one cell, reversal-signed | fails to replicate | **refuted** |

**1. The original P4 hypothesis is refuted, cleanly.** Once the decision time is moved to
T_dec = max(t_b + 600 s, t_e + 10 s) and the placebo is built correctly, bursts selected for holding their
impact over ten minutes do not hold it afterwards. They give it back, relative both to the bursts the filter
rejected and — in two of three periods — to random same-day windows that pass the same filter. Every earlier
version of this result that looked positive is accounted for in `studies/p4_revisit/PRIOR_WORK_INVENTORY.md`: the legacy hidden-
print signing, the circular κ gate, in-sample tuning on NVDA/TSLA, and in this revisit three pipeline defects
(A3, A4, A5) that were found and fixed before the numbers above were read.

**2. An institutional association survives, narrower than the pre-registered test suggested.** The
pre-registered Q2(b)/(c) tests pass in all three periods, but the post-hoc return control shows that part of
the association was the stock's own same-quarter return. What survives: flow from persistent-impact **trade**
bursts co-moves with 13F institutional ownership changes beyond what that return explains, in all three
periods (t 2.45–3.30), and with mutual-fund holdings changes in two of three (t 4.44 and 5.21; not in
2020–21). Other large burst flow loads the opposite way. It is a contemporaneous, aggregate, quarterly
association with slow-moving holdings — not a prediction, not a name-day classifier, and not corroborated by
flow-induced trading. On "can we separate retail from institutions": **partly, on external labels, in
aggregate and by quarter; the retail side itself is not yet shown**, because the retail comparison is exposed
to the same return channel and was not re-tested.

**3. Trade bursts are the more promising family, on two independent grounds.** Their within-tape directional
linkage keeps its sign across all three periods where the submission family's reverses between VAL and TEST;
and their institutional association survives the return control in the confirmatory period where the
submission family's does not.

**4. There is no tradable signal, and the study can say so with authority.** Six pre-registered cells × three
periods, a κ grid, a placebo signal, S_pred, and decile portfolios at two cost levels: one cell passed a
validation gate and failed replication; nothing else came close; every book loses after realistic costs. The
one gross-positive book (trade CLOP in VAL, +7.83 bps/day, SR 2.46) has a deflated-Sharpe probability of
0.0015 net of costs against 311 recorded trials.
