# Metaorder-v1 results

Completed 2026-09-14. Design and amendments: `studies/metaorder/METAORDER_DESIGN.md`. Freezes:
`results/metaorder_v1/freeze_v1.json`–`freeze_v4.json`.

## Bottom line

| step | question | answer |
|---|---|---|
| M1 | one child-cluster rule? | **Tie.** Side-only 5 s streams and 60 s runs are indistinguishable on size evidence in every year; timing evidence swings by year. run60 is kept. |
| M2 | can we score membership in a longer program? | **Yes, but not a directional one.** Same-side links rank out of sample (L1, L3 pass). In 2021, opposite-side links rank almost as strongly (L2 fails). |
| M4 | is program-likeness visible in real time? | **Yes.** The first three children retain ~90% of the whole-burst score's out-of-sample lift (R1, R2 pass). |
| M3 | how well do we recover known parents? | Fast parents (≤ 15 s between children): about half to three-quarters of children, 50–70% burst purity. Slower parents are mostly invisible to run rules. Size linkage finds about half of fixed-clip bursts and a third of randomized ones. Score AUC 0.97 is an upper bound. |
| M5 | does posting against program-like bursts pay? | **No.** Every posting type loses per fill. Program-triggered postings lose most per posting in both years (Q1 fails: +0.33 in 2024, −0.34 in 2021). |

The detection chain is well defined, validated and calibrated:
- child clusters settled up to the run/stream equivalence;
- a real-time program score;
- a linkage score that finds persistent algorithms.

Program-evidence-v1 showed that what it finds is mostly intermediary and arbitrage flow, and M5
shows that trading passively against it is a losing strategy. The fingerprint does not give a
metaorder label or a trading edge.

## M1 — the child-cluster rule

Combined J = ½ (size J + phase J) on names eligible under both rules. D is stream5 minus run60,
with a name-paired bootstrap.

| year | role | names | run60 | stream5 | D [95% CI] | size-only D | phase-only D | decision |
|---|---|---|---|---|---|---|---|---|
| 2024 | seen | 160 | 0.258 | 0.258 | +0.001 [−0.025, +0.025] | −0.019 | +0.020 | tie |
| 2021 | seen | 249 | 0.267 | 0.306 | +0.039 [+0.008, +0.073] | +0.002 | +0.076 | stream5 |
| **2019** | **gate, read once** | 360 | 0.328 | 0.356 | **+0.028 [−0.007, +0.061]** | −0.001 | +0.058 | **tie: keep run60** |
| 2016 | secondary | 127 | 0.286 | 0.325 | +0.039 [+0.010, +0.072] | −0.008 | +0.086 | stream5 |
| 2013 | secondary | 96 | 0.262 | 0.213 | −0.050 [−0.080, −0.019] | −0.016 | −0.083 | run60 |

**Result.** run60 stays the working child-cluster rule. On size evidence the two rules are
indistinguishable in every year (size-only D within ±0.02). They differ only on timing evidence:
side-only 5-second streams keep more whole-second-locked pairs in 2016, 2019 and 2021, and fewer in
2013. The child-cluster boundary is therefore settled only up to this equivalence class, and it is
not worth further search.

## M2 — parent linkage score (fit 2024, scored once on 2021)

Target: identical-size links from a burst's untruncated non-round children to other bursts in the
following 30 minutes, same side, against depth-quartile cross-day chance. Features: size-free burst
features and backward context.

| | 2024 in sample | 2021 out of sample |
|---|---|---|
| same-side excess links per 1,000 pairs, deciles 1→10 | −2.1 … 22.7 | −0.9 … 61.8 |
| **L1** top − bottom, same side | +24.8 [14.0, 54.6] | **+62.7 [31.7, 96.1] — PASS** |
| **L3** Spearman(decile, excess) | 1.00 | **1.00 — PASS** |
| opposite-side excess links, deciles 1→10 | −2.6 … 3.3 | −1.7 … 55.5 |
| **L2** (same − opposite) top − bottom | +18.9 [9.9, 44.7] | **+5.5 [−4.0, +21.8] — FAIL** |

**Reading.** The linkage score finds bursts that belong to longer-lived same-algorithm programs,
out of sample in names and year. In 2024 those programs are directional: same-side links rise 7×
more than opposite-side links across deciles. In 2021 the opposite-side links rise nearly as much,
the two-sided fixed-clip population found by program-evidence-v1 (B4). Directionality therefore
fails confirmation. As built, the score measures membership in a persistent algorithm, not in a
directional metaorder.

Largest standardized coefficients (2024 fit):
- time of day, +2.53 and −2.23 (quadratic);
- truncation share, −1.30;
- log intensity, −0.81;
- opposite-side in-burst share of the previous 30 minutes, −0.55: activity on the other side lowers
  linkage;
- log duration, −0.42; hidden share, −0.38; log depth, −0.37; log packets, +0.38.

## M4 — real-time program score (first three own-side packets)

| | 2024 in sample | 2021 out of sample |
|---|---|---|
| excess repeats per 1,000 pairs, deciles 1→10 | 77.8 … 242.6 | 94.3 … 263.6 |
| **R1** top − bottom | +164.8 [104.5, 249.2] | **+169.3 [107.4, 237.2] — PASS** |
| **R2** Spearman | 0.92 | **0.94 — PASS** |
| whole-burst score, for comparison | +158.6 | +190.4 [151.5, 234.5] |

**Reading.** Program-likeness is visible by the third child. A score that could run in real time
retains about 90% of the whole-burst score's out-of-sample lift. Its dominant inputs are the pace of
the first three packets (duration against mean gap), truncation of the first three, spread, time of
day, and the share of opposite-side burst volume in the last five minutes.

## M3 — calibration on injected synthetic parents (2024 exploration, 50 names × 10 days)

Six parents were generated per name-day, 3,000 in total. 665 were skipped by the 1%
participation cap, leaving 2,335 injected parents with a median of 10 children (10th–90th
percentile 5–40). Only 14% ran their full schedule: the cap binds, so these parents are the
NASDAQ-sized slices a real parent leaves on one venue.

| | all | 5 s interval | 15 s | 30 s | 60 s |
|---|---|---|---|---|---|
| children inside run60 bursts | 51% | 76% | 54% | 42% | 36% |
| mean purity of bursts containing children | 45% | 70% | 49% | 35% | 26% |
| parents with ≥ 1 majority-injected burst | 50% | 90% | 65% | 36% | 10% |

- **Linkage.**
  - Injected-majority bursts show 4.7× chance same-side identical-size links over the next 30
    minutes, against 1.25× for the 455k real bursts in the same name-days.
  - Among parents with at least two majority bursts, 40% of those bursts link to another burst of
    the same parent: fixed clips 49%, randomized clips 31%.
- **Scores.**
  - Pooled AUC for injected-majority against all other bursts: program score 0.973, linkage score
    0.969 (name-day medians 0.979 and 0.968).
  - This is an **upper bound for real programs**. The injected children are untruncated, carry no
    hidden fills and are regularly paced, exactly the features the scores reward.

Expectations written in advance:
- *Fixed clips link more often than randomized clips*: **holds** (49% against 31%).
- *Timer-locked parents score higher on the program score*: **does not hold** (−1.01 against
  −1.03). The score has no phase feature, so a timer signature would need to be added explicitly.

**What this calibrates.**
- *Child clustering.* run60 recovers the children of fast parents (≤ 15 s between children) well
  and mixes them with unrelated flow at 30–50%. Slower parents (30–60 s) are mostly invisible to
  any run rule, because opposite-side trades break the run first.
- *Linkage.* Linking through identical sizes finds about half the later bursts of fixed-clip
  parents and a third for randomized clips.
- *What real programs rarely are.* Real program-like bursts score far below these ideal
  synthetic parents.

## M5 — queue-aware passive orders against program-like bursts

**2024 exploration:** 40 names × 10 days, 231,838 simulated 100-share postings at the touch. Each
joins the back of the displayed queue when a burst's third own-side packet arrives, cancels after
60 s or when a better price appears, and fills only through price-time priority.

| postings | n | fill rate | per-fill markout 1 s | 10 s | 60 s | 300 s | P&L per posting, 60 s | with 0.20¢ rebate |
|---|---|---|---|---|---|---|---|---|
| program-like bursts (score ≥ q80) | 24,610 | 51% | −1.43 | −1.84 | −1.70 | −1.56 | −0.87 | −0.54 |
| middle bursts | 111,571 | 43% | −1.22 | −1.62 | −1.75 | −1.72 | −0.75 | −0.58 |
| least program-like (≤ q20) | 68,609 | 35% | −1.01 | −1.37 | −1.61 | −1.94 | −0.56 | −0.46 |
| placebo times | 27,048 | 27% | −1.37 | −1.66 | −1.92 | −2.12 | −0.53 | −0.33 |

(bps, provider-signed; per-posting P&L counts unfilled postings as zero)

- **Q1 FAILS at exploration.** Per-fill 60 s markout, program minus bottom: +0.33 bps, NW t = 2.38,
  name bootstrap [−0.04, +0.91]. The gate required |t| > 3.
- **Q2 (descriptive).** Per-posting 60 s P&L, program minus placebo: −0.42 bps (t = −6.9).

**Reading.**
- *Every way of posting loses.* A new order at the back of the queue fills mainly when the level is
  being cleared, and the price then moves through it. About half of all fills are trade-throughs.
- *Program triggers lose more per posting, not less.* They raise the fill rate (51% against 27% at
  random times) because the program keeps consuming the level, but they do not reduce the loss per
  fill.
- *Why this differs from the markout result.* Program-evidence-v1 found the *average resting order*
  at the touch earning +0.6–0.7 bps against program bursts. That average is dominated by orders at
  the front of the queue. The marginal order joining at the back is adversely selected, and a maker
  rebate does not change the sign.

Selecting program-like flow is not a liquidity-provision edge for a new passive order.

**2021 confirmation** (read once): 60 names, 595 name-days, 479,543 postings.

| postings | n | fill rate | per-fill markout 1 s | 10 s | 60 s | 300 s | P&L per posting, 60 s | with rebate |
|---|---|---|---|---|---|---|---|---|
| program-like bursts | 63,226 | 50% | −1.84 | −2.12 | −2.06 | −2.15 | −1.03 | −0.78 |
| middle bursts | 240,723 | 39% | −1.57 | −1.82 | −1.77 | −1.79 | −0.70 | −0.54 |
| least program-like | 108,235 | 29% | −1.23 | −1.49 | −1.54 | −1.40 | −0.45 | −0.36 |
| placebo times | 67,359 | 26% | −1.67 | −1.76 | −1.73 | −1.79 | −0.45 | −0.30 |

- **Q1.** Per-fill 60 s markout, program minus bottom: **−0.34 bps (t = −5.6)**, the opposite sign to
  2024 (+0.33, t = 2.4). **Q1 FAILS** in both years.
- **Q2.** Per-posting 60 s P&L, program minus placebo: −0.57 bps (t = −21.4). 2024: −0.42
  (t = −6.9).

A new passive order posted when a program-like burst starts is filled more often and loses more.
The fill is the program's continuation through the level.