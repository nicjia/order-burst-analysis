# Verified Results

The only numbers in this project that come from a design with no known flaw. Anything not
listed here must not appear in `paper.tex` or `main.tex`. Section 2 lists what was excluded
and why, so a discarded number is never silently reintroduced.

**2026-09-02 supersession resolved.** The replacement audit (`hidden-packet-bounds-v1`,
§1.22) has completed. Hidden-liquidity §§1.1–1.14 remain correct reproductions of their stated
*message-level* conventions but must not be cited as magnitudes for economic orders. On
economic packets the bifurcation replicates **in sign** and is much smaller **in magnitude**:
+0.603 for the aggressive leg and −0.056 at the midpoint on the 2023–2024 panel, against
+2.09 and −0.42 in §1.1. Cite §1.22 for magnitudes and §§1.1–1.14 only as evidence about
conventions.

*Correction, 2026-09-13.* This note previously said message counting inflated the headline
roughly 3.5×. That attribution is not supported: §1.1 and §1.22 also differ in burst formation
(same-side runs vs none) and in the measurement base (burst termination vs a one-second
buffer). §1.2 shows the buffer alone moves the per-print contemporaneous-mid estimate from
+1.375 to +0.530. The closest like-for-like comparison available — per print, from t+1s —
is +0.530 on messages against +0.603 on packets (signing conventions still differ slightly),
so there is no evidence that message counting itself inflated the level.

Panel throughout: LOBSTER NASDAQ ITCH, 2023–2024, 474 names (470–473 with usable data),
~236,000 name-days. Inference: the day is the unit; equal-weight within name-day, average
cross-sectionally, Newey–West (10 lags) on the daily-mean series; drop name-days with
|markout| > 1000 bps.

---

## 1. Verified

### 1.1 The bifurcation — array 14132159 (`hidden_emo474`)
Convention: quote rule, abstaining on at-midpoint prints; bursts = runs of ≥3 same-side
prints within 1s gaps; markout from the burst-termination midpoint.

| subset | 3 min | 15 min | 30 min |
|---|---|---|---|
| Aggressive (away from mid) | +2.09 (t=25.3) | +2.24 (t=12.2) | +2.25 (t=13.3) |
| At the midpoint (tick-signed) | −0.42 (t=−6.6) | −0.64 (t=−7.0) | −0.71 (t=−10.7) |

Classifier spread on the same data: quote-abstain +2.09, tick +0.62, CLNV +0.46, EMO +0.03.
Disagreement is monotone in a name's at-midpoint share (Q1 15% share → all four agree;
Q4 91% → only the abstaining rule stays positive).

### 1.2 Construction sensitivity — arrays 14314701 (`hid_ff`), 14367935 (`hid_pp`)
Removing price-conditioning from the construction, same outcome measured throughout:

| formation / signing | 3 min | 15 min | 30 min |
|---|---|---|---|
| same-side runs / contemporaneous mid | +1.482 (38.1) | +1.441 (20.5) | +1.416 (18.3) |
| time clusters / pre-print mid | +0.602 (14.9) | +0.589 (13.1) | +0.614 (13.7) |
| time clusters / outside pre-print quote | +0.029 (0.5) | −0.042 (−0.6) | −0.101 (−1.0) |

Per print, no clustering, same measurement base:

| signing | from t | from t+1s |
|---|---|---|
| contemporaneous midpoint | +1.375 (43.9) | +0.530 (20.8) |
| outside the pre-print quote | **+0.621 (12.4)** | +0.097 (2.1) |

Outside-quote at longer horizons: +0.543 (15 min), +0.495 (30 min). Only 41 of 1,604 hidden
prints per day (~2.5%) are unambiguously signable.

**+0.621 is the conservative headline.** +2.09 is what the common convention yields.

### 1.3 The negative leg is not a tick-rule artifact — array 14227671 (`hid_tp`)
| | 3 min | 15 min | 30 min |
|---|---|---|---|
| at-midpoint (tick-signed) | −0.411 (−9.0) | −0.587 (−10.4) | −0.674 (−11.1) |
| matched placebo (shifted times, same rule) | −0.022 (−1.2) | −0.088 (−3.3) | −0.110 (−2.9) |

Placebo recovers 5.4% at three minutes. Tick-signing the aggressive leg gives −0.064 (−1.6)
against +1.232 (+26.5) under the quote rule — tick conditioning destroys information rather
than manufacturing a negative.

### 1.4 Signing robustness — array 14137566 (`hidden_sr474`)
Baseline +2.00 (23.8); staler mid (1s earlier) −0.02 (−0.4); forward mid (look-ahead)
−1.59 (−39.2); outside-the-quote only +2.09 (12.5).

### 1.5 Spread-scaling law (message-level) — arrays 14368098, 14368227, 14368630
**Superseded by §1.21**, which rebuilds this on economic packets over 474 names and reaches
the same conclusion with a higher correlation. Retained for comparison only.
Across 40 names spanning a fourfold spread range:
`mk3 = 0.033 + 0.709 × half-spread`, cross-name correlation **+0.790**.
Ratio mk3/half-spread by spread quintile: 0.84, 0.60, 0.85, 0.65, 0.71.
**0 of 40 names have mk3 > 2 × half-spread.** 68 burst definitions tested; none positive net.

### 1.6 Huang–Stoll by sweep group — array 14292066 (`hid_sw3`)
| | quoted/2 | effective/2 | impact from t | from t+1s |
|---|---|---|---|---|
| swept | 2.744 | 1.845 | 2.785 (151%) | 0.359 (19.5%) |
| unswept | 2.747 | 1.708 | 0.397 (23.2%) | 0.381 (22.3%) |

Adverse-selection share of the effective half-spread: **22%–51%**, depending on whether
movement coincident with the touch changing counts as information. The 151% figure is not a
possible share and is reported only as evidence that the from-t measurement overstates.

### 1.7 What happens at the touch — array 14292066
Among prints whose touch changes within 100ms: consumption only 20.6%, withdrawal only 28.7%,
both 14.6%, neither 36.1%. Withdrawal is involved in 43% of sweeps, consumption in 35%.

Measured from t+1s all classes converge: consumption +0.354, withdrawal +0.249, both +0.163,
unswept +0.381.

### 1.8 Sweep, non-circular — array 14244524 (`hid_sw2`)
Conditioning on [t, t+100ms], measuring from t+1s: swept +0.374 (9.7), unswept +0.439 (22.0).
From t+5s: +0.203 vs +0.260. **The sweep does not mark differentially informative flow.**
29.8% of aggressive prints are followed by a touch sweep within 100ms.

### 1.9 Hasbrouck VAR specification grid — array 14242281 (`hid_fin`)
Cumulative response to a 1-SD flow innovation (bps), |IRF| ≤ 50 trim:

| specification | 3 min | 10 min | 30 min | % stationary |
|---|---|---|---|---|
| 10s clock, 12 lags, no ridge | +0.043 | +0.047 | +0.047 | 99% |
| 10s clock, 12 lags, ridge | +0.042 | +0.046 | +0.048 | 99% |
| 60s clock, 30 lags, no ridge | +0.017 | −0.047 | −0.145 | 88% |
| 60s clock, 30 lags, ridge | +0.015 | −0.051 | −0.149 | 89% |

Ridge is irrelevant; the sampling clock is decisive. The 10s rows return the same number three
times because VAR(12) on a 10s clock has 120s of memory — every horizon beyond that is the
model's asymptote. Robust claim: decay to zero or below by ten minutes. The thirty-minute cell
is trim-sensitive (at a 1e4 trim its t runs −1.9 to +1.4).

### 1.10 Intraday term structure, full panel — array 14242281
Gross markout: +2.023 (3 min), +2.122, +2.166, +2.131, +2.076, **+2.014 (to close)**.
TOD-stratified placebo ≤ |0.42| at every horizon. Flat across the session on the gross
measure, so the profile does not depend on the placebo adjustment.

### 1.11 Incremental to visible order-flow imbalance — array 14155326 (`hidden_ofi474`)
Univariate: hidden +0.160 (12.2), visible OFI +2.351 (54.6).
Joint: hidden **+0.198 (22.9)**, OFI +2.330 (54.2). Enhancement, not attenuation — classical
suppression with ρ = −0.016, and the algebra reconciles to +0.1980 against +0.1981 reported.

### 1.12 Information events — array 14132159 + `pull_earnings.py`
All name-days +1.910 (29.3); earnings window ±1d +3.002; excluding earnings +1.874;
excluding top decile of moves +1.751; **excluding both (88% of panel) +1.760 (26.8)** —
94% retention. Earnings dates from Yahoo Finance via yfinance (retail-grade source).

### 1.13 Pre-drift decomposition — array 14227671
Outside-the-quote prints, mean pre-drift −30s = +1.301 (37.0):

| horizon | markout | orthogonal to pre-drift | continuation |
|---|---|---|---|
| 3 min | +1.379 (32.0) | +1.407 (36.0) | −0.028 (−1.9) |
| 10 min | +1.228 (23.4) | +1.285 (24.2) | −0.057 (−3.1) |
| 30 min | +1.067 (16.3) | +1.154 (17.3) | −0.088 (−3.2) |

47.5% of the signed move over [−30s, +180s] precedes the print, but the pre-drift does **not**
forecast what follows.

### 1.14 VAR-frequency bridge — array 14292066
Regressing the markout on the VAR's own conditioning set (thirty one-minute signed lags):
raw mean +1.051 (47.9), intercept +0.931 (48.8) — the minute-scale path absorbs **11.5%**,
mean within-name-day R² = 0.35. The bridge fails at the VAR's frequency as well as at 30s.

### 1.15 Multi-day behaviour — `multiday_power.py` (local)
Calendar-time portfolio, overlapping k-day holds, 501 daily observations per horizon:
1d −0.69 (−0.55), 5d −1.11 (−0.42), 10d −1.08 (−0.34), **20d −1.95 (−0.40)**.
Every estimate within 2 bps of zero. SE at 20 days ≈ 4.9 bps.

### 1.16 Trade count and volatility — arrays 14463166 (`harall`), 14482647 (`hincr`)
473 names, 500 dates. Counts forecast realized volatility incrementally to an intraday HAR:
+0.101 (60s), +0.097 (300s). Temporally out-of-sample: 2023 +0.0910 → 2024 +0.1110.

Decomposed: **visible count contributes 93%** (+0.0944), hidden count 7% (+0.0068), and the
hidden term has mean t = **−0.25**, with 1 of 472 names reaching mean t > 2. This is a
replication of Jones–Kaul–Lipson (1994), not a new result.

### 1.17 Point-in-time reversal — array 14489997 (`pitflow`)
1,794 names × 1,028 dates including 12 recovered delistings (SIVB, SBNY, FRC, ATVI, VMW,
SGEN, SPLK, PXD, TWTR, ABMD, HZNP, CTLT). Calibrated 2022, evaluated forward 2023–26:

| signal | 2022 | 2023+ | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| signed visible flow | −1.82 | +0.48 (t 0.86) | +1.29 | +0.63 | −0.26 |
| signed visible / volume | −1.00 | +0.52 (t 0.95) | +1.53 | +0.74 | −0.69 |
| signed hidden flow | −0.19 | +0.49 (t 0.87) | −0.72 | +0.81 | +0.97 |

Not significant. Turnover 0.346/day. The calibration year is negative for all three.

### 1.18 Packet-fragment price discovery and trading — arrays 14575617–14575712

Models fit on 2023 non-holdout names and frozen before 2024. Official test panel: 469 usable
names, 112,779 name-days, 251 dates. On the deterministic 20% held-out-name cohort, the four
formation/post-end specifications estimate only +0.210 to +0.431 bps of the persistent-price
proxy, with t = 0.98–1.60. Every executable continuation strategy is negative: −9.82 to
−12.48 bps at five minutes and −18.10 to −24.09 bps at thirty minutes. Some price outcomes
are significant among names used in fitting, but do not transport to new names. This closes
the formation-state directional strategy; persistent-rate levels are excluded because the
first compact output omitted their unconditional benchmark.

### 1.19 Strict packet-flow continuation — arrays 14593499, 14597703–14597708

Frozen on 2023 non-holdout names and tested once on untouched 2025. Final panel: 473 usable
names (381 seen, 92 held out), 111,521 name-days, 249 dates. The independent schema and
NW(10) recomputation agrees with the production result to `3.6e-15`.

| cohort | 300s delta MSE | NW t | top-decile base residual packets | NW t | incremental MSE fraction |
|---|---:|---:|---:|---:|---:|
| seen names | +0.1160 | 6.26 | +1.4708 | 12.44 | 0.0046% |
| held-out names | +0.1048 | 6.09 | +1.9068 | 10.49 | 0.0032% |

The 60-second count target agrees in sign in both cohorts (delta MSE t = 8.16 and 7.35), and
both signed-log volume horizons corroborate. This verifies incremental continuation of
same-side aggressive packets. It does **not** verify common parentage, institutional identity,
private information, price direction, or executable profitability; the global predictive lift
is statistically precise but economically small.

### 1.20 Liquidity-sensitive-pause mechanism — stopped after arrays 14598875–14598876

The mechanism test was frozen before estimation and stopped at its 2023 training-sign gate,
before any 2025 output. The fit contains 30,674,288 risk rows and 8,384,936 events across 382
non-heldout names. Predeclared order-splitting behavior required the top-decile interaction
with spread change to be negative and the interaction with executable-side depth change to be
positive. Both single and joint models give the opposite signs: spread +0.01270/+0.01271;
contra depth -0.00492/-0.00726. This rejects the specified liquidity-pause fingerprint and
closes that specified fingerprint test. It does not close all anonymous-metaorder mechanism tests. It does not reverse §1.19 and should not
be interpreted causally as a preference for poor liquidity.

### 1.21 Spread-scaling law on economic packets — arrays 14610572, 14610573

Rebuild of §1.5 on `execution_packets.reconstruct_packets`, so the unit is an economic order
rather than an execution message. Definitions were fixed in the extractor docstring before the
panel ran. Panel: 474 names, 228,927 name-days, 500 dates, 2023–2024 only (zero 2025 rows).
Signed economic packets carry **1.87 execution messages each on average** (median 1.72, p95
2.90, max 9.84), which is the size of the message-level weighting error.

Cross-name regression of the name's mean 3-minute markout on its mean half-spread, HC1 errors:

| definition | formation | slope | intercept | R² | ratio median | names clearing 2×half-spread |
|---|---|---:|---:|---:|---:|---:|
| all signed packets | none | +0.629 (26.4) | −0.111 (−1.60) | 0.830 | 0.590 | **0 of 474** |
| 10× volume blocks | size only | +0.776 (24.9) | −0.239 (−2.64) | 0.795 | 0.694 | **0 of 474** |
| run3 | same-sign runs, ≤1s gaps | +0.312 (13.1) | −0.239 (−3.48) | 0.592 | 0.201 | **0 of 474** |
| clust3 | timing only, signed after the fact | +0.297 (14.6) | −0.200 (−3.39) | 0.658 | 0.203 | **0 of 474** |

Cross-name correlation for the headline row is **+0.911** (message-level §1.5: +0.790 on 40
names). The intercept is statistically indistinguishable from zero, so there is no fixed
basis-point component and therefore no spread below which the signal becomes capturable. The
ratio is flat across spread quintiles (0.657, 0.576, 0.563, 0.590, 0.612). **0 of 474 names
clear a round trip in any of the five definitions at any of the three measurement bases**
(15 of 15 cells zero).

Two further readings, both pre-specified:

- **Burst formation is not manufacturing the law.** `run3` conditions its block boundaries on
  sign; `clust3` cuts on timing alone and assigns the sign afterwards. They agree to within
  0.02 bps (0.834 vs 0.821) and 0.02 in slope. This is the circularity that invalidated the
  Hurst validator, tested directly, and it is absent here.
- **Burst formation destroys signal rather than concentrating it.** Every signed packet gives
  +2.053 bps; clustering the same packets into bursts gives +0.83. Clustering more than halves
  the measured footprint, which is the opposite of the order-splitting premise.

**Out-of-sample confirmed — arrays 14621859, 14621860.** Five gates were frozen in
`config/packet_scaling_gate.json` (`b2fea4e099cd`) before the 2025 panel ran, with an explicit
no-respecification rule. Applied once to untouched 2025 (472 names, 249 dates), **all five
pass**: slope +0.5624 (t = 16.78), intercept −0.0978 (**t = −0.79**, indistinguishable from
zero), R² 0.781, ratio median 0.530, **0 of 472 names clearing a round trip**, run3/clust3
formation gap 0.041 bps, slope deviation from training 0.067. The independent auditor
(`audit_packet_scaling.py`, `1349d5e10888` — own CSV reader, QR rather than lstsq, no import
of the production aggregator) reproduces production to `0.0` on slope and `2.8e-17` on
intercept. This is the second result in the project with frozen-gate, single-shot,
independently-recomputed standing.

Measured from a one-second buffer the level collapses to +0.359 (ratio 0.075), the intercept
turns significantly negative (−0.153, t = −4.47), and 33 of 474 names go negative.

### 1.22 Hidden liquidity is not identified on economic packets — arrays 14621851, 14621852

Frozen as `hidden-packet-bounds-v1` before estimation: execution messages collapsed into
timestamp-level economic packets, type-5 `Direction` ignored entirely, no bursts, outcomes
measured from a one-second buffer, and 2023–2024 replication reported separately from 2025
confirmation. Coverage 474 names both periods.

**What can be signed at all.** A hidden packet is defensibly signed only when a same-timestamp
visible execution supplies the native aggressor side ("mixed"), or when its price lies outside
the pre-print quote ("outside").

| | mixed | outside | **unsigned** | at-midpoint |
|---|---:|---:|---:|---:|
| share of hidden packets, 2023–24 | 15.6% | 0.0007% | **84.4%** | 47.9% |
| share of hidden packets, 2025 | 13.1% | 0.0016% | **86.9%** | 43.3% |
| share of hidden volume, 2023–24 | 22.0% | — | **78.0%** | — |

The outside-quote rule, which signed ~2.5% of *messages* in §1.2, signs roughly one packet in
100,000. Collapsing a timestamp usually merges an outside-quote hidden print with visible
executions, so it becomes "mixed" instead.

**Signed markouts, bps (t in parentheses), 3 min / 15 min / 30 min:**

| series | 2023–2024 | 2025 |
|---|---|---|
| away from mid, quote-signed | +0.603 (22.5) / +0.534 (14.3) / +0.525 (12.1) | +0.944 (19.5) / +0.978 (15.4) / +0.955 (18.6) |
| at midpoint, tick-signed | −0.056 (−4.0) / −0.109 (−4.5) / −0.140 (−3.8) | −0.037 (−1.2) / −0.119 (−2.8) / −0.199 (−3.5) |
| defensibly signed only | +0.194 (6.4) / +0.161 (4.4) / +0.167 (4.4) | +0.423 (7.4) / +0.453 (5.8) / +0.452 (5.2) |
| sharp bounds, all hidden | [−9.76, +9.83] | [−11.24, +11.35] |
| sharp bounds, 30 min | [−27.80, +27.87] | [−31.51, +31.63] |

**Two conclusions, both pre-registered.**

1. `conventional_bifurcation_replicates = True`. The frozen gate checks **signs only** (3-minute
   aggressive leg > 0 and midpoint leg < 0 in each period). The aggressive leg is positive and
   significant at every horizon in both periods. The midpoint leg is negative and significant
   at every horizon in 2023–2024, and at 15 and 30 minutes in 2025, but **not at 3 minutes in
   2025 (−0.037, t = −1.2)**. *Corrected 2026-09-13:* this item previously said the midpoint leg
   was significant in both periods.
2. `minimal_positive_identification = False`. Because 84–87% of hidden packets carry no
   defensible sign, the sharp worst-case bounds span roughly ±10 bps at three minutes and ±30
   at thirty. They contain zero by a wide margin at every horizon in both periods. **The
   magnitude of the hidden-liquidity footprint is not identified from anonymous type-5 data.**

The defensibly-signed subset is positive and significant (+0.19 to +0.45), but it is 13–16% of
packets selected precisely because a visible execution accompanied them, which is not a random
sample of concealed trading.

### 1.23 Hidden-footprint spread scaling — arrays 14621853, 14621854

A parallel panel, 472–473 names, both periods. Every defensible subset behaves like
§1.21: **0 names clear 2× the half-spread** under `all_conventional`, `away_quote`, and
`known`, in both 2023–2024 and 2025.

The one cell that appears to clear is `outside` in 2025 — 27 of 107 names. It should not be
cited. That subset is the ~1-in-100,000 packet population of §1.22, its cross-name correlation
is 0.121 against 0.849 for `away_quote`, its intercept is −0.819, and its ratio-by-quintile
profile is non-monotone with a negative first quintile (−0.512, 0.348, 0.600, 1.072, 0.071).
It is small-sample noise, and it is the first thing a referee will point at.

---

### 1.24 Burst-information screen — exploratory, completed 2026-09-13

**Verified computation; its model contrasts are uninformative about bursts (corrected
2026-09-13).** Arrays 14732764, 14732766 and 14732767 supply 1,410 available stock-days (30
confirmed missing), 64,618 valid sampled landmarks over 36 names and 40 dates. Training: 2023,
24 names; evaluation: 2024, 24 seen plus 12 held-out names. Both years had been previously
explored. The 75 fits, 120 contrasts and 90 execution statistics reproduce under the
independent audit (maximum discrepancy 2.84e-14) — the audit verifies arithmetic, not design.

Post-hoc diagnostics (`src_py/diagnose_burst_information_v1.py`, outputs in `posthoc/`) establish
what the screen can and cannot say:

- **Returns and waiting costs are not predicted at all.** Against a zero forecast, all **90 of
  90** return and wait-cost model cells (3 landmarks × 3 targets × 2 cohorts × 5 models) have
  negative out-of-sample R². The best, ridge on state, is −2.6% at the third packet (60s,
  held-out). Relative MSE differences between these models are differences between forecasts
  that all lose to zero.
- **Future signed flow is predicted** (R² against zero of 1–15% across flow cells), from
  trailing flow and book state.
- **The primary flow contrast measures two names.** Targets are raw packet counts. In the
  held-out cohort AMZN contributes 52.5% and AMD 41.1% of baseline MSE and 61% of the burst
  contrast; no training name approaches their activity. The contrast is an extrapolation test
  of tree models, not a test of burst information.
- **Refit on per-name scale-free targets** (flow ÷ trailing signed-packet rate × horizon;
  return ÷ half-spread), adding burst features improves 12 of 32 stage/target/model/cohort
  cells, significantly so in 2 and significantly worse in 5. No consistent burst increment.
- **Design limits:** 1-in-32 landmark sampling left 5,445 / 952 / 5,438 training rows at the
  third / sixth / completion landmarks, evaluated on 20 dates with NW(10). The burst block's
  `depth_imbalance_start` is not signed by burst direction while every target is.

Only the statements above may be cited from this screen. The v1 relative-MSE figures are
listed in §2 as excluded evidence.

### 1.25 Controlled burst recovery and corrected join diagnostic — synthetic only

Thirty paired seeds across nine conditions and three detectors yield 810 independently
audited cells (maximum discrepancy 1.1e-16). Same-side one-second burst pair precision/recall:
isolated parents 100%/60.4%; random pauses 100%/25.8%; dense background 15.6%/13.9%.
These quantify fragmentation and false merging under the specified simulator, not NASDAQ
parent accuracy. An identical observable tape can be assigned different latent parent labels;
this establishes lack of unique identity without generating assumptions, not universal
statistical indistinguishability of splitting and herding.

The session-mixing defect in legacy join training is real (67.3% of legacy training pairs
crossed simulated days) and is excluded in §2. The corrected join model's synthetic accuracy is
**not evidence of reconstruction ability** (corrected 2026-09-13): on the same 2,880 test
pairs, the time gap alone has AUC 0.955 against the model's 0.966, a fixed "gap < 30s" rule
gives 92.7% precision and 88.4% recall, and the simulator's book-state change features have AUC
0.49–0.50 because its book is generated independently of parents. The legacy simulator has
~45 parents and ~190 fragments per day, so consecutive same-side fragments are almost always
one parent; real names produce thousands of qualifying bursts per day. The any-program
participation label (`metaorder_participation.py`) changes the positive class for only 447 of
4,927 simulated fragments (9%), because simulated parents rarely overlap. Details:
`studies/burst_information/BURST_RECONSTRUCTION_RESULTS.md`.

### 1.26 Prices respond to surprise flow, not predicted flow — exploratory, 2026-09-13

Post-hoc, on the burst-information-v1 landmarks (1-in-32 sampled, already-used years). For each
name with at least 150 landmarks in both years, a 2023 OLS of future same-side net packet flow
on trailing 60s/300s signed imbalance and 300s activity predicts 2024 flow (median
out-of-sample R² 7.1% / 9.1% at 300s for third-packet / completion landmarks, ~1% at 60s). The
2024 signed return over the same window is regressed on predicted flow and the surprise:

| landmark, horizon | names | bps per predicted packet (median) | bps per surprise packet (median) | predicted t > 2 | surprise t > 2 |
|---|---:|---:|---:|---:|---:|
| third, 60s | 21 | 0.004 | 0.527 | 6 | 21 |
| third, 300s | 21 | 0.040 | 0.282 | 4 | 19 |
| completion, 60s | 20 | 0.132 | 0.424 | 4 | 20 |
| completion, 300s | 20 | 0.021 | 0.262 | 6 | 18 |

The predicted coefficient is below the surprise coefficient in 16–21 names per cell. This is
the asymmetric-liquidity mechanism (Lillo & Farmer 2004; Farmer, Gerig, Lillo & Mike 2006) in
this panel: predictable continuation is already priced, so detecting ongoing programs from the
public tape does not by itself predict returns. Within-name deciles of a 2023-fitted ridge
return forecast agree: 39 of 40 decile cells (2 landmarks × 2 horizons × 10 deciles) lose money
after paying the half-spread at the reference time. Contemporaneous-window regression; a
mechanism diagnostic, not a strategy. Outputs: `results/burst_information_v1/posthoc/`.

### 1.27 Same-origin fingerprint and burst-definition validation — fingerprint-v1, 2026-09-13

Frozen design with pre-read amendments (`studies/fingerprint/BURST_FINGERPRINT_DESIGN.md`, `freeze_v2`–`freeze_v5`).
Exploration: 2024, 174 names, 3,480 name-days. Confirmation: 2021, 291 disjoint names, 5,702
name-days, read once. Full results and figures: `studies/fingerprint/BURST_FINGERPRINT_RESULTS.md`.

- **Existence (C1 pass).** Same-side untruncated non-round packets with identical sizes recur
  above the depth-matched adjacent-day chance rate. Name-median observed ÷ chance:

  | lag | 2024 | 2021 |
  |---|---|---|
  | 0.5–2 s | 2.16 [2.01, 2.41] | 2.55 [2.33, 2.74] |
  | 2–10 s | 2.05 [1.81, 2.29] | 2.04 [1.95, 2.18] |

  - 97–99% of names are above 1.
  - Controls: round lots 1.00–1.05; excluding hidden fills leaves the ratio unchanged.
  - Activity-plus-depth matching (2024 sensitivity) gives 2.24 / 2.10.
- **Definitions (C3 fail on rank, pass on stability).** On mean per-name Youden J over lags below
  300 s, uninterrupted same-side runs (30–300 s caps) and side-only 2–5 s streams tie.
  - Run 60 s: 0.360 (2024), 0.258 (2021). Run 300 s (pre-declared): 0.361, then 0.257, rank 7.
  - Stream 5 s: 0.347, then 0.263, rank 1.
  - The ranking of all 33 definitions replicates (Spearman 0.954).
  - The winning definitions keep 40–57% of same-origin evidence inside a burst while grouping
    13–22% of same-side pairs. A 60 s side-only stream scores J ≈ 0.05.
- **Book state (C2 fail).** Identical-size recurrences occur at more similar spreads than
  lag-matched size-similar controls (0.90–0.95), but at less similar displayed depth
  (1.05–1.14), in both years.
- **Program score (P1, P2 pass).** A size-free score fit on 2024 bursts ranks 2021 bursts from 51
  to 241 excess identical-size repeats per 1,000 pairs, bottom to top decile (+189 [150, 232];
  monotone). Timing geometry alone gives +132 [93, 180]; context alone gives +17 [−49, 70].

Scope: a lower bound on common origin with relative, not absolute, purity. It is not a
retail/institutional label; LOBSTER has no aggressor identity (§2 row on MPIDs). No return claim.

### 1.28 Evidence that fingerprint bursts are directional execution programs — program-evidence-v1, 2026-09-14

Pre-registered (`studies/program_evidence/PROGRAM_EVIDENCE_DESIGN.md`, freezes `freeze_v1`–`freeze_v3e`). The panels are the
fingerprint-v1 caches: 2024 exploration (174 names) and 2021 confirmation (291 disjoint names).
Full tables: `studies/program_evidence/PROGRAM_EVIDENCE_RESULTS.md`. Arrays: 14738169/70, 14738854, 14738800/04, 14738877.

- **Stage 3b (P1b, P2b pass).** The size-free score refit on run/60 bursts ranks 2021 bursts from
  51 to 241 excess repeats per 1,000 pairs (+190 [151, 234]).
- **Timing fingerprint (A1, A2 pass; A3 fail).**
  - Same-side lags lock within ±10 ms of whole seconds more on the same day than across days:
    1.27 [1.22, 1.31] (2024), 1.14 [1.12, 1.16] (2021).
  - Identical-size pairs are more phase-locked than different-size pairs: 1.62 [1.52, 1.71] and
    1.19 [1.15, 1.23]. The excess grows as the window shrinks to 1 ms.
  - Phase and size J disagree on the best burst definition. Spearman across 33 definitions: 0.60
    (2024), 0.76 (2021). Timing favours 2–10 s streams; size favours runs.
- **Directionality (B1, B2 pass; B3 fail).** Identical-size same-side excess exceeds opposite-side
  excess at 2–10 s, 10–60 s, 60–600 s and 600–3,600 s in both years. Same-side excess persists at
  10–60 min: 1.11–1.15 (2024) and 1.31–1.37 (2021).
  - Opposite-side excess is ≈ 1.0 in 2024 but 1.13–1.42 in 2021.
  - Post-hoc B4: with prices matched within 10 bps, the 2021 opposite-side excess stays at
    1.23–1.37 and the same-side excess at 1.4–2.3. By the pre-written rule, 2021 had many two-sided
    fixed-clip algorithms; this is not a dollar-sizing artifact.
  - Multi-day fixed-size campaigns: 1.014 [1.006, 1.025] in 2024, not replicated in 2021 (1.006
    [0.995, 1.018]).
- **Liquidity-provider markouts (F1 pass).** Touch fills inside program-like bursts pay providers
  +2.51 bps (t = 31.7, 2024) and +2.28 bps (t = 25.7, 2021) more at 60 s than fills inside
  bottom-quintile bursts.
  - Post-hoc F2: within equal truncation-share strata the gap is +0.82 bps in both years
    (t = 5.2, 7.8). About two-thirds of the headline gap is book-sweeping.
- **Cross-name synchrony (G1 pass; G2 fail).**
  - Same-side untruncated packets in different stocks coincide within 1 ms 18.7× (2024) and 28.1×
    (2021) more often than at shifted times.
  - Program-burst packets synchronize more: difference 16.1 [10.7, 22.4] and 7.6 [4.3, 12.1].
  - ETF-overlap slope t = 2.81 (2024; hurdle 3) and 2.26 (2021).
- **Passive side (H1, H2 pass).**
  - Identical non-round limit-order sizes recur on the same side 1.75× (2024) and 2.43× (2021).
  - Aggressive children share an exact size with same-side passive orders within 10 s 1.10× and
    1.13× (opposite side 1.05×, 1.08×).
- **Not supported.**
  - Dollar-sized children (I1 fails; about 100–200 informative pairs).
  - Metaorder physics of 4–5-child fingerprint campaigns: the impact exponent is outside
    [0.3, 0.7] and the reversion intervals span zero.

Scope: none of this labels a burst as institutional. The external WRDS tests and the daily
program-imbalance tests (modules D, E) are in §1.29.

### 1.29 What program flow is: daily panels, WRDS externals, trend and the Tick Size Pilot — program-evidence-v1, 2026-09-14

Point-in-time CRSP top-500 universes intersected with lobster2 coverage: 2024 exploration 163
names, 39,121 name-days; 2021 confirmation 223 names, 55,426 name-days. Arrays: extraction
14738827/28/29, flows 14743294. Program bursts = run/60 bursts with stage-3b score ≥ the 2024 q80.
Fama–MacBeth, NW(10).

- **The original daily idea is null.** Program buy-minus-sell imbalance does not predict next-day,
  open-to-close or next-5-day returns in either year (|t| ≤ 1.5; per s.d. −0.5 and +0.3 bps next
  day).
- **Program imbalance persists less than other flow.** PI→PI 0.21 / 0.18 against NPI→NPI 0.33 /
  0.25; difference t = −8.7 (2024) and −7.1 (2021). The pre-declared direction was the opposite.
- **Mutual-fund trading (CRSP holdings) tracks non-program flow, not program flow.**
  - Quarterly ΔMF on non-program net buying: t = 3.4 (2024) and 2.4 (2021).
  - On program net buying: t = −0.6 and 1.9.
  - Flow-induced trading on program net buying: t = 0.3 and 2.9, not replicated.
- **ETF basket demand tilts program flow, replicated.** The within-type imbalance slope on same-day
  ETF implied demand is larger for program than for other flow by +0.020 (t = 7.3, 2024) and
  +0.017 (t = 5.5, 2021).
- **Retail and block alignment, replicated.**
  - corr(program, BJZZ retail) exceeds corr(other, retail) by +0.028 (t = 3.9) and +0.012 (t = 2.4).
  - corr with ≥ $50k trade imbalance is lower for program flow: −0.071 (t = −10.5) and −0.019
    (t = −3.1).
- **Index changes (one shot, 68 of 103 S&P 500 / Nasdaq-100 events; gate fails).** Over E−5..E−1,
  additions minus deletions:
  - program imbalance z −0.25 (Welch t = −1.45);
  - other imbalance +0.34 (t = 2.07);
  - program minus other −0.59 (t = −2.94).

  Program flow sells into index-addition demand.
- **Trend, 2013–2024 (descriptive, 130-name balanced panel).** Identical-size fingerprint at
  0.5–2 s: 5.6 (2016) → 3.4 (2019) → 2.8 (2021) → 2.6 (2024). Run60 J holds at 0.27–0.37;
  untruncated share 0.45 → 0.36.
- **Tick Size Pilot (exploratory; 144 pilot vs 125 control names, Apr–Sep 2016 vs Nov 2016–Jun
  2017).**
  - Half-spread DiD +9.41 bps [6.63, 12.50]; untruncated share +0.096.
  - Run60 3-minute burst markout DiD −0.83 [−3.15, 1.39].
  - Instrumented markout/half-spread slope −0.13 [−1.09, 0.29], which excludes the cross-sectional
    0.71 of §1.21.

Caveat: lobster2 coverage correlates with performance. Covered names returned 16.5% / 26.4%,
uncovered 4.3% / 1.8%.

Scope: these results characterize flow selected by the program score as intermediary and arbitrage
algorithms (ETF baskets, retail hedging), not institutional parents. They do not show that
institutional parents are absent from the tape.

### 1.30 Child clusters, linkage score, calibration and passive orders — metaorder-v1, 2026-09-14

Pre-registered (`studies/metaorder/METAORDER_DESIGN.md`, freezes `freeze_v1`–`freeze_v4`). Arrays 14745612/13,
14747903, 14747916, 14748434. Full tables: `studies/metaorder/METAORDER_RESULTS.md`.

- **Child-cluster rule (M1 tie).** Combined size + phase J, stream5 minus run60:
  - 2019 (gate) +0.028 [−0.007, +0.061];
  - 2016 +0.039 [0.010, 0.072]; 2013 −0.050 [−0.080, −0.019];
  - seen years: 2021 +0.039, 2024 +0.001.

  Size-only differences are within ±0.02 in every year. run60 stays.
- **Linkage score (L1, L3 pass; L2 fail).** Fit on 2024 run60 bursts, scored on 2021:
  - same-side excess cross-burst identical-size links per 1,000 pairs rise from −0.9 to 61.8 across
    deciles (top − bottom +62.7 [31.7, 96.1], Spearman 1.0);
  - opposite-side links rise to 55.5, so the directional difference is +5.5 [−4.0, 21.8].
- **Real-time score (R1, R2 pass).** First three own-side packets plus backward context: +169.3
  [107.4, 237.2] excess repeats per 1,000 pairs top − bottom in 2021 (Spearman 0.94), against
  +190.4 for the whole-burst score.
- **Synthetic parents (M3, 2024, descriptive).** 2,335 injected parents (median 10 children,
  capped at 1% of daily volume).
  - Children inside run60 bursts: 76% / 54% / 42% / 36% at 5 / 15 / 30 / 60 s intervals.
  - Burst purity: 70% / 49% / 35% / 26%.
  - Among parents with ≥ 2 bursts, size linkage recall is 49% (fixed clip) and 31% (randomized).
  - Score AUC 0.97, an upper bound.
- **Passive orders (M5, Q1 fails both years).** Queue-aware 100-share postings at the touch.
  - Per-fill 60 s markout is negative for every trigger type: −1.5 to −2.1 bps.
  - Program-trigger postings: fill 50–51%, P&L per posting −0.87 (2024) and −1.03 bps (2021),
    against −0.53 and −0.45 at placebo times.
  - Program minus bottom per fill: +0.33 (t = 2.4, 2024) and −0.34 (t = −5.6, 2021).

### 1.31 Informed bursts without leakage: three periods, four cells — p4-revisit-v1, 2026-09-15

Pre-registered (`studies/p4_revisit/P4_REVISIT_DESIGN.md`, freeze `3a33beb216b3`, amendments A1–A6). Aggregations 14760411/12
and the DEV/VAL re-runs; analyses 14760422/23/24. Prior-work audit: `studies/p4_revisit/PRIOR_WORK_INVENTORY.md`. Full tables:
`studies/p4_revisit/P4_REVISIT_RESULTS.md`. 1,267,489 name-days present of 2,435,582 requested, 2012–2025.

Decision-time firewall: T_dec = max(t_b + 600 s, t_e + 10 s); a burst may inform a signal only if
T_dec ≤ the signal clock. Cells are name-split by sha256; VAL, TEST and ERA2 were each read once, in order.

- **H1 refuted: informative impact does not persist past T_dec.** Displacement from the decision mid to the
  close, informative minus pseudo-burst placebo (trade bursts):
  - 2012–16 **−0.60 bps (t −7.80)**; 2020–21 **−0.74 (t −2.19)**; 2022–25 **−0.19 (t −1.52)**.
  - Against the bursts the filter rejected, the gap is −2.04, −3.36 and −2.34 bps (|t| 5.2–14.1).
  - Phase I deciles of D_b/PeakImpact are monotone **inverted** in all three periods.
- **Q2 institutional association: passes as pre-registered, narrower after a post-hoc return control.**
  Quarter FE, PERMNO-clustered; informative minus other large burst flow; 2012–16 / 2020–21 / 2022–25.
  - Pre-registered, trade bursts: 13F ΔIO +0.069 (t 3.40) / +0.067 (4.20) / +0.045 (3.79); CRSP mutual-fund
    Δholdings +0.038 (7.09) / +0.040 (4.10) / +0.054 (7.16). Informative flow loads positive and other large
    burst flow negative in every cell.
  - **Post hoc (2026-09-20), adding the stock's same-quarter return.** "Informative" is chosen by price moving
    the burst's way, so its quarterly flow correlates +0.15 to +0.21 with that return (other flow −0.05 to
    −0.12), and ownership changes co-move with contemporaneous returns. With the control: trade 13F
    **+0.071 (3.30) / +0.039 (2.45) / +0.034 (2.80)**; trade mutual funds **+0.025 (4.44)** / +0.012 (1.19) /
    **+0.041 (5.21)**. Submission bursts fall below t 2 on every test in 2022–25. **Cite the controlled numbers.**
  - Lou (2012) flow-induced trading passes for submission bursts as pre-registered (t 5.28 / 5.48 / 3.49) but
    only 1 of 6 cells survives the return control. Do not cite FIT as corroboration.
  - Retail (BJZZ): informative flow is less retail-aligned (−0.014, −0.013, −0.002), but retail flow is
    contrarian to returns and this was not re-run with the control — not evidence of institutional
    participation. Index events are uninformative (56–87 events).
- **Q2(a) within-tape linkage separates the families.** Informative minus other, same-side minus opposite-side:
  trade +0.0068 / +0.0035 / +0.0046, CI > 0 in all three; submission −0.0002 / −0.0014 / **+0.0012**, i.e. two
  significant opposite answers. Trade bursts are the "more promising" family on this basis, and also because
  only their institutional association survives the return control in 2022–25.
- **Q3 holds: the give-back is predictable at T_dec.** Ridge fit on DEV 2017–19, daily Spearman IC on
  d_close: trade +0.052 (t 11.5) / +0.040 (t 5.97) / +0.039 (t 7.73); submission +0.029 / +0.022 / +0.023.
  VAL and TEST are name-disjoint from the fit; ERA2 is not.
- **Q4 refuted: no tradable daily signal.** Six cells × three periods under Holm with an HLZ t > 3 hurdle.
  One cell passed validation — submission CLOP, t −3.16, Holm p 0.0016, reversal-signed — and **failed to
  replicate in TEST** (t −0.37 against a pre-registered t > 2). Every decile book loses after costs; the one
  gross-positive book (trade CLOP in VAL, +7.83 bps/day, SR 2.46) has a deflated-Sharpe probability of 0.0015
  net against 311 recorded trials.
- **Q0, the legacy pipeline.** All regular-hours type-5 messages carry Direction +1, and the legacy C++
  detector signed every one as a sell: 97.6% net-short name-days against 49–53% corrected. That explains the
  old net-short tilt but not the old 37.8% hit rate (in-sample tuning) and not the flagship Sharpes (NVDA 1.58
  against a 0.45 long-only overnight). The legacy κ gate is circular: it lifts a 3-minute markout from +1.09
  to +11.88 bps regardless of signing.

Coverage is stated with every result: 34% of requested name-days in 2012–16 (NASDAQ names 23%) rising to 80%
in 2022–25 (NASDAQ 81%). The confirmatory cell is the best-covered one.

### 1.32 Execution algorithms carry their size fingerprint across days — fingerprint-multiday-v1, 2026-09-23

Pre-registered (`studies/fingerprint_multiday/FINGERPRINT_MULTIDAY_DESIGN.md`, freeze `49a39405cc61`, amendments A1–A2). Stage-1 jobs
14831007–14831010 over the p4-revisit-v1 per-burst files; no new market data. Tables:
`studies/fingerprint_multiday/FINGERPRINT_MULTIDAY_RESULTS.md`.

- **H1 passes its gate and replicates in four disjoint periods.** Among fingerprint-bearing trade bursts (modal
  untruncated child size repeating inside the burst, non-round), the cross-day share of burst pairs sharing the
  modal size is higher **same side** than opposite side, and the gap decays with the number of days between them.
  D = E(1) − mean E(30,40,50,60): DEV +0.0203 (t 6.78), **TEST +0.0154 (t 23.04)**, VAL +0.0182 (t 6.30),
  ERA2 +0.0163 (t 12.14); 79–93% of names positive. Secondary size set passes too (TEST t 7.07).
- Excess t by lag in TEST: 24.1 (1 day), 18.4 (3), 13.2 (10), 7.0 (20), ~4 (30–60).
- Side-symmetric explanations — two-sided algorithms, round-number conventions, chance — cancel in the
  same-minus-opposite contrast; a side-asymmetric size convention would not decay with the lag.
- **H4 passes.** Episodes (maximal runs of days where one key carries ≥ 3 one-sided bursts; TEST 342,744
  episodes, 676 names, 5.6% lasting 2+ days) have impact concave in participation: log-log slope
  **δ = 0.35, CI [0.12, 0.80]**, excluding 1, with the square-root value inside. Mean impact +16.6 bps at one
  day, +37.6 at two, +72.2 at three. The shape metaorder studies find in broker records, from the public tape.
  Caveat: episodes are observed only while they continue, so stopping is endogenous.
- **H5 fails.** DEV's post-episode reversion (3-day episodes −45.8 bps over 5 days) does not replicate in TEST
  (+5.6). With tradable timing the pre-registered cell (length ≥ 3, five days) is **−2.2 bps, t ≈ −0.1** in
  TEST against DEV's −41.5 (t −3.07). No tradable claim.
- **H2 fails.** Flow linked to the previous day (label amended to require ≥ 3 one-sided bursts at that size on
  both days, purity 0.61 by the mirror placebo) does not beat unlinked flow on 13F ΔIO in TEST: +0.182 (t 1.59)
  against a pre-registered t > 2; mutual funds +0.152 (t 2.29); DEV flat. **Multi-day linkage is a real flow
  structure with no demonstrated institutional content.**

### 1.33 Burst impact holds less when flow is mechanical — forced-flow-v1, 2026-09-23

Pre-registered (`studies/forced_flow/FORCED_FLOW_DESIGN.md`) after a DEV run and before TEST. Name fixed effects, date-clustered.

- **Quarter-end days, informative class (κ = 0.5), P4 `d_close`:** DEV −1.77 bps (t −2.74),
  **TEST −1.37 (t −2.09)** — gate passed. Normal-day means −1.46 and −1.09, so the give-back roughly doubles.
- **Non-informative bursts are unaffected** in both cells (+0.28, t 0.52; +1.01, t 1.25).
- Quarterly expiry does not replicate (+1.38 then −1.02) and is not claimed.
- Timing is exogenous to any one stock's information. On days when more trading is mechanical, what the
  informativeness filter keeps reverts more — the filter is partly labelling predictable non-information flow.

### 1.34 The informativeness filter selects away from institutional-size flow — daily-labels-v1, 2026-09-23

Pre-registered (`studies/daily_labels/DAILY_LABELS_DESIGN.md`). Daily BJZZ TAQ labels: institutional-size imbalance II (≥ $50k
trades) and retail imbalance RI. Day fixed effects, PERMNO-clustered, **same-day return controlled**.

- Informative vs other large burst flow on II — informative +0.49 / +0.34 / +0.97 (t 3.9 / 2.7 / 13.0) against
  other large +1.67 / +2.03 / +2.78 (t 10.3 / 16.2 / 31.1) in DEV / VAL / TEST; difference
  **−1.18, −1.69, −1.81 (t −5.0, −8.1, −14.1)**, replicated on disjoint names and years.
- Burst flow as a whole tracks institutional-size trading; the P4 filter cuts that loading by about two thirds.
  This is the **opposite ordering** to the quarterly 13F result in §1.31 — a daily execution-size label and a
  quarterly ownership-change label disagree about the filter.
- The two regressors are complementary subsets of the same flow, so the contrast is evidence about ordering,
  not a structural magnitude.
- **Primary of that study fails:** program-linked flow minus its mirror placebo on II is +0.99 (t 0.36) in TEST
  against a pre-registered t > 3.
- Retail: informative flow is unrelated to RI; the informative-minus-other contrast is sign-inconsistent across
  cells and is **not** claimed.

### 1.35 Cross-name burst flow, and three further probes — burst-probes, 2026-09-23

Tables: `studies/fingerprint_multiday/BURST_PROBES_RESULTS.md`. Exploration DEV 2017–19 group 0, confirmation TEST 2022–25 groups 1–2.

- **Cross-name flow (jobs 14879698/99).** Minute midpoint grids and burst flow for every covered name; pooled
  regression with name-day fixed effects, date-clustered. Peer burst flow predicts a name's next-minute return:
  **+878 (t 8.2) in DEV and +1128 (t 16.2) in TEST**, beyond its own flow (+64, +107) and its own lagged return.
  **Not stale prices:** peers' contemporaneous returns carry nothing (t 0.69, 0.75). One s.d. of peer flow
  ≈ **1.1 bps** of next-minute return — below a typical half-spread, so not tradable.
- **Multi-day premium.** Episode impact is +22.7 bps (DEV, t 6.19) and +21.5 (TEST, t 5.93) higher for episodes
  of 2+ days than for single-day episodes at equal participation and volatility. Stopping is endogenous, so this
  is association, not the cost of splitting an order.
- **The §1.27 program score predicts cross-day linkage**, a criterion it was not fit on: linked minus mirror
  +0.036 (t 2.89) in DEV, **+0.057 (t 7.10)** in TEST.
- **Nulls, closed:** hidden-liquidity share does not predict permanence (d_close t −0.02, 1.28 M bursts; its
  effect on D_b/peak, t −25, is mechanical); odd-lot bursts track neither retail (t −0.13) nor institutional
  (t 0.08) imbalance; programme intensity does not forecast next-day volatility (t −0.48); no quarterly
  recurrence in the linkage profile.

### 1.36 The 279 burst model rebuilt leak-free: forecasting, oracle and trading — burst-forecasting, 2026-09-24

Code and full tables: `studies/burst_forecasting/` (`docs/FORECASTING_TEST_LIST.md`, `docs/BURST279_RESULTS.md`).
Trade bursts, p4-revisit-v1 samples. Models trained on DEV 2017–19 (group-0 names), scored on VAL 2020–21 and
TEST 2022–25 (groups 1–2: disjoint names and years). Decision time T_dec = max(t_b + 600 s, t_e + 10 s).
Features known by T_dec only; `link_same` / `link_opp` were found to count later bursts and are excluded.

- **Permanence is predictable only where it overlaps the observed move.** AUC for "at least half of the peak
  impact standing at the close, measured from the burst start" 0.64 (TEST), 0.59 at the next open and 0.55 at
  the next close. For the tradable part, the move after T_dec, AUC is 0.511–0.515.
- **The setup is coherent (oracle).** A perfect model trading every burst at T_dec would earn +65 to +79 bps per
  trade net at the close (86–94% winners). Noisy oracles break even at AUC ≈ 0.53.
- **Decision-to-close is forecastable, but not by bursts.** Market-excess (equal-weight intraday index) daily rank
  IC from non-burst predictors alone — the move since the open, the 30-minute pre-move, time of day and spread —
  is +0.038 (t 8.0) in VAL and **+0.040 (t 14.5)** in TEST; the gross mid-to-mid top-minus-bottom decile is
  +12 to +13 bps. Adding the burst's impact path and every burst-structure feature (fingerprint, program score,
  linkage, multi-day, hidden and truncated shares) changes the IC by +0.0002 (t 0.5) and −0.0005 (t −1.0).
- **Next-day close-to-close is not forecastable** from controls, total order-flow imbalance or burst flow, raw
  or market-excess: |IC| < 0.005 in TEST.
- **Trading does not survive costs.** Across 144 trade-burst and 120 submission-burst cells (12 definitions × 3
  exits × 4 rules), fewer cells beat net t > 2 than chance predicts (2 and 0 per period against ≈ 3.3 and ≈ 2.7);
  the single cell passing in both periods (fading multi-day-program bursts to the close, +5.0 / +2.9 bps,
  t 2.02 / 2.08) is what that many correlated tests produce. Once-per-day books (tCLOSE, CLOP, CLCL) lose after
  costs in both periods. Four alternative definitions decided in real time (merged 5-min and 30-min runs,
  fingerprint chains, run60): 0 of 32 cells pass.

## 2. Excluded — do not reintroduce

| Result | Why it is invalid |
|---|---|
| p4-revisit-v1 outputs before amendment A5 (`analysis/*_preA5.json`) | The CRSP cumulative-price-factor ratio was inverted in `p4_aggregate.crsp_frame`, so a split day read as a ~−99% gap. Affects `d_open`, `d_cc`, `phi_open`, `phi_cc` and everything built on them, including a −334 bps/day CLOP decile book. `d_close` and the Phase II features are unaffected and bit-identical. |
| p4-revisit-v1 DEV outputs before A3 and A4 (`*_preA3.json`, `*_preA4.json`) | A3: early-close sessions leave stub quotes, corrupting late mids (−18 to −22 bps class averages, a +60 bps tCLOSE book). A4: the pseudo-burst placebo used real bursts decided after the signal clock and windows preceding their own real burst (placebo tCLOSE t 27.7). |
| p4-revisit-v1 Q2(b)/(c) and (e) cited *without* a same-quarter return control (e.g. "informative flow loads on 13F +0.067, t 4.20" or "informative flow is less retail-aligned" as evidence of institutions) | The informativeness filter selects on price moving the burst's way, so quarterly informative flow tracks the quarter's own return, and ownership changes, flow-induced trading and retail flow all co-move with contemporaneous returns. Use the return-controlled numbers in §1.31. |
| Any P4 result computed with the legacy C++ burst detector | It signs every hidden (type-5) execution as a sell, which is 20–28% of executed volume. See §1.31 Q0. |
| `hid_tp_v1`, array 14225689 | Read the BBO *at* the print, making "outside the quote" endogenous — the print can move the quote it is compared against. Superseded by 14227671. |
| Sweep markout measured **from t** (+3.271 swept vs +0.464 unswept) | Circular: conditions on a quote move inside the first second while the footprint is ~80% impounded in that second. The 18× split vanishes when the windows are made disjoint (§1.8). |
| Pre-trade depletion ratio table (Q1–Q5, d<0.10, d≥1.0) | Conditions on hidden print size over displayed depth, but a type-5 print resting inside the spread consumes no displayed queue. Wrong conditioning variable; superseded by the sweep tests. |
| Pooled "all aggressive" column of the event-time table (−1.541, −1.715, −1.783, +1.159) | Sign assigned against the *contemporaneous* midpoint, so a just-fallen mid mechanically labels a print a buy. Only the outside-the-quote column is usable. |
| Original Hasbrouck VAR, "permanent fraction 1.01" | VAR(12) on a 10s clock carries 120s of memory; the 3- and 10-minute responses were the model's asymptote, not estimates. It also dropped non-stationary name-days, selecting on the parameter under study. |
| Non-overlapping multi-day sort (−8.43 bps at 20 days) | Phase-dependent: sweeping the starting offset across the same data spans −36.2 to +25.5 bps. Superseded by §1.15. |
| Walk-forward reversal Sharpe (+0.79, +1.47, per-year figures) | Calibration window overlapped the evaluation window in calendar time, on an ex-post universe. Superseded by §1.17. |
| Closing-auction imbalance (+9.9 to +20.1 bps) | Imbalance and price move computed over the *same* window — contemporaneous, not predictive. |
| Volume-profile IC (−1.000) | Volume-so-far and rest-of-day volume are mechanically complementary. |
| Single-day "iceberg" signal (+6.92 at 30 min) | One ticker-day. Across 499 days it is −0.197. |
| "N of N names show a positive increment" as evidence | R² is mechanically non-decreasing when a regressor is added. Check the t-statistic. |
| `bq2` same-side Hurst rankings, array 14575613 | Silence/run/Hawkes definitions used packet sign changes to choose block boundaries, then tested block-sign memory against rotated signs. The favorable ranking is mechanically sign-conditioned; sign-blind volume clocks were near zero or negative. Superseded by timing-only `bq3`. |
| Legacy simulation join training and campaign validation | `join_feature_frame` pooled simulated days on an intraday clock; parent IDs restart each day. Cross-day candidates and labels contaminate the join fit. Its failure cannot establish that anonymous reconstruction is exhausted. The separately trained fragment classifier and strict continuation result are not affected by this join-code defect. |
| Initial two-avenue post-completion reversal, array 14575625 | The join model linked only about 2% of fragments, but singleton fragments were still treated as completed campaigns. Decisions were also not sorted chronologically before non-overlap. This is a fade-every-fragment diagnostic, not a campaign-completion test. |
| Hurst reduction as a parent-recovery validator | Sign-blind `bq3` removes the circularity but fails known-parent simulation: definitions with larger positive dH can have materially lower block purity than definitions with zero or negative dH. Hurst may characterize sign flow, but it does not rank parent recovery. |
| "Message counting inflated the hidden headline ~3.5×" (+2.09 → +0.603) | Confounded: the two designs also differ in burst formation and in measuring from termination vs a one-second buffer. The closest like-for-like comparison (per print, from t+1s) is +0.530 vs +0.603. Retracted from the header note, 2026-09-13. |
| Burst-information-v1 "primary test failed" (held-out third-packet 300s flow MSE −0.155%, t = −4.30) | Unnormalized packet-count targets: AMZN and AMD supply 94% of held-out baseline MSE, and no training name has their activity. Tests tree extrapolation to two names, not burst information. See §1.24. |
| Burst-information-v1 return and wait-cost contrasts (e.g. +4.009% vs tree, 60s held-out) and "burst features improve price forecasts" | Every return and wait-cost model is worse than a zero forecast (90 of 90 cells). An MSE improvement between forecasts that all lose to zero is not predictive content. See §1.24. |
| Corrected synthetic join model (Brier 0.0665, precision 92.6%, recall 91.6%) as evidence that reconstruction works | Equivalent to a "gap < 30s" rule (92.7% / 88.4%) in a simulator with ~45 parents a day and book features that carry no parent information (AUC ~0.50). Uninformative about real reconstruction. See §1.25. |
| Fingerprint E3 as first computed (v2 recurrence control, "nearest different-size packet to t_j") | The control's lag to i differs from the matched lag by a random jitter, and state differences grow with lag. On a synthetic tape with no programs it reports 0.57 (0–2 s) and 0.62 (2–10 s, thin name). Replaced by the lag-binned statistic in §1.27. |
| Same-side short-lag size matches against an unmatched cross-day or same-day null | Untruncated sizes are censored by displayed depth, so pairs sharing book state match by chance: a synthetic depth-regime tape with no programs gives ratios above 1.5. Use the depth-matched null (§1.27). |
| LOBSTER column 7 as a participant label for executions | It is NASDAQ MPID attribution and sits on non-marketable market-maker quotes. At most a few hundred attributed executed orders a day in AAPL, and none in KEY or CRWD. |

---

## 3. Provenance

Per-ticker outputs live at `/u/scratch/n/nicjia/order-burst-analysis/results/<group>/out/`,
concatenated to `all.csv` in each. Extractors and aggregators are in `src_py/`, with
checksums and the array-to-table map in `RESULTS_PROVENANCE.md`. `/u/scratch` is not durable
— the cluster git repo (from commit `bc73ebd`) and the local clone are the only safe copies.
