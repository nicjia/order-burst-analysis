# Verified Results

The only numbers in this project that come from a design with no known flaw. Anything not
listed here must not appear in `paper.tex` or `main.tex`. Section 2 lists what was excluded
and why, so a discarded number is never silently reintroduced.

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

### 1.5 Spread-scaling law — arrays 14368098, 14368227, 14368630
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

---

## 2. Excluded — do not reintroduce

| Result | Why it is invalid |
|---|---|
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

---

## 3. Provenance

Per-ticker outputs live at `/u/scratch/n/nicjia/order-burst-analysis/results/<group>/out/`,
concatenated to `all.csv` in each. Extractors and aggregators are in `src_py/`, with
checksums and the array-to-table map in `RESULTS_PROVENANCE.md`. `/u/scratch` is not durable
— the cluster git repo (from commit `bc73ebd`) and the local clone are the only safe copies.
