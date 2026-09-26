# Fingerprint-v1: validating burst definitions on real data without participant IDs

> **Completed 2026-09-13.** Results: `studies/fingerprint/BURST_FINGERPRINT_RESULTS.md`. Gates: C1, P1, P2 pass; C2, C3 fail.

Written 2026-09-13 before any exploration-panel output was read. Goal set by the user: a
well-defined burst that separates flow from one execution program from unrelated flow, allowing
some noise. Alpha is not required.

## Why a fingerprint

LOBSTER cannot label the aggressor of an execution. Probes of KEY, CRWD and AAPL on 2019–2025
dates found:

- **Column 7 is NASDAQ MPID attribution**, never read by any earlier extractor. It sits almost
  only on non-marketable market-maker quotes: at most a few hundred attributed executed orders a
  day in AAPL, essentially all UBSS, and none in KEY or CRWD.
- **Type-5 hidden executions** carry order id 0 and direction +1, with no MPID.
- **Replaces are linkable.** An ITCH replace appears as a delete plus an add at the same
  timestamp, same side, same MPID (10,027 in one KEY day; 96% keep the size). That gives certain
  same-trader chains for *resting* orders only.
- **Aggressor-to-order links are rare.** A delete followed by an execution at the same timestamp
  occurs at 0.2–0.3% of execution timestamps. Leftover posts and hidden-size refills never share
  the timestamp. Tolerance-based matches are confounded with market makers re-quoting after a
  sweep.
- **Coverage gaps.** 2024 has twelve partially downloaded dates (546–555 tickers instead of
  ~2,100): 0301, 0403, 0422, 0605, 0612, 0719, 0730, 0802, 0805, 0906, 1111, 1223. 2021 has none.

The testable signature used instead: when an algorithm slices a parent into identical child
sizes, **same-side packets with the same untruncated size recur at short lags more often than
chance**. Untruncated means the packet neither exhausted displayed touch depth nor walked the
book, so its size is the incoming order's size, not a property of the book. Identical sizes are a
lower bound on common origin, because many algorithms randomize child sizes.

## Measurement (`src_py/fingerprint_packets.py`, `fingerprint_stats.py`)

- **Packets.** Canonical packets come from `execution_packets.py`, with type-5 Direction ignored.
- **Size classes:**
  - untruncated non-round (`u_nonround`, primary);
  - untruncated round lot (`u_round`);
  - truncated (sizes set by the book, so they fingerprint the *passive* side);
  - rare non-round (`u_rare`): sizes with share ≤ 0.2% of untruncated non-round packets on the
    stock's *other* sampled days, excluding both days of the pair;
  - visible non-round (`u_visible`, sensitivity): untruncated non-round packets with no hidden
    execution. Added before any output was read, because an IOC filled only by hidden liquidity
    inside the spread can pass the untruncated test while its size is set by the hidden resting
    order.
- **Pair counts.** Same-side pairs and identical-size matches, by lag bin up to 1,800 s, within
  each day. Opposite-side pairs are a control.
- **Nulls.**
  - *Primary: depth-matched cross-day.* The same statistic between a pair of adjacent trading
    days at the same clock lag, counting only pairs whose packets share an executed-side depth
    quartile (thresholds pooled over the date pair). It is pooled over lags per date pair and
    class. It removes the depth-censoring artifact below, keeps intraday seasonality, and cannot
    contain a same-day pair.
  - *Sensitivity: cross-day unmatched.* Same, without depth matching; exposed to censoring.
  - *Lower bound: within-day long lag.* The same day's match rate at 900–1,800 s. Immune to
    day-to-day shifts but absorbs programs lasting over 15 minutes.
- **Burst definitions:** 3 rules × 11 gaps (0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300 s) ×
  minimum 2 or 3 packets.
  - `run`: same-side runs broken by a sign change or an unsigned packet; the project convention.
  - `stream`: each side's own sequence; other-side packets ignored.
  - `timing`: sign-blind.
- **Recurrence.** Each packet is paired with the next same-size packet, against a control: the
  packet nearest in time whose size is similar but not equal (|s′ − s| ≤ max(1, 0.2 s)), so both
  pairs face the same depth censoring. Recorded:
  - absolute differences in pre-trade spread, log executed-side depth, and signed imbalance;
  - a 0.05 s histogram of recurrence lags.

## Pre-declared quantities (exploration: 2024, 174 names, 10 adjacent-date pairs)

Names are split by `sha256("fingerprint-v1|" + ticker) mod 3`: 0 is exploration (174), 1 or 2 is
confirmation (300). Aggregation in `src_py/aggregate_fingerprint.py`; 95% intervals bootstrap names.

**E1 existence.**
- Primary: `u_nonround` same-side ratio of observed to expected matches under the cross-day null,
  at lags [0.5, 2) and [2, 10) s.
- Also reported: the within-day long-lag null, the opposite-side control, and the `u_rare` and
  `truncated` classes.

**E2 definition quality, lags below 300 s.** Excess X = matches − expected, clipped at zero per
lag bin.
- **TPR** = Σ_b X_within,b / Σ_b X_all,b: the share of same-origin evidence a definition keeps
  inside one burst.
- **FPR** = Σ_b P_within,b / Σ_b P_all,b: the share of all same-side pairs it puts inside one
  burst. Same-origin pairs are a small share of all pairs, so this approximates the false
  positive rate.
- **Youden J** = TPR − FPR. It is scale-free: it needs no estimate of how often children of one
  parent share a size.
- Also reported: enrichment = Σ_b X_within,b / Σ_b P_within,b · (X_all,b / P_all,b), i.e. excess
  inside bursts relative to random same-side pairs at the same lag, plus ROC area per rule across
  gaps.
- **Selection rule, fixed before exploration output was read:**
  - Class `u_nonround`, minimum 3 packets, within-day long-lag null.
  - Choose the rule and gap maximizing J.
  - Report the full ranking and the same ranking under the cross-day null.
  - *Revision note:* the first draft maximized recall × enrichment. The 3-name pilot showed that
    score is degenerate, because a definition putting every pair in one burst scores exactly 1.
    It was replaced by J before any exploration output was aggregated or read.

**E3 state similarity.** Ratio of matched to control mean absolute difference, at recurrence lags
[2, 10), [10, 30) and [30, 120) s, for spread, depth and imbalance. Below 1 means same-size
recurrences happen in more similar book states than lag-matched different-size packets. This
tests the hypothesis that one algorithm acts under similar market states.

**E4 periodicity.** Descriptive only: recurrence-lag peaks at integer seconds against
half-second controls.

## Confirmation (2021, the 300 other names, 10 adjacent-date pairs; applied once, unchanged)

- **C1.** E1 replicates: `u_nonround` same-side cross-day ratio has a lower 95% bound above 1 at
  both [0.5, 2) and [2, 10) s.
- **C2.** E3 replicates for spread and depth: ratio upper 95% bound below 1 at [2, 10) and
  [10, 30) s.
- **C3.** The exploration-selected definition ranks in the 2021 top five by J under the same
  class, null and minimum size. The Spearman rank correlation of J across all 33 `u_nonround`
  minimum-3 rule × gap definitions, between the two periods, exceeds 0.8.

Every gate is reported pass or fail. No parameter, class, null, lag bin or gap grid may change
after exploration output is read. 2021 is a different market regime (pre-2022, heavy retail);
agreement across regimes is part of what is being tested.


## Amendments made before any exploration output was read (2026-09-13)

A code review and synthetic tests found two problems with the first frozen draft; both are fixed.
The first 174-name stage-2 statistics, computed under the draft, were never aggregated or read and
are recomputed from cached packets.

1. **Depth censoring.** A packet counts as untruncated only if its size is below displayed depth,
   and local size distributions move with the book. Packets close in time share book state, so
   they share sizes more often with no common origin. In `tests/test_fingerprint.py` a synthetic
   tape with depth regimes and no programs gives a ratio above 1.5 against the unmatched
   cross-day null and the within-day null, and 1.0 ± 0.1 against the depth-matched null. A
   planted program keeps a ratio above 1.5 against the depth-matched null.
   - A stratified size-shuffle null was tried first and rejected: it absorbs any program that is a
     large share of its time window, which is exactly the signal.
   - The size-similar recurrence control replaces the "any different size" control for the same
     reason: identical sizes under censoring mechanically imply similar depths.
2. **Pooling by chance-match counts.** Pooled ratios and pooled TPR are dominated by names whose
   size distributions concentrate on a few values. In a 3-name synthetic group, one censoring
   name supplied 94% of expected matches and hid two names with planted programs (per-name
   ratios 1.36 and 1.48). Primary statistics are therefore equal-weight across names:
   - E1 and E3: the median of per-name ratios, over names with at least 20 expected matches (E1)
     or 30 matched and 30 control recurrences (E3), with a bootstrap over names;
   - E2: the mean of per-name J.
   Pooled versions are reported alongside.

Revised primary quantities and gates (these replace the E1–E3 and C1–C3 wording above where
they differ):

- **E1 / C1.** `u_nonround`, depth-matched cross-day null. The name-median ratio at [0.5, 2) and
  [2, 10) s has a lower 95% bound above 1 in exploration, and again in 2021.
- **E2 / selection.** `u_nonround`, minimum 3 packets, depth-matched null. Choose the rule and
  gap with the highest mean per-name J over lags below 300 s (names with ≥ 20 expected and ≥ 10
  excess matches).
- **C3.** The selected definition ranks in the 2021 top five by the same statistic. The Spearman
  correlation of mean per-name J across the 33 rule × gap definitions between periods exceeds
  0.8.
- **E3 / C2.** Size-similar control. The name-median matched/control ratio for spread and for
  log depth has an upper 95% bound below 1 at [2, 10) and [10, 30) s, in exploration and again
  in 2021.

## Amendment after a partial exploration read (2026-09-13, evening)

The stage-2 pipeline was checked on the first 57 exploration names, and E3 was seen. The v2 E3
control is biased toward "similar states" at short lags. Its lag to packet i differs from the
matched pair's by a random jitter, and book-state differences grow with lag. On a synthetic tape
with no programs and no state dependence, the v2 control gives matched/control ratios of 0.57
(0–2 s) and 0.955 (2–10 s) in a dense name, and 0.47 / 0.62 / 0.90 (0–2, 2–10, 10–30 s) in a
thin name. The E3 numbers from that partial read are therefore void, and E3/C2 are replaced by:

- **E3 (corrected), `src_py/fingerprint_state.py`.**
  - For each untruncated non-round packet i: j = next same-side packet with the identical size;
    k = next same-side packet with a similar, different size.
  - Both pairs start at i, and differences are accumulated in 40 log-spaced lag bins (0.5–600 s).
  - Within a lag range, control means are reweighted to the matched pairs' bin counts; bins need
    ≥ 20 control pairs, and names ≥ 30 matched pairs.
  - The statistic is the median of per-name ratios, bootstrapped over names
    (`aggregate_fingerprint_state.py`).
  - Synthetic check: ratio within 5% of 1 at 2–10 s with no programs
    (`tests/test_fingerprint.py::StateSimilarityJitterTests`).
- **C2 (replaces the earlier C2).** Corrected E3 name-median ratio for spread and for log depth
  has an upper 95% bound below 1 at 2–10 s and 10–30 s, in exploration and again in 2021.

The corrected statistic had not been computed or read when this amendment was written. The 2021
confirmation outputs of every stage remain unread. E1 and E2 from the same partial read are
unchanged in definition; the full-panel aggregation replaces the partial numbers.

## Stage 3, pre-declared before any result: a program-likeness score (runs only if E1 passes)

- **Burst rows.** For the definition selected in E2, `src_py/fingerprint_burst_rows.py` writes one
  row per burst (≥ 3 same-side packets). Evidence: same-side, same-depth-quartile untruncated
  non-round pairs within 300 s, their identical-size repeats, and the expectation from
  depth-quartile-specific cross-day rates. Features exclude child sizes:
  - timing: packet count, duration, intensity, inter-arrival coefficient of variation and median;
  - execution: truncation share, hidden share, opposite-side and unsigned interleaving;
  - pre-burst book at the first packet: spread, log depth, signed imbalance;
  - context: time of day, trailing 300 s activity.
- **Model.** `src_py/program_score.py` fits a binomial likelihood with success probability
  q + (1 − q)·σ(w·x), with ridge 1 and equal weight per name, on the 2024 exploration names only.
- **Evaluation.** Once, on the 2021 confirmation names.
  - **P1.** Excess repeats per pair in the top score decile minus the bottom decile has a
    name-bootstrap 95% lower bound above 0.
  - **P2.** Spearman correlation between decile and excess per pair exceeds 0.7.
  - Standardized coefficients are reported for interpretation; the pre-burst book-state
    coefficients bear directly on the "algorithms act in similar states" hypothesis.
- **Scope.** The score is a ranking of program-likeness. Its absolute level is not a probability
  of institutional origin.

## What this cannot establish

- Identical sizes are neither necessary nor sufficient for a common parent. Two traders can share
  an algorithm vendor's size choices. The excess over chance is a population lower bound on
  same-origin structure, not a per-burst label.
- No result here identifies institutions versus retail. Most US retail marketable flow is
  internalized off-exchange and never reaches NASDAQ's book. Separating retail needs off-exchange
  data (for example sub-penny TRF prints à la Boehmer, Jones, Zhang & Zhang) or broker records.
- A definition that keeps fingerprints together is well-defined against observable evidence. It
  is not proven to recover whole parents.
