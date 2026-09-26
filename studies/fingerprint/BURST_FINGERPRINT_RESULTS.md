# Fingerprint-v1 results: which burst definitions keep one program's orders together

Completed 2026-09-13. Design, amendments and freezes: `studies/fingerprint/BURST_FINGERPRINT_DESIGN.md` and
`results/fingerprint_v1/freeze*.json`. Exploration: 2024, 174 names, 3,480 name-days, 18.5M
signed packets. Confirmation: 2021, 291 disjoint names (of 300; 9 absent), 5,702 name-days, 36.1M
signed packets, read once after every exploration decision was fixed.

## Bottom line

1. **The tape contains a measurable same-origin signature.** Same-side aggressive orders with an
   identical untruncated, non-round size recur about twice as often as chance within ten seconds.
   The chance rate comes from an adjacent day at the same clock time and book depth. The pattern
   holds in 97–99% of names in both years, while round-lot sizes show no excess.
2. **Simple burst definitions keep about half of that evidence together while grouping a
   minority of unrelated pairs.** Two families tie:
   - an uninterrupted run of same-side packets (any gap cap of 30–300 s);
   - a same-side stream with a 2–5 s gap.

   The pre-declared single winner (run, 300 s) did not keep its rank in 2021 (7th of 33, within
   ±0.016 of first). The ordering of all 33 definitions replicated (Spearman 0.95).
3. **A burst's timing predicts how program-like it is, out of sample.** A score fit on 2024 names
   ranks 2021 bursts from 51 to 241 excess identical-size repeats per 1,000 pairs, bottom to top
   decile. The score uses no child sizes, and timing geometry alone carries most of the lift.
4. **Same-origin children do not wait for the same book.** They trade at more similar spreads
   than size-similar controls at the same lag, but at *less* similar displayed depth, in both
   years, consistent with a program depleting the queue it trades against.

This is a real-data answer to "how well does a burst definition keep one program's orders
together, and how much unrelated flow does it merge in?", stated relative to an observable
fingerprint. It does not separate retail from institutional flow, and it does not give an
absolute purity. See the limits section.

## Pre-declared gates

| gate | test (2021 unless noted) | result |
|---|---|---|
| E1 (2024) | name-median observed/chance, `u_nonround`, depth-matched null, 0.5–2 s and 2–10 s | 2.16 [2.01, 2.41]; 2.05 [1.81, 2.29] |
| **C1** | same, lower 95% bound > 1 | **PASS**: 2.55 [2.33, 2.74]; 2.04 [1.95, 2.18] |
| **C2** | corrected state similarity: spread and depth ratio upper bound < 1 at 2–10 s and 10–30 s | **FAIL**: spread 0.92 [0.90, 0.94] and 0.95 [0.93, 0.96] pass; depth 1.06 [1.04, 1.07] and 1.10 [1.08, 1.11] go the other way |
| **C3** | selected definition (run, 300 s) in the 2021 top 5 **and** Spearman of mean per-name J across 33 definitions > 0.8 | **FAIL**: rank 7 (J 0.257 vs leader 0.263, CIs overlap); Spearman 0.954 passes |
| **P1** | program score, top minus bottom decile excess per 1,000 pairs, lower bound > 0 | **PASS**: +189 [150, 232] |
| **P2** | Spearman(decile, excess) > 0.7 | **PASS**: 1.00 |

## 1. The fingerprint exists and replicates

Observed ÷ chance identical-size matches for same-side untruncated non-round packets. Chance is
the depth-matched cross-day rate, and entries are medians of per-name ratios with 95% bootstrap
intervals over names.

| class | lag | 2024 | names > 1 | 2021 | names > 1 |
|---|---|---|---:|---|---:|
| non-round (primary) | 0.5–2 s | 2.16 [2.01, 2.41] | 98% | 2.55 [2.33, 2.74] | 99% |
| non-round (primary) | 2–10 s | 2.05 [1.81, 2.29] | 97% | 2.04 [1.95, 2.18] | 99% |
| visible only (no hidden fill) | 2–10 s | 2.05 [1.81, 2.29] | 97% | 2.04 [1.95, 2.19] | 100% |
| rare sizes | 2–10 s | 3.56 [2.06, 9.23] (11 names) | 91% | 2.33 [1.73, 2.91] (22 names) | 95% |
| round lots (control) | 2–10 s | 1.04 [1.02, 1.07] | 61% | 1.00 [0.98, 1.01] | 50% |
| truncated (passive side) | 2–10 s | 1.59 [1.51, 1.70] | 99% | 1.40 [1.36, 1.43] | 99% |

![excess by lag, 2024](../../figures/fig_fingerprint_curve_2024.pdf)

- **Decay with lag.** The excess is largest below a second (about 3.7–4.7×), falls to about 2×
  at 2–10 s and 1.7–1.9× at 10–30 s, and settles near 1.2× (2024) to 1.3× (2021) by 30 minutes.
  The long plateau is the part the same-day null absorbs: programs lasting tens of minutes, or
  day-level shifts. Against that null the short-lag excess is 1.74–1.80×, still far above 1.
- **Controls behave as they should.**
  - Round lots, the sizes everyone uses, carry no fingerprint.
  - Excluding packets with any hidden fill changes nothing.
  - Matching local activity terciles as well as depth (exploration, 173 names) gives 2.24× and
    2.10× and leaves the definition ranking unchanged (Spearman 0.993).
- **Passive side.** Truncated packets, whose sizes are set by the book, also repeat at short
  lags: the same displayed size is hit repeatedly, a passive-side fingerprint.
- **Clock timing.** Same-side identical-size recurrence lags spike at whole seconds, against
  half-second controls of 0.79–0.98:

  | lag | 2024 | 2021 |
  |---|---:|---:|
  | 1 s | 1.66× | 1.24× |
  | 10 s | 1.57× | 1.45× |
  | 60 s | 1.96× | 1.21× |

  Opposite-side same-size recurrences show little of this. The fingerprint is consistent with
  wall-clock-scheduled algorithms.

## 2. Which definition keeps same-origin pairs together

Scale-free ROC over lags below 300 s:
- TPR is the share of excess identical-size pairs that fall inside one burst;
- FPR is the share of all same-side depth-matched pairs that do;
- J = TPR − FPR, averaged over names, minimum 3 packets, depth-matched null.

| definition | 2024 J | 2021 J | 2021 TPR | 2021 FPR | 2021 enrichment |
|---|---|---|---:|---:|---:|
| run, 60 s | 0.360 [0.329, 0.391] (rank 3) | 0.258 [0.243, 0.275] (rank 4) | 0.41 | 0.15 | 1.10 |
| run, 300 s (pre-declared) | 0.361 [0.330, 0.393] (rank 1) | 0.257 [0.241, 0.273] (rank 7) | 0.41 | 0.15 | 1.11 |
| stream, 5 s | 0.347 [0.318, 0.376] (rank 6) | 0.263 [0.247, 0.279] (rank 1) | 0.45 | 0.19 | 1.29 |
| stream, 2 s | 0.343 [0.311, 0.376] (rank 7) | 0.261 [0.245, 0.277] (rank 3) | 0.40 | 0.14 | 1.25 |
| timing, 2 s | 0.332 [0.301, 0.361] (rank 12) | 0.262 [0.245, 0.279] (rank 2) | 0.42 | 0.15 | 1.17 |
| stream, 60 s | 0.056 | 0.051 | 0.91 | 0.86 | — |

![ROC, 2021](../../figures/fig_fingerprint_roc_2021.pdf)

- **Most of the separation comes from a few-second time scale, and extra conditions add little.**
  For the run rule, the time cap barely matters beyond 30 s. The run is ended by opposite-side or
  unsigned flow long before the cap binds, so "uninterrupted same-side sequence" is itself the
  definition.
- **Long gaps destroy the definition.** A 60 s side-only stream groups 85% of all pairs and scores
  J ≈ 0.05.
- **J is lower in 2021** (0.26 vs 0.36) because more of 2021's fingerprint evidence falls between
  bursts, spread over longer lags.

**Recommended working definition.** The pre-declared rule did not keep its rank, so this is a
recommendation, not a confirmed winner. The recommendation is **a run of ≥ 3 consecutive
same-side aggressive economic packets, ended by any opposite-side or unsigned execution, with a
60 s gap cap**. It ranks top five in both years and is the simplest member of the tied group.
The side-only 2–5 s stream is an equally supported alternative when interleaved opposite trades
should not break a burst. The tied family keeps 40–57% of same-origin evidence together while
grouping 13–22% of same-side pairs; the recommended run keeps 41–49% while grouping 13–15%.

## 3. Book state: similar spread, different depth (corrected E3)

Identical-size recurrence j versus a size-similar different-size recurrence k from the same
packet i, compared within 40 log-lag bins (control reweighted to the matched lag distribution).
Median of per-name matched ÷ control mean absolute difference:

| lag | spread 2024 | spread 2021 | log depth 2024 | log depth 2021 | imbalance 2024 | imbalance 2021 |
|---|---|---|---|---|---|---|
| 2–10 s | 0.90 [0.86, 0.92] | 0.92 [0.90, 0.94] | 1.05 [1.03, 1.08] | 1.06 [1.04, 1.07] | 0.96 | 0.99 |
| 10–30 s | 0.95 [0.94, 0.97] | 0.95 [0.93, 0.96] | 1.14 [1.11, 1.15] | 1.10 [1.08, 1.11] | 1.01 | 1.01 |
| 30–120 s | 0.94 | 0.95 | 1.18 | 1.10 | 1.04 | 1.02 |

**Reading.**
- Children of one program trade when the spread is similar.
- They do *not* return to similar displayed depth: depth differs more than for unrelated
  size-similar orders, consistent with self-depletion.
- The simple "algorithms act in the same book state" hypothesis is half supported (spread) and
  half contradicted (depth), in both regimes.
- The v2 version of this statistic was biased toward "similar" at short lags by lag jitter in its
  control. That was demonstrated synthetically, and it was replaced before being read.

## 4. A program-likeness score for bursts

For every run-300 s burst: identical-size repeats among its same-side, same-depth-quartile
untruncated non-round pairs, against depth-specific chance.
- **Model.** Binomial likelihood with success probability q + (1 − q)·σ(w·x), ridge, equal weight
  per name, fit on 2024 (838k bursts with at least one testable pair, of 2.43M).
- **Test.** Scored once on 2021 (1.56M testable bursts, of 4.86M).

| 2021 decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| excess per 1,000 pairs | 51 | 59 | 63 | 68 | 74 | 81 | 90 | 101 | 120 | 241 |
| observed ÷ chance | 2.6 | 2.8 | 2.9 | 3.0 | 3.2 | 3.3 | 3.5 | 3.9 | 4.3 | 7.1 |

Post-hoc sensitivities, all fit on 2024 and scored on 2021 (not gates):

| features | top − bottom [95% CI] | names with top > bottom |
|---|---|---:|
| all pre-declared | +189 [146, 233] | 96% |
| without truncation share (it depends on size relative to depth) | +145 [99, 196] | 85% |
| also without spread, depth, imbalance | +132 [89, 183] | 85% |
| timing geometry only (packets, duration, intensity, inter-arrival) | +132 [93, 180] | 83% |
| time of day and activity only | +17 [−49, 70] | 47% |

**Reading.** Program-likeness is carried by burst geometry, not by market context. Without the
truncation share:
- more packets and shorter duration raise the score, and so does a longer median inter-arrival
  (regular pacing without pauses);
- hidden fills lower it.

The score ranks bursts; it is not a probability of institutional origin.

## What this does and does not establish

- **A lower bound, not a label.** Identical child sizes are a lower bound on common origin,
  because many algorithms randomize sizes. Two unrelated traders can share a vendor's size
  choice. Every "purity" statement is relative: J, enrichment and decile lift rank definitions
  and bursts, but no absolute share of noise inside a burst is identified.
- **Not retail versus institutional.** Most US retail marketable orders are internalized
  off-exchange and never reach this book, and LOBSTER carries no aggressor identity. The seventh
  column is MPID attribution on market-maker quotes (see the design document). Separating retail
  needs TRF/TAQ data (sub-penny signing, Boehmer, Jones, Zhang & Zhang 2021) or broker records.
  On NASDAQ lit data the defensible categories are program-like flow and everything else.
- **Two regimes, not a guarantee.** 2021 and 2024 differ (the 2021 retail boom; lower J in 2021).
  The fingerprint replicates in both, while the exact best gap does not.
- **No alpha claim.** See `VERIFIED_RESULTS.md` §1.26 on why detectable programs are already priced.

## Reproduction and provenance

- **Code.**
  - Packets and statistics: `src_py/fingerprint_packets.py`, `fingerprint_stats.py` (v2).
  - Aggregation: `aggregate_fingerprint.py`, `fingerprint_state.py` + `aggregate_fingerprint_state.py`.
  - Burst rows and score: `fingerprint_burst_rows.py`, `collect_burst_rows.py`, `program_score.py`.
  - Gates: `evaluate_fingerprint_gates.py`.
  - Figures and tables: `report_fingerprint.py`.
  - Tests: `tests/test_fingerprint.py`, `tests/test_program_score.py`.
- **Cluster.** Drivers `hoffman2/fingerprint*.sh`, `collect_burst_rows.sh`; arrays listed in
  `RESULTS_PROVENANCE.md`.
- **Outputs.** `results/fingerprint_v1/{explore_2024,confirm_2021}/summary.json`,
  `state_summary.json`; `results/fingerprint_v1/gates.json`, `program_score.json`,
  `program_score_sensitivity.json`, `explore_2024/summary_activity_matched.json`.
- **Figures.** `figures/fig_fingerprint_*.pdf`. PNG and PDF are gitignored, so force-add them when
  committing.
