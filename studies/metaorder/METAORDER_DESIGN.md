# Metaorder-v1: child clusters, parent linkage, calibration, and a passive-side test

Pre-registered 2026-09-14, before any statistic below was computed or read. The user approved the
plan on the same day, after `studies/program_evidence/PROGRAM_EVIDENCE_RESULTS.md`. Results go to `studies/metaorder/METAORDER_RESULTS.md`;
code hashes to `results/metaorder_v1/freeze*.json`.

## Why

Program-evidence-v1 showed that fingerprint bursts are real, mostly directional algorithm activity.
It also showed that a burst is a **fragment**:
- the same program keeps trading after an identical-size chain ends;
- same-side excess outlives any burst;
- linked chains of 4–5 children show no measurable impact physics.

Two further facts shape the plan:
- The timing and size fingerprints disagree about the best boundary.
- The only market fact with a plausible trading use is on the liquidity-provision side: providers
  earn more against program-like flow, +0.8 bps within equal truncation strata.

The plan therefore splits the problem:
1. **Child clusters.** Fix one timing-based rule by a joint criterion and confirm it on an unused
   year.
2. **Parents.** Do not draw boundaries. Score each burst's probability of belonging to a longer
   same-side program, from fingerprint links, and calibrate the score on injected synthetic
   parents.
3. **Trading.** A real-time program score, then a queue-aware simulation of passive orders
   posted against program-like bursts.

Samples follow the project convention:
- exploration: 2024 fingerprint-v1 names (174);
- confirmation: 2021 names (291), read once;
- unused year for the definition gate: 2019 (the 474 fingerprint tickers present that year).

## M1 — the child-cluster rule, confirmed on 2019

**Candidates.** Side-only stream, 5 s gap, ≥ 3 packets (`stream5`), against the incumbent run with
a 60 s cap, ≥ 3 packets (`run60`).

**Statistic.** Combined J = ½ (size J + phase J), averaged over names eligible for both:
- *Size J* is fingerprint-v1's: mean per-name Youden J over lags below 300 s, depth-matched cross-day
  null, `u_nonround`, excess clipped at zero per lag bin, names with ≥ 20 expected and ≥ 10 excess
  matches.
- *Phase J* is program-evidence-v1's A3: lags 0.5–60.5 s, ±10 ms, cross-day concentration baseline,
  names with ≥ 20 expected and ≥ 10 excess locked pairs.

**Already seen** (exploration and confirmation years, both used). Values are from
`src_py/metaorder_m1.py` on names eligible for both statistics under both rules, computed before
any 2019 output existed:

| year | names | combined J run60 | combined J stream5 | D [95% CI] | size-only D | phase-only D |
|---|---|---|---|---|---|---|
| 2024 | 160 | 0.258 | 0.258 | +0.001 [−0.025, +0.025] (tie) | −0.019 | +0.020 |
| 2021 | 249 | 0.267 | 0.306 | +0.039 [+0.008, +0.073] | +0.002 | +0.076 |

**Gate M1 (2019, read once).** D = combined J(stream5) − combined J(run60), name-paired bootstrap
(1,000):
- lower bound > 0: adopt stream5;
- upper bound < 0: keep run60;
- otherwise: a tie, and run60 stays as the incumbent.

2013 and 2016 are reported the same way as secondary replications. The adopted rule is used by
M2–M5.

## M2 — parent linkage: which bursts belong to longer same-side programs

**Link evidence per burst** (`src_py/metaorder_rows.py`). For burst k on side s, ending at b_k:
- *Pairs:* one untruncated non-round packet of k and one packet of the same class in any other
  burst (≥ 3 own-side packets), starting in (b_k, b_k + 1,800 s], in the same executed-side depth
  quartile.
- *Matches:* identical sizes.
- *Expected:* pairs × the depth-quartile-specific cross-day match rate over lags [0, 1,800) s, for the
  same side relation.
- The opposite-side version pairs k's packets with side −s packets in other bursts.

This yields same-side and opposite-side (pairs, matches, expected) per burst.

**Features** (size-free, observable by the end of the burst):
- the fingerprint-v1 burst features;
- backward context:
  - same-side and opposite-side in-burst volume in the preceding 300 s and 1,800 s, as shares of all
    signed volume in the same trailing window (a whole-day denominator would look ahead);
  - log time since the previous same-side burst ended;
  - that burst's log packet count.

No feature uses the forward window.

**Model.** The stage-3 binomial likelihood unchanged (`program_score.py` machinery), with target the
same-side link matches and chance q = expected / pairs. Fit on 2024 exploration; score 2021 once.

**Gates (2021):**

| gate | test |
|---|---|
| **L1** | same-side excess links per 1,000 pairs, top minus bottom score decile: lower bound > 0 |
| **L2** (directional) | L1's difference minus the same difference for opposite-side excess links: lower bound > 0 |
| **L3** | Spearman(decile, same-side excess) > 0.7 |

**Output.** A directional-linkage score for every burst, used by M3 and M5 and in the D and E
panels as a sensitivity.

## M3 — calibration on injected synthetic parents

2024 exploration, 50 names (sha256-ordered) × the first day of each of the 10 date pairs.

**Generator mixture** (fixed now, seeded per name-day), six parents per name-day:
- side ±1;
- start uniform 10:00–15:00; duration 600, 1,800 or 3,600 s;
- child interval Δ ∈ {5, 15, 30, 60} s;
- timing: jitter N(0, 0.1Δ), or timer-locked (whole-second anchor, ±2 ms), half each;
- sizes: a fixed non-round clip c ∈ [101, 999], or c ± 20% uniform, half each;
- each child capped below the prevailing same-side executed depth so it is untruncated.

Injected children inherit the preceding real packet's quote state, with no hidden fill. The
injection does not change other traders' behaviour or the book; this limit is stated in the
results.

**Measured per parent and per burst:**
- child-cluster recall: share of injected children inside bursts;
- purity: injected share of packets in bursts that contain injected children;
- linkage recall: share of injected bursts with a same-side identical-size link to another burst of
  the same parent within 1,800 s; false links to non-injected bursts;
- program-score and linkage-score AUC for injected-majority bursts against all other bursts (a lower
  bound, since the background contains real programs).

Descriptive, no gate. Expectation stated in advance: fixed-clip parents are linked far more often
than randomized-size parents, and timer-locked parents score higher on the program score.

## M4 — a real-time program score

Features are computed on the first three own-side packets of each burst:
- duration to the third packet, and mean and CV of its two gaps;
- truncation and hidden shares of the first three;
- spread, log depth and imbalance at the first packet;
- time of day and trailing 300 s activity;
- the M2 backward context.

The label stays the whole-burst identical-size repeats, so the label may use the future and the
features may not. Fit on 2024; score 2021 once.
- **R1:** top − bottom decile excess repeats per 1,000 pairs, lower bound > 0.
- **R2:** Spearman(decile, excess) > 0.7.

Real-time program triggers use the 2024 80th and 20th percentiles of this score.

## M5 — queue-aware passive orders against program-like bursts

**Data.** Raw LOBSTER messages, downloaded again.
- Exploration: 40 of the 2024 exploration names, confirmation: 60 of the 2021 names, each × the
  first day of each date pair.
- Names are those with median daily packet count between the group's 20th and 90th percentiles,
  taken in sha256 order. The extremes are excluded for compute.

**Triggers,** at the arrival of a burst's third own-side packet:
- *program*: real-time score ≥ q80;
- *bottom*: score ≤ q20;
- *placebo*: uniform times between 10:00 and 15:30, as many as program triggers, random side.

**Order.** 100 shares at the touch on the side the burst trades against (a sell at the best ask
when the burst buys). It joins the back of the displayed queue after all messages at the trigger
timestamp.

**Simulation** (`src_py/queue_sim.py`):
- *Queue ahead:* the displayed resting orders at that price when the order joins.
- *Advance:* executions and cancellations of those orders.
- *Fill:* when executed volume at the level, after joining, exceeds the remaining queue ahead. Hidden
  executions at that price count, since displayed orders have priority, and so does the level being
  exhausted.
- *Cancel:* after 60 s, or when a better price appears on the order's side.

**Outcomes:**
- fill rate;
- per-fill markout at 1, 10, 60 and 300 s, as (fill price − future mid) signed for the provider,
  in bps;
- expected P&L per posting = fill rate × mean markout;
- a scenario adding a 0.20¢/share rebate.

**Gates:**
- **Q1.** Per-fill 60 s markout, program minus bottom triggers, equal weight per name-day,
  Newey–West over days.
  - 2024: |t| > 3.
  - 2021: same sign, t > 2.
- **Q2** (descriptive). Expected P&L per posting for program triggers against placebo.

## Order and freeze points

1. **Cluster jobs** (bundled into long tasks after the 2026-09-14 short-job throttle):
   - module-A statistics for years 2013, 2016, 2019;
   - M2/M4 burst rows for both candidate rules, 2024 and 2021;
   - M5 message downloads and simulation, after M4 is fit.
2. **Order of reads:** M1 read first, then the rule is fixed; M2 and M4 fits on 2024; M3 and M5
   explorations on 2024; then 2021 reads for L, R and Q.

## What this cannot establish

- Linkage is directional algorithmic continuation. Institutional origin needs the external tests
  (program-evidence-v1 module D).
- Synthetic parents calibrate detection on real backgrounds, not realistic interaction with the
  book.
- The queue simulation assumes a small order that does not change other traders' behaviour.
  Latency, fees beyond the stated rebate, and adverse selection from our own presence are not
  modelled.

## Amendments before any output (2026-09-14)

1. **M3 participation cap.** A 3,600 s parent at a 5 s interval with a 500-share clip is 360,000
   shares, more than many names trade on NASDAQ in a day. The child count is therefore capped so a
   parent is at most 1% of the name-day's signed packet volume, and parents with fewer than four
   children are skipped. Clip sizes stay as declared.
2. **M2 context features.** The trailing-window denominator and the previous burst's packet count
   replace the declared "day volume" denominator and "previous burst program score". A whole-day
   denominator would look ahead, and a program score for stream5 did not exist when the rows were
   written.
