# Two-Avenue Research Design

This document records the earlier two-avenue experiments. **The current user mandate and
next design are `PROJECT_GOALS.md` and `studies/burst_information/BURST_PARTICIPATION_DESIGN.md`.** Blanket closure
language below records historical decisions; it does not close the newly authorized
participation/recurrence and alpha objectives.

This document separates two questions that the original project conflated.

## Central idea and decision trail

The central idea is still institutional order splitting: a large parent order may leave a
sequence of same-side aggressive child packets. LOBSTER can show the packet sequence but not
the parent ID. Every test therefore asks one of two narrower questions: does a packet fragment
forecast price discovery, or does it behave like unfinished same-side flow? Neither answer is
automatically proof of an institution or private information.

| Stage | Result | Decision it caused |
|---|---|---|
| Correct economic packets | One incoming order may create multiple messages; type-5 direction is unusable | Collapse timestamp executions and conservatively sign hidden rows before any fragment test. |
| Price-discovery models, 2023→2024 | Held-out-name permanence t = 0.98–1.60; every executable strategy loses 9.8–24.1 bps | Close the directional/trading branch. |
| Campaign joining | Only 1.7% of held-out fragment links accepted | Do not claim reconstructed parents or treat singleton fragments as campaigns. |
| Hurst validation | Sign-conditioned v2 is circular; sign-blind dH is not monotone in simulated parent purity | Abandon Hurst as a parent-recovery ranking and do not launch a full v3 search. |
| Continuation diagnostic, 2024 | High simulation score leaves +1.216 packets of 300s residual flow (t = 8.80), but controls were incomplete | Treat 2024 as exploratory; freeze complete controls and test once on untouched 2025. |

Current run: `strict-continuation-v1`. The first 2023 array 14587812 completed 472 names, but
QQQ and SPY exceeded the 8 GB task limit (exit 137); both are non-holdout training names.
Their isolated 16 GB repairs are 14592987 and 14592988, followed by refreeze 14592989 and a
repeated held one-day 2025 gate 14592990. All four completed cleanly; the repaired freeze has
382 non-holdout names, 2,252,761 sampled fragments, and checksum `fb55d37584cd`. All 474
training outputs are nonempty and the repeated gate passed without inspecting outcome values.
Untouched-2025 array 14593499 finished, with exact-name repairs 14597703--14597707 for five
12 GB failures and unchanged final aggregator 14597708. All repairs exited cleanly. The final
panel has 473 usable names, 111,521 name-days, and 249 dates; CL is the sole empty name.
Production summary checksum is `7c78215db52f`; independent audit checksum is `d29a64098da8`.

## Shared measurement layer

All new work starts from **economic aggressive packets**, not execution messages.
`src_py/execution_packets.py` groups same-timestamp type-4/type-5 executions before burst
formation. Type-4 aggressor sign is native (`-Direction`). Type-5 `Direction` is ignored:
hidden rows inherit a unique same-timestamp type-4 sign, are signed from an execution outside
the pre-event displayed quote, or remain sign 0. Ambiguous hidden packets break directional
fragments.

The original C++ detector is provenance-only. It counts message rows and cannot safely use
type-5 direction. Its future-return κ gate is disabled; new estimators do not call it.

## Avenue 1: classify price-discovering versus noisy fragments

### Claim that can be tested

A fragment's formation-time state may forecast **subsequent persistent price discovery**.
This does not identify private information or an institution. It is an operational forecast:
after a one-second buffer, is the signed midpoint move positive at 5, 15, and 30 minutes?

Two predeclared models are frozen:

1. `price_free`: timing, packet count/volume, size regularity, visible/hidden mix, spread,
   depth, imbalance, intensity, and time of day.
2. `post_end`: the same variables plus price impact observed by fragment termination and the
   ending half-spread. This model is valid only for a decision made after the fragment ends.

Future markouts are training labels in 2023, never formation filters. Both a continuous
median-markout forecast and a binary “positive at every horizon” classifier are fit. Model
coefficients and top-decile cutoffs are then frozen.

### Untouched evaluation

- Calendar training: 2023 only.
- Calendar test: 2024 only.
- Cross-sectional holdout: deterministic 20% of tickers, excluded from fitting.
- Inference: equal weight within name-day, then across names; Newey–West by day.
- Trading: enter at the opposite displayed touch one second after detection, exit at the
  displayed touch, and prevent overlapping positions within a ticker. Continuation P&L is
  reported separately at 5, 15, and 30 minutes.

Success requires both persistent signed price movement on untouched data and positive
executable P&L. A positive midpoint markout with negative touch-to-touch P&L is a measurement
result, not a strategy.

### What “informed” means here

The label is a **price-discovery proxy**, not a truth label for private information. Public
news, correlated order flow, quote revisions, and unobserved off-NASDAQ trading can all move
the price. Literal permanent impact is not observable within one trading day.

## Avenue 2: recover fragments and probabilistic parent campaigns

### Ground-truth laboratory

`src_py/metaorder_simulation.py` generates packet tapes with known parent IDs under:

- genuine order splitting;
- herding by unrelated traders responding to a common signal;
- overlapping parents;
- liquidity-sensitive pauses;
- partial venue observation; and
- a full model combining all mechanisms.

Fragment purity and parent coverage are evaluated only in simulation. A logistic fragment
score and pause-aware join model are fit on simulated parent labels, then frozen before real
data. Real campaign IDs are probabilistic reconstructions, never claimed identities.

### Empirical falsification tests

1. **Matched Hurst placebo.** Real packet boundaries are held fixed. Within each 30-minute
   bin, the signed packet sequence is circularly rotated 50 deterministic times. Thus every
   draw exactly matches time of day, coverage, packet counts, durations, and gaps while
   breaking alignment between detected boundaries and sign flow. The statistic is
   `H(placebo) - H(real)` with an empirical randomization p-value.
2. **Continuation beyond baseline.** A high simulation score must predict future same-side
   packet imbalance after residualizing intensity, time of day, spread, depth, and imbalance.
   This is evidence that a fragment behaves like an unfinished campaign; it is not the
   definition of informed trading.
3. **Simulation transport.** Score calibration and join behavior are compared across each
   simulated stress scenario. Poor real-data transport is reported as model failure.

### What cannot be proven

Anonymous, single-venue data cannot distinguish one parent split into many children from many
traders reacting to the same signal when both generate the same observable tape. Campaign
membership is therefore not point-identified. The publishable claim, if supported, is that a
specified reconstruction passes stronger placebo and ground-truth tests than arbitrary
aggregation—not that NASDAQ parent orders were observed.

## Audit state (2026-08-28)

- One AAPL 2024-01-03 preflight completed: 60,403 packets and 5,802 one-second fragments.
  The conservative rule left 18.5% of packets and 10.7% of volume ambiguous. These are pilot
  diagnostics, not manuscript results.
- Simulation calibration used 80 training and 30 untouched test days. Fragment scoring
  transported better than campaign joining; neither is treated as empirical evidence.
- SGE `14575613` (`bq2`) completed, but its favorable Hurst definitions are excluded:
  same-side runs used sign changes to determine boundaries. The sign-blind volume clocks
  were near zero or negative. `bq3` replaces them with three predeclared timing-only gap
  quantiles whose boundaries are invariant to sign permutations.
- SGE `14575617` (`twoav_tr`) and `14575623` (`twoav_fit`) completed. Frozen 2023 models were
  applied by `14575625` to untouched 2024 names and dates.
- The official OOS panel contains 469 usable names and 112,779 name-days. Formation-time
  price-discovery models do not transport to held-out names; every executable continuation
  specification is negative. This closes Avenue 1 as a directional strategy unless a
  genuinely new, predeclared economic mechanism is proposed.
- The simulation fragment score predicts subsequent same-side flow beyond the first baseline
  in both name cohorts. This is not yet a validated Avenue 2 result because that baseline
  omits some fragment-score inputs and lagged signed-flow state. The next test must use all
  formation inputs plus pre-fragment 60s/300s count and volume imbalance.
- The initial post-completion reversal diagnostic is excluded. The join model accepted only
  about 2% of links, so the code faded singleton fragments as if each were a completed
  campaign, and its non-overlap pass was not chronologically ordered. The repaired test
  requires at least two joined fragments and sorts decisions before enforcing non-overlap.
- SGE `14584362` (`bq3pilot`) was an engineering-only coverage pilot and was superseded before
  inference: its 10%/20% gap cuts rarely produced the 400 blocks required by the Hurst
  estimator. The replacement uses 30%/50%/70% cuts and retains sign-ambiguous packets in the
  timing geometry. No pilot Hurst outcome was used to choose these definitions.
- A 30-day-per-scenario ground-truth simulation then rejected Hurst as a recovery-ranking
  criterion even with sign-blind boundaries. In pure splitting, q50 had an 11.5% high-purity
  block rate and dH = -0.004, whereas the much less pure q70 definition had a 3.1% rate and
  dH = +0.096. Under overlapping parents, q50 had a 26.5% high-purity rate with negative dH,
  while q70 had 7.0% with positive dH. The full bq3 inference run is therefore not warranted.
  Timing-only Hurst compression is not monotone in known parent recovery.

The next Avenue 2 run is restricted to incremental continuation: residualize future 60s and
300s same-side packet flow on every simulation-score input plus pre-fragment count and volume
imbalance over both horizons. Even success is named continuation, not parent reconstruction.

## Frozen strict-continuation test (`strict-continuation-v1`)

The first 2024 continuation result is exploratory because its baseline omitted some score
inputs and lagged signed-flow state. The repair is not evaluated on 2024 again.

- Fit window: 2023, excluding the deterministic 20% name holdout.
- Excluded exploratory year: 2024.
- Final untouched window: 2025, reported separately for seen and held-out names.
- Controls: all thirteen simulation-score inputs plus sign-relative prior total count, count
  imbalance, total volume, and volume imbalance over both 60s and 300s.
- Nested models: fixed ridge penalty 10. The augmented model adds only the frozen simulation
  fragment score to the complete control model. No real return or future-flow outcome enters
  the score.
- Outcomes: future same-side minus opposite packet count at 60s and 300s (primary), plus the
  signed-log transform of future volume imbalance at the same horizons (secondary).
- Inference: name-day means, equal-weight across names by day, Newey-West(10).

The predeclared primary success gate is positive out-of-sample MSE improvement for the
300-second count target with NW t > 2 in both seen and held-out names, together with a
positive top-decile residual against the base model in both cohorts. The 60-second count must
have the same sign. Volume is corroboration only. Failing any primary condition closes the
simulation-score continuation branch. Passing it supports a continuation forecast—not
institutional information, common parentage, or executable trading.

The frozen gate **passes** in untouched 2025. For count_300s, delta MSE is +0.116 (t = 6.26)
in seen names and +0.105 (t = 6.09) in held-out names. Top-decile base residual flow is +1.471
packets (t = 12.44) and +1.907 packets (t = 10.49), respectively. Count_60s agrees in sign
and is significant in both cohorts; both volume horizons corroborate. The overall incremental
MSE fractions are only 0.0046% and 0.0032%, so this is a robust but small continuation signal,
not evidence that anonymous fragments reconstruct institutional parent orders.

## Decision tree after the 2025 test

If the primary continuation gate passes, the next experiment stays on the same causal chain:
estimate the hazard of the next same-side aggressive packet as spread and displayed depth
change after detection. This tests the liquidity-sensitive-pause mechanism—whether flow waits
when execution conditions worsen—without pretending to observe parent IDs. Only after that
would passive execution or order anticipation be simulated using queue-aware executable
prices.

That pass branch is now active. The next design must be frozen before another outcome is
opened: condition the next-same-side-packet hazard on post-detection spread and displayed-depth
changes, controlling for the complete strict-continuation state and baseline time-of-day hazard.

## Frozen liquidity-sensitive-pause test (`liquidity-pause-v1`)

This is a mechanism test, not another burst search. Fragment formation, economic-packet
reconstruction, simulation score, 2023 training names, 2025 seen/held-out cohorts, and the
score top-decile threshold are inherited unchanged from `strict-continuation-v1`.

- Risk episodes start one second after fragment end and use intervals (1,2], (2,5], (5,10],
  (10,30], (30,60], and (60,300] seconds. The event is the first subsequent same-side
  aggressive packet. Quotes and covariates are sampled immediately before each interval.
- To prevent one future packet from being recycled across many detections, candidate fragments
  are accepted chronologically with non-overlapping fixed 300-second windows separately by
  sign. This selection uses only fragment end times, never the future event time.
- Liquidity is directional. Contra depth is displayed ask size after a buy fragment and bid
  size after a sell fragment; same-side depth is the other queue. Time-varying covariates are
  log spread ratio and log contra/same-side depth changes relative to fragment end, plus
  strictly prior one-second signed count and volume state.
- The base pooled-logit hazard contains all 21 strict-continuation controls, fragment score,
  the frozen top-decile indicator, interval baseline-hazard dummies, end liquidity levels,
  all time-varying liquidity main effects, and the risk-time prior-flow controls.
- With ridge penalty fixed at 10, the joint model adds only top-decile interactions with spread,
  contra-depth, and same-side-depth changes. Two single-interaction models add spread or
  contra depth alone. Models fit on 2023 non-heldout names and are frozen before any 2025
  hazard output is inspected.
- Mechanism signs are negative for the top-decile × spread-change coefficient and positive for
  top-decile × contra-depth-change: expensive/thin executable liquidity should delay the next
  same-side packet. Same-side depth is a nuisance interaction, not a directional claim.
- Primary inference is name-day mean out-of-sample log-loss improvement, equal-weighted across
  names by day with Newey-West(10). Success requires the two training interaction signs above,
  positive joint-model improvement with NW t > 2 in both 2025 cohorts, and positive improvement
  from each corresponding single-interaction model in both cohorts. Passing supports a
  liquidity-responsive continuation fingerprint, not common parentage or causality.

Failure leaves the verified continuation result intact but rejects the liquidity-pause
mechanism. No alternative intervals, interactions, or burst definitions will be tried on 2025.

That failure branch is now active. The 2023 fit used 30,674,288 risk rows and 8,384,936
events across 382 non-heldout names. Both the single and joint models violate both directional
requirements: top-decile × spread change is positive (+0.01270 single, +0.01271 joint), while
top-decile × contra-depth change is negative (-0.00492 single, -0.00726 joint). The frozen
model checksum is `6f8380019b26`. No 2025 liquidity-pause output was generated. These signs
are associations in an endogenous risk set, not evidence that traders prefer bad liquidity;
they simply reject the predeclared liquidity-sensitive-pause fingerprint.

The anonymous-metaorder mechanism branch is therefore closed. The verified continuation
result remains useful as a boundary result, but the project now returns to the hidden-liquidity
measurement paper: revalidate hidden-print tables on economic packets and report identification
bounds over the type-5 rows that cannot be signed from pre-event public information.

## Frozen hidden-packet identification test (`hidden-packet-bounds-v1`)

This is the paper branch now in force. It uses individual economic timestamp packets and no
burst formation. Type-5 `Direction` is never read. A hidden execution has a known incoming
aggressor side only when a unique same-timestamp type-4 sign identifies the packet or when a
hidden-only execution lies strictly outside the pre-event displayed quote. All other hidden-only
packets are unsigned.

- Report packet and hidden-volume coverage for mixed visible/hidden, hidden-only outside-quote,
  hidden-only away-from-mid but inside the quote, and exact-midpoint populations.
- Measure midpoint changes from t+1 second to t+3, t+15, and t+30 minutes, so movement
  simultaneous with the execution cannot manufacture the result.
- For known-side packets, report the signed markout. For unsigned packets, report the
  conventional quote/tick-rule estimate only as a convention, alongside sharp worst-case
  bounds obtained by assigning each unsigned sign to minimize or maximize its realized
  markout. Equal packet weight is primary; hidden-volume shares describe coverage.
- Report 2023-2024 replication and 2025 confirmation separately, with name-day means,
  equal cross-sectional day weights, Newey-West(10), and the existing 1,000-bps trim.
- The old bifurcation survives only as a construction-sensitive descriptive fact if the
  away-mid quote-rule estimate is positive and midpoint tick estimate negative in both
  periods. It may not be called aggressive versus passive concealment: type 5 identifies a
  hidden resting execution, not which counterparty chose concealment.
- A minimally identified positive hidden-liquidity footprint requires the lower bound for
  the full hidden-packet population to exceed zero in both periods. If the bounds contain
  zero, the conclusion is non-identification: the sign/classifier convention, not the data
  alone, determines the informational ranking.

### Frozen packet-level spread-scaling companion

`hidden-packet-spread-scaling-v1` recomputes the old spread law without bursts or message-row
counts. The primary population is defensibly signed economic hidden packets (mixed packets
with a unique type-4 sign plus hidden-only packets strictly outside the pre-event quote).
Outside-only, unsigned-away quote-rule, and all-conventional populations are reported as
sensitivity arms. Markout runs from t+1s to t+3m; quoted half-spread is measured immediately
before the packet. Cross-name intercept, slope, correlation, markout/half-spread quintiles,
and the count exceeding twice the half-spread are reported separately for 2023-24 and 2025.
This is a friction-scaling measurement, not a trading test.

If the gate fails, the anonymous-metaorder branch is closed. The project then returns to its
strongest defensible contribution: hidden-liquidity measurement is not identified without a
signing/formation convention. The next paper work would revalidate the core hidden-print
tables on economic packets and report sensitivity/partial-identification bounds across the
unsigned type-5 population. It would not continue tuning burst definitions or trading rules.

No result from these jobs belongs in `paper.tex` or `main.tex` until its design and output are
audited and it is added to `VERIFIED_RESULTS.md` and `RESULTS_PROVENANCE.md`.


## 2026-09-13 addendum: authorized burst-information tests and join defect

The user has explicitly authorized the exploratory comparisons in `studies/burst_information/BURST_INFORMATION_DESIGN.md`.
A code audit found that the legacy simulation join fit pooled fragments across simulated days
on an intraday clock and compared parent IDs that restart daily. That fit/campaign-validation
exercise is excluded as evidence that reconstruction is exhausted. The corrected implementation
is separate (`metaorder_join_v2.py`), and the legacy config remains unchanged for provenance.
The separately trained fragment score and the existing strict-flow continuation measurements
are not invalidated by this join-code defect. New simulated join results are diagnostics only;
no real institutional-parent recovery claim is made.

Correction (2026-09-13): the "new simulated join results" are uninformative in a stronger
sense than "diagnostics only". On the saved test pairs, a "gap < 30 seconds" rule matches the
corrected model's precision and recall, because the legacy simulator has sparse parents and a
book independent of them. See `studies/burst_information/BURST_RECONSTRUCTION_RESULTS.md`. The burst-information screen's
contrasts are likewise uninformative (`VERIFIED_RESULTS.md` §1.24).

