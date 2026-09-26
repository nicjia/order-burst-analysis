# Incremental burst information: exploratory experiment v1

> **Read with the corrections (2026-09-13).** As designed, the model contrasts cannot answer
> Question 1 or 3:
> - targets are raw counts, so the most active names decide every pooled flow comparison;
> - no return or waiting-cost model in the matrix beats a zero forecast;
> - one landmark in 32 leaves a few thousand training rows;
> - `depth_imbalance_start` is not signed by burst direction.
>
> Results and post-hoc diagnostics are in `studies/burst_information/BURST_INFORMATION_RESULTS.md`. Any v2 must report R²
> against a zero forecast and normalize per name.

Authorized 2026-09-13. User priority: publishable information about bursts, with trading
value as a possible application. This reopens a bounded forecast comparison; it does not
revive a claim of identified institutional parents or treat previously inspected years as
fresh holdouts. No prior frozen output or model is overwritten.

## Questions

1. Do burst timing/size features improve future signed packet-flow forecasts beyond flexible
   flow/book and online regime controls?
2. Does the existing simulation-trained score add anything beyond those same inputs?
3. Does any improvement extend to future midpoint returns or indicative execution costs?
4. Why does reconstruction fail: contamination, splitting a parent into fragments,
   overlapping parents, pauses, ambiguous signs, or incomplete venue observation?

## Real-data comparison

The exact names/dates and parameters are in `results/burst_information_v1/design.json`.
Select 24 seen and 12 held-out names by a fixed hash from the existing 474-name universe,
using the existing name-holdout rule. Use 20 evenly spaced dates in each of 2023 and 2024.
Fit only 2023 seen names; evaluate 2024 seen and held-out names separately. Both years have
been used previously. This is a bounded exploratory screen, not final publication inference.
2025 is also previously inspected. Any confirmation sample needs a separate usage audit.

Canonical packets use `execution_packets.py`; type-5 Direction is ignored. Same-side runs
have gaps strictly below one second and a minimum of three packets. Observe each episode
at its third packet, sixth packet if reached, and recognized completion. Completion is the
earlier of the breaking packet and one second after the last packet. No decision occurs at
a retrospectively known endpoint. Missing/ambiguous signs break directional runs.

Keep a deterministic 1/32 sample of landmarks, selected by a hash available at each decision;
do not select using full-day counts, outcomes, or eventual episode length. The first five
minutes and last 301 seconds are excluded. Different stages have different eligible episode
populations; their averages must not be described as causal progression effects.

Every predictor stops at the decision time. All price and flow outcomes start after an
explicit one-second action latency. Horizons are 60 and 300 seconds. Flow means signed
count imbalance, including both same-side and opposing packets. Midpoint returns are separate
targets. The fifth target is the signed extra touch cost of waiting 60 seconds to execute
an otherwise-required one-share market order.

Five models, trained separately for each of three landmarks and five targets:

- Ridge on flow/book state.
- Fixed histogram gradient boosting on the identical state variables.
- The same boosting model plus a fixed online two-state regime filter.
- The same state/regime model plus burst geometry and size features.
- The same state/regime/burst model plus the frozen simulation score.

The state contains signed counts/volume, total activity, ambiguous counts, and observed
returns over 1/5/60/300 seconds, current spread/depth/imbalance, and time of day. The regime
benchmark is a two-state binomial HMM with buy probabilities .35/.65 and symmetric transition
probability .01 per second, filtered on completed one-second bins. It is **not a replication
of Tsaknaki et al.'s score-driven BOCPD**. ClusterLOB is relevant prior art but is not replicated
by these models. The existing simulation score is a representation, not a calibrated real-parent
probability, especially when applied to burst prefixes.

Primary contrast: state/regime/burst versus state/regime, at the third packet, predicting
300-second flow in 2024 held-out names. Secondary contrast: add the simulation score.
All other results are diagnostics; publish the full matrix. No model or hyperparameter is
selected using evaluation outcomes. Boosting settings are fixed in the design JSON.

Training observations are weighted equally within name-day. Report paired loss differences
as name-day means, then equal cross-name means by date. Report NW(10), error-reduction fractions,
and intervals, explicitly noting the weakness of only 20 sampled evaluation dates and that
lags index observed dates. Statistical significance is not a confirmation claim. A subsequent
full contiguous panel is necessary if the screen warrants continuation.

Invalid rows and individual landmark price outcomes above 1,000 bps are explicitly counted;
all models use the same usable sample. This landmark trim differs from the old stated
name-day trim and must not be silently combined with legacy estimates. Missing raw archives
are distinct from failed downloads/extractions. Fail the evaluation on absent or failed receipts,
incorrect ticker/dates, duplicate row IDs, or absent test cohorts.

## Economic diagnostic

Use an identical, chronological, score-independent set of nonoverlapping opportunities for
all policies. Every required order is executed either now (after one-second latency) or after
60 seconds; there is no cost-free abstention. Predict negative waiting cost => wait; otherwise
execute now. Compare against always-now, always-wait, and the state/regime policy. Report both
cohorts and all stages/models, including losses. Verify at least one share of displayed depth
at each possible execution point. Equal fixed per-share fees cancel between the timing choices.

These are indicative infinitesimal touch costs. They do not establish finite-size profitability,
passive fills, consolidated routing performance, or counterfactual market impact. Execution
outputs are diagnostic until the incremental forecast survives stronger/full-panel validation.

## Controlled reconstruction diagnostic

`src_py/burst_recovery_diagnostic.py` generates 30 paired days under nine fixed treatments:
isolated parents, background flow, dense background, overlapping parents, random pauses,
35% venue observation, 20% sign ambiguity, an observationally equivalent no-parent tape,
and combined disturbances. Three fixed detectors: timing-only clusters, same-side runs,
and a size-consistency run sensitivity. No parameter is tuned on recovery outcomes.

Report purity, pair precision/recall, parent coverage, and fragmentation separately. High
purity does not imply full-parent recovery. Unrelated background trades do not share a common
parent merely because their ID is -1. The no-parent tape deliberately has identical observables
to a split-order tape; it is an identification counterexample, not a realistic estimate of
herding prevalence. Other treatments are a synthetic laboratory, not calibrated NASDAQ mechanisms.

The legacy simulator's so-called liquidity-sensitive pauses are random long gaps independent
of its independently generated book state. Neither simulator can validate an institution's
book-adaptive execution policy. That would require a specified adaptive agent and a book model,
or identifiers/parent labels; it must not be inferred from these pause experiments.

## Trial accounting and provenance

This study adds five forecast configurations, three landmarks, five targets: **75 fitted
specifications**, with four paired contrasts across two cohorts (**120 reported comparisons**).
The reconstruction laboratory has 3 definitions x 9 treatments = 27 fixed diagnostic cells.
None is an independent fresh alpha discovery. These supplement, rather than reset, the
historical search ledger. The earlier ledger's 66/~110 totals omit later research phases.

Outputs remain under the gitignored results group. Source, design, and runtime hashes accompany
the final review. Do not promote results into VERIFIED_RESULTS or manuscripts before auditing
coverage, recognition timing, source versions, and aggregation.

Relevant primary literature: [online regimes](https://arxiv.org/abs/2307.02375),
[ClusterLOB](https://arxiv.org/abs/2504.20349),
[synthetic metaorders](https://arxiv.org/abs/2503.18199),
[identification limits](https://arxiv.org/abs/2602.19590),
[splitting versus herding with identifiers](https://arxiv.org/abs/1108.1632).
