# Burst participation and state-dependent recurrence, v2

> **Status 2026-09-13 (review):** paused. Its synthetic milestones rest on the legacy simulator,
> whose book is independent of parents and whose parents rarely overlap. There, a 30-second gap
> rule matches the corrected join model and the any-program label changes only 9% of fragments
> (`studies/burst_information/BURST_RECONSTRUCTION_RESULTS.md`), so they cannot validate real bursts. The real-data
> questions here are now tested by `studies/fingerprint/BURST_FINGERPRINT_DESIGN.md`: whether children of one program
> can be identified from repeated sizes, and whether one algorithm acts in similar book states.
> The label code (`metaorder_participation.py`) is correct and kept. Treat the rest as design
> notes until a simulator with state-dependent execution and realistic parent density exists.

Design and implementation contract, 2026-09-13. This document fixes the research question,
labels and comparison structure. It is not a claim that all experiments below have run or a
completed pre-registration of a confirmation test. Simulator parameters, fitting details and
confirmation gates must be recorded before generating/inspecting their evaluation outcomes.

## Target

For an observed candidate episode, estimate the fraction of executed volume arising from
**any** parent execution program. Also report the packet-count fraction. A mixture of two
parents qualifies as program participation even when neither dominates individually. Keep
single-parent purity as a diagnostic, not as the target.

The primary target is continuous. For descriptive classification use fixed categories:
program-heavy >=80% volume, background/noise-heavy <=20%, mixed in between. Do not discard
mixed cases when evaluating the continuous target. These cutoffs are interpretive defaults,
not optimized trading thresholds. A single program execution can be observed when the rest
of its program executes elsewhere; do not relabel it as noise merely for being a singleton.

Known synthetic parent IDs supply labels. Identified real parent records can eventually
supply external labels after checking their linkage/coverage. Exchange order IDs are not
parent-program IDs. Missing identities must remain unknown, never become negative labels.
Real anonymous prices and later order flow do not supply supervised parent truth. Targets
and features are stored separately; parent IDs, future outcomes, scenario IDs and true
remaining inventory are forbidden predictor inputs.

`src_py/metaorder_participation.py` implements observed count/volume fractions, number of
parents, dominant-parent count fraction and the descriptive classes. It requires an explicit
known-truth source, complete labels, positive volume and valid inclusive fragment bounds.
The legacy fragment scorer and all v1 frozen outputs remain unchanged.

## Observable decision times

1. **Before onset:** use a regular time or event grid that includes non-burst periods, not
   just retrospectively selected burst starts. Features use information strictly before the
   grid point. Forecast the next fixed-window burst onset/side as an observable target;
   latent active-program targets are available only in known-truth simulation.
2. **Early episode:** recognize the third packet using the existing packet convention.
   Compare the strictly pre-first-packet state with the execution sequence observed so far.
3. **After response:** wait for actual completion recognition plus a fixed one-second response
   window. Include only response available by then. Any flow, return or trading outcome begins
   after that decision and an additional fixed one-second action latency.

For comparisons at a given decision, every model sees the same eligible sample and forecasts
identical later outcomes. A pre-state-only model evaluated at the after-response decision is
an ablation, not a claim of a trade made before onset. Threshold/silence-defined burst
completion and all episode history must be computable on an observed prefix of the tape.

## Feature blocks and comparisons

- **Pre-state:** spread, side-specific depth, imbalance, prior flow and returns, replenishment
  and cancellation activity observed before onset, time of day and local activity rate.
- **Execution:** packet size, spacing, volume, intensity, aggressiveness and known/ambiguous
  hidden participation up to the permitted decision.
- **Response:** book depletion/refill, spread/depth recovery and signed price response over
  the fixed observation window. Mechanical contemporaneous impact is not a later outcome.
- **Recurrence:** earlier same-side episodes, their pre-states and state/action relationships,
  elapsed time, intervening opposite activity, and similarity conditional on activity and
  current book state. Limit candidates to the same stock/day and earlier observable episodes.

Compare nested blocks against both regularized linear and fixed flexible state baselines,
plus a simple recent-flow persistence baseline. Compare true-history features with
appropriately time-ordered, state-matched placebo history. Do not select the easiest baseline
or treat a score derived from simulator assumptions as calibrated market probability.

The key recurrence test is whether earlier episodes improve prediction at the **same current
state**. Include state-matched no-burst/no-recurrence opportunities. Shared reaction to public
conditions by independent traders is a hard negative, not evidence of a common parent.
State similarity may reveal a shared policy without identifying a shared institution.

## Simulator needed for meaningful classification development

The legacy simulator's book snapshots are independent and its long pauses random. It can
check label arithmetic, but cannot validate the user's state-dependent execution hypothesis.
Build a separate sequential simulator in which orders consume depth, liquidity replenishes,
and agents observe only pre-action public state. Policies must use explicitly implemented
state-dependent actions rather than a post-hoc label attached to random pauses.

Include spread/depth-sensitive execution, participation-rate and schedule-driven programs,
private remaining inventory and deadlines, concurrent opposing and same-side parents,
partial venue observation, hidden/unknown signs, and unrelated traders who respond to the
same public book state. Include non-program herding with comparable activity/size patterns.
Do not give parents trivially unique sizes, impact functions or book states. If only top-of-
book dynamics are implemented, describe it as such, not as a calibrated full exchange.

Withhold entire policies/parameter ranges as well as independent simulation seeds. Compare
against prevalence and ordinary-state models. Evaluate continuous count/volume error,
calibration, precision-recall with class prevalence, false positives on non-program herding,
and degradation from overlap/thinning. Use day-level uncertainty. Randomly splitting child
packets from one parent between training and testing is prohibited.

## Market translation and alpha

First test incremental next-flow prediction on exploratory years, with names and calendar
separated from model training. Results are observable predictive validity, not confirmed
parent membership. Then test whether any gain extends to future price or an executable
strategy. A persistent program can already be anticipated or have little remaining volume.

Use current book/flow baselines, including the simple linear model that beat the v1 price
model. Record every fit and threshold. Before confirmation, freeze model, direction, horizon,
latency, selection, sizing, exit, costs, risk normalization and economic success gate. Audit
whether the proposed confirmation period was previously inspected. No new confirmation
period is selected or consumed by this design document.

## Completed first milestone and remaining work

Implemented the new known-truth labels with tests for multiple-parent mixtures, volume versus
count weighting, unknown labels, descriptive classes and invalid episode boundaries.
`diagnose_participation_target.py` directly audits labels on 30 legacy simulated days (five
per each of six scenarios), records all scenarios and source hashes, and fits no model.
Output: `results/burst_participation_v2/target_audit.json`.

This audit checks the target change only. The sequential book-adaptive simulator, feature
extractor, fitted v2 classifiers, recurrence comparisons and alpha evaluation remain to be
implemented and run. A finished label audit is not completion of either ultimate project goal.
