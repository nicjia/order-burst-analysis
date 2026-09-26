# forced-flow-v1 — does burst impact hold less when flow is mechanical?

Frozen 2026-09-23 after a DEV exploration run and before TEST.

## Idea

LLM_README §10 lists "forced/uninformed flow windows" as untested. Calendar dates fix when a large share of
trading is mechanical rather than informed: quarter-end (index rebalancing, window dressing, benchmark-tracking
trades) and quarterly expiry (index futures and options settlement). If the P4 informativeness filter — which
keeps bursts whose impact held over ten minutes — is picking up information, its bursts should behave no
differently on those days. If it is picking up price pressure, its bursts should give back *more* on days when
flow is mechanical. Timing is exogenous to any individual stock's information, which nothing else in this
project has.

## Specification (unchanged from the DEV run, `src_py/forced_flow.py`)

Outcome: name-day mean post-decision displacement to the close (P4 `d_close`, bps) of (a) the informative class
κ = 0.5, (b) all large bursts, (c) the non-informative class; and mean |signed large-burst flow| / adv20.
Regressors: quarterly-expiry, other-monthly-expiry, quarter-end and other-month-end dummies, name fixed
effects, standard errors clustered by date, 1/99% winsorization. Trade bursts.

## Exploration (DEV 2017–19, group 0; seen)

Informative-class d_close on quarter-end days −1.77 bps (t −2.74) against a normal-day mean of −1.46; all large
bursts −0.96 (t −1.94); non-informative +0.28 (t 0.52); quarterly expiry +1.38 (t 1.14), wrong sign and
insignificant.

## Confirmation (TEST 2022–25, groups 1–2, disjoint names and years)

**Primary:** the quarter-end coefficient on the informative class is negative with t < −2.
**Secondary, reported:** all-large and non-informative classes, quarterly expiry, and the flow-size regression.
A pass means the informativeness filter is contaminated by predictable mechanical flow; a fail means the
DEV result was noise. Either way it is reported as it falls, and neither changes any p4-revisit-v1 verdict.
