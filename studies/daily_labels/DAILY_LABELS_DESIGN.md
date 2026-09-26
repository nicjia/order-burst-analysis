# daily-labels-v1 — burst flow against daily retail and institutional imbalances

Frozen 2026-09-23 before any regression was run.

## Idea

p4-revisit-v1 could only test institutional content against **quarterly** 13F and mutual-fund holdings, where
the whole panel is a few thousand name-quarters and the surviving effect is t ≈ 2.5–3.3. The BJZZ TAQ extract
already pulled (`data/p4/iid_*.csv.gz`) carries, per name-day, both retail buy/sell volume (subpenny
identification) and **large-trade (≥ $50k) buy/sell volume** — a daily institutional-size proxy. That is a
daily external label on the same names and days as the burst tables, with roughly three orders of magnitude
more observations than the quarterly tests.

## Measures

Per name-day: retail imbalance RI = (buy − sell)/(buy + sell) on retail volume; institutional imbalance
II = the same on ≥ $50k volume. Burst regressors, all divided by adv20: informative flow x_info (κ = 0.5,
trade family), other large flow x_other, program-linked fingerprint flow x_L and its mirror placebo x_M
(fingerprint-multiday-v1 amendment A1, m = 3), and remaining fingerprint flow x_U.

## Specification

Two regressions, one per label, day fixed effects (absorbing market-wide retail and institutional days),
PERMNO-clustered CR1, 1/99% winsorization:

II (or RI) = β_L x_L + β_M x_M + β_U x_U + β_info x_info + β_other x_other + γ·(same-day return, lagged
return, log cap, turnover) + day FE

**The same-day return control is mandatory** — every burst measure here is selected on price moving its way,
and both labels co-move with the day's return.

**Mechanical overlap:** burst volume is part of consolidated volume, and a child of ≥ $50k enters II directly.
The placebo handles it: mirror-linked flow has the same child sizes and the same overlap but no program
continuity, so the primary statistic is the **contrast β_L − β_M**, not β_L itself.

**Primary:** β_L − β_M > 0 in the II regression. **Secondary:** β_L − β_M in the RI regression (expected ≤ 0
if programs are institutional), and β_info − β_other in both.
**Gate:** exploration DEV; confirmation TEST with the same sign and t > 3 (the sample is large enough that t > 2
is weak evidence here). Replication in VAL and ERA2 afterwards.
