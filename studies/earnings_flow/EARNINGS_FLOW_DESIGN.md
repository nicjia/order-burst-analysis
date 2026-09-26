# earnings-flow-v1 — does burst flow before an earnings announcement predict the announcement return?

Frozen 2026-09-21, before any announcement return was joined to burst flow.

## Why

Institutional order flow predicts earnings-announcement returns (Hendershott, Livdan and Schürhoff 2015,
with ANcerno flow). Every burst test so far averaged over all days; information asymmetry is concentrated
before scheduled announcements, so any information bursts carry should be densest there. The p4-revisit-v1
daily tests (Q4) never conditioned on events.

## Data

- Burst flow: p4-revisit-v1 name-day tables (native signs, decision-time firewall), trade family primary.
  Daily x = S / adv20 with the 15:50 clock, where S is informative (S_info, κ = 0.5), other large
  (S_large − S_info) or all (S_all) signed burst volume.
- Announcements: Compustat `fundq.rdq`, linked to PERMNO through CCM (LU/LC, primary links), 2012–2025
  (`data/wrds/comp_rdq_2012_2025.csv.gz`). Day 0 = the first trading day on or after rdq.
- Returns: CRSP v2 daily. Abnormal return = stock return − value-weighted return of that day's point-in-time
  universe (lagged caps). CAR[0,+1] in bps is the outcome; CAR[+2,+21] is secondary (drift).

## Samples (name split sha256("p4-revisit-v1|permno") mod 3)

- **Exploration:** group-0 names, 2012–2019 (ERA2 group 0 plus DEV). Used to check sign and plumbing; no
  specification change is permitted after it except a bug fix, which is logged.
- **Confirmation:** groups 1–2, 2020–2025 (VAL plus TEST). Read once.
- **Replication:** groups 1–2, 2012–2016 (ERA2), read after confirmation.

## Specification

Pre-window: trading days −5 to −1; at least 3 of 5 days present, sum rescaled by 5 / days present.

Primary regression, one per sample, announcement-date fixed effects, PERMNO-clustered CR1, regressors and
outcome winsorized 1/99%:

CAR[0,+1] = β_info · x_info_pre + β_other · x_other_pre + γ₁ CAR[−5,−1] + γ₂ log cap + γ₃ turnover + FE

The pre-window return control is required: informative flow is selected on price moving its way, so it
tracks the pre-window return, which may itself predict the announcement move.

**Primary hypothesis:** β_info > 0. **Confirmation gate:** β_info > 0 with t > 2 in the confirmation sample
(one pre-specified test); t > 3 flagged as strong. Also reported: β_info − β_other.

Secondary, descriptive: all-burst flow x_all_pre alone; submission family; window −10 to −1; CAR[+2,+21];
and a decile sort of CAR[0,+1] on x_info_pre within announcement quarter.
