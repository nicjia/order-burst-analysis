# The MATH 279 burst model, rebuilt leak-free with the new features — 2026-09-24

Code: `studies/burst_forecasting/code/burst279_model.py` (results in `results/burst_forecasting/burst279_v1/results_T.json`). Trade bursts from economic
ITCH packets with native aggressor signs; per-burst p4-revisit-v1 samples.

**Pipeline.** Burst → label whether its impact is permanent (the label may look ahead) → predict that label from
what is known at the decision time T_dec = max(t_b + 10 min, t_e + 10 s) → trade at T_dec, exit at the close.
**Direction** is not predicted: each burst carries its aggressor side.

**No leakage.** Train on 2017–19 (one third of names); test on 2020–21 and on 2022–25 (the other two thirds of
names) — disjoint names *and* years. Hyperparameters fixed in advance; trading thresholds are 2017–19 deciles.
Features known by T_dec only; two linkage features that count bursts up to 30 minutes *after* the burst
(`link_same`, `link_opp`) were found to leak and are excluded. Placebo: the same model trained on labels
shuffled within day.

**Features.** BASE (burst shape and price path so far: size/ADV, children, duration, time of day, peak impact,
displacement at 1 and 10 minutes, retained-impact ratio, spreads, 30-minute pre-burst move, move since the
open) and NEW (identical-size fingerprint, program score — which contains inter-arrival regularity, book
imbalance and execution depth inside the burst — same-side same-size bursts in the previous 30 minutes,
truncated and hidden shares, and yesterday's repeated-clip programs on each side).

## 1. Prediction (gradient boosting, depth 3; AUC 0.5 = chance)

| | 2020–21 AUC | 2022–25 AUC | daily rank IC, 2022–25 |
|---|---|---|---|
| **"Permanent from burst start"** (classic label) | 0.639 | 0.637 | — |
| **"Keeps moving after T_dec"** (tradable label), BASE | 0.515 | 0.512 | +0.032 (t 6.8) |
| same, NEW features only | 0.500 | 0.501 | +0.003 (t 2.7) |
| same, BASE + NEW | 0.514 | 0.511 | +0.032 (t 6.8) |
| placebo | 0.502 | 0.501 | — |

- **The classic label is a trap.** Permanence measured from the burst's start contains the price move already
  seen by T_dec, so it looks very predictable (AUC 0.64). The part you can trade — what happens *after* you
  decide — is barely predictable (AUC 0.51), though reliably so (IC t ≈ 6–7 in both test periods).
- **The new burst-structure features add nothing** to predicting continuation. The top predictors are the
  stock's move since the open, time of day, the pre-burst move and the 10-minute displacement — i.e. intraday
  reversal, not burst identity.

## 2. Trading at T_dec, exit at the close (cost = half the quoted spread + 1 bp)

| strategy | 2020–21 gross (t) | net (t) | 2022–25 gross (t) | net (t) |
|---|---|---|---|---|
| follow every burst | −0.27 (−1.8) | −6.29 | −0.25 (−2.0) | −6.34 |
| **fade bottom decile of P(continue)** | **+8.30 (4.8)** | +0.99 (0.9) | **+5.40 (6.1)** | −1.85 (−0.1) |
| follow top decile | +5.52 (3.9) | −1.80 (−0.8) | +4.31 (5.5) | −3.23 (−2.8) |
| fade bottom decile, tight spread † | +3.46 (3.0) | **+1.18 (1.7)** | +3.32 (4.1) | **+1.07 (1.8)** |
| follow top decile, tight spread † | +2.60 (2.6) | +0.29 (1.2) | +3.25 (4.6) | +0.94 (1.8) |
| placebo, fade bottom decile | +2.20 (2.5) | −5.76 | +1.57 (3.6) | −6.54 |

bps per trade; t on daily P&L, Newey–West. † Spread filter (at or below the 2017–19 median) added after the
2020–21 read; 2022–25 is its first out-of-sample look.

**Reading.** The model's gross edge is real, replicates in two held-out periods on unseen names, and beats the
placebo by 3–6 bps. The round trip costs ~6–7 bps, which consumes it. Restricting to tight-spread bursts leaves
about +1 bp per trade in both periods, at t ≈ 1.7–1.8 — positive twice, not yet significant, and the filter
was chosen after the first test period.
