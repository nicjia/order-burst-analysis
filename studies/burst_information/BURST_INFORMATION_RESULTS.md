# Burst information: completed exploratory tests (corrected 2026-09-13)

**The screen does not show that burst features add or remove forecasting information. None of
its return or waiting-cost models beats a zero forecast, and its primary flow contrast is
dominated by two stocks. It establishes no tradable signal.** An earlier version of this file
reported a "failed primary test" and a "4.009% price-forecast improvement"; both readings are
withdrawn below and listed as excluded evidence in `VERIFIED_RESULTS.md` §2.

## What was run

75 fixed specifications and 120 paired comparisons over 36 stocks and 40 sampled dates:
training on 2023 for 24 stocks, evaluation on 2024 for those stocks plus 12 held-out stocks. Of
1,440 requested stock-days, 1,410 archives were available and 30 confirmed missing; 64,618
valid sampled landmarks. The independent CSV audit reproduces all comparisons (maximum
difference 2.84e-14). It verifies arithmetic, not whether a comparison can support its reading.

## What the screen shows

Post-hoc diagnostics, `src_py/diagnose_burst_information_v1.py` → `results/burst_information_v1/posthoc/`.
All were chosen after the v1 results were read; they are exploratory.

1. **No model predicts signed returns or waiting costs.** Out-of-sample R² against a zero
   forecast is negative in **90 of 90** return and wait-cost model cells. Third packet, 60s
   return, held-out stocks:

   | model | R² vs zero forecast |
   |---|---:|
   | ridge, state | −2.6% |
   | tree, state | −16.4% |
   | tree, state + regime | −14.9% |
   | tree + burst features | −10.3% |
   | tree + burst + simulation score | −11.7% |

   The withdrawn "4.009% improvement" is the move from −14.9% to −10.3%.

2. **Future signed flow is predictable** from trailing flow and book state: R² against zero
   is positive in all 60 flow cells (1.1% to 14.6%).

3. **The primary flow contrast measures AMZN and AMD.** Targets are raw packet counts. In the
   held-out cohort AMZN supplies 52.5% and AMD 41.1% of baseline MSE, and 61% of the burst
   contrast. Their target standard deviations are 136–151 packets against 8–32 for the other
   held-out stocks, and no training stock is close. The withdrawn "−0.155% (t = −4.30)" tests
   how tree models extrapolate to those two stocks.

4. **With per-stock scale-free targets there is still no consistent burst increment.** Flow is
   divided by the trailing signed-packet rate × horizon and returns by the half-spread; the
   same train/test split is refit. Adding burst features improves 12 of 32
   landmark × target × model × cohort cells: 2 significantly better, 5 significantly worse.
   Replacing the unsigned `depth_imbalance_start` with a direction-signed version does not
   change this (9 of 32 improved; 3 significantly better, 8 worse). Adding the simulation score
   to the burst model improves 9 of 32 (1 better, 9 worse).

5. **Predictable flow is priced; surprise flow moves prices.** A per-stock 2023 persistence
   model predicts 2024 same-side flow (median R² 7–9% at 300s). Regressing the 2024 return on
   predicted flow and the surprise gives 0.00–0.13 bps per predicted packet against 0.26–0.53
   bps per surprise packet (medians; surprise t > 2 in 18–21 of about 21 stocks, predicted t > 2
   in 4–6). See `VERIFIED_RESULTS.md` §1.26.

6. **Ranking bursts by predicted return does not clear the spread.** Within-stock 2024 deciles
   of a 2023-fitted ridge return forecast: 39 of 40 decile cells lose after paying the
   half-spread at the reference time, with no monotone ordering. A per-stock fit on 2024 itself
   shows +5 to +16 bps, which is overfitting on a few hundred rows per stock.

The execution diagnostic (one-share "now versus 60s later") uses the same wait-cost forecasts,
all of which lose to zero. Its incremental saving versus the state/regime policy was 0.096 bps
(t = 0.82) held-out and −0.083 bps (t = −0.56) seen: no evidence either way.

## Design limits to fix in any v2

- Sampling one landmark in 32 left 5,445 / 952 / 5,438 training rows at the third / sixth /
  completion landmarks, evaluated on only 20 dates with NW(10).
- Raw count targets and pooled MSE let the most active stocks decide every flow contrast.
  Normalize per stock and report R² against a zero forecast next to every relative MSE.
- `depth_imbalance_start` in the burst block is not signed by burst direction; every target is.
- The seen and held-out cohorts differ sharply in activity (four of the most active stocks are
  held out), so cohort differences mix transport with scale.

## Reconstruction diagnostics

Corrected in `studies/burst_information/BURST_RECONSTRUCTION_RESULTS.md`: the session-mixing join bug is real, but the
corrected join model's synthetic 92.6% / 91.6% equals a "gap < 30 seconds" rule in a simulator
whose book carries no parent information. It is not evidence about real reconstruction.

## Relation to literature

None of these findings establishes first-in-literature novelty. Order splitting and persistent
flow are established ([Toth et al.](https://arxiv.org/abs/1108.1632)); online flow regimes
([Tsaknaki et al.](https://arxiv.org/abs/2307.02375)) and order clustering
([ClusterLOB](https://arxiv.org/abs/2504.20349)) are close prior work; synthetic calibration does
not validate inferred parents ([Maitrier et al.](https://arxiv.org/abs/2503.18199),
[Goliath and Gebbie](https://arxiv.org/abs/2602.19590)). The simple two-state regime filter here
is not a replication of the published score-driven changepoint model.

## Full fixed comparison matrix (as run; see the corrections above before reading)

Positive percentages mean lower MSE for the added block. Every return and wait-cost model in
this matrix loses to a zero forecast, and the flow rows are dominated by the most active
stocks, so these relative MSE changes are not evidence of burst information in either direction.

Parentheses give the paired daily t statistic. Nonlinearity compares tree state to linear state; regime adds the online filter; burst adds burst features; score adds the frozen simulation score.

| Landmark | Target | Cohort | Nonlinearity % (t) | Regime % (t) | Burst % (t) | Score % (t) |
|---|---|---|---:|---:|---:|---:|
| third | flow_60s | seen | -2.165 (-2.46) | -0.328 (-0.76) | +0.771 (+5.66) | -0.593 (-11.82) |
| third | flow_60s | heldout | +3.850 (+1.77) | -0.017 (-0.17) | +0.050 (+0.95) | -0.060 (-2.12) |
| third | flow_300s | seen | +6.417 (+4.35) | +0.291 (+1.92) | -0.310 (-1.46) | -0.211 (-0.93) |
| third | flow_300s | heldout | +6.702 (+2.80) | -0.142 (-1.38) | -0.155 (-4.30) | -0.097 (-0.99) |
| third | return_60s | seen | -2.714 (-5.48) | -0.117 (-0.21) | +0.210 (+0.63) | +0.211 (+1.32) |
| third | return_60s | heldout | -13.499 (-6.13) | +1.306 (+2.00) | +4.009 (+9.25) | -1.323 (-4.21) |
| third | return_300s | seen | -2.642 (-3.66) | +0.682 (+2.32) | -0.111 (-0.28) | +0.396 (+1.32) |
| third | return_300s | heldout | -7.433 (-9.38) | +1.877 (+2.79) | -0.375 (-1.12) | +1.372 (+2.04) |
| third | wait_cost_60s | seen | -3.897 (-5.76) | +0.635 (+2.87) | +0.020 (+0.06) | +0.498 (+0.93) |
| third | wait_cost_60s | heldout | -12.396 (-5.12) | -0.897 (-2.22) | +1.684 (+7.98) | +0.621 (+1.91) |
| sixth | flow_60s | seen | +0.591 (+0.49) | -1.182 (-1.81) | +0.514 (+0.81) | -0.412 (-1.26) |
| sixth | flow_60s | heldout | +0.757 (+1.42) | +0.100 (+1.10) | -0.054 (-0.21) | -0.048 (-0.88) |
| sixth | flow_300s | seen | +4.154 (+1.45) | +1.007 (+6.41) | -0.139 (-0.16) | -0.189 (-0.50) |
| sixth | flow_300s | heldout | +4.004 (+2.65) | +0.027 (+0.29) | +0.172 (+1.89) | +0.061 (+1.32) |
| sixth | return_60s | seen | -3.850 (-1.44) | -0.169 (-0.31) | +0.538 (+1.42) | -0.487 (-0.72) |
| sixth | return_60s | heldout | -2.807 (-0.60) | +1.233 (+0.91) | -1.382 (-0.84) | -0.017 (-0.02) |
| sixth | return_300s | seen | +12.325 (+2.88) | +0.863 (+2.31) | -0.372 (-0.42) | +0.063 (+0.20) |
| sixth | return_300s | heldout | -28.874 (-5.63) | +4.634 (+6.35) | +0.077 (+0.23) | -0.110 (-0.29) |
| sixth | wait_cost_60s | seen | -1.835 (-0.73) | -1.195 (-2.98) | +1.241 (+1.94) | -0.727 (-1.53) |
| sixth | wait_cost_60s | heldout | -1.243 (-0.24) | -0.028 (-0.02) | -0.098 (-0.05) | +1.463 (+1.28) |
| completion | flow_60s | seen | +0.308 (+0.30) | -0.018 (-0.08) | +0.577 (+1.84) | -0.211 (-2.52) |
| completion | flow_60s | heldout | +5.256 (+3.32) | +0.055 (+0.74) | -0.108 (-2.70) | -0.096 (-0.88) |
| completion | flow_300s | seen | +4.023 (+3.75) | +0.056 (+0.29) | +0.554 (+1.86) | -0.061 (-0.18) |
| completion | flow_300s | heldout | +6.385 (+2.46) | -0.132 (-2.89) | -0.030 (-0.47) | -0.116 (-3.31) |
| completion | return_60s | seen | +0.617 (+0.49) | -0.091 (-0.24) | +0.440 (+1.68) | -0.328 (-1.21) |
| completion | return_60s | heldout | -5.096 (-1.90) | +0.190 (+0.41) | +0.811 (+1.04) | -0.855 (-3.99) |
| completion | return_300s | seen | -2.453 (-2.52) | +0.096 (+0.64) | +0.547 (+0.87) | -0.459 (-1.82) |
| completion | return_300s | heldout | -7.318 (-3.64) | -1.498 (-1.89) | +2.313 (+2.52) | -0.973 (-1.18) |
| completion | wait_cost_60s | seen | +1.056 (+0.83) | -0.697 (-2.18) | +0.133 (+0.94) | +0.287 (+2.25) |
| completion | wait_cost_60s | heldout | -3.992 (-1.89) | -0.230 (-0.43) | +0.127 (+0.15) | +0.284 (+0.57) |

## Full execution diagnostic

Savings in bps per required one-share order. All five policies use the same score-independent non-overlapping schedule within each landmark/cohort. The parenthesized t statistics have the same exploratory limits as above.

| Landmark | Cohort | Model | Orders | Saving vs immediate (t) | Saving vs state/regime (t) |
|---|---|---|---:|---:|---:|
| third | heldout | ridge_state | 6770 | -0.0196 (-0.24) | -0.1086 (-0.65) |
| third | heldout | gbt_state | 6770 | +0.1269 (+1.66) | +0.0379 (+0.51) |
| third | heldout | gbt_state_regime | 6770 | +0.0890 (+0.65) | +0.0000 (undefined) |
| third | heldout | gbt_state_regime_burst | 6770 | +0.1854 (+3.39) | +0.0964 (+0.82) |
| third | heldout | gbt_state_regime_burst_score | 6770 | +0.1770 (+2.42) | +0.0880 (+1.12) |
| third | seen | ridge_state | 5004 | -0.4584 (-4.66) | +0.0165 (+0.08) |
| third | seen | gbt_state | 5004 | -0.5180 (-2.14) | -0.0431 (-0.71) |
| third | seen | gbt_state_regime | 5004 | -0.4749 (-2.16) | +0.0000 (undefined) |
| third | seen | gbt_state_regime_burst | 5004 | -0.5580 (-1.57) | -0.0831 (-0.56) |
| third | seen | gbt_state_regime_burst_score | 5004 | -0.3531 (-1.61) | +0.1218 (+2.16) |
| sixth | heldout | ridge_state | 2102 | -0.1488 (-1.16) | -0.3655 (-2.13) |
| sixth | heldout | gbt_state | 2102 | +0.1187 (+0.85) | -0.0980 (-1.11) |
| sixth | heldout | gbt_state_regime | 2102 | +0.2167 (+1.09) | +0.0000 (undefined) |
| sixth | heldout | gbt_state_regime_burst | 2102 | +0.0819 (+0.47) | -0.1348 (-1.60) |
| sixth | heldout | gbt_state_regime_burst_score | 2102 | +0.3782 (+1.56) | +0.1615 (+1.28) |
| sixth | seen | ridge_state | 995 | -0.5823 (-3.64) | -0.0438 (-0.23) |
| sixth | seen | gbt_state | 995 | -0.4058 (-1.61) | +0.1327 (+2.43) |
| sixth | seen | gbt_state_regime | 995 | -0.5385 (-2.45) | +0.0000 (undefined) |
| sixth | seen | gbt_state_regime_burst | 995 | -0.6270 (-2.92) | -0.0886 (-1.13) |
| sixth | seen | gbt_state_regime_burst_score | 995 | -0.6377 (-3.02) | -0.0992 (-2.64) |
| completion | heldout | ridge_state | 6645 | +0.0891 (+0.57) | -0.1286 (-0.73) |
| completion | heldout | gbt_state | 6645 | +0.1620 (+1.79) | -0.0556 (-0.60) |
| completion | heldout | gbt_state_regime | 6645 | +0.2176 (+3.25) | +0.0000 (undefined) |
| completion | heldout | gbt_state_regime_burst | 6645 | +0.2784 (+3.75) | +0.0607 (+0.76) |
| completion | heldout | gbt_state_regime_burst_score | 6645 | +0.2763 (+4.45) | +0.0587 (+0.85) |
| completion | seen | ridge_state | 5006 | -0.0415 (-0.50) | -0.1243 (-1.56) |
| completion | seen | gbt_state | 5006 | +0.1010 (+0.73) | +0.0182 (+0.67) |
| completion | seen | gbt_state_regime | 5006 | +0.0828 (+0.64) | +0.0000 (undefined) |
| completion | seen | gbt_state_regime_burst | 5006 | -0.0001 (-0.00) | -0.0830 (-3.41) |
| completion | seen | gbt_state_regime_burst_score | 5006 | +0.0610 (+0.36) | -0.0218 (-0.47) |

## Reproduction and outputs

Use the project Python environment, one numerical-library thread, and the frozen design. Run `src_py/evaluate_burst_information.py --root results/burst_information_v1`, then `src_py/audit_burst_information.py --root results/burst_information_v1`. The corrections above come from `src_py/diagnose_burst_information_v1.py --root results/burst_information_v1` (run from `src_py/`), which writes `posthoc/diagnostics.json` and six CSV tables without modifying any v1 file.

The result group contains `summary.json`, `independent_audit.json`, `daily_comparisons.csv`, 75 saved models, 15 prediction panels, `model_loss_levels.csv`, and the explicitly post-result name diagnostics. `manifest.json` records extraction sources; `evaluation_manifest.json` records evaluation sources before fitting. `tail_merge_audit.json` accounts for all planned extractions. No confirmation sample was consumed in this study.
