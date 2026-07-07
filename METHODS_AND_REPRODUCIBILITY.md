# Methods & Reproducibility Note

A short guide for a reviewer who wants to know **exactly what was computed, from
what data, with what code** — including the one place where an earlier version of
this note (and the manuscript) reported an **overstated standard error** that has
since been corrected with day-level inference.

---

## 1. The headline question: "why did it suddenly work?"

**It did not.** All three empirical claims are exactly where they were, and the
one that briefly *looked* like it strengthened was an inference artifact that is
now fixed:

| Claim | Correct statistic | Significant? |
|---|---|---|
| Hidden-execution 3-min markout | +1.6 bps, t = 3.4 (pooled, date-clustered) | yes |
| Overnight predictability / reversal **strategy** | OOS Sharpe ≈ 0.8, t ≈ 1.4; fails deflation | **no** |
| Tick-regime **asymmetry** (R3 gradient) | day-level Fama–MacBeth NW **t ≈ −1.6** (raw), **−2.0** (orthogonalized) | **no** (suggestive only) |

**What happened with R3.** The first pass regressed 434 per-name reversal
profitabilities on log price with HC1 (heteroskedasticity-robust) errors and got
**t = −3.81**. That standard error is wrong: every per-name average is taken over
the *same* ~1,000 days, so the 434 observations are not independent — day shocks
correlated with the regressor (there is a common "cheap-name reversal factor",
and the effect is front-loaded into 2023) inflate the naive t. Re-estimating the
gradient **Fama–MacBeth style** (cross-sectional slope each day, Newey–West over
the ~1,000 daily slopes) gives **t = −1.57** raw / **−2.01** orthogonalized, with
a 21-day date-block bootstrap one-sided p of 0.037 / 0.016. The gradient is
**correctly signed and suggestive, but does not reach significance** and does not
clear the Harvey–Liu–Zhu hurdle. This is the same pseudo-replication the paper
polices everywhere ("the day is the unit of inference"); the HC1 version violated
it and has been retracted from §10.1.

So there is no "cross-sectional result that works while the strategy doesn't."
Under correct inference **all three of the strategy Sharpe, the overnight IC, and
the tick-gradient are marginal-to-null** — a consistent, honest picture. What
*does* survive every control is the 3-minute hidden-execution footprint.

**No result was tuned to significance.** The tests came from the editor's
roadmap; each was run once. The unsupportive ones (R9 volatility
state-dependence, t = −1.2; R11 count-COI, did not compute) are reported in the
paper and here, not dropped — and the R3 correction *reduced* our own headline
number, which is the opposite of p-hacking.

---

## 2. Data inputs (Hoffman2, `/u/scratch/n/nicjia/order-burst-analysis`)

| File | Content |
|---|---|
| `close_all.csv` | daily close price panel, dates × tickers |
| `results/sgd_backtests_oos/*_reg_clop_b1p0_i0p5_debug_trades.csv` | **438 files**, one per name: daily burst **`flow_signal`**, SGD `pred`, `side` — the burst-flow signal matrix |
| `results/research/name_relspread_bps.csv` | per-name close relative half-spread (bps), the tick-constraint proxy |
| `universes/full_500.txt` | the fixed start-of-sample list (500 names; 438 complete the walk-forward — attrition is insufficient-burst-data, not delisting) |

`flow_signal` is the daily gated burst order-imbalance from the walk-forward SGD
pipeline (§§3–6). R3 **consumes** it and re-fits nothing.

---

## 3. Code, and what each file does

### `src_py/m7_reversal_baseline.py` — the single canonical harness
- `load_panels()` — reads `close_all.csv` + the 438 signal files into aligned
  daily matrices `FL, PR, SD, R, cpx`.
- `zscore(df)` — cross-sectional z per day, clipped [−4, 4].
- `strat_returns(...)` — the reversal **strategy**: dollar-neutral, unit-gross,
  `H=20` overlapping hold, position `−sign(z)` shifted one day, 1 bp/turnover
  cost → the daily **net** series (Sharpe ≈ 0.8 OOS).
- `build_signals(...)` — produces `flow_orth_ret` (burst flow residualized
  cross-sectionally each day on the 1/5/20-day lagged-return z-scores).

### `src_py/referee_hardening.py` — the roadmap tests
- `per_name_reversal(...)` — per-name mean daily reversal return in bps, **gross
  of costs** (`position × return`, `position = −sign(z(flow)).shift(1)`).
- `r3_asymmetry(...)` — the **naive** cross-sectional regression (HC1). Reports
  t = −3.81 (log price) / +2.48 (spread) and quintile-gap bootstraps resampled by
  name. **These SEs ignore the shared time dimension — do not cite them.** Kept
  only so the correction is transparent.
- `r3_daylevel(..., skip=1|2)` — **the correct inference.** Fama–MacBeth daily
  cross-sectional slopes with Newey–West errors and a 21-day date-block bootstrap.
  Raw: **t = −1.57**; `skip=2` (skip-day, bounce-killed): **t = −1.69**;
  orthogonalized flow: **t = −2.01**. Daily quintile-gap NW t = +1.4 (raw) / +1.7
  (orthogonalized).
- `r3_orthogonalized(...)` — the HC1 bounce control (t = −3.61); superseded by the
  day-level `skip`/orthogonalized runs above.
- `r5 / r9 / r11` — Romano–Wolf (skipped: config grid not in the needed format),
  Nagel state-dependence (**t = −1.2, unsupportive**), count-COI pricing (**did not
  compute**). All reported.

### Section-specific scripts (unchanged)
`hidden_full.py` (474-name hidden markout + placebo + tick-rule), `poisson_test.py`
(inhomogeneous Poisson), `m7_signed_volume.py`, `m10_sign_audit.py`.

---

## 4. Exact commands to reproduce (Hoffman2)

```bash
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source .venv/bin/activate
export OMP_NUM_THREADS=1
python src_py/referee_hardening.py     # R3 (naive + day-level + skip-day) / R5 / R9 / R11
# log: results/research/referee_hardening_v2.log
```

Strategy Sharpe / baselines / Deflated Sharpe: `python src_py/r2_reversal.py`.

---

## 5. The cross-sectional-vs-time-series reconciliation, done correctly

An earlier draft of this note argued the gross gradient is "largely eaten by the
spread," which is *why* the net Sharpe is only ≈ 0.8. **That explanation is wrong
by the paper's own numbers** and has been dropped: turnover is 0.058/day (§10.2),
so even an 8 bps spread costs ≈ 0.5 bps/day against a ≈ 6 bps/day gross mean — a
~7% drag, and re-costing with per-name effective half-spreads moves the OOS Sharpe
only 0.79 → 0.77 (§10.5). The strategy's marginality is a **statistical-power /
volatility** phenomenon, not a cost one.

The legitimate distinction between a gradient and a portfolio Sharpe is real —
uniform day shocks cancel out of a cross-sectional slope but add variance to a
portfolio's time series — but it is *not* load-bearing here, because once the
gradient is estimated with day-level inference it is **also** marginal (t ≈ −1.6
to −2.0). There is no significant-vs-insignificant puzzle to reconcile; there is
one coherent, underpowered pattern.

**Caveats that remain (honestly stated):**
1. **Bounce: tested, not merely asserted.** The skip-day variant (enter t+2, by
   which point mechanical bid–ask bounce has reverted) leaves the day-level slope
   essentially unchanged (−2.0, t = −1.7), and orthogonalizing flow to lagged
   returns if anything strengthens it (t = −2.0). So the concentration is
   **inconsistent with a generic bid–ask-bounce / cheap-stock-reversal
   explanation** — but "inconsistent with", not "ruled out": a heavy day-*t* sell
   burst can position the close toward the bid without a large day-*t* return, a
   flow-specific bounce channel that only a closing-midquote recompute would fully
   exclude (flagged, not yet run).
2. **Code written for this review round**, not independently re-implemented; it
   reuses the same harness as the paper's §10 strategy numbers, so it is at least
   internally consistent.

---

## 6. What did *not* change

The overnight strategy is still a null (Sharpe ≈ 0.8, fails DSR and HLZ); the
`D_b` look-ahead correction still leaves the overnight result null (−0.28 →
−0.20); the 3-minute hidden-execution footprint (+1.6 bps, t = 3.4, 95% placebo
survival) still survives every control. The R3 correction makes the tick-regime
asymmetry a *suggestive, correctly-signed, bounce-robust but statistically
underpowered* pattern — which is exactly how §10.1 now states it.
