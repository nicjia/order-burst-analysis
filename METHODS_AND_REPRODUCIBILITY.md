# Methods & Reproducibility Note

A short guide for a reviewer who wants to know **exactly what was computed, from
what data, with what code** — and, in particular, **why the tick-regime
asymmetry test is significant (t = −3.81) when the tradable reversal strategy is
not (t ≈ 1.4).** The two are different statistical objects; neither number
changed the other.

---

## 1. The headline question: "why did it suddenly work?"

It did **not** — not at the level of a tradable strategy. Three separate
empirical claims live in the paper, at three different significance levels, and
they have stayed put across rounds:

| Claim | Statistic | Significant? | Changed this round? |
|---|---|---|---|
| Hidden-execution 3-min markout | +1.6 bps, t = 3.4 (pooled, date-clustered) | yes | no |
| Overnight predictability / reversal **strategy** | OOS Sharpe ≈ 0.8, t ≈ 1.4; fails deflation | **no** | no |
| Tick-regime **asymmetry** (R3, added this round) | cross-sectional slope t = −3.81 | yes | new test |

The strategy Sharpe and the overnight IC are **still insignificant.** What is new
is R3, and R3 asks a *different question* with a *different test*:

- **Strategy Sharpe** is a **time-series** test of one portfolio's daily net
  return. Its power is set by the number of trading days (~750 out-of-sample)
  and the return volatility, and it is **net of costs**. Low power → t ≈ 1.4.
- **R3** is a **cross-sectional** test across **434 names**: does the reversal
  effect *concentrate* in tick-constrained (cheap, wide-spread) names? Each name
  is one observation, and the test is a regression slope, **gross** of costs.
  434 observations testing a gradient → t = −3.81.

A signal can produce a strong, real cross-sectional gradient and still not be a
profitable standalone strategy once you pay the spread. That is the paper's
actual claim, and the two results are **consistent**, not contradictory. It is
also why the paper explicitly says the *gradient* clears the Harvey–Liu–Zhu t≈3
hurdle while the *tradable Sharpe* does not.

**No code was iterated to reach significance.** R3 is the test the senior editor
asked for (per-name regression on tick-constraint + monotonicity + a momentum
check in large-tick names). It was written once, run once, and reported as-is.
The two companion tests that came back *unsupportive* (R9 volatility
state-dependence, t = −1.16; R11 count-COI, did not compute) are reported too,
in the run log and the response — not dropped.

---

## 2. Data inputs (all on Hoffman2 at `/u/scratch/n/nicjia/order-burst-analysis`)

| File | Content |
|---|---|
| `close_all.csv` | daily close price panel, dates × tickers |
| `results/sgd_backtests_oos/*_reg_clop_b1p0_i0p5_debug_trades.csv` | **438 files**, one per name: the daily burst **`flow_signal`**, the SGD `pred`, and `side`. This is the burst-flow signal matrix. |
| `results/research/name_relspread_bps.csv` | per-name close relative half-spread (bps), used as the tick-constraint proxy |
| `universes/full_500.txt` | the fixed start-of-sample universe list |

The signal (`flow_signal`) is the daily aggregated, gated burst order-imbalance
per name produced by the walk-forward SGD pipeline (Sections 3–6 of the paper).
R3 does **not** re-fit anything — it consumes the already-computed signal.

---

## 3. Code, and what each file does

### `src_py/m7_reversal_baseline.py` — the single canonical harness
Everything downstream reuses these functions so numbers stay consistent with §10.
- `load_panels()` — reads `close_all.csv` + the 438 signal files into aligned
  daily matrices `FL` (flow), `PR` (pred), `SD` (side), `R` (close-to-close
  returns), `cpx` (price).
- `zscore(df)` — cross-sectional z-score per day, clipped to [−4, 4].
- `strat_returns(Zsig, R, mask)` — the reversal **strategy**: dollar-neutral,
  unit-gross, `H=20`-day overlapping hold, position `= −sign(z)` shifted one day
  (no look-ahead), 1 bp/turnover cost. Returns the daily **net** series whose
  Sharpe is ≈ 0.8 OOS.
- `full_sample_universe` / `walkforward_universe` — the bottom-K-by-price
  (tick-constrained) subset, selected either full-sample or re-selected quarterly
  from trailing price only (the walk-forward discipline).

### `src_py/referee_hardening.py` — the roadmap tests added this round
- `per_name_reversal(Zsig, R, cols)` — **the R3 input.** For each name, the
  single-name signed reversal return is `position × return` where
  `position = −sign(z(flow)).shift(1)`; it returns each name's **mean daily
  reversal return in bps** (gross). This is a per-name measure of *how well the
  reversal works on that name*.
- `r3_asymmetry(...)` — the acceptance-condition test:
  1. regress per-name reversal profitability on `log(price)` and on relative
     spread, across the 434 names with a valid estimate, HC1 robust SEs
     → slopes **t = −3.81 (log price)** and **+2.48 (spread)**;
  2. quintile the names by price/spread and bootstrap the tick-constrained-minus-
     large-tick difference → **+5.1 bps (p = 0.001)** / **+5.6 bps (p = 0.002)**;
  3. run the **momentum** construction in the top-100 large-tick names →
     Sharpe −0.51 (t = −0.87): **no continuation there** (this is why the paper
     retracted "continuation in large-tick names").
- `r3_orthogonalized(...)` — the **bounce control.** Reruns the log-price
  regression on `build_signals()['flow_orth_ret']` (burst flow residualized
  cross-sectionally each day on the 1/5/20-day lagged-return z-scores, i.e. with
  any generic short-horizon reversal component removed). Result: slope **t = −3.61
  (log price), still clears |t| > 3** — the tick-constraint gradient survives
  almost unchanged, so it reflects genuine order-flow information, **not** bid–ask
  bounce. (Spread axis: t = +2.41, same sign, just under the strict 3.0.)
- `r5/r9/r11` — Romano–Wolf (skipped: stored config grid not in the needed
  daily-series format), Nagel state-dependence (t = −1.16, **unsupportive**),
  count-COI pricing (did not compute). All reported, none hidden.

### Other section-specific scripts (unchanged this round)
`hidden_full.py` (474-name hidden-execution markout + placebo + tick-rule),
`poisson_test.py` (inhomogeneous Poisson null), `m7_signed_volume.py`,
`m10_sign_audit.py`. These produce the §8/§9/§11 numbers.

---

## 4. Exact commands to reproduce (on Hoffman2)

```bash
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source .venv/bin/activate
export OMP_NUM_THREADS=1
python src_py/referee_hardening.py        # R3/R5/R9/R11; ~1–2 min
# log saved at results/research/referee_hardening_R3R5R9R11.log
```

The reversal strategy Sharpe, baselines, and Deflated Sharpe come from
`python src_py/r2_reversal.py` (which imports `m7_reversal_baseline`).

---

## 5. Honest caveats a reviewer should weigh on R3

1. **R3 is gross of costs.** The +4 bps/day gross reversal in the cheapest
   quintile is measured before the ~8 bps median close spread in those names.
   This is *why* the tradable net Sharpe is only ≈ 0.8 — the gross gradient is
   real, the net edge is largely eaten by the spread. Consistent, not
   contradictory.
2. **Mechanical bid–ask-bounce component — tested and ruled out.** Cheap,
   illiquid stocks have more bounce, so *any* reversal signal looks better there
   before costs (Nagel 2012). We re-ran the log-price gradient on the
   **lagged-return-orthogonalized** flow (`r3_orthogonalized`), which strips out
   any generic short-horizon reversal: the slope is essentially unchanged,
   **t = −3.61, still clearing |t| > 3**. So the concentration is driven by
   genuine order-flow information, not bounce. (This is the check that was
   previously flagged as "not yet run"; it is now run and reported in §10.1.)
3. **This code was written for this review round** and has not been independently
   re-implemented. The harness (`m7_reversal_baseline.py`) is the same one used
   for the paper's §10 numbers, so R3 is at least internally consistent with the
   published strategy figures.

---

## 6. What did *not* change

The overnight strategy is still a null; its Sharpe is still ≈ 0.8 and still fails
the Deflated-Sharpe and Harvey–Liu–Zhu hurdles; the D_b look-ahead correction
still leaves the overnight result a null (−0.28 → −0.20). R3 adds a
cross-sectional characterization of *where* the (untradable) reversal
concentrates; it does not turn the strategy into a profitable one, and the paper
does not claim it does.
