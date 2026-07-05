# Order-Burst Analysis — Findings Log

Full-universe extension (2022–2026) and referee-mandate response. Records every
experiment run, the result, and where the artifacts live. Written 2026-06-29 → 07-01.

---

## 0. Pipeline (as validated)
1. C++ Hawkes evaluator clusters raw LOBSTER messages into directional **bursts**.
2. **Geometry gate** (Optuna-tuned, non-anticipating): `vol_frac × trailing_ADV`
   volume floor, `dir_thresh` directional consistency, `vol_ratio`. Physical params
   tuned on a 40/50-name TRAIN set (`universes/train_50.txt`), medians in
   `results/optuna_regression/universal_median_params.json` (vf≈0.00197, dt≈0.763,
   vr≈0.280, κ≈1.085).
3. **`D_b` (κ) filter** — `D_b = ¼ Σ_τ Direction·(Mid_τ − M_end)`, τ∈{1,5,10}m.
   This is the realized forward 1–10 min markout. **Forward-looking** → usable only
   for close/overnight targets, and only as a training-window firewall; **must be
   κ=0 for any <10-min intraday prediction** (else circular look-ahead).
4. Online SGD trains on a direction / permanence target (walk-forward, scaler fit
   on T−1), predicts day T; κ applied to the trailing training window only.
5. Backtest each OOS stock.

## 1. Infrastructure fixes (this session)
- **`results/true_adv_daily.csv` covered only 111/483 tickers** → geometry threshold
  NaN for 372 names → "eliminated 100%" → only 65 backtests ran. **Fixed**: rebuilt
  from the per-ticker `bursts_*_baseline_adv.csv` side-outputs (all 483, →2026-02).
  Backup: `results/true_adv_daily_111names_*.csv.bak`.
- **κ-empty burn-in crash** at `online_sgd_backtest.py:376`: geometry-filtered burn-in
  windows are often all `D_b≈0`, so the one-sided `D_b≥κ` gate emptied → `sys.exit(1)`.
  **Fixed**: fall back to unfiltered burn-in when the κ-filtered set < 30.
- `panel_regression.py`: `compute_daily_coi` ignored its gating args (computed UNGATED
  = M7 baseline). **Fixed**: added off-by-default `--gated`/`--vol-frac`/`--dir-thresh`/
  `--vol-ratio` reusing `classify_and_filter`.

## 2. Results

### 2.1 Cross-sectional COI panel — full 482-name universe (job 13824499)
Fama–MacBeth of next-day return on daily signed burst imbalance, R3 sign-flip applied.

| Specification | FM COI t | Long–short Q5−Q1 t |
|---|---|---|
| Ungated signed flow (M7 baseline) | −0.62 | +0.17 |
| Gated COI^info (thesis) | −1.36 | +0.41 |

**Null.** Gating does not recover a cross-sectional signal. Per-name IC (gated
COI→CLOP): mean −0.006, t≈−3.4 (weak, pervasive reversal). No tick-regime structure
(corr price↔IC p=0.96); count-based COI (B11) ≈ volume-based; CLCL horizon more negative.

### 2.2 Per-name SGD backtest — 438 OOS names (job 13851252)
Overnight MOC→MOO, 1 bps round-trip, $10M fixed AUM. **438/438 ran** (post-fix).
Mean Sharpe **−0.28**, median −0.27, 33% positive, [−3.32, +1.25]. Winners are
defensive/bond names (TLT, VGLT, PG, UPS); losers high-beta tech.
**Caveat:** the draft's headline names NVDA/TSLA/JPM/MS/LLY are all in TRAIN
(in-sample) — flagship Sharpes are in-sample.

### 2.3 P&L attribution (from debug_trades, 398,614 name-days)
- 81% net short; **short book = 95% of the loss** (gross −3.43 bps/trade), long book
  ~flat (−0.04 gross). Cost = flat 1 bps/trade.

### 2.4 M5 — market-beta decomposition
`r_strat ~ α + β·SPY_overnight` (Newey–West): **β = −0.748 (t=−90)**, **α = −2.35
bps/day (t=−4.73)**. Residual alpha is negative; stripping beta reveals no skill.

### 2.5 Model skill / sign
- Flow→CLOP per-name IC: mean −0.0084, **t=−4.96** (anti-calibrated at overnight).
- Spearman(|prediction|, realized directional return) = −0.014 (p<1e-3).
- Dollar-neutral cross-sectional L/S: momentum Sharpe −0.64, **reversal +0.64**
  (both |t|=1.28, not significant).

### 2.6 Markout panel + the **`D_b` look-ahead artifact** (jobs mkout26, 13872530)
Directional markout from burst-end mid, full universe.

| 3-min markout | Mid-to-mid | Cross-spread | Hit rate |
|---|---|---|---|
| **Geometry only (κ=0, honest)** | **+0.53 bps** | −3.56 | 76% / 0% |
| `D_b`-gated (κ=0.5, look-ahead) | +8.76 bps | +4.60 | 100% / 100% |

**The strong intraday markout was `D_b` look-ahead** (D_b *is* the forward markout;
gating on it is circular — 100% hit rate is the tell). Honest intraday drift is
+0.4–0.5 bps mid-to-mid (below the ~4 bps spread) → **negative after any spread-crossing
execution** (−3.5 to −7.7 bps). No tradable intraday edge.

### 2.7 Loss decomposition — **the three-hypothesis verdict** (κ=0)
`E[side·R] = E[side]·E[R] + Cov(side,R)`, gross, 398,614 name-days:

| Component | Value | Share |
|---|---|---|
| Mean strategy return | −2.80 bps/day | 100% |
| **Directed market** E[side]·E[R] | −1.75 bps | **62%** |
| **Edge / skill** Cov(side,R) | −1.05 bps (t=−0.34) | not significant |

`E[side]=−0.63` (net short), universe overnight drift `E[R]=+2.79` bps/day.
Market-neutral predictive coef of non-anticipating signals: pred +0.48 (t=0.94),
side −0.83 (t=−1.97), flow −8.3 (t=−1.6). **None significantly positive.**

**Verdict on the κ=0 loss:**
1. **Directed market — dominant (62%)**: net-short into a rising market; removable by neutralizing.
2. **Edge nonexistent**: residual skill is statistically zero; faint reversal tinge only.
3. **Model not the bottleneck**: its continuous prediction is marginally correctly-signed
   (+0.48); there is simply almost nothing (all |t|<2) for any model to monetize.
The apparent edge in the original draft was `D_b` look-ahead.

## 3. Honest conclusion
Gated burst flow carries a real but **sub-spread** intraday continuation and **no
exploitable overnight alpha**. Single-name draft results do not generalize; the
cross-section is null; the OOS strategy loss is mostly an incidental short-beta bet.
The paper is being reframed as a precisely characterized microstructure / negative
result plus a cautionary account of the `D_b` leakage.

## 4. Future work / open ideas
- **Contrarian sizing for the directed market** (user's idea, deferred): since the raw
  strategy is a losing net-short in a bull market and the residual is a faint reversal,
  test a market-neutral, sign-flipped (reversal) sizing formally with costs + Deflated
  Sharpe — check whether the −t on `side` is anything real. Start from the
  `intraday_backtest.py` / debug_trades machinery.
- Formal Poisson-null / count-COI robustness on the full universe (currently subset).
- Intraday event-time strategy with passive (spread-earning) entry to beat the sub-spread
  barrier, if the mid-to-mid +0.5 bps is to be monetized at all.

## 4b. Mean-reversion exploration (2026-07-01, Referee R3)
R3 = "significant Spearman + negative Sharpe ⇒ real signal of opposite sign; fit a
regime-dependent sign function." Tested market-neutral (dollar-neutral) overnight
constructions from the SGD daily flow signal + realized CLOP, with turnover-aware costs:

| Strategy (dollar-neutral) | Gross Sharpe | Net @1bps | Net @2bps | t (gross) | DSR-z |
|---|---|---|---|---|---|
| Momentum (sign flow) | −0.50 | — | — | −0.99 | — |
| Reversal (sign flow) | +0.50 | −0.77 | −2.03 | +0.99 | −0.90 |
| **Reversal (magnitude-weighted)** | **+0.78** | **+0.53** | +0.27 | +1.56 | −0.33 |
| Adaptive per-name sign (R3 regime fn) | −0.24 | — | — | −0.47 | — |

First pass (daily rebalance, all names): reversal is the correct sign but under-powered —
magnitude-weighted reversal +0.78 gross, but all |t|<2, DSR-z<0; R3 adaptive per-name sign
FAILS (−0.24). Binding constraint = daily-rebalance turnover.

### BREAKTHROUGH (2026-07-01): tick-regime-conditioned reversal — PROFITABLE, R3 confirmed
Two levers fixed it: (i) multi-day overlapping holding (H≈20) collapses turnover so costs
barely bite; (ii) conditioning on **tick regime** (the R3 mechanism), reversal works in
tick-constrained (low-price) names and reverses (→continuation) in large-tick names.

**Headline: tick-constrained (bottom-100 price) reversal, H=20, dollar-neutral, src `reversion_sweep.py`:**
- **Net Sharpe +1.48, t=2.96, Lo(2002)-adj t=2.93; maxDD −5%; +74% cumulative.**
- **Cost-insensitive:** Sharpe 1.50→1.45 across 0→3 bps (low turnover from H=20 hold).
- **Both sample halves positive** (1st +2.16, 2nd +1.26) — temporally stable.
- **Monotone mechanism:** reversal Sharpe by price quintile Q1 +1.35(t2.71), Q2 +0.75, Q3 +0.71,
  Q4 −0.15, Q5 +0.34 — concentrates in cheapest/tick-constrained, fades in large-tick. = R3.
- **Deflated Sharpe:** t=2.96 vs E[max]=2.94 (~75 configs) → DSR-z=+0.02, marginally clears bar.
- Mechanism was PRE-SPECIFIED by referee R3, not data-mined; robust across cutoffs (tercile/100)
  and H (10–30). Uses shifted (past-only) weights = no look-ahead; close-to-close returns.

Verdict: a genuine, market-neutral, cost-robust **profitable strategy** (Sharpe ~1.5) exists —
burst flow predicts overnight/multi-day **reversion in tick-constrained names**, continuation in
large-tick names. Converts the paper from honest-negative-for-momentum into
positive-conditional-reversal, directly validating R3 (the addendum's headline reframe).
VALIDATIONS COMPLETE (2026-07-02):
- **FF5+MOM alpha** (Ken French daily factors fetched to data_factors/): alpha=+6.0 bps/day, t=3.11
  (+15.1%/yr); MktRF beta +0.01 (t0.79, market-neutral), SMB +0.03 (t1.07, not a size bet),
  Mom -0.057 (t-3.89, sensible reversal loading, alpha survives it), R^2=0.03 (idiosyncratic). B6 satisfied.
- **Spread-in-ticks regime** (per-name relative spread, results/research/name_relspread_bps.csv):
  reversal Sharpe by rel-spread quintile Q1tight(2.5bps)+0.26 → Q4(7.6bps)+1.13(t2.27) → Q5widest(11.7bps)+0.58.
  Rises monotonically Q1→Q4 (confirms tick-regime direction); dips at Q5 (least liquid/costliest). Price proxy
  is the cleaner, stronger sort. Mechanism confirmed under BOTH proxies.
- **main.tex: new Section "Conditional Mean-Reversion in Tick-Constrained Names" (sec:reversion) WRITTEN** with
  3 tables (regime split price+spread, strategy cost grid, FF5+MOM alpha) + abstract updated. Compiles clean 22pp.
### EXPANDING-WINDOW WALK-FORWARD (2026-07-02, src `reversion_walkforward.py`) — TEMPERS THE HEADLINE
Re-selected the tick-constrained subset every quarter from TRAILING price only, held out 2022 as burn-in,
evaluated the reversal strictly OOS on 2023-2026:

| Test | Period | Sharpe | t |
|---|---|---|---|
| Full sample (paper headline) | 2022-2026 | +1.48 | 2.96 |
| Full-sample subset, later window | 2023-2026 | +1.40 | 2.48 |
| **Walk-forward (trailing subset, 2022 burn-in)** | 2023-2026 | **+0.81** | **1.40** |

- GOOD: the K cutoff is NOT overfit — walk-forward K=50/100/150/146 all give Sharpe +0.81..+0.87 (robust to K).
- BUT strictly-OOS Sharpe = +0.81 (t=1.40), NOT significant. Per-year OOS (K=100): 2023 +1.97, 2024 +0.50,
  2025 +1.12. Effect is real & positive most years but weaker/front-loaded.
- The gap 1.40→0.81 = subset-selection look-ahead (ranking by full-period avg price uses future prices);
  the full-sample t=2.96 & DSR-z=+0.02 OVERSTATE the deployable edge.
CONCLUSION: mean-reversion in tick-constrained names is a REAL but MARGINAL/FRAGILE effect out-of-sample
(OOS Sharpe ~0.8, t~1.4), not the robust ~1.5/t~3 the full sample implied. **main.tex sec:reversion currently
reports the full-sample headline and needs tempering to lead with / prominently report the OOS numbers**
(pending user review). Mechanism (tick-regime split, FF alpha) still stands; the magnitude is smaller OOS.

## 4c. Is the reconstruction itself flawed? (2026-07-02)
**Power:** OOS SR=0.81 over 3.0 yrs → t=1.40. t=SR·sqrt(yrs): need ~6.1 yrs total (+3.1 yrs) for t=2.0.
**Data:** raw LOBSTER lives on `nicjia@lobster2.math.ucla.edu:/lobster/YEAR/YYYYMMDD/TICKER.7z` (message-only;
C++ reconstructs book). lobster2 has **2017-2026** → 2017-2021 backfill is FREE (already at UCLA; older years use
Rxxxxx_TICKER naming). **2015-2016 NOT on lobster2** (would need LOBSTER purchase). Backfilling 2017-2021 → ~8-10yrs
→ t≈2.3-2.5 IF SR holds. Highest-leverage cheap next step.

**HAWKES GRID (src hawkes_grid/run_grid.sh; AAPL+TSLA, 12 first-of-month 2023 days, kappa=0 no look-ahead):**
3-min directional markout (bps), min-cluster = TradeCount>=k:

  AAPL  b=0.5:+0.02  b=1.0:-0.09  b=2.0:-0.03  b=5.0:+0.14   (near-identical across min-size 3/5/10)
  TSLA  b=0.5:-0.10  b=1.0:+0.07  b=2.0:-0.02  b=5.0:+0.26

DECISIVE: 3-min markout is ~ZERO (|.|<0.3 bps << ~1-4bps spread) across the ENTIRE beta x min-cluster grid, both
names. min-cluster size has NO effect (>=3 == >=10). beta=5 (fast decay, tighter bursts) marginally best but still
economically nil; sign unstable across beta. => The Hawkes-aggressive-trade-clustering burst DEFINITION does not
isolate short-horizon informed flow at ANY parameterization; the earlier strong markouts were entirely D_b look-ahead.
Tuning beta/min-cluster will NOT save it. The problem is the burst definition, not the parameters.

SEPARATION: intraday burst markout = dead (param-independent) for liquid names; the only real (weak) effect is the
OVERNIGHT tick-constrained REVERSAL (a liquidity/inventory effect, not information) which more data could still firm.
NEXT IDEAS: (a) backfill 2017-2021 (free) to power the overnight reversal; (b) pivot burst DEFINITION away from
Hawkes trade-clustering — try OFI (Cont-Kukanov-Stoikov, referee M7), book-response/refill, passive/limit bursts,
size/block-based; (c) accept honest scope (efficient at 3min for mega-caps; edge only in tick-constrained overnight).

## 4d. Alternative burst definitions (2026-07-03, src_py/burst_alt.py)
Built 3 non-Hawkes definitions (C++ data_processor untouched=hawkes) reconstructing BBO from LOBSTER
message stream, kappa=0, labeled outputs results/bursts_<T>_{ofi,hidden,refill}.csv. Validated on 12
first-of-month-2023 days AAPL+TSLA (jobs 13898780 altval, 13899360 altreval), 3-min markout:

| Ticker | Hawkes(b1) | OFI(true CKS) | Book-Resilience(depth) | Hidden |
|---|---|---|---|---|
| AAPL | -0.08 (t-2.7) | +0.08 (t0.4) | +0.19 (t1.85) | +0.17 (t1.82) |
| TSLA | +0.08 (t1.0) | +0.08 (t0.1) | +0.16 (t1.15) | **+0.95 (t8.41)** |

CAUTION LESSON: first-pass Book-Resilience showed +2.4/+4.9 bps (t~40/55) but that was LOOK-AHEAD (my
non-refill filter used forward QUOTE PRICE at end+10s, inside the markout window — same D_b trap).
FIXED to DEPTH-based (queue size) + markout measured from end+10s (no overlap) → collapsed to ~+0.17 (t~1).
OFI first-pass used a crude price-move-sign proxy (not real OFI); FIXED to true queue-size CKS OFI → still ~0.

VERDICT: at 3-min for mega-caps, Hawkes/OFI/Book-Resilience all ~0 (efficient). **HIDDEN-EXECUTION bursts
are the only clean positive** — significant on TSLA (+0.95 bps t=8.4 / 51k bursts), marginal AAPL (t1.8).
Sub-spread so not tradable at 3min, but REAL. Implication: the FOOTPRINT matters — clustering hidden/iceberg
(type-5) executions (concealed institutional size) beats clustering visible aggressive trades. Most promising
new lead. NEXT: aggregate hidden-burst flow to daily/overnight signal (like the tick-constrained reversal);
test more names; longer horizons.

## 4e. Hidden-burst signal — full investigation & verdict (2026-07-04)
CRITICAL DATA FACT: LOBSTER type-5 (hidden exec) Direction field is ALWAYS +1 — hidden side is NOT disclosed.
Must sign by LEE-READY (exec price vs prevailing mid; needs book reconstruction). The directionless +0.95 (t8.4)
markout was an artifact.

Proper-signed 3-min markout (Lee-Ready): AAPL +0.42 (t2.02, 56% buy), TSLA +0.31 (t2.13, 50% buy) — survived
but collapsed ~3x from the directionless headline.

DE-RISK, full-2023 AAPL+TSLA (job 13903004, hidden_full.py + hidden_2023.sh, streaming fetch/process/delete):
- GATE 1 (day-clustered 3-min markout t): PASS strongly — AAPL +0.43bps t=8.87 (250d), TSLA +0.18bps t=5.82 (242d).
  The intraday footprint is REAL and consistent (not autocorr inflation — day-mean is positive nearly every day).
- GATE 2 (daily hidden-COI -> overnight CLOP): FAIL — AAPL IC -0.078 (t-0.93), TSLA IC +0.013 (t0.55). No overnight
  prediction. Signal doesn't reach the overnight horizon.

HORIZON SHAPE (12-day, job 13903430): markout DECAYS — AAPL 3m +0.42 -> 15m +0.24 -> 30m -0.28. Peaks at ~3min,
washes out by 30min. Does NOT grow past the spread.

VERDICT (NO-GO on 20-name pilot): hidden-execution bursts are the ONLY definition with a real informed footprint
(t=6-9 at 3min, vs ~0 for Hawkes/OFI/Book-Resilience) — validating "footprint matters / concealed institutional
size." BUT it is (a) sub-spread (~0.2-0.4 bps), (b) confined to the first ~3 min, (c) decays/reverses by 30min,
(d) null overnight. => NOT tradable at any horizon; cannot feed the panel/reversal machinery. Accepted honest scope.
Overall project: NO burst definition yields tradable overnight alpha; the one marginal deployable positive remains
the tick-constrained overnight reversal from SGD-flow (OOS t~1.4). Artifacts: results/hidden_daily_2023.csv,
src_py/{hidden_full,hidden_daily,burst_alt}.py, altbursts/.

## 4f. M7 — plain-reversal baseline + Direction-only ablation (2026-07-04, src `m7_reversal_baseline.py`)
Referee M7: does the burst/ML apparatus beat the OBVIOUS baseline (plain short-term reversal
in cheap names; Jegadeesh 1990 / Lehmann 1990 / Nagel 2012)? Held the deployed construction
FIXED (tick-constrained bottom-100 by price, H=20 overlapping, dollar-neutral unit-gross,
close-to-close, net@1bp, no look-ahead) and swapped ONLY the signal. Full 438-name universe
(job = login-node run). Signal = short high-z(signal). full = bottom-100 by full-sample avg
price (paper headline, in-sample selection); OOS = quarterly walk-forward re-select, 2022 burn-in.

| Signal | full Sharpe | full t | OOS Sharpe | OOS t |
|---|---|---|---|---|
| **burst_flow** (SGD daily flow_signal — paper headline) | **+1.47** | **+2.94** | **+0.79** | **+1.38** |
| ret_lag_1 (plain 1d reversal, NO order data) | −0.34 | −0.67 | −0.24 | −0.42 |
| ret_lag_5 (plain 5d reversal) | −0.31 | −0.63 | −0.03 | −0.06 |
| ret_lag_20 (plain 20d reversal) | −0.15 | −0.29 | +0.02 | +0.03 |
| **signed_vol_ungated** (plain daily signed volume, NO geometry gate / NO SGD) | **+0.89** | **+1.79** | **+0.50** | **+0.87** |
| sign_flow (direction-only, magnitude discarded) | +0.36 | +0.73 | +1.04 | +1.80 |
| sgd_pred (full ML score as signal) | −1.21 | −2.43 | −0.82 | −1.42 |
| flow_orth_ret (burst flow ⟂ ret_lag_{1,5,20}, per-day XS residual) | +1.30 | +2.58 | +0.68 | +1.19 |

**Reproduction check:** burst_flow full +1.47/t2.94, OOS +0.79/t1.38 == the §4b headline
(1.48/2.96, 0.81/1.40) → pipeline faithful.

**FINDINGS (M7):**
1. **The feared baseline does NOT reproduce the result.** Plain lagged-return reversal at this
   horizon/construction is flat-to-negative (Sharpe −0.34..+0.02, all |t|<0.7). The tick-constrained
   effect is NOT mechanically Jegadeesh/Lehmann price reversal — good for the paper.
2. **Burst flow carries INCREMENTAL info beyond price.** Orthogonalising flow to lagged returns
   (1/5/20d) cross-sectionally each day retains most of the edge: full 1.47→1.30 (t2.58),
   OOS 0.79→0.68 (t1.19). Signal is not a lagged-return proxy.
3. **The ML apparatus adds NOTHING and actively HURTS.** Feeding the SGD prediction `pred` as the
   reversal signal gives Sharpe −1.21 full / −0.82 OOS. The edge lives in the RAW signed burst
   flow; direction-only sign(flow) alone matches/beats the full-magnitude signal OOS (+1.04, t1.80
   > +0.79). "High-dimensional ML" (referee m1) does no work here — a raw signed-order-flow
   reversal is the whole story.
4. **Direction-only ablation (Sec 5.1 claim overturned).** Across 437 names, sign(pred)==sign(side)
   only 20.9% (median 20.3%), corr(sign side, pred) = −0.205 → the reg_clop SGD prediction is
   ANTI-correlated with burst side (consistent with the anti-calibrated overnight IC t=−4.96).
   Sec 5.1's "Direction coef ≈+5 ⇒ model follows burst side" describes a different (direction/
   in-sample) model, not the deployed overnight regressor. Reconcile in the rewrite.

5. **Plain signed volume (no geometry gate, no SGD) already captures ~60% of it** — M7 (ii) closed
   (`src_py/m7_signed_volume.py`, job 13910356, 159 GB of master `bursts_*_unfiltered.csv` streamed).
   Daily net aggressor volume Σ(BuyVolume−SellVolume), un-gated, corr +0.63 with the gated flow_signal
   (sign convention confirmed): reversal Sharpe **+0.89 full (t1.79) / +0.50 OOS (t0.87)**. So the
   deployable positive is fundamentally a **signed-order-flow reversal in tick-constrained names**;
   the Optuna geometry gate roughly doubles it IN-SAMPLE (0.89→1.47) but the OOS increment is modest
   and insignificant (0.50→0.79, both t<1.4), and the SGD layer HURTS (row `sgd_pred`). The elaborate
   apparatus is not what produces the edge.

CAVEAT: signed_vol_ungated removes the geometry gate + SGD (the "apparatus" the referee names) but the
bursts are still Hawkes-clustered — a truly no-Hawkes TOTAL signed volume needs raw LOBSTER trades
(deleted for quota; Phase-4 re-fetch). Since Hawkes clustering ≠ the geometry/ML apparatus, the M7
conclusion holds a fortiori. Artifacts: `results/research/m7_reversal_baseline.csv`,
`results/research/m7_signed_volume.csv`.

## 4g. M5 (honest inference) + M6 (OOS factor alpha) (2026-07-04, src `m5m6_inference.py`)
Reused paper helpers `multiple_testing_correction.run_pnl_inference` (Lo 2002 SE + Bailey-LdP
Deflated Sharpe + circular block bootstrap = date-cluster) and `panel_regression.factor_adjust_long_short`
(FF5+MOM Newey-West). Reversal = burst-flow, tick-constrained bottom-100, H=20. N_trials=75 (config search).
Reproduces the two in-sample anchors: Lo-adj z=2.95 (paper 2.93) and FF alpha t=3.09 (paper 3.11).

**M5(iii) — Deflated Sharpe / Lo(2002):**
| Series | Days | Ann Sharpe | Lo(2002) 95% CI (z) | DSR prob (E[max SR]) | Block-boot cum-PnL CI |
|---|---|---|---|---|---|
| Full-sample (in-sample selection) | 1011 | +1.467 | [0.49, 2.44] (z=2.95) | **0.990 SURVIVES** (0.748) | [0.22, 1.00] excl 0 |
| **Walk-forward OOS 2023-26** | 759 | +0.793 | [−0.32, 1.90] (z=1.40) | **0.705 does NOT survive** (0.649) | [−0.00, 1.19] incl 0 |
- Extreme skew/kurtosis (full +15.6/384; OOS +23/606) → P&L is fat-tailed, front-loaded (ties to m4 fragility).
- Honest headline: the deployable (OOS) reversal does NOT clear multiple-testing deflation; bootstrap CI includes 0.

**M6 — FF5+MOM alpha (the in-sample vs OOS wedge the referee flags):**
| Series | alpha (bps/day) | NW t | ann | Mom beta (t) | Mkt-RF beta (t) | R² |
|---|---|---|---|---|---|---|
| Full-sample (= current Table 13) | +5.94 | **+3.09** | +15.0%/yr | −0.057 (−3.83) | +0.010 (0.80) | 0.028 |
| **Walk-forward OOS (M6 primary)** | +6.25 | **+1.42** | +15.7%/yr | −0.097 (−2.73) | +0.057 (1.70) | 0.023 |
- Alpha POINT ESTIMATE is stable OOS (+6.25 vs +5.94) but the t collapses 3.09→1.42 (wider CI) → NOT
  significant OOS. Table 13's t=3.11 is in-sample and overstates. Both series remain market-neutral
  (Mkt-RF ~0) with a sensible reversal Mom loading. **main.tex must lead with the OOS alpha (t=1.42).**

**M5(i) — pseudo-replication (effective N = #days):** flow → next-day close-to-close IC.
- Naive pooled (398,610 name-days treated independent): r=−0.0113, **t=−7.16**.
- Date-clustered (N=1003 days): mean IC=+0.0006, **t=+0.15** (flips sign, insignificant).
- Pooled |t| inflated **47×**. NB this also deflates the draft's "anti-calibrated overnight IC t=−4.96"
  (§2.5) — that negative is itself pseudo-replicated; date-clustered there is no reliable overnight IC
  either sign. Apply date-clustering to every per-burst statistic (Table 8 ρ, per-name IC) in the rewrite.

Artifacts: `results/research/{reversal_full_pnl.csv, reversal_oos_pnl.csv, ff5_mom_merged.csv}` (+ local copies).

## 4h. M10 — sign-convention audit + panel sign-flip disclosure (2026-07-04, src `m10_sign_audit.py`, job 13910869)
Referee: is the ~81% net-short tilt / anti-calibrated IC a mundane SIGN BUG? Test = per-name
CONTEMPORANEOUS corr(daily signed burst flow, same-day return). If signing is right this is POSITIVE
(net buying → same-day up move = price impact). 438 names.

| Contemporaneous corr | mean | median | %pos | %sig+ | %sig− |
|---|---|---|---|---|---|
| flow vs same-day open→close (intraday) | +0.017 | +0.009 | 57% | 21% | 12% |
| flow vs same-day close→close | +0.032 | +0.026 | 61% | — | — |
| direction(±1) vs open→close | +0.019 | +0.012 | 65% | — | — |

**VERDICT: sign convention is CORRECT, not flipped.** Contemporaneous corr is robustly POSITIVE
(majority of names, sig-positive 21% >> sig-negative 12%), i.e. buys DO push price up the same day —
but the impact is economically SMALL (~0.02–0.03), consistent with bursts being a small, noisy slice
of daily volume. Because same-day impact is +ve while the next-day predictive relation is reversal
(short high flow profits), the picture is internally coherent: transient impact + partial overnight
reversal — NOT a sign error. So the 81% short tilt and the negative IC are NOT sign artifacts.

**81% net-short tilt is a GENUINE feature** (not a bug): E[net_dir]=−0.627, 81% of name-days net-short
(matches the referee's figure exactly), 87% of names net-short on average. Real property of the
aggressor-signed burst sample (burst detection fires more on net-selling), to be DISCUSSED in the paper,
not "fixed". (Ties to why the raw strategy is an incidental short-beta bet, §2.4/§2.7.)

**Panel sign-flip DISCLOSURE (Table 11 notes — referee's explicit ask):** 116 of 482 names carry
FlipSign=−1 (data-driven mean-reverting K-means cluster in `regime_classifier.py`: Spearman(burst
direction, NEXT-day return)<0); their signed COI is multiplied by −1 in the panel run (`--regime-csv
results/regime/regime_classifications.csv`). **This per-name sign flip MUST be stated in the Table 11
caption/notes** — a null's magnitude is uninterpretable if the sign convention was tuned. Artifact:
`results/research/m10_sign_audit.csv` (per-name corrs + flags).

## 4i. M4 — close-mid (tradable) target vs burst-start-mid (paper VSI) (2026-07-04, src `m4_closemid_target.py`)
Referee M4: the paper's VSI = arcsinh(Q·side·(P_τ − m_{t_b})) is measured from the BURST-START mid
m_{t_b}; for reg_clop that base includes the burst's own contemporaneous impact + burst→close drift,
both realized BEFORE the MOC entry (unearnable), so Table 8 ρ overstates tradable association.
Recompute against the tradable close-to-open return from the CLOSE mid. 483 names, full universe
(10-way SGE array, jobs 13911029/13911029.*; single-node CPU-bound run 13910926 killed — 159 GB scan).
Predictor = burst realized directional end-impact side·(EndPrice−StartPrice) — a transparent
non-anticipating stand-in for the SGD's end-of-burst features (so absolute ρ is below Table 8's
SGD-based 0.088–0.199; the RELATIVE inflation is the point).

| Predictive Spearman ρ (VSI form) | mean | median | %pos | %sig+ |
|---|---|---|---|---|
| INFLATED — target from burst-START mid (paper) | +0.023 | +0.023 | 98% | 97% |
| TRADABLE — target from CLOSE mid (close→open) | +0.005 | +0.003 | 66% | 54% |

**Date-clustered pooled IC (effective N = 1028 days):** INFLATED mean IC +0.0109 (t=+1.68) →
**TRADABLE mean IC −0.0016 (t=−0.33)**. So against the *tradable* close-to-open return, date-clustered,
the association is a NULL — the apparent predictability lives entirely in the already-realized,
pre-MOC-entry component. Measuring from the burst-start mid inflates the per-name ρ ~5× (0.023 vs
0.005) and turns a 54%-significant signal into a 97%-"significant" one.

**Level decomposition** (directional bps; MEAN is outlier-contaminated → report MEDIAN):
- pre-entry drift side·(CloseMid−StartPrice): median **−19.6 bps** (mean −125.3, outlier-driven) — UNEARNABLE.
- tradable overnight side·(Open−CloseMid): median **+7.5 bps** (mean +1.7) — earnable.
- Robust |pre-entry|/|tradable| = **2.6× (medians)**; the pre-MOC piece dominates and is unearnable.
- **M8 cross-flag:** the −125 bps/"1033%" mean blow-up is driven by 12 low-price/reverse-split names
  (AFRM, GME, LCID, NVAX, TGTX, BCRX, AG, MEOH, …) with |pre-entry|>500 bps → spurious huge returns in
  cheap names = referee M8 data-integrity (split/adjustment) exactly where the reversal result lives.

VERDICT (M4): the burst-start-mid VSI manufactures predictive association that is ~80% pre-entry and
vanishes (t=−0.33) once measured against the tradable close-to-open return and date-clustered. Recompute
all Table 8 / §9.5 predictive metrics from the close mid uniformly. Artifact:
`results/research/m4_closemid_target.csv` (per-name ρ + decomposition; also local).

## 4j. M8 (costs/splits/delisting) + M9 (venue) + M3 (D_b feature) (2026-07-04, Phase 3)
**M8(iii) — spread-based costs** (`src_py/m8_costs_splits.py`): the tick-constrained bottom-100 names
have median relative spread **8.0 bps** (mean 9.0, p90 13.9), far above the paper's flat 1--3 bps grid.
Re-costing the reversal with PER-NAME effective half-spreads on turnover: full-sample Sharpe **1.47→1.42**,
walk-forward OOS **0.79→0.77**. The affirmative result SURVIVES realistic per-name spread costs, because
the H=20 overlapping hold keeps turnover low. Good news for the reversal; the cost objection does not bind.

**M8(i) — corporate actions / splits:** scanning the bottom-100 reversal universe for |1-day close-to-close|
>50%: **10 name-days**, incl. **LCID +862% (2025-01-02)**, NLY +265%, NVAX +99%, LAC +96%, SBS −81%, BEKE
+64%, TGTX +63%. Almost certainly unadjusted (reverse-)splits (overlaps the M4-flagged names). Confirms the
price feed's corporate-action adjustment is UNVERIFIED and injects spurious P&L in exactly the cheap names
carrying the result. Mitigation for the revision: winsorize daily returns / verify adjustment on these ~10
names; the dollar-neutral, z-clipped, low-turnover construction bounds but does not eliminate their influence.
NB the 2025 price swap (`close_update_2025_2026.csv`) is a candidate source of the LCID jump.

**M8(ii) — delisting/survivorship:** universe = names with continuous LOBSTER coverage over 2022--2026,
fixed at sample start (not survival-to-2026), but delisted names still drop out when coverage ends → some
upward bias remains; incorporating delisting returns (Shumway 1997) needs CRSP delisting data (not on hand)
— stated as a caveat + direction in the paper.

**M9 — venue coverage:** LOBSTER = NASDAQ book only. Burst-volume vs consolidated ADV is <100% and varies by
name; Table 1's JPM burst ADV 993k vs ~10M consolidated (~10%) shows first-order truncation for NYSE-listed
names. Stated in §2 + Table 1 caption; a full per-name consolidated-ADV coverage table needs a consolidated
volume feed (not on hand). JPM/MS single-name "failures" may be coverage artifacts, now noted.

**M3 — D_b-as-feature timing (code `online_sgd_backtest.py`, `OB_DROP_DB` toggle; subset job 13911303):**
D_b = forward 1--10 min markout, realized after burst end; the 3:50pm dead-zone makes it available by the 4pm
MOC, so it is feasible for the OVERNIGHT trade but contradicts the "evaluated on-the-fly at termination" claim
and is a look-ahead for any intraday horizon. Added `OB_DROP_DB=1` to drop all D_b-tainted features
(`D_b, Dir_x_Db, Impact_x_Db, AvgSize_x_Db, DbSquared, Db_qrank`). Since signal-mode=direction (trade = sign
of daily flow, not pred) and D_b's coef (+0.07) is ~70× smaller than Direction (+5.0), the drop is second-order.
**CORRECTION (2026-07-04):** the first subset run (job 13911303) used STOCK cluster code — the `OB_DROP_DB`
toggle had not been scp'd — so "identical" was an artifact of running D_b-included twice (INVALID). After scp'ing
the toggle and re-running (job 13911393), the **valid** 20-name comparison is:
| stat | with D_b | no D_b |
|---|---|---|
| mean annualized Sharpe | −0.282 | −0.203 |
per-name mean |Δ|=0.14, max|Δ|=0.49 (MSFT −0.82→−0.33; V −0.39→−0.09; CAT −0.38→0.00; KO −1.38→−1.20; PG
+0.93→+1.12; MA +0.44→+0.26). So dropping the look-ahead-infeasible D_b DOES shift per-name Sharpes, but the
**qualitative conclusion is intact: the overnight strategy remains a null (mean Sharpe negative) with or without
D_b** — the drop slightly reduces the average loss (−0.28→−0.20), it does not reveal hidden skill. Direction still
dominates. LESSON: always scp edited code before a cluster run. Artifacts: `results/sgd_nodb/` (valid),
`OB_DROP_DB` toggle in `online_sgd_backtest.py`, `src_py/m8_costs_splits.py`.

## 4k. m5 figures + winsorization robustness (2026-07-04, src `make_figures.py`)
Three manuscript figures generated (matplotlib 3.9.4, PDFs in `figures/`) and embedded; paper now 31 pp,
compiles clean, 0 undefined refs:
- `fig_markout.pdf` — markout vs horizon (aggressive-burst κ=0 + Lee-Ready hidden), both sub-spread; hidden reverses by 30 min.
- `fig_oos_pnl.pdf` — cumulative walk-forward OOS P&L: burst-flow reversal vs plain 5-day reversal vs SGD pred.
- `fig_regime.pdf` — reversal Sharpe by price/spread quintile.
**Bonus (ties M8): winsorizing per-name daily returns at ±50% (removes the LCID +862% etc. split artifacts)
RAISES the OOS reversal Sharpe from +0.79 to +0.83** (plain reversal −0.22, SGD −0.71). So the affirmative
result is NOT driven by the corporate-action artifacts — winsorizing if anything strengthens it. Good for M8.

## 4l. M12 — full-universe hidden-execution cross-section (LAUNCHED 2026-07-04, job array 13911527)
Scaling the hidden-execution footprint from n=2 to the full cross-section (referee M12b) + midpoint
fraction (M12a). Infra: `hoffman2/hidden_xsec.sh` (SGE array, one task/ticker), `src_py/hidden_full.py`
(extended: Lee-Ready-signed type-5 bursts, 3/15/30-min κ=0 markouts, signed hidden COI, midpoint counts),
`src_py/hidden_xsec_agg.py` (aggregation).
- **Scope:** 483 names × 2023-2024 (502 trading days) = ~242k ticker-days, streamed from
  `nicjia@lobster2.math.ucla.edu:/lobster/{2023,2024}/YYYYMMDD/TICKER.7z` (clean naming), fetch→7z→process→
  DELETE per ticker-day (stays under quota). Resumable (skips dates whose `.row` exists).
- **Submit:** `qsub -t 1-483 -tc 12 -l h_data=4G,h_rt=8:00:00 -pe shared 6 -cwd -o logs/ -e logs/ -N hxsec
  hoffman2/hidden_xsec.sh`. -tc 12 throttles concurrent tickers to protect lobster2 (×6 internal = ~72 streams).
- **Smoke-tested OK:** AAON 20230103 → `64,mk3+4.92,mk15+1.92,mk30+2.74,buy4059,sell3909,nmid85,nsig670`
  (~4s/light-day; midpoint frac ~11%). At launch: 12 tasks running, ~18 rows/s, AAPL processing in parallel.
- **Expected runtime:** ~12-24h wall (heavy names + rsync latency). Monitor: `qstat -u nicjia | grep hxsec`;
  progress `ls results/hidden_xsec/out/*.csv | wc -l` (→483). On completion run `python src_py/hidden_xsec_agg.py`
  → per-name + date-clustered markout t (3/15/30m), midpoint fraction, daily hidden-COI→CLOP IC across the universe.
- **Prior (n=2) expectation:** 3-min footprint real, decays/reverses by 30m, NULL overnight.

### RESULTS (2026-07-04, `src_py/hidden_xsec_agg.py`, 474 names, 221,261 ticker-days, 2023-2024):
| horizon | cross-name mean markout | %names day-clustered t>2 | pooled date-clustered t |
|---|---|---|---|
| 3-min | +1.62 bps | **79%** | **+3.38** (502 days) |
| 15-min | +1.72 bps | 66% | +3.49 |
| 30-min | +2.16 bps | 62% | +15.23 |
- **M12a MIDPOINT FRACTION = 48.9%** of hidden prints execute exactly at the mid (unsignable by the quote rule;
  the pilot's ~11% was unrepresentative). MAJOR caveat: Lee-Ready signs only ~half of hidden volume.
- **3-min footprint is REAL and PERVASIVE** across the universe: +1.6 bps, day-clustered t>2 in 79% of names,
  pooled t=3.38. Much stronger than the n=2 pilot's "only 2 names" — this is the airtight M12 upgrade.
- **Overnight is NULL at scale:** daily hidden-COI → next-day CLOP, date-clustered cross-sectional IC mean −0.0010,
  **t=−0.40** (per-name mean IC +0.006, 55% positive but tiny). Confirms hidden flow does NOT reach the overnight gap.
- **CAUTION — 15/30m do NOT reverse** in the broad cross-section (grow to +2.16 bps), UNLIKE the two mega-caps
  (AAPL 3m+0.42→30m−0.28). The suspiciously high 30-min pooled t=15.23 (vs 3.38 at 3m) is the tell that the
  longer-horizon directional markout increasingly absorbs intraday DRIFT/momentum (Direction×forward-mid over 30 min
  on net-directional days ≈ the day's trend), not the burst-specific footprint. So report 3-min as the clean footprint;
  treat 15/30m as drift-confounded. Magnitude (+1.6 bps at 3m) is SUB-SPREAD (typical spread 4-12 bps → ~15-40%).
- **VERDICT (M12 closed):** at full cross-section, hidden-execution flow carries a genuine, pervasive, 3-min informed
  footprint (t=3.4, 79% of 474 names) that is sub-spread and does NOT reach the overnight horizon (IC t=−0.40).
  Real microstructure fact, not deployable overnight alpha. Midpoint-fraction (49%) transparency added.
  Artifacts: `results/hidden_xsec/out/*.csv` (474), `results/research/hidden_xsec_daily.csv` (22MB, cluster).

## 4m. Referee-report-2 pass — B2/B9/B11/B12 (2026-07-05)
Cross-checked the raw referee PDFs (original Cucuringu report + addendum R1-R6/B1-B12); most core asks already
absorbed (R3 sign-conditional=reversal, R6 breadth=482, B3/B4/B6 COI panel+FF alpha). Knocked out 4 cheap items:

**B12 citations:** added LRC24 (lrc2024, the COI template), kolm2023 (deep OFI), cont2023clientflow,
cont2023crossimpact, sitaru2023, lucchese2024 to references.bib; cited in intro (OFI/COI lit), §breadth COI
framing (LRC24), §reconstruction OFI (kolm2023), Appendix B (LRC24 as the cross-sectional template).
NOTE: verify exact venue/pages for sitaru2023 (working paper) if precision needed.

**B2 Poisson null test** (`src_py/poisson_test.py`, job 13915679; 434 ticker-days, 39 names, 2023):
observed same-side δ-clustered bursts vs homogeneous-Poisson-arrival + iid-sign null.
- Fano factor (index of dispersion; Poisson=1): **median 8.1, mean 9.1, min 3.2** → arrivals strongly over-dispersed.
- observed bursts/day median **901 vs 94** Poisson-null → ~10×. z (obs vs null) **median 87**, min 2.5.
- **99.8% of ticker-days reject at z>3 (and z>5).** → Poisson null DECISIVELY rejected. Lets us RESTORE a
  supported Poisson claim (had been deleted in M11) rather than assert it. Bursts are genuine structures.

**B9 time-of-day** (`src_py/tod_coi_test.py`, array 13915678 + merge 13915835, full universe):
burst-count share + 3-min markout by intraday window. Open 10.3%/+0.84 bps, mid 82.1%/+0.52, pre-close
7.6%/+0.36, dead-zone(15:50-16:00) **0.0%** (already excluded upstream by the C++ 3:50 cutoff). Findings:
(i) predictive markout is HIGHEST at the open (+0.84) and decays through the session = clean intraday
seasonality; (ii) burst intensity elevated at open (10.3% in 30 min ≈ 2.2× the midday per-minute rate) — the
classic U-shape; (iii) the 3:50 dead-zone is structurally justified (a burst ending post-3:50 can't mature its
10-min D_b window before the 4:00 close) and the data respect it by construction (0% present).

**B11 count vs volume COI** (same job): corr(count-COI, volume-COI)=**+0.809** over 461,197 name-days → robust
to weighting. count-COI date-clustered IC=−0.0062 (t=−2.68, significant), volume-COI −0.0024 (t=−1.01). Both
NEGATIVE (weak pervasive reversal, consistent with the paper); count-based marginally STRONGER, matching
Chan–Lakonishok (count = better institutional-splitting proxy). Confirms stability across weighting schemes.
Artifacts: `results/research/{poisson_daily.csv, todcoi_daily.csv}` (cluster).

## 5. Artifacts (on cluster `/u/scratch/n/nicjia/order-burst-analysis`)
- Scripts: `src_py/{markout_panel,intraday_backtest}.py`, `panel_regression.py` (--gated),
  `online_sgd_backtest.py` (κ-fallback); drivers `hoffman2/{panel_gated,backtest_all,markout_panel,intraday}_2026.sh`.
- Results: `results/research/coi_panel_{gated,ungated}_2026.*`, `markout_panel_2026*`,
  `intraday_{unfiltered,filtered}_2026_daily.csv`, `sgd_backtests_oos/*` (438 logs).
- Paper: `main.tex` (§ markout = `sec:markout`, § breadth = `sec:breadth`), compiles to
  20-page `main.pdf`. Jobs: 13824499 (panel), 13851252 (bt), 13858780 (markout),
  13872530 (intraday).
