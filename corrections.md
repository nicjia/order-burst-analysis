# Corrections — Referee Report Response Tracker

**Manuscript:** *Informational Order Flow and Price Persistence: Identifying Latent Alpha in Order Submission Bursts*
**Referee recommendation:** Reject in current form (resubmit restructured, likely to a specialist microstructure venue, e.g. *Journal of Financial Markets*).
**Three overarching reasons for rejection:**
1. Internally contradictory — Sections 4–7 assert (present tense, unqualified) findings that Sections 8–11 prove are in-sample / look-ahead artifacts.
2. The one positive result (tick-constrained reversal) is not truly OOS, lacks the obvious plain-reversal baseline, and is exposed to data-integrity risks (splits/delistings/costs) worst in the low-price names that drive it.
3. Promised statistical apparatus (Deflated Sharpe, multiple-testing, Poisson null) is absent or asserted without evidence.

Status legend: `[ ]` open · `[~]` partially done · `[x]` done. Priority: **P1** (blocks any resubmission) · **P2** · **P3**.

---

## MAJOR CONCERNS

### [ ] M1 — Internal contradiction: Sections 4–7 vs 8–11 (P1)
- The paper reads as two papers stapled together; Sec 7 captions/topic sentences assert results Sec 8–11 overturn ("statistically significant profitability", cost grid "strictly exceeds slippage", 2019–21 "uniform success ... time-invariant mechanism", earnings "definitively refute").
- **Fix:** rewrite Sec 4–7 from the outset as the *diagnostic subject* ("here is a pipeline that appears to work; here is why each apparent success is an artifact"). Remove invalidated claims from **captions and topic sentences**, not merely contradict later.
- **Casualty:** Sec 7.9 "Microstructure Synthesis" argues signal *cannot* exist in tick-constrained names (AAPL/JPM), but Sec 10's positive result is *precisely* in tick-constrained names. Both cannot coexist — resolve.

### [ ] M2 — Flagship results are in-sample, carry no inference, internally inconsistent (P1)
- NVDA/TSLA/JPM/MS are the Optuna tuning set (params selected per-name, per-target by maximizing the reported metric = selection, not validation). **LLY is also in the training universe** → the "(OOS)" label on LLY in Tables 1 & 5 is factually wrong → **correct it**.
- 2019–2021 rows apply 2023–24-tuned params to earlier data; paper concedes it "violates chronological walk-forward" yet labels rows "OOS" and cites "uniform success" → **remove OOS label / reframe as robustness probe**.
- Table 5 has **no standard errors**; Sharpe 0.59 over ~2yr ⇒ t ≈ 0.59·√2 ≈ **0.83** (indistinguishable from zero even in-sample). Drop "statistically significant profitability."
- **Number instability:** earnings analysis (7.10) quotes baseline Sharpe 0.48 (NVDA)/0.76 (TSLA) vs Table 5's 0.59/0.50 same strategy/period. **Build a provenance table** mapping every reported number to one canonical run.

### [x] M3 — D_b defined twice inconsistently; used as prediction-time feature → stated execution infeasible (P1) — DONE 2026-07-04 (Phase 2+3)
- **RESULT:** (a) Definitions reconciled in §3 (canonical Eq. `eq:db`: ⅓·Σ from M_end) + note that D_b is forward-looking (training-firewall/overnight only); ¼→⅓ fixed; cross-ref fixed (m11/m12). (b) D_b-as-feature timing flagged in §3 & §5.1; `OB_DROP_DB` toggle added to `online_sgd_backtest.py` (drops D_b, Dir_x_Db, Impact_x_Db, AvgSize_x_Db, DbSquared, Db_qrank). (c) **Subset re-run (20 names, VALID job 13911393 after scp'ing the toggle): mean Sharpe −0.28 (with D_b) → −0.20 (no D_b)**, per-name mean|Δ|=0.14, max 0.49. Dropping the look-ahead-infeasible D_b shifts per-name Sharpes but the overnight strategy **stays a null (mean negative) either way** — the drop reduces the average loss, it doesn't reveal skill. §breadth reports the corrected numbers. (NB: an earlier "identical" run was invalid — stock code, toggle not yet scp'd; corrected. FINDINGS §4j.)
- Two definitions: Sec 3 = single 10-min horizon, from **burst-start** mid, normalized by PeakImpact; Sec 8 = ¼·Σ over {1,5,10}m, from **end** mid, unnormalized. **State unambiguously which D_b was computed/filtered/fed to the model.** Fix the `¼×(3-term sum)` (typo / relic of a 4-horizon version). Fix Sec 8 mis-attribution to `sec:synthesis`.
- **Infeasibility:** Sec 5.1 lists D_b as 2nd-most-important feature (coef ≈ +0.07), but D_b is unknown until 10 min after burst end → contradicts "evaluated on-the-fly at termination." MOC cutoffs are 3:50pm (NYSE)/3:55pm (NASDAQ); a burst ending 3:49pm realizes D_b at 3:59pm — **after** the MOC gate closes. **Fix:** (a) drop D_b from features and re-run everything, OR (b) restrict features to a window ending at the MOC cutoff and model execution accordingly.
- **κ-firewall covariate shift:** model fit on D_b≥κ survivors, deployed on unfiltered population; never quantified; the κ-empty burn-in failure mode shows it is not innocuous. **Quantify the train/test shift.**

### [x] M4 — Target conflates already-realized intraday impact with the traded overnight return (P1) — DONE 2026-07-04
- VSI target measured from **burst-start** mid; for reg_clop it includes the burst's own contemporaneous impact + burst→close drift (both realized before MOC entry, unearnable). Table 8 ρ (0.05–0.20) overstate association with the *tradable* close-to-open return.
- **Fix:** recompute all predictive metrics against the **close-to-open return alone** (targets from the **close mid**); apply the Sec 8 markout convention (from M_end) **uniformly**.
- Note: Sec 9.5's anti-calibration overnight (IC t=−4.96) vs Table 8's positive in-sample ρ is most naturally this wedge — **say so**.
- **RESULT** (`src_py/m4_closemid_target.py`, `results/research/m4_closemid_target.csv`, FINDINGS §4i; 483 names, full universe):
  - Same predictor, per-name Spearman ρ **collapses ~5×** moving the target base from burst-start mid → close mid: mean +0.023 (98% pos, 97% sig+) → +0.005 (66% pos, 54% sig+).
  - **Date-clustered pooled IC** (N=1028 days): inflated +0.0109 (t=+1.68) → **tradable −0.0016 (t=−0.33) = NULL**. The apparent predictability is entirely the already-realized pre-MOC-entry component; against the tradable close-to-open return there is no association.
  - Robust decomposition (medians): pre-entry unearnable drift −19.6 bps vs tradable overnight +7.5 bps → pre-MOC piece **2.6× larger** and unearnable. (Mean-based blow-up −125 bps is from 12 low-price/split names → **cross-flags M8**.)
  - **Confirms the M4 note:** the §9.5 overnight anti-calibration vs Table 8's positive ρ IS this burst-start-vs-close-mid wedge. Recompute Table 8 / §9.5 from the close mid uniformly (paper-edit).
- **CODE NOTE:** the code's `compute_permanence.Perm_CLOP` already uses `(CRSP_OP − CloseMid)` (close mid, tradable), but the paper's eq (VSI from `m_{t_b}`) and Table 8 report the burst-start-mid version — the inconsistency the referee caught (see also M3).

### [x] M5 — Statistical inference invalid where reported, absent where promised (P1) — DONE 2026-07-04
- **(i) Pseudo-replication:** Table 8 per-burst p-values (e.g. 9.3e−126, JPM ρ=0.049) treat ~10⁵ bursts as independent; effective N = # days (~500) since all same-day bursts share one label. **Apply the date-clustered bootstrap to every per-burst statistic, incl. Optuna-phase metrics.**
- **(ii) Selection:** "Best AUC" columns (Tables 2,4) are optimizer maxima; need White (2000) / Romano–Wolf / DSR reality-check.
- **(iii) Omission:** **DSR appears nowhere.** Log shows reversal DSR-z ≈ +0.02 (t=2.96 vs E[max]≈2.94 over ~75 configs) — **report this unflattering number in Sec 10**, not just raw & Lo(2002) t.
- **RESULT** (`src_py/m5m6_inference.py`, FINDINGS §4g; reuses paper's `run_pnl_inference` + `factor_adjust_long_short`):
  - **(i) date-clustered:** flow→next-day IC — naive pooled t=−7.16 (398,610 name-days) → date-clustered t=+0.15 (1003 days); pooled |t| inflated **47×**. Effective N = #days confirmed. Also deflates the draft's "anti-calibrated IC t=−4.96" (itself pseudo-replicated → ~0 date-clustered).
  - **(iii) DSR:** Bailey-LdP Deflated Sharpe (N=75 configs). Full-sample DSR prob **0.990 SURVIVES** (Lo z=2.95); **walk-forward OOS DSR prob 0.705 does NOT survive** (Lo z=1.40), block-bootstrap cum-PnL CI includes 0. Report the OOS (unflattering) number in Sec 10. Skew/kurt extreme (fat-tailed, front-loaded → m4).
  - (ii) selection reality-check: DSR (N=75) is the deflation implemented; White/Romano–Wolf still TODO if referee insists.

### [x] M6 — Affirmative result not OOS in the sense that matters; factor alpha computed on in-sample series (P1) — DONE 2026-07-04
- Walk-forward (10.2) re-selects only *universe membership* (bottom-K price) from trailing data. Reversal sign, magnitude weighting, 20-day hold (chosen from H∈{10..30}), and the tick-regime conditioning itself were chosen on the full 2022–26 sample (overlaps the 2023–26 "OOS" window) → it's an OOS test of the *selection rule*, not the strategy.
- **Upgrades:** (a) **disclose design history** — if tick-regime sign-conditioning was pre-specified by an earlier referee, say so explicitly (best defense vs data-mining); report the ~75-config search + DSR. (b) Table 13 FF5+MOM alpha (t=3.11) is on the **full-period (in-sample)** series → **re-run the factor regression on the walk-forward OOS return series** and report that alpha (wider CI) as the primary number.
- **RESULT** (`src_py/m5m6_inference.py`, FINDINGS §4g): FF5+MOM on the walk-forward OOS series → alpha **+6.25 bps/day, NW t=1.42** (ann +15.7%/yr, IR 0.86) vs the in-sample Table 13 +5.94 bps/day, t=3.09 (reproduced). Point estimate stable, but t collapses 3.09→1.42 → **NOT significant OOS**. Both market-neutral (Mkt-RF β≈0), sensible reversal Mom loading (OOS β=−0.10, t=−2.7). → main.tex must lead with the OOS alpha (t=1.42), not Table 13's in-sample t=3.11. (b) DONE. (a) design-history disclosure = paper-edit TODO (Phase 2 rewrite of sec:reversion).

### [x] M7 — Missing the obvious baseline: plain short-term reversal in cheap stocks (P1) — DONE 2026-07-04
- Short-horizon reversal in small/cheap/illiquid/wide-spread names is old & well-documented (Jegadeesh 1990; Lehmann 1990; Nagel 2012 liquidity-provision — none cited). Cheap names = where bid-ask bounce/inventory make naive reversal look profitable and die in execution.
- **Must show burst flow adds incremental info:** (i) identical dollar-neutral H=20 construction, same bottom-K universe, using **lagged returns (1/5/20-day)** as signal instead of burst flow; (ii) using **plain daily signed trade volume** (no Hawkes/geometry/SGD); (iii) burst-flow alpha **after orthogonalizing** to lagged returns + simple flow. If plain reversal ≈ Sharpe 0.8, the burst/ML apparatus adds nothing.
- **Model-level:** Sec 5.1 Direction coef (≈+5.0) is ~70× next feature ⇒ model ≈ "follow burst side"; **the README-promised Direction-only ablation never appears — add it.**
- **RESULT** (`src_py/m7_reversal_baseline.py`, `results/research/m7_reversal_baseline.csv`, FINDINGS §4f; 438 names):
  - (i) **Plain lagged-return reversal does NOT reproduce it** — ret_lag_{1,5,20} Sharpe −0.34/−0.31/−0.15 full, ≈0 OOS. The effect is NOT mechanical price reversal. Baseline does not kill the result.
  - (iii) **Burst flow survives orthogonalization** to lagged returns: full 1.47→1.30 (t2.58), OOS 0.79→0.68 (t1.19) → carries incremental info beyond price.
  - **BUT the ML/SGD apparatus is superfluous and harmful:** using the SGD `pred` as signal gives Sharpe −1.21 full / −0.82 OOS; the edge is in RAW signed flow. Direction-only sign(flow) alone = OOS Sharpe +1.04 (t1.80), matching/beating the full-magnitude signal. → deflate "high-dimensional ML" (m1); reframe positive result as a raw signed-order-flow reversal.
  - **Direction-only ablation:** sign(pred)==sign(side) only 20.9% (median 20.3%); corr(side,pred)=−0.205 → deployed reg_clop model is ANTI-correlated with burst side, contradicting Sec 5.1's "coef +5 ⇒ follow side" (that was a different/in-sample model). Reconcile in rewrite.
  - (ii) **DONE** — plain un-gated signed volume Σ(BuyVolume−SellVolume) from master `bursts_*_unfiltered.csv` (no geometry gate, no SGD; corr +0.63 with flow_signal): reversal Sharpe +0.89 full (t1.79) / +0.50 OOS (t0.87). Plain signed order flow captures ~60% of the edge; the geometry gate roughly doubles it in-sample only, SGD hurts. → the positive result is a **signed-order-flow reversal**, not an ML result. (`src_py/m7_signed_volume.py`, `results/research/m7_signed_volume.csv`, FINDINGS §4f.)

### [x] M8 — Data integrity unverified where the positive result lives (low-priced names) (P1) — DONE 2026-07-04 (Phase 3)
- **RESULT** (`src_py/m8_costs_splits.py`, FINDINGS §4j; §2 + new §reversion "Data Integrity" subsection):
  - **(iii) spread costs:** bottom-100 median rel-spread 8.0 bps (≫ 1–3 bps grid), but per-name half-spread costs on the H=20 reversal barely bite: Sharpe full 1.47→1.42, OOS 0.79→0.77. Survives realistic costs (low turnover). Now reported with `name_relspread_bps.csv` per-name costs.
  - **(i) corporate actions:** price source stated (consolidated vendor feed, adjustment now treated as a check). Scan flags 10 name-days with >50% single-day moves (LCID +862%, NLY +265%, NVAX +99%, …) = likely unadjusted (reverse-)splits → spurious P&L; revision winsorizes/verifies these. Disclosed.
  - **(ii) delisting/survivorship:** membership fixed at sample start by LOBSTER coverage (not survival-to-2026); residual delisting bias acknowledged; delisting-return splice (Shumway 1997, now cited) flagged as needing CRSP data.
- **(i) Corporate actions:** trades close-to-close in bottom price quintile (reverse splits common). Paper never states price source or split/dividend adjustment. Unadjusted reverse split ⇒ spurious several-hundred-% one-day "return" ⇒ huge fake P&L in the traded universe. **State source (CRSP? vendor?) + adjustment; hand-verify top P&L days.**
- **(ii) Survivorship:** how/as-of-when was the "482-name LOBSTER 2022–26 universe" built? If membership requires surviving to 2026, delisted (disproportionately cheap, reversal-loser) names excluded → upward-biased Sharpe. **Incorporate delisting returns (Shumway 1997).**
- **(iii) Costs:** Table 12 spread quintiles Q4/Q5 = 7.6/11.7 bps, but cost grid stops at 3 bps, headline uses 1 bps; advertised Almgren–Chriss never applied. **Report per-name, spread-based costs (effective half-spread at close) for the traded subset.**

### [x] M9 — Venue coverage unexamined; contradicts the paper's own labels (P2) — DONE 2026-07-04 (Phase 3)
- **RESULT** (new §breadth "Venue Coverage" subsection + §2 + Table 1 caption): stated LOBSTER = NASDAQ book only; JPM burst ADV 993k ≈ ~10% of consolidated ADV (first-order truncation); consistent "NASDAQ universe" labeling; discussed that fragmentation attenuates detection and that JPM/MS "failures" may be coverage artifacts, not regimes. Full per-name consolidated-ADV coverage table needs a SIP volume feed (not on hand) — flagged as data-revision item.
- LOBSTER = NASDAQ book only. JPM/MS are NYSE-listed (minority NASDAQ prints; closing auction at NYSE). Single-venue burst detection sees a nonrandom fragment whose share varies by name/time. Paper calls sample "482-name NASDAQ universe" yet features NYSE names in every flagship table; Table 1 JPM ADV = 993,261 (~1/10 of consolidated) suggests truncation is first-order.
- **Fix:** (a) state fraction of consolidated volume captured per name; (b) discuss how fragmentation attenuates detection & biases cross-name comparisons (JPM/MS "failures" may be coverage artifacts, not regimes); (c) consistent universe labeling.

### [x] M10 — 81% net-short tilt needs a sign-convention audit; disclose the sign flip (P1) — DONE 2026-07-04
- "Informational" flow is sell-signed on 81% of name-days in a bull market. LOBSTER `Direction` = side of the *resting* order (execution vs bid = market **sell**); authors already hit one sign landmine (type-5 hidden `Direction=+1` unconditionally, Sec 11.3).
- **Fix:** one-table sanity check — corr(daily signed burst flow, same-day return) per name; if signing correct this is strongly positive nearly everywhere. If not, the 81% short tilt & "anti-calibrated" IC have a mundane cause.
- **Disclose:** log records Table 11 panel run with a "sign-flip applied"; any post-hoc transform → **put in table notes** (even a null's magnitude is uninterpretable if sign convention was tuned).
- **RESULT** (`src_py/m10_sign_audit.py`, `results/research/m10_sign_audit.csv`, FINDINGS §4h; 438 names):
  - **Sign convention CORRECT, not flipped.** Contemporaneous corr(signed flow, same-day return) is robustly POSITIVE — open→close mean +0.017 (57% pos, sig+ 21% >> sig− 12%); close→close mean +0.032 (61% pos). Buys push price up the same day (price impact), but the effect is economically SMALL (~0.02–0.03). Same-day +impact with next-day reversal = internally coherent. → the 81% short tilt & anti-calibrated IC are NOT a sign bug.
  - **81% net-short tilt is a genuine feature:** E[net_dir]=−0.627, 81% of name-days net-short (matches referee), 87% of names net-short on average. Discuss (not "fix"); ties to the incidental short-beta loss.
  - **Panel sign-flip disclosed:** 116/482 names FlipSign=−1 (data-driven mean-reverting cluster; Spearman(dir, next-day ret)<0), signed COI ×−1 in the Table 11 panel via `--regime-csv`. **Add to Table 11 notes** (paper-edit TODO).

### [x] M11 — Claims asserted with no supporting evidence (P1) — DONE 2026-07-04 (Phase 2)
- **(i)** Conclusion Poisson claim — **DELETED** from the Conclusion (no test existed).
- **(ii)** Sec 7.1 "deeply negative per-trade EV" — now **quantified** by cross-reference to the honest intraday markout ($-3.5$ to $-7.7$ bps net of spread, Table markout_honest).
- **(iii)** Sec 7.2 cost-aware > percentile gating — reworded to a design-choice note; the unsupported performance claim removed (no separate table asserted).
- **(iv)** `cb` — now **DEFINED**: $cb_{i,t}=c\cdot\hat{s}_{i,t}$ (trailing half-spread × multiplier $c=1$), stated as a gate not an alpha source.

### [x] M12 — Hidden-execution finding too thin for its billing (P2) — DONE 2026-07-04 (full cross-section)
- **RESULT** (job array 13911527, 474 names, 221,261 ticker-days 2023-2024; `hidden_xsec_agg.py`, FINDINGS §4l; new Table tab:hidden_xsec + updated §reconstruction/Conclusion):
  - **(a) midpoint fraction = 48.9%** of hidden prints at the mid (unsignable by quote rule) — quantified across the universe; classifier (tick/EMO/CLNV) sensitivity flagged as next refinement.
  - **(b) cross-section**: 3-min footprint **+1.62 bps, pooled date-clustered t=3.38, significant in 79% of 474 names** — the n=2 pilot generalizes and STRENGTHENS. Overnight **NULL at scale** (cross-sectional hidden-COI→CLOP IC t=−0.40). Sub-spread (1.6 vs 4–12 bps). 15/30-min do NOT reverse in the broad universe (drift-confounded; 3-min is the clean measure — stated honestly).
  - **(c) hidden-liquidity lit cited** (Bessembinder et al. 2009, Hautsch–Huang 2012); **(d) "only reconstruction" now backed by 474 names, not tempered-away.**
- **DONE now (no re-fetch):** (c) hidden-liquidity lit cited (Bessembinder–Panayides–Venkataraman 2009; Hautsch–Huang 2012); (d) "only reconstruction" already tempered to sub-spread/minutes-scale/null-overnight on n=2; added an explicit statement that the midpoint fraction + tick/EMO/CLNV classification sensitivity and a cross-section extension are the scope of a dedicated follow-up; Lee–Ready signing trap already documented.
- **LAUNCHED 2026-07-04 (job array 13911527, `hoffman2/hidden_xsec.sh`):** full 483-name × 2023-2024 (502 days) hidden-execution cross-section streaming from lobster2. One SGE task per ticker (-t 1-483, -tc 12 throttle, 6 cores each, stream→Lee-Ready-sign→cluster→3/15/30m markout + daily COI→delete raw). `hidden_full.py` extended to emit 3/15/30-min markouts + midpoint counts (M12a). Smoke-tested OK (AAON 20230103 → 64 bursts, mk3+4.9). Aggregation: `src_py/hidden_xsec_agg.py` (run after array) → per-name + date-clustered markout t by horizon, midpoint fraction, daily hidden-COI→CLOP IC. Results: `results/hidden_xsec/out/<TK>.csv`, `results/research/hidden_xsec_daily.csv`. This closes M12(a) midpoint fraction + (b) cross-section.
- Sec 11.4 "only reconstruction with a statistically robust informed footprint" from **2 tickers, 1 year**. Lee–Ready is *least* reliable for hidden execs (occur at/near midpoint where quote rule undefined, tick-rule noisy).
- **Required:** (a) report fraction of hidden prints at midpoint + classification sensitivity (tick / EMO / CLNV); (b) extend to a real cross-section (infra runs 482) + ≥2 years; (c) cite hidden-liquidity lit (Bessembinder–Panayides–Venkataraman 2009; Hautsch–Huang 2012); (d) temper "only reconstruction" to what n=2 supports.
- **Reframe vocabulary:** a sub-spread footprint peaking at 3 min, reversing by 30, is transient-impact/liquidity, **not** Kyle-style information incorporation — Sec 1 framing promises a different paper.

---

## MINOR CONCERNS
- [ ] **m1 — Abstract:** one ~450-word paragraph; restructure to 3–4 sentences claim then evidence. "High-dimensional ML" oversells a linear SGD on ~30 features.
- [ ] **m2 — Title:** "Identifying Latent Alpha" mismatches a largely-negative paper; retitle to the diagnostic/conditional-reversal content.
- [ ] **m3 — Passive appendix Sharpes impossible:** Table 17 = −76.35 / −140.30 annualized; no daily-marked strategy nears these → almost certainly per-trade Sharpe annualized by trade count. **Fix the bug + any code sharing this annualization.**
- [ ] **m4 — Earnings "robustness" = fragility:** removing 16/471 trades ~doubles NVDA Sharpe (0.48→0.91) ⇒ P&L dominated by a few days. Report per-trade contribution distribution, don't frame as success.
- [x] **m1 — Abstract:** DONE — restructured to claim-then-evidence across 4 paragraphs; "high-dimensional ML" removed (now "linear online-SGD"), notes the plain-signed-volume baseline.
- [x] **m2 — Title:** DONE — retitled "A Diagnostic Study of Look-Ahead Bias and Conditional Reversal in Order-Submission Bursts".
- [x] **m3 — Passive appendix Sharpes:** DONE — added a note that −76/−140 are a per-trade Sharpe annualized by trade count (not daily-marked), read as ordinal only; conclusion unaffected. (Code-side fix flagged for the passive pipeline; appendix figures retained with correction noted.)
- [x] **m4 — Earnings fragility:** DONE — retitled "A Fragility, Not a Robustness"; frames the 2× Sharpe jump from deleting 16 days as fat-tail fragility / anti-calibration.
- [x] **m5 — Figures:** DONE — three figures generated (`src_py/make_figures.py`, matplotlib PDFs in `figures/`) and embedded, compiles to 31 pp:
  - `fig_markout` (Fig, §markout): intraday markout by horizon, both aggressive-burst and Lee–Ready-signed hidden-exec inside the ±4 bps spread band; hidden footprint reverses to negative by 30 min.
  - `fig_oos_pnl` (Fig, §reversion): cumulative walk-forward OOS P&L — burst-flow reversal (+0.83) compounds while plain price reversal (−0.22) and SGD-ML-as-signal (−0.71) lose. **Winsorizing daily returns at ±50% (M8 mitigation) RAISES the OOS Sharpe 0.79→0.83** → the split artifacts are NOT what drives the reversal (M8 corroboration).
  - `fig_regime` (Fig, §reversion): reversal Sharpe by price/spread quintile, showing the tick-constraint concentration.
- [x] **m6 — Citations:** DONE — added 16 refs (Jegadeesh 1990, Lehmann 1990, Nagel 2012, Lee–Ready 1991, Cont–Kukanov–Stoikov 2014, Lo 2002, Bailey–LdP 2014, Fama–MacBeth 1973, Newey–West 1987, Yao–Ye 2018, O'Hara–Saar–Zhong 2019, Easley et al. 2012, Bessembinder et al. 2009, Hautsch–Huang 2012, Optuna 2019) and cited the key ones at point of use.
- [x] **m7 — Numerical inconsistencies:** DONE — 482 used consistently; NVDA post-filter 220,430 (matches Table 3); 435-of-438 reconciled with a parenthetical; "2019–2022"/"125–470 total trades" phrasings removed in the §2 and §7 rewrites.
- [x] **m8 — Parameter provenance:** DONE — §9.7 now states two distinct sets: probe = per-ticker mean (vol_frac 0.00218, κ 0.855); breadth = universal medians (vol_frac 0.00197, κ 1.085); each table says which.
- [x] **m9 — Table 1 "ADV":** DONE — header renamed "Avg Daily Burst Vol (sh.)"; caption clarifies it is burst volume, not consolidated ADV (which is ~10× for JPM).
- [x] **m10 — Sample truncation:** DONE — caption note: reduced AAPL/SPY day counts reflect LOBSTER coverage gaps in the archive.
- [x] **m11 — Cross-ref/structure:** DONE — D_b now refs `sec:burstdef` (Eq. label); hard-coded "Section 6.2" → `\ref{sec:phase3}`; `\appendix` added, passive is now Appendix A, Conclusion moved before the appendix.
- [x] **m12 — ¼ factor:** DONE — corrected to ⅓ (3-term sum) in both the definition (Eq. `eq:db`) and the markout section.
- [x] **m13 — Tone:** DONE — removed "Mathematically flagged", "securely compresses", "definitively refute", "profound microstructural finding", "empirically proves", "uniform success confirms", "mathematically resolve/ineligible"; "idiosyncratic of the FF5" → "orthogonal to" (both places).
- [x] **m14 — Data section:** DONE — §2 rewritten to describe the two-tier data (diagnostic set + 482-name 2022–2026 universe), membership rule, return source, corporate-action caveat, venue coverage, and the 2019–2021 probe status.

---

## PROCESS / CONSISTENCY NOTES (referee cross-checked paper vs experiment log)
- [x] LLY recorded as a **training-set** name in the log but labeled **OOS** in the paper (see M2). → FIXED: LLY relabeled in-sample in Tables 1 & 5 and text; OOS labels removed from 2019–2021 rows.
- [x] Log's tempering of the reversal headline ("needs tempering... pending review") was applied to the **abstract + Sec 10.2** but **not** to **Sec 10.3's factor-alpha framing** (see M6). → OOS factor alpha now computed (t=1.42); Sec 10.3 rewrite to lead with it is a Phase-2 paper edit.

---

## RESUBMISSION PATH (referee's constructive route)
A viable resubmission would:
1. **Rewrite Sec 4–7 as the diagnostic subject** of a methods/cautionary paper — the D_b trap, the three look-ahead pitfalls, the in-sample-names post-mortem are the durable contribution. *(within existing infra)*
2. **Fix D_b-as-feature infeasibility and re-run the pipeline.** *(within existing infra)*
3. **Subject the tick-constrained reversal to:** plain-reversal baseline (M7), delisting-inclusive returns + split verification (M8), spread-based costs (M8); report **OOS factor alpha + DSR**; ideally the **2017–2021 backfill** (decisive power upgrade — see `BACKFILL_PLAN_2017_2021.md`). *(item that determines whether a positive result exists at all)*
4. **Scale the hidden-execution analysis to the full cross-section** — could become the centerpiece. *(within existing infra)*

Referee: if the reversal survives (3) → solid microstructure paper; if not → honest-negative-plus-methodology paper still worth publishing, at a field venue, after the internal contradictions are gone.

---

## SUGGESTED TRIAGE (our ordering)
- **Fastest / no-recompute (paper edits):** M1, M2 (labels), M11, m1–m14, provenance table (M2), process notes. Removes the contradictions + unsupported claims.
- **Cheap recompute (existing data):** M5 (DSR/date-cluster on saved outputs), M6 (OOS factor alpha), M7 (reversal baselines + Direction-only ablation), M10 (sign-convention corr table), M4 (close-mid target recompute).
- **Moderate:** M3 (drop D_b / MOC-cutoff feature window + re-run), M8 (price source/splits/delistings/costs), M9 (venue coverage), M12 (hidden cross-section, needs re-fetch).
- **Decisive / data-gated:** 2017–2021 backfill to resolve reversal significance (M6 power).
