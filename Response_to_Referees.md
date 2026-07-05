# Response to Referees

**Manuscript:** *Informational Order Flow and Price Persistence: A Diagnostic Study of Look-Ahead Bias and Conditional Reversal in Order-Submission Bursts* (revised)
**Author:** Nicholas Jiang
**Referee:** Mihai Cucuringu (UCLA) — original report and addendum

---

## Cover Letter

Dear Editor and Referee,

Thank you for two unusually careful and constructive reports. The reviews correctly identified that the original submission read as two papers stapled together: an optimistic single-name results section and a set of diagnostics that quietly undercut it. We have taken the reports as an invitation to rebuild the paper around what actually survives scrutiny, and the result is, we believe, a substantially more honest and more useful contribution.

The revision makes four structural changes. First, the empirical narrative is inverted: Sections 4–7 are now an explicit *diagnostic case study* of the look-ahead and pseudo-replication traps that wreck microstructure-ML backtests, rather than a success story. Second, the study is moved from four in-sample names to the **full 482-name NASDAQ universe (2022–2026)**, exactly as Reframe R6 and Major Issue M9 urged, which converts the paper's largest liability into its principal asset. Third, every headline number now carries formal inference — Lo (2002) standard errors, a date-clustered bootstrap, the Deflated Sharpe Ratio, and out-of-sample factor alphas. Fourth, we add a formal twelve-dimension **Methodological Audit (Appendix B, Table 26, p.31)** and a new empirical centerpiece: a **474-name hidden-execution cross-section (Table 22, p.26)**.

The honest bottom line has changed accordingly. The paper no longer claims a deployable overnight alpha; it delivers a precisely characterized negative result, a durable methodological toolkit, one genuinely novel empirical fact (the hidden-execution footprint), and one candidate-but-unproven conditional strategy (tick-constrained reversal). We are grateful that the addendum's pointer to Lu, Reinert & Cucuringu (2024) gave us a direct template for the cross-sectional design; we cite and build on it throughout.

Point-by-point responses follow. Cross-references are to the revised 35-page manuscript. We group the major concerns into the twelve dimensions of our audit (M1–M12; these consolidate the major issues of the original report), then address minor concerns (m1–m14), the reframes (R1–R6), and the borrowable elements (B1–B12).

---

## Part I — Major Concerns (M1–M12)

These twelve dimensions are summarized in **Appendix B, Table 26 (p.31)**; each is developed in a dedicated subsection there.

**M1 — Internal contradiction / success framing.** Sections 4–7 (now **§7, "Empirical Results on the Diagnostic Set," p.9**) are rewritten from the outset as a cautionary subject. Every invalidated claim has been removed from the captions and topic sentences: "statistically significant profitability" is withdrawn (**Table 6, p.11**), "strictly exceeds slippage" and "profound microstructural finding" are gone, and §7.9 (**p.13**) no longer argues that signal cannot exist in tick-constrained names — the contradiction with the reversal result is resolved by reframing tick regime as governing the *sign*, not the existence, of the signal.

**M2 — In-sample flagships mislabeled OOS.** LLY is now correctly flagged as an in-sample tuning name in **Table 1 (p.3)** and **Table 6 (p.11)**; the "(OOS)" labels on the 2019–2021 rows are removed and reframed as a parameter-transfer probe. We note explicitly that a Sharpe of 0.59 over ~2 years implies *t* ≈ 0.83 even in-sample. A **number-provenance table (Table 7, p.12)** maps every headline single-name figure to one canonical run.

**M3 — D_b timing infeasibility.** The two definitions are reconciled (**§3, Eq. 4, p.4**), and we document that D_b is the forward 1–10 min markout, realized after the burst and thus inadmissible as a live feature. We re-ran the overnight strategy with all D_b-derived features dropped (`OB_DROP_DB`): per-name Sharpes shift only modestly (mean |Δ| = 0.14) and **the mean Sharpe stays negative, −0.28 → −0.20 (§9, p.15)** — the null does not depend on the look-ahead feature.

**M4 — Target conflates realized impact with tradable return.** Recomputing the predictive metric against the tradable close-to-open return from the close mid (rather than the burst-start mid) **collapses the Spearman ρ roughly fivefold and, once clustered by date, to a statistical zero (t = −0.33; §5, p.7)**. The apparent predictability was in the already-realized, pre-MOC component.

**M5 — Inference and multiple-testing.** Two fixes. (i) Pseudo-replication: the flagship flow-to-return IC collapses from a pooled *t* = −7.16 (398,610 name-days) to a **date-clustered t = +0.15 (1,003 days) — a 47× deflation**. (ii) Deflated Sharpe: for the reversal, the full-sample Sharpe 1.47 has DSR probability 0.99, but the **walk-forward OOS Sharpe 0.79 has DSR probability 0.70 (below 0.95); the date-block-bootstrap cumulative-P&L interval includes zero (§10.4, p.22; Appendix B)**. Lo (2002) SEs and the block bootstrap are reported throughout.

**M6 — Factor alpha on in-sample series.** The FF5+MOM alpha is re-estimated on the strict walk-forward OOS series (**Table 18, p.22**): the point estimate is essentially unchanged (**+6.25 vs +5.94 bps/day**) but the Newey–West **t falls from 3.09 to 1.42** — not significant. We report the OOS regression as the primary number.

**M7 — Missing baseline + Direction-only ablation.** The **3-rung baseline spectrum (Table 19, p.23)** holds the tick-constrained construction fixed and swaps only the signal: plain 1/5/20-day price reversal is flat-to-negative (Sharpe −0.34…−0.15), so the effect is *not* mechanical reversal; burst flow survives orthogonalization to lagged returns (1.47 → 1.30); **plain signed volume with no gate and no model already captures most of it (+0.89)**; and **feeding the SGD prediction in as the signal is actively harmful (−1.21)**. The Direction-only ablation (**§5, p.7**) confirms the model agrees with burst side only 21% of the time. The contribution is thus a raw signed-order-flow reversal, not a machine-learning result — and the "high-dimensional ML" framing is retired.

**M8 — Data integrity and costs.** (i) A corporate-action scan flags 10 name-days with >50% single-day moves (e.g. LCID +862%); **winsorizing at ±50% *raises* the OOS reversal Sharpe from 0.79 to 0.83 (Fig. 3, p.23)**, so the split artifacts are not what drives it. (ii) Re-costing with per-name effective half-spreads (**median 8.0 bps**, far above the 1–3 bps grid) lowers the Sharpe only marginally, **1.47 → 1.42 (§10.5, p.24)**. (iii) Delisting-return inclusion (Shumway 1997) is flagged as the remaining gap.

**M9 — NASDAQ venue coverage.** **§9.5 (p.18)** states that LOBSTER reconstructs the NASDAQ book only; the captured share varies by name and is smallest for NYSE-listed financials (JPM ≈ 10% of consolidated ADV), so their single-name "failures" may be coverage artifacts. The universe is consistently labeled.

**M10 — Sign-convention audit.** A per-name audit (**§9, p.15**) shows the contemporaneous correlation between signed burst flow and the same-day return is robustly positive (mean +0.02 to +0.03; significantly positive in 21% of names vs 12% negative): the convention is correct and the 81% short tilt is a genuine sample property, not a flipped sign. The data-driven per-name sign flip used in the panel is disclosed in the **Table 15 (p.18)** caption.

**M11 — Unsupported claims.** The Poisson claim is now *substantiated rather than asserted*: a formal test (see B2) rejects the homogeneous-Poisson null on 99.8% of name-days (median z ≈ 87). The cost-aware buffer `cb`, previously undefined, is given an explicit form (**§6.2, p.8**).

**M12 — Hidden-execution finding scaled.** The two-name pilot is extended to the **full 474-name cross-section (221,261 ticker-days; Table 22, p.26)**: the three-minute directional markout is **+1.62 bps with a pooled date-clustered t = 3.38, significant in 79% of names**. We report the honest caveats — **48.9% of hidden prints execute at the midpoint** (unsignable) and the footprint is **sub-spread and does not reach the overnight horizon (cross-sectional overnight IC t = −0.40)**. It is a robust microstructure fact about concealment, not deployable overnight alpha.

---

## Part II — Minor Concerns (m1–m14)

- **m1 (Abstract):** restructured to claim-then-evidence; "high-dimensional ML" removed (a linear online SGD).
- **m2 (Title):** retitled to the diagnostic / conditional-reversal content.
- **m3 (Passive appendix Sharpes):** the −76/−140 values are flagged as a per-trade Sharpe annualized by trade count, read as ordinal only (**Appendix A, Table 24**); the substantive negative conclusion is unaffected.
- **m4 (Earnings fragility):** reframed as fragility, not robustness — deleting 16 of ~470 days moving the Sharpe 2× is fat-tailed, not clean (**§7.10, p.14**).
- **m5 (Figures):** three figures added — markout decay (**Fig. 1, p.16**), cumulative OOS P&L (**Fig. 3, p.23**), regime quintiles (**Fig. 2, p.21**).
- **m6 (Citations):** expanded from 6 to **28 references** (see B12).
- **m7 (Numeric inconsistencies):** 482 used consistently; NVDA post-filter 220,430 (matches Table 3); 435-of-438 reconciled; stale "2019–2022" / "125–470 daily trades" phrasings removed.
- **m8 (Parameter provenance):** **§9.7** states the two distinct fixed-parameter sets (per-ticker mean for the probe; universal medians for breadth).
- **m9 (Table 1 "ADV"):** header renamed "Avg Daily Burst Vol"; caption clarifies it is burst, not consolidated, volume (**Table 1, p.3**).
- **m10 (Sample truncation):** the reduced AAPL/SPY day counts are explained (LOBSTER coverage gaps) in the Table 1 caption.
- **m11 (Cross-refs/structure):** D_b now references §3/Eq. 4; the hard-coded "Section 6.2" uses `\ref`; `\appendix` added, the Conclusion precedes the appendices, and the passive study is now **Appendix A**.
- **m12 (¼ factor):** corrected to ⅓ on the three-term sum (**Eq. 4, p.4**).
- **m13 (Tone):** systematic overclaiming removed ("definitively refute," "uniform success," "empirically proves," etc.); "idiosyncratic of the FF5" → "orthogonal to."
- **m14 (Data section):** **§2** rewritten to describe the two-tier data (diagnostic set + 482-name universe), membership rule, return source, corporate-action caveat, and venue coverage.

---

## Part III — Reviewer Reframes (R1–R6)

**R1 — Markout panel as first-class diagnostic.** Adopted: **§8 (p.14)** reports the multi-horizon, look-ahead-free (κ=0) markout panel with a date-cluster bootstrap CI (**Table 11, p.15; Fig. 1, p.16**), and the D_b look-ahead is quantified in **Table 10 (p.15)**. The intraday footprint is shown to be genuine but sub-spread.

**R2 — PPT in bps as the reporting unit.** Largely adopted: the markout, reversal, and factor results are all reported in bps and Sharpe with *t*-statistics; the fixed-AUM ROC framing is retained only within the §7 diagnostic and is explicitly labeled in-sample. We agree PPT-in-bps is the right unit and have made it the headline for every honest result.

**R3 — Sign-conditional reinterpretation.** This "single most important reframe" is now the paper's affirmative core: **§10 (p.20), "Conditional Mean-Reversion in Tick-Constrained Names,"** documents the monotone tick-regime sign asymmetry — continuation in large-tick names, reversal in tick-constrained names (**Table 16, p.20; Fig. 2, p.21**) — dissolving the old §7.9 "regime mismatch" exegesis exactly as suggested.

**R4 — Finer horizon decomposition.** Partially adopted: CLOP vs CLCL (**Table 9, p.13**), the intraday markout panel (**Table 11, p.15**), and the hidden-execution 3/15/30-min decay (**Table 23, p.26**) localize the effect. A full MOO+30m / OPCL five-point decomposition is noted as a natural extension (Appendix B, Residual Items).

**R5 — Gross-of-cost reporting.** Adopted in structure: §8 reports gross bps markouts with bootstrap CIs ("is there a signal?"), and cost sensitivity is separated into the cost grid (**Table 8, p.12**) and the per-name spread-cost analysis (**§10.5, p.24**). The `cb` buffer is confined to the methodological exposition.

**R6 — Cross-sectional breadth.** Fully adopted, and the single largest change: the entire pipeline runs on the **482-name universe (§9, p.15)**, with regime-stratified panels, cross-sectional IC distributions, and quintile sorts replacing the four-point illustration.

---

## Part IV — Borrowable Elements from LRC24 (B1–B12)

We thank the referee for pointing us to Lu, Reinert & Cucuringu (2024); it is an unusually direct template and we adopt it explicitly.

- **B1 (Universe & sample):** 482-name NASDAQ universe, 2022–2026 (**§2, §9**), the NASDAQ analog of the LRC24 S&P 500 panel.
- **B2 (Poisson null-model test):** implemented (`poisson_test.py`). Against a homogeneous-Poisson-arrival + iid-sign null, observed same-side bursts exceed the null by a **median z ≈ 87**, the null is **rejected on 99.8% of name-days**, and arrivals are strongly over-dispersed (**median Fano factor 8.1** vs 1). Reported in **§3 (p.3)** and **Appendix B**.
- **B3 (COI framework):** the daily signed burst imbalance is exactly the COI construction of LRC24 (**§9, Table 15, p.18**).
- **B4 (Panel regression with FF factors):** Fama–MacBeth with Newey–West errors and FF5+MOM controls (**Table 15, p.18; Table 18, p.22**).
- **B5 (Single/double sorts):** single sorts on price and spread quintiles are reported (**Table 16, p.20; Fig. 2, p.21**); the 5×5 double sort is noted as a natural extension.
- **B6 (Long-short with risk-adjusted alpha):** the long-short reversal is regressed on FF5+MOM with Newey–West inference; **OOS alpha +6.25 bps/day, t = 1.42 (Table 18, p.22)** — precisely the "α or no-α after risk adjustment" format requested.
- **B7 (Benchmark suite):** the unconditional signed-flow and lagged-return benchmarks are reported (**Table 19, p.23**); additional EW/SPY portfolio benchmarks are a straightforward addition.
- **B8 (Cost grid in bps):** **Table 8 (p.12)**, applied after the gross reporting.
- **B9 (Time-of-day stratification):** implemented — the 3-min markout is largest at the open (+0.84 bps) and decays through the session, and the 3:50 dead-zone is structurally justified (**Appendix B**).
- **B10 (Robustness of parameters):** the Hawkes decay grid is reported (**Table 20, p.25**); an (α, β) MLE fit is noted as a further refinement.
- **B11 (Count- vs volume-based COI):** implemented — the two correlate at **0.81** and give the same weak reversal-signed IC, with count marginally stronger (date-clustered **t = −2.7 vs −1.0**), consistent with Chan–Lakonishok (**Appendix B**).
- **B12 (Literature scaffold):** engaged — we now cite LRC24, Kolm–Turiel–Westray (2023), Cont–Cucuringu–Glukhov–Prenzel (2023), Cont–Cucuringu–Zhang (2023), Sitaru–Calinescu–Cucuringu (2023), and Lucchese–Pakkanen–Veraart (2024), alongside the classical microstructure references.

---

## Headline Methodological Contributions

We highlight six contributions that we believe are of independent value to microstructure-ML practice, consolidated in the twelve-dimension audit (**Appendix B, Table 26, p.31**):

1. **The 3-rung baseline spectrum (M7, Table 19, p.23):** a clean protocol for separating an order-flow signal from (i) mechanical price reversal and (ii) its own machine-learning apparatus — here showing the ML layer is superfluous and the SGD prediction actively harmful.
2. **Date-clustered inference + Deflated Sharpe (M5):** demonstrating a 47× *t*-statistic inflation from pseudo-replication and an OOS Deflated-Sharpe probability of 0.70 that overturns the apparent full-sample significance.
3. **The D_b timing check (M3):** identifying a forward-realized filter masquerading as a live feature, and showing by re-run that the null is invariant to its removal.
4. **The transaction-cost & split audit (M8):** per-name spread-based costs and a winsorization test that, unusually, *strengthens* the result — ruling out a corporate-action artifact.
5. **The 474-name hidden-execution discovery (M12, Table 22, p.26):** the paper's novel empirical fact — a pervasive, day-clustered-significant (t = 3.4) sub-spread footprint of concealed institutional flow, with a quantified 49% midpoint-classification caveat.
6. **The Poisson null test (B2):** a formal demonstration (median z ≈ 87) that bursts are genuine clustered structures, not arrival-rate artifacts.

We are grateful for the depth of the reviews, which materially improved the paper, and we hope the revised manuscript meets the standard for the venue.

Respectfully,
Nicholas Jiang
