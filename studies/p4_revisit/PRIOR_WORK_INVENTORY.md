# Prior research inventory: what was tried, what it found, what can be trusted

Written 2026-09-14 before the P4 revisit (`studies/p4_revisit/P4_REVISIT_PLAN.md`).

**Sources read.**
- The proposal `UCLA_279__P4_informed_bursts.pdf` (v1, 2026-01-23).
- The git history (195 commits, 2026-02-04 onward).
- `archive/docs/PROJECT_LIFETIME_RESULTS_SUMMARY_20260407.md` §1–11 and §14, and `results/*.md`.
- `archive/passive/PASSIVE_FINAL_REPORT.md`, `archive/docs/FINDINGS_LOG.md` §0–5, `archive/docs/corrections.md`.
- `archive/docs/METHODS_AND_REPRODUCIBILITY.md`, `archive/docs/STRATEGY_AND_REPLICATION.md`, `archive/docs/walkthrough.md`, `archive/docs/process.md`.
- `archive/docs/BACKFILL_PLAN_2017_2021.md`, `studies/two_avenue/TWO_AVENUE_DESIGN.md`, `DEFINITIONS_TRIED.md`, `VERIFIED_RESULTS.md`.
- `main.tex` §§1–12 and the appendices, and the abstract and outline of `paper.tex`.
- The legacy code: `src_cpp/burst.cpp` and `main.cpp` at HEAD, `src_py/compute_permanence.py`.

---

## 0. Summary

1. **The P4 pipeline was never run as written.**
   - The implemented filter differed from the proposal: a volume-scaled D_b ≥ κ in dollar-shares, not D_b ≥ κ·PeakImpact.
   - The long-horizon targets were measured from the burst-start mid.
   - Parameters were tuned on the names and years later reported.
   - The rule-based Phase III that P4 suggests (informative bursts above a size percentile, aggregated to S_i,t, tested against CLOP and CLCL with controls) was never run on a point-in-time cross-section.
2. **A defect not recorded anywhere before.**
   - The April–July C++ detector counted every hidden (type-5) execution as a sell.
   - Since mid-2014, ITCH reports the hidden side as "buy" for every such print. Verified on real files: 100% of type-5 messages carry Direction +1.
   - Hidden prints are 20–28% of executed shares, so every C++-based burst direction, flow signal and P4 result was sign-biased toward selling.
   - **Q0a (2023, 680 name-days, legacy rule replicated) confirms it explains the old 81% net-short tilt:**
     - legacy signing: 78.5% of directional bursts are sells; 97.6% of name-days net short;
     - without type 5, or with native packets: 51% sells; 49–53% of name-days net short.
   - *Corrected 2026-09-15.* An earlier draft of this item also attributed two further anomalies to the bias. Q0a does not support that:
     - the 37.8% hit rate is not reproduced: the start-mid one-minute hit rate is 75% legacy vs 79% corrected;
     - the +0.02 same-day correlation is small under every stream (−0.003 to +0.05).
3. **Every "promising" number has a named defect.** Look-ahead (the κ gate), in-sample selection, target overlap, survivorship, pseudo-replication, and the sign bias above. None of them survived a clean re-test.
4. **What is clean.**
   - The September 2026 work: native-sign economic packets, fingerprint-validated bursts, point-in-time panels.
   - It shows real directional algorithms. There is no daily-imbalance alpha and no passive-posting alpha.
   - It never applied P4's permanence filter.
5. **The P4 question is therefore still open**, with correct signs and firewalls: do bursts whose impact persists carry information past the decision time, and are they parent-order children?

---

## 1. Timeline

| era | dates | question | main artifacts |
|---|---|---|---|
| A. P4 build | Feb 4 – Apr 14 2026 | Phase I–III on NVDA/TSLA/JPM/MS | `src_cpp/`, `compute_permanence.py`, `optuna_physical_sweep.py`, `online_sgd_backtest.py`, lifetime summary |
| B. P4 refinements | May 13–22 2026 | tCLOSE with t+10 execution; Hawkes features; RandomForest/HistGB; `phase3_flow` to fix SGD look-ahead; passive bursts | `results/fixed_aum_backtest_results.md`, `transaction_cost_sensitivity.md`, `LLY_SGD_Backtest_Analysis.md`, `archive/passive/` |
| C. Breadth + referee rounds | Jun 6 – Jul 6 2026 | 482-name 2022–26 run; referee M1–M12, R1–R6, B1–B12 | `archive/docs/FINDINGS_LOG.md`, `archive/docs/corrections.md`, `Response_to_Referees.md`, `main.tex` §§4–12 |
| D. Hidden-liquidity paper | Jul – Aug 2026 | Hidden-execution footprint and its conventions | `paper.tex`, `VERIFIED_RESULTS.md` §1.1–1.17 |
| E. Two avenues | Aug 28 – Sep 13 2026 | Economic packets; fragment price discovery; campaign joining; continuation; spread scaling | `studies/two_avenue/TWO_AVENUE_DESIGN.md`, `VERIFIED_RESULTS.md` §1.18–1.26 |
| F. Well-defined bursts | Sep 13–14 2026 | Fingerprint validation; program evidence; metaorder linkage | `BURST_FINGERPRINT_*`, `PROGRAM_EVIDENCE_*`, `METAORDER_*`, `VERIFIED_RESULTS.md` §1.27–1.30 |

---

## 2. P4 as written vs. as implemented

| element | P4 proposal | April–July implementation | consequence |
|---|---|---|---|
| Event | Limit-order **submission** bursts (t, q, side, level, distance to mid); trade bursts as the §6 alternative | Aggressive **execution messages** (types 4 and 5), silence then Hawkes clustering | Submission bursts were never the main object. Passive Hawkes bursts were tried on 5–7 names only |
| Unit | Orders | ITCH execution **messages**. One sweep is several messages (1.87 per packet, §1.21) | Counts and direction ratios are message-weighted |
| Trade sign | side of the order | `Direction == −1` → buy, **else sell**, including type 5 | **Hidden prints counted as sells** (§3) |
| Burst direction | same-side sequence | count ratio ≥ dir_thresh and minority/majority volume ≤ vol_ratio, both Optuna-tuned | Mixed clusters labelled; sign bias pushed labels to sell |
| PeakImpact | max\|m(t_b+τ) − m(t_b)\| over seconds; signed variant suggested | directional extreme mid within 10 s of start | close to P4 |
| D_b | mean over h ∈ {1,3,5,10} min of side·(m(t_b+h) − m(t_b)), **from initiation** | Code: ¼ Σ **Q_b**·side·(Mid(t_end+h) − StartPrice). Text: ⅓ Σ over {1,5,10} from M_end. Three inconsistent versions | Volume scaling makes κ a dollar-share floor, which selects large bursts, not persistence |
| Filter | D_b ≥ κ·PeakImpact, κ ∈ (0,1) | D_b ≥ κ (κ 0.2–1.8 in dollar-shares), **applied upstream in C++ before any modelling** | Ratio form never tested; short-horizon targets gated by their own outcome until commit 0e4079b |
| Permanence | ratio φ = (x − m_tb)/PeakImpact, x ∈ {c_t, o_t+1, c_t+1} | arcsinh(Q_b·side·(x − ref)); ref = start mid (text, Optuna ρ) or close mid (code, CLOP/CLCL) | Start-mid base inflates predictability about 5× (M4) |
| tCLOSE | log(c_t / P_tb), from burst time | Code: CloseMid − Mid_10m (entry at t_end + 10 min). Optuna/text versions from start | P4's tCLOSE overlaps D_b's own window, which is mechanical |
| Phase I | descriptive: how permanence varies with characteristics | Skipped; went to classification AUC | Never done cleanly at breadth |
| Phase II | predict permanence from features known shortly after t_b (D_b allowed for long horizons) | Optuna over detector parameters maximizing AUC on the reported names; online SGD with D_b features | In-sample selection; model later anti-correlated with side (−0.205) |
| Phase III | S_i,t = Σ Î_b Q_b; regress fret(H) on S plus controls (vol, volume, market, sector) | Per-name MOC→MOO SGD backtests; FM COI panel without controls for the stock's own intraday return | P4 regression (5.1) with controls never run; universe ex post |
| Rule-based variant (P4 bracket) | informative bursts (3.2) + Q_b above an order-splitting percentile | never run | **the gap** |

---

## 3. New defect: hidden executions signed as sells

**Code.** `src_cpp/burst.cpp` at HEAD (and every version since 2026-02-04):

```cpp
// LOBSTER Direction: -1 = Buyer-initiated, 1 = Seller-initiated
if (msg.direction == -1) { buy_count_++;  buy_volume_ += msg.size; }
else                     { sell_count_++; sell_volume_ += msg.size; }
```

This is applied to type 4 **and type 5** (`is_trade = type == 4 || type == 5`).

**Data.** Since 2014-07-14, NASDAQ ITCH sets the Buy/Sell Indicator of non-displayed trade messages to "B" regardless of the resting side. Checked 2026-09-14 on raw lobster2 files:

| name-day | type-4 msgs (Dir −1 / +1) | type-5 msgs | type-5 Direction | hidden share of executed shares |
|---|---|---|---|
| AAPL 2019-06-03 | 41,598 / 57,945 | 19,695 | **all +1** | 19.8% |
| MS 2023-06-01 | 5,906 / 6,099 | 3,775 | **all +1** | 27.9% |

**What it contaminates.** Everything built on C++ burst CSVs:
- the Optuna AUC and ρ tables;
- the SGD backtests (NVDA/TSLA/JPM/MS/LLY, the 438-name breadth run);
- the COI panels;
- `flow_signal`, and therefore the tick-constrained reversal;
- M7 "signed volume";
- the overnight–intraday Sample C;
- the passive pipeline's volume ADV denominator. The ADV itself is unsigned, so this one is unaffected.

**What Q0a shows** (`results/p4_revisit_v1/q0/q0a_summary.json`). Python replica of the C++ rule (`src_py/p4_q0_legacy.py`), 40 names × 20 dates in 2023, 680 name-days present.

| stream | sells among directional bursts (count / volume) | net-short name-days | 1-min hit, end mid / start mid | corr(flow, open→close), pooled / within name |
|---|---|---|---|
| legacy (type 5 = sell) | 78.5% / 86.4% | **97.6%** | 51.6% / 74.9% | +0.023 / +0.006 |
| type 4 only | 51.5% / 51.1% | 52.5% | 54.8% / 79.1% | +0.017 / +0.049 |
| native packets | 51.0% / 50.1% | 49.0% | 55.4% / 78.6% | −0.003 / +0.019 |

- **Explained by the bias.**
  - The 81% net-short tilt of name-days (M10, `FINDINGS_LOG` §4h).
  - With it, the part of the breadth loss that came from being net short into a rising market (62%, `FINDINGS_LOG` §2.7).
- **Not explained by the bias.**
  - The 37.8% every-burst NVDA hit rate (lifetime summary §8). The legacy start-mid hit rate here is 75%. A more likely cause is direction-0 bursts counted as misses (not verified).
  - The +0.02 same-day correlation, which is small for correctly signed streams too.

**Not affected.**
- `hist_flow.py` (Sample A, 2017–21): Lee–Ready for type 5, at-midpoint prints dropped.
- The hidden-liquidity panels (Lee–Ready, then packets).
- Economic packets (type-5 Direction ignored).
- The point-in-time reversal §1.17 (native visible flow).
- All September 2026 work (native-sign packets).

---

## 4. Every "promising" number and what happened to it

Status: **LEAK** = look-ahead or target overlap; **IN-SAMPLE** = parameters, names or universe chosen on the evaluation data; **SIGN** = §3 bias; **FRAGILE** = not significant on correct inference or a clean re-test; **STANDS** = survived a clean test.

| # | result (as first reported) | source | defects | clean re-test | status |
|---|---|---|---|---|
| 1 | Gated intraday markout +8.76 bps at 3 min, 100% hit rate; AAPL +3.2 to +9.1 bps, t ≈ 200 | `FINDINGS_LOG` §2.6; `archive/docs/process.md` | κ gate is the forward markout | κ = 0: +0.53 bps (sub-spread; −3.6 crossing) | **LEAK** |
| 2 | Optuna AUC cls_1m 0.62–0.65; cls_clop 0.51–0.59 | lifetime §3 | best-of-search on reported names; per-burst pseudo-replication; short targets κ-filtered before 0e4079b; SIGN | never re-run | **IN-SAMPLE + SIGN** |
| 3 | Regression Optuna ρ 0.05–0.20 (reg_clop) | `main.tex` Table optuna_regression | start-mid target includes realized pre-close move; per-burst p-values; SIGN | close-mid, date-clustered: IC −0.0016 (t −0.33) | **LEAK** |
| 4 | NVDA reg_clop Sharpe 1.58, TSLA 1.57 (2023–24, $10M fixed AUM); "survives 5 bps" | `results/fixed_aum_backtest_results.md` | tuning names and years; κ upstream; D_b features; no overnight-drift/beta benchmark (NVDA rallied ~8× in-sample); SIGN | 438 OOS names: mean −0.28, 33% positive | **IN-SAMPLE** |
| 5 | LLY 2019–21 Sharpe 0.26, "universality confirmed" | `LLY_SGD_Backtest_Analysis.md` | 96% long trades in a rising stock = beta; LLY in the training universe | — | **IN-SAMPLE / beta** |
| 6 | NVDA reg_close cost-aware Sharpe 0.34 (74 trades, 2019–22) | lifetime §6 | 74 trades, one name, one of 15 grid cells | — | **FRAGILE** |
| 7 | JPM phase3_flow CLCL Sharpe 1.33 (4 names, 2019–22) | lifetime §14.9 | one of four names; NVDA −0.91 | — | **FRAGILE** |
| 8 | Tick-constrained reversal Sharpe 1.48 (t 2.96), FF5+MOM α t 3.11 | `FINDINGS_LOG` §4b, §4g | ex-post (survivor) universe; subset selected on full-period price; calibration overlaps evaluation; SIGN | walk-forward 0.79 (t 1.38, DSR 0.705 fails); PIT 2023+ +0.48 (t 0.86) | **IN-SAMPLE, then null** |
| 9 | Burst flow adds information beyond price reversal (orthogonalized 1.30) | §4f | same as #8 | PIT null | **null** |
| 10 | Count-COI → CLOP IC t −2.68 | §4m | fails HLZ; SIGN | — | **FRAGILE** |
| 11 | Overnight follow-flow +2.46 (t 4.83), Sample A 2017–21 | `main.tex` §tugofwar | confined to 2020–21 (+4.14 vs +0.55); Yahoo opens; 2022 liquidity screen (look-ahead universe); includes 15:50–16:00 flow | not re-tested with 15:50 cut or official prices | **unresolved** |
| 12 | Hidden overnight +2.44 (t 3.36), Sample B 2023–24 | `main.tex` §tugofwar | at-mid dropped; Yahoo opens | buffered mid-to-mid t 1.58 | **FRAGILE** |
| 13 | Hidden footprint +2.09 bps at 3 min, permanent to the close | §1.1 | convention-dependent | packets +0.60; not identified, bounds ±10 bps (§1.22) | **construction artifact** |
| 14 | Book-resilience +2.4/+4.9 bps (t 40–55) | §4d | forward quote in filter | depth-based +0.17 (t ~1) | **LEAK** |
| 15 | Directionless hidden markout +0.95 (t 8.4) | §4d | type-5 Direction | Lee–Ready +0.31 | **SIGN** |
| 16 | Closing-auction imbalance +9.9 to +20.1 bps | `VERIFIED_RESULTS` §2 | same-window imbalance and move | — | **LEAK** |
| 17 | Passive Sharpe −76/−140 | passive report | per-trade Sharpe annualized by trade count | per-trade capture −2.6 to −3.6 bps | **bug** (conclusion null) |
| 18 | Multi-day campaign reversal; intensity → volatility | `main.tex` conclusion; §1.16 | beta; replication of Jones–Kaul–Lipson | vol: real (visible count) | vol **STANDS**, not new |
| 19 | Fragment price discovery t 0.98–1.60; strict continuation (flow → flow) | §1.18–1.19 | — | frozen 2025 | continuation **STANDS** (no price) |
| 20 | Spread-scaling law (markout ≈ 0.63× half-spread, 0/474 clear) | §1.21 | — | frozen 2025 gates pass | **STANDS** |
| 21 | Fingerprint, program evidence, metaorder | §1.27–1.30 | — | 2021/2019 confirmations | **STANDS** (no alpha) |

---

## 5. What the clean evidence says (inputs to the revisit)

- **Bursts are real common-origin algorithms.** Same-size children recur about 2× chance, and timers are phase-locked. Run60 is the working rule.
- **Program-like flow is intermediary.**
  - It sells into index additions.
  - It follows ETF baskets.
  - Its imbalance persists less than other flow.
- **Tape-detected flow is priced.** Prices move on surprise flow, not predicted flow (§1.26), and every burst markout is about 0.6–0.7× the half-spread (§1.21).
- **Point-in-time daily flow signals are null** (§1.17, §1.29).
- **Hidden-liquidity magnitude is not identified** on packets (§1.22).

None of these conditions on **realized persistence**. That is P4's key idea: select bursts whose impact did not revert, then ask whether they carry information past the decision time and whether they look like parent-order children.

---

## 6. Never tested (the gap the revisit fills)

1. D_b ≥ κ·PeakImpact on correctly signed bursts (ratio form, κ ∈ (0,1)).
2. The P4 rule-based Phase III: informative bursts with Q_b above an order-splitting size percentile, aggregated to S_i,t, then (5.1) against CLOP and CLCL with controls. Point-in-time universe, both listings.
3. tCLOSE from a fixed decision clock (the P4 tCLOSE overlaps D_b).
4. Phase I descriptive permanence at breadth: its dependence on size, duration, peak impact, D_b, placement and state.
5. Phase II with honest timing: features known by T_dec = t_end + 10 min.
6. Submission bursts (P4's primary object) through the same D_b/permanence machinery.
7. Persistent-impact bursts against parent-order evidence:
   - internal: identical-size linkage, timers, concavity;
   - external: mutual-fund flow-induced trading, 13F changes, index events, BJZZ retail.
8. The 2012–2016 archive, NYSE-listed names in 2017–2021, and 2022–2024 at scale on native signs.

---

## 7. Consequences for the revisit design

- **Q0 reproduced the sign bias (done 2026-09-15; §3).**
  - The net-short tilt reappears only under the legacy convention.
  - The low hit rate and the weak same-day correlation do not; they need other explanations.
- **Use the P4 definitions verbatim where they are leak-free:**
  - D_b from initiation over {1,3,5,10} min;
  - ratio filter against PeakImpact;
  - permanence ratio.

  Add the firewalls:
  - T_dec = max(t_b + 10 min, t_end);
  - CLOP/CLCL only from bursts with T_dec ≤ 15:50;
  - tCLOSE only from a fixed clock.
- **Report the momentum confound explicitly.** Realized-D_b selection picks bursts whose price kept moving. Controls:
  - the stock's own open-to-decision return;
  - all-burst flow;
  - pseudo-bursts through the same filter.
- **No parameter search on evaluation data.** The legacy search counted 66 definitions and ~110 configurations, and more since, so any Q4 claim is held to t > 3 and a deflated Sharpe.
