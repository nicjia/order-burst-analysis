Let me pull the key numbers from the referee-mandate test logs that earlier agents ran (ablation, baselines, Poisson, cost grid, etc.) so the summary is grounded in actual results rather than filenames.

This pulls the earlier referee-mandate tests into view — and they change the picture in an important, more *positive* way than the overnight-COI panel alone suggested. Here's the complete summary.

---

# Order-Burst Analysis — Full Project Summary

## 1. The question
Do intraday limit-order-book **submission bursts** (temporally clustered, same-side aggressive orders) carry information that predicts future price moves? The paper detects bursts from LOBSTER data via a C++ engine, gates them down to "informational" bursts through a geometry cascade (fractional-ADV volume floor, directional consistency, volume ratio), and tests whether the surviving flow predicts returns.

## 2. Infrastructure & data (built by prior agents)
- **C++ parser** on UCLA Hoffman2 reconstructs BBO and clusters bursts (Hawkes β=1.0 fixed); Python layer computes permanence labels + microstructure features.
- **Universe:** 500-name NASDAQ list → **482 tickers** with valid burst+permanence data (18 truly absent from LOBSTER), **2022 → early-2026**.
- **Prices:** CRSP extended with 2025–26 daily OHLC (fetched 6/28, 500/500). Permanence relabeled through 2026 (the ~8h step — done and safe).
- **Anti-bias design:** physical params tuned by Optuna on a 40–50 name TRAIN set only; κ (impact filter) applied to training windows only, never to prediction candidates.

## 3. All tests conducted, and their results

### A. Per-burst markout decay — *the core signal* (referee R1) — **STRONGLY POSITIVE intraday**
Directional markouts on AAPL (representative), **gated/informational bursts**:

| Horizon | Mean | t-stat | Win% | Naive Sharpe |
|---|---|---|---|---|
| 1 min | +3.16 bps | 211 | 77% | 11.4 |
| 3 min | +5.53 bps | 249 | 83% | 13.4 |
| 5 min | +7.11 bps | 263 | 86% | 14.1 |
| 10 min | +9.12 bps | 233 | 82% | 12.5 |
| **CLOP (overnight)** | **+2.66 bps** | **4.1** | 51% | 0.22 |
| **CLCL (next close)** | **−3.72 bps** | **−5.0** | 48% | −0.27 |

Gating gives a huge uplift over raw bursts (raw 3m ≈ +0.9 bps vs filtered +5.5). **The signal is concentrated at minutes-to-hours, decays to marginal by the overnight gap, and reverses by the next close.** ⚠️ The 11–14 Sharpes are per-burst-markout artifacts (overlapping windows, no capacity/costs) — real *sign & decay structure*, not tradable Sharpe. They need bootstrap CIs (R1) before any tradability claim.

### B. Time-of-day stratification (B9) — morning strongest
Early (09:30–10:30) bursts have the largest markouts (1m: +1.23 vs midday +0.94 vs close +0.82); the burst→close markout is **+7.1 bps (t=6) for morning** but **−5.4 (midday)**. Predictive content is front-loaded to the open.

### C. Poisson null-model test (B2) — **bursts are real**
Both raw and filtered bursts reject the homogeneous-Poisson null (p≈0, over-dispersed, CV>1). Bursts are genuine clustering, not arrival-rate artifacts. Clean positive methodological result.

### D. Cross-sectional COI panel — *the referee headline (B3/B4/R6)* — **NULL overnight**
Full 482-name universe, 2026 data (my fresh runs today):

| | Fama-MacBeth COI | Long-short Q5−Q1 |
|---|---|---|
| Ungated (M7 baseline) | t = −0.62 | +0.21 bps/day, t=0.17 |
| Gated (COI^info thesis) | t = −1.36 | +0.77 bps/day, t=0.41 |

Gating doesn't rescue it. Per-name IC is a **weak, pervasive reversal** (mean IC −0.006 to −0.007, t = −3.4 to −4.2 across variants). Count-based COI (B11) ≈ volume; CLCL horizon *more* negative; **no tick-regime structure** (corr price↔IC p=0.96). The R3 sign-conditional reframe does **not** hold at breadth.

### E. Direction ablation (M2) — Direction is *not* the dominant feature
On overnight CLOP, full-model Spearman ρ ≈ 0.027; removing Direction *raises* it to 0.038; Direction-only ρ = −0.13. So overnight the ML signal is weak and Direction isn't carrying it. (Refutes the "it's just a sign classifier" worry — but on a horizon where the model barely predicts anyway.)

### F. Per-name adaptive SGD backtest — **mostly negative at breadth**
The paper's actual strategy (overnight MOC-in/MOO-out). The 65-name subset that ran averaged **Sharpe −0.29**; GOOGL −0.53. It trades the *overnight* horizon — the decayed part. Full 482-name sweep is **running now (job bt26)** to confirm.

### G. Multiple-testing / Deflated Sharpe (M3/M4)
On the research subset (100 trials × 39 results): **41% survive Bonferroni, 44% Holm.** Lo(2002) SE + DSR are computed per-name inside the backtest (GOOGL's overnight Sharpe did *not* survive DSR).

### H. Transaction costs (M6/B8)
Flat 0–5 bps grid + Almgren-Chriss scaffold. Raw intraday markouts (~0.9 bps at 1m) turn net-negative above ~1 bps round-trip; filtered/gated markouts (3–9 bps) survive 1 bps but are eaten at 3–5 bps.

## 4. The synthesis
**The alpha is intraday and decays overnight.** Gated informational bursts strongly predict same-direction moves over 1–60 minutes (front-loaded to the open); by the overnight gap the effect is marginal (per-burst t≈4 but null cross-sectionally); by next close it reverses. The paper's headline strategy trades the **overnight** horizon — precisely where the signal has decayed — which is why per-name Sharpes are ~0 and the cross-sectional COI is null. The referees anticipated exactly this (R1: "the author punted on burst-to-close… the markout at that horizon was weak"; R4: "alpha is in the overnight gap and gets eaten during the next session").

## 5. What changed this session & current state
- Root-caused & fixed the systemic backtest failure (ADV table covered 111/483 names → rebuilt to all 483).
- Found the panel was testing **ungated** flow, implemented **gated COI^info**, ran gated-vs-ungated → null (above).
- Exhausted the positive-continuation variants (counts, horizons, tick-regime) → all negative overnight.
- **Live now:** job **bt26** (full 482-name SGD sweep) running; monitor will report the Sharpe distribution.

## 6. The live positive thread
The strong **intraday markout** result (Section A) is where the paper's real, defensible alpha lives — the referees explicitly asked for that markout panel as the first-class diagnostic. The natural next move is to build the **multi-horizon markout panel with bootstrap CIs across the full universe** (not just AAPL) and reframe the paper around the intraday decay curve, treating overnight as the decayed tail rather than the headline. That's a genuinely positive, referee-aligned story the data supports — unlike the overnight cross-sectional version.

Want me to build that full-universe markout panel next (it's the honest home of the positive result), or wait for bt26 first?

The bt26 monitor fired. Let me read the sweep result.

The full sweep completed cleanly — **the fixes worked: 438/438 names ran, 0 failures** (vs 65 before). And it **corroborates the null**: mean Sharpe −0.28, median −0.27, only 33% positive, range [−3.32, +1.25].

Two things I want to verify before I report: (1) that these are real walk-forwards, not fast degenerate exits, and (2) crucially — **do the paper's headline names (NVDA/TSLA) still show their strong Sharpes on the 2026 data, or did they collapse too?** Let me check.