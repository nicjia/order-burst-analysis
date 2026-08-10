# The Informational Bifurcation of Hidden Liquidity

Measuring the adverse-selection content of non-displayed executions from raw NASDAQ ITCH
messages, on a 474-name panel over 2023–2024.

**Paper: [`paper.pdf`](paper.pdf) — 21 pages. Source: [`paper.tex`](paper.tex).**

---

## The finding

Hidden liquidity is usually treated as one object with one informativeness number. It is
neither. Splitting LOBSTER type-5 (non-displayed) executions by where they print relative to
the midpoint separates two economically distinct populations:

| Hidden execution subset | 3 min | 15 min | 30 min |
|---|---|---|---|
| **Aggressive** (away from the midpoint) | **+2.09** | +2.24 | +2.25 |
| **At the midpoint** (tick-signed) | **−0.42** | −0.64 | −0.71 |

*Signed midpoint markout, bps. 474 names, 199,627 name-days.*

Pooling them — which every classifier that force-signs midpoint prints must do — averages an
informed population against an uninformed one and returns a statistical zero. That is why the
four standard trade-sign rules disagree about hidden-liquidity informativeness by two full
basis points on identical data (+2.09 under quote-rule abstention, +0.03 under EMO).

This is the self-selection equilibrium of Zhu (2014), measured *within* one venue and one
order type — venue, fee schedule, and clientele held fixed, only execution aggressiveness
varying — rather than across venues, where self-selection is confounded with venue effects.

**The aggressive footprint behaves as an adverse-selection quantity.** It is a stable ~⅓ of
the quoted half-spread (cross-name correlation +0.85; ratio centred on 0.34, IQR 0.28–0.42;
across spread quintiles spanning 5× in width the share moves only 0.29→0.38). Realized
volatility predicts it univariately but collapses to zero (t = 0.2) once spread is controlled.
It is 51% of the effective half-spread on the Huang–Stoll accounting, incremental to
Cont–Kukanov–Stoikov order-flow imbalance, and retains 94% of its value after excluding
earnings windows and large-move days.

**What the paper does not claim.** Causality is withdrawn — 47.5% of the signed move precedes
the print on unambiguously-signed trades. Displacement is not excluded — 30% of aggressive
prints are followed by the touch being swept within 100 ms, and that minority carries 75% of
the measured level. And because LOBSTER reconstructs the NASDAQ book alone, price discovery
cannot be separated from a local midpoint catching up to a consolidated quote. The
bifurcation is immune to all three (it is a contemporaneous comparison on one feed); the
magnitudes are provisional on them.

---

## Two documents

| File | What it is |
|---|---|
| **`paper.tex` / `paper.pdf`** | **The submission draft.** 21 pages, 20 tables. Only results that survived scrutiny. |
| `main.tex` / `main.pdf` | The full internal record, ~55 pages. Every finding including the earlier burst/reversal strategy work, retained deliberately and not cut. |

The two differ in scope, not in facts. `main.tex` preserves the project's full history —
including a cross-sectional reversal strategy that failed deflated-Sharpe adjustment, was
built on an ex-post universe, and had overlapping calibration and evaluation windows. That
work is not in the submission draft and is not claimed as a result.

---

## Reproducing the tables

Every table in `paper.tex` maps to a script. All extraction scripts take
`--msg <message_file> --ticker <TK>` and emit one CSV row per ticker-day; the `hoffman2/*.sh`
drivers fan them across the panel as SGE array jobs.

| Paper table | Extractor | Aggregator |
|---|---|---|
| `tab:signing`, `tab:signing_quartile`, `tab:decomp`, `tab:regime_ts` | `src_py/hidden_emo_clnv.py` | — |
| `tab:tickplacebo`, `tab:decomposition` | `src_py/hidden_tickplacebo.py` | `src_py/agg_tickplacebo.py` |
| `tab:signrobust` | `src_py/hidden_signrobust.py` | — |
| `tab:hidden_term`, `tab:vargrid` | `src_py/hidden_final.py` | `src_py/agg_final.py` |
| `tab:spread`, `tab:scaling` | `src_py/hidden_spread_decomp.py` | `src_py/footprint_determinants.py` |
| `tab:ofi` | `src_py/hidden_vs_ofi.py` | — |
| `tab:news` | `src_py/pull_earnings.py` | — |
| `tab:depletion` | `src_py/hidden_depletion.py` | — |
| `tab:sweep` | `src_py/hidden_sweep2.py` | — |
| `tab:preprint` | `src_py/hidden_preprint.py` | — |
| `tab:hasbrouck` | `src_py/hidden_hasbrouck2.py` | — |
| `tab:multiday` | `src_py/multiday_power.py` (runs locally, no cluster) | — |

`src_py/burst_alt.py` is the shared BBO reconstruction module every extractor imports:
`reconstruct(msg_path) → (bt, bmid, bb, ba, bbsz, basz, ofi, trades)`, plus `mid_at` and
`bbo_at` for time-indexed lookup.

Inference convention throughout: the **day** is the unit, markouts are equal-weighted within
name-day then averaged cross-sectionally, and *t*-statistics are Newey–West with 10 lags on
the daily-mean series. Name-days whose markout exceeds 1,000 bps in absolute value at any
horizon are dropped (56 of 221,261 on the primary panel).

---

## Four ways this measurement misleads

Documented because each cost a full run to discover, and each looks correct until tested:

1. **A VAR reporting horizons beyond its own memory.** A VAR(12) on a 10-second clock carries
   120 seconds of memory, so 3-, 10- and 30-minute responses all return its asymptote — the
   same number three times (+0.043, +0.047, +0.047). The original "permanent impact" result
   was that fixed point. Ridge shrinkage is irrelevant; only the sampling clock matters.
2. **A non-overlapping sort whose sign depends on sampling phase.** The 20-day reversion
   estimate spans −36.2 to +25.5 bps across starting offsets of the *same data*. A
   calendar-time portfolio with overlapping holds gives −1.95 bps (t = −0.40).
3. **A sweep test whose conditioning window overlaps its outcome.** Conditioning on the quote
   moving within 1 s and then measuring a markout that starts at the print gives an 18×
   split; separating the windows collapses it to zero.
4. **A tick-rule placebo that cleared rather than convicted.** The at-midpoint negative
   markout looked like it could be manufactured by tick-rule conditioning. A matched placebo
   recovers 5.4% of it and is statistically zero.

---

## Infrastructure

- **Data**: LOBSTER message files, streamed per ticker-day from `lobster2.math.ucla.edu` and
  reconstructed in memory. Roughly 40% of ticker-days are absent from the staged archive.
- **Compute**: UCLA Hoffman2, SGE array jobs — one task per ticker, `xargs -P6` over its dates.
  A full 474-name pass over 2023–2024 takes 3–6 hours at 50 concurrent tasks.
- **`src_cpp/`**: the original C++ burst detector (a self-exciting decaying counter used as a
  clustering heuristic; the excitation increment is fixed at unity as a normalization, and the
  decay rate and threshold are the two free parameters). Not load-bearing for the hidden-liquidity
  results, which do not use it.
- **`archive/`, `passive/`, model-zoo scripts**: earlier phases — Optuna calibration, online-SGD
  walk-forward backtests, passive-burst analysis. Superseded, retained for provenance.

## Limitations that are not going away

Single-venue signing (no SIP/NBBO), a universe that is not point-in-time (delisted names are
absent from the staged archive rather than truncated), and no trader identities — bursts are
anonymous message-level events, not parent orders. Each is stated in `paper.tex` §8 with what
it does and does not affect.
