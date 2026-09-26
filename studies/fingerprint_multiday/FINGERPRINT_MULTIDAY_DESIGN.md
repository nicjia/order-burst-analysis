# fingerprint-multiday-v1 — do execution algorithms carry their fingerprint across days?

Frozen 2026-09-21, before any cross-day statistic was computed, as part of a search for burst ideas not yet
explored.

## Why this is new

Fingerprint-v1 (`VERIFIED_RESULTS.md` §1.27) showed that identical untruncated child sizes recur *within* a
burst at about twice the depth-matched rate, and used **cross-day** size matches as its chance baseline.
Metaorder-v1 linked bursts within 30 minutes. Nothing has asked whether the same algorithm returns on later
days, and whether it keeps its side. Multi-day execution is the classic signature of institutional orders
(pension and mutual-fund packages routinely span several days); intermediaries rarely carry a directional
position overnight. If a size fingerprint persists across days *on the same side*, the tape carries a
multi-day marker that needs no external data to compute.

## Data

p4-revisit-v1 stage-1 files (`results/p4_revisit_v1/out/<CELL>/<PERMNO>/<DATE>.npz`), trade bursts only: run
rule, 60 s, economic packets, native signs. A burst is **fingerprint-bearing** when its modal *untruncated*
child size repeats inside the burst (`mode_count >= 2`). Stage 1 (`src_py/fp_multiday_extract.py`) keeps, per
(permno, date, side, modal size): number of bursts, executed volume, children. It reads no prices or outcomes.

Size sets: **primary** — modal size not a multiple of 100 (the fingerprint-v1 rule); **secondary** — modal size
≥ 10 and not a multiple of 10.

Samples (name split and periods from p4-revisit-v1): **exploration DEV** 2017–19, group-0 names;
**confirmation TEST** 2022–25, groups 1–2 (disjoint names and years); then VAL 2020–21 and ERA2 2012–16 as
further replications, read after TEST. Lags are counted on the exchange trading calendar; a pair of days
enters only if both are present for the name.

## H1 — directional persistence across days (primary)

For name i and lag k trading days, over all present day pairs (t, t+k), with c_{side,t}(s) the number of
fingerprint bursts of that side and modal size s on day t, and n_{side,t} = Σ_s c_{side,t}(s):

- matches same side: P_same(k) = Σ_t Σ_s [c_B,t(s) c_B,t+k(s) + c_S,t(s) c_S,t+k(s)];
  pairs same side: N_same(k) = Σ_t [n_B,t n_B,t+k + n_S,t n_S,t+k];
- P_opp, N_opp likewise with buy paired to sell;
- directional excess E_i(k) = P_same/N_same − P_opp/N_opp.

**Primary statistic:** D_i = E_i(1) − mean(E_i(30), E_i(40), E_i(50), E_i(60)).

Why this contrast: coincidences, round-number conventions and two-sided (market-making) algorithms are
symmetric in side, so they add equally to same- and opposite-side rates. A persistent *side-asymmetric* size
convention would give an excess that does not decay with k. Only multi-day directional programs give an excess
that is large at one day and gone by 30–60 days. Price drift that changes share counts of dollar-sized clips
lowers all rates with k, on both sides alike.

**Inference:** one D_i per name with at least 100 present day pairs at lag 1 and at each far lag; cross-name
mean, t-statistic and a name bootstrap (2,000 draws).
**Gate:** mean D > 0 with t > 3 in DEV; same sign with t > 2 in TEST. Secondary size set and the profile
E(k), k = 1, 2, 3, 5, 10, 20, reported descriptively.

## H2 — is backward-linked flow institutional? (run only if H1 passes TEST)

Linked flow L_{i,t}: signed volume on day t of fingerprint bursts whose (side, modal size) also appears on day
t−1 for the same name — observable at the end of day t. Unlinked flow U_{i,t}: the rest of the fingerprint
signed volume. Daily values divided by adv20, summed within the calendar quarter (≥ 20 days).

Regression (p4-revisit-v1 conventions: quarter fixed effects, PERMNO-clustered CR1, 1/99% winsorization):
13F ΔIO on qL, qU, previous-quarter return, log cap, turnover **and the same-quarter return** (the control the
p4 post-hoc check showed is required for any price-related selection). Mutual-fund Δholdings as the second
label.
**Gate:** β_L − β_U > 0 with t > 2 in TEST, for 13F; mutual funds reported alongside. 13F quarters need median
filers ≥ 100 (p4 amendment A6).

## H3 — does linked flow predict flow and returns? (descriptive unless H1 passes)

Next-day signed fingerprint flow on L_t and U_t (daily Fama–MacBeth, NW(10)); next-day close-to-close and
next-5-day returns on L_t and U_t with own-return, size, turnover and volatility controls. Hurdle for any
return claim: t > 3 in DEV and same sign t > 2 in TEST, net of a 2 bps per side cost in any portfolio version.

## What would count as a finding

H1 passing in TEST is a structural finding about algorithmic execution (the tape carries multi-day fingerprints).
H2 passing in TEST would make it the tape-only institutional marker the project has lacked. A failure of either
is reported as it falls.

---

## Amendment A1 — 2026-09-21, after H1 was confirmed on TEST and before H2 was run on TEST

**What was seen.** H1 passed its gate in DEV (D +0.0203, t 6.78) and TEST (D +0.0154, t 23.04; 671 names),
and replicated in VAL (t 6.30) and ERA2 (t 12.14). H2 as written was run on DEV only. Its link label ("the
(side, size) also appears on day t−1") marks 55% of fingerprint volume as linked, and neither linked nor
unlinked flow loads on 13F (t 0.28, 0.21). Common sizes recur daily on both sides, so the label mostly marks
conventions.

**Diagnostic (flow structure only, no prices, 13F or returns).** Compare *same-side* links with *mirror*
links, where the same size traded on the opposite side the day before and only on the other side today.
Coincidences are side-symmetric, so purity = 1 − mirror volume / same-link volume estimates the program share
of the label. On DEV (`src_py/fp_multiday_purity.py`): rarity filters leave purity at 19–24%; requiring
the size to recur in at least m one-sided bursts on both days gives purity 0.23 / 0.48 / 0.61 / 0.73 / 0.85 for
m = 1 / 2 / 3 / 5 / 10.

**Amended H2 label.** Program-linked flow L_{i,t}: signed volume of day-t bursts with (side, size) such that
the size appears in at least **m = 3** bursts on that side and none on the other side, on both t−1 and t.
Mirror flow M_{i,t}: the same with the side switched overnight (placebo). U_{i,t}: all other fingerprint flow.
Regression as before (13F ΔIO; same-quarter return control; quarter FE; PERMNO-clustered), with qL, qM and qU.
**Gate unchanged in spirit:** β_L − β_U > 0 with t > 2 in TEST for 13F. β_L − β_M reported as the placebo
contrast. m = 5 reported as a sensitivity. Nothing about H1 changes.

---

## Amendment A2 — 2026-09-23, H2 outcome and the episode study (H4, H5) fixed before any episode outcome was computed

**H2 outcome (TEST, amended label, same-quarter return control).** Program-linked flow beats other fingerprint
flow on 13F ΔIO by +0.182 but at **t 1.59**, below the pre-registered t > 2: the primary institutional gate
**fails**. Mutual-fund Δholdings: +0.152 (t 2.29), passes. The mirror placebo is flat in both (t 0.05, −1.19).
DEV showed neither (t −0.60, −0.82). Reported as a weak, one-label result, not an institutional marker.

**Episodes.** For name i, side σ and size s, an episode is a maximal run of consecutive present trading days on
which the key carries at least 3 one-sided bursts (the A1 label). Episode length L days, executed fingerprint
volume Q (NASDAQ, this key only, a lower bound on the parent), participation φ = Q / (L · adv20).

### H4 — impact shape

Signed impact I = σ · (cumulative CRSP return from the close before the episode to the close of its last day),
in bps, and in units of the name's 20-day volatility. Episodes binned by φ decile within cell; per bin, mean I
and mean φ. Estimate δ in log mean-I = const + δ · log mean-φ across bins (bins with positive mean I).
**Square root** predicts δ ≈ 0.5, **linear** δ ≈ 1. Reported with the bin table; gate: δ CI (name bootstrap,
2,000 draws) excludes 1 in TEST, and the L ≥ 2 profile is concave.

### H5 — after the episode

Signed cumulative abnormal return (stock minus the value-weighted universe) over the 1, 5 and 20 trading days
after the last episode day, by φ decile. Metaorder literature expects partial decay of the peak impact.
Any trading claim needs t > 3 in DEV, same sign t > 2 in TEST, and to survive 2 bps per side.

Both are new questions on cells already read for H1/H2; they are exploratory in DEV and confirmed in TEST.

---

## Amendment A3 — 2026-09-23, fixed before the TEST episode file existed

H5 measured decay from the last episode day, but an episode's end is only observable once a day passes with no
continuation. The tradable version enters at the close of d1+1 and measures from d1+2
(`src_py/fp_multiday_h5_tradable.py`). DEV, signed by the programme's side: episodes of length 3 give
−41.5 bps over five days (t −3.07, n 481) and −61.2 over twenty (t −2.29); length ≥ 4 give −128.9 over twenty
(t −2.52, n 116); lengths 1 and 2 and the top participation decile are flat (|t| ≤ 1.5).

Ten cells were inspected, so the DEV t of −3.07 is worth about t 2.4 after a Bonferroni correction, and DEV is
the exploration sample in any case.

**Confirmation cell, fixed now:** episodes of length **≥ 3 pooled**, five-day horizon, tradable timing, signed
by side. **Gate:** negative with t < −2 in TEST, and the mean below −4 bps so that a 2 bps per side round trip
leaves something. Twenty-day horizon and the length profile are reported alongside. A pass would be the first
tradable-looking result in this project to clear a pre-registered out-of-sample gate; it would still need a
portfolio implementation with costs and capacity before being called a strategy.
