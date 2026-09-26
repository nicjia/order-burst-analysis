# fingerprint-multiday-v1 — results

Design: `studies/fingerprint_multiday/FINGERPRINT_MULTIDAY_DESIGN.md` (frozen `49a39405cc61`; amendments A1, A2). Stage-1 tables:
`src_py/fp_multiday_extract.py`, jobs 14831007–14831010, from the p4-revisit-v1 per-burst files
(1.27 M name-days, 2012–2025). No new market data was read.

## H1 — execution algorithms carry their size fingerprint across days, on the same side

For each name and lag k, among fingerprint-bearing trade bursts (modal untruncated child size repeating inside
the burst, size not a multiple of 100), the share of cross-day burst pairs sharing the modal size, same side
minus opposite side. Primary statistic D = E(1) − mean E(30, 40, 50, 60).

| cell | period | names | D | t | bootstrap CI | names with D > 0 |
|---|---|---|---|---|---|---|
| **DEV** (exploration) | 2017–19 | 206 | +0.0203 | 6.78 | [+0.014, +0.026] | 93% |
| **TEST** (confirmation) | 2022–25 | 671 | **+0.0154** | **23.04** | [+0.0141, +0.0167] | 93% |
| VAL | 2020–21 | 411 | +0.0182 | 6.30 | [+0.013, +0.024] | 90% |
| ERA2 | 2012–16 | 446 | +0.0163 | 12.14 | [+0.014, +0.019] | 79% |

**Gate (DEV t > 3, TEST same sign t > 2): passed.** The secondary size set (≥ 10, not a multiple of 10) also
passes: DEV +0.0095 (t 7.63), TEST +0.0043 (t 7.07).

Lag profile of the same-minus-opposite excess, t-statistics:

| lag (trading days) | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|
| DEV | 9.2 | 8.6 | 5.7 | 2.5 | 1.7 | 1.9 | −0.6 | 0.8 | 1.4 | 4.8 |
| TEST | 24.1 | 21.9 | 18.4 | 17.6 | 13.2 | 7.0 | 4.2 | 4.1 | 4.0 | 3.1 |

Two-sided algorithms, round-number conventions and coincidences are symmetric in side and cancel in this
contrast; only a *directional* program that keeps its clip size produces a same-side excess that decays with
the gap between days. The total match rate (both sides) also falls with the lag — 0.23 to 0.12 over 60 days in
DEV — which is configuration persistence rather than direction.

**No quarterly recurrence.** DEV showed a bump at lag 60 (t 4.8), about one calendar quarter, which would
suggest rebalancing algorithms returning on a quarterly cycle. Extending the TEST profile to lags 40–252 shows
a smooth decay instead — t 5.2 (40), 4.8 (60), 4.3 (63), 1.8 (66), 0.3 (70), −0.8 (80), −0.2 (126), 1.1 (189),
3.2 (252). The DEV bump was noise; the excess simply dies out around 60–70 trading days.

**What is new.** Fingerprint-v1 (§1.27) established identical child sizes *within* a burst and used cross-day
matches as its chance baseline; metaorder-v1 linked bursts within 30 minutes. This is the first evidence in the
project that the same directional algorithm returns on later days, and it holds in four disjoint periods.

## H2 — is backward-linked flow institutional? Primary gate fails

The frozen label ("the (side, size) also appears on day t−1") marks 55% of fingerprint volume and is mostly
convention: neither linked nor unlinked flow loads on 13F in DEV (t 0.28, 0.21). Amendment A1 replaced it with
a label requiring the size to recur in at least 3 one-sided bursts on both days, chosen on a purity diagnostic
that uses flow structure only — the mirror (same size, opposite side yesterday) placebo gives
purity 0.23 / 0.48 / 0.61 / 0.73 / 0.85 at m = 1 / 2 / 3 / 5 / 10. With m = 3:

| cell | 13F ΔIO, L − U | mutual funds, L − U | mirror placebo (13F) |
|---|---|---|---|
| DEV | −0.066 (t −0.60) | −0.066 (t −0.82) | +0.23 (t 0.79) |
| **TEST** | +0.182 (**t 1.59**) | +0.152 (t 2.29) | +0.02 (t 0.05) |

**Gate (13F, t > 2 in TEST): fails.** The mutual-fund label passes on its own; one label out of two, with the
exploration cell flat, is not evidence. Multi-day fingerprint linkage is real (H1) but does not, by itself,
mark institutional flow.

## H4 — impact is concave in participation (passes)

Episodes: maximal runs of consecutive days on which one (name, side, modal size) key carries ≥ 3 one-sided
bursts. TEST has 342,744 episodes on 676 names (DEV 60,179 on 207); 5.6% run 2 days or more, the longest 12.
Participation φ = Q / (L · adv20) uses only the fingerprint-size volume on NASDAQ, a lower bound on the parent.

Signed impact by participation decile (TEST, bps, from the close before the episode to its last close):

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| impact | −13.8 | +5.1 | +14.4 | +15.6 | +23.0 | +25.0 | +30.2 | +29.9 | +32.1 | +18.2 |
| t | −6.2 | 2.6 | 7.7 | 8.5 | 12.4 | 13.2 | 15.3 | 14.6 | 13.9 | 6.6 |

Log-log slope of mean impact on mean participation: **δ = 0.35, name-bootstrap CI [0.12, 0.80]**, which
**excludes 1** — the pre-registered gate. Impact is concave in size, with the square-root value 0.5 inside the
interval; DEV gave 0.85 with CI [0.50, 1.33], which excludes neither. Mean impact by episode length rises from
+16.6 bps (1 day) to +37.6 (2) and +72.2 (3).

This is the shape the metaorder literature finds in proprietary broker records, recovered here from the public
tape alone. Two cautions: episodes are detected only while they continue, so stopping is endogenous, and the
first decile's negative mean shows the smallest episodes are dominated by noise.

## H5 — no decay that survives out of sample, and nothing tradable

Raw decay from the last episode day, DEV, looked like the textbook pattern: 3-day episodes gave back −45.8 bps
over the next 5 days and −73.4 over 20. **It does not replicate**: in TEST the same cells are +5.6 and −9.8.

With tradable timing (amendment A3: an episode's end is known only after a quiet day, so entry is the close of
d1+1 and returns start at d1+2), DEV showed −41.5 bps over five days for 3-day episodes (t −3.07) and −128.9
over twenty for longer ones (t −2.52). The pre-registered confirmation cell — episodes of length ≥ 3 pooled,
five-day horizon — gives **−2.2 bps in TEST (t ≈ −0.1, n 2,136)** against a gate of t < −2 and a mean below
−4 bps. **Fails.** Single- and two-day episodes and the top participation decile are flat in both cells.

Post-episode reversion is therefore not established, and no part of this study supports a trade.
