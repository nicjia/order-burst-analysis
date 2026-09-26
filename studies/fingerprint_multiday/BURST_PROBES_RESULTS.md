# burst-probes, 2026-09-23 — the remaining ideas from the 2026-09-23 list

Six further ideas, each explored on DEV (2017–19, group 0) and, where exploration was not null, confirmed on
TEST (2022–25, groups 1–2 — disjoint names and years). Code: `src_py/fp_crossname.py`,
`probe_score_linkage.py`, `burst_feature_probes.py`, `probes_retail_vol.py`. Outputs under
`results/burst_probes_v1/` and `results/fp_multiday_v1/`.

## Confirmed

### Cross-name burst flow predicts the next minute, and it is not stale prices (idea 6)

LLM_README §10 listed cross-sectional lead–lag as untested. Per date, every covered name's minute midpoint grid
and trade bursts: f[i,m] is signed burst volume started in minute m over the name's burst volume that day;
P[i,m] is the mean f over the *other* names. Pooled over (name, minute) with name-day fixed effects and
standard errors clustered by date (jobs 14879698/99):

| | own flow | **peer flow** | own lagged return | peers' contemporaneous return |
|---|---|---|---|---|
| DEV (754 dates, 53.8 M obs) | +64.1 (t 4.5) | **+878 (t 8.2)** | −0.003 (t −15.9) | +0.005 (t 0.69) |
| TEST (998 dates, 200.3 M obs) | +106.9 (t 18.0) | **+1128 (t 16.2)** | −0.067 (t −5.3) | +0.018 (t 0.75) |

The obvious alternative is non-synchronous trading: a name whose midpoint has not updated catches up to the
market's last move. That is ruled out directly — peers' *returns* in the same minute carry no predictive power
at all (t 0.69, 0.75), while peers' *flow* does. A one-standard-deviation move in peer flow (σ ≈ 0.00097) is
worth **≈ 1.1 bps** of next-minute return in TEST and 0.85 in DEV. That is below a typical half-spread for
these names, so it is a measurement result, not a trade — the spread-scaling law (§1.21) again.

### Multi-day programmes move price more than single-day ones at equal participation (idea 8)

Episode impact on a multi-day dummy, log participation and log volatility, day fixed effects, PERMNO clusters:
**DEV +22.7 bps (t 6.19), TEST +21.5 bps (t 5.93)**. Caveat: an episode continues or stops endogenously, and a
programme that keeps going is one whose price kept moving, so this is association, not the cost of splitting.

### The within-day program score predicts cross-day linkage (idea 4)

The program score of `VERIFIED_RESULTS.md` §1.27 was fit on *within-day* size repetition. Cross-day linkage is
a criterion it never saw. Keys with ≥ 3 one-sided bursts, classified linked / mirror / unlinked:

| | linked | mirror | unlinked | linked − mirror |
|---|---|---|---|---|
| DEV mean score | −2.397 | −2.446 | −2.465 | +0.036 (t 2.89) |
| TEST mean score | −2.316 | −2.399 | −2.455 | **+0.057 (t 7.10)** |

Two independently constructed measures of "this is a programme" agree. The gap is small in score units.

## Null — explored and closed

- **Hidden-liquidity share (idea 7).** Share of a burst's volume executed against hidden liquidity does not
  predict post-decision permanence: d_close t −0.02 on 1.28 M DEV bursts (d_open t 1.26). It does predict the
  within-burst ratio D_b/peak strongly (t −25.1), which is mechanical — trading at the midpoint moves the quote
  less. Not carried to confirmation.
- **Odd-lot bursts as a retail label (idea 5).** Name-day signed burst volume in modal sizes < 100 is unrelated
  to the BJZZ retail imbalance (t −0.13) *and* to the institutional imbalance (t 0.08), while ≥ 100-share burst
  flow loads on the institutional label (t 10.6). Lit odd-lot bursts are neither retail nor institutional-size.
- **Programme intensity as a volatility forecaster (idea 10).** Programme-linked volume does not forecast the
  next day's absolute return (t −0.48); other fingerprint volume does (t 4.71). No increment over §1.16.
- **Quarterly recurrence (idea 9).** The DEV bump at lag 60 was noise; the TEST profile decays smoothly and is
  gone by ~70 trading days.
- **Post-episode reversion (idea 2).** See `studies/fingerprint_multiday/FINGERPRINT_MULTIDAY_RESULTS.md` H5: DEV −41.5 bps over five days
  (t −3.07) became −2.2 bps (t −0.1) in TEST.
