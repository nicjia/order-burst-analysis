# Project goal: well-defined bursts

User mandate, 2026-09-13 (replaces the earlier "alpha OR classification" statement of the same
day). **The goal is a well-defined burst: a definition that separates executions from one
execution program from unrelated flow, allowing a little noise. Alpha is not required.**

## What counts as success

A burst definition is well-defined when it is:

1. **Observable** — computed from the tape available at each moment, with no future prices and
   no sign-conditioned boundaries that manufacture the structure being tested.
2. **Validated against evidence of common origin on real data.** Synthetic agreement is a
   development check only; the legacy simulator's accuracy numbers do not transfer (see
   `studies/burst_information/BURST_RECONSTRUCTION_RESULTS.md`).
3. **Quantified in both directions** — how much same-origin evidence it keeps together and how
   many unrelated trades it merges in — with the noise level stated rather than assumed away.
4. **Stable** across names not used to choose it and across calendar regimes.

## What the data can and cannot identify

- **Aggressors are not labeled in LOBSTER.** Column 7 is NASDAQ MPID attribution, almost only on
  non-marketable market-maker quotes. Hidden executions have no order id. Links from a trader's
  resting order to an aggressive execution are rare.
- **Resting orders are linkable.** Replace chains (same-timestamp delete + add) are certain
  same-trader links, but only for passive activity.
- **Common origin of aggressive flow is testable statistically, not per trade.** Identical
  untruncated child sizes recur at short lags beyond chance; see `studies/fingerprint/BURST_FINGERPRINT_DESIGN.md`.
- **Retail versus institutional is not identifiable from NASDAQ lit data alone.** Most retail
  marketable orders are internalized off-exchange and never reach this book. A retail label
  needs off-exchange prints (e.g. TAQ/TRF sub-penny signing, Boehmer–Jones–Zhang–Zhang 2021) or
  broker records. On this data the achievable separation is **program-like** flow (repeated
  children of one algorithm) versus everything else.

## Settled context (do not re-derive)

- **No directional alpha from bursts.** The spread-scaling law holds out of sample, and every
  tested strategy loses. On the burst-information panel, no return model beats a zero forecast,
  and prices respond to surprise flow, not predicted flow (`VERIFIED_RESULTS.md` §1.21, §1.24,
  §1.26).
- **Flow continuation is real but small** (§1.19). Future flow is predictable; burst features
  add no consistent increment once scale is handled (§1.24).

## Current state (2026-09-13)

**Fingerprint-v1 is complete** (`studies/fingerprint/BURST_FINGERPRINT_RESULTS.md`). The real-data evidence answers
three questions:
- **Do program children exist on the tape?** Yes: identical child sizes recur about 2× chance.
- **Which burst definitions keep them together?** Uninterrupted same-side runs or 2–5 s side-only
  streams.
- **Can a burst's program-likeness be scored without sizes?** Yes, out of sample.

Absolute purity and retail/institutional labels remain unidentified on this data.

**Program-evidence-v1 and metaorder-v1 (2026-09-14, complete).**
- **Detection chain, validated and calibrated:**
  - child clusters (run60, equivalent to 5 s side-only streams);
  - a real-time program score;
  - a persistent-algorithm linkage score.
- **What it finds is mostly intermediary and arbitrage flow.** It does not predict returns, track
  mutual-fund trades or front-run index additions. It does track ETF creation and redemption
  demand.
- **Passive liquidity provision against it loses** in a queue-aware simulation.
- **Implication for the mandate.** The well-defined burst exists, but as a detector of algorithms,
  not a retail/institutional or metaorder label.

**Next, if pursued** (external labels are now the binding constraint):
- Broker parent-order data, or licensed 13F/FactSet ownership, to test institutional content
  directly.
- A phase feature in the program score (M3 showed timer-locked parents are not yet scored higher).
- Multi-venue data (TAQ is licensed) to follow a parent across venues.
- **External labels.** TAQ/TRF retail signing or broker parent records to calibrate purity.
- **Richer fingerprints.** Near-identical sizes and clock-phase alignment, to raise recall beyond
  exact size matches.
- **Order-lineage links on the passive side.** Replace chains give certain same-trader links.
