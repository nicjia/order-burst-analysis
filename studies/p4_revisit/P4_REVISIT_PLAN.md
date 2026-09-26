# P4 revisit: informed bursts, done without leakage — plan (draft for user review)

Written 2026-09-14 after reading the original proposal (`UCLA_279__P4_informed_bursts.pdf`,
v1 Jan 23, 2026), `archive/docs/PROJECT_LIFETIME_RESULTS_SUMMARY_20260407.md` and `archive/docs/FINDINGS_LOG.md`. **No new
work has been run for this plan.** It needs the user's decisions (last section) before anything is
frozen.

## 1. The original idea, restated

- **Institutions split parents into small children.** A cluster of same-side child orders (a burst)
  is the visible trace of a parent.
- **Reason backwards from price impact.** A burst whose impact *persists* (does not revert within
  minutes) is more likely an informed parent's child than liquidity-driven noise.
- **Pipeline (P4):**
  - PeakImpact(b), and the decay D_b over 1, 3, 5 and 10 minutes;
  - keep informative bursts (D_b ≥ κ · PeakImpact);
  - Phase I: describe permanence to the close, the next open and the next close;
  - Phase II: predict permanence from early information;
  - Phase III: aggregate predicted-informative signed size into S_i,t and test it against
    tCLOSE, CLOP and CLCL returns.
- **Goal from the start:** detect who is trading what, and whether it carries long-horizon
  information.

## 2. What was actually done, and where it leaked

| P4 element | what was run | reported result | defect found later | status |
|---|---|---|---|---|
| Burst detection | C++ silence/Hawkes clusters; Optuna-tuned silence, ADV fraction, direction ratio, κ | — | Direction from price against the contemporaneous mid can manufacture bursts (formation circularity) | superseded by native-sign packets |
| κ filter | kept D_b ≥ κ, with D_b = forward 1–10 min markout from burst end | intraday 3-min markout +8.76 bps, 100% hit rate (κ = 0.5) | **D_b is the forward markout, so gating on it is circular.** Honest κ = 0: +0.5 bps, below the spread (`FINDINGS_LOG` §2.6) | intraday edge retracted |
| Phase I | walk-forward logistic AUC, Optuna over physical parameters | cls_1m 0.64; cls_close 0.55; cls_clop 0.56 | κ filtering applied before short targets (commit 0e4079b); parameters selected on evaluation data | inflated; never re-run cleanly |
| Phase III overnight | cross-sectional flow panel (482 names); per-name online SGD MOC→MOO | FM t = −0.62 ungated, −1.36 gated; SGD Sharpe −0.28 on 438 OOS names | flagship names (NVDA, TSLA, JPM, MS, LLY) in training; 62% of the loss was short beta | null |
| Reversal strategy | tick-constrained reversal | OOS Sharpe ~0.8 | ex-post universe; calibration overlapping evaluation | dead on a clean design (§1.17) |
| Daily program imbalance (2026-09) | program-score-filtered flow → next-day returns | null in 2024 and 2021 | filter was program-likeness, not permanence | null (§1.29) |

**What has never been run cleanly** (the gap this plan fills):
1. **Realized-D_b filter for close and overnight horizons.** A burst ending by 15:50 has a known D_b
   at the close, so an overnight signal built from those bursts is **not** look-ahead. It was
   never evaluated with every other firewall in place (point-in-time universe, forward-only
   parameters, controls for the stock's own intraday return).
2. **The κ · PeakImpact normalization** of P4 eq. 3.3. The C++ used D_b ≥ κ.
3. **Phase II with honest timing** (features available by the decision time only).
4. **Evidence that persistent-impact bursts come from parent orders.** Internal (linkage into
   same-side campaigns) and external (institutional and retail proxies).
5. **The NYSE-listed half of the archive.** On 2024-06-03: 794 NYSE-listed and 731 NASDAQ-listed
   common stocks. For NYSE-listed names the book is NASDAQ's venue; 1,144 NYSE tickers, including
   LLY, UNH and JPM, are still pending download.
6. **Submission bursts** (P4 §2 primary object: same-side limit-order submissions).

## 3. Definitions (to freeze before any outcome is computed)

**Events, two parallel families:**
- **Trade bursts** (P4 §6):
  - economic packets with native ITCH aggressor sign, clustered by the validated run rule
    (same side, ≥ 3 packets, pause < 60 s);
  - the original silence rule (0.5, 1, 2 s) for continuity with the April work.
- **Submission bursts** (P4 §2): same-side limit-order adds within a silence window and within
  k ticks of the touch, excluding ITCH replace halves.

**Per burst b** (side s, start t_s, end t_e):
- signed size Q_b = s × volume;
- reference mid m_ref = mid just before the first event;
- **PeakImpact** = max over τ ∈ [0, t_e − t_s + 5 s] of s × (m(t_s + τ) − m_ref), signed and
  floored at one tick;
- **D_b** = mean over h ∈ {1, 3, 5, 10} min of s × (m(t_e + h) − m_ref);
- **informative** ⇔ D_b ≥ κ · PeakImpact;
- **decision time** T_dec = t_e + 10 min. Nothing about b may be used before T_dec.

**Permanence** at x ∈ {close_t, open_t+1, close_t+1}: s × (x − m_ref) in bps, and the P4 ratio
winsorized at small PeakImpact.

## 4. Leakage firewalls (non-negotiable)

1. **Time ordering.**
   - *CLOP and CLCL signals:* bursts with T_dec ≤ 16:00 only. Entry in the closing auction at c_t,
     exit at open_t+1 or close_t+1.
   - *tCLOSE signal:* a fixed decision clock (e.g. 15:30) with bursts whose T_dec ≤ 15:30. The
     return runs from the mid at 15:30 to the close, **never from t_b** (the original tCLOSE
     overlapped D_b).
2. **Parameters forward-only.** κ, the cluster rule, size percentile, θ and model hyperparameters
   are chosen on the training period only. There is no Optuna or grid search on evaluation data.
3. **Samples.** Train and test are split in time **and** in names (sha256 split). A second era
   (2019) is read once.
4. **Point-in-time universe.** CRSP top-N by dollar volume at the prior year end, including later
   delistings, intersected with archive coverage; the coverage bias is reported (it was large in
   the 2026-09 panels).
5. **Controls that separate burst information from return momentum** (the κ filter selects bursts
   whose price kept moving):
   - the stock's own open-to-15:50 and last-hour return;
   - unfiltered signed flow;
   - all-burst flow;
   - market and short-term reversal factors, volatility, volume, spread.
6. **Placebos.**
   - *Pseudo-bursts* at random times on the same name-day, passed through the same
     D_b ≥ κ·PeakImpact filter. If they "predict" as well, the effect is return momentum, not
     bursts.
   - *Sign-flipped* bursts.
   - *A deliberately leaky version* (D_b measured after the close) to confirm the pipeline can
     detect leakage.
7. **Portfolios and inference.** Dollar- and beta-neutral portfolios; auction execution costs for
   CLOP and CLCL, half-spread for tCLOSE. Pre-registered grid size, HLZ t > 3 in training, deflated
   Sharpe on test.

## 5. Questions, in order, with go/no-go gates

| step | question | data | gate to continue |
|---|---|---|---|
| **Q0** | Reproduce the April result, then remove each leak in turn and show how much of it survives | NVDA/TSLA/JPM/MS 2019–2022 archive | written audit table: every promising number labeled survive / leak |
| **Q1** (Phase I) | Does κ-filtered impact persist **after T_dec**, beyond unfiltered bursts and pseudo-bursts? | training years | post-T_dec displacement (to close and to next open) filtered − unfiltered and filtered − pseudo-burst, \|t\| > 3 |
| **Q2** (link to parents) | Are persistent-impact bursts children of parents? | training years | (a) linkage: excess same-side identical-size links (metaorder-v1 M2) above unfiltered; (b) external: loads on institutional proxies above unfiltered, not more on retail |
| **Q3** (Phase II) | Can permanence be predicted from information available by T_dec? | training → test | out-of-sample rank IC > 0 with CI |
| **Q4** (Phase III) | Does S_i,t = Σ informative Q_b predict tCLOSE, CLOP or CLCL beyond controls? | train → test (and 2019 once) | FM t > 3 train, same sign t > 2 test; neutral portfolio net of costs |
| **Q5** | Where does it live? | test | NASDAQ- vs NYSE-listed, liquidity terciles, tick-constrained names |

The user has said a null Q4 is acceptable if Q2 separates institutional from other flow. Q2 is
therefore a standalone deliverable, not a stepping stone.

## 6. How the classification would be justified (Q2 in detail)

There are no per-burst labels on any venue without broker IDs, so the claim must rest on
convergent, pre-registered, replicated aggregate evidence:

1. **Internal: splitting.**
   - Persistent-impact bursts should belong to multi-burst same-side campaigns more often (identical
     child sizes and whole-second timers across bursts, measured against chance).
   - Their impact should be concave in campaign size.
2. **External: institutional proxies**, relative to unfiltered flow:
   - buying before S&P 500 and Nasdaq-100 additions (the 2026-09 post-hoc lead was in large,
     book-sweeping bursts);
   - mutual-fund flow-induced trading (CRSP, available);
   - quarterly 13F changes (public SEC data; needs a download OK);
   - Russell reconstitution names (public lists).
3. **External: retail proxies.** Filtered flow should **not** track BJZZ off-exchange retail
   imbalance (TAQ, available), nor attention and meme days.
4. **Outcome.**
   - All three hold and replicate out of sample: "persistent-impact bursts are
     institutional-parent flow", at the stock-day or stock-quarter level.
   - They fail: we say so, and the impact criterion does not identify institutions either.

## 7. Data and compute

- **Price data.** CRSP daily ends 2024-12-31, so every return test uses years ≤ 2024. The archive
  runs to 2026.
- **Proposed samples:**
  - train 2022–2023, name split 0;
  - test 2024, name splits 1–2;
  - second era 2019, read once;
  - a first Q0/Q1 look on one month of 2023.
- **Universe.** Point-in-time top 500 or 1,000 across NASDAQ- and NYSE-listed names in the archive
  (NYSE downloads still pending for some large names).
- **Extraction.**
  - 500 names × 3 years ≈ 380k name-days, reusing `contig_packets.sh`.
  - Submission bursts need a new extractor pass over raw messages.
  - Mid paths at 1 s resolution are needed for PeakImpact and D_b (a small addition to the packet
    cache).
  - All jobs as long bundled shards on `bertozzi_pod.q` (`-l highp`) to avoid the short-job
    throttle.

## 8. Decisions needed from the user

1. **Years.** Train 2022–2023, test 2024, second era 2019 — or different?
2. **Universe.** Include NYSE-listed names (NASDAQ venue only) or NASDAQ-listed names only? Top 500
   or 1,000?
3. **Events.** Trade bursts only (fast, validated), or also submission bursts (P4's primary object,
   a new extraction)?
4. **Institutional labels.** Download public SEC 13F data sets (a few GB) for Q2?
5. **Q0 first?** Reproducing the April numbers and stripping leaks one at a time costs little and
   settles which past results were real.
