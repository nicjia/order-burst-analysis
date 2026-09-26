# daily-labels-v1 — results

Design: `studies/daily_labels/DAILY_LABELS_DESIGN.md` (frozen 2026-09-23 before any regression). Code: `src_py/daily_labels.py`.
Labels per name-day from the BJZZ TAQ extract: retail imbalance RI (subpenny identification) and
institutional-size imbalance II (trades ≥ $50k). Day fixed effects, PERMNO-clustered CR1, the **same-day return**
among the controls, regressors winsorized over their non-zero values.

## Primary — program-linked flow against its placebo: fails

| cell | II: β(program-linked) − β(mirror) |
|---|---|
| DEV | +9.84 (t 2.43) |
| **TEST** | **+0.99 (t 0.36)** |

**Gate (t > 3 in TEST): fails.** Program-linked flow (fingerprint-multiday-v1 amendment A1) is not
distinguishable from its mirror placebo on the institutional label. Together with the 13F failure in
`studies/fingerprint_multiday/FINGERPRINT_MULTIDAY_RESULTS.md`, the multi-day link is a real flow structure (H1) with no demonstrated
institutional content.

## Secondary — the informativeness filter selects *away* from institutional-size flow (replicates)

Coefficients on ADV-normalized signed burst flow, same-day return controlled:

| cell | institutional imbalance: informative | other large | difference |
|---|---|---|---|
| DEV 2017–19 (208 names, 137 k name-days) | +0.49 (t 3.93) | +1.67 (t 10.26) | **−1.18 (t −5.00)** |
| VAL 2020–21 (415 names, 191 k) | +0.34 (t 2.65) | +2.03 (t 16.16) | **−1.69 (t −8.14)** |
| TEST 2022–25 (673 names, 509 k) | +0.97 (t 12.98) | +2.78 (t 31.09) | **−1.81 (t −14.08)** |

Both classes of burst flow move with institutional-size trading on the same day, so burst flow is broadly
institutional — but the P4 informativeness filter **cuts the loading by about two thirds** in every cell, on
disjoint names and years. The ordering is the opposite of the quarterly 13F result (`VERIFIED_RESULTS.md`
§1.31), where informative flow loaded more than other large flow.

**Caveat, stated before the numbers are used.** The two regressors are complementary subsets of the same large
burst flow, so their coefficients are attenuated by their own measurement error to different degrees; the
contrast is therefore evidence about *ordering*, not about a structural magnitude. The labels also differ in
kind: 13F is a quarterly change in ownership, II is a daily execution-size imbalance.

## Retail

Informative flow is unrelated to retail imbalance in DEV (t −0.37) and TEST (t −0.67); other large flow is
positive in VAL (t 3.83) and TEST (t 4.15) but not DEV (t −0.67). The informative-minus-other contrast is
negative in VAL (t −3.76) and TEST (t −3.23) and positive in DEV (t 0.24) — two of three cells, sign-inconsistent,
**not claimed**. Lit NASDAQ burst flow remains essentially unrelated to retail activity, as expected.
