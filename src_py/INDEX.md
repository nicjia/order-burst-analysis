# src_py index

Every script here backs a result in `VERIFIED_RESULTS.md`, a study document, a manuscript or a test, or is
imported by one that does. Scripts are grouped by study: by the `RESULTS_PROVENANCE.md` section that pins
them, otherwise by name. Descriptions are each file's first docstring line. The files are unchanged, so their
sha256 pins hold, and that is why they stay in one flat folder; this index is the map. Superseded scripts
are in `archive/src_py/` (see `archive/MANIFEST.md`). The burst-forecasting study keeps its own code in
`studies/burst_forecasting/code/`. Generated 2026-09-25.

## Shared modules — book reconstruction, economic packets, burst rules, fingerprint statistics, data access

| script | description |
|---|---|
| `burst_alt.py` | burst_alt.py — alternative burst definitions (NOT Hawkes; C++ data_processor untouched). |
| `execution_packets.py` | Canonical reconstruction of economic aggressive execution packets. |
| `fingerprint_packets.py` | Stage 1 of fingerprint-v1: cache one ticker-day of economic packets as a compact npz. |
| `fingerprint_state.py` | fingerprint-v1 E3, corrected: do identical-size recurrences occur in more similar book states? |
| `fingerprint_stats.py` | Stage 2 of fingerprint-v1: same-origin size fingerprints and burst-definition validation. |
| `p4_packets.py` | Vectorized economic execution packets, identical to execution_packets.reconstruct_packets. |
| `program_bursts.py` | Run/60 bursts and the stage-3b program score on one day of cached packets (vectorized). |
| `summarize.py` | — |
| `wrds_access.py` | WRDS PostgreSQL access for program-evidence-v1. |

## Hidden liquidity and the spread-scaling law — `paper.tex`; ledger §1.1–1.14, §1.21–1.23

| script | description |
|---|---|
| `agg_final.py` | agg_final.py — aggregate results/hid_fin into the three remaining referee tables. |
| `agg_tickplacebo.py` | agg_tickplacebo.py — aggregate results/hid_tp into the two referee tables. |
| `aggregate_hidden_packet_bounds.py` | Day-unit aggregation for hidden-packet identification bounds. |
| `aggregate_hidden_packet_spread_scaling.py` | Cross-name spread scaling for economic hidden-execution packets. |
| `aggregate_packet_scaling.py` | Cross-name aggregation of the packet-level spread-scaling law. |
| `audit_packet_scaling.py` | Independent recomputation and frozen-gate application for the spread-scaling law. |
| `footprint_determinants.py` | footprint_determinants.py — cross-sectional determinants of the permanent hidden- |
| `hidden_depletion.py` | hidden_depletion.py — a NON-CIRCULAR test of the mechanical quote-displacement account. |
| `hidden_emo_clnv.py` | hidden_emo_clnv.py — re-sign hidden (type-5) executions under FOUR canonical trade-sign |
| `hidden_final.py` | hidden_final.py — three remaining referee tests in ONE pass over the message stream, so |
| `hidden_freeform.py` | hidden_freeform.py — is the footprint an artifact of price-conditioned burst FORMATION? |
| `hidden_full.py` | hidden_full.py — process ONE LOBSTER message file into a daily hidden-flow row. |
| `hidden_hasbrouck2.py` | hidden_hasbrouck2.py — re-estimated Hasbrouck (1991) VAR addressing two referee objections |
| `hidden_incr.py` | hidden_incr.py — does NON-DISPLAYED execution count add anything over displayed count? |
| `hidden_mechanical.py` | hidden_mechanical.py — does the permanent footprint survive the mechanical |
| `hidden_packet_bounds.py` | Economic-packet revalidation and sign bounds for hidden executions. |
| `hidden_packet_spread_scaling.py` | Packet-level hidden-execution footprint and quoted-spread scaling. |
| `hidden_perprint.py` | hidden_perprint.py — the decisive check: does the footprint exist WITHOUT any burst at all? |
| `hidden_preprint.py` | hidden_preprint.py — does the burst move the price, or follow a price already moving? |
| `hidden_signrobust.py` | hidden_signrobust.py — stale-mid robustness for the aggressive-hidden footprint (referee |
| `hidden_spread_decomp.py` | hidden_spread_decomp.py — Huang-Stoll (1996) effective/realized/price-impact spread |
| `hidden_sweep2.py` | hidden_sweep2.py — non-circular version of the sequential-sweep test. |
| `hidden_sweep3.py` | hidden_sweep3.py — four referee analyses in one pass. |
| `hidden_term.py` | hidden_term.py — term-structure extension of hidden_full.py (referee #3). |
| `hidden_tickplacebo.py` | hidden_tickplacebo.py — two referee items on one pass over the message stream. |
| `hidden_vs_ofi.py` | hidden_vs_ofi.py — is the hidden-execution footprint INCREMENTAL to visible order-flow |
| `hidden_xsec_agg.py` | hidden_xsec_agg.py — aggregate the full-universe hidden-execution cross-section (M12). |
| `multiday_power.py` | multiday_power.py — re-run the multi-day reversion test with an efficient design. |
| `packet_spread_scaling.py` | Packet-level recomputation of the spread-scaling law (VERIFIED_RESULTS.md 1.5). |
| `pull_earnings.py` | pull_earnings.py — fetch earnings announcement dates for the 474-name hidden panel so the |

## Referee audits and robustness checks (M-series, Poisson null, time-of-day, costs, validators) — cited by the ledger, `main.tex` or the prior-work inventory

| script | description |
|---|---|
| `m10_sign_audit.py` | m10_sign_audit.py — Referee M10: sign-convention audit + net-short tilt. |
| `m4_closemid_target.py` | m4_closemid_target.py — Referee M4. (sharded for a job array) |
| `m5m6_inference.py` | m5m6_inference.py — Referee M5 (honest inference) + M6 (OOS factor alpha). |
| `m7_reversal_baseline.py` | m7_reversal_baseline.py — Referee M7. |
| `m7_signed_volume.py` | m7_signed_volume.py — Referee M7 (ii): the PLAIN signed-volume baseline. |
| `m8_costs_splits.py` | m8_costs_splits.py — Referee M8 (iii) spread-based costs + (i) split/adjustment check. |
| `multiple_testing_correction.py` | multiple_testing_correction.py — Sharpe Inference & Multiple Testing Correction |
| `naive_baseline_markout.py` | naive_baseline_markout.py — Microstructure Baseline: Unconditional Burst PnL |
| `poisson_baseline_test.py` | poisson_baseline_test.py — Reviewer B2: Poisson Null-Model Structural Test |
| `poisson_test.py` | poisson_test.py — Referee B2: is burst clustering an artifact of Poisson arrivals? |
| `posthoc_components.py` | POST HOC (2026-09-14, after program-evidence-v1 D1/D4 were read): which component of NASDAQ signed flow |
| `referee_hardening.py` | referee_hardening.py — senior-editor roadmap, matrix-based items (run on cluster). |
| `time_of_day_analysis.py` | time_of_day_analysis.py — Reviewer B9: Time-of-Day Stratification |
| `tod_coi_test.py` | tod_coi_test.py — Referee B9 (time-of-day stratification) + B11 (count vs volume COI). |
| `transaction_cost_grid.py` | transaction_cost_grid.py — Reviewer R5/B8: Transaction Cost Sensitivity Grid |
| `validate_hidden_packet_bounds.py` | Schema and conditional-finiteness gate for hidden-packet-bound rows. |
| `validate_strict_continuation_output.py` | Schema and finiteness gate for strict-continuation compact output. |

## Overnight / tug-of-war and reversal checks — §1.15–1.17 and `main.tex`

| script | description |
|---|---|
| `beta_hedged_markout.py` | beta_hedged_markout.py — Factor Regression: Alpha vs Beta Decomposition |
| `buffered_mids.py` | buffered_mids.py — extract LOBSTER intraday midpoints at 9:35 and 15:55 (plus 9:30 open |
| `dlret_splice.py` | dlret_splice.py — CRSP delisting-return splicing with a with/without TOGGLE. |
| `fig_tugofwar.py` | fig_tugofwar.py — figures/fig_tugofwar.pdf: cumulative P&L of the three session legs |
| `overnight_buffer_test.py` | overnight_buffer_test.py — does the overnight leg survive a 5-minute auction buffer? |
| `panel_regression.py` | panel_regression.py — Reviewer R3/B4/B5: Panel Regressions & COI Construction |
| `pit_flow.py` | pit_flow.py — daily flow panel for a point-in-time universe (delisted names included). |
| `pivot_returns.py` | pivot_returns.py — Build ticker×date pivot tables from daily CRSP files. |
| `pull_opens.py` | Pull split/div-adjusted Open+Close for the 493 panel names (2017-2021) from |
| `regime_classifier.py` | regime_classifier.py — Automated Microstructural Regime Classification |
| `reversion_walkforward.py` | reversion_walkforward.py — expanding-window OOS test of the tick-constrained reversal. |
| `tugofwar_2022_2026.py` | tugofwar_2022_2026.py — the referee's decisive test: run the Section 12 |
| `tugofwar_2023.py` | tugofwar_2023.py — replication test of the overnight/intraday decomposition on an |

## Figures

| script | description |
|---|---|
| `make_figures.py` | make_figures.py — Referee m5: generate the three manuscript figures as PDFs. |

## Two avenues: fragments, strict continuation, liquidity pause — §1.18–1.20, `studies/two_avenue/`

| script | description |
|---|---|
| `aggregate_liquidity_pause.py` | Name-day/Newey-West(10) aggregation for liquidity-pause-v1. |
| `aggregate_stage2.py` | Aggregate program-evidence-v1 modules C (campaigns), F (markouts), G (synchrony), H (passive). |
| `aggregate_strict_continuation.py` | Day-unit Newey-West aggregation for the frozen 2025 continuation test. |
| `aggregate_two_avenue_oos.py` | Aggregate compact 2024 name-day summaries with the project's inference convention. |
| `audit_strict_continuation.py` | Independent coverage, schema, and NW(10) audit for strict-continuation-v1. |
| `burst_quality.py` | burst_quality.py — rank burst definitions by how well they aggregate order splits. |
| `burst_quality_v2.py` | Packet-level burst validation with geometry-exact, multi-draw Hurst placebos. |
| `burst_quality_v3.py` | Sign-blind packet-cluster validation with geometry-exact Hurst placebos. |
| `fit_frozen_models.py` | Fit and freeze Avenue-1 and flow-baseline models from 2023 sampled fragments only. |
| `fit_liquidity_pause.py` | Fit and freeze the four predeclared 2023 liquidity-pause hazard models. |
| `fit_strict_continuation.py` | Fit and freeze nested continuation models from 2023 non-holdout names only. |
| `fragment_reconstruction.py` | Price-free fragment formation and outcome attachment for the two research avenues. |
| `liquidity_pause_common.py` | Frozen specification and risk-set construction for liquidity-pause-v1. |
| `liquidity_pause_extract.py` | Extract frozen liquidity-pause discrete hazard rows for one ticker-day. |
| `liquidity_pause_oos_day.py` | Apply frozen liquidity-pause hazards to one untouched 2025 ticker-day. |
| `strict_continuation_common.py` | Frozen specification shared by the strict 2023-to-2025 continuation test. |
| `strict_continuation_extract.py` | Extract price-free fragment rows for the frozen strict-continuation experiment. |
| `strict_continuation_oos_day.py` | Apply frozen nested models to one untouched 2025 ticker-day. |
| `two_avenue_evaluate.py` | Frozen temporal/name holdout evaluation for both packet-fragment avenues. |
| `two_avenue_extract.py` | Shared daily extractor for informed-flow and latent-parent research avenues. |
| `two_avenue_oos_day.py` | Apply frozen 2023 models to one untouched 2024 ticker-day and emit one summary row. |
| `validate_bq3_simulation.py` | Ground-truth stress test for the sign-blind timing-cluster definitions. |
| `validate_liquidity_pause_output.py` | Schema/finiteness gate for compact liquidity-pause OOS output. |
| `validate_liquidity_pause_rows.py` | Schema/finiteness gate for liquidity-pause risk-set rows. |

## Burst information and reconstruction — §1.24–1.26, `studies/burst_information/`

| script | description |
|---|---|
| `aggregate_markouts_trunc.py` | Aggregate post-hoc F2: program-minus-bottom markouts within burst truncation-share strata. |
| `aggregate_price_null.py` | Aggregate post-hoc module B4: identical-size matches under price-matched nulls (one group). |
| `audit_burst_information.py` | Independent CSV/weighted-day/HAC recomputation; does not import production evaluator. |
| `audit_burst_recovery.py` | Independently enumerate pair labels for every saved synthetic recovery cell. |
| `audit_join_sessions.py` | Audit persisted session-join labels and probabilities without importing join code. |
| `burst_information_extract.py` | Prospective burst landmarks and matched-state prediction labels, exploratory v1. |
| `burst_recovery_diagnostic.py` | Paired mechanism ablations. Synthetic labels are not evidence of real parent identity. |
| `collect_burst_rows.py` | Concatenate per-name burst rows into one compact file of usable rows (pairs > 0) plus coverage counts. |
| `diagnose_burst_information_v1.py` | Post-hoc diagnostics for burst-information-v1 (added 2026-09-13, after its results were read). |
| `diagnose_join_sessions.py` | Paired legacy/session-corrected join fits on new synthetic train and test days. |
| `diagnose_participation_target.py` | Audit the target change on legacy synthetic tapes; no fitting or market claims. |
| `evaluate_burst_information.py` | Frozen exploratory state/regime/burst comparisons on identical prospective decisions. |
| `ranked_burst_pilot.py` | Exploratory quote-snapshot screen; NOT a validated execution backtest. |
| `report_burst_information.py` | — |

## Fingerprint validation — §1.27, `studies/fingerprint/`

| script | description |
|---|---|
| `aggregate_fingerprint.py` | Aggregate fingerprint-v1 statistics for one group (exploration or confirmation). |
| `aggregate_fingerprint_state.py` | Aggregate the corrected E3 statistic (fingerprint_state.py): lag-held-fixed state similarity. |
| `evaluate_fingerprint_gates.py` | Apply the frozen fingerprint-v1 confirmation gates (BURST_FINGERPRINT_DESIGN.md) exactly once. |
| `fingerprint_burst_rows.py` | Stage 2b of fingerprint-v1: one row per burst, for a program-likeness score. |
| `report_fingerprint.py` | Figures and markdown tables for fingerprint-v1 summaries (exploration and confirmation). |

## Program evidence — §1.28–1.29, `studies/program_evidence/`

| script | description |
|---|---|
| `aggregate_evidence.py` | Aggregate program-evidence-v1 modules A, B and I across names (one period), with gates. |
| `analyze_daily_flow.py` | program-evidence-v1 modules D1-D3 and E1-E3 on a contiguous daily-flow panel (one year/group). |
| `analyze_events.py` | program-evidence-v1 module D4: program flow around S&P 500 and Nasdaq-100 changes (one shot). |
| `analyze_years.py` | program-evidence-v1 module J: fingerprint and burst descriptors across years; Tick Size Pilot DiD. |
| `build_event_jobs.py` | Job files for module D4: packet extraction windows around S&P 500 and Nasdaq-100 changes. |
| `build_pit_universe.py` | Point-in-time universes for program-evidence-v1 contiguous panels (modules D and E). |
| `build_year_jobs.py` | Job files for program-evidence-v1 module J (descriptive trend and Tick Size Pilot). |
| `daily_flow.py` | program-evidence-v1 modules D and E: daily program and other signed flow for one name. |
| `evidence_campaigns.py` | program-evidence-v1 module C: price paths of fingerprint-linked campaigns and matched placebos. |
| `evidence_formulas.py` | Statistics of program-evidence-v1 modules A, B and I, shared by the aggregator and the tests. |
| `evidence_markouts.py` | program-evidence-v1 module F: liquidity-provider markouts by burst program score (one ticker). |
| `evidence_markouts_trunc.py` | Post-hoc module F2 (after the F1 read): markouts by program score within truncation-share strata. |
| `evidence_price_null.py` | program-evidence-v1 module B4 (post-hoc, after the 2021 module B read): price-matched nulls. |
| `evidence_stats.py` | program-evidence-v1, modules A, B and I, from fingerprint-v1 cached packets (one ticker). |
| `evidence_sync.py` | program-evidence-v1 module G: cross-name synchrony of untruncated non-round packets (one date). |
| `hist_flow.py` | hist_flow.py — lightweight daily-flow extractor for the 2017-2021 historical |
| `program_score.py` | Stage 3 of fingerprint-v1: a program-likeness score for bursts, validated on real data. |
| `program_score_run60.py` | Stage 3b of program-evidence-v1: the fingerprint-v1 program score for run/60 bursts. |
| `report_program_evidence.py` | Figures for program-evidence-v1 (PNG for review, PDF for the manuscript). |
| `tsp_stats.py` | program-evidence-v1 module J: per-day tape descriptors for the multi-year and Tick Size Pilot panels. |
| `wrds_pull_evidence.py` | WRDS extracts for program-evidence-v1 modules D, E and G2 (cached under gitignored data/wrds/). |

## Metaorders, linkage, passive orders — §1.30, `studies/metaorder/`

| script | description |
|---|---|
| `aggregate_inject.py` | Aggregate metaorder-v1 M3 (synthetic parent calibration), 2024 exploration. |
| `aggregate_passive.py` | Aggregate metaorder-v1 M5 (queue-aware passive orders) for one group. |
| `metaorder_features.py` | Metaorder-v1 per-burst quantities for any fingerprint-v1 burst rule (vectorized, one day). |
| `metaorder_fit.py` | Metaorder-v1 M2 and M4 fits (and the rule's program score for M3), exploration 2024 -> confirmation 2021. |
| `metaorder_inference.py` | metaorder_inference.py — can the parent order be reverse-engineered from anonymized |
| `metaorder_inject.py` | Metaorder-v1 M3: inject synthetic parents into real cached packet days and measure detection (one ticker). |
| `metaorder_join_v2.py` | Session-scoped join candidates. Legacy join functions remain unchanged for provenance. |
| `metaorder_m1.py` | Metaorder-v1 gate M1: combined size and phase J for stream5 against run60, one year. |
| `metaorder_models.py` | Simulation-calibrated fragment scoring and pause-aware campaign stitching. |
| `metaorder_participation.py` | Known-truth targets for any-program participation; never infer truth from prices. |
| `metaorder_passive.py` | Metaorder-v1 M5 driver: real-time burst triggers and queue-aware passive orders for one ticker-day. |
| `metaorder_rows.py` | Metaorder-v1 stage 1 (M2, M4): one row per burst for rules run60 and stream5, one ticker. |
| `metaorder_simulation.py` | Ground-truth simulator for anonymous metaorder-fragment reconstruction. |
| `passive_extract.py` | program-evidence-v1 module H, stage 1: non-round limit-order submissions for one ticker-day. |
| `passive_stats.py` | program-evidence-v1 module H, stage 2: passive identical-size fingerprints (one ticker). |
| `queue_sim.py` | Metaorder-v1 M5: queue-aware simulation of small passive orders posted at trigger times. |

## P4 revisit: informed bursts without leakage — §1.31, `studies/p4_revisit/`

| script | description |
|---|---|
| `p4_aggregate.py` | P4 revisit v1, stage 2 pass 1: name-day statistics and a per-burst sample from the stage-1 npz files. |
| `p4_alt_tickers.py` | P4 revisit v1: retry list for name-days whose archive file is filed under a later ticker of the same company. |
| `p4_analyze.py` | P4 revisit v1, stage 3: pre-registered tests Q1, Q2(a) and Q4 on one cell (P4_REVISIT_DESIGN.md section 6). |
| `p4_coverage.py` | P4 revisit v1: archive coverage of the requested name-days (P4_REVISIT_DESIGN.md section 2). |
| `p4_external.py` | P4 revisit v1: external institutional and retail proxies for Q2 (P4_REVISIT_DESIGN.md section 6). |
| `p4_external_tests.py` | P4 revisit v1, stage 3: Q2(b)-(e) external institutional and retail proxies (P4_REVISIT_DESIGN.md section 6). |
| `p4_extract.py` | P4 revisit v1, stage 1: trade and submission bursts with P4 impact measures for one ticker-day. |
| `p4_phase2.py` | P4 revisit v1, Q3 (Phase II): predict post-decision persistence from information available at T_dec. |
| `p4_q0_legacy.py` | P4 revisit v1, Q0a: does the legacy detector's hidden-execution signing produce the old anomalies? |
| `p4_q0_report.py` | P4 revisit v1, Q0a/Q0c summary: legacy-detector anomalies under the three trade streams. |
| `p4_q2_return_control.py` | P4 revisit v1, POST-HOC robustness check (not pre-registered): does the Q2 institutional association survive a |
| `p4_universe.py` | P4 revisit v1: point-in-time universes, CRSP v2 daily data and extraction job lists. |

## Multi-day fingerprint, cross-name flow, probes — §1.32, §1.35, `studies/fingerprint_multiday/`

| script | description |
|---|---|
| `burst_feature_probes.py` | burst-feature probes (ideas 7 and 10): what the per-burst sample says about hidden-liquidity share. |
| `fp_crossname.py` | cross-name-v1 (idea 6, listed untested in LLM_README section 10): does burst flow in OTHER names predict a |
| `fp_multiday_episodes.py` | fingerprint-multiday-v1 H4/H5 (design 5c419eebb3ae): impact shape and decay of tape-detected program episodes. |
| `fp_multiday_extract.py` | fingerprint-multiday-v1, stage 1 (cluster): compact per-name-day fingerprint tables from p4-revisit-v1 npz files. |
| `fp_multiday_extract2.py` | fingerprint-multiday-v1, stage 1b (cluster): as stage 1, plus the mean program score per key (idea 4): compact per- |
| `fp_multiday_h1.py` | fingerprint-multiday-v1 H1 (FINGERPRINT_MULTIDAY_DESIGN.md, frozen 49a39405cc61): cross-day same-side excess. |
| `fp_multiday_h2.py` | fingerprint-multiday-v1 H2 (design frozen 49a39405cc61): is backward-linked fingerprint flow institutional? |
| `fp_multiday_h5_tradable.py` | fingerprint-multiday-v1 H5, tradable timing: an episode's end is only known once a day passes with no |
| `fp_multiday_links.py` | fingerprint-multiday-v1, amendment A1 link labels (fixed on DEV, before TEST H2). |
| `fp_multiday_purity.py` | fingerprint-multiday-v1 amendment A1 diagnostic: link purity by repetition count (flow structure only). |
| `probe_score_linkage.py` | Idea 4: does the program score (fit on within-day size repetition, VERIFIED 1.27) predict CROSS-DAY linkage, |
| `probes_retail_vol.py` | Idea 5: are odd-lot bursts retail? Idea 10: does multi-day program intensity forecast volatility? |

## Forced flow — §1.33

| script | description |
|---|---|
| `forced_flow.py` | forced-flow-v1: does burst impact hold less on days when flow is mechanical? |

## Daily retail / institutional labels — §1.34

| script | description |
|---|---|
| `daily_labels.py` | daily-labels-v1 (DAILY_LABELS_DESIGN.md): burst flow vs daily retail and institutional (>= $50k) imbalances. |

## Earnings flow — null

| script | description |
|---|---|
| `earnings_flow.py` | earnings-flow-v1 (EARNINGS_FLOW_DESIGN.md, frozen bb92aa0d85c6): pre-announcement burst flow vs CAR[0,+1]. |

## Legacy pipeline (Feb–May 2026), kept because `studies/p4_revisit/PRIOR_WORK_INVENTORY.md` audits it — no verified result

| script | description |
|---|---|
| `ablation_study.py` | ablation_study.py — Feature Ablation: Direction Dominance Test |
| `aggregate_burst_quality_v2.py` | Daily-unit Newey-West aggregation for corrected packet-level Hurst placebos. |
| `aggregate_results.py` | aggregate_results.py — Post-HPC Cross-Sectional Aggregator |
| `burst_validate.py` | burst_validate.py — do our bursts behave like order splits, and does the result depend on |
| `burst_zoo.py` | burst_zoo.py — exploration harness for alternative burst definitions. |
| `burst_zoo2.py` | burst_zoo2.py — iteration 2: does the block-print signal scale with the spread? |
| `data_quality.py` | data_quality.py — data-integrity infrastructure for the 2017-2021 (and full) sweep. |
| `idea_zoo2.py` | idea_zoo2.py — remaining ideas, plus the test that decides the volatility result. |
| `optuna_regression_sweep.py` | optuna_regression_sweep.py |
| `silence_optimized_sweep.py` | silence_optimized_sweep.py |
| `train_model_zoo.py` | train_model_zoo.py — Comprehensive Model Zoo for Permanence Prediction |
| `compute_permanence.py` | compute_permanence.py — Phase I Permanence Calculation |
| `online_sgd_backtest.py` | — |
| `optuna_physical_sweep.py` | optuna_physical_sweep.py |
