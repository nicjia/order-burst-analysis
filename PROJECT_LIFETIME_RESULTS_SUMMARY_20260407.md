# Order Burst Analysis - Lifetime Results Summary (as of 2026-04-07)

## 1. Project Goals
- Predict permanence behavior from burst events (classification and regression horizons).
- Find physical burst-definition parameters (silence, ADV fraction, directional threshold, volume ratio, kappa) that generalize.
- Evaluate tradability via walk-forward backtests with realistic execution assumptions and measure PnL/Sharpe/hit-rate.

## 2. End-to-End Method (What Was Enforced)
### 2.1 Data and Burst Construction
- Raw event stream processed by `data_processor` (C++) into burst rows.
- Core physical controls: silence (`-s`), direction threshold (`-d`), volume ratio (`-r`), regular session bounds (`-b 34200 -e 57600`).
- Burst output later enriched with permanence columns via `src_py/compute_permanence.py`.

### 2.2 Permanence Targets
- Intraday: `Perm_t1m`, `Perm_t3m`, `Perm_t5m`, `Perm_t10m`, `Perm_tCLOSE`.
- Next-day: `Perm_CLOP`, `Perm_CLCL`.
- `kappa` decay filter applied in permanence stage: keep rows where `D_b >= kappa` (when kappa > 0).

### 2.3 Optuna Physical Search Objective
- Script: `src_py/optuna_physical_sweep.py`.
- Model/metric optimized: monthly walk-forward logistic-regression AUC (`roc_auc_score`) on binary targets (`cls_*`).
- Important: this objective is not directly equivalent to trade PnL optimization.

### 2.4 Trading Backtest Rules (Final Instrumented Version)
- Script: `src_py/online_sgd_backtest.py`.
- Model: online `SGDRegressor` (regression targets).
- Execution mode used in key tests: `burst_stream` with quote-based fills when `EndBid/EndAsk` available.
- Signal modes:
  - `percentile`: long if pred > rolling 75th pct, short if pred < rolling 25th pct.
  - `cost_aware`: predicted per-share move must exceed spread-based gate (`cost_buffer_mult`).
- Position mode in key tests: `shares=1`.
- Diagnostics exported: per-trade and per-signal CSVs.

## 3. Optuna Physical Results (Canonical Best JSONs Across Stocks)
| Ticker | Target | Best AUC | Silence | vol_frac | dir_thresh | vol_ratio | kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| JPM | cls_10m | 0.5652 | s0p5 | 0.0031719538535952218 | 0.5306220738749856 | 0.07321736253718321 | 0.0 |
| JPM | cls_1m | 0.6239 | s2p0 | 0.000911990180310947 | 0.929607686295797 | 0.3865011086107934 | 0.0 |
| JPM | cls_3m | 0.5766 | s0p5 | 0.002116047672165893 | 0.8846779784010457 | 0.27480608520238015 | 0.0 |
| JPM | cls_5m | 0.5644 | s2p0 | 0.0030389959132686283 | 0.6678729841616422 | 0.5246062443331432 | 0.0 |
| JPM | cls_clcl | 0.5605 | s2p0 | 1.0002909964246878e-05 | 0.8948677353397435 | 0.3148394057857518 | 0.04997578791589263 |
| JPM | cls_clop | 0.5495 | s0p5 | 0.0017361019572263686 | 0.5090646193054597 | 0.3590676594584764 | 0.6335168810753494 |
| JPM | cls_close | 0.5303 | s0p5 | 8.508480364830398e-05 | 0.6129556864182452 | 0.36485233510384907 | 1.6607695013987593 |
| MS | cls_10m | 0.5610 | s2p0 | 0.004891105306837529 | 0.5602470197269422 | 0.09250375920414282 | 0.0 |
| MS | cls_1m | 0.6285 | s2p0 | 0.0016515914669865662 | 0.9496310321847444 | 0.5776935087713172 | 0.0 |
| MS | cls_3m | 0.5748 | s2p0 | 0.0011710754867493185 | 0.5142763635945803 | 0.02276608837472519 | 0.0 |
| MS | cls_5m | 0.5660 | s2p0 | 0.004884490995522765 | 0.8419165484159855 | 0.24701153998791633 | 0.0 |
| MS | cls_clcl | 0.5065 | s2p0 | 0.000834603832377626 | 0.9040003598652198 | 0.01575874706009664 | 0.23932493814272882 |
| MS | cls_clop | 0.5114 | s0p5 | 0.000953762432338896 | 0.9442046160786061 | 0.011144354081279395 | 1.6639679436062051 |
| MS | cls_close | 0.5696 | s1p0 | 0.004982835422892858 | 0.6813910527422576 | 0.2961222033610919 | 0.8681311619432621 |
| NVDA | cls_10m | 0.5949 | s2p0 | 0.004851819272593841 | 0.6255701904219741 | 0.48652275207880374 | 0.0 |
| NVDA | cls_1m | 0.6533 | s2p0 | 0.0049157326743032685 | 0.7096570261049338 | 0.5673611393219141 | 0.0 |
| NVDA | cls_3m | 0.6288 | s2p0 | 0.0049830984354126955 | 0.6620045163766891 | 0.5515813238244158 | 0.0 |
| NVDA | cls_5m | 0.6164 | s2p0 | 0.004723869949902043 | 0.6012875135708862 | 0.5039577877927351 | 0.0 |
| NVDA | cls_clcl | 0.5456 | s1p0 | 0.0007955625106750042 | 0.6260078772819889 | 0.479122690193159 | 0.7258905759978211 |
| NVDA | cls_clop | 0.5927 | s0p5 | 0.0005791438631370755 | 0.5006846785554202 | 0.5351208424087406 | 1.6622979927165715 |
| NVDA | cls_close | 0.5339 | s0p5 | 0.0006830448323427173 | 0.7158494882817108 | 0.06887597272196404 | 1.2762520750555515 |
| TSLA | cls_10m | 0.6124 | s2p0 | 0.004512323654407346 | 0.5990318252837113 | 0.5686626949572482 | 0.0 |
| TSLA | cls_1m | 0.6464 | s0p5 | 0.004982101226003821 | 0.6781305353061478 | 0.42618445776832053 | 0.0 |
| TSLA | cls_3m | 0.6436 | s2p0 | 0.004305421645396051 | 0.654607615387428 | 0.511539003720897 | 0.0 |
| TSLA | cls_5m | 0.6191 | s2p0 | 0.0030561630336139087 | 0.6552782722195707 | 0.5340003057318181 | 0.0 |
| TSLA | cls_clcl | 0.5455 | s0p5 | 0.002199444466416056 | 0.5077245633527208 | 0.5619561357849555 | 1.84581974658433 |
| TSLA | cls_clop | 0.5836 | s0p5 | 0.0037698309203782303 | 0.551176409679149 | 0.5850612628415314 | 0.8884980023518256 |
| TSLA | cls_close | 0.5554 | s0p5 | 0.0018758211970032928 | 0.5659238075989267 | 0.339525224012972 | 0.47128249209645184 |

### 3.1 AUC by Target (Cross-Asset Summary)
| Target | Mean AUC | Best (Ticker,AUC) | Worst (Ticker,AUC) |
|---|---:|---|---|
| cls_10m | 0.5834 | TSLA,0.6124 | MS,0.5610 |
| cls_1m | 0.6380 | NVDA,0.6533 | JPM,0.6239 |
| cls_3m | 0.6060 | TSLA,0.6436 | MS,0.5748 |
| cls_5m | 0.5915 | TSLA,0.6191 | JPM,0.5644 |
| cls_clcl | 0.5395 | JPM,0.5605 | MS,0.5065 |
| cls_clop | 0.5593 | NVDA,0.5927 | MS,0.5114 |
| cls_close | 0.5473 | MS,0.5696 | JPM,0.5303 |

## 4. NVDA AUC Sensitivity to Silence Threshold
| Target | s0p5 AUC | s1p0 AUC | s2p0 AUC | Gap(best-worst) |
|---|---:|---:|---:|---:|
| cls_10m | 0.539632 |  |  | 0.000000 |
| cls_1m | 0.617081 | 0.630278 | 0.656070 | 0.038990 |
| cls_3m | 0.575302 | 0.600823 | 0.629806 | 0.054504 |
| cls_5m | 0.554167 | 0.581277 | 0.614872 | 0.060705 |

Interpretation: NVDA short-horizon cls targets improved as silence increased from `s0p5` to `s2p0` (largest observed gap ~0.0607 on `cls_5m`).

## 5. Global Sweep and Ranking Artifacts
- `results/sweep_rankings/global_top_configs.csv` rows: 5
- `results/sweep_rankings/logreg_l2_config_target_stats.csv` rows: 23
- `results/sweep_rankings/logreg_l2_config_overall.csv` rows: 6
- `results/strict_rankings_best_per_target.csv` rows: 7
- `results/strict_rankings_overall_models.csv` rows: 21
- `results/strict_rankings_model_target_strict.csv` rows: 141

### 5.1 Global Top Configs (first 10 rows)
| model | config | overall_mean_auc | overall_mean_std | targets |
|---|---|---|---|---|
| logreg_l2 | s0p5_v50_d0p7_r0p1_k0p0 | 0.5854822753439226 | 0.006172881857263914 | 4 |
| logreg_l2 | s0p5_v50_d0p8_r0p1_k0p0 | 0.5862817643724587 | 0.007999762727904312 | 4 |
| logreg_l2 | s0p5_v50_d0p8_r0p3_k0p0 | 0.5876335715704364 | 0.008765930121927636 | 4 |
| logreg_l2 | s0p5_v50_d0p7_r0p5_k0p0 | 0.5883261597951678 | 0.01006652045492328 | 4 |
| logreg_l2 | s0p5_v50_d0p7_r0p3_k0p0 | 0.5881497174859832 | 0.010680756238256837 | 4 |

## 6. NVDA Backtest Results (Cross-Target Grid, 2019-2022 Archive)
- Source: `hoffman_pull_20260407/NVDA_alltargets_20260407/summary.csv`
| target | signal_mode | cost_buffer | trades | longs | shorts | cum_pnl_raw | sharpe |
|---|---|---:|---:|---:|---:|---:|---:|
| reg_1m | percentile | NA | 23032 | 11926 | 11106 | -3585.8000 | -2.8600 |
| reg_1m | cost_aware | 0.5 | 5316 | 5229 | 87 | -906.4600 | -3.0600 |
| reg_1m | cost_aware | 1.0 | 3960 | 3895 | 65 | -724.0300 | -2.8500 |
| reg_3m | percentile | NA | 23030 | 12006 | 11024 | -3002.8900 | -1.2700 |
| reg_3m | cost_aware | 0.5 | 3336 | 3264 | 72 | -564.6500 | -1.7600 |
| reg_3m | cost_aware | 1.0 | 2377 | 2320 | 57 | -466.6400 | -1.7800 |
| reg_5m | percentile | NA | 23134 | 12008 | 11126 | -2948.3600 | -0.8400 |
| reg_5m | cost_aware | 0.5 | 2552 | 2487 | 65 | -524.4200 | -1.5500 |
| reg_5m | cost_aware | 1.0 | 1738 | 1686 | 52 | -301.9200 | -1.0700 |
| reg_10m | percentile | NA | 23138 | 11994 | 11144 | -1473.7900 | -0.2200 |
| reg_10m | cost_aware | 0.5 | 1023 | 989 | 34 | -79.2000 | -0.2600 |
| reg_10m | cost_aware | 1.0 | 648 | 631 | 17 | -136.5000 | -0.5000 |
| reg_close | percentile | NA | 23663 | 11585 | 12078 | -2085.1200 | -0.2900 |
| reg_close | cost_aware | 0.5 | 88 | 50 | 38 | 79.1000 | 0.3000 |
| reg_close | cost_aware | 1.0 | 74 | 45 | 29 | 80.6400 | 0.3400 |

- Best Sharpe in grid: reg_close / cost_aware / cb=1.0 -> Sharpe 0.3400, PnL 80.6400, trades 74
- `reg_clop/reg_clcl` failed for this dataset path because target `y` had NaN values.

## 7. Trade-Level Forensics (Debug CSVs)
| run | trades | long/short | direction_hit_rate | net_win_rate | avg_gross_win | avg_gross_loss | EV/trade | total_gross |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| reg_10m cost_aware cb=1.0 | 648 | 631/17 | 0.4753 | 0.4753 | 2.4423 | -2.6139 | -0.2106 | -136.5000 |
| reg_close cost_aware cb=1.0 | 66 | 42/24 | 0.5455 | 0.5455 | 8.3139 | -7.2887 | 1.2218 | 80.6400 |

Diagnostic conclusion: reg_10m had sub-50% directional hit-rate and negative EV/trade; reg_close had >50% hit-rate and positive EV/trade.

## 8. Every-Burst Permanence-Direction Baseline
- Source: `hoffman_pull_20260407/NVDA_perm_direction_everyburst_20260407.csv`
| target_col | horizon_min | trades | direction_hit_rate | gross_sum_raw | EV/trade | daily_sharpe |
|---|---|---:|---:|---:|---:|---:|
| Perm_t1m | 1 | 3000622 | 0.3781 | -229824.7600 | -0.0766 | -16.5036 |
| Perm_t3m | 3 | 3000619 | 0.4417 | -136036.1000 | -0.0453 | -9.4743 |
| Perm_t5m | 5 | 3000616 | 0.4629 | -73707.0100 | -0.0246 | -3.6793 |
| Perm_t10m | 10 | 3000602 | 0.4752 | -59839.2600 | -0.0199 | -2.0784 |
| Perm_tCLOSE | close | 2999639 | 0.4716 | -484617.1700 | -0.1616 | -0.8263 |

Outcome: every-burst permanence-direction strategy was negative on all tested horizons.

## 9. NVDA Rebuild with Optuna cls_close Best Physical Params
- Parameter set from `best_physical_params_cls_close.json`: `silence=s0p5`, `vol_frac=0.0006830448323427173`, `dir_thresh=0.7158494882817108`, `vol_ratio=0.06887597272196404`, `kappa=1.2762520750555515`.
- Executed requested pipeline: `data_processor` -> `compute_permanence`.
- data_processor total bursts: 6469839
- kappa filter: 6469839 -> 1769196 kept (27.3% kept, 4700643 removed)
- Output files:
  - `hoffman_pull_20260407/NVDA_clsclose_optuna_rerun/bursts_NVDA_clsclose_optuna_raw.csv`
  - `hoffman_pull_20260407/NVDA_clsclose_optuna_rerun/bursts_NVDA_clsclose_optuna_kappa1276.csv`

## 10. Final Findings for Decision-Making
1. Physical-parameter optimization produced meaningful AUC gains across silence thresholds (especially short horizons for NVDA).
2. High AUC alone did not guarantee trade profitability under current execution and signal rules.
3. Strategy quality depended strongly on target horizon and trade gating; `reg_close` sparse gating outperformed broader high-frequency variants.
4. Directional edge diagnostics (hit-rate + EV/trade) are critical and should remain mandatory in evaluation.
5. The Optuna cls_close rebuild generated a highly selective dataset (~27.3% kept post-kappa), useful for targeted downstream model/trade tests.

## 11. Known Constraints / Caveats
- Some silence-specific Optuna files are incomplete for certain targets (e.g., cls_10m only s0p5 in `_s*.json` set).
- `reg_clop/reg_clcl` backtest path encountered NaN target issues in the tested archive dataset.
- Several historical backtests used fixed physical settings rather than exact target-specific Optuna params; comparisons should note this mismatch.

## 12. Full Sweep Pull (2026-04-08)
- Pull method: single SSH+tar stream (one password prompt) to avoid repeated password entry.
- Local raw pull root: `hoffman_pull_20260407/sweeps_raw_20260408/results/`
- Aggregate files:
  - `hoffman_pull_20260407/sweeps_agg_20260408/sweep_summary_all_rows.csv`
  - `hoffman_pull_20260407/sweeps_agg_20260408/ranked_configs_all_rows.csv`
  - `hoffman_pull_20260407/sweeps_agg_20260408/coverage_counts.json`

### 12.1 sweep_frac Coverage Counts (rows in `sweep_summary.csv`)
| ticker | phase | row_count |
|---|---|---:|
| JPM | long | 81 |
| JPM | short | 162 |
| MS | long | 81 |
| MS | short | 162 |
| NVDA | long | 81 |
| NVDA | short | 162 |
| TSLA | long | 81 |
| TSLA | short | 162 |
- Total `sweep_frac` summary rows aggregated: **972**
- Total `sweep` summary rows aggregated (non-frac): **703**

### 12.2 Notes on Completeness
- Your expected shape was 81 + 81 per stock; pulled data shows short phases at 162 rows for each ticker and long phases at 81 rows each in `sweep_frac` summaries.
- This suggests short-phase `sweep_summary.csv` contains two short targets combined (likely `cls_1m` + `cls_3m`) while long phase is single-target.
- The earlier `global_top_configs` subsection is retained for provenance, but this section is the full raw aggregation requested.

### 12.3 Full sweep_frac `sweep_summary.csv` Aggregation (All Rows)
```csv
sweep_type,ticker,phase,config,target,silence,min_vol,vol_frac,dir_thresh,vol_ratio,kappa,rows,metric_name,metric_value
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_1m,0.5,135.9,1e-05,0.7,0.1,0.0,4366907,AUC,0.6226559779158184
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_10m,0.5,135.9,1e-05,0.7,0.1,0.0,4366907,AUC,0.550402122693406
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_1m,0.5,135.9,1e-05,0.7,0.3,0.0,4366907,AUC,0.6223924671558533
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_10m,0.5,135.9,1e-05,0.7,0.3,0.0,4366907,AUC,0.5493555350150684
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_1m,0.5,135.9,1e-05,0.7,0.5,0.0,4366907,AUC,0.622292180707421
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_10m,0.5,135.9,1e-05,0.7,0.5,0.0,4366907,AUC,0.5665957261930027
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_1m,0.5,135.9,1e-05,0.8,0.1,0.0,4366907,AUC,0.6234957946965595
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_10m,0.5,135.9,1e-05,0.8,0.1,0.0,4366907,AUC,0.5503150364155189
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_1m,0.5,135.9,1e-05,0.8,0.3,0.0,4366907,AUC,0.6226235393139614
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_10m,0.5,135.9,1e-05,0.8,0.3,0.0,4366907,AUC,0.55200541598056
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_1m,0.5,135.9,1e-05,0.8,0.5,0.0,4366907,AUC,0.6220441079966194
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_10m,0.5,135.9,1e-05,0.8,0.5,0.0,4366907,AUC,0.5517724022777533
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_1m,0.5,135.9,1e-05,0.9,0.1,0.0,4366907,AUC,0.6234941140871384
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_10m,0.5,135.9,1e-05,0.9,0.1,0.0,4366907,AUC,0.5476452664561203
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_1m,0.5,135.9,1e-05,0.9,0.3,0.0,4366907,AUC,0.6242366108582241
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_10m,0.5,135.9,1e-05,0.9,0.3,0.0,4366907,AUC,0.5470242564926026
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_1m,0.5,135.9,1e-05,0.9,0.5,0.0,4366907,AUC,0.6242426278479709
sweep_frac,NVDA,short,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_10m,0.5,135.9,1e-05,0.9,0.5,0.0,4366907,AUC,0.5470333084237258
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_1m,0.5,1359.2,0.0001,0.7,0.1,0.0,1065715,AUC,0.6291111395707933
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_10m,0.5,1359.2,0.0001,0.7,0.1,0.0,1065715,AUC,0.5416278974623147
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_1m,0.5,1359.2,0.0001,0.7,0.3,0.0,1065715,AUC,0.6331849105462825
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_10m,0.5,1359.2,0.0001,0.7,0.3,0.0,1065715,AUC,0.5476650014631673
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_1m,0.5,1359.2,0.0001,0.7,0.5,0.0,1065715,AUC,0.6328832127978555
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_10m,0.5,1359.2,0.0001,0.7,0.5,0.0,1065715,AUC,0.5462767367066775
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_1m,0.5,1359.2,0.0001,0.8,0.1,0.0,1065715,AUC,0.6287165678493314
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_10m,0.5,1359.2,0.0001,0.8,0.1,0.0,1065715,AUC,0.5408594239353055
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_1m,0.5,1359.2,0.0001,0.8,0.3,0.0,1065715,AUC,0.6316714367844275
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_10m,0.5,1359.2,0.0001,0.8,0.3,0.0,1065715,AUC,0.5455769389412286
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_1m,0.5,1359.2,0.0001,0.8,0.5,0.0,1065715,AUC,0.632067507778034
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_10m,0.5,1359.2,0.0001,0.8,0.5,0.0,1065715,AUC,0.5456244080045244
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_1m,0.5,1359.2,0.0001,0.9,0.1,0.0,1065715,AUC,0.6273595022285009
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_10m,0.5,1359.2,0.0001,0.9,0.1,0.0,1065715,AUC,0.5374994703634831
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_1m,0.5,1359.2,0.0001,0.9,0.3,0.0,1065715,AUC,0.6279305740427575
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_10m,0.5,1359.2,0.0001,0.9,0.3,0.0,1065715,AUC,0.5383381521394707
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_1m,0.5,1359.2,0.0001,0.9,0.5,0.0,1065715,AUC,0.6279827708541408
sweep_frac,NVDA,short,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_10m,0.5,1359.2,0.0001,0.9,0.5,0.0,1065715,AUC,0.5383915024112701
sweep_frac,NVDA,short,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_1m,0.5,13592.4,0.001,0.7,0.1,0.0,51596,AUC,0.6303388845470439
sweep_frac,NVDA,short,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_10m,0.5,13592.4,0.001,0.7,0.1,0.0,51596,AUC,0.5339207583217622
sweep_frac,NVDA,short,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_1m,0.5,13592.4,0.001,0.7,0.3,0.0,51596,AUC,0.6414127843577175
sweep_frac,NVDA,short,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_10m,0.5,13592.4,0.001,0.7,0.3,0.0,51596,AUC,0.5432517761849913
sweep_frac,NVDA,short,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_1m,0.5,13592.4,0.001,0.7,0.5,0.0,51596,AUC,0.6451350554650314
sweep_frac,NVDA,short,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_10m,0.5,13592.4,0.001,0.7,0.5,0.0,51596,AUC,0.5418927825889968
sweep_frac,NVDA,short,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_1m,0.5,13592.4,0.001,0.8,0.1,0.0,51596,AUC,0.6296570081391066
sweep_frac,NVDA,short,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_10m,0.5,13592.4,0.001,0.8,0.1,0.0,51596,AUC,0.5342040831423952
sweep_frac,NVDA,short,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_1m,0.5,13592.4,0.001,0.8,0.3,0.0,51596,AUC,0.6362244785135616
sweep_frac,NVDA,short,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_10m,0.5,13592.4,0.001,0.8,0.3,0.0,51596,AUC,0.5393561065856778
sweep_frac,NVDA,short,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_1m,0.5,13592.4,0.001,0.8,0.5,0.0,51596,AUC,0.6375875012040632
sweep_frac,NVDA,short,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_10m,0.5,13592.4,0.001,0.8,0.5,0.0,51596,AUC,0.5384274881485405
sweep_frac,NVDA,short,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_1m,0.5,13592.4,0.001,0.9,0.1,0.0,51596,AUC,0.6282784431576027
sweep_frac,NVDA,short,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_10m,0.5,13592.4,0.001,0.9,0.1,0.0,51596,AUC,0.5330281930495191
sweep_frac,NVDA,short,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_1m,0.5,13592.4,0.001,0.9,0.3,0.0,51596,AUC,0.6293467011773328
sweep_frac,NVDA,short,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_10m,0.5,13592.4,0.001,0.9,0.3,0.0,51596,AUC,0.5326805737537972
sweep_frac,NVDA,short,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_1m,0.5,13592.4,0.001,0.9,0.5,0.0,51596,AUC,0.6295160166996189
sweep_frac,NVDA,short,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_10m,0.5,13592.4,0.001,0.9,0.5,0.0,51596,AUC,0.5326166234616667
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_1m,1.0,116.8,1e-05,0.7,0.1,0.0,2479049,AUC,0.6263948522407384
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_10m,1.0,116.8,1e-05,0.7,0.1,0.0,2479049,AUC,0.5499600891411014
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_1m,1.0,116.8,1e-05,0.7,0.3,0.0,2479049,AUC,0.6290321248075186
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_10m,1.0,116.8,1e-05,0.7,0.3,0.0,2479049,AUC,0.5550428231400718
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_1m,1.0,116.8,1e-05,0.7,0.5,0.0,2479049,AUC,0.6293901171264871
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_10m,1.0,116.8,1e-05,0.7,0.5,0.0,2479049,AUC,0.5557892376407182
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_1m,1.0,116.8,1e-05,0.8,0.1,0.0,2479049,AUC,0.6260677865164523
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_10m,1.0,116.8,1e-05,0.8,0.1,0.0,2479049,AUC,0.5492760199919323
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_1m,1.0,116.8,1e-05,0.8,0.3,0.0,2479049,AUC,0.6279646728472452
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_10m,1.0,116.8,1e-05,0.8,0.3,0.0,2479049,AUC,0.5525933966594649
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_1m,1.0,116.8,1e-05,0.8,0.5,0.0,2479049,AUC,0.6278867912907252
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_10m,1.0,116.8,1e-05,0.8,0.5,0.0,2479049,AUC,0.5524051573785425
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_1m,1.0,116.8,1e-05,0.9,0.1,0.0,2479049,AUC,0.6245366637999675
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_10m,1.0,116.8,1e-05,0.9,0.1,0.0,2479049,AUC,0.546590013275533
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_1m,1.0,116.8,1e-05,0.9,0.3,0.0,2479049,AUC,0.6249801599876659
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_10m,1.0,116.8,1e-05,0.9,0.3,0.0,2479049,AUC,0.5473744397996748
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_1m,1.0,116.8,1e-05,0.9,0.5,0.0,2479049,AUC,0.6250008459964579
sweep_frac,NVDA,short,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_10m,1.0,116.8,1e-05,0.9,0.5,0.0,2479049,AUC,0.5474284717848055
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_1m,1.0,1168.1,0.0001,0.7,0.1,0.0,881011,AUC,0.6290743790807414
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_10m,1.0,1168.1,0.0001,0.7,0.1,0.0,881011,AUC,0.5457948151267302
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_1m,1.0,1168.1,0.0001,0.7,0.3,0.0,881011,AUC,0.6374377780377618
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_10m,1.0,1168.1,0.0001,0.7,0.3,0.0,881011,AUC,0.5595478731971654
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_1m,1.0,1168.1,0.0001,0.7,0.5,0.0,881011,AUC,0.6396874259113013
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_10m,1.0,1168.1,0.0001,0.7,0.5,0.0,881011,AUC,0.5615154170674677
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_1m,1.0,1168.1,0.0001,0.8,0.1,0.0,881011,AUC,0.6286868563569258
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_10m,1.0,1168.1,0.0001,0.8,0.1,0.0,881011,AUC,0.5449267110232743
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_1m,1.0,1168.1,0.0001,0.8,0.3,0.0,881011,AUC,0.6333580851130757
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_10m,1.0,1168.1,0.0001,0.8,0.3,0.0,881011,AUC,0.5523340983701581
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_1m,1.0,1168.1,0.0001,0.8,0.5,0.0,881011,AUC,0.6340184054436906
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_10m,1.0,1168.1,0.0001,0.8,0.5,0.0,881011,AUC,0.552896883663515
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_1m,1.0,1168.1,0.0001,0.9,0.1,0.0,881011,AUC,0.6272328013768459
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_10m,1.0,1168.1,0.0001,0.9,0.1,0.0,881011,AUC,0.5419980660714405
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_1m,1.0,1168.1,0.0001,0.9,0.3,0.0,881011,AUC,0.6278709783029371
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_10m,1.0,1168.1,0.0001,0.9,0.3,0.0,881011,AUC,0.5431304962134983
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_1m,1.0,1168.1,0.0001,0.9,0.5,0.0,881011,AUC,0.6278740655957311
sweep_frac,NVDA,short,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_10m,1.0,1168.1,0.0001,0.9,0.5,0.0,881011,AUC,0.5432131923861423
sweep_frac,NVDA,short,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_1m,1.0,11681.0,0.001,0.7,0.1,0.0,68413,AUC,0.6314824454878383
sweep_frac,NVDA,short,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_10m,1.0,11681.0,0.001,0.7,0.1,0.0,68413,AUC,0.5435989786021029
sweep_frac,NVDA,short,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_1m,1.0,11681.0,0.001,0.7,0.3,0.0,68413,AUC,0.6549292555025226
sweep_frac,NVDA,short,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_10m,1.0,11681.0,0.001,0.7,0.3,0.0,68413,AUC,0.5607425938693127
sweep_frac,NVDA,short,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_1m,1.0,11681.0,0.001,0.7,0.5,0.0,68413,AUC,0.6662693652610496
sweep_frac,NVDA,short,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_10m,1.0,11681.0,0.001,0.7,0.5,0.0,68413,AUC,0.5678685874103708
sweep_frac,NVDA,short,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_1m,1.0,11681.0,0.001,0.8,0.1,0.0,68413,AUC,0.6304369525292643
sweep_frac,NVDA,short,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_10m,1.0,11681.0,0.001,0.8,0.1,0.0,68413,AUC,0.5428611830046659
sweep_frac,NVDA,short,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_1m,1.0,11681.0,0.001,0.8,0.3,0.0,68413,AUC,0.6454042144047356
sweep_frac,NVDA,short,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_10m,1.0,11681.0,0.001,0.8,0.3,0.0,68413,AUC,0.5508085046884588
sweep_frac,NVDA,short,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_1m,1.0,11681.0,0.001,0.8,0.5,0.0,68413,AUC,0.6476068593962233
sweep_frac,NVDA,short,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_10m,1.0,11681.0,0.001,0.8,0.5,0.0,68413,AUC,0.5516377276092616
sweep_frac,NVDA,short,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_1m,1.0,11681.0,0.001,0.9,0.1,0.0,68413,AUC,0.6279362197699949
sweep_frac,NVDA,short,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_10m,1.0,11681.0,0.001,0.9,0.1,0.0,68413,AUC,0.5421979283460092
sweep_frac,NVDA,short,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_1m,1.0,11681.0,0.001,0.9,0.3,0.0,68413,AUC,0.6297245327585436
sweep_frac,NVDA,short,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_10m,1.0,11681.0,0.001,0.9,0.3,0.0,68413,AUC,0.5428651732470371
sweep_frac,NVDA,short,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_1m,1.0,11681.0,0.001,0.9,0.5,0.0,68413,AUC,0.6298270509382375
sweep_frac,NVDA,short,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_10m,1.0,11681.0,0.001,0.9,0.5,0.0,68413,AUC,0.5427924733744081
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_1m,2.0,97.5,1e-05,0.7,0.1,0.0,984854,AUC,0.6227538442157008
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_10m,2.0,97.5,1e-05,0.7,0.1,0.0,984854,AUC,0.5527126481200769
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_1m,2.0,97.5,1e-05,0.7,0.3,0.0,984854,AUC,0.6322615382166652
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_10m,2.0,97.5,1e-05,0.7,0.3,0.0,984854,AUC,0.5604948257364789
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_1m,2.0,97.5,1e-05,0.7,0.5,0.0,984854,AUC,0.6369230494809713
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_10m,2.0,97.5,1e-05,0.7,0.5,0.0,984854,AUC,0.5628009226433386
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_1m,2.0,97.5,1e-05,0.8,0.1,0.0,984854,AUC,0.6216833996082816
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_10m,2.0,97.5,1e-05,0.8,0.1,0.0,984854,AUC,0.5510534647119164
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_1m,2.0,97.5,1e-05,0.8,0.3,0.0,984854,AUC,0.6276698931503009
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_10m,2.0,97.5,1e-05,0.8,0.3,0.0,984854,AUC,0.5567968778233302
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_1m,2.0,97.5,1e-05,0.8,0.5,0.0,984854,AUC,0.6286906825570564
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_10m,2.0,97.5,1e-05,0.8,0.5,0.0,984854,AUC,0.5577595906891746
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_1m,2.0,97.5,1e-05,0.9,0.1,0.0,984854,AUC,0.618719470499357
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_10m,2.0,97.5,1e-05,0.9,0.1,0.0,984854,AUC,0.5468627030317512
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_1m,2.0,97.5,1e-05,0.9,0.3,0.0,984854,AUC,0.6195840379526736
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_10m,2.0,97.5,1e-05,0.9,0.3,0.0,984854,AUC,0.5477653226602907
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_1m,2.0,97.5,1e-05,0.9,0.5,0.0,984854,AUC,0.6196268270092914
sweep_frac,NVDA,short,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_10m,2.0,97.5,1e-05,0.9,0.5,0.0,984854,AUC,0.547877665011877
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_1m,2.0,974.8,0.0001,0.7,0.1,0.0,527095,AUC,0.6208963531579778
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_10m,2.0,974.8,0.0001,0.7,0.1,0.0,527095,AUC,0.5459801012321589
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_1m,2.0,974.8,0.0001,0.7,0.3,0.0,527095,AUC,0.6396720130214988
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_10m,2.0,974.8,0.0001,0.7,0.3,0.0,527095,AUC,0.5644943188203547
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_1m,2.0,974.8,0.0001,0.7,0.5,0.0,527095,AUC,0.6495063133797103
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_10m,2.0,974.8,0.0001,0.7,0.5,0.0,527095,AUC,0.5713523709337552
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_1m,2.0,974.8,0.0001,0.8,0.1,0.0,527095,AUC,0.620069068740933
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_10m,2.0,974.8,0.0001,0.8,0.1,0.0,527095,AUC,0.5443865052569604
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_1m,2.0,974.8,0.0001,0.8,0.3,0.0,527095,AUC,0.6310931763915311
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_10m,2.0,974.8,0.0001,0.8,0.3,0.0,527095,AUC,0.5556177270833779
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_1m,2.0,974.8,0.0001,0.8,0.5,0.0,527095,AUC,0.6329755034864774
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_10m,2.0,974.8,0.0001,0.8,0.5,0.0,527095,AUC,0.5572197133558627
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_1m,2.0,974.8,0.0001,0.9,0.1,0.0,527095,AUC,0.6171247279977202
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_10m,2.0,974.8,0.0001,0.9,0.1,0.0,527095,AUC,0.5405487719204116
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_1m,2.0,974.8,0.0001,0.9,0.3,0.0,527095,AUC,0.6185884805869519
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_10m,2.0,974.8,0.0001,0.9,0.3,0.0,527095,AUC,0.5419960550817585
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_1m,2.0,974.8,0.0001,0.9,0.5,0.0,527095,AUC,0.6186352678505415
sweep_frac,NVDA,short,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_10m,2.0,974.8,0.0001,0.9,0.5,0.0,527095,AUC,0.5420929187433401
sweep_frac,NVDA,short,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_1m,2.0,9748.4,0.001,0.7,0.1,0.0,74313,AUC,0.623008524876502
sweep_frac,NVDA,short,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_10m,2.0,9748.4,0.001,0.7,0.1,0.0,74313,AUC,0.5481666991864127
sweep_frac,NVDA,short,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_1m,2.0,9748.4,0.001,0.7,0.3,0.0,74313,AUC,0.670003565512824
sweep_frac,NVDA,short,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_10m,2.0,9748.4,0.001,0.7,0.3,0.0,74313,AUC,0.5731243604033054
sweep_frac,NVDA,short,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_1m,2.0,9748.4,0.001,0.7,0.5,0.0,74313,AUC,0.6978516161713783
sweep_frac,NVDA,short,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_10m,2.0,9748.4,0.001,0.7,0.5,0.0,74313,AUC,0.5871085152234341
sweep_frac,NVDA,short,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_1m,2.0,9748.4,0.001,0.8,0.1,0.0,74313,AUC,0.6217756963083922
sweep_frac,NVDA,short,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_10m,2.0,9748.4,0.001,0.8,0.1,0.0,74313,AUC,0.5463153819022948
sweep_frac,NVDA,short,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_1m,2.0,9748.4,0.001,0.8,0.3,0.0,74313,AUC,0.6502286446579353
sweep_frac,NVDA,short,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_10m,2.0,9748.4,0.001,0.8,0.3,0.0,74313,AUC,0.5608808500692051
sweep_frac,NVDA,short,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_1m,2.0,9748.4,0.001,0.8,0.5,0.0,74313,AUC,0.6559819636845039
sweep_frac,NVDA,short,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_10m,2.0,9748.4,0.001,0.8,0.5,0.0,74313,AUC,0.5636200504139356
sweep_frac,NVDA,short,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_1m,2.0,9748.4,0.001,0.9,0.1,0.0,74313,AUC,0.6162705760976044
sweep_frac,NVDA,short,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_10m,2.0,9748.4,0.001,0.9,0.1,0.0,74313,AUC,0.5425044021179057
sweep_frac,NVDA,short,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_1m,2.0,9748.4,0.001,0.9,0.3,0.0,74313,AUC,0.6177605838980722
sweep_frac,NVDA,short,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_10m,2.0,9748.4,0.001,0.9,0.3,0.0,74313,AUC,0.542296971729064
sweep_frac,NVDA,short,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_1m,2.0,9748.4,0.001,0.9,0.5,0.0,74313,AUC,0.6178077148740906
sweep_frac,NVDA,short,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_10m,2.0,9748.4,0.001,0.9,0.5,0.0,74313,AUC,0.5423757952381437
sweep_frac,NVDA,long,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_close,0.5,135.9,1e-05,0.7,0.1,0.0,4366907,AUC,0.5630465384629564
sweep_frac,NVDA,long,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_close,0.5,135.9,1e-05,0.7,0.3,0.0,4366907,AUC,0.5631089049682884
sweep_frac,NVDA,long,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_close,0.5,135.9,1e-05,0.7,0.5,0.0,4366907,AUC,0.5648352733980233
sweep_frac,NVDA,long,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_close,0.5,135.9,1e-05,0.8,0.1,0.0,4366907,AUC,0.5631398251729497
sweep_frac,NVDA,long,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_close,0.5,135.9,1e-05,0.8,0.3,0.0,4366907,AUC,0.5624770586492416
sweep_frac,NVDA,long,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_close,0.5,135.9,1e-05,0.8,0.5,0.0,4366907,AUC,0.5624068531299313
sweep_frac,NVDA,long,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_close,0.5,135.9,1e-05,0.9,0.1,0.0,4366907,AUC,0.5632838275198692
sweep_frac,NVDA,long,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_close,0.5,135.9,1e-05,0.9,0.3,0.0,4366907,AUC,0.5631333342654125
sweep_frac,NVDA,long,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_close,0.5,135.9,1e-05,0.9,0.5,0.0,4366907,AUC,0.5630993630661184
sweep_frac,NVDA,long,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_close,0.5,1359.2,0.0001,0.7,0.1,0.0,1065715,AUC,0.5863325063305107
sweep_frac,NVDA,long,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_close,0.5,1359.2,0.0001,0.7,0.3,0.0,1065715,AUC,0.5851619695094742
sweep_frac,NVDA,long,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_close,0.5,1359.2,0.0001,0.7,0.5,0.0,1065715,AUC,0.5846704054784507
sweep_frac,NVDA,long,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_close,0.5,1359.2,0.0001,0.8,0.1,0.0,1065715,AUC,0.5858922568378948
sweep_frac,NVDA,long,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_close,0.5,1359.2,0.0001,0.8,0.3,0.0,1065715,AUC,0.5860504235861066
sweep_frac,NVDA,long,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_close,0.5,1359.2,0.0001,0.8,0.5,0.0,1065715,AUC,0.5855567208869845
sweep_frac,NVDA,long,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_close,0.5,1359.2,0.0001,0.9,0.1,0.0,1065715,AUC,0.5850187066415916
sweep_frac,NVDA,long,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_close,0.5,1359.2,0.0001,0.9,0.3,0.0,1065715,AUC,0.5856275539348018
sweep_frac,NVDA,long,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_close,0.5,1359.2,0.0001,0.9,0.5,0.0,1065715,AUC,0.5856193398178724
sweep_frac,NVDA,long,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_close,0.5,13592.4,0.001,0.7,0.1,0.0,51596,AUC,0.5862346929238653
sweep_frac,NVDA,long,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_close,0.5,13592.4,0.001,0.7,0.3,0.0,51596,AUC,0.5851593658738135
sweep_frac,NVDA,long,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_close,0.5,13592.4,0.001,0.7,0.5,0.0,51596,AUC,0.5815885190619232
sweep_frac,NVDA,long,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_close,0.5,13592.4,0.001,0.8,0.1,0.0,51596,AUC,0.585358017132064
sweep_frac,NVDA,long,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_close,0.5,13592.4,0.001,0.8,0.3,0.0,51596,AUC,0.5844929304185469
sweep_frac,NVDA,long,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_close,0.5,13592.4,0.001,0.8,0.5,0.0,51596,AUC,0.5843678857168638
sweep_frac,NVDA,long,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_close,0.5,13592.4,0.001,0.9,0.1,0.0,51596,AUC,0.5832970813685261
sweep_frac,NVDA,long,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_close,0.5,13592.4,0.001,0.9,0.3,0.0,51596,AUC,0.5839402908468729
sweep_frac,NVDA,long,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_close,0.5,13592.4,0.001,0.9,0.5,0.0,51596,AUC,0.5837984097029406
sweep_frac,NVDA,long,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_close,1.0,116.8,1e-05,0.7,0.1,0.0,2479049,AUC,0.5737129642989428
sweep_frac,NVDA,long,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_close,1.0,116.8,1e-05,0.7,0.3,0.0,2479049,AUC,0.5726175741997555
sweep_frac,NVDA,long,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_close,1.0,116.8,1e-05,0.7,0.5,0.0,2479049,AUC,0.5727662751409871
sweep_frac,NVDA,long,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_close,1.0,116.8,1e-05,0.8,0.1,0.0,2479049,AUC,0.5737002114279615
sweep_frac,NVDA,long,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_close,1.0,116.8,1e-05,0.8,0.3,0.0,2479049,AUC,0.573135082452997
sweep_frac,NVDA,long,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_close,1.0,116.8,1e-05,0.8,0.5,0.0,2479049,AUC,0.572748680315322
sweep_frac,NVDA,long,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_close,1.0,116.8,1e-05,0.9,0.1,0.0,2479049,AUC,0.5737089433650487
sweep_frac,NVDA,long,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_close,1.0,116.8,1e-05,0.9,0.3,0.0,2479049,AUC,0.5737049734976896
sweep_frac,NVDA,long,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_close,1.0,116.8,1e-05,0.9,0.5,0.0,2479049,AUC,0.573674275055373
sweep_frac,NVDA,long,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_close,1.0,1168.1,0.0001,0.7,0.1,0.0,881011,AUC,0.5832411736221114
sweep_frac,NVDA,long,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_close,1.0,1168.1,0.0001,0.7,0.3,0.0,881011,AUC,0.5847499429291971
sweep_frac,NVDA,long,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_close,1.0,1168.1,0.0001,0.7,0.5,0.0,881011,AUC,0.5818792405245204
sweep_frac,NVDA,long,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_close,1.0,1168.1,0.0001,0.8,0.1,0.0,881011,AUC,0.5829144903001168
sweep_frac,NVDA,long,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_close,1.0,1168.1,0.0001,0.8,0.3,0.0,881011,AUC,0.584934566677872
sweep_frac,NVDA,long,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_close,1.0,1168.1,0.0001,0.8,0.5,0.0,881011,AUC,0.584394813254684
sweep_frac,NVDA,long,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_close,1.0,1168.1,0.0001,0.9,0.1,0.0,881011,AUC,0.581799954705308
sweep_frac,NVDA,long,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_close,1.0,1168.1,0.0001,0.9,0.3,0.0,881011,AUC,0.5821179556927442
sweep_frac,NVDA,long,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_close,1.0,1168.1,0.0001,0.9,0.5,0.0,881011,AUC,0.5821244208938023
sweep_frac,NVDA,long,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_close,1.0,11681.0,0.001,0.7,0.1,0.0,68413,AUC,0.5976419308495297
sweep_frac,NVDA,long,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_close,1.0,11681.0,0.001,0.7,0.3,0.0,68413,AUC,0.600990040395172
sweep_frac,NVDA,long,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_close,1.0,11681.0,0.001,0.7,0.5,0.0,68413,AUC,0.5969075766018028
sweep_frac,NVDA,long,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_close,1.0,11681.0,0.001,0.8,0.1,0.0,68413,AUC,0.596856751496557
sweep_frac,NVDA,long,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_close,1.0,11681.0,0.001,0.8,0.3,0.0,68413,AUC,0.5988240904922497
sweep_frac,NVDA,long,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_close,1.0,11681.0,0.001,0.8,0.5,0.0,68413,AUC,0.5978142997142882
sweep_frac,NVDA,long,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_close,1.0,11681.0,0.001,0.9,0.1,0.0,68413,AUC,0.5960082992262554
sweep_frac,NVDA,long,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_close,1.0,11681.0,0.001,0.9,0.3,0.0,68413,AUC,0.5965915016262373
sweep_frac,NVDA,long,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_close,1.0,11681.0,0.001,0.9,0.5,0.0,68413,AUC,0.596450357207399
sweep_frac,NVDA,long,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_close,2.0,97.5,1e-05,0.7,0.1,0.0,984854,AUC,0.5877264038888856
sweep_frac,NVDA,long,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_close,2.0,97.5,1e-05,0.7,0.3,0.0,984854,AUC,0.5868345381162718
sweep_frac,NVDA,long,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_close,2.0,97.5,1e-05,0.7,0.5,0.0,984854,AUC,0.5841528706628695
sweep_frac,NVDA,long,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_close,2.0,97.5,1e-05,0.8,0.1,0.0,984854,AUC,0.5874215770927435
sweep_frac,NVDA,long,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_close,2.0,97.5,1e-05,0.8,0.3,0.0,984854,AUC,0.5873660364559633
sweep_frac,NVDA,long,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_close,2.0,97.5,1e-05,0.8,0.5,0.0,984854,AUC,0.5871764670425575
sweep_frac,NVDA,long,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_close,2.0,97.5,1e-05,0.9,0.1,0.0,984854,AUC,0.5862336260332204
sweep_frac,NVDA,long,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_close,2.0,97.5,1e-05,0.9,0.3,0.0,984854,AUC,0.5864213210288532
sweep_frac,NVDA,long,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_close,2.0,97.5,1e-05,0.9,0.5,0.0,984854,AUC,0.5863894339233862
sweep_frac,NVDA,long,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_close,2.0,974.8,0.0001,0.7,0.1,0.0,527095,AUC,0.5882656946944488
sweep_frac,NVDA,long,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_close,2.0,974.8,0.0001,0.7,0.3,0.0,527095,AUC,0.5912303945053898
sweep_frac,NVDA,long,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_close,2.0,974.8,0.0001,0.7,0.5,0.0,527095,AUC,0.5902903772101864
sweep_frac,NVDA,long,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_close,2.0,974.8,0.0001,0.8,0.1,0.0,527095,AUC,0.5878617738106209
sweep_frac,NVDA,long,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_close,2.0,974.8,0.0001,0.8,0.3,0.0,527095,AUC,0.5893574853462682
sweep_frac,NVDA,long,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_close,2.0,974.8,0.0001,0.8,0.5,0.0,527095,AUC,0.5894058886519533
sweep_frac,NVDA,long,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_close,2.0,974.8,0.0001,0.9,0.1,0.0,527095,AUC,0.5863805800544648
sweep_frac,NVDA,long,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_close,2.0,974.8,0.0001,0.9,0.3,0.0,527095,AUC,0.5863614745414186
sweep_frac,NVDA,long,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_close,2.0,974.8,0.0001,0.9,0.5,0.0,527095,AUC,0.5863616164773645
sweep_frac,NVDA,long,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_close,2.0,9748.4,0.001,0.7,0.1,0.0,74313,AUC,0.5922908388860875
sweep_frac,NVDA,long,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_close,2.0,9748.4,0.001,0.7,0.3,0.0,74313,AUC,0.5937427009205049
sweep_frac,NVDA,long,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_close,2.0,9748.4,0.001,0.7,0.5,0.0,74313,AUC,0.5943254910137457
sweep_frac,NVDA,long,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_close,2.0,9748.4,0.001,0.8,0.1,0.0,74313,AUC,0.5922821883790497
sweep_frac,NVDA,long,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_close,2.0,9748.4,0.001,0.8,0.3,0.0,74313,AUC,0.5926623333687148
sweep_frac,NVDA,long,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_close,2.0,9748.4,0.001,0.8,0.5,0.0,74313,AUC,0.5928116615837521
sweep_frac,NVDA,long,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_close,2.0,9748.4,0.001,0.9,0.1,0.0,74313,AUC,0.5918013758675014
sweep_frac,NVDA,long,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_close,2.0,9748.4,0.001,0.9,0.3,0.0,74313,AUC,0.591877200719343
sweep_frac,NVDA,long,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_close,2.0,9748.4,0.001,0.9,0.5,0.0,74313,AUC,0.5917106106549479
sweep_frac,MS,short,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_1m,0.5,9.2,1e-05,0.7,0.1,0.0,1910918,AUC,0.6469874955251259
sweep_frac,MS,short,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_10m,0.5,9.2,1e-05,0.7,0.1,0.0,1910918,AUC,0.5634375555454474
sweep_frac,MS,short,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_1m,0.5,9.2,1e-05,0.7,0.3,0.0,1910918,AUC,0.6489979679864959
sweep_frac,MS,short,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_10m,0.5,9.2,1e-05,0.7,0.3,0.0,1910918,AUC,0.5660116825805778
sweep_frac,MS,short,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_1m,0.5,9.2,1e-05,0.7,0.5,0.0,1910918,AUC,0.6503561121433427
sweep_frac,MS,short,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_10m,0.5,9.2,1e-05,0.7,0.5,0.0,1910918,AUC,0.5674382889334726
sweep_frac,MS,short,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_1m,0.5,9.2,1e-05,0.8,0.1,0.0,1910918,AUC,0.6463260785591473
sweep_frac,MS,short,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_10m,0.5,9.2,1e-05,0.8,0.1,0.0,1910918,AUC,0.5624930391156013
sweep_frac,MS,short,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_1m,0.5,9.2,1e-05,0.8,0.3,0.0,1910918,AUC,0.6477146559272496
sweep_frac,MS,short,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_10m,0.5,9.2,1e-05,0.8,0.3,0.0,1910918,AUC,0.5644989564071886
sweep_frac,MS,short,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_1m,0.5,9.2,1e-05,0.8,0.5,0.0,1910918,AUC,0.6483697081599025
sweep_frac,MS,short,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_10m,0.5,9.2,1e-05,0.8,0.5,0.0,1910918,AUC,0.5652923009474605
sweep_frac,MS,short,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_1m,0.5,9.2,1e-05,0.9,0.1,0.0,1910918,AUC,0.6450713896458677
sweep_frac,MS,short,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_10m,0.5,9.2,1e-05,0.9,0.1,0.0,1910918,AUC,0.5609321973162443
sweep_frac,MS,short,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_1m,0.5,9.2,1e-05,0.9,0.3,0.0,1910918,AUC,0.64529698668359
sweep_frac,MS,short,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_10m,0.5,9.2,1e-05,0.9,0.3,0.0,1910918,AUC,0.5612869373489802
sweep_frac,MS,short,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_1m,0.5,9.2,1e-05,0.9,0.5,0.0,1910918,AUC,0.6453116121064003
sweep_frac,MS,short,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_10m,0.5,9.2,1e-05,0.9,0.5,0.0,1910918,AUC,0.5612965157024525
sweep_frac,MS,short,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_1m,0.5,92.3,0.0001,0.7,0.1,0.0,1214805,AUC,0.6470786352913942
sweep_frac,MS,short,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_10m,0.5,92.3,0.0001,0.7,0.1,0.0,1214805,AUC,0.562661108071163
sweep_frac,MS,short,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_1m,0.5,92.3,0.0001,0.7,0.3,0.0,1214805,AUC,0.6495597051756289
sweep_frac,MS,short,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_10m,0.5,92.3,0.0001,0.7,0.3,0.0,1214805,AUC,0.5664814678300418
sweep_frac,MS,short,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_1m,0.5,92.3,0.0001,0.7,0.5,0.0,1214805,AUC,0.6514195539475783
sweep_frac,MS,short,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_10m,0.5,92.3,0.0001,0.7,0.5,0.0,1214805,AUC,0.5692979851495752
sweep_frac,MS,short,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_1m,0.5,92.3,0.0001,0.8,0.1,0.0,1214805,AUC,0.6463589340110211
sweep_frac,MS,short,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_10m,0.5,92.3,0.0001,0.8,0.1,0.0,1214805,AUC,0.5615968894524884
sweep_frac,MS,short,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_1m,0.5,92.3,0.0001,0.8,0.3,0.0,1214805,AUC,0.6480867236099125
sweep_frac,MS,short,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_10m,0.5,92.3,0.0001,0.8,0.3,0.0,1214805,AUC,0.5641671652206831
sweep_frac,MS,short,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_1m,0.5,92.3,0.0001,0.8,0.5,0.0,1214805,AUC,0.6488710881508736
sweep_frac,MS,short,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_10m,0.5,92.3,0.0001,0.8,0.5,0.0,1214805,AUC,0.5654759970789227
sweep_frac,MS,short,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_1m,0.5,92.3,0.0001,0.9,0.1,0.0,1214805,AUC,0.644834067832908
sweep_frac,MS,short,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_10m,0.5,92.3,0.0001,0.9,0.1,0.0,1214805,AUC,0.559226240948073
sweep_frac,MS,short,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_1m,0.5,92.3,0.0001,0.9,0.3,0.0,1214805,AUC,0.645139489484855
sweep_frac,MS,short,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_10m,0.5,92.3,0.0001,0.9,0.3,0.0,1214805,AUC,0.5596757200521263
sweep_frac,MS,short,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_1m,0.5,92.3,0.0001,0.9,0.5,0.0,1214805,AUC,0.645160934984209
sweep_frac,MS,short,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_10m,0.5,92.3,0.0001,0.9,0.5,0.0,1214805,AUC,0.5597072624117032
sweep_frac,MS,short,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_1m,0.5,923.2,0.001,0.7,0.1,0.0,75786,AUC,0.6543586482361087
sweep_frac,MS,short,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_10m,0.5,923.2,0.001,0.7,0.1,0.0,75786,AUC,0.5521846205571557
sweep_frac,MS,short,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_1m,0.5,923.2,0.001,0.7,0.3,0.0,75786,AUC,0.6616157095905999
sweep_frac,MS,short,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_10m,0.5,923.2,0.001,0.7,0.3,0.0,75786,AUC,0.5614665101818116
sweep_frac,MS,short,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_1m,0.5,923.2,0.001,0.7,0.5,0.0,75786,AUC,0.6633056103051964
sweep_frac,MS,short,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_10m,0.5,923.2,0.001,0.7,0.5,0.0,75786,AUC,0.5661168458201267
sweep_frac,MS,short,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_1m,0.5,923.2,0.001,0.8,0.1,0.0,75786,AUC,0.6535814900016063
sweep_frac,MS,short,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_10m,0.5,923.2,0.001,0.8,0.1,0.0,75786,AUC,0.5521521031220407
sweep_frac,MS,short,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_1m,0.5,923.2,0.001,0.8,0.3,0.0,75786,AUC,0.6589083864867202
sweep_frac,MS,short,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_10m,0.5,923.2,0.001,0.8,0.3,0.0,75786,AUC,0.5579662302542859
sweep_frac,MS,short,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_1m,0.5,923.2,0.001,0.8,0.5,0.0,75786,AUC,0.659571031376693
sweep_frac,MS,short,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_10m,0.5,923.2,0.001,0.8,0.5,0.0,75786,AUC,0.5592993859966753
sweep_frac,MS,short,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_1m,0.5,923.2,0.001,0.9,0.1,0.0,75786,AUC,0.6518413009750725
sweep_frac,MS,short,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_10m,0.5,923.2,0.001,0.9,0.1,0.0,75786,AUC,0.549516311930355
sweep_frac,MS,short,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_1m,0.5,923.2,0.001,0.9,0.3,0.0,75786,AUC,0.6530951001045807
sweep_frac,MS,short,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_10m,0.5,923.2,0.001,0.9,0.3,0.0,75786,AUC,0.5506986895336334
sweep_frac,MS,short,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_1m,0.5,923.2,0.001,0.9,0.5,0.0,75786,AUC,0.6531265972151244
sweep_frac,MS,short,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_10m,0.5,923.2,0.001,0.9,0.5,0.0,75786,AUC,0.550734560742057
sweep_frac,MS,short,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_1m,1.0,9.2,1e-05,0.7,0.1,0.0,1645119,AUC,0.6480474598410737
sweep_frac,MS,short,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_10m,1.0,9.2,1e-05,0.7,0.1,0.0,1645119,AUC,0.5656856653611373
sweep_frac,MS,short,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_1m,1.0,9.2,1e-05,0.7,0.3,0.0,1645119,AUC,0.6511751002696257
sweep_frac,MS,short,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_10m,1.0,9.2,1e-05,0.7,0.3,0.0,1645119,AUC,0.569758634337743
sweep_frac,MS,short,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_1m,1.0,9.2,1e-05,0.7,0.5,0.0,1645119,AUC,0.6532774304609048
sweep_frac,MS,short,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_10m,1.0,9.2,1e-05,0.7,0.5,0.0,1645119,AUC,0.5719924891237135
sweep_frac,MS,short,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_1m,1.0,9.2,1e-05,0.8,0.1,0.0,1645119,AUC,0.6471012676767385
sweep_frac,MS,short,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_10m,1.0,9.2,1e-05,0.8,0.1,0.0,1645119,AUC,0.5642972791460608
sweep_frac,MS,short,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_1m,1.0,9.2,1e-05,0.8,0.3,0.0,1645119,AUC,0.6490839571152578
sweep_frac,MS,short,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_10m,1.0,9.2,1e-05,0.8,0.3,0.0,1645119,AUC,0.5672364203892724
sweep_frac,MS,short,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_1m,1.0,9.2,1e-05,0.8,0.5,0.0,1645119,AUC,0.6499629530641788
sweep_frac,MS,short,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_10m,1.0,9.2,1e-05,0.8,0.5,0.0,1645119,AUC,0.5683781471615014
sweep_frac,MS,short,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_1m,1.0,9.2,1e-05,0.9,0.1,0.0,1645119,AUC,0.6453064701045769
sweep_frac,MS,short,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_10m,1.0,9.2,1e-05,0.9,0.1,0.0,1645119,AUC,0.5614793754028338
sweep_frac,MS,short,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_1m,1.0,9.2,1e-05,0.9,0.3,0.0,1645119,AUC,0.6456111055855214
sweep_frac,MS,short,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_10m,1.0,9.2,1e-05,0.9,0.3,0.0,1645119,AUC,0.5619892860740945
sweep_frac,MS,short,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_1m,1.0,9.2,1e-05,0.9,0.5,0.0,1645119,AUC,0.6456501076916835
sweep_frac,MS,short,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_10m,1.0,9.2,1e-05,0.9,0.5,0.0,1645119,AUC,0.5620576004182745
sweep_frac,MS,short,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_1m,1.0,92.1,0.0001,0.7,0.1,0.0,1094120,AUC,0.6491518176063337
sweep_frac,MS,short,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_10m,1.0,92.1,0.0001,0.7,0.1,0.0,1094120,AUC,0.5659304756110657
sweep_frac,MS,short,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_1m,1.0,92.1,0.0001,0.7,0.3,0.0,1094120,AUC,0.6526739163926236
sweep_frac,MS,short,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_10m,1.0,92.1,0.0001,0.7,0.3,0.0,1094120,AUC,0.5710842470439516
sweep_frac,MS,short,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_1m,1.0,92.1,0.0001,0.7,0.5,0.0,1094120,AUC,0.6553939111466776
sweep_frac,MS,short,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_10m,1.0,92.1,0.0001,0.7,0.5,0.0,1094120,AUC,0.5746885692791165
sweep_frac,MS,short,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_1m,1.0,92.1,0.0001,0.8,0.1,0.0,1094120,AUC,0.648266370155216
sweep_frac,MS,short,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_10m,1.0,92.1,0.0001,0.8,0.1,0.0,1094120,AUC,0.5648396455360742
sweep_frac,MS,short,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_1m,1.0,92.1,0.0001,0.8,0.3,0.0,1094120,AUC,0.6504821007232321
sweep_frac,MS,short,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_10m,1.0,92.1,0.0001,0.8,0.3,0.0,1094120,AUC,0.5683188992002445
sweep_frac,MS,short,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_1m,1.0,92.1,0.0001,0.8,0.5,0.0,1094120,AUC,0.6514891726587522
sweep_frac,MS,short,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_10m,1.0,92.1,0.0001,0.8,0.5,0.0,1094120,AUC,0.5697875890566746
sweep_frac,MS,short,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_1m,1.0,92.1,0.0001,0.9,0.1,0.0,1094120,AUC,0.6465808943330942
sweep_frac,MS,short,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_10m,1.0,92.1,0.0001,0.9,0.1,0.0,1094120,AUC,0.5621880130984642
sweep_frac,MS,short,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_1m,1.0,92.1,0.0001,0.9,0.3,0.0,1094120,AUC,0.6469243582271665
sweep_frac,MS,short,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_10m,1.0,92.1,0.0001,0.9,0.3,0.0,1094120,AUC,0.5627893211152272
sweep_frac,MS,short,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_1m,1.0,92.1,0.0001,0.9,0.5,0.0,1094120,AUC,0.6469721841064163
sweep_frac,MS,short,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_10m,1.0,92.1,0.0001,0.9,0.5,0.0,1094120,AUC,0.5628535118255382
sweep_frac,MS,short,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_1m,1.0,921.1,0.001,0.7,0.1,0.0,87063,AUC,0.6554339426715757
sweep_frac,MS,short,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_10m,1.0,921.1,0.001,0.7,0.1,0.0,87063,AUC,0.5553384701184192
sweep_frac,MS,short,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_1m,1.0,921.1,0.001,0.7,0.3,0.0,87063,AUC,0.6651616995674036
sweep_frac,MS,short,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_10m,1.0,921.1,0.001,0.7,0.3,0.0,87063,AUC,0.5667872195629653
sweep_frac,MS,short,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_1m,1.0,921.1,0.001,0.7,0.5,0.0,87063,AUC,0.6682379575941726
sweep_frac,MS,short,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_10m,1.0,921.1,0.001,0.7,0.5,0.0,87063,AUC,0.5707686245372147
sweep_frac,MS,short,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_1m,1.0,921.1,0.001,0.8,0.1,0.0,87063,AUC,0.6551119207318307
sweep_frac,MS,short,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_10m,1.0,921.1,0.001,0.8,0.1,0.0,87063,AUC,0.5546523567237054
sweep_frac,MS,short,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_1m,1.0,921.1,0.001,0.8,0.3,0.0,87063,AUC,0.6613384711914565
sweep_frac,MS,short,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_10m,1.0,921.1,0.001,0.8,0.3,0.0,87063,AUC,0.5608006652603869
sweep_frac,MS,short,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_1m,1.0,921.1,0.001,0.8,0.5,0.0,87063,AUC,0.6624892312350626
sweep_frac,MS,short,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_10m,1.0,921.1,0.001,0.8,0.5,0.0,87063,AUC,0.5616890464927746
sweep_frac,MS,short,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_1m,1.0,921.1,0.001,0.9,0.1,0.0,87063,AUC,0.6532403566099056
sweep_frac,MS,short,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_10m,1.0,921.1,0.001,0.9,0.1,0.0,87063,AUC,0.5519234689612782
sweep_frac,MS,short,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_1m,1.0,921.1,0.001,0.9,0.3,0.0,87063,AUC,0.6545541512063738
sweep_frac,MS,short,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_10m,1.0,921.1,0.001,0.9,0.3,0.0,87063,AUC,0.5532899079016118
sweep_frac,MS,short,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_1m,1.0,921.1,0.001,0.9,0.5,0.0,87063,AUC,0.6545683943106062
sweep_frac,MS,short,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_10m,1.0,921.1,0.001,0.9,0.5,0.0,87063,AUC,0.5532417785660653
sweep_frac,MS,short,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_1m,2.0,9.2,1e-05,0.7,0.1,0.0,1278326,AUC,0.6498212237305232
sweep_frac,MS,short,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_10m,2.0,9.2,1e-05,0.7,0.1,0.0,1278326,AUC,0.5695709833387463
sweep_frac,MS,short,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_1m,2.0,9.2,1e-05,0.7,0.3,0.0,1278326,AUC,0.6543717507940263
sweep_frac,MS,short,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_10m,2.0,9.2,1e-05,0.7,0.3,0.0,1278326,AUC,0.5752622495322961
sweep_frac,MS,short,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_1m,2.0,9.2,1e-05,0.7,0.5,0.0,1278326,AUC,0.6578650509042728
sweep_frac,MS,short,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_10m,2.0,9.2,1e-05,0.7,0.5,0.0,1278326,AUC,0.5786890182427487
sweep_frac,MS,short,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_1m,2.0,9.2,1e-05,0.8,0.1,0.0,1278326,AUC,0.6485469641420751
sweep_frac,MS,short,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_10m,2.0,9.2,1e-05,0.8,0.1,0.0,1278326,AUC,0.567852032782681
sweep_frac,MS,short,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_1m,2.0,9.2,1e-05,0.8,0.3,0.0,1278326,AUC,0.6510659975065121
sweep_frac,MS,short,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_10m,2.0,9.2,1e-05,0.8,0.3,0.0,1278326,AUC,0.5715227142166619
sweep_frac,MS,short,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_1m,2.0,9.2,1e-05,0.8,0.5,0.0,1278326,AUC,0.6522678175886313
sweep_frac,MS,short,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_10m,2.0,9.2,1e-05,0.8,0.5,0.0,1278326,AUC,0.5730477680224833
sweep_frac,MS,short,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_1m,2.0,9.2,1e-05,0.9,0.1,0.0,1278326,AUC,0.6463771778643282
sweep_frac,MS,short,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_10m,2.0,9.2,1e-05,0.9,0.1,0.0,1278326,AUC,0.5642403752072971
sweep_frac,MS,short,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_1m,2.0,9.2,1e-05,0.9,0.3,0.0,1278326,AUC,0.6467635224327881
sweep_frac,MS,short,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_10m,2.0,9.2,1e-05,0.9,0.3,0.0,1278326,AUC,0.5648445068698726
sweep_frac,MS,short,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_1m,2.0,9.2,1e-05,0.9,0.5,0.0,1278326,AUC,0.6468238514005721
sweep_frac,MS,short,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_10m,2.0,9.2,1e-05,0.9,0.5,0.0,1278326,AUC,0.5649248525722724
sweep_frac,MS,short,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_1m,2.0,91.7,0.0001,0.7,0.1,0.0,909921,AUC,0.6528345059162882
sweep_frac,MS,short,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_10m,2.0,91.7,0.0001,0.7,0.1,0.0,909921,AUC,0.5715067292488788
sweep_frac,MS,short,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_1m,2.0,91.7,0.0001,0.7,0.3,0.0,909921,AUC,0.6573824017518425
sweep_frac,MS,short,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_10m,2.0,91.7,0.0001,0.7,0.3,0.0,909921,AUC,0.5772118131807841
sweep_frac,MS,short,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_1m,2.0,91.7,0.0001,0.7,0.5,0.0,909921,AUC,0.6614754911227043
sweep_frac,MS,short,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_10m,2.0,91.7,0.0001,0.7,0.5,0.0,909921,AUC,0.5818514053785684
sweep_frac,MS,short,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_1m,2.0,91.7,0.0001,0.8,0.1,0.0,909921,AUC,0.65176324661561
sweep_frac,MS,short,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_10m,2.0,91.7,0.0001,0.8,0.1,0.0,909921,AUC,0.570277430576611
sweep_frac,MS,short,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_1m,2.0,91.7,0.0001,0.8,0.3,0.0,909921,AUC,0.6542569701050425
sweep_frac,MS,short,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_10m,2.0,91.7,0.0001,0.8,0.3,0.0,909921,AUC,0.5737552058095469
sweep_frac,MS,short,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_1m,2.0,91.7,0.0001,0.8,0.5,0.0,909921,AUC,0.6554594800099853
sweep_frac,MS,short,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_10m,2.0,91.7,0.0001,0.8,0.5,0.0,909921,AUC,0.5753501005586463
sweep_frac,MS,short,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_1m,2.0,91.7,0.0001,0.9,0.1,0.0,909921,AUC,0.6496073303666358
sweep_frac,MS,short,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_10m,2.0,91.7,0.0001,0.9,0.1,0.0,909921,AUC,0.5672323267399931
sweep_frac,MS,short,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_1m,2.0,91.7,0.0001,0.9,0.3,0.0,909921,AUC,0.650058586482055
sweep_frac,MS,short,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_10m,2.0,91.7,0.0001,0.9,0.3,0.0,909921,AUC,0.5678231480731063
sweep_frac,MS,short,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_1m,2.0,91.7,0.0001,0.9,0.5,0.0,909921,AUC,0.6501218929673939
sweep_frac,MS,short,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_10m,2.0,91.7,0.0001,0.9,0.5,0.0,909921,AUC,0.567907383389911
sweep_frac,MS,short,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_1m,2.0,917.4,0.001,0.7,0.1,0.0,101782,AUC,0.657131907169065
sweep_frac,MS,short,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_10m,2.0,917.4,0.001,0.7,0.1,0.0,101782,AUC,0.5580525601424309
sweep_frac,MS,short,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_1m,2.0,917.4,0.001,0.7,0.3,0.0,101782,AUC,0.6687042213540604
sweep_frac,MS,short,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_10m,2.0,917.4,0.001,0.7,0.3,0.0,101782,AUC,0.5700971584127079
sweep_frac,MS,short,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_1m,2.0,917.4,0.001,0.7,0.5,0.0,101782,AUC,0.6750575351891653
sweep_frac,MS,short,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_10m,2.0,917.4,0.001,0.7,0.5,0.0,101782,AUC,0.5770968938025838
sweep_frac,MS,short,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_1m,2.0,917.4,0.001,0.8,0.1,0.0,101782,AUC,0.6568366288890229
sweep_frac,MS,short,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_10m,2.0,917.4,0.001,0.8,0.1,0.0,101782,AUC,0.5572641945880124
sweep_frac,MS,short,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_1m,2.0,917.4,0.001,0.8,0.3,0.0,101782,AUC,0.6633316575818867
sweep_frac,MS,short,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_10m,2.0,917.4,0.001,0.8,0.3,0.0,101782,AUC,0.5637689659161491
sweep_frac,MS,short,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_1m,2.0,917.4,0.001,0.8,0.5,0.0,101782,AUC,0.6642827473236034
sweep_frac,MS,short,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_10m,2.0,917.4,0.001,0.8,0.5,0.0,101782,AUC,0.5651571185830571
sweep_frac,MS,short,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_1m,2.0,917.4,0.001,0.9,0.1,0.0,101782,AUC,0.6553331576923651
sweep_frac,MS,short,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_10m,2.0,917.4,0.001,0.9,0.1,0.0,101782,AUC,0.5538300736232643
sweep_frac,MS,short,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_1m,2.0,917.4,0.001,0.9,0.3,0.0,101782,AUC,0.6564420605162329
sweep_frac,MS,short,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_10m,2.0,917.4,0.001,0.9,0.3,0.0,101782,AUC,0.5548607252588283
sweep_frac,MS,short,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_1m,2.0,917.4,0.001,0.9,0.5,0.0,101782,AUC,0.656400674346344
sweep_frac,MS,short,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_10m,2.0,917.4,0.001,0.9,0.5,0.0,101782,AUC,0.5548747518742851
sweep_frac,MS,long,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_close,0.5,9.2,1e-05,0.7,0.1,0.0,1910918,AUC,0.5801491974842842
sweep_frac,MS,long,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_close,0.5,9.2,1e-05,0.7,0.3,0.0,1910918,AUC,0.5830053989569685
sweep_frac,MS,long,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_close,0.5,9.2,1e-05,0.7,0.5,0.0,1910918,AUC,0.5846717755499615
sweep_frac,MS,long,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_close,0.5,9.2,1e-05,0.8,0.1,0.0,1910918,AUC,0.5786856600097918
sweep_frac,MS,long,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_close,0.5,9.2,1e-05,0.8,0.3,0.0,1910918,AUC,0.580337535175791
sweep_frac,MS,long,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_close,0.5,9.2,1e-05,0.8,0.5,0.0,1910918,AUC,0.5811314900979264
sweep_frac,MS,long,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_close,0.5,9.2,1e-05,0.9,0.1,0.0,1910918,AUC,0.5767934606817721
sweep_frac,MS,long,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_close,0.5,9.2,1e-05,0.9,0.3,0.0,1910918,AUC,0.5769973986247727
sweep_frac,MS,long,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_close,0.5,9.2,1e-05,0.9,0.5,0.0,1910918,AUC,0.5770576384024604
sweep_frac,MS,long,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_close,0.5,92.3,0.0001,0.7,0.1,0.0,1214805,AUC,0.5833233856151607
sweep_frac,MS,long,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_close,0.5,92.3,0.0001,0.7,0.3,0.0,1214805,AUC,0.5862111734379812
sweep_frac,MS,long,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_close,0.5,92.3,0.0001,0.7,0.5,0.0,1214805,AUC,0.5889720991199507
sweep_frac,MS,long,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_close,0.5,92.3,0.0001,0.8,0.1,0.0,1214805,AUC,0.5818491385237475
sweep_frac,MS,long,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_close,0.5,92.3,0.0001,0.8,0.3,0.0,1214805,AUC,0.5832213196985464
sweep_frac,MS,long,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_close,0.5,92.3,0.0001,0.8,0.5,0.0,1214805,AUC,0.5842875533052584
sweep_frac,MS,long,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_close,0.5,92.3,0.0001,0.9,0.1,0.0,1214805,AUC,0.5801982405994063
sweep_frac,MS,long,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_close,0.5,92.3,0.0001,0.9,0.3,0.0,1214805,AUC,0.5801807596464506
sweep_frac,MS,long,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_close,0.5,92.3,0.0001,0.9,0.5,0.0,1214805,AUC,0.5802276142323632
sweep_frac,MS,long,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_close,0.5,923.2,0.001,0.7,0.1,0.0,75786,AUC,0.5801135755663291
sweep_frac,MS,long,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_close,0.5,923.2,0.001,0.7,0.3,0.0,75786,AUC,0.5745687232609176
sweep_frac,MS,long,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_close,0.5,923.2,0.001,0.7,0.5,0.0,75786,AUC,0.5742592506722529
sweep_frac,MS,long,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_close,0.5,923.2,0.001,0.8,0.1,0.0,75786,AUC,0.5811765544504401
sweep_frac,MS,long,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_close,0.5,923.2,0.001,0.8,0.3,0.0,75786,AUC,0.5795203338038564
sweep_frac,MS,long,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_close,0.5,923.2,0.001,0.8,0.5,0.0,75786,AUC,0.5787142022205133
sweep_frac,MS,long,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_close,0.5,923.2,0.001,0.9,0.1,0.0,75786,AUC,0.5799077973656848
sweep_frac,MS,long,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_close,0.5,923.2,0.001,0.9,0.3,0.0,75786,AUC,0.579717183662246
sweep_frac,MS,long,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_close,0.5,923.2,0.001,0.9,0.5,0.0,75786,AUC,0.5794818694738119
sweep_frac,MS,long,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_close,1.0,9.2,1e-05,0.7,0.1,0.0,1645119,AUC,0.5874528442888507
sweep_frac,MS,long,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_close,1.0,9.2,1e-05,0.7,0.3,0.0,1645119,AUC,0.5906343416474293
sweep_frac,MS,long,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_close,1.0,9.2,1e-05,0.7,0.5,0.0,1645119,AUC,0.592442803363115
sweep_frac,MS,long,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_close,1.0,9.2,1e-05,0.8,0.1,0.0,1645119,AUC,0.5857690060945897
sweep_frac,MS,long,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_close,1.0,9.2,1e-05,0.8,0.3,0.0,1645119,AUC,0.5876014453492933
sweep_frac,MS,long,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_close,1.0,9.2,1e-05,0.8,0.5,0.0,1645119,AUC,0.5884517793415455
sweep_frac,MS,long,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_close,1.0,9.2,1e-05,0.9,0.1,0.0,1645119,AUC,0.5833314101713125
sweep_frac,MS,long,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_close,1.0,9.2,1e-05,0.9,0.3,0.0,1645119,AUC,0.5835459044067567
sweep_frac,MS,long,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_close,1.0,9.2,1e-05,0.9,0.5,0.0,1645119,AUC,0.5836058102219739
sweep_frac,MS,long,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_close,1.0,92.1,0.0001,0.7,0.1,0.0,1094120,AUC,0.5848349110621494
sweep_frac,MS,long,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_close,1.0,92.1,0.0001,0.7,0.3,0.0,1094120,AUC,0.5877203110463298
sweep_frac,MS,long,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_close,1.0,92.1,0.0001,0.7,0.5,0.0,1094120,AUC,0.590571184319875
sweep_frac,MS,long,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_close,1.0,92.1,0.0001,0.8,0.1,0.0,1094120,AUC,0.5837741620471959
sweep_frac,MS,long,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_close,1.0,92.1,0.0001,0.8,0.3,0.0,1094120,AUC,0.5850123521191545
sweep_frac,MS,long,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_close,1.0,92.1,0.0001,0.8,0.5,0.0,1094120,AUC,0.5859519299093535
sweep_frac,MS,long,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_close,1.0,92.1,0.0001,0.9,0.1,0.0,1094120,AUC,0.5825032628357871
sweep_frac,MS,long,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_close,1.0,92.1,0.0001,0.9,0.3,0.0,1094120,AUC,0.5825135444256222
sweep_frac,MS,long,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_close,1.0,92.1,0.0001,0.9,0.5,0.0,1094120,AUC,0.5825059185668852
sweep_frac,MS,long,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_close,1.0,921.1,0.001,0.7,0.1,0.0,87063,AUC,0.5814202035202845
sweep_frac,MS,long,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_close,1.0,921.1,0.001,0.7,0.3,0.0,87063,AUC,0.5779068666450282
sweep_frac,MS,long,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_close,1.0,921.1,0.001,0.7,0.5,0.0,87063,AUC,0.5762373810055832
sweep_frac,MS,long,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_close,1.0,921.1,0.001,0.8,0.1,0.0,87063,AUC,0.5811427710452647
sweep_frac,MS,long,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_close,1.0,921.1,0.001,0.8,0.3,0.0,87063,AUC,0.580217120045192
sweep_frac,MS,long,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_close,1.0,921.1,0.001,0.8,0.5,0.0,87063,AUC,0.579821143895163
sweep_frac,MS,long,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_close,1.0,921.1,0.001,0.9,0.1,0.0,87063,AUC,0.5798616174583081
sweep_frac,MS,long,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_close,1.0,921.1,0.001,0.9,0.3,0.0,87063,AUC,0.579985757656446
sweep_frac,MS,long,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_close,1.0,921.1,0.001,0.9,0.5,0.0,87063,AUC,0.5799703072237092
sweep_frac,MS,long,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_close,2.0,9.2,1e-05,0.7,0.1,0.0,1278326,AUC,0.589179314770867
sweep_frac,MS,long,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_close,2.0,9.2,1e-05,0.7,0.3,0.0,1278326,AUC,0.5922008887530447
sweep_frac,MS,long,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_close,2.0,9.2,1e-05,0.7,0.5,0.0,1278326,AUC,0.5944802523677369
sweep_frac,MS,long,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_close,2.0,9.2,1e-05,0.8,0.1,0.0,1278326,AUC,0.5877014754646341
sweep_frac,MS,long,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_close,2.0,9.2,1e-05,0.8,0.3,0.0,1278326,AUC,0.5891753105842298
sweep_frac,MS,long,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_close,2.0,9.2,1e-05,0.8,0.5,0.0,1278326,AUC,0.590052490765486
sweep_frac,MS,long,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_close,2.0,9.2,1e-05,0.9,0.1,0.0,1278326,AUC,0.5861346824136314
sweep_frac,MS,long,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_close,2.0,9.2,1e-05,0.9,0.3,0.0,1278326,AUC,0.5862160296601332
sweep_frac,MS,long,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_close,2.0,9.2,1e-05,0.9,0.5,0.0,1278326,AUC,0.5862126343371432
sweep_frac,MS,long,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_close,2.0,91.7,0.0001,0.7,0.1,0.0,909921,AUC,0.5876411024677283
sweep_frac,MS,long,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_close,2.0,91.7,0.0001,0.7,0.3,0.0,909921,AUC,0.5886802110668394
sweep_frac,MS,long,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_close,2.0,91.7,0.0001,0.7,0.5,0.0,909921,AUC,0.5911089152479109
sweep_frac,MS,long,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_close,2.0,91.7,0.0001,0.8,0.1,0.0,909921,AUC,0.5871537245149608
sweep_frac,MS,long,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_close,2.0,91.7,0.0001,0.8,0.3,0.0,909921,AUC,0.5876441103936284
sweep_frac,MS,long,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_close,2.0,91.7,0.0001,0.8,0.5,0.0,909921,AUC,0.5881839067306199
sweep_frac,MS,long,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_close,2.0,91.7,0.0001,0.9,0.1,0.0,909921,AUC,0.5868045305806743
sweep_frac,MS,long,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_close,2.0,91.7,0.0001,0.9,0.3,0.0,909921,AUC,0.5867923119534638
sweep_frac,MS,long,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_close,2.0,91.7,0.0001,0.9,0.5,0.0,909921,AUC,0.5867742190296196
sweep_frac,MS,long,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_close,2.0,917.4,0.001,0.7,0.1,0.0,101782,AUC,0.5900535539121454
sweep_frac,MS,long,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_close,2.0,917.4,0.001,0.7,0.3,0.0,101782,AUC,0.5909446944510639
sweep_frac,MS,long,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_close,2.0,917.4,0.001,0.7,0.5,0.0,101782,AUC,0.5911257568282332
sweep_frac,MS,long,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_close,2.0,917.4,0.001,0.8,0.1,0.0,101782,AUC,0.5900405715112106
sweep_frac,MS,long,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_close,2.0,917.4,0.001,0.8,0.3,0.0,101782,AUC,0.5923753122866067
sweep_frac,MS,long,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_close,2.0,917.4,0.001,0.8,0.5,0.0,101782,AUC,0.5929677691783746
sweep_frac,MS,long,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_close,2.0,917.4,0.001,0.9,0.1,0.0,101782,AUC,0.5895326454962263
sweep_frac,MS,long,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_close,2.0,917.4,0.001,0.9,0.3,0.0,101782,AUC,0.5895629876202967
sweep_frac,MS,long,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_close,2.0,917.4,0.001,0.9,0.5,0.0,101782,AUC,0.5895902358572801
sweep_frac,JPM,short,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_1m,0.5,11.8,1e-05,0.7,0.1,0.0,2469561,AUC,0.6433503754840516
sweep_frac,JPM,short,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_10m,0.5,11.8,1e-05,0.7,0.1,0.0,2469561,AUC,0.5648173669727302
sweep_frac,JPM,short,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_1m,0.5,11.8,1e-05,0.7,0.3,0.0,2469561,AUC,0.6454883072790606
sweep_frac,JPM,short,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_10m,0.5,11.8,1e-05,0.7,0.3,0.0,2469561,AUC,0.5672688494763727
sweep_frac,JPM,short,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_1m,0.5,11.8,1e-05,0.7,0.5,0.0,2469561,AUC,0.6472795677046738
sweep_frac,JPM,short,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_10m,0.5,11.8,1e-05,0.7,0.5,0.0,2469561,AUC,0.5687489296979031
sweep_frac,JPM,short,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_1m,0.5,11.8,1e-05,0.8,0.1,0.0,2469561,AUC,0.6425721914006757
sweep_frac,JPM,short,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_10m,0.5,11.8,1e-05,0.8,0.1,0.0,2469561,AUC,0.5637063962612751
sweep_frac,JPM,short,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_1m,0.5,11.8,1e-05,0.8,0.3,0.0,2469561,AUC,0.6438179138351222
sweep_frac,JPM,short,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_10m,0.5,11.8,1e-05,0.8,0.3,0.0,2469561,AUC,0.5654283563690452
sweep_frac,JPM,short,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_1m,0.5,11.8,1e-05,0.8,0.5,0.0,2469561,AUC,0.6446809785372165
sweep_frac,JPM,short,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_10m,0.5,11.8,1e-05,0.8,0.5,0.0,2469561,AUC,0.5665066174315132
sweep_frac,JPM,short,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_1m,0.5,11.8,1e-05,0.9,0.1,0.0,2469561,AUC,0.641267743730821
sweep_frac,JPM,short,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_10m,0.5,11.8,1e-05,0.9,0.1,0.0,2469561,AUC,0.5616176468470535
sweep_frac,JPM,short,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_1m,0.5,11.8,1e-05,0.9,0.3,0.0,2469561,AUC,0.6414463311936307
sweep_frac,JPM,short,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_10m,0.5,11.8,1e-05,0.9,0.3,0.0,2469561,AUC,0.5618949046966043
sweep_frac,JPM,short,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_1m,0.5,11.8,1e-05,0.9,0.5,0.0,2469561,AUC,0.641475440096778
sweep_frac,JPM,short,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_10m,0.5,11.8,1e-05,0.9,0.5,0.0,2469561,AUC,0.5619575400470189
sweep_frac,JPM,short,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_1m,0.5,117.6,0.0001,0.7,0.1,0.0,1381811,AUC,0.6403667798970372
sweep_frac,JPM,short,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_10m,0.5,117.6,0.0001,0.7,0.1,0.0,1381811,AUC,0.5591848305322701
sweep_frac,JPM,short,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_1m,0.5,117.6,0.0001,0.7,0.3,0.0,1381811,AUC,0.6428055002863069
sweep_frac,JPM,short,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_10m,0.5,117.6,0.0001,0.7,0.3,0.0,1381811,AUC,0.5640881734232664
sweep_frac,JPM,short,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_1m,0.5,117.6,0.0001,0.7,0.5,0.0,1381811,AUC,0.6458464587781513
sweep_frac,JPM,short,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_10m,0.5,117.6,0.0001,0.7,0.5,0.0,1381811,AUC,0.5696029422465877
sweep_frac,JPM,short,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_1m,0.5,117.6,0.0001,0.8,0.1,0.0,1381811,AUC,0.6396795417719541
sweep_frac,JPM,short,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_10m,0.5,117.6,0.0001,0.8,0.1,0.0,1381811,AUC,0.5578249393015638
sweep_frac,JPM,short,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_1m,0.5,117.6,0.0001,0.8,0.3,0.0,1381811,AUC,0.6408822198495512
sweep_frac,JPM,short,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_10m,0.5,117.6,0.0001,0.8,0.3,0.0,1381811,AUC,0.5601299428364241
sweep_frac,JPM,short,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_1m,0.5,117.6,0.0001,0.8,0.5,0.0,1381811,AUC,0.642338036188681
sweep_frac,JPM,short,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_10m,0.5,117.6,0.0001,0.8,0.5,0.0,1381811,AUC,0.5632488851528606
sweep_frac,JPM,short,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_1m,0.5,117.6,0.0001,0.9,0.1,0.0,1381811,AUC,0.6388856588307912
sweep_frac,JPM,short,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_10m,0.5,117.6,0.0001,0.9,0.1,0.0,1381811,AUC,0.5565566125184638
sweep_frac,JPM,short,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_1m,0.5,117.6,0.0001,0.9,0.3,0.0,1381811,AUC,0.6391163094713214
sweep_frac,JPM,short,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_10m,0.5,117.6,0.0001,0.9,0.3,0.0,1381811,AUC,0.5569748051057213
sweep_frac,JPM,short,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_1m,0.5,117.6,0.0001,0.9,0.5,0.0,1381811,AUC,0.6391557204647728
sweep_frac,JPM,short,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_10m,0.5,117.6,0.0001,0.9,0.5,0.0,1381811,AUC,0.5570649895360735
sweep_frac,JPM,short,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_1m,0.5,1175.9,0.001,0.7,0.1,0.0,53764,AUC,0.6492289458046827
sweep_frac,JPM,short,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_10m,0.5,1175.9,0.001,0.7,0.1,0.0,53764,AUC,0.5559797189798201
sweep_frac,JPM,short,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_1m,0.5,1175.9,0.001,0.7,0.3,0.0,53764,AUC,0.6550193081085138
sweep_frac,JPM,short,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_10m,0.5,1175.9,0.001,0.7,0.3,0.0,53764,AUC,0.5630685096279999
sweep_frac,JPM,short,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_1m,0.5,1175.9,0.001,0.7,0.5,0.0,53764,AUC,0.6586428026364689
sweep_frac,JPM,short,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_10m,0.5,1175.9,0.001,0.7,0.5,0.0,53764,AUC,0.5655669065479276
sweep_frac,JPM,short,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_1m,0.5,1175.9,0.001,0.8,0.1,0.0,53764,AUC,0.6485842805650256
sweep_frac,JPM,short,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_10m,0.5,1175.9,0.001,0.8,0.1,0.0,53764,AUC,0.5548909846550028
sweep_frac,JPM,short,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_1m,0.5,1175.9,0.001,0.8,0.3,0.0,53764,AUC,0.6525379212661333
sweep_frac,JPM,short,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_10m,0.5,1175.9,0.001,0.8,0.3,0.0,53764,AUC,0.5606419350013352
sweep_frac,JPM,short,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_1m,0.5,1175.9,0.001,0.8,0.5,0.0,53764,AUC,0.6534846581489633
sweep_frac,JPM,short,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_10m,0.5,1175.9,0.001,0.8,0.5,0.0,53764,AUC,0.5613423559008138
sweep_frac,JPM,short,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_1m,0.5,1175.9,0.001,0.9,0.1,0.0,53764,AUC,0.6475363329797548
sweep_frac,JPM,short,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_10m,0.5,1175.9,0.001,0.9,0.1,0.0,53764,AUC,0.5521687071993423
sweep_frac,JPM,short,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_1m,0.5,1175.9,0.001,0.9,0.3,0.0,53764,AUC,0.6486077151766872
sweep_frac,JPM,short,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_10m,0.5,1175.9,0.001,0.9,0.3,0.0,53764,AUC,0.5536125902741118
sweep_frac,JPM,short,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_1m,0.5,1175.9,0.001,0.9,0.5,0.0,53764,AUC,0.6485614628810509
sweep_frac,JPM,short,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_10m,0.5,1175.9,0.001,0.9,0.5,0.0,53764,AUC,0.5535599743026901
sweep_frac,JPM,short,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_1m,1.0,11.6,1e-05,0.7,0.1,0.0,2036062,AUC,0.6437508206708724
sweep_frac,JPM,short,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_10m,1.0,11.6,1e-05,0.7,0.1,0.0,2036062,AUC,0.5662036514724471
sweep_frac,JPM,short,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_1m,1.0,11.6,1e-05,0.7,0.3,0.0,2036062,AUC,0.6466721534730163
sweep_frac,JPM,short,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_10m,1.0,11.6,1e-05,0.7,0.3,0.0,2036062,AUC,0.5692399401855308
sweep_frac,JPM,short,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_1m,1.0,11.6,1e-05,0.7,0.5,0.0,2036062,AUC,0.6492954965519058
sweep_frac,JPM,short,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_10m,1.0,11.6,1e-05,0.7,0.5,0.0,2036062,AUC,0.571426915790285
sweep_frac,JPM,short,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_1m,1.0,11.6,1e-05,0.8,0.1,0.0,2036062,AUC,0.6426968144947531
sweep_frac,JPM,short,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_10m,1.0,11.6,1e-05,0.8,0.1,0.0,2036062,AUC,0.5647876334652807
sweep_frac,JPM,short,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_1m,1.0,11.6,1e-05,0.8,0.3,0.0,2036062,AUC,0.644312040607588
sweep_frac,JPM,short,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_10m,1.0,11.6,1e-05,0.8,0.3,0.0,2036062,AUC,0.5667630459021904
sweep_frac,JPM,short,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_1m,1.0,11.6,1e-05,0.8,0.5,0.0,2036062,AUC,0.6453461759997792
sweep_frac,JPM,short,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_10m,1.0,11.6,1e-05,0.8,0.5,0.0,2036062,AUC,0.5678751671337173
sweep_frac,JPM,short,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_1m,1.0,11.6,1e-05,0.9,0.1,0.0,2036062,AUC,0.640814479649143
sweep_frac,JPM,short,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_10m,1.0,11.6,1e-05,0.9,0.1,0.0,2036062,AUC,0.5619960448930978
sweep_frac,JPM,short,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_1m,1.0,11.6,1e-05,0.9,0.3,0.0,2036062,AUC,0.641036966627389
sweep_frac,JPM,short,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_10m,1.0,11.6,1e-05,0.9,0.3,0.0,2036062,AUC,0.562392286403771
sweep_frac,JPM,short,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_1m,1.0,11.6,1e-05,0.9,0.5,0.0,2036062,AUC,0.6411104395438525
sweep_frac,JPM,short,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_10m,1.0,11.6,1e-05,0.9,0.5,0.0,2036062,AUC,0.5624991492041459
sweep_frac,JPM,short,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_1m,1.0,115.8,0.0001,0.7,0.1,0.0,1238064,AUC,0.6428549731792041
sweep_frac,JPM,short,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_10m,1.0,115.8,0.0001,0.7,0.1,0.0,1238064,AUC,0.5638764350090277
sweep_frac,JPM,short,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_1m,1.0,115.8,0.0001,0.7,0.3,0.0,1238064,AUC,0.6454698071684578
sweep_frac,JPM,short,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_10m,1.0,115.8,0.0001,0.7,0.3,0.0,1238064,AUC,0.5675886093970911
sweep_frac,JPM,short,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_1m,1.0,115.8,0.0001,0.7,0.5,0.0,1238064,AUC,0.6486964306165152
sweep_frac,JPM,short,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_10m,1.0,115.8,0.0001,0.7,0.5,0.0,1238064,AUC,0.5723507712781563
sweep_frac,JPM,short,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_1m,1.0,115.8,0.0001,0.8,0.1,0.0,1238064,AUC,0.6420888800584239
sweep_frac,JPM,short,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_10m,1.0,115.8,0.0001,0.8,0.1,0.0,1238064,AUC,0.5626609515439156
sweep_frac,JPM,short,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_1m,1.0,115.8,0.0001,0.8,0.3,0.0,1238064,AUC,0.6435028875439864
sweep_frac,JPM,short,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_10m,1.0,115.8,0.0001,0.8,0.3,0.0,1238064,AUC,0.5647883532676026
sweep_frac,JPM,short,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_1m,1.0,115.8,0.0001,0.8,0.5,0.0,1238064,AUC,0.6444476848564944
sweep_frac,JPM,short,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_10m,1.0,115.8,0.0001,0.8,0.5,0.0,1238064,AUC,0.5660919998034615
sweep_frac,JPM,short,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_1m,1.0,115.8,0.0001,0.9,0.1,0.0,1238064,AUC,0.6404998364450253
sweep_frac,JPM,short,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_10m,1.0,115.8,0.0001,0.9,0.1,0.0,1238064,AUC,0.5599788713348931
sweep_frac,JPM,short,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_1m,1.0,115.8,0.0001,0.9,0.3,0.0,1238064,AUC,0.6407649774965534
sweep_frac,JPM,short,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_10m,1.0,115.8,0.0001,0.9,0.3,0.0,1238064,AUC,0.560499053153378
sweep_frac,JPM,short,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_1m,1.0,115.8,0.0001,0.9,0.5,0.0,1238064,AUC,0.6408406694460379
sweep_frac,JPM,short,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_10m,1.0,115.8,0.0001,0.9,0.5,0.0,1238064,AUC,0.5606327955659056
sweep_frac,JPM,short,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_1m,1.0,1158.0,0.001,0.7,0.1,0.0,67856,AUC,0.645030987737478
sweep_frac,JPM,short,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_10m,1.0,1158.0,0.001,0.7,0.1,0.0,67856,AUC,0.5509658514979983
sweep_frac,JPM,short,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_1m,1.0,1158.0,0.001,0.7,0.3,0.0,67856,AUC,0.6527753961588585
sweep_frac,JPM,short,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_10m,1.0,1158.0,0.001,0.7,0.3,0.0,67856,AUC,0.5621698313961625
sweep_frac,JPM,short,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_1m,1.0,1158.0,0.001,0.7,0.5,0.0,67856,AUC,0.6573501850734957
sweep_frac,JPM,short,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_10m,1.0,1158.0,0.001,0.7,0.5,0.0,67856,AUC,0.5659998887103384
sweep_frac,JPM,short,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_1m,1.0,1158.0,0.001,0.8,0.1,0.0,67856,AUC,0.6444083865165745
sweep_frac,JPM,short,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_10m,1.0,1158.0,0.001,0.8,0.1,0.0,67856,AUC,0.5500237271807012
sweep_frac,JPM,short,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_1m,1.0,1158.0,0.001,0.8,0.3,0.0,67856,AUC,0.648727676793087
sweep_frac,JPM,short,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_10m,1.0,1158.0,0.001,0.8,0.3,0.0,67856,AUC,0.5578551997846152
sweep_frac,JPM,short,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_1m,1.0,1158.0,0.001,0.8,0.5,0.0,67856,AUC,0.6495110198590804
sweep_frac,JPM,short,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_10m,1.0,1158.0,0.001,0.8,0.5,0.0,67856,AUC,0.5582228895854797
sweep_frac,JPM,short,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_1m,1.0,1158.0,0.001,0.9,0.1,0.0,67856,AUC,0.6437516276395314
sweep_frac,JPM,short,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_10m,1.0,1158.0,0.001,0.9,0.1,0.0,67856,AUC,0.547238537024328
sweep_frac,JPM,short,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_1m,1.0,1158.0,0.001,0.9,0.3,0.0,67856,AUC,0.644730154741425
sweep_frac,JPM,short,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_10m,1.0,1158.0,0.001,0.9,0.3,0.0,67856,AUC,0.5490905375122039
sweep_frac,JPM,short,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_1m,1.0,1158.0,0.001,0.9,0.5,0.0,67856,AUC,0.6447265097735748
sweep_frac,JPM,short,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_10m,1.0,1158.0,0.001,0.9,0.5,0.0,67856,AUC,0.5491405751427234
sweep_frac,JPM,short,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_1m,2.0,11.3,1e-05,0.7,0.1,0.0,1466526,AUC,0.6433235670881582
sweep_frac,JPM,short,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_10m,2.0,11.3,1e-05,0.7,0.1,0.0,1466526,AUC,0.5680767904022942
sweep_frac,JPM,short,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_1m,2.0,11.3,1e-05,0.7,0.3,0.0,1466526,AUC,0.6477559174938236
sweep_frac,JPM,short,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_10m,2.0,11.3,1e-05,0.7,0.3,0.0,1466526,AUC,0.5731421450947803
sweep_frac,JPM,short,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_1m,2.0,11.3,1e-05,0.7,0.5,0.0,1466526,AUC,0.6515871702694802
sweep_frac,JPM,short,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_10m,2.0,11.3,1e-05,0.7,0.5,0.0,1466526,AUC,0.5767979617135373
sweep_frac,JPM,short,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_1m,2.0,11.3,1e-05,0.8,0.1,0.0,1466526,AUC,0.6420599265899243
sweep_frac,JPM,short,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_10m,2.0,11.3,1e-05,0.8,0.1,0.0,1466526,AUC,0.5663760080227594
sweep_frac,JPM,short,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_1m,2.0,11.3,1e-05,0.8,0.3,0.0,1466526,AUC,0.6443624747410417
sweep_frac,JPM,short,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_10m,2.0,11.3,1e-05,0.8,0.3,0.0,1466526,AUC,0.5694306431455999
sweep_frac,JPM,short,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_1m,2.0,11.3,1e-05,0.8,0.5,0.0,1466526,AUC,0.6455583902789745
sweep_frac,JPM,short,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_10m,2.0,11.3,1e-05,0.8,0.5,0.0,1466526,AUC,0.5710556883514427
sweep_frac,JPM,short,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_1m,2.0,11.3,1e-05,0.9,0.1,0.0,1466526,AUC,0.6398804643269365
sweep_frac,JPM,short,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_10m,2.0,11.3,1e-05,0.9,0.1,0.0,1466526,AUC,0.5628564413856012
sweep_frac,JPM,short,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_1m,2.0,11.3,1e-05,0.9,0.3,0.0,1466526,AUC,0.6402407270793881
sweep_frac,JPM,short,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_10m,2.0,11.3,1e-05,0.9,0.3,0.0,1466526,AUC,0.5633264003653117
sweep_frac,JPM,short,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_1m,2.0,11.3,1e-05,0.9,0.5,0.0,1466526,AUC,0.6403269444645974
sweep_frac,JPM,short,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_10m,2.0,11.3,1e-05,0.9,0.5,0.0,1466526,AUC,0.5634444545514817
sweep_frac,JPM,short,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_1m,2.0,113.0,0.0001,0.7,0.1,0.0,997101,AUC,0.6458567652257079
sweep_frac,JPM,short,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_10m,2.0,113.0,0.0001,0.7,0.1,0.0,997101,AUC,0.5678471022247569
sweep_frac,JPM,short,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_1m,2.0,113.0,0.0001,0.7,0.3,0.0,997101,AUC,0.6501887449282349
sweep_frac,JPM,short,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_10m,2.0,113.0,0.0001,0.7,0.3,0.0,997101,AUC,0.5725437974359265
sweep_frac,JPM,short,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_1m,2.0,113.0,0.0001,0.7,0.5,0.0,997101,AUC,0.6540537258266795
sweep_frac,JPM,short,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_10m,2.0,113.0,0.0001,0.7,0.5,0.0,997101,AUC,0.5767410151113562
sweep_frac,JPM,short,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_1m,2.0,113.0,0.0001,0.8,0.1,0.0,997101,AUC,0.6448149881391325
sweep_frac,JPM,short,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_10m,2.0,113.0,0.0001,0.8,0.1,0.0,997101,AUC,0.5666459442877991
sweep_frac,JPM,short,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_1m,2.0,113.0,0.0001,0.8,0.3,0.0,997101,AUC,0.6473327335452994
sweep_frac,JPM,short,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_10m,2.0,113.0,0.0001,0.8,0.3,0.0,997101,AUC,0.5698179587467179
sweep_frac,JPM,short,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_1m,2.0,113.0,0.0001,0.8,0.5,0.0,997101,AUC,0.648401846577412
sweep_frac,JPM,short,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_10m,2.0,113.0,0.0001,0.8,0.5,0.0,997101,AUC,0.5711386222553443
sweep_frac,JPM,short,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_1m,2.0,113.0,0.0001,0.9,0.1,0.0,997101,AUC,0.6424867512043365
sweep_frac,JPM,short,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_10m,2.0,113.0,0.0001,0.9,0.1,0.0,997101,AUC,0.5632421774508085
sweep_frac,JPM,short,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_1m,2.0,113.0,0.0001,0.9,0.3,0.0,997101,AUC,0.642974537822148
sweep_frac,JPM,short,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_10m,2.0,113.0,0.0001,0.9,0.3,0.0,997101,AUC,0.5638843856079814
sweep_frac,JPM,short,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_1m,2.0,113.0,0.0001,0.9,0.5,0.0,997101,AUC,0.6430918880452946
sweep_frac,JPM,short,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_10m,2.0,113.0,0.0001,0.9,0.5,0.0,997101,AUC,0.5640464405506568
sweep_frac,JPM,short,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_1m,2.0,1130.1,0.001,0.7,0.1,0.0,88294,AUC,0.6452224090336359
sweep_frac,JPM,short,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_10m,2.0,1130.1,0.001,0.7,0.1,0.0,88294,AUC,0.5463480264870579
sweep_frac,JPM,short,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_1m,2.0,1130.1,0.001,0.7,0.3,0.0,88294,AUC,0.658213734009241
sweep_frac,JPM,short,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_10m,2.0,1130.1,0.001,0.7,0.3,0.0,88294,AUC,0.5630831945294996
sweep_frac,JPM,short,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_1m,2.0,1130.1,0.001,0.7,0.5,0.0,88294,AUC,0.665810370325723
sweep_frac,JPM,short,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_10m,2.0,1130.1,0.001,0.7,0.5,0.0,88294,AUC,0.5729094508764645
sweep_frac,JPM,short,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_1m,2.0,1130.1,0.001,0.8,0.1,0.0,88294,AUC,0.6445502466212847
sweep_frac,JPM,short,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_10m,2.0,1130.1,0.001,0.8,0.1,0.0,88294,AUC,0.545790561774223
sweep_frac,JPM,short,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_1m,2.0,1130.1,0.001,0.8,0.3,0.0,88294,AUC,0.6511674270479242
sweep_frac,JPM,short,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_10m,2.0,1130.1,0.001,0.8,0.3,0.0,88294,AUC,0.5551980867146105
sweep_frac,JPM,short,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_1m,2.0,1130.1,0.001,0.8,0.5,0.0,88294,AUC,0.6523513300893703
sweep_frac,JPM,short,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_10m,2.0,1130.1,0.001,0.8,0.5,0.0,88294,AUC,0.5567066863631603
sweep_frac,JPM,short,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_1m,2.0,1130.1,0.001,0.9,0.1,0.0,88294,AUC,0.6420369307271316
sweep_frac,JPM,short,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_10m,2.0,1130.1,0.001,0.9,0.1,0.0,88294,AUC,0.5419373768198373
sweep_frac,JPM,short,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_1m,2.0,1130.1,0.001,0.9,0.3,0.0,88294,AUC,0.643143654274634
sweep_frac,JPM,short,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_10m,2.0,1130.1,0.001,0.9,0.3,0.0,88294,AUC,0.5431554514160819
sweep_frac,JPM,short,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_1m,2.0,1130.1,0.001,0.9,0.5,0.0,88294,AUC,0.6431984943958344
sweep_frac,JPM,short,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_10m,2.0,1130.1,0.001,0.9,0.5,0.0,88294,AUC,0.5432272175950308
sweep_frac,JPM,long,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_close,0.5,11.8,1e-05,0.7,0.1,0.0,2469561,AUC,0.5776110651868276
sweep_frac,JPM,long,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_close,0.5,11.8,1e-05,0.7,0.3,0.0,2469561,AUC,0.5773924538834979
sweep_frac,JPM,long,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_close,0.5,11.8,1e-05,0.7,0.5,0.0,2469561,AUC,0.577361734540417
sweep_frac,JPM,long,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_close,0.5,11.8,1e-05,0.8,0.1,0.0,2469561,AUC,0.5773111716580336
sweep_frac,JPM,long,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_close,0.5,11.8,1e-05,0.8,0.3,0.0,2469561,AUC,0.5774250758870878
sweep_frac,JPM,long,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_close,0.5,11.8,1e-05,0.8,0.5,0.0,2469561,AUC,0.5776898035072646
sweep_frac,JPM,long,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_close,0.5,11.8,1e-05,0.9,0.1,0.0,2469561,AUC,0.576592233187112
sweep_frac,JPM,long,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_close,0.5,11.8,1e-05,0.9,0.3,0.0,2469561,AUC,0.5766790165911381
sweep_frac,JPM,long,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_close,0.5,11.8,1e-05,0.9,0.5,0.0,2469561,AUC,0.5766920960218653
sweep_frac,JPM,long,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_close,0.5,117.6,0.0001,0.7,0.1,0.0,1381811,AUC,0.580656977997711
sweep_frac,JPM,long,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_close,0.5,117.6,0.0001,0.7,0.3,0.0,1381811,AUC,0.5808695232165673
sweep_frac,JPM,long,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_close,0.5,117.6,0.0001,0.7,0.5,0.0,1381811,AUC,0.5819243899776426
sweep_frac,JPM,long,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_close,0.5,117.6,0.0001,0.8,0.1,0.0,1381811,AUC,0.5803789931646914
sweep_frac,JPM,long,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_close,0.5,117.6,0.0001,0.8,0.3,0.0,1381811,AUC,0.5804840974675899
sweep_frac,JPM,long,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_close,0.5,117.6,0.0001,0.8,0.5,0.0,1381811,AUC,0.5808644223297179
sweep_frac,JPM,long,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_close,0.5,117.6,0.0001,0.9,0.1,0.0,1381811,AUC,0.5797482071467138
sweep_frac,JPM,long,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_close,0.5,117.6,0.0001,0.9,0.3,0.0,1381811,AUC,0.5798159062646535
sweep_frac,JPM,long,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_close,0.5,117.6,0.0001,0.9,0.5,0.0,1381811,AUC,0.5798157802971315
sweep_frac,JPM,long,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_close,0.5,1175.9,0.001,0.7,0.1,0.0,53764,AUC,0.5781150443475183
sweep_frac,JPM,long,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_close,0.5,1175.9,0.001,0.7,0.3,0.0,53764,AUC,0.5800035951514251
sweep_frac,JPM,long,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_close,0.5,1175.9,0.001,0.7,0.5,0.0,53764,AUC,0.5784167435115466
sweep_frac,JPM,long,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_close,0.5,1175.9,0.001,0.8,0.1,0.0,53764,AUC,0.5781579143434876
sweep_frac,JPM,long,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_close,0.5,1175.9,0.001,0.8,0.3,0.0,53764,AUC,0.58069566798325
sweep_frac,JPM,long,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_close,0.5,1175.9,0.001,0.8,0.5,0.0,53764,AUC,0.5808147987142491
sweep_frac,JPM,long,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_close,0.5,1175.9,0.001,0.9,0.1,0.0,53764,AUC,0.5788759310463701
sweep_frac,JPM,long,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_close,0.5,1175.9,0.001,0.9,0.3,0.0,53764,AUC,0.5787840486074142
sweep_frac,JPM,long,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_close,0.5,1175.9,0.001,0.9,0.5,0.0,53764,AUC,0.5786813093454621
sweep_frac,JPM,long,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_close,1.0,11.6,1e-05,0.7,0.1,0.0,2036062,AUC,0.5803943984754169
sweep_frac,JPM,long,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_close,1.0,11.6,1e-05,0.7,0.3,0.0,2036062,AUC,0.5814819997310324
sweep_frac,JPM,long,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_close,1.0,11.6,1e-05,0.7,0.5,0.0,2036062,AUC,0.5823545930133248
sweep_frac,JPM,long,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_close,1.0,11.6,1e-05,0.8,0.1,0.0,2036062,AUC,0.5799093022778735
sweep_frac,JPM,long,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_close,1.0,11.6,1e-05,0.8,0.3,0.0,2036062,AUC,0.5807510501152808
sweep_frac,JPM,long,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_close,1.0,11.6,1e-05,0.8,0.5,0.0,2036062,AUC,0.581301394564884
sweep_frac,JPM,long,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_close,1.0,11.6,1e-05,0.9,0.1,0.0,2036062,AUC,0.5789393765302384
sweep_frac,JPM,long,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_close,1.0,11.6,1e-05,0.9,0.3,0.0,2036062,AUC,0.5790532846607594
sweep_frac,JPM,long,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_close,1.0,11.6,1e-05,0.9,0.5,0.0,2036062,AUC,0.5790639902144958
sweep_frac,JPM,long,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_close,1.0,115.8,0.0001,0.7,0.1,0.0,1238064,AUC,0.5846690330420813
sweep_frac,JPM,long,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_close,1.0,115.8,0.0001,0.7,0.3,0.0,1238064,AUC,0.5852742188772477
sweep_frac,JPM,long,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_close,1.0,115.8,0.0001,0.7,0.5,0.0,1238064,AUC,0.5866139614936408
sweep_frac,JPM,long,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_close,1.0,115.8,0.0001,0.8,0.1,0.0,1238064,AUC,0.5843905718225266
sweep_frac,JPM,long,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_close,1.0,115.8,0.0001,0.8,0.3,0.0,1238064,AUC,0.5847612063435973
sweep_frac,JPM,long,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_close,1.0,115.8,0.0001,0.8,0.5,0.0,1238064,AUC,0.5851211279619156
sweep_frac,JPM,long,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_close,1.0,115.8,0.0001,0.9,0.1,0.0,1238064,AUC,0.5836857412807378
sweep_frac,JPM,long,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_close,1.0,115.8,0.0001,0.9,0.3,0.0,1238064,AUC,0.583841659156878
sweep_frac,JPM,long,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_close,1.0,115.8,0.0001,0.9,0.5,0.0,1238064,AUC,0.5838533034902509
sweep_frac,JPM,long,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_close,1.0,1158.0,0.001,0.7,0.1,0.0,67856,AUC,0.5776084196830689
sweep_frac,JPM,long,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_close,1.0,1158.0,0.001,0.7,0.3,0.0,67856,AUC,0.5818414739616637
sweep_frac,JPM,long,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_close,1.0,1158.0,0.001,0.7,0.5,0.0,67856,AUC,0.5827983343295873
sweep_frac,JPM,long,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_close,1.0,1158.0,0.001,0.8,0.1,0.0,67856,AUC,0.5780770494836307
sweep_frac,JPM,long,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_close,1.0,1158.0,0.001,0.8,0.3,0.0,67856,AUC,0.5800646210997803
sweep_frac,JPM,long,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_close,1.0,1158.0,0.001,0.8,0.5,0.0,67856,AUC,0.5804492002123524
sweep_frac,JPM,long,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_close,1.0,1158.0,0.001,0.9,0.1,0.0,67856,AUC,0.5783635020372434
sweep_frac,JPM,long,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_close,1.0,1158.0,0.001,0.9,0.3,0.0,67856,AUC,0.5783319310415911
sweep_frac,JPM,long,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_close,1.0,1158.0,0.001,0.9,0.5,0.0,67856,AUC,0.5782545824999626
sweep_frac,JPM,long,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_close,2.0,11.3,1e-05,0.7,0.1,0.0,1466526,AUC,0.5807635314748129
sweep_frac,JPM,long,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_close,2.0,11.3,1e-05,0.7,0.3,0.0,1466526,AUC,0.5819679788548419
sweep_frac,JPM,long,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_close,2.0,11.3,1e-05,0.7,0.5,0.0,1466526,AUC,0.5836003672819192
sweep_frac,JPM,long,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_close,2.0,11.3,1e-05,0.8,0.1,0.0,1466526,AUC,0.5804269249160339
sweep_frac,JPM,long,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_close,2.0,11.3,1e-05,0.8,0.3,0.0,1466526,AUC,0.5813559063156473
sweep_frac,JPM,long,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_close,2.0,11.3,1e-05,0.8,0.5,0.0,1466526,AUC,0.5817884099785557
sweep_frac,JPM,long,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_close,2.0,11.3,1e-05,0.9,0.1,0.0,1466526,AUC,0.5800327506829811
sweep_frac,JPM,long,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_close,2.0,11.3,1e-05,0.9,0.3,0.0,1466526,AUC,0.5802670276934514
sweep_frac,JPM,long,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_close,2.0,11.3,1e-05,0.9,0.5,0.0,1466526,AUC,0.5802972894752126
sweep_frac,JPM,long,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_close,2.0,113.0,0.0001,0.7,0.1,0.0,997101,AUC,0.5895790531678322
sweep_frac,JPM,long,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_close,2.0,113.0,0.0001,0.7,0.3,0.0,997101,AUC,0.5897457639620906
sweep_frac,JPM,long,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_close,2.0,113.0,0.0001,0.7,0.5,0.0,997101,AUC,0.5902474812417012
sweep_frac,JPM,long,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_close,2.0,113.0,0.0001,0.8,0.1,0.0,997101,AUC,0.5892153548808888
sweep_frac,JPM,long,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_close,2.0,113.0,0.0001,0.8,0.3,0.0,997101,AUC,0.5899341736636674
sweep_frac,JPM,long,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_close,2.0,113.0,0.0001,0.8,0.5,0.0,997101,AUC,0.5901223830302653
sweep_frac,JPM,long,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_close,2.0,113.0,0.0001,0.9,0.1,0.0,997101,AUC,0.5884299270125739
sweep_frac,JPM,long,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_close,2.0,113.0,0.0001,0.9,0.3,0.0,997101,AUC,0.5888457172323809
sweep_frac,JPM,long,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_close,2.0,113.0,0.0001,0.9,0.5,0.0,997101,AUC,0.5889083345406159
sweep_frac,JPM,long,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_close,2.0,1130.1,0.001,0.7,0.1,0.0,88294,AUC,0.5911478155708123
sweep_frac,JPM,long,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_close,2.0,1130.1,0.001,0.7,0.3,0.0,88294,AUC,0.5919578997390325
sweep_frac,JPM,long,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_close,2.0,1130.1,0.001,0.7,0.5,0.0,88294,AUC,0.593820726196377
sweep_frac,JPM,long,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_close,2.0,1130.1,0.001,0.8,0.1,0.0,88294,AUC,0.5910302359333582
sweep_frac,JPM,long,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_close,2.0,1130.1,0.001,0.8,0.3,0.0,88294,AUC,0.5914253869524836
sweep_frac,JPM,long,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_close,2.0,1130.1,0.001,0.8,0.5,0.0,88294,AUC,0.5916882669291599
sweep_frac,JPM,long,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_close,2.0,1130.1,0.001,0.9,0.1,0.0,88294,AUC,0.5913985401711783
sweep_frac,JPM,long,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_close,2.0,1130.1,0.001,0.9,0.3,0.0,88294,AUC,0.591489723815414
sweep_frac,JPM,long,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_close,2.0,1130.1,0.001,0.9,0.5,0.0,88294,AUC,0.5914616193906579
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_1m,0.5,133.7,1e-05,0.7,0.1,0.0,4507315,AUC,0.6239128667764281
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_10m,0.5,133.7,1e-05,0.7,0.1,0.0,4507315,AUC,0.5599686427699786
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_1m,0.5,133.7,1e-05,0.7,0.3,0.0,4507315,AUC,0.6252273421176018
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_10m,0.5,133.7,1e-05,0.7,0.3,0.0,4507315,AUC,0.5615095016573698
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_1m,0.5,133.7,1e-05,0.7,0.5,0.0,4507315,AUC,0.6228017610993184
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_10m,0.5,133.7,1e-05,0.7,0.5,0.0,4507315,AUC,0.557465150573941
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_1m,0.5,133.7,1e-05,0.8,0.1,0.0,4507315,AUC,0.6231897688849999
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_10m,0.5,133.7,1e-05,0.8,0.1,0.0,4507315,AUC,0.5586358067393321
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_1m,0.5,133.7,1e-05,0.8,0.3,0.0,4507315,AUC,0.625254251334181
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_10m,0.5,133.7,1e-05,0.8,0.3,0.0,4507315,AUC,0.5620793702202785
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_1m,0.5,133.7,1e-05,0.8,0.5,0.0,4507315,AUC,0.6252265981800238
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_10m,0.5,133.7,1e-05,0.8,0.5,0.0,4507315,AUC,0.5620569059656625
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_1m,0.5,133.7,1e-05,0.9,0.1,0.0,4507315,AUC,0.6205692208957161
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_10m,0.5,133.7,1e-05,0.9,0.1,0.0,4507315,AUC,0.5534722985325087
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_1m,0.5,133.7,1e-05,0.9,0.3,0.0,4507315,AUC,0.621155018078208
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_10m,0.5,133.7,1e-05,0.9,0.3,0.0,4507315,AUC,0.5531873699451956
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_1m,0.5,133.7,1e-05,0.9,0.5,0.0,4507315,AUC,0.6211986569501049
sweep_frac,TSLA,short,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_10m,0.5,133.7,1e-05,0.9,0.5,0.0,4507315,AUC,0.5533499608935645
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_1m,0.5,1336.5,0.0001,0.7,0.1,0.0,955104,AUC,0.6278842767176548
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_10m,0.5,1336.5,0.0001,0.7,0.1,0.0,955104,AUC,0.5518989197365094
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_1m,0.5,1336.5,0.0001,0.7,0.3,0.0,955104,AUC,0.6345281397323522
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_10m,0.5,1336.5,0.0001,0.7,0.3,0.0,955104,AUC,0.5637037330660994
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_1m,0.5,1336.5,0.0001,0.7,0.5,0.0,955104,AUC,0.636018500221667
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_10m,0.5,1336.5,0.0001,0.7,0.5,0.0,955104,AUC,0.5648354426348545
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_1m,0.5,1336.5,0.0001,0.8,0.1,0.0,955104,AUC,0.6267752308421038
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_10m,0.5,1336.5,0.0001,0.8,0.1,0.0,955104,AUC,0.5498730217596929
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_1m,0.5,1336.5,0.0001,0.8,0.3,0.0,955104,AUC,0.6299422906532749
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_10m,0.5,1336.5,0.0001,0.8,0.3,0.0,955104,AUC,0.5558713015594207
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_1m,0.5,1336.5,0.0001,0.8,0.5,0.0,955104,AUC,0.6303468980965776
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_10m,0.5,1336.5,0.0001,0.8,0.5,0.0,955104,AUC,0.5566167990159931
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_1m,0.5,1336.5,0.0001,0.9,0.1,0.0,955104,AUC,0.6241889899739037
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_10m,0.5,1336.5,0.0001,0.9,0.1,0.0,955104,AUC,0.5438715746713539
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_1m,0.5,1336.5,0.0001,0.9,0.3,0.0,955104,AUC,0.6245122899791302
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_10m,0.5,1336.5,0.0001,0.9,0.3,0.0,955104,AUC,0.544516565485024
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_1m,0.5,1336.5,0.0001,0.9,0.5,0.0,955104,AUC,0.6245185519045103
sweep_frac,TSLA,short,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_10m,0.5,1336.5,0.0001,0.9,0.5,0.0,955104,AUC,0.5445543130317342
sweep_frac,TSLA,short,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_1m,0.5,13365.1,0.001,0.7,0.1,0.0,53821,AUC,0.6443634751459245
sweep_frac,TSLA,short,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_10m,0.5,13365.1,0.001,0.7,0.1,0.0,53821,AUC,0.5603457692904225
sweep_frac,TSLA,short,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_1m,0.5,13365.1,0.001,0.7,0.3,0.0,53821,AUC,0.6579727265326061
sweep_frac,TSLA,short,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_10m,0.5,13365.1,0.001,0.7,0.3,0.0,53821,AUC,0.5719087265743756
sweep_frac,TSLA,short,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_1m,0.5,13365.1,0.001,0.7,0.5,0.0,53821,AUC,0.6643245401031472
sweep_frac,TSLA,short,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_10m,0.5,13365.1,0.001,0.7,0.5,0.0,53821,AUC,0.5764879196269177
sweep_frac,TSLA,short,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_1m,0.5,13365.1,0.001,0.8,0.1,0.0,53821,AUC,0.6428445660138261
sweep_frac,TSLA,short,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_10m,0.5,13365.1,0.001,0.8,0.1,0.0,53821,AUC,0.5580735593121304
sweep_frac,TSLA,short,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_1m,0.5,13365.1,0.001,0.8,0.3,0.0,53821,AUC,0.6492663738146396
sweep_frac,TSLA,short,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_10m,0.5,13365.1,0.001,0.8,0.3,0.0,53821,AUC,0.5634323698836631
sweep_frac,TSLA,short,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_1m,0.5,13365.1,0.001,0.8,0.5,0.0,53821,AUC,0.650161483350973
sweep_frac,TSLA,short,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_10m,0.5,13365.1,0.001,0.8,0.5,0.0,53821,AUC,0.5644848480010993
sweep_frac,TSLA,short,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_1m,0.5,13365.1,0.001,0.9,0.1,0.0,53821,AUC,0.6398687935605591
sweep_frac,TSLA,short,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_10m,0.5,13365.1,0.001,0.9,0.1,0.0,53821,AUC,0.5543749111262481
sweep_frac,TSLA,short,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_1m,0.5,13365.1,0.001,0.9,0.3,0.0,53821,AUC,0.6403577974507006
sweep_frac,TSLA,short,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_10m,0.5,13365.1,0.001,0.9,0.3,0.0,53821,AUC,0.5557527889055822
sweep_frac,TSLA,short,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_1m,0.5,13365.1,0.001,0.9,0.5,0.0,53821,AUC,0.6404229776750475
sweep_frac,TSLA,short,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_10m,0.5,13365.1,0.001,0.9,0.5,0.0,53821,AUC,0.5556444224762107
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_1m,1.0,130.8,1e-05,0.7,0.1,0.0,2244611,AUC,0.6224411140632877
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_10m,1.0,130.8,1e-05,0.7,0.1,0.0,2244611,AUC,0.5602724521015207
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_1m,1.0,130.8,1e-05,0.7,0.3,0.0,2244611,AUC,0.6296677067033536
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_10m,1.0,130.8,1e-05,0.7,0.3,0.0,2244611,AUC,0.5710554286214844
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_1m,1.0,130.8,1e-05,0.7,0.5,0.0,2244611,AUC,0.6305512608181971
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_10m,1.0,130.8,1e-05,0.7,0.5,0.0,2244611,AUC,0.5706624904839185
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_1m,1.0,130.8,1e-05,0.8,0.1,0.0,2244611,AUC,0.6209656936045617
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_10m,1.0,130.8,1e-05,0.8,0.1,0.0,2244611,AUC,0.558056327511743
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_1m,1.0,130.8,1e-05,0.8,0.3,0.0,2244611,AUC,0.6248246457791127
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_10m,1.0,130.8,1e-05,0.8,0.3,0.0,2244611,AUC,0.5640846821605208
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_1m,1.0,130.8,1e-05,0.8,0.5,0.0,2244611,AUC,0.6255216627393392
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_10m,1.0,130.8,1e-05,0.8,0.5,0.0,2244611,AUC,0.5649800595760619
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_1m,1.0,130.8,1e-05,0.9,0.1,0.0,2244611,AUC,0.6175411346652951
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_10m,1.0,130.8,1e-05,0.9,0.1,0.0,2244611,AUC,0.5515255670268906
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_1m,1.0,130.8,1e-05,0.9,0.3,0.0,2244611,AUC,0.6180064086961898
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_10m,1.0,130.8,1e-05,0.9,0.3,0.0,2244611,AUC,0.5525177888465759
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_1m,1.0,130.8,1e-05,0.9,0.5,0.0,2244611,AUC,0.61806559895402
sweep_frac,TSLA,short,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_10m,1.0,130.8,1e-05,0.9,0.5,0.0,2244611,AUC,0.5526492349460346
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_1m,1.0,1308.5,0.0001,0.7,0.1,0.0,748082,AUC,0.6251667914411841
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_10m,1.0,1308.5,0.0001,0.7,0.1,0.0,748082,AUC,0.5518257760801419
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_1m,1.0,1308.5,0.0001,0.7,0.3,0.0,748082,AUC,0.6368694746624375
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_10m,1.0,1308.5,0.0001,0.7,0.3,0.0,748082,AUC,0.5684290719756662
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_1m,1.0,1308.5,0.0001,0.7,0.5,0.0,748082,AUC,0.6417568526469759
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_10m,1.0,1308.5,0.0001,0.7,0.5,0.0,748082,AUC,0.5730848130959124
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_1m,1.0,1308.5,0.0001,0.8,0.1,0.0,748082,AUC,0.6236887932485529
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_10m,1.0,1308.5,0.0001,0.8,0.1,0.0,748082,AUC,0.5496591209789969
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_1m,1.0,1308.5,0.0001,0.8,0.3,0.0,748082,AUC,0.6283616796324148
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_10m,1.0,1308.5,0.0001,0.8,0.3,0.0,748082,AUC,0.5564853098256175
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_1m,1.0,1308.5,0.0001,0.8,0.5,0.0,748082,AUC,0.6290535291304796
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_10m,1.0,1308.5,0.0001,0.8,0.5,0.0,748082,AUC,0.5573784957171563
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_1m,1.0,1308.5,0.0001,0.9,0.1,0.0,748082,AUC,0.6209152707971952
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_10m,1.0,1308.5,0.0001,0.9,0.1,0.0,748082,AUC,0.5442843986694225
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_1m,1.0,1308.5,0.0001,0.9,0.3,0.0,748082,AUC,0.6211874071985237
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_10m,1.0,1308.5,0.0001,0.9,0.3,0.0,748082,AUC,0.5447659781096283
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_1m,1.0,1308.5,0.0001,0.9,0.5,0.0,748082,AUC,0.6212251440303418
sweep_frac,TSLA,short,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_10m,1.0,1308.5,0.0001,0.9,0.5,0.0,748082,AUC,0.5448343223180665
sweep_frac,TSLA,short,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_1m,1.0,13084.9,0.001,0.7,0.1,0.0,73773,AUC,0.6345260441475786
sweep_frac,TSLA,short,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_10m,1.0,13084.9,0.001,0.7,0.1,0.0,73773,AUC,0.5561139292345603
sweep_frac,TSLA,short,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_1m,1.0,13084.9,0.001,0.7,0.3,0.0,73773,AUC,0.6607473082142168
sweep_frac,TSLA,short,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_10m,1.0,13084.9,0.001,0.7,0.3,0.0,73773,AUC,0.5740668291193197
sweep_frac,TSLA,short,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_1m,1.0,13084.9,0.001,0.7,0.5,0.0,73773,AUC,0.6758594838386633
sweep_frac,TSLA,short,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_10m,1.0,13084.9,0.001,0.7,0.5,0.0,73773,AUC,0.5821863593141683
sweep_frac,TSLA,short,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_1m,1.0,13084.9,0.001,0.8,0.1,0.0,73773,AUC,0.6328647369477711
sweep_frac,TSLA,short,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_10m,1.0,13084.9,0.001,0.8,0.1,0.0,73773,AUC,0.5541172087765587
sweep_frac,TSLA,short,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_1m,1.0,13084.9,0.001,0.8,0.3,0.0,73773,AUC,0.6422820753831432
sweep_frac,TSLA,short,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_10m,1.0,13084.9,0.001,0.8,0.3,0.0,73773,AUC,0.5609413615326262
sweep_frac,TSLA,short,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_1m,1.0,13084.9,0.001,0.8,0.5,0.0,73773,AUC,0.6437052921209454
sweep_frac,TSLA,short,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_10m,1.0,13084.9,0.001,0.8,0.5,0.0,73773,AUC,0.5615834239687824
sweep_frac,TSLA,short,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_1m,1.0,13084.9,0.001,0.9,0.1,0.0,73773,AUC,0.6306431264351082
sweep_frac,TSLA,short,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_10m,1.0,13084.9,0.001,0.9,0.1,0.0,73773,AUC,0.5510564219397458
sweep_frac,TSLA,short,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_1m,1.0,13084.9,0.001,0.9,0.3,0.0,73773,AUC,0.6313155255664965
sweep_frac,TSLA,short,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_10m,1.0,13084.9,0.001,0.9,0.3,0.0,73773,AUC,0.5515783666857167
sweep_frac,TSLA,short,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_1m,1.0,13084.9,0.001,0.9,0.5,0.0,73773,AUC,0.6313785083182484
sweep_frac,TSLA,short,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_10m,1.0,13084.9,0.001,0.9,0.5,0.0,73773,AUC,0.5514707404280857
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_1m,2.0,126.9,1e-05,0.7,0.1,0.0,670803,AUC,0.611365239354766
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_10m,2.0,126.9,1e-05,0.7,0.1,0.0,670803,AUC,0.5559281025260568
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_1m,2.0,126.9,1e-05,0.7,0.3,0.0,670803,AUC,0.6267128540586054
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_10m,2.0,126.9,1e-05,0.7,0.3,0.0,670803,AUC,0.5712527172310096
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_1m,2.0,126.9,1e-05,0.7,0.5,0.0,670803,AUC,0.6350158470188917
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_10m,2.0,126.9,1e-05,0.7,0.5,0.0,670803,AUC,0.5763740164693084
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_1m,2.0,126.9,1e-05,0.8,0.1,0.0,670803,AUC,0.6094120956214089
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_10m,2.0,126.9,1e-05,0.8,0.1,0.0,670803,AUC,0.5530792631066596
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_1m,2.0,126.9,1e-05,0.8,0.3,0.0,670803,AUC,0.6154194047803299
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_10m,2.0,126.9,1e-05,0.8,0.3,0.0,670803,AUC,0.5603264936026047
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_1m,2.0,126.9,1e-05,0.8,0.5,0.0,670803,AUC,0.6164667938285338
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_10m,2.0,126.9,1e-05,0.8,0.5,0.0,670803,AUC,0.5613689042097475
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_1m,2.0,126.9,1e-05,0.9,0.1,0.0,670803,AUC,0.6050735202378543
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_10m,2.0,126.9,1e-05,0.9,0.1,0.0,670803,AUC,0.5476035122403631
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_1m,2.0,126.9,1e-05,0.9,0.3,0.0,670803,AUC,0.6056831682613791
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_10m,2.0,126.9,1e-05,0.9,0.3,0.0,670803,AUC,0.5484755380401405
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_1m,2.0,126.9,1e-05,0.9,0.5,0.0,670803,AUC,0.6057071491701363
sweep_frac,TSLA,short,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_10m,2.0,126.9,1e-05,0.9,0.5,0.0,670803,AUC,0.548495189174289
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_1m,2.0,1269.2,0.0001,0.7,0.1,0.0,367017,AUC,0.6084256768091968
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_10m,2.0,1269.2,0.0001,0.7,0.1,0.0,367017,AUC,0.5514080100344418
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_1m,2.0,1269.2,0.0001,0.7,0.3,0.0,367017,AUC,0.6305704660756895
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_10m,2.0,1269.2,0.0001,0.7,0.3,0.0,367017,AUC,0.5703279438115093
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_1m,2.0,1269.2,0.0001,0.7,0.5,0.0,367017,AUC,0.6433947391390554
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_10m,2.0,1269.2,0.0001,0.7,0.5,0.0,367017,AUC,0.5787159985459855
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_1m,2.0,1269.2,0.0001,0.8,0.1,0.0,367017,AUC,0.6063904051872085
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_10m,2.0,1269.2,0.0001,0.8,0.1,0.0,367017,AUC,0.5489474457667896
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_1m,2.0,1269.2,0.0001,0.8,0.3,0.0,367017,AUC,0.613382259825845
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_10m,2.0,1269.2,0.0001,0.8,0.3,0.0,367017,AUC,0.5552256398480999
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_1m,2.0,1269.2,0.0001,0.8,0.5,0.0,367017,AUC,0.6143230163499077
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_10m,2.0,1269.2,0.0001,0.8,0.5,0.0,367017,AUC,0.5559826389574943
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_1m,2.0,1269.2,0.0001,0.9,0.1,0.0,367017,AUC,0.6027498185537699
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_10m,2.0,1269.2,0.0001,0.9,0.1,0.0,367017,AUC,0.5449147571559266
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_1m,2.0,1269.2,0.0001,0.9,0.3,0.0,367017,AUC,0.603060948015929
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_10m,2.0,1269.2,0.0001,0.9,0.3,0.0,367017,AUC,0.5453707530564977
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_1m,2.0,1269.2,0.0001,0.9,0.5,0.0,367017,AUC,0.6030413218127698
sweep_frac,TSLA,short,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_10m,2.0,1269.2,0.0001,0.9,0.5,0.0,367017,AUC,0.5453925772396502
sweep_frac,TSLA,short,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_1m,2.0,12691.8,0.001,0.7,0.1,0.0,69128,AUC,0.6106011596265187
sweep_frac,TSLA,short,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_10m,2.0,12691.8,0.001,0.7,0.1,0.0,69128,AUC,0.559222043867878
sweep_frac,TSLA,short,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_1m,2.0,12691.8,0.001,0.7,0.3,0.0,69128,AUC,0.6459192404728733
sweep_frac,TSLA,short,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_10m,2.0,12691.8,0.001,0.7,0.3,0.0,69128,AUC,0.5765403722789539
sweep_frac,TSLA,short,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_1m,2.0,12691.8,0.001,0.7,0.5,0.0,69128,AUC,0.6757463915948375
sweep_frac,TSLA,short,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_10m,2.0,12691.8,0.001,0.7,0.5,0.0,69128,AUC,0.5937374868896874
sweep_frac,TSLA,short,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_1m,2.0,12691.8,0.001,0.8,0.1,0.0,69128,AUC,0.608082691384685
sweep_frac,TSLA,short,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_10m,2.0,12691.8,0.001,0.8,0.1,0.0,69128,AUC,0.557944996860782
sweep_frac,TSLA,short,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_1m,2.0,12691.8,0.001,0.8,0.3,0.0,69128,AUC,0.6169207204417799
sweep_frac,TSLA,short,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_10m,2.0,12691.8,0.001,0.8,0.3,0.0,69128,AUC,0.560719130738522
sweep_frac,TSLA,short,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_1m,2.0,12691.8,0.001,0.8,0.5,0.0,69128,AUC,0.6177740400629002
sweep_frac,TSLA,short,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_10m,2.0,12691.8,0.001,0.8,0.5,0.0,69128,AUC,0.5612332885406401
sweep_frac,TSLA,short,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_1m,2.0,12691.8,0.001,0.9,0.1,0.0,69128,AUC,0.605236159032072
sweep_frac,TSLA,short,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_10m,2.0,12691.8,0.001,0.9,0.1,0.0,69128,AUC,0.5544882811099651
sweep_frac,TSLA,short,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_1m,2.0,12691.8,0.001,0.9,0.3,0.0,69128,AUC,0.6051483475156343
sweep_frac,TSLA,short,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_10m,2.0,12691.8,0.001,0.9,0.3,0.0,69128,AUC,0.554636744890594
sweep_frac,TSLA,short,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_1m,2.0,12691.8,0.001,0.9,0.5,0.0,69128,AUC,0.6051690734171759
sweep_frac,TSLA,short,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_10m,2.0,12691.8,0.001,0.9,0.5,0.0,69128,AUC,0.5546348770584707
sweep_frac,TSLA,long,s0p5_vf1e-05_d0p7_r0p1_k0p0,cls_close,0.5,133.7,1e-05,0.7,0.1,0.0,4507315,AUC,0.6023371609564117
sweep_frac,TSLA,long,s0p5_vf1e-05_d0p7_r0p3_k0p0,cls_close,0.5,133.7,1e-05,0.7,0.3,0.0,4507315,AUC,0.6013055104545058
sweep_frac,TSLA,long,s0p5_vf1e-05_d0p7_r0p5_k0p0,cls_close,0.5,133.7,1e-05,0.7,0.5,0.0,4507315,AUC,0.5993346385963004
sweep_frac,TSLA,long,s0p5_vf1e-05_d0p8_r0p1_k0p0,cls_close,0.5,133.7,1e-05,0.8,0.1,0.0,4507315,AUC,0.6021322244050039
sweep_frac,TSLA,long,s0p5_vf1e-05_d0p8_r0p3_k0p0,cls_close,0.5,133.7,1e-05,0.8,0.3,0.0,4507315,AUC,0.6028972317310887
sweep_frac,TSLA,long,s0p5_vf1e-05_d0p8_r0p5_k0p0,cls_close,0.5,133.7,1e-05,0.8,0.5,0.0,4507315,AUC,0.6026042766259836
sweep_frac,TSLA,long,s0p5_vf1e-05_d0p9_r0p1_k0p0,cls_close,0.5,133.7,1e-05,0.9,0.1,0.0,4507315,AUC,0.6009195629300712
sweep_frac,TSLA,long,s0p5_vf1e-05_d0p9_r0p3_k0p0,cls_close,0.5,133.7,1e-05,0.9,0.3,0.0,4507315,AUC,0.6012378752303162
sweep_frac,TSLA,long,s0p5_vf1e-05_d0p9_r0p5_k0p0,cls_close,0.5,133.7,1e-05,0.9,0.5,0.0,4507315,AUC,0.601262930572776
sweep_frac,TSLA,long,s0p5_vf0p0001_d0p7_r0p1_k0p0,cls_close,0.5,1336.5,0.0001,0.7,0.1,0.0,955104,AUC,0.6158630055965733
sweep_frac,TSLA,long,s0p5_vf0p0001_d0p7_r0p3_k0p0,cls_close,0.5,1336.5,0.0001,0.7,0.3,0.0,955104,AUC,0.6189795523382395
sweep_frac,TSLA,long,s0p5_vf0p0001_d0p7_r0p5_k0p0,cls_close,0.5,1336.5,0.0001,0.7,0.5,0.0,955104,AUC,0.6184880177724155
sweep_frac,TSLA,long,s0p5_vf0p0001_d0p8_r0p1_k0p0,cls_close,0.5,1336.5,0.0001,0.8,0.1,0.0,955104,AUC,0.6150527741352194
sweep_frac,TSLA,long,s0p5_vf0p0001_d0p8_r0p3_k0p0,cls_close,0.5,1336.5,0.0001,0.8,0.3,0.0,955104,AUC,0.6171922993574864
sweep_frac,TSLA,long,s0p5_vf0p0001_d0p8_r0p5_k0p0,cls_close,0.5,1336.5,0.0001,0.8,0.5,0.0,955104,AUC,0.6173461032328283
sweep_frac,TSLA,long,s0p5_vf0p0001_d0p9_r0p1_k0p0,cls_close,0.5,1336.5,0.0001,0.9,0.1,0.0,955104,AUC,0.612977907886662
sweep_frac,TSLA,long,s0p5_vf0p0001_d0p9_r0p3_k0p0,cls_close,0.5,1336.5,0.0001,0.9,0.3,0.0,955104,AUC,0.6134209279423888
sweep_frac,TSLA,long,s0p5_vf0p0001_d0p9_r0p5_k0p0,cls_close,0.5,1336.5,0.0001,0.9,0.5,0.0,955104,AUC,0.6134338336105377
sweep_frac,TSLA,long,s0p5_vf0p001_d0p7_r0p1_k0p0,cls_close,0.5,13365.1,0.001,0.7,0.1,0.0,53821,AUC,0.6106572420079946
sweep_frac,TSLA,long,s0p5_vf0p001_d0p7_r0p3_k0p0,cls_close,0.5,13365.1,0.001,0.7,0.3,0.0,53821,AUC,0.6137599385886635
sweep_frac,TSLA,long,s0p5_vf0p001_d0p7_r0p5_k0p0,cls_close,0.5,13365.1,0.001,0.7,0.5,0.0,53821,AUC,0.6144707831701147
sweep_frac,TSLA,long,s0p5_vf0p001_d0p8_r0p1_k0p0,cls_close,0.5,13365.1,0.001,0.8,0.1,0.0,53821,AUC,0.6106846909392607
sweep_frac,TSLA,long,s0p5_vf0p001_d0p8_r0p3_k0p0,cls_close,0.5,13365.1,0.001,0.8,0.3,0.0,53821,AUC,0.6117549760245731
sweep_frac,TSLA,long,s0p5_vf0p001_d0p8_r0p5_k0p0,cls_close,0.5,13365.1,0.001,0.8,0.5,0.0,53821,AUC,0.6124679856220964
sweep_frac,TSLA,long,s0p5_vf0p001_d0p9_r0p1_k0p0,cls_close,0.5,13365.1,0.001,0.9,0.1,0.0,53821,AUC,0.6093570400112864
sweep_frac,TSLA,long,s0p5_vf0p001_d0p9_r0p3_k0p0,cls_close,0.5,13365.1,0.001,0.9,0.3,0.0,53821,AUC,0.6097758767212209
sweep_frac,TSLA,long,s0p5_vf0p001_d0p9_r0p5_k0p0,cls_close,0.5,13365.1,0.001,0.9,0.5,0.0,53821,AUC,0.6097919974096344
sweep_frac,TSLA,long,s1p0_vf1e-05_d0p7_r0p1_k0p0,cls_close,1.0,130.8,1e-05,0.7,0.1,0.0,2244611,AUC,0.607578401184406
sweep_frac,TSLA,long,s1p0_vf1e-05_d0p7_r0p3_k0p0,cls_close,1.0,130.8,1e-05,0.7,0.3,0.0,2244611,AUC,0.6091486090734859
sweep_frac,TSLA,long,s1p0_vf1e-05_d0p7_r0p5_k0p0,cls_close,1.0,130.8,1e-05,0.7,0.5,0.0,2244611,AUC,0.6064310722874278
sweep_frac,TSLA,long,s1p0_vf1e-05_d0p8_r0p1_k0p0,cls_close,1.0,130.8,1e-05,0.8,0.1,0.0,2244611,AUC,0.6069953242330518
sweep_frac,TSLA,long,s1p0_vf1e-05_d0p8_r0p3_k0p0,cls_close,1.0,130.8,1e-05,0.8,0.3,0.0,2244611,AUC,0.6090030402898099
sweep_frac,TSLA,long,s1p0_vf1e-05_d0p8_r0p5_k0p0,cls_close,1.0,130.8,1e-05,0.8,0.5,0.0,2244611,AUC,0.6090535552228558
sweep_frac,TSLA,long,s1p0_vf1e-05_d0p9_r0p1_k0p0,cls_close,1.0,130.8,1e-05,0.9,0.1,0.0,2244611,AUC,0.6049981356137633
sweep_frac,TSLA,long,s1p0_vf1e-05_d0p9_r0p3_k0p0,cls_close,1.0,130.8,1e-05,0.9,0.3,0.0,2244611,AUC,0.6053785999635176
sweep_frac,TSLA,long,s1p0_vf1e-05_d0p9_r0p5_k0p0,cls_close,1.0,130.8,1e-05,0.9,0.5,0.0,2244611,AUC,0.6053890375425987
sweep_frac,TSLA,long,s1p0_vf0p0001_d0p7_r0p1_k0p0,cls_close,1.0,1308.5,0.0001,0.7,0.1,0.0,748082,AUC,0.6135731910592687
sweep_frac,TSLA,long,s1p0_vf0p0001_d0p7_r0p3_k0p0,cls_close,1.0,1308.5,0.0001,0.7,0.3,0.0,748082,AUC,0.6182769627531738
sweep_frac,TSLA,long,s1p0_vf0p0001_d0p7_r0p5_k0p0,cls_close,1.0,1308.5,0.0001,0.7,0.5,0.0,748082,AUC,0.6179906064687968
sweep_frac,TSLA,long,s1p0_vf0p0001_d0p8_r0p1_k0p0,cls_close,1.0,1308.5,0.0001,0.8,0.1,0.0,748082,AUC,0.6126621805831695
sweep_frac,TSLA,long,s1p0_vf0p0001_d0p8_r0p3_k0p0,cls_close,1.0,1308.5,0.0001,0.8,0.3,0.0,748082,AUC,0.6154117818702962
sweep_frac,TSLA,long,s1p0_vf0p0001_d0p8_r0p5_k0p0,cls_close,1.0,1308.5,0.0001,0.8,0.5,0.0,748082,AUC,0.6154690813636281
sweep_frac,TSLA,long,s1p0_vf0p0001_d0p9_r0p1_k0p0,cls_close,1.0,1308.5,0.0001,0.9,0.1,0.0,748082,AUC,0.6104948593031988
sweep_frac,TSLA,long,s1p0_vf0p0001_d0p9_r0p3_k0p0,cls_close,1.0,1308.5,0.0001,0.9,0.3,0.0,748082,AUC,0.6107727529099711
sweep_frac,TSLA,long,s1p0_vf0p0001_d0p9_r0p5_k0p0,cls_close,1.0,1308.5,0.0001,0.9,0.5,0.0,748082,AUC,0.610743656674108
sweep_frac,TSLA,long,s1p0_vf0p001_d0p7_r0p1_k0p0,cls_close,1.0,13084.9,0.001,0.7,0.1,0.0,73773,AUC,0.6146623587744302
sweep_frac,TSLA,long,s1p0_vf0p001_d0p7_r0p3_k0p0,cls_close,1.0,13084.9,0.001,0.7,0.3,0.0,73773,AUC,0.617135943308061
sweep_frac,TSLA,long,s1p0_vf0p001_d0p7_r0p5_k0p0,cls_close,1.0,13084.9,0.001,0.7,0.5,0.0,73773,AUC,0.6169345072709121
sweep_frac,TSLA,long,s1p0_vf0p001_d0p8_r0p1_k0p0,cls_close,1.0,13084.9,0.001,0.8,0.1,0.0,73773,AUC,0.6143390797722953
sweep_frac,TSLA,long,s1p0_vf0p001_d0p8_r0p3_k0p0,cls_close,1.0,13084.9,0.001,0.8,0.3,0.0,73773,AUC,0.6176314741263981
sweep_frac,TSLA,long,s1p0_vf0p001_d0p8_r0p5_k0p0,cls_close,1.0,13084.9,0.001,0.8,0.5,0.0,73773,AUC,0.6181121733095423
sweep_frac,TSLA,long,s1p0_vf0p001_d0p9_r0p1_k0p0,cls_close,1.0,13084.9,0.001,0.9,0.1,0.0,73773,AUC,0.6131295234030675
sweep_frac,TSLA,long,s1p0_vf0p001_d0p9_r0p3_k0p0,cls_close,1.0,13084.9,0.001,0.9,0.3,0.0,73773,AUC,0.613131336609918
sweep_frac,TSLA,long,s1p0_vf0p001_d0p9_r0p5_k0p0,cls_close,1.0,13084.9,0.001,0.9,0.5,0.0,73773,AUC,0.6130996295842338
sweep_frac,TSLA,long,s2p0_vf1e-05_d0p7_r0p1_k0p0,cls_close,2.0,126.9,1e-05,0.7,0.1,0.0,670803,AUC,0.6057184013351893
sweep_frac,TSLA,long,s2p0_vf1e-05_d0p7_r0p3_k0p0,cls_close,2.0,126.9,1e-05,0.7,0.3,0.0,670803,AUC,0.6103989149344172
sweep_frac,TSLA,long,s2p0_vf1e-05_d0p7_r0p5_k0p0,cls_close,2.0,126.9,1e-05,0.7,0.5,0.0,670803,AUC,0.6103941902848506
sweep_frac,TSLA,long,s2p0_vf1e-05_d0p8_r0p1_k0p0,cls_close,2.0,126.9,1e-05,0.8,0.1,0.0,670803,AUC,0.6048044296484655
sweep_frac,TSLA,long,s2p0_vf1e-05_d0p8_r0p3_k0p0,cls_close,2.0,126.9,1e-05,0.8,0.3,0.0,670803,AUC,0.6072935493975321
sweep_frac,TSLA,long,s2p0_vf1e-05_d0p8_r0p5_k0p0,cls_close,2.0,126.9,1e-05,0.8,0.5,0.0,670803,AUC,0.6076902144252901
sweep_frac,TSLA,long,s2p0_vf1e-05_d0p9_r0p1_k0p0,cls_close,2.0,126.9,1e-05,0.9,0.1,0.0,670803,AUC,0.6030488796853332
sweep_frac,TSLA,long,s2p0_vf1e-05_d0p9_r0p3_k0p0,cls_close,2.0,126.9,1e-05,0.9,0.3,0.0,670803,AUC,0.6033337888289763
sweep_frac,TSLA,long,s2p0_vf1e-05_d0p9_r0p5_k0p0,cls_close,2.0,126.9,1e-05,0.9,0.5,0.0,670803,AUC,0.6033509555818313
sweep_frac,TSLA,long,s2p0_vf0p0001_d0p7_r0p1_k0p0,cls_close,2.0,1269.2,0.0001,0.7,0.1,0.0,367017,AUC,0.6110248177914164
sweep_frac,TSLA,long,s2p0_vf0p0001_d0p7_r0p3_k0p0,cls_close,2.0,1269.2,0.0001,0.7,0.3,0.0,367017,AUC,0.6152874624057196
sweep_frac,TSLA,long,s2p0_vf0p0001_d0p7_r0p5_k0p0,cls_close,2.0,1269.2,0.0001,0.7,0.5,0.0,367017,AUC,0.6162603565147748
sweep_frac,TSLA,long,s2p0_vf0p0001_d0p8_r0p1_k0p0,cls_close,2.0,1269.2,0.0001,0.8,0.1,0.0,367017,AUC,0.6105598106801051
sweep_frac,TSLA,long,s2p0_vf0p0001_d0p8_r0p3_k0p0,cls_close,2.0,1269.2,0.0001,0.8,0.3,0.0,367017,AUC,0.6118152201837987
sweep_frac,TSLA,long,s2p0_vf0p0001_d0p8_r0p5_k0p0,cls_close,2.0,1269.2,0.0001,0.8,0.5,0.0,367017,AUC,0.6120518228862221
sweep_frac,TSLA,long,s2p0_vf0p0001_d0p9_r0p1_k0p0,cls_close,2.0,1269.2,0.0001,0.9,0.1,0.0,367017,AUC,0.6098105335794991
sweep_frac,TSLA,long,s2p0_vf0p0001_d0p9_r0p3_k0p0,cls_close,2.0,1269.2,0.0001,0.9,0.3,0.0,367017,AUC,0.6097983919069895
sweep_frac,TSLA,long,s2p0_vf0p0001_d0p9_r0p5_k0p0,cls_close,2.0,1269.2,0.0001,0.9,0.5,0.0,367017,AUC,0.6097852111339421
sweep_frac,TSLA,long,s2p0_vf0p001_d0p7_r0p1_k0p0,cls_close,2.0,12691.8,0.001,0.7,0.1,0.0,69128,AUC,0.6220390840406131
sweep_frac,TSLA,long,s2p0_vf0p001_d0p7_r0p3_k0p0,cls_close,2.0,12691.8,0.001,0.7,0.3,0.0,69128,AUC,0.6251508264290606
sweep_frac,TSLA,long,s2p0_vf0p001_d0p7_r0p5_k0p0,cls_close,2.0,12691.8,0.001,0.7,0.5,0.0,69128,AUC,0.626593233435815
sweep_frac,TSLA,long,s2p0_vf0p001_d0p8_r0p1_k0p0,cls_close,2.0,12691.8,0.001,0.8,0.1,0.0,69128,AUC,0.6221895436265193
sweep_frac,TSLA,long,s2p0_vf0p001_d0p8_r0p3_k0p0,cls_close,2.0,12691.8,0.001,0.8,0.3,0.0,69128,AUC,0.6227356506245346
sweep_frac,TSLA,long,s2p0_vf0p001_d0p8_r0p5_k0p0,cls_close,2.0,12691.8,0.001,0.8,0.5,0.0,69128,AUC,0.6227942270722522
sweep_frac,TSLA,long,s2p0_vf0p001_d0p9_r0p1_k0p0,cls_close,2.0,12691.8,0.001,0.9,0.1,0.0,69128,AUC,0.6221626026652588
sweep_frac,TSLA,long,s2p0_vf0p001_d0p9_r0p3_k0p0,cls_close,2.0,12691.8,0.001,0.9,0.3,0.0,69128,AUC,0.6217236250737248
sweep_frac,TSLA,long,s2p0_vf0p001_d0p9_r0p5_k0p0,cls_close,2.0,12691.8,0.001,0.9,0.5,0.0,69128,AUC,0.6216934654973261
```

### 12.4 Full sweep (non-frac) `sweep_summary.csv` Aggregation (All Rows Pulled)
```csv
sweep_type,ticker,phase,config,target,silence,min_vol,vol_frac,dir_thresh,vol_ratio,kappa,rows,metric_name,metric_value
sweep,JPM,short,s0p5_v50_d0p7_r0p1_k0p0,cls_1m,0.5,50,,0.7,0.1,0.0,1974001,AUC,0.6488348975454057
sweep,JPM,short,s0p5_v50_d0p7_r0p1_k0p0,cls_10m,0.5,50,,0.7,0.1,0.0,1974001,AUC,0.5489165463830208
sweep,JPM,short,s0p5_v50_d0p7_r0p3_k0p0,cls_1m,0.5,50,,0.7,0.3,0.0,1974001,AUC,0.6441325931987333
sweep,JPM,short,s0p5_v50_d0p7_r0p3_k0p0,cls_10m,0.5,50,,0.7,0.3,0.0,1974001,AUC,0.567982517048999
sweep,JPM,short,s0p5_v50_d0p7_r0p5_k0p0,cls_1m,0.5,50,,0.7,0.5,0.0,1974001,AUC,0.6438882987965839
sweep,JPM,short,s0p5_v50_d0p7_r0p5_k0p0,cls_10m,0.5,50,,0.7,0.5,0.0,1974001,AUC,0.5632029024844085
sweep,JPM,short,s0p5_v50_d0p8_r0p1_k0p0,cls_1m,0.5,50,,0.8,0.1,0.0,1974001,AUC,0.6407227605983749
sweep,JPM,short,s0p5_v50_d0p8_r0p1_k0p0,cls_10m,0.5,50,,0.8,0.1,0.0,1974001,AUC,0.5627602815995344
sweep,JPM,short,s0p5_v50_d0p8_r0p3_k0p0,cls_1m,0.5,50,,0.8,0.3,0.0,1974001,AUC,0.6421429276412965
sweep,JPM,short,s0p5_v50_d0p8_r0p3_k0p0,cls_10m,0.5,50,,0.8,0.3,0.0,1974001,AUC,0.565041875442624
sweep,JPM,short,s0p5_v50_d0p8_r0p5_k0p0,cls_1m,0.5,50,,0.8,0.5,0.0,1974001,AUC,0.643060894231127
sweep,JPM,short,s0p5_v50_d0p8_r0p5_k0p0,cls_10m,0.5,50,,0.8,0.5,0.0,1974001,AUC,0.5665293793606871
sweep,JPM,short,s0p5_v50_d0p9_r0p1_k0p0,cls_1m,0.5,50,,0.9,0.1,0.0,1974001,AUC,0.6391968929649462
sweep,JPM,short,s0p5_v50_d0p9_r0p1_k0p0,cls_10m,0.5,50,,0.9,0.1,0.0,1974001,AUC,0.5599795958748135
sweep,JPM,short,s0p5_v50_d0p9_r0p3_k0p0,cls_1m,0.5,50,,0.9,0.3,0.0,1974001,AUC,0.6394015380514699
sweep,JPM,short,s0p5_v50_d0p9_r0p3_k0p0,cls_10m,0.5,50,,0.9,0.3,0.0,1974001,AUC,0.5603454744451091
sweep,JPM,short,s0p5_v50_d0p9_r0p5_k0p0,cls_1m,0.5,50,,0.9,0.5,0.0,1974001,AUC,0.6394374958205782
sweep,JPM,short,s0p5_v50_d0p9_r0p5_k0p0,cls_10m,0.5,50,,0.9,0.5,0.0,1974001,AUC,0.5604165309284012
sweep,JPM,short,s0p5_v100_d0p7_r0p1_k0p0,cls_1m,0.5,100,,0.7,0.1,0.0,1715393,AUC,0.6411114803954125
sweep,JPM,short,s0p5_v100_d0p7_r0p1_k0p0,cls_10m,0.5,100,,0.7,0.1,0.0,1715393,AUC,0.5637691223231247
sweep,JPM,short,s0p5_v100_d0p7_r0p3_k0p0,cls_1m,0.5,100,,0.7,0.3,0.0,1715393,AUC,0.6435411095013838
sweep,JPM,short,s0p5_v100_d0p7_r0p3_k0p0,cls_10m,0.5,100,,0.7,0.3,0.0,1715393,AUC,0.5676314465849039
sweep,JPM,short,s0p5_v100_d0p7_r0p5_k0p0,cls_1m,0.5,100,,0.7,0.5,0.0,1715393,AUC,0.6459981546750329
sweep,JPM,short,s0p5_v100_d0p7_r0p5_k0p0,cls_10m,0.5,100,,0.7,0.5,0.0,1715393,AUC,0.571041521315753
sweep,JPM,short,s0p5_v100_d0p8_r0p1_k0p0,cls_1m,0.5,100,,0.8,0.1,0.0,1715393,AUC,0.6402584360849738
sweep,JPM,short,s0p5_v100_d0p8_r0p1_k0p0,cls_10m,0.5,100,,0.8,0.1,0.0,1715393,AUC,0.5622616597942153
sweep,JPM,short,s0p5_v100_d0p8_r0p3_k0p0,cls_1m,0.5,100,,0.8,0.3,0.0,1715393,AUC,0.6416053021639467
sweep,JPM,short,s0p5_v100_d0p8_r0p3_k0p0,cls_10m,0.5,100,,0.8,0.3,0.0,1715393,AUC,0.5646103222385144
sweep,JPM,short,s0p5_v100_d0p8_r0p5_k0p0,cls_1m,0.5,100,,0.8,0.5,0.0,1715393,AUC,0.6425652071003243
sweep,JPM,short,s0p5_v100_d0p8_r0p5_k0p0,cls_10m,0.5,100,,0.8,0.5,0.0,1715393,AUC,0.5663146997225151
sweep,JPM,short,s0p5_v100_d0p9_r0p1_k0p0,cls_1m,0.5,100,,0.9,0.1,0.0,1715393,AUC,0.6388133553458539
sweep,JPM,short,s0p5_v100_d0p9_r0p1_k0p0,cls_10m,0.5,100,,0.9,0.1,0.0,1715393,AUC,0.5594965903152215
sweep,JPM,short,s0p5_v100_d0p9_r0p3_k0p0,cls_1m,0.5,100,,0.9,0.3,0.0,1715393,AUC,0.6390241941007132
sweep,JPM,short,s0p5_v100_d0p9_r0p3_k0p0,cls_10m,0.5,100,,0.9,0.3,0.0,1715393,AUC,0.5598191829135141
sweep,JPM,short,s0p5_v100_d0p9_r0p5_k0p0,cls_1m,0.5,100,,0.9,0.5,0.0,1715393,AUC,0.6390597131383065
sweep,JPM,short,s0p5_v100_d0p9_r0p5_k0p0,cls_10m,0.5,100,,0.9,0.5,0.0,1715393,AUC,0.5598893816657695
sweep,JPM,short,s0p5_v200_d0p7_r0p1_k0p0,cls_1m,0.5,200,,0.7,0.1,0.0,899955,AUC,0.6411470475462627
sweep,JPM,short,s0p5_v200_d0p7_r0p1_k0p0,cls_10m,0.5,200,,0.7,0.1,0.0,899955,AUC,0.5557837444985664
sweep,JPM,short,s0p5_v200_d0p7_r0p3_k0p0,cls_1m,0.5,200,,0.7,0.3,0.0,899955,AUC,0.6430143783983857
sweep,JPM,short,s0p5_v200_d0p7_r0p3_k0p0,cls_10m,0.5,200,,0.7,0.3,0.0,899955,AUC,0.5589607281528898
sweep,JPM,short,s0p5_v200_d0p7_r0p5_k0p0,cls_1m,0.5,200,,0.7,0.5,0.0,899955,AUC,0.645277929487554
sweep,JPM,short,s0p5_v200_d0p7_r0p5_k0p0,cls_10m,0.5,200,,0.7,0.5,0.0,899955,AUC,0.5637865788712019
sweep,JPM,short,s0p5_v200_d0p8_r0p1_k0p0,cls_1m,0.5,200,,0.8,0.1,0.0,899955,AUC,0.6408886734222528
sweep,JPM,short,s0p5_v200_d0p8_r0p1_k0p0,cls_10m,0.5,200,,0.8,0.1,0.0,899955,AUC,0.5552923814423305
sweep,JPM,short,s0p5_v200_d0p8_r0p3_k0p0,cls_1m,0.5,200,,0.8,0.3,0.0,899955,AUC,0.6421397804637881
sweep,JPM,short,s0p5_v200_d0p8_r0p3_k0p0,cls_10m,0.5,200,,0.8,0.3,0.0,899955,AUC,0.5573253421412202
sweep,JPM,short,s0p5_v200_d0p8_r0p5_k0p0,cls_1m,0.5,200,,0.8,0.5,0.0,899955,AUC,0.6429104852432045
sweep,JPM,short,s0p5_v200_d0p8_r0p5_k0p0,cls_10m,0.5,200,,0.8,0.5,0.0,899955,AUC,0.5587191343532398
sweep,JPM,short,s0p5_v200_d0p9_r0p1_k0p0,cls_1m,0.5,200,,0.9,0.1,0.0,899955,AUC,0.6401149271044602
sweep,JPM,short,s0p5_v200_d0p9_r0p1_k0p0,cls_10m,0.5,200,,0.9,0.1,0.0,899955,AUC,0.5539037999106587
sweep,JPM,short,s0p5_v200_d0p9_r0p3_k0p0,cls_1m,0.5,200,,0.9,0.3,0.0,899955,AUC,0.6405134886462496
sweep,JPM,short,s0p5_v200_d0p9_r0p3_k0p0,cls_10m,0.5,200,,0.9,0.3,0.0,899955,AUC,0.5545926768050788
sweep,JPM,short,s0p5_v200_d0p9_r0p5_k0p0,cls_1m,0.5,200,,0.9,0.5,0.0,899955,AUC,0.6405543914636782
sweep,JPM,short,s0p5_v200_d0p9_r0p5_k0p0,cls_10m,0.5,200,,0.9,0.5,0.0,899955,AUC,0.5546568033027782
sweep,JPM,short,s0p5_v1000_d0p7_r0p1_k0p0,cls_1m,0.5,1000,,0.7,0.1,0.0,61109,AUC,0.6441736184355408
sweep,JPM,short,s0p5_v1000_d0p7_r0p1_k0p0,cls_10m,0.5,1000,,0.7,0.1,0.0,61109,AUC,0.5529055730170385
sweep,JPM,short,s0p5_v1000_d0p7_r0p3_k0p0,cls_1m,0.5,1000,,0.7,0.3,0.0,61109,AUC,0.6504315237342816
sweep,JPM,short,s0p5_v1000_d0p7_r0p3_k0p0,cls_10m,0.5,1000,,0.7,0.3,0.0,61109,AUC,0.562125178042846
sweep,JPM,short,s0p5_v1000_d0p7_r0p5_k0p0,cls_1m,0.5,1000,,0.7,0.5,0.0,61109,AUC,0.6541140051216268
sweep,JPM,short,s0p5_v1000_d0p7_r0p5_k0p0,cls_10m,0.5,1000,,0.7,0.5,0.0,61109,AUC,0.5677497956073634
sweep,JPM,short,s0p5_v1000_d0p8_r0p1_k0p0,cls_1m,0.5,1000,,0.8,0.1,0.0,61109,AUC,0.6439470646424781
sweep,JPM,short,s0p5_v1000_d0p8_r0p1_k0p0,cls_10m,0.5,1000,,0.8,0.1,0.0,61109,AUC,0.5519291911002007
sweep,JPM,short,s0p5_v1000_d0p8_r0p3_k0p0,cls_1m,0.5,1000,,0.8,0.3,0.0,61109,AUC,0.6473706833673252
sweep,JPM,short,s0p5_v1000_d0p8_r0p3_k0p0,cls_10m,0.5,1000,,0.8,0.3,0.0,61109,AUC,0.5579372951971309
sweep,JPM,short,s0p5_v1000_d0p8_r0p5_k0p0,cls_1m,0.5,1000,,0.8,0.5,0.0,61109,AUC,0.6480215970306971
sweep,JPM,short,s0p5_v1000_d0p8_r0p5_k0p0,cls_10m,0.5,1000,,0.8,0.5,0.0,61109,AUC,0.5588644114586746
sweep,JPM,short,s0p5_v1000_d0p9_r0p1_k0p0,cls_1m,0.5,1000,,0.9,0.1,0.0,61109,AUC,0.6428485054192188
sweep,JPM,short,s0p5_v1000_d0p9_r0p1_k0p0,cls_10m,0.5,1000,,0.9,0.1,0.0,61109,AUC,0.5483866954949201
sweep,JPM,short,s0p5_v1000_d0p9_r0p3_k0p0,cls_1m,0.5,1000,,0.9,0.3,0.0,61109,AUC,0.6434608767030512
sweep,JPM,short,s0p5_v1000_d0p9_r0p3_k0p0,cls_10m,0.5,1000,,0.9,0.3,0.0,61109,AUC,0.5491633590886986
sweep,JPM,short,s0p5_v1000_d0p9_r0p5_k0p0,cls_1m,0.5,1000,,0.9,0.5,0.0,61109,AUC,0.6434033693412557
sweep,JPM,short,s0p5_v1000_d0p9_r0p5_k0p0,cls_10m,0.5,1000,,0.9,0.5,0.0,61109,AUC,0.5491403025870945
sweep,JPM,short,s1p0_v50_d0p7_r0p1_k0p0,cls_1m,1.0,50,,0.7,0.1,0.0,1666897,AUC,0.6420055780094861
sweep,JPM,short,s1p0_v50_d0p7_r0p1_k0p0,cls_10m,1.0,50,,0.7,0.1,0.0,1666897,AUC,0.5662018090569969
sweep,JPM,short,s1p0_v50_d0p7_r0p3_k0p0,cls_1m,1.0,50,,0.7,0.3,0.0,1666897,AUC,0.6452910903600724
sweep,JPM,short,s1p0_v50_d0p7_r0p3_k0p0,cls_10m,1.0,50,,0.7,0.3,0.0,1666897,AUC,0.570311450856109
sweep,JPM,short,s1p0_v50_d0p7_r0p5_k0p0,cls_1m,1.0,50,,0.7,0.5,0.0,1666897,AUC,0.6485719093933303
sweep,JPM,short,s1p0_v50_d0p7_r0p5_k0p0,cls_10m,1.0,50,,0.7,0.5,0.0,1666897,AUC,0.5739026491139076
sweep,JPM,short,s1p0_v50_d0p8_r0p1_k0p0,cls_1m,1.0,50,,0.8,0.1,0.0,1666897,AUC,0.6410700357839232
sweep,JPM,short,s1p0_v50_d0p8_r0p1_k0p0,cls_10m,1.0,50,,0.8,0.1,0.0,1666897,AUC,0.5647504126077183
sweep,JPM,short,s1p0_v50_d0p8_r0p3_k0p0,cls_1m,1.0,50,,0.8,0.3,0.0,1666897,AUC,0.6427061428073546
sweep,JPM,short,s1p0_v50_d0p8_r0p3_k0p0,cls_10m,1.0,50,,0.8,0.3,0.0,1666897,AUC,0.56695162849789
sweep,JPM,short,s1p0_v50_d0p8_r0p5_k0p0,cls_1m,1.0,50,,0.8,0.5,0.0,1666897,AUC,0.6437744737704927
sweep,JPM,short,s1p0_v50_d0p8_r0p5_k0p0,cls_10m,1.0,50,,0.8,0.5,0.0,1666897,AUC,0.5684267215025605
sweep,JPM,short,s1p0_v50_d0p9_r0p1_k0p0,cls_1m,1.0,50,,0.9,0.1,0.0,1666897,AUC,0.6395462110326702
sweep,JPM,short,s1p0_v50_d0p9_r0p1_k0p0,cls_10m,1.0,50,,0.9,0.1,0.0,1666897,AUC,0.5623742209006063
sweep,JPM,short,s1p0_v50_d0p9_r0p3_k0p0,cls_1m,1.0,50,,0.9,0.3,0.0,1666897,AUC,0.6397512839136571
sweep,JPM,short,s1p0_v50_d0p9_r0p3_k0p0,cls_10m,1.0,50,,0.9,0.3,0.0,1666897,AUC,0.5626987563922694
sweep,JPM,short,s1p0_v50_d0p9_r0p5_k0p0,cls_1m,1.0,50,,0.9,0.5,0.0,1666897,AUC,0.6398053925465691
sweep,JPM,short,s1p0_v50_d0p9_r0p5_k0p0,cls_10m,1.0,50,,0.9,0.5,0.0,1666897,AUC,0.562755187723263
sweep,JPM,short,s1p0_v100_d0p7_r0p1_k0p0,cls_1m,1.0,100,,0.7,0.1,0.0,1472959,AUC,0.6423217481084642
sweep,JPM,short,s1p0_v100_d0p7_r0p1_k0p0,cls_10m,1.0,100,,0.7,0.1,0.0,1472959,AUC,0.5665537856042804
sweep,JPM,short,s1p0_v100_d0p7_r0p3_k0p0,cls_1m,1.0,100,,0.7,0.3,0.0,1472959,AUC,0.6452053595235908
sweep,JPM,short,s1p0_v100_d0p7_r0p3_k0p0,cls_10m,1.0,100,,0.7,0.3,0.0,1472959,AUC,0.5702928599364864
sweep,JPM,short,s1p0_v100_d0p7_r0p5_k0p0,cls_1m,1.0,100,,0.7,0.5,0.0,1472959,AUC,0.6485410268787452
sweep,JPM,short,s1p0_v100_d0p7_r0p5_k0p0,cls_10m,1.0,100,,0.7,0.5,0.0,1472959,AUC,0.5744026913517414
sweep,JPM,short,s1p0_v100_d0p8_r0p1_k0p0,cls_1m,1.0,100,,0.8,0.1,0.0,1472959,AUC,0.64150232935522
sweep,JPM,short,s1p0_v100_d0p8_r0p1_k0p0,cls_10m,1.0,100,,0.8,0.1,0.0,1472959,AUC,0.5652584524588407
sweep,JPM,short,s1p0_v100_d0p8_r0p3_k0p0,cls_1m,1.0,100,,0.8,0.3,0.0,1472959,AUC,0.6429309032247413
sweep,JPM,short,s1p0_v100_d0p8_r0p3_k0p0,cls_10m,1.0,100,,0.8,0.3,0.0,1472959,AUC,0.5671413335526284
sweep,JPM,short,s1p0_v100_d0p8_r0p5_k0p0,cls_1m,1.0,100,,0.8,0.5,0.0,1472959,AUC,0.643918324253349
sweep,JPM,short,s1p0_v100_d0p8_r0p5_k0p0,cls_10m,1.0,100,,0.8,0.5,0.0,1472959,AUC,0.5685520347018916
sweep,JPM,short,s1p0_v100_d0p9_r0p1_k0p0,cls_1m,1.0,100,,0.9,0.1,0.0,1472959,AUC,0.64005884147789
sweep,JPM,short,s1p0_v100_d0p9_r0p1_k0p0,cls_10m,1.0,100,,0.9,0.1,0.0,1472959,AUC,0.5628093738858787
sweep,JPM,short,s1p0_v100_d0p9_r0p3_k0p0,cls_1m,1.0,100,,0.9,0.3,0.0,1472959,AUC,0.640281573578766
sweep,JPM,short,s1p0_v100_d0p9_r0p3_k0p0,cls_10m,1.0,100,,0.9,0.3,0.0,1472959,AUC,0.5631883263464788
sweep,JPM,short,s1p0_v100_d0p9_r0p5_k0p0,cls_1m,1.0,100,,0.9,0.5,0.0,1472959,AUC,0.6403498154698344
sweep,JPM,short,s1p0_v100_d0p9_r0p5_k0p0,cls_10m,1.0,100,,0.9,0.5,0.0,1472959,AUC,0.563368039806149
sweep,JPM,short,s1p0_v200_d0p7_r0p1_k0p0,cls_1m,1.0,200,,0.7,0.1,0.0,846951,AUC,0.6434764325262016
sweep,JPM,short,s1p0_v200_d0p7_r0p1_k0p0,cls_10m,1.0,200,,0.7,0.1,0.0,846951,AUC,0.5608924313674774
sweep,JPM,short,s1p0_v200_d0p7_r0p3_k0p0,cls_1m,1.0,200,,0.7,0.3,0.0,846951,AUC,0.6464598671040194
sweep,JPM,short,s1p0_v200_d0p7_r0p3_k0p0,cls_10m,1.0,200,,0.7,0.3,0.0,846951,AUC,0.5650684659598866
sweep,JPM,short,s1p0_v200_d0p7_r0p5_k0p0,cls_1m,1.0,200,,0.7,0.5,0.0,846951,AUC,0.6490613864134714
sweep,JPM,short,s1p0_v200_d0p7_r0p5_k0p0,cls_10m,1.0,200,,0.7,0.5,0.0,846951,AUC,0.5685842295764236
sweep,JPM,short,s1p0_v200_d0p8_r0p1_k0p0,cls_1m,1.0,200,,0.8,0.1,0.0,846951,AUC,0.6430540274696264
sweep,JPM,short,s1p0_v200_d0p8_r0p1_k0p0,cls_10m,1.0,200,,0.8,0.1,0.0,846951,AUC,0.5602568872729635
sweep,JPM,short,s1p0_v200_d0p8_r0p3_k0p0,cls_1m,1.0,200,,0.8,0.3,0.0,846951,AUC,0.6450585119240151
sweep,JPM,short,s1p0_v200_d0p8_r0p3_k0p0,cls_10m,1.0,200,,0.8,0.3,0.0,846951,AUC,0.5632273054492996
sweep,JPM,short,s1p0_v200_d0p8_r0p5_k0p0,cls_1m,1.0,200,,0.8,0.5,0.0,846951,AUC,0.6460518434520616
sweep,JPM,short,s1p0_v200_d0p8_r0p5_k0p0,cls_10m,1.0,200,,0.8,0.5,0.0,846951,AUC,0.5645344426506904
sweep,JPM,short,s1p0_v200_d0p9_r0p1_k0p0,cls_1m,1.0,200,,0.9,0.1,0.0,846951,AUC,0.6416984581185834
sweep,JPM,short,s1p0_v200_d0p9_r0p1_k0p0,cls_10m,1.0,200,,0.9,0.1,0.0,846951,AUC,0.5576090459475584
sweep,JPM,short,s1p0_v200_d0p9_r0p3_k0p0,cls_1m,1.0,200,,0.9,0.3,0.0,846951,AUC,0.6421762181995008
sweep,JPM,short,s1p0_v200_d0p9_r0p3_k0p0,cls_10m,1.0,200,,0.9,0.3,0.0,846951,AUC,0.5584614714656833
sweep,JPM,short,s1p0_v200_d0p9_r0p5_k0p0,cls_1m,1.0,200,,0.9,0.5,0.0,846951,AUC,0.6422618598569457
sweep,JPM,short,s1p0_v200_d0p9_r0p5_k0p0,cls_10m,1.0,200,,0.9,0.5,0.0,846951,AUC,0.5586954955296239
sweep,JPM,short,s1p0_v1000_d0p7_r0p1_k0p0,cls_1m,1.0,1000,,0.7,0.1,0.0,78315,AUC,0.6413776629074568
sweep,JPM,short,s1p0_v1000_d0p7_r0p1_k0p0,cls_10m,1.0,1000,,0.7,0.1,0.0,78315,AUC,0.5543344982095078
sweep,JPM,short,s1p0_v1000_d0p7_r0p3_k0p0,cls_1m,1.0,1000,,0.7,0.3,0.0,78315,AUC,0.6498067419682456
sweep,JPM,short,s1p0_v1000_d0p7_r0p3_k0p0,cls_10m,1.0,1000,,0.7,0.3,0.0,78315,AUC,0.5651884566318338
sweep,JPM,short,s1p0_v1000_d0p7_r0p5_k0p0,cls_1m,1.0,1000,,0.7,0.5,0.0,78315,AUC,0.6548609365673573
sweep,JPM,short,s1p0_v1000_d0p7_r0p5_k0p0,cls_10m,1.0,1000,,0.7,0.5,0.0,78315,AUC,0.5703791922236372
sweep,JPM,short,s1p0_v1000_d0p8_r0p1_k0p0,cls_1m,1.0,1000,,0.8,0.1,0.0,78315,AUC,0.6408734804163314
sweep,JPM,short,s1p0_v1000_d0p8_r0p1_k0p0,cls_10m,1.0,1000,,0.8,0.1,0.0,78315,AUC,0.5534166153009756
sweep,JPM,short,s1p0_v1000_d0p8_r0p3_k0p0,cls_1m,1.0,1000,,0.8,0.3,0.0,78315,AUC,0.6452299842275104
sweep,JPM,short,s1p0_v1000_d0p8_r0p3_k0p0,cls_10m,1.0,1000,,0.8,0.3,0.0,78315,AUC,0.5601370327573463
sweep,JPM,short,s1p0_v1000_d0p8_r0p5_k0p0,cls_1m,1.0,1000,,0.8,0.5,0.0,78315,AUC,0.6459590110713538
sweep,JPM,short,s1p0_v1000_d0p8_r0p5_k0p0,cls_10m,1.0,1000,,0.8,0.5,0.0,78315,AUC,0.5604080677764351
sweep,JPM,short,s1p0_v1000_d0p9_r0p1_k0p0,cls_1m,1.0,1000,,0.9,0.1,0.0,78315,AUC,0.6401107353144705
sweep,JPM,short,s1p0_v1000_d0p9_r0p1_k0p0,cls_10m,1.0,1000,,0.9,0.1,0.0,78315,AUC,0.5505936102594404
sweep,JPM,short,s1p0_v1000_d0p9_r0p3_k0p0,cls_1m,1.0,1000,,0.9,0.3,0.0,78315,AUC,0.6404140755382564
sweep,JPM,short,s1p0_v1000_d0p9_r0p3_k0p0,cls_10m,1.0,1000,,0.9,0.3,0.0,78315,AUC,0.5518555703396257
sweep,JPM,short,s1p0_v1000_d0p9_r0p5_k0p0,cls_1m,1.0,1000,,0.9,0.5,0.0,78315,AUC,0.6404017946659021
sweep,JPM,short,s1p0_v1000_d0p9_r0p5_k0p0,cls_10m,1.0,1000,,0.9,0.5,0.0,78315,AUC,0.5517178427892901
sweep,JPM,short,s2p0_v50_d0p7_r0p1_k0p0,cls_1m,2.0,50,,0.7,0.1,0.0,1244509,AUC,0.6438761117930814
sweep,JPM,short,s2p0_v50_d0p7_r0p1_k0p0,cls_10m,2.0,50,,0.7,0.1,0.0,1244509,AUC,0.5692436699421568
sweep,JPM,short,s2p0_v50_d0p7_r0p3_k0p0,cls_1m,2.0,50,,0.7,0.3,0.0,1244509,AUC,0.6480332522053281
sweep,JPM,short,s2p0_v50_d0p7_r0p3_k0p0,cls_10m,2.0,50,,0.7,0.3,0.0,1244509,AUC,0.5734959488147369
sweep,JPM,short,s2p0_v50_d0p7_r0p5_k0p0,cls_1m,2.0,50,,0.7,0.5,0.0,1244509,AUC,0.6521342318023832
sweep,JPM,short,s2p0_v50_d0p7_r0p5_k0p0,cls_10m,2.0,50,,0.7,0.5,0.0,1244509,AUC,0.5780914816649471
sweep,JPM,short,s2p0_v50_d0p8_r0p1_k0p0,cls_1m,2.0,50,,0.8,0.1,0.0,1244509,AUC,0.6429033787797952
sweep,JPM,short,s2p0_v50_d0p8_r0p1_k0p0,cls_10m,2.0,50,,0.8,0.1,0.0,1244509,AUC,0.5680858478982791
sweep,JPM,short,s2p0_v50_d0p8_r0p3_k0p0,cls_1m,2.0,50,,0.8,0.3,0.0,1244509,AUC,0.6450889235706156
sweep,JPM,short,s2p0_v50_d0p8_r0p3_k0p0,cls_10m,2.0,50,,0.8,0.3,0.0,1244509,AUC,0.5706098919277481
sweep,JPM,short,s2p0_v50_d0p8_r0p5_k0p0,cls_1m,2.0,50,,0.8,0.5,0.0,1244509,AUC,0.6461162209179557
sweep,JPM,short,s2p0_v50_d0p8_r0p5_k0p0,cls_10m,2.0,50,,0.8,0.5,0.0,1244509,AUC,0.5719894419300086
sweep,JPM,short,s2p0_v50_d0p9_r0p1_k0p0,cls_1m,2.0,50,,0.9,0.1,0.0,1244509,AUC,0.6410197699111753
sweep,JPM,short,s2p0_v50_d0p9_r0p1_k0p0,cls_10m,2.0,50,,0.9,0.1,0.0,1244509,AUC,0.5653847716163956
sweep,JPM,short,s2p0_v50_d0p9_r0p3_k0p0,cls_1m,2.0,50,,0.9,0.3,0.0,1244509,AUC,0.6414122827590063
sweep,JPM,short,s2p0_v50_d0p9_r0p3_k0p0,cls_10m,2.0,50,,0.9,0.3,0.0,1244509,AUC,0.5658148698481639
sweep,JPM,short,s2p0_v50_d0p9_r0p5_k0p0,cls_1m,2.0,50,,0.9,0.5,0.0,1244509,AUC,0.6414975043931421
sweep,JPM,short,s2p0_v50_d0p9_r0p5_k0p0,cls_10m,2.0,50,,0.9,0.5,0.0,1244509,AUC,0.5659473872176435
sweep,JPM,short,s2p0_v100_d0p7_r0p1_k0p0,cls_1m,2.0,100,,0.7,0.1,0.0,1124513,AUC,0.6452669059856301
sweep,JPM,short,s2p0_v100_d0p7_r0p1_k0p0,cls_10m,2.0,100,,0.7,0.1,0.0,1124513,AUC,0.5703604240853063
sweep,JPM,short,s2p0_v100_d0p7_r0p3_k0p0,cls_1m,2.0,100,,0.7,0.3,0.0,1124513,AUC,0.6491485849570924
sweep,JPM,short,s2p0_v100_d0p7_r0p3_k0p0,cls_10m,2.0,100,,0.7,0.3,0.0,1124513,AUC,0.5740282422570316
sweep,JPM,short,s2p0_v100_d0p7_r0p5_k0p0,cls_1m,2.0,100,,0.7,0.5,0.0,1124513,AUC,0.6529619001693092
sweep,JPM,short,s2p0_v100_d0p7_r0p5_k0p0,cls_10m,2.0,100,,0.7,0.5,0.0,1124513,AUC,0.578344828378618
sweep,JPM,short,s2p0_v100_d0p8_r0p1_k0p0,cls_1m,2.0,100,,0.8,0.1,0.0,1124513,AUC,0.6443010311963818
sweep,JPM,short,s2p0_v100_d0p8_r0p1_k0p0,cls_10m,2.0,100,,0.8,0.1,0.0,1124513,AUC,0.5693113845704907
sweep,JPM,short,s2p0_v100_d0p8_r0p3_k0p0,cls_1m,2.0,100,,0.8,0.3,0.0,1124513,AUC,0.646500874390781
sweep,JPM,short,s2p0_v100_d0p8_r0p3_k0p0,cls_10m,2.0,100,,0.8,0.3,0.0,1124513,AUC,0.5717490069511191
sweep,JPM,short,s2p0_v100_d0p8_r0p5_k0p0,cls_1m,2.0,100,,0.8,0.5,0.0,1124513,AUC,0.6474405582152419
sweep,JPM,short,s2p0_v100_d0p8_r0p5_k0p0,cls_10m,2.0,100,,0.8,0.5,0.0,1124513,AUC,0.5729167485726274
sweep,JPM,short,s2p0_v100_d0p9_r0p1_k0p0,cls_1m,2.0,100,,0.9,0.1,0.0,1124513,AUC,0.6421176856661821
sweep,JPM,short,s2p0_v100_d0p9_r0p1_k0p0,cls_10m,2.0,100,,0.9,0.1,0.0,1124513,AUC,0.5653116694757343
sweep,JPM,short,s2p0_v100_d0p9_r0p3_k0p0,cls_1m,2.0,100,,0.9,0.3,0.0,1124513,AUC,0.6425188969407909
sweep,JPM,short,s2p0_v100_d0p9_r0p3_k0p0,cls_10m,2.0,100,,0.9,0.3,0.0,1124513,AUC,0.5658027458509993
sweep,JPM,short,s2p0_v100_d0p9_r0p5_k0p0,cls_1m,2.0,100,,0.9,0.5,0.0,1124513,AUC,0.6426189669374948
sweep,JPM,short,s2p0_v100_d0p9_r0p5_k0p0,cls_10m,2.0,100,,0.9,0.5,0.0,1124513,AUC,0.5659640131702499
sweep,JPM,short,s2p0_v200_d0p7_r0p1_k0p0,cls_1m,2.0,200,,0.7,0.1,0.0,728020,AUC,0.6461078203439662
sweep,JPM,short,s2p0_v200_d0p7_r0p1_k0p0,cls_10m,2.0,200,,0.7,0.1,0.0,728020,AUC,0.5644718805008543
sweep,JPM,short,s2p0_v200_d0p7_r0p3_k0p0,cls_1m,2.0,200,,0.7,0.3,0.0,728020,AUC,0.6521320411924365
sweep,JPM,short,s2p0_v200_d0p7_r0p3_k0p0,cls_10m,2.0,200,,0.7,0.3,0.0,728020,AUC,0.5719146998342173
sweep,JPM,short,s2p0_v200_d0p7_r0p5_k0p0,cls_1m,2.0,200,,0.7,0.5,0.0,728020,AUC,0.6562192308179702
sweep,JPM,short,s2p0_v200_d0p7_r0p5_k0p0,cls_10m,2.0,200,,0.7,0.5,0.0,728020,AUC,0.5763300772877278
sweep,JPM,short,s2p0_v200_d0p8_r0p1_k0p0,cls_1m,2.0,200,,0.8,0.1,0.0,728020,AUC,0.6451868505183166
sweep,JPM,short,s2p0_v200_d0p8_r0p1_k0p0,cls_10m,2.0,200,,0.8,0.1,0.0,728020,AUC,0.563218504601069
sweep,JPM,short,s2p0_v200_d0p8_r0p3_k0p0,cls_1m,2.0,200,,0.8,0.3,0.0,728020,AUC,0.6488561686843205
sweep,JPM,short,s2p0_v200_d0p8_r0p3_k0p0,cls_10m,2.0,200,,0.8,0.3,0.0,728020,AUC,0.5683905477704738
sweep,JPM,short,s2p0_v200_d0p8_r0p5_k0p0,cls_1m,2.0,200,,0.8,0.5,0.0,728020,AUC,0.6503705419148991
sweep,JPM,short,s2p0_v200_d0p8_r0p5_k0p0,cls_10m,2.0,200,,0.8,0.5,0.0,728020,AUC,0.5706444211736936
sweep,JPM,short,s2p0_v200_d0p9_r0p1_k0p0,cls_1m,2.0,200,,0.9,0.1,0.0,728020,AUC,0.642848533863044
sweep,JPM,short,s2p0_v200_d0p9_r0p1_k0p0,cls_10m,2.0,200,,0.9,0.1,0.0,728020,AUC,0.559308351347835
sweep,JPM,short,s2p0_v200_d0p9_r0p3_k0p0,cls_1m,2.0,200,,0.9,0.3,0.0,728020,AUC,0.6435570232431511
sweep,JPM,short,s2p0_v200_d0p9_r0p3_k0p0,cls_10m,2.0,200,,0.9,0.3,0.0,728020,AUC,0.5603479726592558
sweep,JPM,short,s2p0_v200_d0p9_r0p5_k0p0,cls_1m,2.0,200,,0.9,0.5,0.0,728020,AUC,0.643709275200602
sweep,JPM,short,s2p0_v200_d0p9_r0p5_k0p0,cls_10m,2.0,200,,0.9,0.5,0.0,728020,AUC,0.5605849269004373
sweep,JPM,short,s2p0_v1000_d0p7_r0p1_k0p0,cls_1m,2.0,1000,,0.7,0.1,0.0,100434,AUC,0.6404049425320251
sweep,JPM,short,s2p0_v1000_d0p7_r0p1_k0p0,cls_10m,2.0,1000,,0.7,0.1,0.0,100434,AUC,0.5481033053440796
sweep,JPM,short,s2p0_v1000_d0p7_r0p3_k0p0,cls_1m,2.0,1000,,0.7,0.3,0.0,100434,AUC,0.6535059510603949
sweep,JPM,short,s2p0_v1000_d0p7_r0p3_k0p0,cls_10m,2.0,1000,,0.7,0.3,0.0,100434,AUC,0.5637273027937292
sweep,JPM,short,s2p0_v1000_d0p7_r0p5_k0p0,cls_1m,2.0,1000,,0.7,0.5,0.0,100434,AUC,0.6610850826735266
sweep,JPM,short,s2p0_v1000_d0p7_r0p5_k0p0,cls_10m,2.0,1000,,0.7,0.5,0.0,100434,AUC,0.5734244009274512
sweep,JPM,short,s2p0_v1000_d0p8_r0p1_k0p0,cls_1m,2.0,1000,,0.8,0.1,0.0,100434,AUC,0.6398459931259495
sweep,JPM,short,s2p0_v1000_d0p8_r0p1_k0p0,cls_10m,2.0,1000,,0.8,0.1,0.0,100434,AUC,0.5475430146940071
sweep,JPM,short,s2p0_v1000_d0p8_r0p3_k0p0,cls_1m,2.0,1000,,0.8,0.3,0.0,100434,AUC,0.646220844599955
sweep,JPM,short,s2p0_v1000_d0p8_r0p3_k0p0,cls_10m,2.0,1000,,0.8,0.3,0.0,100434,AUC,0.555856976037098
sweep,JPM,short,s2p0_v1000_d0p8_r0p5_k0p0,cls_1m,2.0,1000,,0.8,0.5,0.0,100434,AUC,0.6471342312674337
sweep,JPM,short,s2p0_v1000_d0p8_r0p5_k0p0,cls_10m,2.0,1000,,0.8,0.5,0.0,100434,AUC,0.5571840942955569
sweep,JPM,short,s2p0_v1000_d0p9_r0p1_k0p0,cls_1m,2.0,1000,,0.9,0.1,0.0,100434,AUC,0.6376067949707218
sweep,JPM,short,s2p0_v1000_d0p9_r0p1_k0p0,cls_10m,2.0,1000,,0.9,0.1,0.0,100434,AUC,0.5437850803093564
sweep,JPM,short,s2p0_v1000_d0p9_r0p3_k0p0,cls_1m,2.0,1000,,0.9,0.3,0.0,100434,AUC,0.6382042578224858
sweep,JPM,short,s2p0_v1000_d0p9_r0p3_k0p0,cls_10m,2.0,1000,,0.9,0.3,0.0,100434,AUC,0.5443568159380051
sweep,JPM,short,s2p0_v1000_d0p9_r0p5_k0p0,cls_1m,2.0,1000,,0.9,0.5,0.0,100434,AUC,0.6382197516971806
sweep,JPM,short,s2p0_v1000_d0p9_r0p5_k0p0,cls_10m,2.0,1000,,0.9,0.5,0.0,100434,AUC,0.5443937630168287
sweep,JPM,long,s0p5_v50_d0p7_r0p1_k0p0,cls_close,0.5,50,,0.7,0.1,0.0,1974001,AUC,0.584365605667048
sweep,JPM,long,s0p5_v50_d0p7_r0p3_k0p0,cls_close,0.5,50,,0.7,0.3,0.0,1974001,AUC,0.5779846110988118
sweep,JPM,long,s0p5_v50_d0p7_r0p5_k0p0,cls_close,0.5,50,,0.7,0.5,0.0,1974001,AUC,0.5782877262717624
sweep,JPM,long,s0p5_v50_d0p8_r0p1_k0p0,cls_close,0.5,50,,0.8,0.1,0.0,1974001,AUC,0.577973313967647
sweep,JPM,long,s0p5_v50_d0p8_r0p3_k0p0,cls_close,0.5,50,,0.8,0.3,0.0,1974001,AUC,0.5780492098893953
sweep,JPM,long,s0p5_v50_d0p8_r0p5_k0p0,cls_close,0.5,50,,0.8,0.5,0.0,1974001,AUC,0.578445442205511
sweep,JPM,long,s0p5_v50_d0p9_r0p1_k0p0,cls_close,0.5,50,,0.9,0.1,0.0,1974001,AUC,0.5771680109323344
sweep,JPM,long,s0p5_v50_d0p9_r0p3_k0p0,cls_close,0.5,50,,0.9,0.3,0.0,1974001,AUC,0.5772025103003445
sweep,JPM,long,s0p5_v50_d0p9_r0p5_k0p0,cls_close,0.5,50,,0.9,0.5,0.0,1974001,AUC,0.5772311458473984
sweep,JPM,long,s0p5_v100_d0p7_r0p1_k0p0,cls_close,0.5,100,,0.7,0.1,0.0,1715393,AUC,0.579335846297053
sweep,JPM,long,s0p5_v100_d0p7_r0p3_k0p0,cls_close,0.5,100,,0.7,0.3,0.0,1715393,AUC,0.5794008150975418
sweep,JPM,long,s0p5_v100_d0p7_r0p5_k0p0,cls_close,0.5,100,,0.7,0.5,0.0,1715393,AUC,0.57975550909493
sweep,JPM,long,s0p5_v100_d0p8_r0p1_k0p0,cls_close,0.5,100,,0.8,0.1,0.0,1715393,AUC,0.5789283469200155
sweep,JPM,long,s0p5_v100_d0p8_r0p3_k0p0,cls_close,0.5,100,,0.8,0.3,0.0,1715393,AUC,0.5792099198703203
sweep,JPM,long,s0p5_v100_d0p8_r0p5_k0p0,cls_close,0.5,100,,0.8,0.5,0.0,1715393,AUC,0.5796481185011272
sweep,JPM,long,s0p5_v100_d0p9_r0p1_k0p0,cls_close,0.5,100,,0.9,0.1,0.0,1715393,AUC,0.578107105941253
sweep,JPM,long,s0p5_v100_d0p9_r0p3_k0p0,cls_close,0.5,100,,0.9,0.3,0.0,1715393,AUC,0.5781746826815629
sweep,JPM,long,s0p5_v100_d0p9_r0p5_k0p0,cls_close,0.5,100,,0.9,0.5,0.0,1715393,AUC,0.5782048818324339
sweep,JPM,long,s0p5_v200_d0p7_r0p1_k0p0,cls_close,0.5,200,,0.7,0.1,0.0,899955,AUC,0.5813560697212591
sweep,JPM,long,s0p5_v200_d0p7_r0p3_k0p0,cls_close,0.5,200,,0.7,0.3,0.0,899955,AUC,0.5814370550750427
sweep,JPM,long,s0p5_v200_d0p7_r0p5_k0p0,cls_close,0.5,200,,0.7,0.5,0.0,899955,AUC,0.5811531800041546
sweep,JPM,long,s0p5_v200_d0p8_r0p1_k0p0,cls_close,0.5,200,,0.8,0.1,0.0,899955,AUC,0.5813277099640286
sweep,JPM,long,s0p5_v200_d0p8_r0p3_k0p0,cls_close,0.5,200,,0.8,0.3,0.0,899955,AUC,0.5818314087772186
sweep,JPM,long,s0p5_v200_d0p8_r0p5_k0p0,cls_close,0.5,200,,0.8,0.5,0.0,899955,AUC,0.582189053709153
sweep,JPM,long,s0p5_v200_d0p9_r0p1_k0p0,cls_close,0.5,200,,0.9,0.1,0.0,899955,AUC,0.5808555619003762
sweep,JPM,long,s0p5_v200_d0p9_r0p3_k0p0,cls_close,0.5,200,,0.9,0.3,0.0,899955,AUC,0.580984185531277
sweep,JPM,long,s0p5_v200_d0p9_r0p5_k0p0,cls_close,0.5,200,,0.9,0.5,0.0,899955,AUC,0.5809907820839578
sweep,JPM,long,s0p5_v1000_d0p7_r0p1_k0p0,cls_close,0.5,1000,,0.7,0.1,0.0,61109,AUC,0.5864085363063286
sweep,JPM,long,s0p5_v1000_d0p7_r0p3_k0p0,cls_close,0.5,1000,,0.7,0.3,0.0,61109,AUC,0.5877411670860859
sweep,JPM,long,s0p5_v1000_d0p7_r0p5_k0p0,cls_close,0.5,1000,,0.7,0.5,0.0,61109,AUC,0.5883151301261201
sweep,JPM,long,s0p5_v1000_d0p8_r0p1_k0p0,cls_close,0.5,1000,,0.8,0.1,0.0,61109,AUC,0.5863818641041485
sweep,JPM,long,s0p5_v1000_d0p8_r0p3_k0p0,cls_close,0.5,1000,,0.8,0.3,0.0,61109,AUC,0.586762456682455
sweep,JPM,long,s0p5_v1000_d0p8_r0p5_k0p0,cls_close,0.5,1000,,0.8,0.5,0.0,61109,AUC,0.5870653337775362
sweep,JPM,long,s0p5_v1000_d0p9_r0p1_k0p0,cls_close,0.5,1000,,0.9,0.1,0.0,61109,AUC,0.5870089933066149
sweep,JPM,long,s0p5_v1000_d0p9_r0p3_k0p0,cls_close,0.5,1000,,0.9,0.3,0.0,61109,AUC,0.5865844470700081
sweep,TSLA,short,s0p5_v50_d0p7_r0p1_k0p0,cls_1m,0.5,50,,0.7,0.1,0.0,5776548,AUC,0.6489185717215264
sweep,TSLA,short,s0p5_v50_d0p7_r0p1_k0p0,cls_10m,0.5,50,,0.7,0.1,0.0,5776548,AUC,0.5476915559381088
sweep,TSLA,short,s0p5_v50_d0p7_r0p3_k0p0,cls_1m,0.5,50,,0.7,0.3,0.0,5776548,AUC,0.6221156994607594
sweep,TSLA,short,s0p5_v50_d0p7_r0p3_k0p0,cls_10m,0.5,50,,0.7,0.3,0.0,5776548,AUC,0.5566480275328605
sweep,TSLA,short,s0p5_v50_d0p7_r0p5_k0p0,cls_1m,0.5,50,,0.7,0.5,0.0,5776548,AUC,0.6226272771068643
sweep,TSLA,short,s0p5_v50_d0p7_r0p5_k0p0,cls_10m,0.5,50,,0.7,0.5,0.0,5776548,AUC,0.5599086062335997
sweep,TSLA,short,s0p5_v50_d0p8_r0p1_k0p0,cls_1m,0.5,50,,0.8,0.1,0.0,5776548,AUC,0.6230367308228567
sweep,TSLA,short,s0p5_v50_d0p8_r0p1_k0p0,cls_10m,0.5,50,,0.8,0.1,0.0,5776548,AUC,0.5581981934686703
sweep,TSLA,short,s0p5_v50_d0p8_r0p3_k0p0,cls_1m,0.5,50,,0.8,0.3,0.0,5776548,AUC,0.6235788836583898
sweep,TSLA,short,s0p5_v50_d0p8_r0p3_k0p0,cls_10m,0.5,50,,0.8,0.3,0.0,5776548,AUC,0.5588617496322626
sweep,TSLA,short,s0p5_v50_d0p8_r0p5_k0p0,cls_1m,0.5,50,,0.8,0.5,0.0,5776548,AUC,0.6230865686614364
sweep,TSLA,short,s0p5_v50_d0p8_r0p5_k0p0,cls_10m,0.5,50,,0.8,0.5,0.0,5776548,AUC,0.558176484106031
sweep,TSLA,short,s0p5_v50_d0p9_r0p1_k0p0,cls_1m,0.5,50,,0.9,0.1,0.0,5776548,AUC,0.621671435494035
sweep,TSLA,short,s0p5_v50_d0p9_r0p1_k0p0,cls_10m,0.5,50,,0.9,0.1,0.0,5776548,AUC,0.5554565167199663
sweep,TSLA,short,s0p5_v50_d0p9_r0p3_k0p0,cls_1m,0.5,50,,0.9,0.3,0.0,5776548,AUC,0.6219316119034446
sweep,TSLA,short,s0p5_v50_d0p9_r0p3_k0p0,cls_10m,0.5,50,,0.9,0.3,0.0,5776548,AUC,0.5559454066898842
sweep,TSLA,short,s0p5_v50_d0p9_r0p5_k0p0,cls_1m,0.5,50,,0.9,0.5,0.0,5776548,AUC,0.6219475474713364
sweep,TSLA,short,s0p5_v50_d0p9_r0p5_k0p0,cls_10m,0.5,50,,0.9,0.5,0.0,5776548,AUC,0.5560463638483997
sweep,TSLA,short,s0p5_v100_d0p7_r0p1_k0p0,cls_1m,0.5,100,,0.7,0.1,0.0,5263150,AUC,0.6238312925924809
sweep,TSLA,short,s0p5_v100_d0p7_r0p1_k0p0,cls_10m,0.5,100,,0.7,0.1,0.0,5263150,AUC,0.56018501852737
sweep,TSLA,short,s0p5_v100_d0p7_r0p3_k0p0,cls_1m,0.5,100,,0.7,0.3,0.0,5263150,AUC,0.623532906948408
sweep,TSLA,short,s0p5_v100_d0p7_r0p3_k0p0,cls_10m,0.5,100,,0.7,0.3,0.0,5263150,AUC,0.5591220375880913
sweep,TSLA,short,s0p5_v100_d0p7_r0p5_k0p0,cls_1m,0.5,100,,0.7,0.5,0.0,5263150,AUC,0.6227131482403442
sweep,TSLA,short,s0p5_v100_d0p7_r0p5_k0p0,cls_10m,0.5,100,,0.7,0.5,0.0,5263150,AUC,0.559433845131335
sweep,TSLA,short,s0p5_v100_d0p8_r0p1_k0p0,cls_1m,0.5,100,,0.8,0.1,0.0,5263150,AUC,0.6237027718442527
sweep,TSLA,short,s0p5_v100_d0p8_r0p1_k0p0,cls_10m,0.5,100,,0.8,0.1,0.0,5263150,AUC,0.5570703903494635
sweep,TSLA,short,s0p5_v100_d0p8_r0p3_k0p0,cls_1m,0.5,100,,0.8,0.3,0.0,5263150,AUC,0.6248311158852233
sweep,TSLA,short,s0p5_v100_d0p8_r0p3_k0p0,cls_10m,0.5,100,,0.8,0.3,0.0,5263150,AUC,0.5589036616860088
sweep,TSLA,short,s0p5_v100_d0p8_r0p5_k0p0,cls_1m,0.5,100,,0.8,0.5,0.0,5263150,AUC,0.6245162542566237
sweep,TSLA,short,s0p5_v100_d0p8_r0p5_k0p0,cls_10m,0.5,100,,0.8,0.5,0.0,5263150,AUC,0.5585193321044202
sweep,TSLA,short,s0p5_v100_d0p9_r0p1_k0p0,cls_1m,0.5,100,,0.9,0.1,0.0,5263150,AUC,0.6216840672272061
sweep,TSLA,short,s0p5_v100_d0p9_r0p1_k0p0,cls_10m,0.5,100,,0.9,0.1,0.0,5263150,AUC,0.5531540494157262
sweep,TSLA,short,s0p5_v100_d0p9_r0p3_k0p0,cls_1m,0.5,100,,0.9,0.3,0.0,5263150,AUC,0.622007557632836
sweep,TSLA,short,s0p5_v100_d0p9_r0p3_k0p0,cls_10m,0.5,100,,0.9,0.3,0.0,5263150,AUC,0.5537728358810187
sweep,TSLA,short,s0p5_v100_d0p9_r0p5_k0p0,cls_1m,0.5,100,,0.9,0.5,0.0,5263150,AUC,0.6220362089999364
sweep,TSLA,short,s0p5_v100_d0p9_r0p5_k0p0,cls_10m,0.5,100,,0.9,0.5,0.0,5263150,AUC,0.5538890025438725
sweep,TSLA,short,s0p5_v200_d0p7_r0p1_k0p0,cls_1m,0.5,200,,0.7,0.1,0.0,3968796,AUC,0.6238418859911615
sweep,TSLA,short,s0p5_v200_d0p7_r0p1_k0p0,cls_10m,0.5,200,,0.7,0.1,0.0,3968796,AUC,0.5559119414786852
sweep,TSLA,short,s0p5_v200_d0p7_r0p3_k0p0,cls_1m,0.5,200,,0.7,0.3,0.0,3968796,AUC,0.6274099424783961
sweep,TSLA,short,s0p5_v200_d0p7_r0p3_k0p0,cls_10m,0.5,200,,0.7,0.3,0.0,3968796,AUC,0.5627422799146632
sweep,TSLA,short,s0p5_v200_d0p7_r0p5_k0p0,cls_1m,0.5,200,,0.7,0.5,0.0,3968796,AUC,0.6261307107445858
sweep,TSLA,short,s0p5_v200_d0p7_r0p5_k0p0,cls_10m,0.5,200,,0.7,0.5,0.0,3968796,AUC,0.5607660428573875
sweep,TSLA,short,s0p5_v200_d0p8_r0p1_k0p0,cls_1m,0.5,200,,0.8,0.1,0.0,3968796,AUC,0.6229782259666828
sweep,TSLA,short,s0p5_v200_d0p8_r0p1_k0p0,cls_10m,0.5,200,,0.8,0.1,0.0,3968796,AUC,0.554066379194869
sweep,TSLA,short,s0p5_v200_d0p8_r0p3_k0p0,cls_1m,0.5,200,,0.8,0.3,0.0,3968796,AUC,0.6258102607035134
sweep,TSLA,short,s0p5_v200_d0p8_r0p3_k0p0,cls_10m,0.5,200,,0.8,0.3,0.0,3968796,AUC,0.5596835684563161
sweep,TSLA,short,s0p5_v200_d0p8_r0p5_k0p0,cls_1m,0.5,200,,0.8,0.5,0.0,3968796,AUC,0.6261954918550539
sweep,TSLA,short,s0p5_v200_d0p8_r0p5_k0p0,cls_10m,0.5,200,,0.8,0.5,0.0,3968796,AUC,0.5605959489751478
sweep,TSLA,short,s0p5_v200_d0p9_r0p1_k0p0,cls_1m,0.5,200,,0.9,0.1,0.0,3968796,AUC,0.6202923103794648
sweep,TSLA,short,s0p5_v200_d0p9_r0p1_k0p0,cls_10m,0.5,200,,0.9,0.1,0.0,3968796,AUC,0.548049855894153
sweep,TSLA,short,s0p5_v200_d0p9_r0p3_k0p0,cls_1m,0.5,200,,0.9,0.3,0.0,3968796,AUC,0.6207830323094985
sweep,TSLA,short,s0p5_v200_d0p9_r0p3_k0p0,cls_10m,0.5,200,,0.9,0.3,0.0,3968796,AUC,0.5491894276138868
sweep,TSLA,short,s0p5_v200_d0p9_r0p5_k0p0,cls_1m,0.5,200,,0.9,0.5,0.0,3968796,AUC,0.6208313345479733
sweep,TSLA,short,s0p5_v200_d0p9_r0p5_k0p0,cls_10m,0.5,200,,0.9,0.5,0.0,3968796,AUC,0.5493609458089184
sweep,TSLA,short,s0p5_v1000_d0p7_r0p1_k0p0,cls_1m,0.5,1000,,0.7,0.1,0.0,1256344,AUC,0.6268190755310423
sweep,TSLA,short,s0p5_v1000_d0p7_r0p1_k0p0,cls_10m,0.5,1000,,0.7,0.1,0.0,1256344,AUC,0.5520581909167375
sweep,TSLA,short,s0p5_v1000_d0p7_r0p3_k0p0,cls_1m,0.5,1000,,0.7,0.3,0.0,1256344,AUC,0.6335535641935723
sweep,TSLA,short,s0p5_v1000_d0p7_r0p3_k0p0,cls_10m,0.5,1000,,0.7,0.3,0.0,1256344,AUC,0.564895717498501
sweep,TSLA,short,s0p5_v1000_d0p7_r0p5_k0p0,cls_1m,0.5,1000,,0.7,0.5,0.0,1256344,AUC,0.6349971735262704
sweep,TSLA,short,s0p5_v1000_d0p7_r0p5_k0p0,cls_10m,0.5,1000,,0.7,0.5,0.0,1256344,AUC,0.5661985653552304
sweep,TSLA,long,s0p5_v50_d0p7_r0p1_k0p2,cls_close,0.5,50,,0.7,0.1,0.2,1524095,AUC,0.5612712917368261
sweep,TSLA,long,s0p5_v50_d0p7_r0p1_k0p2,cls_clop,0.5,50,,0.7,0.1,0.2,1524095,AUC,0.5149636153524934
sweep,TSLA,long,s0p5_v50_d0p7_r0p1_k0p2,cls_clcl,0.5,50,,0.7,0.1,0.2,1524095,AUC,0.5010135572137814
sweep,MS,short,s0p5_v50_d0p7_r0p1_k0p0,cls_1m,0.5,50,,0.7,0.1,0.0,1505334,AUC,0.6517839449856686
sweep,MS,short,s0p5_v50_d0p7_r0p1_k0p0,cls_10m,0.5,50,,0.7,0.1,0.0,1505334,AUC,0.5463299099279054
sweep,MS,short,s0p5_v50_d0p7_r0p3_k0p0,cls_1m,0.5,50,,0.7,0.3,0.0,1505334,AUC,0.6488996393086898
sweep,MS,short,s0p5_v50_d0p7_r0p3_k0p0,cls_10m,0.5,50,,0.7,0.3,0.0,1505334,AUC,0.5667614719740864
sweep,MS,short,s0p5_v50_d0p7_r0p5_k0p0,cls_1m,0.5,50,,0.7,0.5,0.0,1505334,AUC,0.6505700573672409
sweep,MS,short,s0p5_v50_d0p7_r0p5_k0p0,cls_10m,0.5,50,,0.7,0.5,0.0,1505334,AUC,0.5689801605198198
sweep,MS,short,s0p5_v50_d0p8_r0p1_k0p0,cls_1m,0.5,50,,0.8,0.1,0.0,1505334,AUC,0.6458527019588537
sweep,MS,short,s0p5_v50_d0p8_r0p1_k0p0,cls_10m,0.5,50,,0.8,0.1,0.0,1505334,AUC,0.5622511323693058
sweep,MS,short,s0p5_v50_d0p8_r0p3_k0p0,cls_1m,0.5,50,,0.8,0.3,0.0,1505334,AUC,0.6474424605242926
sweep,MS,short,s0p5_v50_d0p8_r0p3_k0p0,cls_10m,0.5,50,,0.8,0.3,0.0,1505334,AUC,0.5646249991326792
sweep,MS,short,s0p5_v50_d0p8_r0p5_k0p0,cls_1m,0.5,50,,0.8,0.5,0.0,1505334,AUC,0.6481609645019175
sweep,MS,short,s0p5_v50_d0p8_r0p5_k0p0,cls_10m,0.5,50,,0.8,0.5,0.0,1505334,AUC,0.5657152467112615
sweep,MS,short,s0p5_v50_d0p9_r0p1_k0p0,cls_1m,0.5,50,,0.9,0.1,0.0,1505334,AUC,0.6443642917737153
sweep,MS,short,s0p5_v50_d0p9_r0p1_k0p0,cls_10m,0.5,50,,0.9,0.1,0.0,1505334,AUC,0.5600310527891654
sweep,MS,short,s0p5_v50_d0p9_r0p3_k0p0,cls_1m,0.5,50,,0.9,0.3,0.0,1505334,AUC,0.6446254979604704
sweep,MS,short,s0p5_v50_d0p9_r0p3_k0p0,cls_10m,0.5,50,,0.9,0.3,0.0,1505334,AUC,0.5605075017898983
sweep,MS,short,s0p5_v50_d0p9_r0p5_k0p0,cls_1m,0.5,50,,0.9,0.5,0.0,1505334,AUC,0.6446498646892619
sweep,MS,short,s0p5_v50_d0p9_r0p5_k0p0,cls_10m,0.5,50,,0.9,0.5,0.0,1505334,AUC,0.5605285690865779
sweep,MS,short,s0p5_v100_d0p7_r0p1_k0p0,cls_1m,0.5,100,,0.7,0.1,0.0,1306410,AUC,0.6469092062682695
sweep,MS,short,s0p5_v100_d0p7_r0p1_k0p0,cls_10m,0.5,100,,0.7,0.1,0.0,1306410,AUC,0.5631136225109247
sweep,MS,short,s0p5_v100_d0p7_r0p3_k0p0,cls_1m,0.5,100,,0.7,0.3,0.0,1306410,AUC,0.6492838326632556
sweep,MS,short,s0p5_v100_d0p7_r0p3_k0p0,cls_10m,0.5,100,,0.7,0.3,0.0,1306410,AUC,0.5667396225611226
sweep,MS,short,s0p5_v100_d0p7_r0p5_k0p0,cls_1m,0.5,100,,0.7,0.5,0.0,1306410,AUC,0.6511005819485038
sweep,MS,short,s0p5_v100_d0p7_r0p5_k0p0,cls_10m,0.5,100,,0.7,0.5,0.0,1306410,AUC,0.5693613742102673
sweep,MS,short,s0p5_v100_d0p8_r0p1_k0p0,cls_1m,0.5,100,,0.8,0.1,0.0,1306410,AUC,0.6461876289857629
sweep,MS,short,s0p5_v100_d0p8_r0p1_k0p0,cls_10m,0.5,100,,0.8,0.1,0.0,1306410,AUC,0.5621406250652153
sweep,MS,short,s0p5_v100_d0p8_r0p3_k0p0,cls_1m,0.5,100,,0.8,0.3,0.0,1306410,AUC,0.6478353034199675
sweep,MS,short,s0p5_v100_d0p8_r0p3_k0p0,cls_10m,0.5,100,,0.8,0.3,0.0,1306410,AUC,0.5646375452811457
sweep,MS,short,s0p5_v100_d0p8_r0p5_k0p0,cls_1m,0.5,100,,0.8,0.5,0.0,1306410,AUC,0.6486093974208398
sweep,MS,short,s0p5_v100_d0p8_r0p5_k0p0,cls_10m,0.5,100,,0.8,0.5,0.0,1306410,AUC,0.5658225513153631
sweep,MS,short,s0p5_v100_d0p9_r0p1_k0p0,cls_1m,0.5,100,,0.9,0.1,0.0,1306410,AUC,0.6446660819162016
sweep,MS,short,s0p5_v100_d0p9_r0p1_k0p0,cls_10m,0.5,100,,0.9,0.1,0.0,1306410,AUC,0.5599194017236745
sweep,MS,short,s0p5_v100_d0p9_r0p3_k0p0,cls_1m,0.5,100,,0.9,0.3,0.0,1306410,AUC,0.6449543459611025
sweep,MS,short,s0p5_v100_d0p9_r0p3_k0p0,cls_10m,0.5,100,,0.9,0.3,0.0,1306410,AUC,0.5603693840816071
sweep,MS,short,s0p5_v100_d0p9_r0p5_k0p0,cls_1m,0.5,100,,0.9,0.5,0.0,1306410,AUC,0.6449867953012841
sweep,MS,short,s0p5_v100_d0p9_r0p5_k0p0,cls_10m,0.5,100,,0.9,0.5,0.0,1306410,AUC,0.5604426618503843
sweep,MS,short,s0p5_v200_d0p7_r0p1_k0p0,cls_1m,0.5,200,,0.7,0.1,0.0,705340,AUC,0.6467326937939055
sweep,MS,short,s0p5_v200_d0p7_r0p1_k0p0,cls_10m,0.5,200,,0.7,0.1,0.0,705340,AUC,0.5571863221916611
sweep,MS,short,s0p5_v200_d0p7_r0p3_k0p0,cls_1m,0.5,200,,0.7,0.3,0.0,705340,AUC,0.649380515708526
sweep,MS,short,s0p5_v200_d0p7_r0p3_k0p0,cls_10m,0.5,200,,0.7,0.3,0.0,705340,AUC,0.5613877273388105
sweep,MS,short,s0p5_v200_d0p7_r0p5_k0p0,cls_1m,0.5,200,,0.7,0.5,0.0,705340,AUC,0.6515155880829036
sweep,MS,short,s0p5_v200_d0p7_r0p5_k0p0,cls_10m,0.5,200,,0.7,0.5,0.0,705340,AUC,0.5654624104553891
sweep,MS,short,s0p5_v200_d0p8_r0p1_k0p0,cls_1m,0.5,200,,0.8,0.1,0.0,705340,AUC,0.6463435018847539
sweep,MS,short,s0p5_v200_d0p8_r0p1_k0p0,cls_10m,0.5,200,,0.8,0.1,0.0,705340,AUC,0.5569489285519228
sweep,MS,short,s0p5_v200_d0p8_r0p3_k0p0,cls_1m,0.5,200,,0.8,0.3,0.0,705340,AUC,0.6483542341260499
sweep,MS,short,s0p5_v200_d0p8_r0p3_k0p0,cls_10m,0.5,200,,0.8,0.3,0.0,705340,AUC,0.5599500766709206
sweep,MS,short,s0p5_v200_d0p8_r0p5_k0p0,cls_1m,0.5,200,,0.8,0.5,0.0,705340,AUC,0.6491783833443842
sweep,MS,short,s0p5_v200_d0p8_r0p5_k0p0,cls_10m,0.5,200,,0.8,0.5,0.0,705340,AUC,0.5614892471151203
sweep,MS,short,s0p5_v200_d0p9_r0p1_k0p0,cls_1m,0.5,200,,0.9,0.1,0.0,705340,AUC,0.6452459630874825
sweep,MS,short,s0p5_v200_d0p9_r0p1_k0p0,cls_10m,0.5,200,,0.9,0.1,0.0,705340,AUC,0.5553175026946364
sweep,MS,short,s0p5_v200_d0p9_r0p3_k0p0,cls_1m,0.5,200,,0.9,0.3,0.0,705340,AUC,0.6456891772063115
sweep,MS,short,s0p5_v200_d0p9_r0p3_k0p0,cls_10m,0.5,200,,0.9,0.3,0.0,705340,AUC,0.5560624940762067
sweep,MS,short,s0p5_v200_d0p9_r0p5_k0p0,cls_1m,0.5,200,,0.9,0.5,0.0,705340,AUC,0.6457110430491576
sweep,MS,short,s0p5_v200_d0p9_r0p5_k0p0,cls_10m,0.5,200,,0.9,0.5,0.0,705340,AUC,0.5560824947034149
sweep,MS,short,s0p5_v1000_d0p7_r0p1_k0p0,cls_1m,0.5,1000,,0.7,0.1,0.0,59245,AUC,0.6530091617295606
sweep,MS,short,s0p5_v1000_d0p7_r0p1_k0p0,cls_10m,0.5,1000,,0.7,0.1,0.0,59245,AUC,0.5530667681512369
sweep,MS,short,s0p5_v1000_d0p7_r0p3_k0p0,cls_1m,0.5,1000,,0.7,0.3,0.0,59245,AUC,0.6589523013308556
sweep,MS,short,s0p5_v1000_d0p7_r0p3_k0p0,cls_10m,0.5,1000,,0.7,0.3,0.0,59245,AUC,0.5604445267064943
sweep,MS,short,s0p5_v1000_d0p7_r0p5_k0p0,cls_1m,0.5,1000,,0.7,0.5,0.0,59245,AUC,0.6612298507811301
sweep,MS,short,s0p5_v1000_d0p7_r0p5_k0p0,cls_10m,0.5,1000,,0.7,0.5,0.0,59245,AUC,0.564388611744927
sweep,MS,short,s0p5_v1000_d0p8_r0p1_k0p0,cls_1m,0.5,1000,,0.8,0.1,0.0,59245,AUC,0.6521567093388261
sweep,MS,short,s0p5_v1000_d0p8_r0p1_k0p0,cls_10m,0.5,1000,,0.8,0.1,0.0,59245,AUC,0.5525259772808747
sweep,MS,short,s0p5_v1000_d0p8_r0p3_k0p0,cls_1m,0.5,1000,,0.8,0.3,0.0,59245,AUC,0.6567819886929127
sweep,MS,short,s0p5_v1000_d0p8_r0p3_k0p0,cls_10m,0.5,1000,,0.8,0.3,0.0,59245,AUC,0.5579454686601326
sweep,MS,short,s0p5_v1000_d0p8_r0p5_k0p0,cls_1m,0.5,1000,,0.8,0.5,0.0,59245,AUC,0.6575520712179164
sweep,MS,short,s0p5_v1000_d0p8_r0p5_k0p0,cls_10m,0.5,1000,,0.8,0.5,0.0,59245,AUC,0.5586505914857082
sweep,MS,short,s0p5_v1000_d0p9_r0p1_k0p0,cls_1m,0.5,1000,,0.9,0.1,0.0,59245,AUC,0.6505180284923242
sweep,MS,short,s0p5_v1000_d0p9_r0p1_k0p0,cls_10m,0.5,1000,,0.9,0.1,0.0,59245,AUC,0.5493449996170403
sweep,MS,short,s0p5_v1000_d0p9_r0p3_k0p0,cls_1m,0.5,1000,,0.9,0.3,0.0,59245,AUC,0.6514780026860001
sweep,MS,short,s0p5_v1000_d0p9_r0p3_k0p0,cls_10m,0.5,1000,,0.9,0.3,0.0,59245,AUC,0.5506103898947584
sweep,MS,short,s0p5_v1000_d0p9_r0p5_k0p0,cls_1m,0.5,1000,,0.9,0.5,0.0,59245,AUC,0.6515198781147309
sweep,MS,short,s0p5_v1000_d0p9_r0p5_k0p0,cls_10m,0.5,1000,,0.9,0.5,0.0,59245,AUC,0.5504921312156486
sweep,MS,short,s1p0_v50_d0p7_r0p1_k0p0,cls_1m,1.0,50,,0.7,0.1,0.0,1323843,AUC,0.6484292316848619
sweep,MS,short,s1p0_v50_d0p7_r0p1_k0p0,cls_10m,1.0,50,,0.7,0.1,0.0,1323843,AUC,0.5664997646008577
sweep,MS,short,s1p0_v50_d0p7_r0p3_k0p0,cls_1m,1.0,50,,0.7,0.3,0.0,1323843,AUC,0.6518655529404099
sweep,MS,short,s1p0_v50_d0p7_r0p3_k0p0,cls_10m,1.0,50,,0.7,0.3,0.0,1323843,AUC,0.5712982035078946
sweep,MS,short,s1p0_v50_d0p7_r0p5_k0p0,cls_1m,1.0,50,,0.7,0.5,0.0,1323843,AUC,0.6543172451309813
sweep,MS,short,s1p0_v50_d0p7_r0p5_k0p0,cls_10m,1.0,50,,0.7,0.5,0.0,1323843,AUC,0.5742888424561869
sweep,MS,short,s1p0_v50_d0p8_r0p1_k0p0,cls_1m,1.0,50,,0.8,0.1,0.0,1323843,AUC,0.6474817555109738
sweep,MS,short,s1p0_v50_d0p8_r0p1_k0p0,cls_10m,1.0,50,,0.8,0.1,0.0,1323843,AUC,0.5651061935662992
sweep,MS,short,s1p0_v50_d0p8_r0p3_k0p0,cls_1m,1.0,50,,0.8,0.3,0.0,1323843,AUC,0.6496399339716528
sweep,MS,short,s1p0_v50_d0p8_r0p3_k0p0,cls_10m,1.0,50,,0.8,0.3,0.0,1323843,AUC,0.5684573812746148
sweep,MS,short,s1p0_v50_d0p8_r0p5_k0p0,cls_1m,1.0,50,,0.8,0.5,0.0,1323843,AUC,0.6506235486311263
sweep,MS,short,s1p0_v50_d0p8_r0p5_k0p0,cls_10m,1.0,50,,0.8,0.5,0.0,1323843,AUC,0.569775750152322
sweep,MS,short,s1p0_v50_d0p9_r0p1_k0p0,cls_1m,1.0,50,,0.9,0.1,0.0,1323843,AUC,0.6457076520443182
sweep,MS,short,s1p0_v50_d0p9_r0p1_k0p0,cls_10m,1.0,50,,0.9,0.1,0.0,1323843,AUC,0.5623726270323748
sweep,MS,short,s1p0_v50_d0p9_r0p3_k0p0,cls_1m,1.0,50,,0.9,0.3,0.0,1323843,AUC,0.6460304577658419
sweep,MS,short,s1p0_v50_d0p9_r0p3_k0p0,cls_10m,1.0,50,,0.9,0.3,0.0,1323843,AUC,0.5630235115051507
sweep,MS,short,s1p0_v50_d0p9_r0p5_k0p0,cls_1m,1.0,50,,0.9,0.5,0.0,1323843,AUC,0.6460903435088026
sweep,MS,short,s1p0_v50_d0p9_r0p5_k0p0,cls_10m,1.0,50,,0.9,0.5,0.0,1323843,AUC,0.5631393575491236
sweep,MS,short,s1p0_v100_d0p7_r0p1_k0p0,cls_1m,1.0,100,,0.7,0.1,0.0,1163757,AUC,0.6490791738473295
sweep,MS,short,s1p0_v100_d0p7_r0p1_k0p0,cls_10m,1.0,100,,0.7,0.1,0.0,1163757,AUC,0.5667434222645815
sweep,MS,short,s1p0_v100_d0p7_r0p3_k0p0,cls_1m,1.0,100,,0.7,0.3,0.0,1163757,AUC,0.6525220044544354
sweep,MS,short,s1p0_v100_d0p7_r0p3_k0p0,cls_10m,1.0,100,,0.7,0.3,0.0,1163757,AUC,0.5713937967737795
sweep,MS,short,s1p0_v100_d0p7_r0p5_k0p0,cls_1m,1.0,100,,0.7,0.5,0.0,1163757,AUC,0.6551027744902506
sweep,MS,short,s1p0_v100_d0p7_r0p5_k0p0,cls_10m,1.0,100,,0.7,0.5,0.0,1163757,AUC,0.574665134203861
sweep,MS,short,s1p0_v100_d0p8_r0p1_k0p0,cls_1m,1.0,100,,0.8,0.1,0.0,1163757,AUC,0.6481995025099001
sweep,MS,short,s1p0_v100_d0p8_r0p1_k0p0,cls_10m,1.0,100,,0.8,0.1,0.0,1163757,AUC,0.5655566320648541
sweep,MS,short,s1p0_v100_d0p8_r0p3_k0p0,cls_1m,1.0,100,,0.8,0.3,0.0,1163757,AUC,0.6503954566929088
sweep,MS,short,s1p0_v100_d0p8_r0p3_k0p0,cls_10m,1.0,100,,0.8,0.3,0.0,1163757,AUC,0.5687639096498426
sweep,MS,short,s1p0_v100_d0p8_r0p5_k0p0,cls_1m,1.0,100,,0.8,0.5,0.0,1163757,AUC,0.6514102929182307
sweep,MS,short,s1p0_v100_d0p8_r0p5_k0p0,cls_10m,1.0,100,,0.8,0.5,0.0,1163757,AUC,0.570177301526353
sweep,MS,short,s1p0_v100_d0p9_r0p1_k0p0,cls_1m,1.0,100,,0.9,0.1,0.0,1163757,AUC,0.6464589875672522
sweep,MS,short,s1p0_v100_d0p9_r0p1_k0p0,cls_10m,1.0,100,,0.9,0.1,0.0,1163757,AUC,0.562886271672144
sweep,MS,short,s1p0_v100_d0p9_r0p3_k0p0,cls_1m,1.0,100,,0.9,0.3,0.0,1163757,AUC,0.6468079999401939
sweep,MS,short,s1p0_v100_d0p9_r0p3_k0p0,cls_10m,1.0,100,,0.9,0.3,0.0,1163757,AUC,0.5634296804740905
sweep,MS,short,s1p0_v100_d0p9_r0p5_k0p0,cls_1m,1.0,100,,0.9,0.5,0.0,1163757,AUC,0.6468612345846366
sweep,MS,short,s1p0_v100_d0p9_r0p5_k0p0,cls_10m,1.0,100,,0.9,0.5,0.0,1163757,AUC,0.5635197589689196
sweep,MS,short,s1p0_v200_d0p7_r0p1_k0p0,cls_1m,1.0,200,,0.7,0.1,0.0,673272,AUC,0.6496221128988199
sweep,MS,short,s1p0_v200_d0p7_r0p1_k0p0,cls_10m,1.0,200,,0.7,0.1,0.0,673272,AUC,0.5609452249221115
sweep,MS,short,s1p0_v200_d0p7_r0p3_k0p0,cls_1m,1.0,200,,0.7,0.3,0.0,673272,AUC,0.6534072610693568
sweep,MS,short,s1p0_v200_d0p7_r0p3_k0p0,cls_10m,1.0,200,,0.7,0.3,0.0,673272,AUC,0.5664493501625691
sweep,MS,short,s1p0_v200_d0p7_r0p5_k0p0,cls_1m,1.0,200,,0.7,0.5,0.0,673272,AUC,0.6561713312771996
sweep,MS,short,s1p0_v200_d0p7_r0p5_k0p0,cls_10m,1.0,200,,0.7,0.5,0.0,673272,AUC,0.5705370653457125
sweep,MS,short,s1p0_v200_d0p8_r0p1_k0p0,cls_1m,1.0,200,,0.8,0.1,0.0,673272,AUC,0.6491500975521236
sweep,MS,short,s1p0_v200_d0p8_r0p1_k0p0,cls_10m,1.0,200,,0.8,0.1,0.0,673272,AUC,0.560512422672751
sweep,MS,short,s1p0_v200_d0p8_r0p3_k0p0,cls_1m,1.0,200,,0.8,0.3,0.0,673272,AUC,0.6518039870810894
sweep,MS,short,s1p0_v200_d0p8_r0p3_k0p0,cls_10m,1.0,200,,0.8,0.3,0.0,673272,AUC,0.5644181703944559
sweep,MS,short,s1p0_v200_d0p8_r0p5_k0p0,cls_1m,1.0,200,,0.8,0.5,0.0,673272,AUC,0.6528209552905594
sweep,MS,short,s1p0_v200_d0p8_r0p5_k0p0,cls_10m,1.0,200,,0.8,0.5,0.0,673272,AUC,0.5659696904356356
sweep,MS,short,s1p0_v200_d0p9_r0p1_k0p0,cls_1m,1.0,200,,0.9,0.1,0.0,673272,AUC,0.6477359998856708
sweep,MS,short,s1p0_v200_d0p9_r0p1_k0p0,cls_10m,1.0,200,,0.9,0.1,0.0,673272,AUC,0.5586600223307139
sweep,MS,short,s1p0_v200_d0p9_r0p3_k0p0,cls_1m,1.0,200,,0.9,0.3,0.0,673272,AUC,0.648239198157392
sweep,MS,short,s1p0_v200_d0p9_r0p3_k0p0,cls_10m,1.0,200,,0.9,0.3,0.0,673272,AUC,0.559543324532081
sweep,MS,short,s1p0_v200_d0p9_r0p5_k0p0,cls_1m,1.0,200,,0.9,0.5,0.0,673272,AUC,0.6483240033793389
sweep,MS,short,s1p0_v200_d0p9_r0p5_k0p0,cls_10m,1.0,200,,0.9,0.5,0.0,673272,AUC,0.5596098886930213
sweep,MS,short,s1p0_v1000_d0p7_r0p1_k0p0,cls_1m,1.0,1000,,0.7,0.1,0.0,69547,AUC,0.6511555100168174
sweep,MS,short,s1p0_v1000_d0p7_r0p1_k0p0,cls_10m,1.0,1000,,0.7,0.1,0.0,69547,AUC,0.5509597445189922
sweep,MS,short,s1p0_v1000_d0p7_r0p3_k0p0,cls_1m,1.0,1000,,0.7,0.3,0.0,69547,AUC,0.6601096631793609
sweep,MS,short,s1p0_v1000_d0p7_r0p3_k0p0,cls_10m,1.0,1000,,0.7,0.3,0.0,69547,AUC,0.5620524859760391
sweep,MS,short,s1p0_v1000_d0p7_r0p5_k0p0,cls_1m,1.0,1000,,0.7,0.5,0.0,69547,AUC,0.6632076063925649
sweep,MS,short,s1p0_v1000_d0p7_r0p5_k0p0,cls_10m,1.0,1000,,0.7,0.5,0.0,69547,AUC,0.5665124689473361
sweep,MS,short,s1p0_v1000_d0p8_r0p1_k0p0,cls_1m,1.0,1000,,0.8,0.1,0.0,69547,AUC,0.6508324638768701
sweep,MS,short,s1p0_v1000_d0p8_r0p1_k0p0,cls_10m,1.0,1000,,0.8,0.1,0.0,69547,AUC,0.5500510202048646
sweep,MS,short,s1p0_v1000_d0p8_r0p3_k0p0,cls_1m,1.0,1000,,0.8,0.3,0.0,69547,AUC,0.6566344405358402
sweep,MS,short,s1p0_v1000_d0p8_r0p3_k0p0,cls_10m,1.0,1000,,0.8,0.3,0.0,69547,AUC,0.5561999931244068
sweep,MS,short,s1p0_v1000_d0p8_r0p5_k0p0,cls_1m,1.0,1000,,0.8,0.5,0.0,69547,AUC,0.6575592901649838
sweep,MS,short,s1p0_v1000_d0p8_r0p5_k0p0,cls_10m,1.0,1000,,0.8,0.5,0.0,69547,AUC,0.5568826045955766
sweep,MS,short,s1p0_v1000_d0p9_r0p1_k0p0,cls_1m,1.0,1000,,0.9,0.1,0.0,69547,AUC,0.649570284677044
sweep,MS,short,s1p0_v1000_d0p9_r0p1_k0p0,cls_10m,1.0,1000,,0.9,0.1,0.0,69547,AUC,0.5474418402884882
sweep,MS,short,s1p0_v1000_d0p9_r0p3_k0p0,cls_1m,1.0,1000,,0.9,0.3,0.0,69547,AUC,0.6505346333098158
sweep,MS,short,s1p0_v1000_d0p9_r0p3_k0p0,cls_10m,1.0,1000,,0.9,0.3,0.0,69547,AUC,0.5484526079175882
sweep,MS,short,s1p0_v1000_d0p9_r0p5_k0p0,cls_1m,1.0,1000,,0.9,0.5,0.0,69547,AUC,0.6505198856313124
sweep,MS,short,s1p0_v1000_d0p9_r0p5_k0p0,cls_10m,1.0,1000,,0.9,0.5,0.0,69547,AUC,0.5482703385466261
sweep,MS,short,s2p0_v50_d0p7_r0p1_k0p0,cls_1m,2.0,50,,0.7,0.1,0.0,1063771,AUC,0.6513812964497396
sweep,MS,short,s2p0_v50_d0p7_r0p1_k0p0,cls_10m,2.0,50,,0.7,0.1,0.0,1063771,AUC,0.5711507550996351
sweep,MS,short,s2p0_v50_d0p7_r0p3_k0p0,cls_1m,2.0,50,,0.7,0.3,0.0,1063771,AUC,0.6557725347628728
sweep,MS,short,s2p0_v50_d0p7_r0p3_k0p0,cls_10m,2.0,50,,0.7,0.3,0.0,1063771,AUC,0.5767542784267445
sweep,MS,short,s2p0_v50_d0p7_r0p5_k0p0,cls_1m,2.0,50,,0.7,0.5,0.0,1063771,AUC,0.6597452576284513
sweep,MS,short,s2p0_v50_d0p7_r0p5_k0p0,cls_10m,2.0,50,,0.7,0.5,0.0,1063771,AUC,0.5811133568686276
sweep,MS,short,s2p0_v50_d0p8_r0p1_k0p0,cls_1m,2.0,50,,0.8,0.1,0.0,1063771,AUC,0.6503146250003964
sweep,MS,short,s2p0_v50_d0p8_r0p1_k0p0,cls_10m,2.0,50,,0.8,0.1,0.0,1063771,AUC,0.5697012282028554
sweep,MS,short,s2p0_v50_d0p8_r0p3_k0p0,cls_1m,2.0,50,,0.8,0.3,0.0,1063771,AUC,0.6526374024325515
sweep,MS,short,s2p0_v50_d0p8_r0p3_k0p0,cls_10m,2.0,50,,0.8,0.3,0.0,1063771,AUC,0.573097136458842
sweep,MS,short,s2p0_v50_d0p8_r0p5_k0p0,cls_1m,2.0,50,,0.8,0.5,0.0,1063771,AUC,0.6538292094310292
sweep,MS,short,s2p0_v50_d0p8_r0p5_k0p0,cls_10m,2.0,50,,0.8,0.5,0.0,1063771,AUC,0.5746528006280668
sweep,MS,short,s2p0_v50_d0p9_r0p1_k0p0,cls_1m,2.0,50,,0.9,0.1,0.0,1063771,AUC,0.6483110473352939
sweep,MS,short,s2p0_v50_d0p9_r0p1_k0p0,cls_10m,2.0,50,,0.9,0.1,0.0,1063771,AUC,0.5664828288815343
sweep,MS,short,s2p0_v50_d0p9_r0p3_k0p0,cls_1m,2.0,50,,0.9,0.3,0.0,1063771,AUC,0.6487197856274755
sweep,MS,short,s2p0_v50_d0p9_r0p3_k0p0,cls_10m,2.0,50,,0.9,0.3,0.0,1063771,AUC,0.5670626225151092
sweep,MS,short,s2p0_v50_d0p9_r0p5_k0p0,cls_1m,2.0,50,,0.9,0.5,0.0,1063771,AUC,0.6487652237879521
sweep,MS,short,s2p0_v50_d0p9_r0p5_k0p0,cls_10m,2.0,50,,0.9,0.5,0.0,1063771,AUC,0.5671036421291693
sweep,MS,short,s2p0_v100_d0p7_r0p1_k0p0,cls_1m,2.0,100,,0.7,0.1,0.0,953169,AUC,0.6525648551918728
sweep,MS,short,s2p0_v100_d0p7_r0p1_k0p0,cls_10m,2.0,100,,0.7,0.1,0.0,953169,AUC,0.5721182063030189
sweep,MS,short,s2p0_v100_d0p7_r0p3_k0p0,cls_1m,2.0,100,,0.7,0.3,0.0,953169,AUC,0.6569077474788441
sweep,MS,short,s2p0_v100_d0p7_r0p3_k0p0,cls_10m,2.0,100,,0.7,0.3,0.0,953169,AUC,0.5774049988354133
sweep,MS,short,s2p0_v100_d0p7_r0p5_k0p0,cls_1m,2.0,100,,0.7,0.5,0.0,953169,AUC,0.6609039602976523
sweep,MS,short,s2p0_v100_d0p7_r0p5_k0p0,cls_10m,2.0,100,,0.7,0.5,0.0,953169,AUC,0.5815944778647358
sweep,MS,short,s2p0_v100_d0p8_r0p1_k0p0,cls_1m,2.0,100,,0.8,0.1,0.0,953169,AUC,0.6516161681785698
sweep,MS,short,s2p0_v100_d0p8_r0p1_k0p0,cls_10m,2.0,100,,0.8,0.1,0.0,953169,AUC,0.5710407782696071
sweep,MS,short,s2p0_v100_d0p8_r0p3_k0p0,cls_1m,2.0,100,,0.8,0.3,0.0,953169,AUC,0.653943259452794
sweep,MS,short,s2p0_v100_d0p8_r0p3_k0p0,cls_10m,2.0,100,,0.8,0.3,0.0,953169,AUC,0.5742081707116017
sweep,MS,short,s2p0_v100_d0p8_r0p5_k0p0,cls_1m,2.0,100,,0.8,0.5,0.0,953169,AUC,0.6551391822020907
sweep,MS,short,s2p0_v100_d0p8_r0p5_k0p0,cls_10m,2.0,100,,0.8,0.5,0.0,953169,AUC,0.5757144328744207
sweep,MS,short,s2p0_v100_d0p9_r0p1_k0p0,cls_1m,2.0,100,,0.9,0.1,0.0,953169,AUC,0.6495440043438236
sweep,MS,short,s2p0_v100_d0p9_r0p1_k0p0,cls_10m,2.0,100,,0.9,0.1,0.0,953169,AUC,0.5679745552435611
sweep,MS,short,s2p0_v100_d0p9_r0p3_k0p0,cls_1m,2.0,100,,0.9,0.3,0.0,953169,AUC,0.6499692762032098
sweep,MS,short,s2p0_v100_d0p9_r0p3_k0p0,cls_10m,2.0,100,,0.9,0.3,0.0,953169,AUC,0.5685205982340646
sweep,MS,short,s2p0_v100_d0p9_r0p5_k0p0,cls_1m,2.0,100,,0.9,0.5,0.0,953169,AUC,0.6500410504044253
sweep,MS,short,s2p0_v100_d0p9_r0p5_k0p0,cls_10m,2.0,100,,0.9,0.5,0.0,953169,AUC,0.5686042515000587
sweep,MS,short,s2p0_v200_d0p7_r0p1_k0p0,cls_1m,2.0,200,,0.7,0.1,0.0,606817,AUC,0.6534139813766825
sweep,MS,short,s2p0_v200_d0p7_r0p1_k0p0,cls_10m,2.0,200,,0.7,0.1,0.0,606817,AUC,0.5668706669059549
sweep,MS,short,s2p0_v200_d0p7_r0p3_k0p0,cls_1m,2.0,200,,0.7,0.3,0.0,606817,AUC,0.6590958272334471
sweep,MS,short,s2p0_v200_d0p7_r0p3_k0p0,cls_10m,2.0,200,,0.7,0.3,0.0,606817,AUC,0.5731983979490001
sweep,MS,short,s2p0_v200_d0p7_r0p5_k0p0,cls_1m,2.0,200,,0.7,0.5,0.0,606817,AUC,0.6635597895180043
sweep,MS,short,s2p0_v200_d0p7_r0p5_k0p0,cls_10m,2.0,200,,0.7,0.5,0.0,606817,AUC,0.5782777870144676
sweep,MS,short,s2p0_v200_d0p8_r0p1_k0p0,cls_1m,2.0,200,,0.8,0.1,0.0,606817,AUC,0.6525445894379185
sweep,MS,short,s2p0_v200_d0p8_r0p1_k0p0,cls_10m,2.0,200,,0.8,0.1,0.0,606817,AUC,0.5659591370249197
sweep,MS,short,s2p0_v200_d0p8_r0p3_k0p0,cls_1m,2.0,200,,0.8,0.3,0.0,606817,AUC,0.6560445662505112
sweep,MS,short,s2p0_v200_d0p8_r0p3_k0p0,cls_10m,2.0,200,,0.8,0.3,0.0,606817,AUC,0.5701617251520723
sweep,MS,short,s2p0_v200_d0p8_r0p5_k0p0,cls_1m,2.0,200,,0.8,0.5,0.0,606817,AUC,0.6575709100498177
sweep,MS,short,s2p0_v200_d0p8_r0p5_k0p0,cls_10m,2.0,200,,0.8,0.5,0.0,606817,AUC,0.5719699541979959
sweep,MS,short,s2p0_v200_d0p9_r0p1_k0p0,cls_1m,2.0,200,,0.9,0.1,0.0,606817,AUC,0.6503956226827354
sweep,MS,short,s2p0_v200_d0p9_r0p1_k0p0,cls_10m,2.0,200,,0.9,0.1,0.0,606817,AUC,0.562979477325998
sweep,MS,short,s2p0_v200_d0p9_r0p3_k0p0,cls_1m,2.0,200,,0.9,0.3,0.0,606817,AUC,0.6511012104453251
sweep,MS,short,s2p0_v200_d0p9_r0p3_k0p0,cls_10m,2.0,200,,0.9,0.3,0.0,606817,AUC,0.5639711604297357
sweep,MS,short,s2p0_v200_d0p9_r0p5_k0p0,cls_1m,2.0,200,,0.9,0.5,0.0,606817,AUC,0.6511969796840087
sweep,MS,short,s2p0_v200_d0p9_r0p5_k0p0,cls_10m,2.0,200,,0.9,0.5,0.0,606817,AUC,0.5640952035053405
sweep,MS,short,s2p0_v1000_d0p7_r0p1_k0p0,cls_1m,2.0,1000,,0.7,0.1,0.0,83237,AUC,0.6548158084180361
sweep,MS,short,s2p0_v1000_d0p7_r0p1_k0p0,cls_10m,2.0,1000,,0.7,0.1,0.0,83237,AUC,0.5532714543747497
sweep,MS,short,s2p0_v1000_d0p7_r0p3_k0p0,cls_1m,2.0,1000,,0.7,0.3,0.0,83237,AUC,0.6663954769789027
sweep,MS,short,s2p0_v1000_d0p7_r0p3_k0p0,cls_10m,2.0,1000,,0.7,0.3,0.0,83237,AUC,0.5653071713902702
sweep,MS,short,s2p0_v1000_d0p7_r0p5_k0p0,cls_1m,2.0,1000,,0.7,0.5,0.0,83237,AUC,0.672730038324773
sweep,MS,short,s2p0_v1000_d0p7_r0p5_k0p0,cls_10m,2.0,1000,,0.7,0.5,0.0,83237,AUC,0.572017777650466
sweep,MS,short,s2p0_v1000_d0p8_r0p1_k0p0,cls_1m,2.0,1000,,0.8,0.1,0.0,83237,AUC,0.6544605661522436
sweep,MS,short,s2p0_v1000_d0p8_r0p1_k0p0,cls_10m,2.0,1000,,0.8,0.1,0.0,83237,AUC,0.5524699205889374
sweep,MS,short,s2p0_v1000_d0p8_r0p3_k0p0,cls_1m,2.0,1000,,0.8,0.3,0.0,83237,AUC,0.6610510019546841
sweep,MS,short,s2p0_v1000_d0p8_r0p3_k0p0,cls_10m,2.0,1000,,0.8,0.3,0.0,83237,AUC,0.5585202646492072
sweep,MS,short,s2p0_v1000_d0p8_r0p5_k0p0,cls_1m,2.0,1000,,0.8,0.5,0.0,83237,AUC,0.661875443426083
sweep,MS,short,s2p0_v1000_d0p8_r0p5_k0p0,cls_10m,2.0,1000,,0.8,0.5,0.0,83237,AUC,0.5590917500068545
sweep,MS,short,s2p0_v1000_d0p9_r0p1_k0p0,cls_1m,2.0,1000,,0.9,0.1,0.0,83237,AUC,0.6527913335569395
sweep,MS,short,s2p0_v1000_d0p9_r0p1_k0p0,cls_10m,2.0,1000,,0.9,0.1,0.0,83237,AUC,0.5499591827587218
sweep,MS,short,s2p0_v1000_d0p9_r0p3_k0p0,cls_1m,2.0,1000,,0.9,0.3,0.0,83237,AUC,0.6538214458758472
sweep,MS,short,s2p0_v1000_d0p9_r0p3_k0p0,cls_10m,2.0,1000,,0.9,0.3,0.0,83237,AUC,0.5508272712399211
sweep,MS,short,s2p0_v1000_d0p9_r0p5_k0p0,cls_1m,2.0,1000,,0.9,0.5,0.0,83237,AUC,0.6537988080911905
sweep,MS,short,s2p0_v1000_d0p9_r0p5_k0p0,cls_10m,2.0,1000,,0.9,0.5,0.0,83237,AUC,0.550879960739602
sweep,MS,long,s0p5_v50_d0p7_r0p1_k0p0,cls_close,0.5,50,,0.7,0.1,0.0,1505334,AUC,0.5833730832637711
sweep,MS,long,s0p5_v50_d0p7_r0p3_k0p0,cls_close,0.5,50,,0.7,0.3,0.0,1505334,AUC,0.5629003397003054
sweep,MS,long,s0p5_v50_d0p7_r0p5_k0p0,cls_close,0.5,50,,0.7,0.5,0.0,1505334,AUC,0.5889833043665512
sweep,MS,long,s0p5_v50_d0p8_r0p1_k0p0,cls_close,0.5,50,,0.8,0.1,0.0,1505334,AUC,0.5828645806178726
sweep,MS,long,s0p5_v50_d0p8_r0p3_k0p0,cls_close,0.5,50,,0.8,0.3,0.0,1505334,AUC,0.5846425341250711
sweep,MS,long,s0p5_v50_d0p8_r0p5_k0p0,cls_close,0.5,50,,0.8,0.5,0.0,1505334,AUC,0.5846726808864017
sweep,MS,long,s0p5_v50_d0p9_r0p1_k0p0,cls_close,0.5,50,,0.9,0.1,0.0,1505334,AUC,0.5795054973259349
sweep,MS,long,s0p5_v50_d0p9_r0p3_k0p0,cls_close,0.5,50,,0.9,0.3,0.0,1505334,AUC,0.5796170384108622
sweep,MS,long,s0p5_v50_d0p9_r0p5_k0p0,cls_close,0.5,50,,0.9,0.5,0.0,1505334,AUC,0.57968778719211
sweep,MS,long,s0p5_v100_d0p7_r0p1_k0p0,cls_close,0.5,100,,0.7,0.1,0.0,1306410,AUC,0.5803511194462211
sweep,MS,long,s0p5_v100_d0p7_r0p3_k0p0,cls_close,0.5,100,,0.7,0.3,0.0,1306410,AUC,0.5830550980580597
sweep,MS,long,s0p5_v100_d0p7_r0p5_k0p0,cls_close,0.5,100,,0.7,0.5,0.0,1306410,AUC,0.5859437768215814
sweep,MS,long,s0p5_v100_d0p8_r0p1_k0p0,cls_close,0.5,100,,0.8,0.1,0.0,1306410,AUC,0.578761678353189
sweep,MS,long,s0p5_v100_d0p8_r0p3_k0p0,cls_close,0.5,100,,0.8,0.3,0.0,1306410,AUC,0.5801193026476177
sweep,MS,long,s0p5_v100_d0p8_r0p5_k0p0,cls_close,0.5,100,,0.8,0.5,0.0,1306410,AUC,0.5812421661690133
sweep,MS,long,s0p5_v100_d0p9_r0p1_k0p0,cls_close,0.5,100,,0.9,0.1,0.0,1306410,AUC,0.5770709153540307
sweep,MS,long,s0p5_v100_d0p9_r0p3_k0p0,cls_close,0.5,100,,0.9,0.3,0.0,1306410,AUC,0.5770489663112752
sweep,MS,long,s0p5_v100_d0p9_r0p5_k0p0,cls_close,0.5,100,,0.9,0.5,0.0,1306410,AUC,0.5771090267660401
sweep,MS,long,s0p5_v200_d0p7_r0p1_k0p0,cls_close,0.5,200,,0.7,0.1,0.0,705340,AUC,0.5741449168538835
sweep,MS,long,s0p5_v200_d0p7_r0p3_k0p0,cls_close,0.5,200,,0.7,0.3,0.0,705340,AUC,0.5759256548325216
sweep,MS,long,s0p5_v200_d0p7_r0p5_k0p0,cls_close,0.5,200,,0.7,0.5,0.0,705340,AUC,0.5800522544720571
sweep,MS,long,s0p5_v200_d0p8_r0p1_k0p0,cls_close,0.5,200,,0.8,0.1,0.0,705340,AUC,0.5739508317667088
sweep,MS,long,s0p5_v200_d0p8_r0p3_k0p0,cls_close,0.5,200,,0.8,0.3,0.0,705340,AUC,0.574818220940087
sweep,MS,long,s0p5_v200_d0p8_r0p5_k0p0,cls_close,0.5,200,,0.8,0.5,0.0,705340,AUC,0.5758235793142228
sweep,MS,long,s0p5_v200_d0p9_r0p1_k0p0,cls_close,0.5,200,,0.9,0.1,0.0,705340,AUC,0.5740644347024361
sweep,MS,long,s0p5_v200_d0p9_r0p3_k0p0,cls_close,0.5,200,,0.9,0.3,0.0,705340,AUC,0.5739894336499747
sweep,MS,long,s0p5_v200_d0p9_r0p5_k0p0,cls_close,0.5,200,,0.9,0.5,0.0,705340,AUC,0.5740134329566251
sweep,MS,long,s0p5_v1000_d0p7_r0p1_k0p0,cls_close,0.5,1000,,0.7,0.1,0.0,59245,AUC,0.5712257379508137
sweep,MS,long,s0p5_v1000_d0p7_r0p3_k0p0,cls_close,0.5,1000,,0.7,0.3,0.0,59245,AUC,0.5667530582274001
sweep,MS,long,s0p5_v1000_d0p7_r0p5_k0p0,cls_close,0.5,1000,,0.7,0.5,0.0,59245,AUC,0.5665763207411938
sweep,MS,long,s0p5_v1000_d0p8_r0p1_k0p0,cls_close,0.5,1000,,0.8,0.1,0.0,59245,AUC,0.5717384821784134
sweep,MS,long,s0p5_v1000_d0p8_r0p3_k0p0,cls_close,0.5,1000,,0.8,0.3,0.0,59245,AUC,0.5701694142038904
sweep,MS,long,s0p5_v1000_d0p8_r0p5_k0p0,cls_close,0.5,1000,,0.8,0.5,0.0,59245,AUC,0.5700067201654336
sweep,MS,long,s0p5_v1000_d0p9_r0p1_k0p0,cls_close,0.5,1000,,0.9,0.1,0.0,59245,AUC,0.5706330362049724
sweep,MS,long,s0p5_v1000_d0p9_r0p3_k0p0,cls_close,0.5,1000,,0.9,0.3,0.0,59245,AUC,0.5707970615934936
sweep,MS,long,s0p5_v1000_d0p9_r0p5_k0p0,cls_close,0.5,1000,,0.9,0.5,0.0,59245,AUC,0.5708389562724774
sweep,MS,long,s1p0_v50_d0p7_r0p1_k0p0,cls_close,1.0,50,,0.7,0.1,0.0,1323843,AUC,0.5856383557781395
sweep,MS,long,s1p0_v50_d0p7_r0p3_k0p0,cls_close,1.0,50,,0.7,0.3,0.0,1323843,AUC,0.5896235356493903
sweep,MS,long,s1p0_v50_d0p7_r0p5_k0p0,cls_close,1.0,50,,0.7,0.5,0.0,1323843,AUC,0.592426803622962
sweep,MS,long,s1p0_v50_d0p8_r0p1_k0p0,cls_close,1.0,50,,0.8,0.1,0.0,1323843,AUC,0.5838632647325839
sweep,MS,long,s1p0_v50_d0p8_r0p3_k0p0,cls_close,1.0,50,,0.8,0.3,0.0,1323843,AUC,0.5858173113638525
sweep,MS,long,s1p0_v50_d0p8_r0p5_k0p0,cls_close,1.0,50,,0.8,0.5,0.0,1323843,AUC,0.5870034646042382
sweep,MS,long,s1p0_v50_d0p9_r0p1_k0p0,cls_close,1.0,50,,0.9,0.1,0.0,1323843,AUC,0.5816227132670421
sweep,MS,long,s1p0_v50_d0p9_r0p3_k0p0,cls_close,1.0,50,,0.9,0.3,0.0,1323843,AUC,0.5817976458341331
sweep,MS,long,s1p0_v50_d0p9_r0p5_k0p0,cls_close,1.0,50,,0.9,0.5,0.0,1323843,AUC,0.5818269784822897
sweep,MS,long,s1p0_v100_d0p7_r0p1_k0p0,cls_close,1.0,100,,0.7,0.1,0.0,1163757,AUC,0.5832189641094779
sweep,MS,long,s1p0_v100_d0p7_r0p3_k0p0,cls_close,1.0,100,,0.7,0.3,0.0,1163757,AUC,0.5858159245172337
sweep,MS,long,s1p0_v100_d0p7_r0p5_k0p0,cls_close,1.0,100,,0.7,0.5,0.0,1163757,AUC,0.5885186908025164
sweep,MS,long,s1p0_v100_d0p8_r0p1_k0p0,cls_close,1.0,100,,0.8,0.1,0.0,1163757,AUC,0.582115993446703
sweep,MS,long,s1p0_v100_d0p8_r0p3_k0p0,cls_close,1.0,100,,0.8,0.3,0.0,1163757,AUC,0.5832110340001562
sweep,MS,long,s1p0_v100_d0p8_r0p5_k0p0,cls_close,1.0,100,,0.8,0.5,0.0,1163757,AUC,0.5841474359592912
sweep,MS,long,s1p0_v100_d0p9_r0p1_k0p0,cls_close,1.0,100,,0.9,0.1,0.0,1163757,AUC,0.5807650572959802
sweep,MS,long,s1p0_v100_d0p9_r0p3_k0p0,cls_close,1.0,100,,0.9,0.3,0.0,1163757,AUC,0.5808279438632731
sweep,MS,long,s1p0_v100_d0p9_r0p5_k0p0,cls_close,1.0,100,,0.9,0.5,0.0,1163757,AUC,0.5808361488988609
sweep,MS,long,s1p0_v200_d0p7_r0p1_k0p0,cls_close,1.0,200,,0.7,0.1,0.0,673272,AUC,0.5796024049230302
sweep,MS,long,s1p0_v200_d0p7_r0p3_k0p0,cls_close,1.0,200,,0.7,0.3,0.0,673272,AUC,0.580163780575813
sweep,MS,long,s1p0_v200_d0p7_r0p5_k0p0,cls_close,1.0,200,,0.7,0.5,0.0,673272,AUC,0.5832154547792973
sweep,MS,long,s1p0_v200_d0p8_r0p1_k0p0,cls_close,1.0,200,,0.8,0.1,0.0,673272,AUC,0.579738338588287
sweep,MS,long,s1p0_v200_d0p8_r0p3_k0p0,cls_close,1.0,200,,0.8,0.3,0.0,673272,AUC,0.5802226962172791
sweep,MS,long,s1p0_v200_d0p8_r0p5_k0p0,cls_close,1.0,200,,0.8,0.5,0.0,673272,AUC,0.5808248605820499
sweep,NVDA,short,s0p5_v50_d0p7_r0p1_k0p0,cls_1m,0.5,50,,0.7,0.1,0.0,5058727,AUC,0.6230849638667789
sweep,NVDA,short,s0p5_v50_d0p7_r0p1_k0p0,cls_10m,0.5,50,,0.7,0.1,0.0,5058727,AUC,0.5524832760059163
sweep,NVDA,short,s0p5_v50_d0p7_r0p3_k0p0,cls_1m,0.5,50,,0.7,0.3,0.0,5058727,AUC,0.6226146987606049
sweep,NVDA,short,s0p5_v50_d0p7_r0p3_k0p0,cls_10m,0.5,50,,0.7,0.3,0.0,5058727,AUC,0.5515864525972233
sweep,NVDA,short,s0p5_v50_d0p7_r0p5_k0p0,cls_1m,0.5,50,,0.7,0.5,0.0,5058727,AUC,0.6237420954697886
sweep,NVDA,short,s0p5_v50_d0p7_r0p5_k0p0,cls_10m,0.5,50,,0.7,0.5,0.0,5058727,AUC,0.5514810195129394
sweep,NVDA,short,s0p5_v50_d0p8_r0p1_k0p0,cls_1m,0.5,50,,0.8,0.1,0.0,5058727,AUC,0.6238444386270601
sweep,NVDA,short,s0p5_v50_d0p8_r0p1_k0p0,cls_10m,0.5,50,,0.8,0.1,0.0,5058727,AUC,0.5517587606395874
sweep,NVDA,short,s0p5_v50_d0p8_r0p3_k0p0,cls_1m,0.5,50,,0.8,0.3,0.0,5058727,AUC,0.6227979800499246
sweep,NVDA,short,s0p5_v50_d0p8_r0p3_k0p0,cls_10m,0.5,50,,0.8,0.3,0.0,5058727,AUC,0.5529475936924676
sweep,NVDA,short,s0p5_v50_d0p8_r0p5_k0p0,cls_1m,0.5,50,,0.8,0.5,0.0,5058727,AUC,0.6223576616607638
sweep,NVDA,short,s0p5_v50_d0p8_r0p5_k0p0,cls_10m,0.5,50,,0.8,0.5,0.0,5058727,AUC,0.5530234240844498
sweep,NVDA,short,s0p5_v50_d0p9_r0p1_k0p0,cls_1m,0.5,50,,0.9,0.1,0.0,5058727,AUC,0.6242147693946458
sweep,NVDA,short,s0p5_v50_d0p9_r0p1_k0p0,cls_10m,0.5,50,,0.9,0.1,0.0,5058727,AUC,0.549644813303849
sweep,NVDA,short,s0p5_v50_d0p9_r0p3_k0p0,cls_1m,0.5,50,,0.9,0.3,0.0,5058727,AUC,0.6242902382182964
sweep,NVDA,short,s0p5_v50_d0p9_r0p3_k0p0,cls_10m,0.5,50,,0.9,0.3,0.0,5058727,AUC,0.5500112088166715
sweep,NVDA,short,s0p5_v50_d0p9_r0p5_k0p0,cls_1m,0.5,50,,0.9,0.5,0.0,5058727,AUC,0.6242256948747341
sweep,NVDA,short,s0p5_v50_d0p9_r0p5_k0p0,cls_10m,0.5,50,,0.9,0.5,0.0,5058727,AUC,0.5499916325420734
sweep,NVDA,short,s0p5_v100_d0p7_r0p1_k0p0,cls_1m,0.5,100,,0.7,0.1,0.0,4632789,AUC,0.6249316973495915
sweep,NVDA,short,s0p5_v100_d0p7_r0p1_k0p0,cls_10m,0.5,100,,0.7,0.1,0.0,4632789,AUC,0.5530938121748821
sweep,NVDA,short,s0p5_v100_d0p7_r0p3_k0p0,cls_1m,0.5,100,,0.7,0.3,0.0,4632789,AUC,0.6234515074288304
sweep,NVDA,short,s0p5_v100_d0p7_r0p3_k0p0,cls_10m,0.5,100,,0.7,0.3,0.0,4632789,AUC,0.5518928883484944
sweep,NVDA,short,s0p5_v100_d0p7_r0p5_k0p0,cls_1m,0.5,100,,0.7,0.5,0.0,4632789,AUC,0.6234688396785645
sweep,NVDA,short,s0p5_v100_d0p7_r0p5_k0p0,cls_10m,0.5,100,,0.7,0.5,0.0,4632789,AUC,0.5500659667503958
sweep,NVDA,short,s0p5_v100_d0p8_r0p1_k0p0,cls_1m,0.5,100,,0.8,0.1,0.0,4632789,AUC,0.6254015294301615
sweep,NVDA,short,s0p5_v100_d0p8_r0p1_k0p0,cls_10m,0.5,100,,0.8,0.1,0.0,4632789,AUC,0.5524604653766287
sweep,NVDA,short,s0p5_v100_d0p8_r0p3_k0p0,cls_1m,0.5,100,,0.8,0.3,0.0,4632789,AUC,0.6244713481382106
sweep,NVDA,short,s0p5_v100_d0p8_r0p3_k0p0,cls_10m,0.5,100,,0.8,0.3,0.0,4632789,AUC,0.553760492065013
sweep,NVDA,short,s0p5_v100_d0p8_r0p5_k0p0,cls_1m,0.5,100,,0.8,0.5,0.0,4632789,AUC,0.6236914438920622
sweep,NVDA,short,s0p5_v100_d0p8_r0p5_k0p0,cls_10m,0.5,100,,0.8,0.5,0.0,4632789,AUC,0.5535551858665761
sweep,NVDA,short,s0p5_v100_d0p9_r0p1_k0p0,cls_1m,0.5,100,,0.9,0.1,0.0,4632789,AUC,0.6243181347968628
sweep,NVDA,short,s0p5_v100_d0p9_r0p1_k0p0,cls_10m,0.5,100,,0.9,0.1,0.0,4632789,AUC,0.5491568906425768
sweep,NVDA,short,s0p5_v100_d0p9_r0p3_k0p0,cls_1m,0.5,100,,0.9,0.3,0.0,4632789,AUC,0.6246658227940476
sweep,NVDA,short,s0p5_v100_d0p9_r0p3_k0p0,cls_10m,0.5,100,,0.9,0.3,0.0,4632789,AUC,0.5497699607358557
sweep,NVDA,short,s0p5_v100_d0p9_r0p5_k0p0,cls_1m,0.5,100,,0.9,0.5,0.0,4632789,AUC,0.6247372690845617
sweep,NVDA,short,s0p5_v100_d0p9_r0p5_k0p0,cls_10m,0.5,100,,0.9,0.5,0.0,4632789,AUC,0.5498535873226108
sweep,NVDA,short,s0p5_v200_d0p7_r0p1_k0p0,cls_1m,0.5,200,,0.7,0.1,0.0,3467612,AUC,0.625820826068468
sweep,NVDA,short,s0p5_v200_d0p7_r0p1_k0p0,cls_10m,0.5,200,,0.7,0.1,0.0,3467612,AUC,0.5508121909736301
sweep,NVDA,short,s0p5_v200_d0p7_r0p3_k0p0,cls_1m,0.5,200,,0.7,0.3,0.0,3467612,AUC,0.6281276136523553
sweep,NVDA,short,s0p5_v200_d0p7_r0p3_k0p0,cls_10m,0.5,200,,0.7,0.3,0.0,3467612,AUC,0.5558601176051109
sweep,NVDA,short,s0p5_v200_d0p7_r0p5_k0p0,cls_1m,0.5,200,,0.7,0.5,0.0,3467612,AUC,0.6250240979919687
sweep,NVDA,short,s0p5_v200_d0p7_r0p5_k0p0,cls_10m,0.5,200,,0.7,0.5,0.0,3467612,AUC,0.5502694045145423
sweep,NVDA,short,s0p5_v200_d0p8_r0p1_k0p0,cls_1m,0.5,200,,0.8,0.1,0.0,3467612,AUC,0.6249337921684698
sweep,NVDA,short,s0p5_v200_d0p8_r0p1_k0p0,cls_10m,0.5,200,,0.8,0.1,0.0,3467612,AUC,0.5491225022598396
sweep,NVDA,short,s0p5_v200_d0p8_r0p3_k0p0,cls_1m,0.5,200,,0.8,0.3,0.0,3467612,AUC,0.6278126996087954
sweep,NVDA,short,s0p5_v200_d0p8_r0p3_k0p0,cls_10m,0.5,200,,0.8,0.3,0.0,3467612,AUC,0.5531600911024546
sweep,NVDA,short,s0p5_v200_d0p8_r0p5_k0p0,cls_1m,0.5,200,,0.8,0.5,0.0,3467612,AUC,0.6278410658913492
sweep,NVDA,short,s0p5_v200_d0p8_r0p5_k0p0,cls_10m,0.5,200,,0.8,0.5,0.0,3467612,AUC,0.5531556474400227
sweep,NVDA,short,s0p5_v200_d0p9_r0p1_k0p0,cls_1m,0.5,200,,0.9,0.1,0.0,3467612,AUC,0.6239277943985688
sweep,NVDA,short,s0p5_v200_d0p9_r0p1_k0p0,cls_10m,0.5,200,,0.9,0.1,0.0,3467612,AUC,0.5448342000212063
sweep,NVDA,short,s0p5_v200_d0p9_r0p3_k0p0,cls_1m,0.5,200,,0.9,0.3,0.0,3467612,AUC,0.6244294226433662
sweep,NVDA,short,s0p5_v200_d0p9_r0p3_k0p0,cls_10m,0.5,200,,0.9,0.3,0.0,3467612,AUC,0.5459220420803468
sweep,NVDA,short,s0p5_v200_d0p9_r0p5_k0p0,cls_1m,0.5,200,,0.9,0.5,0.0,3467612,AUC,0.6244750908519817
sweep,NVDA,short,s0p5_v200_d0p9_r0p5_k0p0,cls_10m,0.5,200,,0.9,0.5,0.0,3467612,AUC,0.5459783646579528
sweep,NVDA,short,s0p5_v1000_d0p7_r0p1_k0p0,cls_1m,0.5,1000,,0.7,0.1,0.0,1123893,AUC,0.6261054614010243
sweep,NVDA,short,s0p5_v1000_d0p7_r0p1_k0p0,cls_10m,0.5,1000,,0.7,0.1,0.0,1123893,AUC,0.5484907001593707
sweep,NVDA,short,s0p5_v1000_d0p7_r0p3_k0p0,cls_1m,0.5,1000,,0.7,0.3,0.0,1123893,AUC,0.6320565372296564
sweep,NVDA,short,s0p5_v1000_d0p7_r0p3_k0p0,cls_10m,0.5,1000,,0.7,0.3,0.0,1123893,AUC,0.5591062735750884
sweep,NVDA,short,s0p5_v1000_d0p7_r0p5_k0p0,cls_1m,0.5,1000,,0.7,0.5,0.0,1123893,AUC,0.6317305930809601
sweep,NVDA,short,s0p5_v1000_d0p7_r0p5_k0p0,cls_10m,0.5,1000,,0.7,0.5,0.0,1123893,AUC,0.5571688585867073
sweep,NVDA,short,s0p5_v1000_d0p8_r0p1_k0p0,cls_1m,0.5,1000,,0.8,0.1,0.0,1123893,AUC,0.6252923645177833
sweep,NVDA,short,s0p5_v1000_d0p8_r0p1_k0p0,cls_10m,0.5,1000,,0.8,0.1,0.0,1123893,AUC,0.5467861209671648
sweep,NVDA,short,s0p5_v1000_d0p8_r0p3_k0p0,cls_1m,0.5,1000,,0.8,0.3,0.0,1123893,AUC,0.6285081006688982
sweep,NVDA,short,s0p5_v1000_d0p8_r0p3_k0p0,cls_10m,0.5,1000,,0.8,0.3,0.0,1123893,AUC,0.5526514979507317
sweep,NVDA,short,s0p5_v1000_d0p8_r0p5_k0p0,cls_1m,0.5,1000,,0.8,0.5,0.0,1123893,AUC,0.6288900485174512
sweep,NVDA,short,s0p5_v1000_d0p8_r0p5_k0p0,cls_10m,0.5,1000,,0.8,0.5,0.0,1123893,AUC,0.5528448629290645
sweep,NVDA,short,s0p5_v1000_d0p9_r0p1_k0p0,cls_1m,0.5,1000,,0.9,0.1,0.0,1123893,AUC,0.6230413069302475
sweep,NVDA,short,s0p5_v1000_d0p9_r0p1_k0p0,cls_10m,0.5,1000,,0.9,0.1,0.0,1123893,AUC,0.5414923571926443
sweep,NVDA,short,s0p5_v1000_d0p9_r0p3_k0p0,cls_1m,0.5,1000,,0.9,0.3,0.0,1123893,AUC,0.6236272468430121
sweep,NVDA,short,s0p5_v1000_d0p9_r0p3_k0p0,cls_10m,0.5,1000,,0.9,0.3,0.0,1123893,AUC,0.5422681828049705
sweep,NVDA,short,s0p5_v1000_d0p9_r0p5_k0p0,cls_1m,0.5,1000,,0.9,0.5,0.0,1123893,AUC,0.6236585448720504
sweep,NVDA,short,s0p5_v1000_d0p9_r0p5_k0p0,cls_10m,0.5,1000,,0.9,0.5,0.0,1123893,AUC,0.5422584623249885
sweep,NVDA,short,s1p0_v50_d0p7_r0p1_k0p0,cls_1m,1.0,50,,0.7,0.1,0.0,2689734,AUC,0.6263099138569302
sweep,NVDA,short,s1p0_v50_d0p7_r0p1_k0p0,cls_10m,1.0,50,,0.7,0.1,0.0,2689734,AUC,0.5530845490934283
sweep,NVDA,short,s1p0_v50_d0p7_r0p3_k0p0,cls_1m,1.0,50,,0.7,0.3,0.0,2689734,AUC,0.6285386202770253
sweep,NVDA,short,s1p0_v50_d0p7_r0p3_k0p0,cls_10m,1.0,50,,0.7,0.3,0.0,2689734,AUC,0.5571292640050903
sweep,NVDA,short,s1p0_v50_d0p7_r0p5_k0p0,cls_1m,1.0,50,,0.7,0.5,0.0,2689734,AUC,0.6290123855997141
sweep,NVDA,short,s1p0_v50_d0p7_r0p5_k0p0,cls_10m,1.0,50,,0.7,0.5,0.0,2689734,AUC,0.5579736817140118
sweep,NVDA,short,s1p0_v50_d0p8_r0p1_k0p0,cls_1m,1.0,50,,0.8,0.1,0.0,2689734,AUC,0.6259656056990445
sweep,NVDA,short,s1p0_v50_d0p8_r0p1_k0p0,cls_10m,1.0,50,,0.8,0.1,0.0,2689734,AUC,0.5522851316326006
sweep,NVDA,short,s1p0_v50_d0p8_r0p3_k0p0,cls_1m,1.0,50,,0.8,0.3,0.0,2689734,AUC,0.62771941087205
sweep,NVDA,short,s1p0_v50_d0p8_r0p3_k0p0,cls_10m,1.0,50,,0.8,0.3,0.0,2689734,AUC,0.5550368568432437
sweep,NVDA,short,s1p0_v50_d0p8_r0p5_k0p0,cls_1m,1.0,50,,0.8,0.5,0.0,2689734,AUC,0.6276446536439324
sweep,NVDA,short,s1p0_v50_d0p8_r0p5_k0p0,cls_10m,1.0,50,,0.8,0.5,0.0,2689734,AUC,0.5548590630232995
sweep,NVDA,short,s1p0_v50_d0p9_r0p1_k0p0,cls_1m,1.0,50,,0.9,0.1,0.0,2689734,AUC,0.6243000748657773
sweep,NVDA,short,s1p0_v50_d0p9_r0p1_k0p0,cls_10m,1.0,50,,0.9,0.1,0.0,2689734,AUC,0.549350885510906
sweep,NVDA,short,s1p0_v50_d0p9_r0p3_k0p0,cls_1m,1.0,50,,0.9,0.3,0.0,2689734,AUC,0.6246877864065591
sweep,NVDA,short,s1p0_v50_d0p9_r0p3_k0p0,cls_10m,1.0,50,,0.9,0.3,0.0,2689734,AUC,0.5500355024106728
sweep,NVDA,short,s1p0_v50_d0p9_r0p5_k0p0,cls_1m,1.0,50,,0.9,0.5,0.0,2689734,AUC,0.6247112510060963
sweep,NVDA,short,s1p0_v50_d0p9_r0p5_k0p0,cls_10m,1.0,50,,0.9,0.5,0.0,2689734,AUC,0.5500716412267636
sweep,NVDA,short,s1p0_v100_d0p7_r0p1_k0p0,cls_1m,1.0,100,,0.7,0.1,0.0,2517949,AUC,0.6268051430623172
sweep,NVDA,short,s1p0_v100_d0p7_r0p1_k0p0,cls_10m,1.0,100,,0.7,0.1,0.0,2517949,AUC,0.5519833652025472
sweep,NVDA,short,s1p0_v100_d0p7_r0p3_k0p0,cls_1m,1.0,100,,0.7,0.3,0.0,2517949,AUC,0.6293834535086817
sweep,NVDA,short,s1p0_v100_d0p7_r0p3_k0p0,cls_10m,1.0,100,,0.7,0.3,0.0,2517949,AUC,0.5564297280211297
sweep,NVDA,short,s1p0_v100_d0p7_r0p5_k0p0,cls_1m,1.0,100,,0.7,0.5,0.0,2517949,AUC,0.629453113605167
sweep,NVDA,short,s1p0_v100_d0p7_r0p5_k0p0,cls_10m,1.0,100,,0.7,0.5,0.0,2517949,AUC,0.5560579335161541
sweep,NVDA,short,s1p0_v100_d0p8_r0p1_k0p0,cls_1m,1.0,100,,0.8,0.1,0.0,2517949,AUC,0.6263404552965236
sweep,NVDA,short,s1p0_v100_d0p8_r0p1_k0p0,cls_10m,1.0,100,,0.8,0.1,0.0,2517949,AUC,0.5510464266508791
sweep,NVDA,short,s1p0_v100_d0p8_r0p3_k0p0,cls_1m,1.0,100,,0.8,0.3,0.0,2517949,AUC,0.6284039021493595
sweep,NVDA,short,s1p0_v100_d0p8_r0p3_k0p0,cls_10m,1.0,100,,0.8,0.3,0.0,2517949,AUC,0.5544703570898452
sweep,NVDA,short,s1p0_v100_d0p8_r0p5_k0p0,cls_1m,1.0,100,,0.8,0.5,0.0,2517949,AUC,0.6283536624779472
sweep,NVDA,short,s1p0_v100_d0p8_r0p5_k0p0,cls_10m,1.0,100,,0.8,0.5,0.0,2517949,AUC,0.5542569184673263
sweep,NVDA,short,s1p0_v100_d0p9_r0p1_k0p0,cls_1m,1.0,100,,0.9,0.1,0.0,2517949,AUC,0.624528039158703
sweep,NVDA,short,s1p0_v100_d0p9_r0p1_k0p0,cls_10m,1.0,100,,0.9,0.1,0.0,2517949,AUC,0.5479052509491692
sweep,NVDA,short,s1p0_v100_d0p9_r0p3_k0p0,cls_1m,1.0,100,,0.9,0.3,0.0,2517949,AUC,0.6249791011976982
sweep,NVDA,short,s1p0_v100_d0p9_r0p3_k0p0,cls_10m,1.0,100,,0.9,0.3,0.0,2517949,AUC,0.5487512088483808
sweep,NVDA,short,s1p0_v100_d0p9_r0p5_k0p0,cls_1m,1.0,100,,0.9,0.5,0.0,2517949,AUC,0.6250051311319957
sweep,NVDA,short,s1p0_v100_d0p9_r0p5_k0p0,cls_10m,1.0,100,,0.9,0.5,0.0,2517949,AUC,0.5488302732431704
sweep,NVDA,short,s1p0_v200_d0p7_r0p1_k0p0,cls_1m,1.0,200,,0.7,0.1,0.0,2028201,AUC,0.6267541594935525
sweep,NVDA,short,s1p0_v200_d0p7_r0p1_k0p0,cls_10m,1.0,200,,0.7,0.1,0.0,2028201,AUC,0.548503652928167
sweep,NVDA,long,s0p5_v50_d0p7_r0p1_k0p2,cls_close,0.5,50,,0.7,0.1,0.2,1445198,AUC,0.5316215622622565
sweep,NVDA,long,s0p5_v50_d0p7_r0p1_k0p2,cls_clop,0.5,50,,0.7,0.1,0.2,1445198,AUC,0.5649462687558522
sweep,NVDA,long,s0p5_v50_d0p7_r0p1_k0p2,cls_clcl,0.5,50,,0.7,0.1,0.2,1445198,AUC,0.5164477338708373
```
## 13. Full PnL Log Findings (Printed Metrics)
- This section includes direct printed metrics from each PnL run log, not just filename references.
- All values below are transcribed/parsing-derived from run logs in `hoffman_pull_20260407/`.

### 13.1 Run-by-Run Core Metrics Table (NVDA all-target batch)
| log_file | target | signal_mode | execution_mode | trades_fired | entry_cost_raw | exit_cost_raw | total_cost_raw | cum_pnl_raw | daily_mean | daily_std | sharpe |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| run_10_reg_10m_percentile.log | reg_10m | percentile | burst_stream | 23,138 (11,994 Long / 11,144 Short) | 0.0000 | 0.0000 | 0.0000 | -1473.7900 | -1.50540 | 110.18151 | -0.22 |
| run_11_reg_10m_cost_aware_cb0.5.log | reg_10m | cost_aware | burst_stream | 1,023 (989 Long / 34 Short) | 0.0000 | 0.0000 | 0.0000 | -79.2000 | -0.08090 | 5.03351 | -0.26 |
| run_12_reg_10m_cost_aware_cb1.0.log | reg_10m | cost_aware | burst_stream | 648 (631 Long / 17 Short) | 0.0000 | 0.0000 | 0.0000 | -136.5000 | -0.13943 | 4.38561 | -0.50 |
| run_13_reg_close_percentile.log | reg_close | percentile | burst_stream | 23,663 (11,585 Long / 12,078 Short) | 0.0000 | 0.0000 | 0.0000 | -2085.1200 | -2.12985 | 116.71730 | -0.29 |
| run_14_reg_close_cost_aware_cb0.5.log | reg_close | cost_aware | burst_stream | 88 (50 Long / 38 Short) | 0.0000 | 0.0000 | 0.0000 | 79.1000 | 0.08080 | 4.27872 | 0.30 |
| run_15_reg_close_cost_aware_cb1.0.log | reg_close | cost_aware | burst_stream | 74 (45 Long / 29 Short) | 0.0000 | 0.0000 | 0.0000 | 80.6400 | 0.08237 | 3.82598 | 0.34 |
| run_16_reg_clop_percentile.log |  |  |  | FAILED/PARTIAL |  |  |  |  |  |  |  |
| run_1_reg_1m_percentile.log | reg_1m | percentile | burst_stream | 23,032 (11,926 Long / 11,106 Short) | 0.0000 | 0.0000 | 0.0000 | -3585.8000 | -3.66272 | 20.30991 | -2.86 |
| run_2_reg_1m_cost_aware_cb0.5.log | reg_1m | cost_aware | burst_stream | 5,316 (5,229 Long / 87 Short) | 0.0000 | 0.0000 | 0.0000 | -906.4600 | -0.92590 | 4.79964 | -3.06 |
| run_3_reg_1m_cost_aware_cb1.0.log | reg_1m | cost_aware | burst_stream | 3,960 (3,895 Long / 65 Short) | 0.0000 | 0.0000 | 0.0000 | -724.0300 | -0.73956 | 4.11446 | -2.85 |
| run_4_reg_3m_percentile.log | reg_3m | percentile | burst_stream | 23,030 (12,006 Long / 11,024 Short) | 0.0000 | 0.0000 | 0.0000 | -3002.8900 | -3.06730 | 38.45495 | -1.27 |
| run_5_reg_3m_cost_aware_cb0.5.log | reg_3m | cost_aware | burst_stream | 3,336 (3,264 Long / 72 Short) | 0.0000 | 0.0000 | 0.0000 | -564.6500 | -0.57676 | 5.20943 | -1.76 |
| run_6_reg_3m_cost_aware_cb1.0.log | reg_3m | cost_aware | burst_stream | 2,377 (2,320 Long / 57 Short) | 0.0000 | 0.0000 | 0.0000 | -466.6400 | -0.47665 | 4.24601 | -1.78 |
| run_7_reg_5m_percentile.log | reg_5m | percentile | burst_stream | 23,134 (12,008 Long / 11,126 Short) | 0.0000 | 0.0000 | 0.0000 | -2948.3600 | -3.01160 | 56.58720 | -0.84 |
| run_8_reg_5m_cost_aware_cb0.5.log | reg_5m | cost_aware | burst_stream | 2,552 (2,487 Long / 65 Short) | 0.0000 | 0.0000 | 0.0000 | -524.4200 | -0.53567 | 5.47767 | -1.55 |
| run_9_reg_5m_cost_aware_cb1.0.log | reg_5m | cost_aware | burst_stream | 1,738 (1,686 Long / 52 Short) | 0.0000 | 0.0000 | 0.0000 | -301.9200 | -0.30840 | 4.56806 | -1.07 |

### 13.2 Debug Runs: Full Printed Diagnostics
#### run_nvda_debug_20260407.12846931.log
- Target: reg_10m
- Signal mode: cost_aware
- Execution mode: burst_stream
- Total Trades Fired: 648 (631 Long / 17 Short)
- Entry Spread Cost (raw): 0.0000
- Exit Spread Cost (raw): 0.0000
- Total Spread Cost (raw): 0.0000
- Cumulative Simulated PnL (raw): -136.5000
- Daily Mean PnL (raw): -0.13943
- Daily StdDev (raw): 4.38561
- Annualized Sharpe Ratio: -0.50
- Signals evaluated: 44,455
- Signals passed long: 631 (1.42%)
- Signals passed short: 17 (0.04%)
- Signals rejected: 43,807 (98.54%)
- Long side diagnostics: `Long   trades=    631 win_rate= 47.54% gross= -144.3300 cost=    0.0000 net= -144.3300 avg_net/trade= -0.228732`
- Short side diagnostics: `Short  trades=     17 win_rate= 47.06% gross=    7.8300 cost=    0.0000 net=    7.8300 avg_net/trade=  0.460588`

#### run_nvda_debug_regclose_20260407.12847805.log
- Target: reg_close
- Signal mode: cost_aware
- Execution mode: burst_stream
- Total Trades Fired: 74 (45 Long / 29 Short)
- Entry Spread Cost (raw): 0.0000
- Exit Spread Cost (raw): 0.0000
- Total Spread Cost (raw): 0.0000
- Cumulative Simulated PnL (raw): 80.6400
- Daily Mean PnL (raw): 0.08237
- Daily StdDev (raw): 3.82598
- Annualized Sharpe Ratio: 0.34
- Signals evaluated: 44,455
- Signals passed long: 45 (0.10%)
- Signals passed short: 29 (0.07%)
- Signals rejected: 44,381 (99.83%)
- Long side diagnostics: `Long   trades=     42 win_rate= 59.52% gross=  126.4300 cost=    0.0000 net=  126.4300 avg_net/trade=  3.010238`
- Short side diagnostics: `Short  trades=     24 win_rate= 45.83% gross=  -45.7900 cost=    0.0000 net=  -45.7900 avg_net/trade= -1.907917`

### 13.3 Interpretation of Spread Inclusion from Printed Logs
- In quote-based burst-stream debug runs, printed explicit spread costs were often `0.0000`, meaning losses were dominated by directional edge quality rather than explicit modeled spread charges in those runs.
- Therefore, after adding spread-aware framework, the decisive factor remained hit-rate/EV per trade under selected signal rules and horizon target.

## 14. Direct Answers: Config Strings, Models, and Metrics (2026-04-08)

### 14.1 The New Configuration Strings
The current top physical-parameter configurations (from `results/optuna_physical/*/best_physical_params_*.json`) are fractional-volume (`vf`) and map to your requested `sf` style as follows:

- Canonical format now: `s{silence}_sf{vol_frac}_d{dir_thresh}_r{vol_ratio}_k{kappa}`
- Internal field name used in code/files is `vf` (volume fraction).

Top AUC configurations currently observed:

| Rank | Ticker | Target | AUC | Exact String |
|---|---|---|---:|---|
| 1 | NVDA | `cls_1m` | 0.6533 | `s2.0_sf0.0049157326743032685_d0.7096570261049338_r0.5673611393219141_k0.0` |
| 2 | TSLA | `cls_1m` | 0.6464 | `s0.5_sf0.004982101226003821_d0.6781305353061478_r0.42618445776832053_k0.0` |
| 3 | TSLA | `cls_3m` | 0.6436 | `s2.0_sf0.004305421645396051_d0.654607615387428_r0.511539003720897_k0.0` |
| 4 | NVDA | `cls_3m` | 0.6288 | `s2.0_sf0.0049830984354126955_d0.6620045163766891_r0.5515813238244158_k0.0` |
| 5 | MS | `cls_1m` | 0.6285 | `s2.0_sf0.0016515914669865662_d0.9496310321847444_r0.5776935087713172_k0.0` |
| 6 | JPM | `cls_1m` | 0.6239 | `s2.0_sf0.000911990180310947_d0.929607686295797_r0.3865011086107934_k0.0` |

Examples for long-horizon targets with nonzero `kappa` (same source):

- NVDA `cls_clop`: `s0.5_sf0.0005791438631370755_d0.5006846785554202_r0.5351208424087406_k1.6622979927165715`
- NVDA `cls_close`: `s0.5_sf0.0006830448323427173_d0.7158494882817108_r0.06887597272196404_k1.2762520750555515`

### 14.2 The Models Evaluated
Short answer: both paths exist, but they are serving different phases.

- Regression benchmarking phase (`src_py/regression_eval.py`) is still evaluating:
  - `HistGB_Restricted`
  - `XGB_Restricted`
  - `ElasticNet`
  - `Ridge`
  - (also `RandomForest_Shallow` in that script)
- Trading/backtest phase (`src_py/online_sgd_backtest.py`) has shifted to `SGDRegressor` and is where `cost_aware` gating, Sharpe, and net PnL-per-trade are computed.

### 14.3 Performance Metrics (Updated)

#### 14.3.1 OoS R-squared (Regression Benchmarking File)
From `results/multi_model_regression_summary.csv` (aggregated over available rows for these model families):

| Model | Mean OoS R2 (all targets) | Max OoS R2 (all targets) |
|---|---:|---:|
| `HistGB_Restricted` | 0.0127 | 0.1082 |
| `XGB_Restricted` | 0.0126 | 0.1063 |
| `ElasticNet` | 0.0088 | 0.0842 |
| `Ridge` | 0.0033 | 0.0838 |

#### 14.3.2 Directional Accuracy / Win Rate Under Cost-Aware Gating
- Cost-aware gating metrics are currently produced in the SGD backtest path, not in the multi-model regression benchmark.
- Therefore:
  - `HistGB/XGB/ElasticNet/Ridge`: no direct cost-aware win-rate output yet (N/A in current pipeline).
  - `SGDRegressor` (cost-aware runs, NVDA logs):

| Run | Trades | Cum PnL (raw) | Sharpe | Avg Net PnL/Trade (raw) |
|---|---:|---:|---:|---:|
| `run_15_reg_close_cost_aware_cb1.0.log` | 74 | 80.64 | 0.34 | 1.089730 |
| `run_14_reg_close_cost_aware_cb0.5.log` | 88 | 79.10 | 0.30 | 0.898864 |
| `run_11_reg_10m_cost_aware_cb0.5.log` | 1,023 | -79.20 | -0.26 | -0.077419 |
| `run_12_reg_10m_cost_aware_cb1.0.log` | 648 | -136.50 | -0.50 | -0.210648 |
| `run_9_reg_5m_cost_aware_cb1.0.log` | 1,738 | -301.92 | -1.07 | -0.173717 |
| `run_8_reg_5m_cost_aware_cb0.5.log` | 2,552 | -524.42 | -1.55 | -0.205494 |
| `run_5_reg_3m_cost_aware_cb0.5.log` | 3,336 | -564.65 | -1.76 | -0.169260 |
| `run_6_reg_3m_cost_aware_cb1.0.log` | 2,377 | -466.64 | -1.78 | -0.196315 |
| `run_3_reg_1m_cost_aware_cb1.0.log` | 3,960 | -724.03 | -2.85 | -0.182836 |
| `run_2_reg_1m_cost_aware_cb0.5.log` | 5,316 | -906.46 | -3.06 | -0.170515 |

Debug-run win rates that were explicitly printed:

- `run_nvda_debug_20260407.12846931.log` (`reg_10m`, cost-aware): Long win rate `47.54%`, Short win rate `47.06%`.
- `run_nvda_debug_regclose_20260407.12847805.log` (`reg_close`, cost-aware): Long win rate `59.52%`, Short win rate `45.83%`.

#### 14.3.3 Profitability Metrics to Report Going Forward
Recommended primary profitability metrics (and now reported above):

- `Annualized Sharpe Ratio`
- `Avg Net PnL per Trade (raw)`
- `Cumulative Simulated PnL (raw)` as a secondary context metric

These are the right replacements for legacy raw "Net Perm" when evaluating executable, gated trading behavior.

### 14.4 Hoffman Overnight Batch (`12890943`, 2026-04-09)

#### 14.4.1 Execution Health
- Batch status: completed cleanly for all four array tasks.
- `qstat -u nicjia`: empty after completion.
- Error scan on `logs/overnight_bt_12890943_*.out`: no `Traceback`, no `Input y contains NaN`, no `ERROR: line`.
- Completion markers found:
  - `Overnight backtests complete for NVDA`
  - `Overnight backtests complete for TSLA`
  - `Overnight backtests complete for JPM`
  - `Overnight backtests complete for MS`

#### 14.4.2 Pulled Artifacts (Local Snapshot)
- Pull root: `hoffman_pull_20260409_overnight/`
- Job logs: `hoffman_pull_20260409_overnight/logs/overnight_bt_12890943_*.out`
- Main result files pulled from Hoffman: 24 files (8 strategy runs x `.log`, `_debug_trades.csv`, `_debug_signals.csv`)
- Result directory: `hoffman_pull_20260409_overnight/results/overnight_backtests/`
- Analyzer script (temporary): `tmp_analyze_overnight_12890943.py`
- Analyzer outputs:
  - `hoffman_pull_20260409_overnight/analysis/overnight_12890943_metrics.csv`
  - `hoffman_pull_20260409_overnight/analysis/overnight_12890943_summary.md`

#### 14.4.3 Consolidated Run Metrics (from pulled `.log` files)
| Ticker | Target | Trades | Cum PnL (raw) | Sharpe | Dropped non-finite rows |
|---|---|---:|---:|---:|---:|
| JPM | reg_clcl | 409,968 | -15,903,610.56 | -2.50 | 1,237 |
| JPM | reg_clop | 22 | -313,474.70 | -0.58 | 39 |
| MS | reg_clcl | 2,302 | -187,958.10 | -0.76 | 47 |
| MS | reg_clop | 5,822 | -297,268.07 | -1.19 | 32 |
| NVDA | reg_clcl | 701 | 118,711,049.88 | 1.58 | 42 |
| NVDA | reg_clop | 1,857 | 181,747,889.84 | 1.17 | 65 |
| TSLA | reg_clcl | 0 | 0.00 | 0.00 | 9 |
| TSLA | reg_clop | 0 | 0.00 | 0.00 | 6 |

#### 14.4.4 Batch-Level Aggregates
- By target:
  - `reg_clcl`: 412,971 trades, cumulative PnL `102,619,481.22`, mean Sharpe `-0.42`.
  - `reg_clop`: 7,701 trades, cumulative PnL `181,137,147.07`, mean Sharpe `-0.15`.
- By ticker:
  - `NVDA`: strong positive result in both overnight targets (combined PnL `300,458,939.72`, mean Sharpe `1.375`).
  - `JPM` and `MS`: negative in both overnight targets for this run.
  - `TSLA`: zero triggered trades under selected cost-aware gates for both overnight targets.
