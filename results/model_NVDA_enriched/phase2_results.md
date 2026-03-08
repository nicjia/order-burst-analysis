# Phase II — Predicting Permanence

Data: `results/bursts_NVDA_filtered.csv` (768,498 bursts)

Walk-forward CV: 21 monthly folds (2023-04 → 2024-12)

## Classification: P(φ_tCLOSE > 1)

| Model | AUC | Accuracy | Precision | Recall | F1 | Brier |
|-------|-----|----------|-----------|--------|----|----- -|
| LightGBM | 0.5392 | 0.5307 | 0.5310 | 0.9123 | 0.6713 | 0.2490 |
| LogReg (monthly mean AUC) | 0.5498 | — | — | — | — | — |

## Regression: φ_tCLOSE (winsorized)

| Metric | Value |
|--------|-------|
| RMSE | 272.6416 |
| MAE | 122.2080 |
| R2 | 0.0005 |
| DirAcc | 0.5380 |

## Feature Importance (last fold)

| Feature | Gain |
|---------|------|
| D_b | 4059.7 |
| Direction | 2430.3 |
| PeakImpact | 1453.3 |
| RecentBurstVol | 932.6 |
| RecentBurstCount | 393.3 |
| ImpactPerShare | 95.5 |
| AvgTradeSize | 32.1 |
| BurstVolume | 25.0 |
| TradeCount | 0.0 |
| Duration | 0.0 |
| PriceChange | 0.0 |
| TimeOfDay | 0.0 |
| LogVolume | 0.0 |
| LogPeakImpact | 0.0 |
| LogSpread | 0.0 |
| DepthRatio | 0.0 |
| LogTradeIntensity | 0.0 |
| SpreadXVolume | 0.0 |
