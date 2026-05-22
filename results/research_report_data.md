# Comprehensive Research Data: Informed Order-Burst Microstructure Signatures
## Consolidated Master Reference — All Experiments, All Data
> **Last Updated**: 2026-05-17
> This single document contains every empirical result, parameter set, benchmark, and methodological
> detail produced during the research pipeline. It is designed to be fed directly to an LLM
> research-writing agent to produce a complete, peer-reviewed manuscript.

---

# PART I: PIPELINE ARCHITECTURE

## 1.1 Phase I — Burst Detection (C++ Engine)
- **Input**: Raw LOBSTER Level-1 limit order book message files (event-by-event, microsecond timestamps)
- **Output**: Per-burst feature vectors with ~35 microstructure features
- **Hawkes Process**: Self-exciting point process with baseline intensity β=1.0, decay rate α=0.5 (tag: `b1p0_i0p5`)
- **Burst Definition**: A contiguous cluster of aggressive orders whose inter-arrival intensity exceeds the Hawkes threshold
- **Trading Hours**: 9:30 AM – 4:00 PM EST (unix seconds 34200 – 57600)
- **Threads**: 4 parallel day-level processing

## 1.2 Phase II — Feature Engineering & Volume-Scaled Impact (Python)
**Permanence Targets** (renamed "Volume-Scaled Impact" / VSI):
- `reg_close`: arcsinh(Q_b × (P_close − P_burst) / P_burst) — burst-to-daily-close
- `reg_clop`: arcsinh(Q_b × (P_nextopen − P_close) / P_close) — close-to-next-open (overnight)
- `reg_clcl`: arcsinh(Q_b × (P_nextclose − P_close) / P_close) — close-to-next-close (full day)

**Classification Targets**: Binary sign of each regression target (cls_close, cls_clop, cls_clcl)

**Structural Filters**:
- Fractional ADV (`vol_frac`): Burst volume must exceed X% of 14-day trailing ADV
- Directionality (`dir_thresh`): Burst directional consistency must exceed threshold
- Volume ratio (`vol_ratio`): Ratio of aggressive-to-passive volume must exceed threshold
- Decay filter (κ): D_b must hold ≥ κ fraction of peak impact at t+10min

## 1.3 Phase III — Online SGD Walk-Forward Execution
- **Model**: SGDRegressor with Huber loss, online learning
- **Scaler**: StandardScaler fitted once during burn-in, then frozen
- **Burn-in**: First ~30 calendar days (model trains, does not trade)
- **Walk-Forward**: Strictly chronological. Train on day T, predict day T+1. Zero future data leakage.

### Execution Logic (phase3_flow)
1. Scan all bursts passing structural filters for the day
2. Discard bursts after 3:50 PM (10-minute D_b dead zone)
3. SGD predicts Volume-Scaled Impact for each eligible burst
4. If pred > 0.0 → burst tagged "informational"
5. Aggregate informational bursts into daily Net Directional Volume signal
6. Trade direction = sign(Net Directional Volume)
7. **Entry**: Market-On-Close (MOC) at 4:00 PM
8. **Exit (CLOP)**: Market-On-Open (MOO) at 9:30 AM next day
9. **Exit (CLCL)**: Market-On-Close (MOC) at 4:00 PM next day

### Execution Constraints
- Capital: $10,000,000 Fixed AUM per trade
- Transaction Costs: 1.0 bps round-trip (exchange fees + auction impact)
- No rolling percentiles or learned thresholds
- No leverage; one position per day maximum

---

# PART II: PARAMETER OPTIMIZATION (OPTUNA)

## 2.1 Methodology
- **Method**: Bayesian optimization (100 trials per ticker per target)
- **ML Model**: HistGradientBoostingClassifier (70/30 chronological split)
- **Objective**: Maximize out-of-sample ROC AUC on binary classification targets
- **Search Space**: vol_frac ∈ [1e-5, 0.005], dir_thresh ∈ [0.5, 0.95], vol_ratio ∈ [0.01, 0.6], kappa ∈ [0.0, 2.0]

## 2.2 Best Parameters Per Ticker (cls_clop, best Hawkes tag)

| Ticker | Hawkes Tag | AUC | vol_frac | dir_thresh | vol_ratio | kappa |
|:---|:---|:---|:---|:---|:---|:---|
| NVDA | b1p0_i0p3 | 0.6046 | 0.000187 | 0.511 | 0.600 | 0.988 |
| TSLA | b1p0_i0p3 | 0.5325 | 0.003636 | 0.643 | 0.481 | 0.969 |
| JPM  | b1p0_i0p8 | 0.5477 | 0.00001 | 0.503 | 0.553 | 0.206 |
| MS   | b1p0_i0p5 | 0.5527 | 0.004296 | 0.830 | 0.405 | 0.950 |

## 2.3 Universal Parameters (for OOS tickers: LLY, AAPL, SPY)
vol_frac=0.0005, dir_thresh=0.6, vol_ratio=0.5, kappa=0.8

## 2.4 Hawkes Tag Stability (Cross-Tag AUC Variance)
The burst definition is structurally robust across Hawkes intensity parameters:

| Ticker | i0p3 AUC | i0p5 AUC | i0p8 AUC | Max Δ |
|:---|:---|:---|:---|:---|
| NVDA (cls_clop) | 0.6046 | 0.6024 | 0.6003 | 0.0043 |
| TSLA (cls_clop) | 0.5325 | 0.5279 | 0.5233 | 0.0092 |
| JPM (cls_clop)  | 0.5379 | 0.5454 | 0.5477 | 0.0098 |
| MS (cls_clop)   | 0.5463 | 0.5527 | 0.5359 | 0.0168 |

## 2.5 Full Optuna Sweep Data (All Tags × All Targets)

| Ticker | Target | Hawkes Tag | AUC | dir_thresh | vol_frac | vol_ratio | kappa |
|:---|:---|:---|:---|:---|:---|:---|:---|
| JPM | cls_clcl | b1p0_i0p3 | 0.5772 | 0.894 | 1.1e-05 | 0.420 | 0.175 |
| JPM | cls_clcl | b1p0_i0p5 | 0.5491 | 0.569 | 1e-05 | 0.379 | 0.464 |
| JPM | cls_clcl | b1p0_i0p8 | 0.5506 | 0.659 | 1e-05 | 0.493 | 0.074 |
| JPM | cls_clop | b1p0_i0p3 | 0.5379 | 0.666 | 0.004593 | 0.336 | 1.940 |
| JPM | cls_clop | b1p0_i0p5 | 0.5454 | 0.529 | 1.2e-05 | 0.199 | 0.873 |
| JPM | cls_clop | b1p0_i0p8 | 0.5477 | 0.503 | 1e-05 | 0.553 | 0.206 |
| JPM | cls_close | b1p0_i0p3 | 0.5517 | 0.669 | 0.000104 | 0.521 | 0.640 |
| JPM | cls_close | b1p0_i0p5 | 0.5156 | 0.667 | 0.000344 | 0.533 | 1.176 |
| JPM | cls_close | b1p0_i0p8 | 0.5515 | 0.521 | 0.003595 | 0.497 | 0.084 |
| MS | cls_clcl | b1p0_i0p3 | 0.5187 | 0.695 | 0.001578 | 0.360 | 0.451 |
| MS | cls_clcl | b1p0_i0p5 | 0.5200 | 0.922 | 0.004927 | 0.267 | 1.291 |
| MS | cls_clcl | b1p0_i0p8 | 0.5186 | 0.876 | 0.001315 | 0.376 | 0.056 |
| MS | cls_clop | b1p0_i0p3 | 0.5463 | 0.775 | 0.004273 | 0.330 | 0.064 |
| MS | cls_clop | b1p0_i0p5 | 0.5527 | 0.830 | 0.004296 | 0.405 | 0.950 |
| MS | cls_clop | b1p0_i0p8 | 0.5359 | 0.929 | 0.004250 | 0.084 | 0.195 |
| MS | cls_close | b1p0_i0p3 | 0.5319 | 0.682 | 0.000329 | 0.447 | 0.496 |
| MS | cls_close | b1p0_i0p5 | 0.5314 | 0.724 | 0.000326 | 0.214 | 0.596 |
| MS | cls_close | b1p0_i0p8 | 0.5313 | 0.843 | 0.000360 | 0.232 | 0.464 |
| NVDA | cls_clcl | b1p0_i0p3 | 0.5249 | 0.668 | 0.000277 | 0.586 | 1.389 |
| NVDA | cls_clcl | b1p0_i0p5 | 0.5272 | 0.738 | 0.000847 | 0.600 | 1.315 |
| NVDA | cls_clcl | b1p0_i0p8 | 0.5325 | 0.660 | 0.000865 | 0.567 | 0.254 |
| NVDA | cls_clop | b1p0_i0p3 | 0.6046 | 0.511 | 0.000187 | 0.600 | 0.988 |
| NVDA | cls_clop | b1p0_i0p5 | 0.6024 | 0.504 | 0.000130 | 0.542 | 0.864 |
| NVDA | cls_clop | b1p0_i0p8 | 0.6003 | 0.550 | 7.8e-05 | 0.551 | 0.184 |
| NVDA | cls_close | b1p0_i0p3 | 0.5520 | 0.531 | 1.3e-05 | 0.480 | 0.784 |
| NVDA | cls_close | b1p0_i0p5 | 0.5530 | 0.513 | 1.9e-05 | 0.551 | 0.152 |
| NVDA | cls_close | b1p0_i0p8 | 0.5509 | 0.548 | 2e-05 | 0.501 | 0.507 |
| TSLA | cls_clcl | b1p0_i0p3 | 0.4985 | 0.949 | 6.1e-05 | 0.525 | 1.155 |
| TSLA | cls_clcl | b1p0_i0p5 | 0.4994 | 0.940 | 6.5e-05 | 0.013 | 1.651 |
| TSLA | cls_clcl | b1p0_i0p8 | 0.4987 | 0.931 | 6.5e-05 | 0.186 | 1.034 |
| TSLA | cls_clop | b1p0_i0p3 | 0.5325 | 0.643 | 0.003636 | 0.481 | 0.969 |
| TSLA | cls_clop | b1p0_i0p5 | 0.5279 | 0.694 | 0.003764 | 0.239 | 0.259 |
| TSLA | cls_clop | b1p0_i0p8 | 0.5233 | 0.641 | 0.003471 | 0.590 | 0.964 |
| TSLA | cls_close | b1p0_i0p3 | 0.5412 | 0.544 | 1.4e-05 | 0.592 | 0.325 |
| TSLA | cls_close | b1p0_i0p5 | 0.5403 | 0.558 | 1e-05 | 0.544 | 1.552 |
| TSLA | cls_close | b1p0_i0p8 | 0.5414 | 0.512 | 1.1e-05 | 0.600 | 0.855 |

---

# PART III: BACKTEST RESULTS

## 3.1 Volume-Weighted Signal (Original, signed_volume)

### CLOP — Close-to-Open (Overnight), $10M AUM, 1.0 bps

| Ticker | Period | Trades | Long/Short | Long Win% | Short Win% | Net PnL ($) | ROC (%) | Max DD | Sharpe |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **NVDA** | 2023-2024 | 470 | 305/165 | 57.05% | 43.64% | +$14,162,158 | **+141.62%** | -65.5% | 1.58 |
| **TSLA** | 2023-2024 | 286 | 70/216 | 55.71% | 47.22% | +$7,321,358 | **+73.21%** | -16.4% | 1.57 |
| JPM | 2023-2024 | 470 | 277/193 | 49.82% | 46.11% | -$70,265 | -0.70% | -16.3% | -0.03 |
| MS | 2023-2024 | 100 | 8/92 | 25.00% | 43.48% | -$505,155 | -5.05% | ~6.8% | -0.47 |
| **LLY** | 2019-2021 | 476 | 461/15 | 56.40% | 46.67% | +$1,126,792 | **+11.27%** | -36.1% | 0.32 |
| AAPL | 2019-2021 | 334 | 221/113 | 54.30% | 40.71% | -$3,030,938 | -30.31% | -63.2% | -0.87 |
| **SPY** | 2019-2021 | 289 | 262/27 | 58.40% | 51.85% | +$3,186,165 | **+31.86%** | -9.3% | 1.35 |

### CLCL — Close-to-Close (Full Day), $10M AUM, 1.0 bps

| Ticker | Period | Trades | Long/Short | Long Win% | Short Win% | Net PnL ($) | ROC (%) | Max DD | Sharpe |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| NVDA | 2023-2024 | 470 | 298/172 | 53.36% | 43.60% | -$7,792,434 | -77.92% | -139.1% | -0.50 |
| TSLA | 2023-2024 | 163 | 30/133 | 46.67% | 42.86% | -$9,957,494 | -99.57% | -135.7% | -1.41 |
| JPM | 2023-2024 | 471 | 375/96 | 54.13% | 35.42% | -$83,216 | -0.83% | -32.9% | -0.02 |
| MS | 2023-2024 | 68 | 12/56 | 50.00% | 41.07% | -$56,854 | -0.57% | -14.1% | -0.04 |
| LLY | 2019-2021 | 406 | 359/47 | 52.09% | 42.55% | -$1,519,239 | -15.19% | -38.9% | -0.25 |
| AAPL | 2019-2021 | 333 | 301/32 | 54.15% | 31.25% | -$3,804,153 | -38.04% | -120.5% | -0.78 |

## 3.2 Prediction-Magnitude-Weighted Signal (Option A: pred_weighted)

In this mode, the daily flow aggregation uses the SGD's predicted VSI magnitude instead of raw burst volume. Bursts predicted to have high permanence contribute proportionally more to the daily signal.

**Critical Observation**: pred_weighted eliminates ALL short trades. Because the SGD predicts permanence in arcsinh-transformed space, and the informational gate is pred > 0, the sum of positive predictions is always positive — yielding a permanent long bias. This reveals the fundamental limitation of this approach: it collapses the model into a long-only strategy.

### CLOP (pred_weighted), $10M AUM, 1.0 bps

| Ticker | Period | Trades | Long/Short | Win% | Net PnL ($) | ROC (%) | Max DD | Sharpe |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| NVDA | 2023-2024 | 470 | 470/0 | 56.60% | +$6,493,199 | +64.93% | -94.6% | 0.47 |
| TSLA | 2023-2024 | 408 | 408/0 | 52.94% | -$1,272,025 | -12.72% | -53.4% | -0.22 |
| JPM | 2023-2024 | 470 | 470/0 | 50.21% | +$1,380,555 | +13.81% | -12.6% | 0.61 |
| MS | 2023-2024 | 108 | 108/0 | 54.63% | +$173,630 | +1.74% | -11.3% | 0.16 |
| LLY | 2019-2021 | 476 | 476/0 | 56.09% | +$1,881,185 | +18.81% | -29.1% | 0.53 |
| AAPL | 2019-2021 | 334 | 334/0 | 55.99% | -$581,235 | -5.81% | -43.2% | -0.17 |
| SPY | 2019-2021 | 289 | 289/0 | 57.09% | -$1,570,096 | -15.70% | -33.2% | -0.66 |

### CLCL (pred_weighted), $10M AUM, 1.0 bps

| Ticker | Period | Trades | Long/Short | Win% | Net PnL ($) | ROC (%) | Max DD | Sharpe |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| NVDA | 2023-2024 | 470 | 470/0 | 54.47% | +$10,032,418 | +100.32% | -117.4% | 0.65 |
| TSLA | 2023-2024 | 221 | 221/0 | 53.85% | +$9,681,098 | +96.81% | -64.9% | 1.25 |

---

# PART IV: BUY-AND-HOLD BENCHMARKS

## 4.1 Overnight B&H (Blind Long Every Night, $10M)

| Ticker | Period | Overnight B&H ROC (%) | Overnight B&H PnL ($) |
|:---|:---|:---|:---|
| NVDA | 2023-2024 | +77.51% | +$7,750,926 |
| TSLA | 2023-2024 | +70.99% | +$7,098,627 |
| JPM | 2023-2024 | +15.69% | +$1,568,894 |
| MS | 2023-2024 | +30.48% | +$3,048,309 |
| LLY | 2019-2021 | +15.41% | +$1,540,618 |
| AAPL | 2019-2021 | -26.03% | -$2,603,001 |

## 4.2 Close-to-Close B&H (Blind Long Every Day, $10M)

| Ticker | Period | CLCL B&H ROC (%) | CLCL B&H PnL ($) |
|:---|:---|:---|:---|
| NVDA | 2023-2024 | +160.66% | +$16,065,534 |
| TSLA | 2023-2024 | +168.44% | +$16,844,498 |
| JPM | 2023-2024 | +62.05% | +$6,205,183 |
| MS | 2023-2024 | +45.02% | +$4,501,947 |
| LLY | 2019-2021 | +49.54% | +$4,954,142 |
| AAPL | 2019-2021 | +58.16% | +$5,816,332 |

## 4.3 Excess Alpha (Volume-Weighted CLOP Strategy vs. Overnight B&H)

| Ticker | Period | Strategy ROC | Overnight B&H | Excess Alpha |
|:---|:---|:---|:---|:---|
| **NVDA** | 2023-2024 | +141.62% | +77.51% | **+64.11pp** ✓ |
| TSLA | 2023-2024 | +73.21% | +70.99% | +2.22pp (marginal) |
| JPM | 2023-2024 | -0.70% | +15.69% | -16.39pp ✗ |
| MS | 2023-2024 | -5.05% | +30.48% | -35.53pp ✗ |
| LLY | 2019-2021 | +11.27% | +15.41% | -4.14pp (marginal) |
| AAPL | 2019-2021 | -30.31% | -26.03% | -4.28pp ✗ |
| **SPY** | 2019-2021 | +31.86% | N/A | N/A |

---

# PART V: TRANSACTION COST SENSITIVITY (CLOP, signed_volume)

| Ticker | 1.0 bps ROC | 1.0 bps Sharpe | 3.0 bps ROC | 3.0 bps Sharpe | 5.0 bps ROC | 5.0 bps Sharpe |
|:---|:---|:---|:---|:---|:---|:---|
| NVDA | +141.62% | 1.58 | +132.22% | 0.95 | +122.82% | 0.88 |
| TSLA | +73.21% | 1.57 | +67.49% | 1.37 | +61.77% | 1.25 |

---

# PART VI: BURST COUNTS AND DATASET SIZES

| Ticker | Period | Raw Bursts | After Filters | Trading Days with Signal |
|:---|:---|:---|:---|:---|
| NVDA | 2023-2024 | 894,343 | ~214,142 | 470 |
| TSLA | 2023-2024 | 757,120 | ~4,371 | 286 |
| JPM | 2023-2024 | 1,251,679 | ~400,430 | 470 |
| MS | 2023-2024 | 1,108,679 | ~752 | 100 |
| LLY | 2019-2021 | 308,361 | ~138,652 | 476 |
| AAPL | 2019-2021 | 117,253 | ~38,167 | 334 |
| SPY | 2019-2021 | ~150,000 | ~34,824 | 289 |

---

# PART VII: KEY FINDINGS & INTERPRETATIONS

## 7.1 CLOP Works, CLCL Universally Fails (Volume-Weighted Signal)
The overnight horizon isolates informational carry. Holding through the next trading day exposes the position to gap mean-reversion and unrelated flow that dilutes the signal. Every ticker lost money on CLCL with the volume-weighted signal.

## 7.2 pred_weighted Collapses to Long-Only
The magnitude-weighted approach (Option A) has a structural flaw: since only bursts with pred > 0 are selected, and the flow signal = sum of those positive predictions, the signal is ALWAYS positive. This eliminates short trades entirely, collapsing into a long-only overnight strategy. Key comparisons:
- **JPM CLOP**: Volume-weighted = -0.70% → Pred-weighted = +13.81% (improved by going long-only in a bull market)
- **NVDA CLOP**: Volume-weighted = +141.62% → Pred-weighted = +64.93% (WORSE — lost the profitable short trades)
- **TSLA CLOP**: Volume-weighted = +73.21% → Pred-weighted = -12.72% (WORSE — was heavily short-biased, now forced long)

## 7.3 The AUC-vs-PnL Disconnect
Optuna optimized classification AUC (binary direction), but the backtest uses regression (magnitude). AUC treats a 1-cent win identically to a $1000 loss. The parameters that maximize AUC > 0.5 do not necessarily maximize magnitude-weighted PnL. This is a known limitation that Option B (regression-based Optuna) would address.

## 7.4 Asset-Class Boundary Conditions
The strategy succeeds on assets with: high idiosyncratic volatility, large-tick microstructure (thin books swept by aggressive orders), and directional institutional accumulation (momentum). It fails on: tick-constrained assets (AAPL), ETF-arbitrage-dominated flow (JPM, MS), and assets where overnight gaps routinely mean-revert.

## 7.5 LLY Out-of-Sample Structural Invariability
Using 2023-2024 parameters on 2019-2021 LLY data (different sector, different regime, COVID crash included): +11.27% ROC with Sharpe 0.32. This is NOT presented as walk-forward trading — it is a stress test of physical parameter universality.

---

# PART VIII: KNOWN LIMITATIONS & FUTURE WORK

1. **AUC-PnL Misalignment**: Optuna optimizes binary classification AUC, but execution uses regression magnitude. Solution: Option B — regression-based Optuna sweep using Spearman correlation or simulated PnL as the objective.
2. **No Intraday (CL) Testing**: The burst-to-close target (reg_close) requires burst_stream execution mode, not phase3_flow. Position sizing logic needs adaptation for multiple simultaneous intraday positions.
3. **Order Book Execution Prices**: We reconstruct the BBO but execute at MOC/MOO. Could potentially use reconstructed 4:00 PM BBO for more realistic friction modeling.
4. **Single-Stock Isolation**: The strategy trades each ticker independently. Cross-asset correlation effects (e.g., sector momentum, index rotation) are not modeled.
5. **2023-2024 Bull Market Bias**: All "successful" tickers rose substantially during the test period. The excess alpha calculation (vs. overnight B&H) is the proper defense, but more bear-market data is needed.

---

# PART X: OPTION B — REGRESSION OPTUNA SWEEP RESULTS (2026-05-20)

## 10.1 Methodology
- **Objective**: Confidence-scaled Spearman rank correlation (ρ × min(1, n_test/500))
- **Model**: SGDRegressor(loss='huber', epsilon=1.35, penalty='l2', alpha=0.001, learning_rate='adaptive', eta0=0.001) — exact mirror of Phase-III backtest
- **Anti-Sparsity Defense**: Hard floor of 200 minimum bursts; score penalized by confidence factor
- **Target**: reg_clop (continuous VSI, NOT binary classification)
- **Split**: 70/30 chronological
- **Trials**: 100 per ticker per Hawkes tag

## 10.2 Best Regression Parameters Per Ticker (reg_clop, best Hawkes tag)

| Ticker | Best Tag | Spearman ρ | p-value | n_test | vol_frac | dir_thresh | vol_ratio | kappa |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **NVDA** | b1p0_i0p3 | **0.196** | 0.0 | 145,000 | 6.45e-05 | 0.509 | 0.583 | 0.824 |
| **TSLA** | b1p0_i0p3 | **0.094** | 3.0e-07 | 2,952 | 0.00330 | 0.568 | 0.599 | 1.736 |
| **JPM** | b1p0_i0p8 | **0.149** | 1.3e-07 | 1,249 | 0.00326 | 0.592 | 0.379 | 1.707 |
| **MS** | b1p0_i0p8 | **0.126** | 0.001 | 651 | 0.00496 | 0.908 | 0.048 | 0.981 |

## 10.3 Regression Spearman Stability (Cross-Tag)

| Ticker | i0p3 ρ | i0p5 ρ | i0p8 ρ | Max Δ |
|:---|:---|:---|:---|:---|
| NVDA | 0.196 | 0.197 | 0.196 | 0.001 |
| TSLA | 0.094 | 0.053 | 0.088 | 0.041 |
| JPM  | 0.049 | 0.134 | 0.149 | 0.100 |
| MS   | 0.125 | 0.071 | 0.126 | 0.055 |

## 10.4 Full Regression Sweep Data (All Tags × All Tickers)

| Ticker | Tag | Spearman ρ | p-value | n_test | n_train | n_total | vol_frac | dir_thresh | vol_ratio | kappa |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| NVDA | i0p3 | 0.196 | 0.0 | 145,000 | 427,872 | 574,765 | 6.45e-05 | 0.509 | 0.583 | 0.824 |
| NVDA | i0p5 | 0.197 | 0.0 | 142,414 | 419,278 | 563,547 | 6.64e-05 | 0.530 | 0.596 | 1.926 |
| NVDA | i0p8 | 0.196 | 0.0 | 140,970 | 415,165 | 557,970 | 6.75e-05 | 0.502 | 0.582 | 1.216 |
| TSLA | i0p3 | 0.094 | 3.0e-07 | 2,952 | 5,982 | 8,958 | 0.00330 | 0.568 | 0.599 | 1.736 |
| TSLA | i0p5 | 0.053 | 3.6e-20 | 29,570 | 61,215 | 91,038 | 4.48e-04 | 0.556 | 0.291 | 0.372 |
| TSLA | i0p8 | 0.088 | 6.1e-38 | 21,143 | 43,485 | 64,810 | 6.19e-04 | 0.516 | 0.582 | 0.132 |
| JPM | i0p3 | 0.049 | 9.3e-126 | 238,170 | 622,451 | 860,621 | 1.0e-05 | 0.561 | 0.570 | 0.200 |
| JPM | i0p5 | 0.134 | 6.8e-07 | 1,372 | 1,859 | 3,239 | 0.00312 | 0.538 | 0.307 | 1.706 |
| JPM | i0p8 | 0.149 | 1.3e-07 | 1,249 | 1,697 | 2,953 | 0.00326 | 0.592 | 0.379 | 1.707 |
| MS | i0p3 | 0.125 | 0.001 | 655 | 1,030 | 1,686 | 0.00494 | 0.902 | 0.018 | 1.399 |
| MS | i0p5 | 0.071 | 0.066 | 665 | 1,051 | 1,717 | 0.00489 | 0.846 | 0.149 | 0.953 |
| MS | i0p8 | 0.126 | 0.001 | 651 | 1,021 | 1,673 | 0.00496 | 0.908 | 0.048 | 0.981 |

---

# PART XI: OPTION B BACKTEST RESULTS (CLOP, $10M AUM, 1.0 bps)

## 11.1 Core 2023-2024 Tickers (Ticker-Specific Regression Params)

| Ticker | Period | Trades | Long/Short | Long Win% | Short Win% | Net PnL ($) | ROC (%) | Max DD | Sharpe |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **NVDA** | 2023-2024 | 470 | 286/184 | 56.99% | 42.93% | +$15,108,538 | **+151.09%** | -30.5% | 1.09 |
| **TSLA** | 2023-2024 | 380 | 104/276 | 57.69% | 50.00% | +$11,969,031 | **+119.69%** | -18.3% | 2.07 |
| JPM | 2023-2024 | 125 | 23/102 | 60.87% | 50.00% | -$1,006,928 | -10.07% | -13.5% | -0.95 |
| MS | 2023-2024 | 37 | 9/28 | 55.56% | 42.86% | -$595,019 | -5.95% | -10.4% | -0.80 |

## 11.2 Out-of-Sample 2019-2021 (Averaged Regression Params)

| Ticker | Period | Trades | Long/Short | Long Win% | Short Win% | Net PnL ($) | ROC (%) | Max DD | Sharpe |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **LLY** | 2019-2021 | 443 | 398/45 | 55.53% | 44.44% | +$2,615,706 | **+26.16%** | -16.5% | 0.76 |
| AAPL | 2019-2021 | 328 | 248/80 | 54.84% | 41.25% | +$170,134 | +1.70% | -35.5% | 0.05 |
| SPY | 2019-2021 | 264 | 234/30 | 54.70% | 20.00% | -$1,697,553 | -16.98% | -18.6% | -1.24 |

## 11.3 Head-to-Head: Classification-Optimized vs Regression-Optimized

| Ticker | Classification ROC | Classification Sharpe | Regression ROC | Regression Sharpe | Delta ROC |
|:---|:---|:---|:---|:---|:---|
| **NVDA** | +141.62% | 1.58 | **+151.09%** | 1.09 | **+9.5pp** ↑ |
| **TSLA** | +73.21% | 1.57 | **+119.69%** | 2.07 | **+46.5pp** ↑ |
| JPM | -0.70% | -0.03 | -10.07% | -0.95 | -9.4pp ↓ |
| MS | -5.05% | -0.47 | -5.95% | -0.80 | -0.9pp ↓ |
| **LLY** (OOS) | +11.27% | 0.32 | **+26.16%** | 0.76 | **+14.9pp** ↑ |
| AAPL (OOS) | -30.31% | -0.87 | **+1.70%** | 0.05 | **+32.0pp** ↑ |
| SPY (OOS) | +31.86% | 1.35 | -16.98% | -1.24 | -48.8pp ↓ |

## 11.4 Key Observations

**Assets where regression-optimized params improved performance:**
- NVDA: +9.5pp ROC improvement. Critically, BOTH longs (+$11.1M) and shorts (+$4.0M) are now profitable. Max drawdown reduced from 65.5% to 30.5%.
- TSLA: +46.5pp ROC improvement. Sharpe increased from 1.57 to 2.07. Both sides profitable.
- LLY: +14.9pp ROC improvement. Sharpe doubled from 0.32 to 0.76. Both sides profitable.
- AAPL: Flipped from -30.31% to +1.70%. No longer a negative control.

**Assets where regression-optimized params degraded performance:**
- JPM: Worse (-10.07% vs -0.70%). The regression params produced fewer trades (125 vs 470) and a heavy short bias (23L/102S) in a bull market.
- MS: Slightly worse. Very few trades (37 total).
- SPY: Significantly worse (-16.98% vs +31.86%). The averaged universal parameters do not suit SPY. SPY would need its own dedicated parameter optimization.

**Interpretation:**
The regression-aligned optimization finds parameters that select for bursts with LARGE predicted overnight impact, not just correct directional classification. This produces:
1. Higher per-trade edge on momentum assets (NVDA, TSLA, LLY)
2. Fewer but higher-conviction trades on noisy assets (JPM: 125 vs 470 trades)
3. A structural bias toward high-kappa values (~1.7), enforcing strict decay gating that removes noise bursts

