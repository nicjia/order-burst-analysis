# Passive Limit Order Burst Analysis — Intraday Alpha Lab (v2)

## Executive Summary

After correcting for look-ahead bias, **0 out of 80 tested strategies generated positive post-cost alpha** across all 5 tickers (JPM, MS, NVDA, LLY, AAPL). The previous v1 results showing 84-91% win rates were entirely an artifact of using the `D_b` (directional balance) feature, which is computed from post-burst mid-prices — the very values we were trying to predict. Once this look-ahead violation was removed, the passive limit order signal collapsed to noise.

---

## Methodology

### 1. Data Source & Burst Detection

- **Source**: LOBSTER L3 message data (2019–2024), processed through a custom C++ Hawkes process detector
- **Trigger**: Hawkes intensity excitation on **Type 1 (limit order submissions)** at BBO ± 3 tick levels
- **Burst lifecycle**: Begins when Hawkes intensity crosses threshold; terminates when intensity decays below threshold
- **Direction**: Majority submission side (BidRatio vs AskRatio) determines burst direction (+1 = buy, -1 = sell)
- **ADV Baseline**: 14-day trailing average of **Traded Volume** (Types 4/5 only), grounding `vol_frac` thresholds in economic reality rather than HFT quote spam

### 2. Look-Ahead Constraint (Critical Fix from v1)

For intraday predictions, **only features known at the exact moment the burst terminates** may be used. The following were explicitly **excluded** in v2:

| Excluded Feature | Reason |
|-----------------|--------|
| `D_b` (directional balance) | Computed from post-burst mid-prices at 1m/3m/5m/10m — **this IS the target variable** |
| `PeakImpact` | Uses `tau_max=10s` window that can extend past burst end |
| `QueueExhaustionRate` | Requires post-burst order book state |
| `CloseMid`, `Perm_*` | End-of-day / next-day target variables |

**This was the critical flaw in v1.** The `D_b` feature is computed as the average of `Volume × Direction × (Mid_Nm - StartPrice)` across N=1,3,5,10 minutes. By including `D_b` as a predictor of 5-minute drift, the model was effectively given the answer in the feature set, producing artificially high win rates (84-91%).

### 3. Feature Set (30 Features, All Known at Burst End)

| Category | Features | Count |
|----------|----------|-------|
| Burst aggregates | Volume, SubmissionCount, BidSubCount, AskSubCount, BidSubVolume, AskSubVolume, BidRatio, AskRatio, MinMaxVolRatio, Duration | 10 |
| Book snapshot (at burst end) | Spread, BidVolBest, AskVolBest, BidDepth5, AskDepth5, BookImbalance | 6 |
| Pre-burst lookback | Volatility60s, Momentum5s, Momentum30s, Momentum60s, TradeCount5m, TradeVolume5m | 6 |
| Microstructure | SubmissionSizeVariance, RoundLotPct, HawkesPeakIntensity | 3 |
| Cancellation activity | CancelCount, CancelVolume, BidCancelCount, AskCancelCount, BidCancelVolume, AskCancelVolume, CancelRatio, PreBurstCancelRate | 8 |

### 4. Target Variable

For each intraday horizon N ∈ {3min, 5min, 10min}:

$$y_i = \text{arcsinh}\!\left(\text{direction}_i \times \frac{\text{Mid}_{t+N} - \text{EndPrice}_i}{\text{EndPrice}_i} \times 10000\right)$$

where `EndPrice` is the BBO midpoint at burst termination and `Mid_{t+N}` is the midpoint N minutes later. The `arcsinh` transformation preserves sign while compressing outliers.

### 5. Transaction Cost Model

Each trade pays the **actual BBO spread at burst end time**, converted to basis points:

$$c_i = \frac{\text{Spread}_i}{\text{EndPrice}_i} \times 10000$$

This represents the cost of crossing the spread to enter a position at burst detection time. We conservatively assume mid-price exit (no exit spread charged).

| Ticker | Median Spread (bps) |
|--------|-------------------|
| JPM | 1.55 |
| MS | 2.03 |
| NVDA | 2.32 |
| LLY | 3.31 |
| AAPL | 0.74 |

### 6. Model & Validation

- **Model**: `SGDRegressor` (Huber loss, L2 penalty, α=0.001, adaptive learning rate, max_iter=1000)
- **Validation**: Strict walk-forward — for each test month $m$, train on all months $< m$
- **No pre-filtering**: ALL directed bursts used (no Optuna-optimized physical parameter gates)

### 7. Entry Strategies Tested

| ID | Strategy | Entry Criterion |
|----|----------|----------------|
| F | Baseline | Trade every directed burst, no ML model |
| A | Quartile (25%) | Enter when prediction falls in top/bottom 25% |
| B | Cost-Aware | Enter only when |predicted drift| > actual per-burst spread |
| C | Tight (10%) | Enter when prediction falls in top/bottom 10% |
| C2 | Tight (5%) | Enter when prediction falls in top/bottom 5% |

---

## Results

### JPM (581,781 total bursts, 301,979 directed, median spread 1.55 bps)

| Strategy | N Trades | Win Rate | Mean Capture (bps) | Spearman ρ |
|----------|----------|----------|-------------------|------------|
| F-Baseline-Mid_5m | 301,979 | 41.6% | -1.51 | — |
| F-Baseline-Mid_10m | 301,979 | 44.1% | -1.54 | — |
| A-Q25-Mid_5m | ~147K | 29.8% | -2.20 | 0.024 |
| B-CostAware-Mid_10m | 2,519 | 47.2% | -0.63 | — |
| C-Tight10-Mid_10m | ~29K | 30.7% | -2.57 | — |

### MS (761,274 total bursts, 414,071 directed, median spread 2.03 bps)

| Strategy | N Trades | Win Rate | Mean Capture (bps) | Spearman ρ |
|----------|----------|----------|-------------------|------------|
| F-Baseline-Mid_5m | 414,071 | 40.4% | -1.96 | — |
| A-Q25-Mid_10m | ~193K | 32.7% | -2.58 | -0.008 |
| B-CostAware-Mid_10m | 263 | 46.8% | -1.09 | — |

### NVDA (58,973 total bursts, 17,000 directed, median spread 2.32 bps)

| Strategy | N Trades | Win Rate | Mean Capture (bps) | Spearman ρ |
|----------|----------|----------|-------------------|------------|
| F-Baseline-Mid_5m | 17,000 | 42.7% | -2.37 | — |
| A-Q25-Mid_10m | ~8.3K | 34.4% | -2.58 | 0.000 |
| B-CostAware-Mid_10m | 772 | 46.2% | -1.10 | — |

### LLY (1,979,050 total bursts, 1,865,282 directed, median spread 3.31 bps — Pure OOS)

| Strategy | N Trades | Win Rate | Mean Capture (bps) | Spearman ρ |
|----------|----------|----------|-------------------|------------|
| F-Baseline-Mid_10m | 1,865,282 | 40.8% | -3.16 | — |
| A-Q25-Mid_10m | ~900K | 19.6% | -5.33 | 0.001 |
| B-CostAware-Mid_10m | 4,733 | 41.8% | -0.93 | — |

### AAPL (61,322 total bursts, 55,956 directed, median spread 0.74 bps — Pure OOS)

| Strategy | N Trades | Win Rate | Mean Capture (bps) | Spearman ρ |
|----------|----------|----------|-------------------|------------|
| F-Baseline-Mid_5m | 55,956 | 44.3% | -0.74 | — |
| A-Q25-Mid_10m | ~27K | 43.8% | -0.96 | -0.009 |
| B-CostAware-Mid_10m | 601 | 42.8% | -0.74 | — |

---

## Analysis

### 1. Why the v1 Results Were a Mirage

The v1 alpha lab reported 84-91% win rates and +1.5 bps capture using Strategy B (Cost-Aware Gating). This was entirely caused by including `D_b` as a feature. `D_b` is defined as:

$$D_b = \frac{1}{K}\sum_{k \in \{1,3,5,10\}} V_b \cdot d_b \cdot (\text{Mid}_{t+k} - P_{\text{start}})$$

This is literally a weighted average of the future mid-price drifts — the same quantities we are trying to predict. Including it as a feature constitutes a severe information leak, creating an autoregressive loop where the model trivially "predicts" the target from a linear combination of future prices.

### 2. The Passive Signal is Real but Sub-Spread

Across all tickers, the **directional baseline** (Strategy F) shows consistent sub-50% win rates (24-46%), meaning passive burst direction alone predicts the subsequent drift at rates barely above chance after accounting for the bid-ask spread. The ML model's Spearman correlations (0.00 to 0.025) are statistically significant in some cases (JPM: p<0.001) but economically meaningless — the predicted alpha magnitude is consistently smaller than the transaction cost.

### 3. Cost-Aware Gating (B) Reduces Losses but Cannot Generate Profit

Strategy B achieves the smallest losses by filtering out trades where predicted alpha < spread. In the best case (JPM, 10m horizon), it achieves 47.2% win rate and -0.63 bps per trade. This means the model *does* have some discriminative power — it successfully avoids the worst trades — but the residual signal is insufficient to overcome the spread.

### 4. AAPL: Tick-Constrained Regime

AAPL's exceptionally tight spread (0.74 bps) means the cost barrier is lowest, yet it still cannot generate profit. This suggests the signal degradation is not purely a friction problem — the passive submission signal genuinely does not predict short-horizon price moves at the tick-constrained level.

### 5. LLY: High Spread Regime (Pure OOS)

LLY's wide spread (3.31 bps) creates an insurmountable friction barrier. Even with cost-aware gating, the model loses 0.93 bps per trade. LLY also has the lowest win rates across all strategies, suggesting the passive signal is weakest in pharmaceutical/healthcare stocks.

---

## Conclusion

**Passive limit order submissions do not contain actionable short-horizon alpha** once look-ahead bias is properly eliminated. While the aggressive trade-based burst detector (the main pipeline) benefits from mechanical price impact that persists beyond the burst, passive submissions lack this momentum transfer mechanism. The informational content of a passive bid accumulation is too diffuse and too easily absorbed by the continuous limit order book to generate a tradeable edge within 3-10 minutes.

The key lesson: `D_b` must never be used as a feature for intraday prediction targets. It may only be used as a filtering criterion for overnight targets where the execution window (MOC to MOO) is temporally separated from the feature calculation window by several hours.