# Appendix: Passive Limit Order Burst Analysis

## 1. Objective and Motivation
The primary pipeline of this paper focuses on aggressive execution bursts (market orders crossing the spread) and their ability to generate mechanical price impact. As a comparative experiment, we sought to isolate and evaluate **passive institutional intent**—specifically, whether bursts of passive limit order submissions (adds to the order book) contain predictive informational alpha regarding short-term and overnight price drift.

## 2. Methodology Divergence

Analyzing passive submissions required several fundamental architectural changes compared to the aggressive execution pipeline:

### 2.1. Burst Detection and Triggers
*   **Type 1 Triggers Only**: The Hawkes process was strictly configured to excite only on `Type 1` (limit order submission) messages at the Best Bid and Offer (BBO) and the next two levels (L1–L3). Market orders (`Type 4/5`) and cancellations (`Type 2/3`) were tracked as feature metrics but did not trigger burst detection.
*   **Directional Imbalance**: Burst direction was defined by the ratio of Bid submissions to Ask submissions. A burst was labeled bullish (+1) if the `BidRatio` exceeded a directional threshold, and bearish (-1) if the `AskRatio` exceeded the threshold.
*   **ADV Denominator Correction**: In the aggressive pipeline, volume fraction thresholds (`vol_frac`) are scaled against total daily volume. For passive submissions, HFT market-making bots generate millions of micro-quotes (quote spam), artificially bloating the raw submission volume. To ground the threshold in economic reality, the ADV baseline was calculated using **Traded Volume (Types 4/5)**.

### 2.2. Strict Look-Ahead Avoidance
Passive bursts do not generate immediate mechanical impact. Therefore, the target variables are pure price drifts (Close-to-Open gaps or intraday N-minute drifts). To prevent look-ahead bias, **all features were strictly restricted to information available at the exact millisecond the burst terminated**. 
*   Features like `D_b` (which averages future price drifts) and `QueueExhaustionRate` (which looks at post-burst book state) were explicitly excluded.
*   The feature set consisted of 30 variables, including burst aggregates (volume, duration), book snapshots (spread, depth, imbalance), pre-burst lookbacks (volatility, momentum), and cancellation interactions (`PreBurstCancelRate`).

### 2.3. Transaction Cost Modeling
Unlike aggressive bursts, which cross the spread to enter and generate momentum, a strategy trading on passive intent must actively pay the spread to enter the position. 
*   For **Overnight targets** (Close-to-Open), the model pays a strict **3.0 bps round-trip friction** to account for the slippage of crossing the Market-On-Close (MOC) and Market-On-Open (MOO) auctions.
*   For **Intraday targets** (3m, 5m, 10m), the model pays the **actual BBO spread** at the exact time of burst termination (typically 0.74 bps for AAPL, 1.55 bps for JPM, up to 3.31 bps for LLY).

### 2.4. Target Formulations
1.  **Overnight (Close-to-Open)**: $y = \text{arcsinh}\left(\text{direction} \times \frac{\text{Open}_{t+1} - \text{CloseMid}_t}{\text{CloseMid}_t} \times 10000\right)$
2.  **Intraday (N-minute)**: $y = \text{arcsinh}\left(\text{direction} \times \frac{\text{Mid}_{t+N} - \text{EndPrice}_t}{\text{EndPrice}_t} \times 10000\right)$

---

## 3. Empirical Results

We conducted rigorous Out-of-Sample (OOS) walk-forward backtests on five tickers (JPM, MS, NVDA, LLY, AAPL) using both overnight and intraday horizons.

### 3.1. The Structural Absence of Passive Flow in TSLA/SPY
The first empirical finding was structural: ultra-liquid, momentum-driven assets like TSLA and SPY simply do not contain detectable passive accumulation footprints. Even after adjusting the ADV baseline to bypass HFT noise, the pipeline detected fewer than 50 valid passive bursts for TSLA over two years. Institutions trading these assets appear to almost exclusively utilize aggressive crossing strategies.

### 3.2. Overnight Horizon (Close-to-Open) Results
Using a strict Leave-One-Out (LOO) consensus parameter set to prevent information leakage, the overnight strategy yielded negative Sharpe ratios across all tested assets.

| Ticker | Gated Trades | Win Rate | Mean Capture (bps) | Ann. Sharpe | Spearman ρ |
|--------|--------------|----------|-------------------|-------------|------------|
| JPM | 34,908 | 47.7% | -2.56 bps | -76.35 | 0.0887 (p<0.001) |
| MS | 36,450 | 42.3% | -3.07 bps | -81.81 | -0.0050 |
| NVDA | 1,520 | 39.5% | -3.56 bps | -35.13 | -0.0524 |
| LLY | 125,228 | 45.6% | -3.05 bps | -140.30 | 0.0006 |
| AAPL | 140 | 45.0% | -3.21 bps | -9.50 | -0.0039 |

**Analysis**: The Spearman correlation for JPM (0.0887) is highly statistically significant, indicating that the passive footprint *does* successfully predict the direction of the overnight gap. However, the raw expected value of this drift is only ~0.44 bps. When the 3.0 bps MOC/MOO auction friction is applied, the strategy bleeds money.

### 3.3. Intraday Horizon Results
To test if the alpha decays before the overnight close, we evaluated 3-minute, 5-minute, and 10-minute intraday holds, applying the actual BBO spread as the transaction cost constraint. We tested multiple ML gating mechanisms, including "Cost-Aware Gating" (only entering trades where the predicted magnitude exceeds the actual spread).

**0 out of 80 tested intraday strategies generated positive post-cost alpha.** 

| Ticker | Best Strategy | Win Rate | Mean Capture (bps) | Spearman ρ |
|--------|---------------|----------|-------------------|------------|
| JPM | Cost-Aware Gate (10m) | 47.2% | -0.63 bps | 0.0253 |
| MS | Cost-Aware Gate (10m) | 46.8% | -1.09 bps | -0.0084 |
| NVDA | Cost-Aware Gate (10m) | 46.2% | -1.10 bps | 0.0004 |
| LLY | Cost-Aware Gate (5m) | 45.5% | -0.72 bps | 0.0047 |
| AAPL | Cost-Aware Gate (3m) | 46.4% | -0.50 bps | 0.0156 |

**Analysis**: While Cost-Aware Gating minimized losses by successfully avoiding wide-spread regimes, the residual informational signal was simply too diffuse. Even in a tick-constrained asset like AAPL with an exceptionally tight median spread (0.74 bps), the strategy could not overcome execution friction.

---

## 4. Conclusion

The passive limit order experiment yields a definitive negative result, which serves as a powerful contrast to the aggressive execution pipeline. 

While passive institutional accumulation footprints contain statistically significant directional information (as evidenced by JPM's rank correlation), they fundamentally lack the mechanical momentum transfer characteristic of aggressive market orders. Because the limit order book continuously absorbs and diffuses passive intent, the resulting price drift is simply too weak to survive real-world execution friction—whether that friction is the 3.0 bps closing auction or the 0.74 bps continuous spread. 

To generate actionable microstructure alpha, algorithms must track aggressive executions that actively force price discovery.
