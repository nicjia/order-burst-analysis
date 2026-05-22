#!/usr/bin/env python3
"""
passive_alpha_lab_v2.py — Look-Ahead-Safe Strategy Iteration

CRITICAL DESIGN CONSTRAINT:
  For short-horizon intraday targets (1m/3m/5m/10m), we may ONLY use features
  that are known at the EXACT MOMENT the burst terminates. Any feature that
  requires information from after the burst end time is a look-ahead violation.

LOOK-AHEAD AUDIT:
  ✓ SAFE: Volume, SubmissionCount, BidSubCount, AskSubCount, BidSubVolume,
          AskSubVolume, BidRatio, AskRatio, MinMaxVolRatio, Spread, BidVolBest,
          AskVolBest, BidDepth5, AskDepth5, BookImbalance, Volatility60s,
          Momentum5s/30s/60s, TradeCount5m, TradeVolume5m, SubmissionSizeVariance,
          RoundLotPct, HawkesPeakIntensity, CancelCount, CancelVolume,
          BidCancelCount, AskCancelCount, BidCancelVolume, AskCancelVolume,
          CancelRatio, PreBurstCancelRate, Duration, StartPrice, EndPrice,
          Direction
  ✗ REMOVED: D_b (requires post-burst mid prices), QueueExhaustionRate (post-burst),
             PeakImpact/PeakPrice (uses tau_max window that can extend past burst end),
             Mid_1m/3m/5m/10m (these are targets, never features),
             CloseMid, Perm_* (target variables)

FILTERING AUDIT:
  ✓ vol_frac     → filters on Volume (known at burst end)
  ✓ dir_thresh   → filters on BidRatio/AskRatio (known at burst end)
  ✓ vol_ratio    → filters on MinMaxVolRatio (known at burst end)
  ✓ max_cancel   → filters on CancelRatio (known at burst end)
  ✗ kappa        → filters on D_b (look-ahead!) — NOT USED HERE

METHODOLOGY:
  1. Load all directed bursts (Direction != 0)
  2. No Optuna filtering — we use ALL directed bursts (the ML model itself must
     learn to distinguish signal from noise; pre-filtering with physical params
     is equivalent to the Optuna regime and already tested)
  3. Construct intraday targets: arcsinh(direction × (Mid_Nm - EndPrice)/EndPrice × 10000)
  4. Walk-forward SGD: train on all months < test month, predict test month
  5. Evaluate with actual per-burst spread cost (Spread/EndPrice × 10000 bps)
  6. Multiple gating strategies: quartile (25%), tight (5/10%), cost-aware
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── STRICTLY LOOK-AHEAD-SAFE FEATURES ──
# Every feature here is known at burst end time. No exceptions.
SAFE_FEATURE_COLS = [
    # Burst-internal aggregates (accumulated during burst lifecycle)
    'Volume', 'SubmissionCount', 'BidSubCount', 'AskSubCount',
    'BidSubVolume', 'AskSubVolume', 'BidRatio', 'AskRatio', 'MinMaxVolRatio',
    'Duration',
    # Order book snapshot at burst end
    'Spread', 'BidVolBest', 'AskVolBest', 'BidDepth5', 'AskDepth5', 'BookImbalance',
    # Pre-burst lookback features (trailing windows BEFORE burst)
    'Volatility60s', 'Momentum5s', 'Momentum30s', 'Momentum60s',
    'TradeCount5m', 'TradeVolume5m',
    # Burst-internal microstructure
    'SubmissionSizeVariance', 'RoundLotPct', 'HawkesPeakIntensity',
    # Cancellation features (events during burst lifecycle)
    'CancelCount', 'CancelVolume', 'BidCancelCount', 'AskCancelCount',
    'BidCancelVolume', 'AskCancelVolume', 'CancelRatio',
    # Pre-burst cancel rate (50ms window BEFORE burst start)
    'PreBurstCancelRate',
]

# Columns explicitly EXCLUDED due to look-ahead:
# 'D_b'                 — requires post-burst mid prices (1m/3m/5m/10m average)
# 'PeakImpact'          — uses tau_max window that can extend past burst end
# 'QueueExhaustionRate' — post-burst calculation
# 'CloseMid'            — end-of-day price (future)
# 'Mid_1m/3m/5m/10m'    — these ARE the targets
# 'Perm_tCLOSE/CLOP/CLCL' — target variables


def load_data(ticker):
    path = f"results/passive/passive_bursts_{ticker}_raw_filtered.csv"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found"); sys.exit(1)
    cols = pd.read_csv(path, nrows=0).columns
    float_cols = [c for c in cols if c not in ('Date','Time','Ticker')]
    df = pd.read_csv(path, dtype={c:'float32' for c in float_cols}, low_memory=True)
    df['Date'] = df['Date'].astype(str).str.replace('-','').astype(int)
    return df


def get_safe_features(df):
    """Extract ONLY look-ahead-safe features."""
    cols = [c for c in SAFE_FEATURE_COLS if c in df.columns]
    X = df[cols].fillna(0).values.astype(np.float32)
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0), cols


def walk_forward_predict(df, target_col):
    """Strictly walk-forward SGD. Returns (y_true, y_pred, eval_df)."""
    df = df.copy()
    df['Month'] = pd.to_datetime(df['Date'].astype(str)).dt.to_period('M')
    months = sorted(df['Month'].unique())
    if len(months) < 3:
        return None, None, None

    all_yt, all_yp, eval_indices = [], [], []
    for i in range(2, len(months)):
        train_df = df[df['Month'].isin(months[:i])]
        test_df  = df[df['Month'] == months[i]]
        if len(train_df) < 30 or len(test_df) < 5:
            continue

        y_tr = train_df[target_col].fillna(0).values
        y_te = test_df[target_col].fillna(0).values
        X_tr, _ = get_safe_features(train_df)
        X_te, _ = get_safe_features(test_df)

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        m = SGDRegressor(loss='huber', epsilon=1.35, penalty='l2', alpha=0.001,
                         learning_rate='adaptive', eta0=0.001, max_iter=1000, random_state=42)
        m.fit(X_tr, y_tr)
        all_yt.extend(y_te)
        all_yp.extend(m.predict(X_te))
        eval_indices.extend(test_df.index.tolist())

    if len(all_yt) < 30:
        return None, None, None
    return np.array(all_yt), np.array(all_yp), df.loc[eval_indices]


def evaluate(y_true, y_pred, cost_bps, percentile=25, label=""):
    p_lo = np.percentile(y_pred, percentile)
    p_hi = np.percentile(y_pred, 100 - percentile)
    mask = (y_pred <= p_lo) | (y_pred >= p_hi)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 10: return None

    c = cost_bps[mask] if isinstance(cost_bps, np.ndarray) else cost_bps
    capture = np.sign(yp) * yt - c
    rho, pval = spearmanr(yt, yp)
    daily_n = len(capture) / 500
    sharpe = (capture.mean() / max(capture.std(), 1e-9)) * np.sqrt(252 * daily_n)

    return {
        'label': label, 'n_trades': len(capture),
        'win_rate': float(round((capture > 0).mean(), 4)),
        'mean_capture_bps': float(round(capture.mean(), 4)),
        'sum_capture': float(round(capture.sum(), 2)),
        'sharpe': float(round(sharpe, 2)),
        'spearman_rho': float(round(rho, 4)) if not np.isnan(rho) else 0.0,
        'p_value': float(round(pval, 6)) if not np.isnan(pval) else 1.0,
    }


def cost_aware_evaluate(y_true, y_pred, spread_bps, label=""):
    mask = np.abs(y_pred) > spread_bps
    if mask.sum() < 10: return None
    yt, yp, sp = y_true[mask], y_pred[mask], spread_bps[mask]
    capture = np.sign(yp) * yt - sp
    rho, pval = spearmanr(yt, yp)
    daily_n = len(capture) / 500
    sharpe = (capture.mean() / max(capture.std(), 1e-9)) * np.sqrt(252 * daily_n)
    return {
        'label': label, 'n_trades': int(mask.sum()),
        'win_rate': float(round((capture > 0).mean(), 4)),
        'mean_capture_bps': float(round(capture.mean(), 4)),
        'sum_capture': float(round(capture.sum(), 2)),
        'sharpe': float(round(sharpe, 2)),
        'spearman_rho': float(round(rho, 4)) if not np.isnan(rho) else 0.0,
    }


def directional_baseline(df, horizon_col, cost_bps_val):
    """No-model baseline: trade every directed burst's direction, pay median spread."""
    end_p = df['EndPrice'].values
    mid_h = df[horizon_col].values
    direction = df['Direction'].values
    drift_bps = direction * (mid_h - end_p) / end_p * 10000
    capture = drift_bps - cost_bps_val
    return {
        'label': f'F-Baseline-{horizon_col}', 'n_trades': len(capture),
        'win_rate': float(round((capture > 0).mean(), 4)),
        'mean_capture_bps': float(round(capture.mean(), 4)),
        'sum_capture': float(round(capture.sum(), 2)),
    }


def run_ticker(ticker, results):
    print(f"\n{'='*60}")
    print(f"  ALPHA LAB v2 (Look-Ahead-Safe) — {ticker}")
    print(f"{'='*60}")

    df = load_data(ticker)
    directed = df[df['Direction'] != 0].copy()
    print(f"  Total: {len(df):,}  Directed: {len(directed):,} ({100*len(directed)/len(df):.1f}%)")

    if len(directed) < 100:
        print(f"  SKIP: <100 directed bursts")
        results.append({'ticker': ticker, 'label': 'SKIP', 'reason': f'{len(directed)} directed'})
        return

    spread_bps = (directed['Spread'].values / directed['EndPrice'].values * 10000).astype(np.float32)
    median_spread = float(np.median(spread_bps))
    print(f"  Median spread: {median_spread:.2f} bps")
    print(f"  Features used: {len([c for c in SAFE_FEATURE_COLS if c in directed.columns])}")

    # F: No-model baselines
    for h in ['Mid_1m', 'Mid_3m', 'Mid_5m', 'Mid_10m']:
        r = directional_baseline(directed, h, median_spread)
        r['ticker'] = ticker
        results.append(r)
        print(f"  [F] {r['label']}: WR={r['win_rate']:.1%} Cap={r['mean_capture_bps']:.2f}")

    # ML strategies per horizon
    for horizon in ['Mid_3m', 'Mid_5m', 'Mid_10m']:
        tcol = f'_tgt_{horizon}'
        directed[tcol] = np.arcsinh(
            directed['Direction'] * (directed[horizon] - directed['EndPrice']) / directed['EndPrice'] * 10000
        )

        y_true, y_pred, df_eval = walk_forward_predict(directed, tcol)
        if y_true is None:
            print(f"  [{horizon}] Walk-forward failed (insufficient data)")
            continue

        eval_spread = (df_eval['Spread'].values / df_eval['EndPrice'].values * 10000).astype(np.float32)

        # A: Quartile gate
        r = evaluate(y_true, y_pred, eval_spread, percentile=25, label=f'A-Q25-{horizon}')
        if r: r['ticker'] = ticker; results.append(r)
        if r: print(f"  [A] {r['label']}: WR={r['win_rate']:.1%} Cap={r['mean_capture_bps']:.2f} ρ={r['spearman_rho']}")

        # B: Cost-aware
        r = cost_aware_evaluate(y_true, y_pred, eval_spread, label=f'B-CostAware-{horizon}')
        if r: r['ticker'] = ticker; results.append(r)
        if r: print(f"  [B] {r['label']}: WR={r['win_rate']:.1%} Cap={r['mean_capture_bps']:.2f} n={r['n_trades']}")

        # C: Tight 10%
        r = evaluate(y_true, y_pred, eval_spread, percentile=10, label=f'C-Tight10-{horizon}')
        if r: r['ticker'] = ticker; results.append(r)
        if r: print(f"  [C] {r['label']}: WR={r['win_rate']:.1%} Cap={r['mean_capture_bps']:.2f}")

        # C2: Tight 5%
        r = evaluate(y_true, y_pred, eval_spread, percentile=5, label=f'C-Tight05-{horizon}')
        if r: r['ticker'] = ticker; results.append(r)
        if r: print(f"  [C2] {r['label']}: WR={r['win_rate']:.1%} Cap={r['mean_capture_bps']:.2f}")


def write_report(results, out_path):
    lines = [
        "# Passive Alpha Lab v2 — Look-Ahead-Safe Results",
        "",
        "## Methodology",
        "",
        "### Signal Generation",
        "- **Trigger**: Hawkes process excites on Type 1 (limit order submissions) at BBO ± 3 tick levels",
        "- **Burst lifecycle**: Begins when Hawkes intensity > threshold; ends when intensity decays below threshold",
        "- **Direction**: Determined by BidRatio vs AskRatio during the burst (majority side wins)",
        "- **ADV Baseline**: 14-day trailing average of **Traded Volume** (Types 4/5), NOT submission volume",
        "",
        "### Look-Ahead Constraint",
        "For intraday predictions, we use ONLY features available at the exact moment",
        "the burst terminates. The following are explicitly **excluded**:",
        "- `D_b` (directional balance) — requires post-burst mid prices",
        "- `PeakImpact` — uses a tau_max window that can extend past burst end",
        "- `QueueExhaustionRate` — post-burst calculation",
        "- `CloseMid`, `Perm_*` — future/target variables",
        "",
        "### Feature Set (30 features, all known at burst end)",
        "| Category | Features |",
        "|----------|----------|",
        "| Burst aggregates | Volume, SubmissionCount, BidSubCount, AskSubCount, BidSubVolume, AskSubVolume, BidRatio, AskRatio, MinMaxVolRatio, Duration |",
        "| Book snapshot | Spread, BidVolBest, AskVolBest, BidDepth5, AskDepth5, BookImbalance |",
        "| Pre-burst lookback | Volatility60s, Momentum5s/30s/60s, TradeCount5m, TradeVolume5m |",
        "| Microstructure | SubmissionSizeVariance, RoundLotPct, HawkesPeakIntensity |",
        "| Cancellations | CancelCount, CancelVolume, Bid/AskCancelCount, Bid/AskCancelVolume, CancelRatio, PreBurstCancelRate |",
        "",
        "### Target Variable",
        "$$y = \\text{arcsinh}\\left(\\text{direction} \\times \\frac{\\text{Mid}_{t+N} - \\text{EndPrice}}{\\text{EndPrice}} \\times 10000\\right)$$",
        "Where N ∈ {3min, 5min, 10min} and EndPrice is the mid-price at burst termination.",
        "",
        "### Transaction Cost",
        "Each trade pays the **actual BBO spread at burst end time**, converted to basis points:",
        "$$c_i = \\frac{\\text{Spread}_i}{\\text{EndPrice}_i} \\times 10000$$",
        "This is the cost of crossing the spread once to enter the position. We assume exit",
        "at the mid-price N minutes later (conservative: no exit spread charged).",
        "",
        "### Model & Validation",
        "- **Model**: SGDRegressor (Huber loss, L2 penalty, α=0.001, adaptive learning rate)",
        "- **Validation**: Strict walk-forward — train on all months before test month, predict test month",
        "- **No filtering**: ALL directed bursts are used (no Optuna pre-filtering)",
        "",
        "### Entry Strategies",
        "| Strategy | Entry Criterion |",
        "|----------|----------------|",
        "| F (Baseline) | Trade every directed burst, no model |",
        "| A (Quartile) | Enter top/bottom 25% of predicted drift |",
        "| B (Cost-Aware) | Enter when |predicted drift| > actual spread |",
        "| C (Tight 10%) | Enter top/bottom 10% of predicted drift |",
        "| C2 (Tight 5%) | Enter top/bottom 5% of predicted drift |",
        "",
        "---",
        "",
        "## Results",
        "",
        "| Ticker | Strategy | N Trades | Win Rate | Mean Cap (bps) | Sum Cap | Sharpe | Spearman ρ |",
        "|--------|----------|----------|----------|----------------|---------|--------|------------|",
    ]

    for r in sorted(results, key=lambda x: (x.get('ticker',''), x.get('label',''))):
        if r.get('label') == 'SKIP':
            lines.append(f"| {r['ticker']} | SKIP | — | — | — | — | — | {r.get('reason','')} |")
        else:
            wr = r.get('win_rate', 0)
            mc = r.get('mean_capture_bps', 0)
            sc = r.get('sum_capture', 0)
            sh = r.get('sharpe', '—')
            rho = r.get('spearman_rho', '—')
            pv = r.get('p_value', '—')
            bold = "**" if mc > 0 else ""
            lines.append(
                f"| {r['ticker']} | {r['label']} | {r.get('n_trades','—'):,} "
                f"| {wr:.1%} | {bold}{mc:.2f}{bold} | {sc:,.0f} | {sh} | {rho} |"
            )

    # Profitable summary
    profitable = [r for r in results if r.get('mean_capture_bps', -999) > 0]
    lines.extend(["", "## Profitable Strategies (Post-Cost)", ""])
    if profitable:
        profitable.sort(key=lambda x: x['mean_capture_bps'], reverse=True)
        lines.append("| Ticker | Strategy | N Trades | Win Rate | Mean Cap (bps) | Sum Cap |")
        lines.append("|--------|----------|----------|----------|----------------|---------|")
        for r in profitable:
            lines.append(
                f"| {r['ticker']} | {r['label']} | {r['n_trades']:,} "
                f"| {r['win_rate']:.1%} | **{r['mean_capture_bps']:.2f}** | {r['sum_capture']:,.0f} |"
            )
    else:
        lines.append("**No strategies achieved positive post-cost expected value.**")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    # Train tickers + OOS tickers
    TRAIN = ['JPM', 'MS', 'NVDA']
    OOS   = ['LLY', 'AAPL']
    ALL   = TRAIN + OOS

    results = []
    for ticker in ALL:
        try:
            run_ticker(ticker, results)
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({'ticker': ticker, 'label': 'SKIP', 'reason': str(e)})

    write_report(results, "passive/ALPHA_LAB_V2_RESULTS.md")

    # Also dump raw JSON for appendix generation
    with open("results/oos_passive/alpha_lab_v2_raw.json", 'w') as f:
        json.dump(results, f, indent=2)

    profitable = [r for r in results if r.get('mean_capture_bps', -999) > 0]
    print(f"\nTotal strategies: {len(results)}")
    print(f"Profitable: {len(profitable)}")
