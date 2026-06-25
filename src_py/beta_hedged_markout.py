#!/usr/bin/env python3
"""
beta_hedged_markout.py — Factor Regression: Alpha vs Beta Decomposition

Regresses the strategy's daily PnL against:
  1. The underlying asset's daily return (stock beta)
  2. SPY daily return (market beta)
  3. FF5 + MOM factor returns (if available)

Proves the alpha is orthogonal to passive directional exposure.

Uses Newey-West HAC standard errors throughout.

Usage:
    # Single-ticker
    python3 src_py/beta_hedged_markout.py \
        --trades-csv results/sgd_backtests_oos/NVDA_reg_clop_debug_trades.csv \
        --close-csv close_all.csv \
        --ticker NVDA

    # With Fama-French factors
    python3 src_py/beta_hedged_markout.py \
        --trades-csv results/sgd_backtests_oos/NVDA_reg_clop_debug_trades.csv \
        --close-csv close_all.csv \
        --ticker NVDA \
        --factor-csv data/ff5_mom_daily.csv
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def newey_west_se(y, X, lags=None):
    """
    OLS with Newey-West HAC standard errors.

    Parameters
    ----------
    y : array (T,)
    X : array (T, k) — should include a constant column
    lags : int or None — Newey-West bandwidth; None = auto (floor(4*(T/100)^{2/9}))

    Returns
    -------
    dict with keys: beta, se, t_stat, p_value, r_squared
    """
    T, k = X.shape
    if lags is None:
        lags = max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))

    # OLS estimates
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # Newey-West covariance
    S = np.zeros((k, k))
    for j in range(lags + 1):
        if j == 0:
            Gj = (X * resid[:, None]).T @ (X * resid[:, None]) / T
        else:
            w = 1.0 - j / (lags + 1.0)
            Gj_forward = (X[j:] * resid[j:, None]).T @ (X[:-j] * resid[:-j, None]) / T
            Gj = w * (Gj_forward + Gj_forward.T)
        S += Gj

    cov = XtX_inv @ S @ XtX_inv * T
    se = np.sqrt(np.diag(cov))
    t_stat = beta / np.where(se > 0, se, 1e-12)
    p_value = 2.0 * (1.0 - scipy_stats.t.cdf(np.abs(t_stat), df=max(T - k, 1)))

    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "beta": beta,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "r_squared": r_squared,
        "nw_lags": lags,
        "n_obs": T,
        "residuals": resid,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Alpha vs Beta decomposition with factor regressions")
    ap.add_argument("--trades-csv", required=True,
                    help="Trade-level diagnostics CSV from online_sgd_backtest.py")
    ap.add_argument("--close-csv", required=True,
                    help="Close price matrix CSV")
    ap.add_argument("--ticker", required=True,
                    help="Ticker symbol")
    ap.add_argument("--factor-csv", default=None,
                    help="Optional: FF5+MOM daily factor returns CSV")
    ap.add_argument("--open-csv", default=None,
                    help="Optional: Open price matrix CSV (for CLOP-style returns)")
    args = ap.parse_args()

    print(f"\n{'='*80}")
    print(f"  ALPHA VS BETA DECOMPOSITION")
    print(f"  Ticker: {args.ticker}")
    print(f"  Trades: {args.trades_csv}")
    print(f"{'='*80}")

    # ── Load trade-level PnL ──
    trades = pd.read_csv(args.trades_csv)
    if trades.empty:
        print("ERROR: No trades found.")
        sys.exit(1)

    # Aggregate to daily PnL
    trades["day_date"] = pd.to_datetime(trades["day"]).dt.strftime("%Y%m%d").astype(int)
    daily_pnl = trades.groupby("day_date")["net_raw"].sum().rename("strategy_pnl")

    print(f"  Trades: {len(trades):,}")
    print(f"  Trading days with PnL: {len(daily_pnl)}")

    # ── Load market data ──
    close_px = pd.read_csv(args.close_csv, index_col="date")
    close_px.index = pd.Index(close_px.index).astype(int)

    # Stock daily return
    if args.ticker not in close_px.columns:
        print(f"ERROR: {args.ticker} not found in close price matrix.")
        sys.exit(1)

    stock_ret = close_px[args.ticker].pct_change().rename("stock_return")

    # SPY daily return (market)
    if "SPY" in close_px.columns:
        mkt_ret = close_px["SPY"].pct_change().rename("market_return")
    else:
        print("WARNING: SPY not found in close matrix. Using stock return only.")
        mkt_ret = pd.Series(dtype=float, name="market_return")

    # ── Merge all series ──
    panel = pd.DataFrame(daily_pnl)
    panel = panel.join(stock_ret, how="left")
    if not mkt_ret.empty:
        panel = panel.join(mkt_ret, how="left")
    panel = panel.dropna()

    if len(panel) < 30:
        print(f"ERROR: Only {len(panel)} observations. Need at least 30.")
        sys.exit(1)

    print(f"  Merged observations: {len(panel)}")
    print(f"  Date range: {panel.index.min()} to {panel.index.max()}")

    # ── Regression 1: Strategy PnL ~ alpha + beta_stock * R_stock ──
    y = panel["strategy_pnl"].values
    X_names = ["intercept", "stock_return"]
    X = np.column_stack([np.ones(len(y)), panel["stock_return"].values])

    results_stock = newey_west_se(y, X)

    print(f"\n  {'='*70}")
    print(f"  REGRESSION 1: PnL = α + β × R_stock")
    print(f"  {'='*70}")
    print(f"  {'Variable':<20} {'Coef':>12} {'NW SE':>12} {'t-stat':>10} {'p-value':>10}")
    print(f"  {'-'*70}")
    for i, name in enumerate(X_names):
        sig = ""
        if results_stock["p_value"][i] < 0.01:
            sig = " ***"
        elif results_stock["p_value"][i] < 0.05:
            sig = " **"
        elif results_stock["p_value"][i] < 0.10:
            sig = " *"
        print(f"  {name:<20} {results_stock['beta'][i]:>12.6f} "
              f"{results_stock['se'][i]:>12.6f} "
              f"{results_stock['t_stat'][i]:>10.2f} "
              f"{results_stock['p_value'][i]:>10.4f}{sig}")
    print(f"  R²: {results_stock['r_squared']:.4f}  |  NW lags: {results_stock['nw_lags']}  |  N: {results_stock['n_obs']}")

    # ── Regression 2: Strategy PnL ~ alpha + beta_stock + beta_market ──
    if "market_return" in panel.columns:
        X_names2 = ["intercept", "stock_return", "market_return"]
        X2 = np.column_stack([
            np.ones(len(y)),
            panel["stock_return"].values,
            panel["market_return"].values,
        ])
        results_mkt = newey_west_se(y, X2)

        print(f"\n  {'='*70}")
        print(f"  REGRESSION 2: PnL = α + β₁ × R_stock + β₂ × R_market")
        print(f"  {'='*70}")
        print(f"  {'Variable':<20} {'Coef':>12} {'NW SE':>12} {'t-stat':>10} {'p-value':>10}")
        print(f"  {'-'*70}")
        for i, name in enumerate(X_names2):
            sig = ""
            if results_mkt["p_value"][i] < 0.01:
                sig = " ***"
            elif results_mkt["p_value"][i] < 0.05:
                sig = " **"
            elif results_mkt["p_value"][i] < 0.10:
                sig = " *"
            print(f"  {name:<20} {results_mkt['beta'][i]:>12.6f} "
                  f"{results_mkt['se'][i]:>12.6f} "
                  f"{results_mkt['t_stat'][i]:>10.2f} "
                  f"{results_mkt['p_value'][i]:>10.4f}{sig}")
        print(f"  R²: {results_mkt['r_squared']:.4f}  |  NW lags: {results_mkt['nw_lags']}  |  N: {results_mkt['n_obs']}")

    # ── Regression 3: FF5 + MOM (if available) ──
    if args.factor_csv and os.path.exists(args.factor_csv):
        print(f"\n  Loading factor data from {args.factor_csv}...")
        factors = pd.read_csv(args.factor_csv)

        # Standard FF format: Date, Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
        if "Date" in factors.columns:
            factors["Date"] = factors["Date"].astype(int)
            factors = factors.set_index("Date")
        elif "date" in factors.columns:
            factors["date"] = factors["date"].astype(int)
            factors = factors.set_index("date")

        # Expected factor columns
        factor_cols = []
        for col_name in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "UMD"]:
            if col_name in factors.columns:
                factor_cols.append(col_name)

        if factor_cols:
            # Merge factors with panel
            panel_ff = panel.join(factors[factor_cols], how="inner")
            panel_ff = panel_ff.dropna()

            if len(panel_ff) >= 30:
                y_ff = panel_ff["strategy_pnl"].values
                X_ff_names = ["intercept"] + factor_cols
                X_ff = np.column_stack([
                    np.ones(len(y_ff)),
                    panel_ff[factor_cols].values
                ])
                results_ff = newey_west_se(y_ff, X_ff)

                print(f"\n  {'='*70}")
                print(f"  REGRESSION 3: PnL = α + Σ βᵢ × Factor_i  (FF5+MOM)")
                print(f"  {'='*70}")
                print(f"  {'Variable':<20} {'Coef':>12} {'NW SE':>12} {'t-stat':>10} {'p-value':>10}")
                print(f"  {'-'*70}")
                for i, name in enumerate(X_ff_names):
                    sig = ""
                    if results_ff["p_value"][i] < 0.01:
                        sig = " ***"
                    elif results_ff["p_value"][i] < 0.05:
                        sig = " **"
                    elif results_ff["p_value"][i] < 0.10:
                        sig = " *"
                    print(f"  {name:<20} {results_ff['beta'][i]:>12.6f} "
                          f"{results_ff['se'][i]:>12.6f} "
                          f"{results_ff['t_stat'][i]:>10.2f} "
                          f"{results_ff['p_value'][i]:>10.4f}{sig}")
                print(f"  R²: {results_ff['r_squared']:.4f}  |  NW lags: {results_ff['nw_lags']}  |  N: {results_ff['n_obs']}")

                # Information ratio = alpha / residual_std
                alpha = results_ff["beta"][0]
                resid_std = np.std(results_ff["residuals"], ddof=1)
                ir = (alpha / resid_std * np.sqrt(252)) if resid_std > 0 else 0.0
                print(f"\n  Annualized Alpha:       {alpha * 252:.4f}")
                print(f"  Residual Std (daily):   {resid_std:.6f}")
                print(f"  Information Ratio:      {ir:.2f}")
            else:
                print(f"  WARNING: Only {len(panel_ff)} observations after factor merge. Skipping FF regression.")
        else:
            print(f"  WARNING: No recognized factor columns in {args.factor_csv}")
    else:
        print(f"\n  No factor CSV provided. Skipping FF5+MOM regression.")
        print(f"  Use --factor-csv to enable multi-factor alpha decomposition.")

    # ── Summary ──
    alpha_daily = results_stock["beta"][0]
    alpha_ann = alpha_daily * 252
    alpha_t = results_stock["t_stat"][0]

    print(f"\n  {'='*70}")
    print(f"  SUMMARY: {args.ticker}")
    print(f"  {'='*70}")
    print(f"  Daily alpha (vs stock):      {alpha_daily:>+.6f}")
    print(f"  Annualized alpha (vs stock): {alpha_ann:>+.4f}")
    print(f"  Alpha t-stat (NW):           {alpha_t:>.2f}")
    print(f"  Stock beta:                  {results_stock['beta'][1]:>.4f}")

    if alpha_t > 1.96:
        print(f"  → Alpha is STATISTICALLY SIGNIFICANT at 5% level")
    elif alpha_t > 1.64:
        print(f"  → Alpha is marginally significant at 10% level")
    else:
        print(f"  → Alpha is NOT statistically significant")

    if abs(results_stock["beta"][1]) < 0.1:
        print(f"  → Strategy is UNCORRELATED with underlying (|β| < 0.1)")
    else:
        print(f"  → Strategy has residual directional exposure (β = {results_stock['beta'][1]:.4f})")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
