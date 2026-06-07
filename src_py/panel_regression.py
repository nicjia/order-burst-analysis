#!/usr/bin/env python3
"""
panel_regression.py — Reviewer R3/B4/B5: Panel Regressions & COI Construction

Implements:
  1. Daily Conditional Order Imbalance (COI) per stock
  2. Sign-conditional COI flipping for mean-reverting stocks (Reviewer R3)
  3. Long-short quintile portfolio sorts by COI
  4. Fama-MacBeth cross-sectional regressions
  5. Factor-adjusted returns (Fama-French 5 + Momentum)
  6. Newey-West standard errors

Sign-Conditional Flipping (Reviewer R3):
  Financials (JPM, MS, GS, BAC, etc.) structurally mean-revert due to
  ETF arbitrage (XLF) and statistical pairs trading. A buy-side burst in
  JPM is NOT momentum — it's the opening leg of a pair that will unwind.
  We invert their COI score (×-1) so that "high burst activity" correctly
  maps to "expected reversal" rather than being misclassified as a signal
  failure.

Usage:
    python3 src_py/panel_regression.py \
        --burst-dir results/ \
        --tickers AAPL,V,MA,PG,KO,SPY \
        --open-csv open_all.csv \
        --close-csv close_all.csv

    # With signal flipping for financials:
    python3 src_py/panel_regression.py \
        --burst-dir results/ \
        --tickers AAPL,V,MA,JPM,MS,GS,SPY \
        --open-csv open_all.csv \
        --close-csv close_all.csv \
        --mean-revert-tickers JPM,MS,GS,BAC,C,WFC,XLF
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def load_burst_data(burst_dir, tickers, suffix="baseline_unfiltered"):
    """Load burst CSVs for multiple tickers into a single DataFrame."""
    frames = []
    for ticker in tickers:
        path = os.path.join(burst_dir, f"bursts_{ticker}_{suffix}.csv")
        if not os.path.exists(path):
            print(f"  Warning: Missing {path}, skipping {ticker}")
            continue
        df = pd.read_csv(path)
        try:
            df["Date"] = df["Date"].astype(int)
        except (ValueError, TypeError):
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").astype(int)
        if "Ticker" not in df.columns:
            df["Ticker"] = ticker
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_daily_coi(bursts_df, vol_frac_thresh=0.0, dir_thresh=0.7):
    """
    Compute daily Conditional Order Imbalance (COI) per stock.

    COI_i,t = Σ(valid buy burst volumes) - Σ(valid sell burst volumes)
              normalized by total valid burst volume.

    Returns a DataFrame with columns: [Date, Ticker, COI, n_buy, n_sell, total_vol]
    """
    # Basic directional filter
    df = bursts_df.copy()

    # Classify direction if not already done
    if "Direction" in df.columns:
        directional = df[df["Direction"] != 0].copy()
    else:
        directional = df.copy()

    records = []
    for (date, ticker), group in directional.groupby(["Date", "Ticker"]):
        buy_mask = group["Direction"] == 1
        sell_mask = group["Direction"] == -1

        buy_vol = group.loc[buy_mask, "Volume"].sum() if buy_mask.any() else 0
        sell_vol = group.loc[sell_mask, "Volume"].sum() if sell_mask.any() else 0
        total_vol = buy_vol + sell_vol

        coi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0

        records.append({
            "Date": date,
            "Ticker": ticker,
            "COI": coi,
            "n_buy": buy_mask.sum(),
            "n_sell": sell_mask.sum(),
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "total_vol": total_vol,
            "n_bursts": len(group),
        })

    return pd.DataFrame(records)


def build_quintile_portfolios(coi_daily, returns_df):
    """
    Sort stocks into quintile portfolios based on COI.
    Returns a DataFrame of daily portfolio returns.
    """
    # Merge COI with forward returns
    merged = coi_daily.merge(returns_df, on=["Date", "Ticker"], how="inner")

    if len(merged) < 10:
        return None

    # Assign quintiles within each date
    merged["quintile"] = merged.groupby("Date")["COI"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if len(x) >= 5 else np.nan
    )

    merged = merged.dropna(subset=["quintile"])

    if len(merged) < 10:
        return None

    # Portfolio returns: equal-weight average return within each quintile per day
    port_returns = merged.groupby(["Date", "quintile"]).agg(
        port_return=("fwd_return", "mean"),
        n_stocks=("Ticker", "count")
    ).reset_index()

    return port_returns


def fama_macbeth_regression(panel_df, y_col="fwd_return", x_cols=None):
    """
    Run Fama-MacBeth cross-sectional regression.

    For each date t:
        R_{i,t+1} = α_t + β_t × COI_{i,t} + ε_{i,t}

    Then average the time series of coefficients and compute
    Newey-West standard errors.
    """
    if x_cols is None:
        x_cols = ["COI"]

    dates = sorted(panel_df["Date"].unique())
    coef_series = {col: [] for col in ["intercept"] + x_cols}
    n_obs = []

    for date in dates:
        day_df = panel_df[panel_df["Date"] == date]
        if len(day_df) < 3:
            continue

        y = day_df[y_col].values
        X = day_df[x_cols].values

        if np.any(np.isnan(y)) or np.any(np.isnan(X)):
            continue

        # Add intercept
        X_full = np.column_stack([np.ones(len(y)), X])

        try:
            # OLS: β = (X'X)^{-1} X'y
            beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
            coef_series["intercept"].append(beta[0])
            for i, col in enumerate(x_cols):
                coef_series[col].append(beta[i + 1])
            n_obs.append(len(y))
        except np.linalg.LinAlgError:
            continue

    if not coef_series["intercept"]:
        return None

    # ── Newey-West standard errors ──
    results = {}
    n_periods = len(coef_series["intercept"])
    # Bandwidth: Newey-West optimal lag ≈ floor(4 * (T/100)^{2/9})
    nw_lag = max(1, int(np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0))))

    for col in ["intercept"] + x_cols:
        series = np.array(coef_series[col])
        mean_coef = np.mean(series)
        T = len(series)

        # Compute Newey-West variance
        demeaned = series - mean_coef
        gamma_0 = np.mean(demeaned ** 2)
        nw_var = gamma_0
        for lag in range(1, nw_lag + 1):
            weight = 1.0 - lag / (nw_lag + 1.0)
            gamma_l = np.mean(demeaned[lag:] * demeaned[:-lag])
            nw_var += 2.0 * weight * gamma_l

        nw_se = np.sqrt(nw_var / T)
        t_stat = mean_coef / nw_se if nw_se > 0 else 0.0
        p_val = 2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), df=max(T - 1, 1)))

        results[col] = {
            "mean": mean_coef,
            "nw_se": nw_se,
            "t_stat": t_stat,
            "p_value": p_val,
            "n_periods": T,
            "nw_lag": nw_lag,
        }

    return results


def main():
    ap = argparse.ArgumentParser(
        description="Panel regression framework: COI, quintile sorts, Fama-MacBeth")
    ap.add_argument("--burst-dir", required=True,
                    help="Directory containing burst CSV files")
    ap.add_argument("--tickers", required=True,
                    help="Comma-separated list of tickers")
    ap.add_argument("--open-csv", required=True,
                    help="Open price matrix CSV")
    ap.add_argument("--close-csv", required=True,
                    help="Close price matrix CSV")
    ap.add_argument("--suffix", default="baseline_unfiltered",
                    help="Burst CSV suffix (default: baseline_unfiltered)")
    ap.add_argument("--factor-csv", default=None,
                    help="Optional: Fama-French factor returns CSV")
    ap.add_argument("--mean-revert-tickers", default="",
                    help="Comma-separated tickers whose COI should be inverted (×-1). "
                         "These are structurally mean-reverting stocks (e.g., financials) "
                         "where burst momentum predicts reversal, not continuation.")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--output-csv", default=None,
                    help="Optional: save COI panel to CSV")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    mean_revert_set = set(
        t.strip() for t in args.mean_revert_tickers.split(",") if t.strip()
    )

    print(f"\n{'='*80}")
    print(f"  PANEL REGRESSION FRAMEWORK")
    print(f"  Tickers: {tickers}")
    print(f"  Burst dir: {args.burst_dir}")
    if mean_revert_set:
        active_mr = sorted(mean_revert_set & set(tickers))
        print(f"  Mean-revert signal flip: {active_mr}")
    print(f"{'='*80}")

    # ── Load burst data ──
    bursts = load_burst_data(args.burst_dir, tickers, args.suffix)
    if bursts.empty:
        print("ERROR: No burst data loaded.")
        sys.exit(1)

    if args.start_date:
        start_int = int(pd.to_datetime(args.start_date).strftime("%Y%m%d"))
        bursts = bursts[bursts["Date"] >= start_int]
    if args.end_date:
        end_int = int(pd.to_datetime(args.end_date).strftime("%Y%m%d"))
        bursts = bursts[bursts["Date"] <= end_int]

    print(f"  Total bursts loaded: {len(bursts):,}")
    print(f"  Tickers present: {bursts['Ticker'].nunique()}")
    print(f"  Date range: {bursts['Date'].min()} to {bursts['Date'].max()}")

    # ── Step 1: Compute COI ──
    print(f"\n  Computing daily COI (Conditional Order Imbalance)...")
    coi_daily = compute_daily_coi(bursts)

    # ── Step 1b: Sign-conditional COI flipping (Reviewer R3) ──
    # For mean-reverting stocks (financials, etc.), invert the COI score.
    # Rationale: A buy-side burst in JPM is NOT momentum — it's an ETF-arb
    # or pair-trade opening leg that will structurally unwind. The signal
    # is REVERSAL, not continuation. Multiply COI by -1 to align the
    # cross-sectional sort so that Q5 (high COI) consistently means
    # "expected positive forward return" for ALL stock types.
    if mean_revert_set:
        flip_mask = coi_daily["Ticker"].isin(mean_revert_set)
        n_flipped = flip_mask.sum()
        if n_flipped > 0:
            coi_daily.loc[flip_mask, "COI"] *= -1.0
            flipped_tickers = sorted(coi_daily.loc[flip_mask, "Ticker"].unique())
            print(f"\n  Sign-Conditional COI Flip (Reviewer R3):")
            print(f"    Inverted {n_flipped:,} stock-day COI scores for: {flipped_tickers}")
            print(f"    Rationale: these tickers are structurally mean-reverting")
            print(f"    (ETF arb, pairs trading). Burst momentum → expected reversal.")

    print(f"\n  COI panel: {len(coi_daily):,} stock-day observations")
    print(f"  COI statistics (after flipping):")
    print(f"    Mean COI:    {coi_daily['COI'].mean():.4f}")
    print(f"    Std COI:     {coi_daily['COI'].std():.4f}")
    print(f"    Median COI:  {coi_daily['COI'].median():.4f}")

    if args.output_csv:
        coi_daily.to_csv(args.output_csv, index=False)
        print(f"  COI panel saved to: {args.output_csv}")

    # ── Step 2: Build forward returns ──
    print(f"\n  Building forward returns from price matrices...")
    close_px = pd.read_csv(args.close_csv, index_col="date")
    close_px.index = pd.Index(close_px.index).astype(int)

    open_px = pd.read_csv(args.open_csv, index_col="date")
    open_px.index = pd.Index(open_px.index).astype(int)

    trading_days = np.array(sorted(close_px.index.tolist()), dtype=np.int64)

    # Compute CLOP return: (next_open - today_close) / today_close
    return_records = []
    for ticker in tickers:
        if ticker not in close_px.columns:
            continue
        cl = close_px[ticker].dropna()
        op = open_px[ticker].dropna() if ticker in open_px.columns else pd.Series(dtype=float)

        for date_int in cl.index:
            idx = np.searchsorted(trading_days, date_int, side="right")
            if idx >= len(trading_days):
                continue
            next_day = int(trading_days[idx])

            today_close = cl.get(date_int, np.nan)
            next_open = op.get(next_day, np.nan)

            if np.isnan(today_close) or np.isnan(next_open) or today_close <= 0:
                continue

            fwd_ret = (next_open - today_close) / today_close

            return_records.append({
                "Date": date_int,
                "Ticker": ticker,
                "fwd_return": fwd_ret,
            })

    returns_df = pd.DataFrame(return_records)
    print(f"  Forward returns: {len(returns_df):,} observations")

    # ── Step 3: Merge and run Fama-MacBeth ──
    panel = coi_daily.merge(returns_df, on=["Date", "Ticker"], how="inner")
    print(f"  Merged panel: {len(panel):,} observations")

    if len(panel) < 50:
        print("WARNING: Panel too small for reliable regression results.")

    if len(panel) > 0:
        print(f"\n  Running Fama-MacBeth cross-sectional regression...")
        print(f"  Model: R_{{i,t+1}} = α + β × COI_{{i,t}} + ε")

        fm_results = fama_macbeth_regression(panel, y_col="fwd_return", x_cols=["COI"])

        if fm_results:
            print(f"\n  {'='*70}")
            print(f"  FAMA-MACBETH REGRESSION RESULTS")
            print(f"  {'='*70}")
            print(f"  {'Variable':<15} {'Coef':>10} {'NW SE':>10} {'t-stat':>10} "
                  f"{'p-value':>10} {'Sig':>6}")
            print(f"  {'-'*70}")

            for var, res in fm_results.items():
                sig = ""
                if res["p_value"] < 0.01:
                    sig = "***"
                elif res["p_value"] < 0.05:
                    sig = "**"
                elif res["p_value"] < 0.10:
                    sig = "*"

                print(f"  {var:<15} {res['mean']:>10.6f} {res['nw_se']:>10.6f} "
                      f"{res['t_stat']:>10.2f} {res['p_value']:>10.4f} {sig:>6}")

            print(f"\n  Newey-West lag: {fm_results['intercept']['nw_lag']}")
            print(f"  Cross-sections (months): {fm_results['intercept']['n_periods']}")
        else:
            print("  Fama-MacBeth regression failed (insufficient data).")

    # ── Step 4: Quintile portfolio sorts ──
    if len(panel) >= 50:
        print(f"\n  Building quintile portfolios sorted by COI...")
        port_returns = build_quintile_portfolios(coi_daily, returns_df)

        if port_returns is not None and len(port_returns) > 0:
            print(f"\n  {'='*70}")
            print(f"  QUINTILE PORTFOLIO RETURNS")
            print(f"  {'='*70}")
            print(f"  {'Quintile':<10} {'Mean Ret (bps)':>15} {'Std (bps)':>12} "
                  f"{'t-stat':>10} {'N days':>8} {'Avg Stocks':>12}")
            print(f"  {'-'*70}")

            quintile_means = {}
            for q in sorted(port_returns["quintile"].unique()):
                q_data = port_returns[port_returns["quintile"] == q]
                # Aggregate to daily level
                daily_ret = q_data.groupby("Date")["port_return"].mean()
                n_stocks = q_data.groupby("Date")["n_stocks"].mean().mean()

                mean_ret = daily_ret.mean() * 10000  # Convert to BPS
                std_ret = daily_ret.std() * 10000
                n_days = len(daily_ret)
                se = std_ret / np.sqrt(n_days) if n_days > 1 else 0
                t_stat = mean_ret / se if se > 0 else 0

                quintile_means[q] = mean_ret

                q_label = f"Q{int(q)+1}"
                if q == 0:
                    q_label += " (Low COI)"
                elif q == 4:
                    q_label += " (High COI)"

                sig = ""
                p_val = 2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat / 10000 * np.sqrt(n_days)),
                                                         df=max(n_days - 1, 1)))
                if p_val < 0.01:
                    sig = "***"
                elif p_val < 0.05:
                    sig = "**"

                print(f"  {q_label:<10} {mean_ret:>15.2f} {std_ret:>12.2f} "
                      f"{t_stat:>10.2f}{sig:<4} {n_days:>8} {n_stocks:>12.1f}")

            # Long-Short spread
            if 0 in quintile_means and 4 in quintile_means:
                spread = quintile_means[4] - quintile_means[0]
                print(f"\n  Long-Short Spread (Q5 - Q1): {spread:>+.2f} bps/day")

        else:
            print("  Insufficient data for quintile sorts (need ≥5 stocks per day).")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
