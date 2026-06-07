#!/usr/bin/env python3
"""
time_of_day_analysis.py — Reviewer B9: Time-of-Day Stratification

Groups filtered bursts into time-of-day regimes and reports Gross PnL (BPS),
t-statistics, and trade counts separately for each bucket.

Regimes:
  - Early Morning:  09:30 - 10:30  (34200 - 37800 SPM)
  - Midday:         10:30 - 14:30  (37800 - 52200 SPM)
  - Late Afternoon:  14:30 - 16:00  (52200 - 57600 SPM)

Usage:
    python3 src_py/time_of_day_analysis.py results/bursts_NVDA_baseline_filtered.csv
    python3 src_py/time_of_day_analysis.py results/bursts_NVDA_baseline_unfiltered.csv --ticker NVDA
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# Time-of-day regime boundaries (seconds past midnight)
REGIMES = {
    "Early (09:30-10:30)": (34200.0, 37800.0),
    "Midday (10:30-14:30)": (37800.0, 52200.0),
    "Close  (14:30-16:00)": (52200.0, 57600.0),
}

# Horizons to evaluate
HORIZON_MAP = {
    "1m":      "Mid_1m",
    "3m":      "Mid_3m",
    "5m":      "Mid_5m",
    "10m":     "Mid_10m",
    "tCLOSE":  "CloseMid",
}


def compute_bps(df, exit_col, entry_col="StartPrice", dir_col="Direction"):
    """Compute directional PnL in basis points."""
    if exit_col not in df.columns:
        return None
    entry = df[entry_col].astype(float)
    exit_px = df[exit_col].astype(float)
    direction = df[dir_col].astype(float)
    valid = (entry > 0) & exit_px.notna() & direction.notna() & (direction != 0)
    if valid.sum() < 5:
        return None
    bps = direction[valid] * (exit_px[valid] - entry[valid]) / entry[valid] * 10000.0
    return bps


def format_stat_line(regime, horizon, bps):
    """Format a single statistics line."""
    if bps is None or len(bps) < 5:
        return f"  {regime:<24} {horizon:<10}    N/A"

    n = len(bps)
    mean = np.mean(bps)
    std = np.std(bps, ddof=1)
    se = std / np.sqrt(n) if n > 1 else 0.0
    t_stat = mean / se if se > 0 else 0.0
    p_val = 2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), df=max(n - 1, 1)))
    median = np.median(bps)
    win_pct = 100.0 * (bps > 0).sum() / n

    sig = ""
    if p_val < 0.01:
        sig = "***"
    elif p_val < 0.05:
        sig = "**"
    elif p_val < 0.10:
        sig = "*"

    return (f"  {regime:<24} {horizon:<10}  N={n:>6,}  "
            f"Mean={mean:>8.2f} bps  Median={median:>8.2f}  "
            f"Std={std:>8.2f}  t={t_stat:>6.2f}{sig:<4}  Win%={win_pct:>5.1f}%")


def main():
    ap = argparse.ArgumentParser(
        description="Time-of-day stratification analysis (Reviewer B9)")
    ap.add_argument("data_csv", help="Path to bursts CSV")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    args = ap.parse_args()

    # ── Load data ──
    df = pd.read_csv(args.data_csv)
    try:
        df["Date"] = df["Date"].astype(int)
    except (ValueError, TypeError):
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").astype(int)

    if args.ticker:
        df = df[df["Ticker"] == args.ticker].copy()
    if args.start_date:
        start_int = int(pd.to_datetime(args.start_date).strftime("%Y%m%d"))
        df = df[df["Date"] >= start_int].copy()
    if args.end_date:
        end_int = int(pd.to_datetime(args.end_date).strftime("%Y%m%d"))
        df = df[df["Date"] <= end_int].copy()

    if "StartTime" not in df.columns:
        print("ERROR: CSV must contain 'StartTime' column.")
        sys.exit(1)

    # Handle mixed direction
    if "Direction" in df.columns:
        mixed = df["Direction"] == 0
        if mixed.any():
            buy_dom = df.loc[mixed, "BuyVolume"] >= df.loc[mixed, "SellVolume"]
            df.loc[mixed & buy_dom, "Direction"] = 1
            df.loc[mixed & ~buy_dom, "Direction"] = -1

    print(f"\n{'='*110}")
    print(f"  TIME-OF-DAY STRATIFICATION ANALYSIS")
    print(f"  Data: {args.data_csv}")
    print(f"  Total bursts: {len(df):,}  |  Days: {df['Date'].nunique()}")
    print(f"{'='*110}")

    # ── Assign regime labels ──
    start_times = df["StartTime"].astype(float)
    df["Regime"] = "Unknown"
    for regime_name, (t_start, t_end) in REGIMES.items():
        mask = (start_times >= t_start) & (start_times < t_end)
        df.loc[mask, "Regime"] = regime_name

    # ── Report burst distribution ──
    print(f"\n  Burst Distribution by Regime:")
    print(f"  {'Regime':<24}  {'Count':>8}  {'Pct':>7}  {'Avg Volume':>12}  {'Avg Duration':>13}")
    print(f"  {'-'*75}")
    for regime_name in REGIMES:
        subset = df[df["Regime"] == regime_name]
        n = len(subset)
        pct = 100.0 * n / max(len(df), 1)
        avg_vol = subset["Volume"].mean() if "Volume" in subset.columns and n > 0 else 0
        avg_dur = (subset["EndTime"] - subset["StartTime"]).mean() if "EndTime" in subset.columns and n > 0 else 0
        print(f"  {regime_name:<24}  {n:>8,}  {pct:>6.1f}%  {avg_vol:>12,.0f}  {avg_dur:>10.2f} sec")

    # ── Report markout by regime and horizon ──
    for horizon_label, horizon_col in HORIZON_MAP.items():
        if horizon_col not in df.columns:
            continue

        print(f"\n  Horizon: {horizon_label}")
        print(f"  {'Regime':<24} {'Horizon':<10}  {'N':>6}  {'Mean':>11}  {'Median':>11}  "
              f"{'Std':>8}  {'t-stat':>10}  {'Win%':>7}")
        print(f"  {'-'*100}")

        # All regimes
        bps_all = compute_bps(df, exit_col=horizon_col)
        print(format_stat_line("ALL", horizon_label, bps_all))

        # Per regime
        for regime_name in REGIMES:
            subset = df[df["Regime"] == regime_name]
            bps = compute_bps(subset, exit_col=horizon_col)
            print(format_stat_line(regime_name, horizon_label, bps))

    # ── Cross-regime ANOVA test for heterogeneity ──
    print(f"\n  Cross-Regime Heterogeneity (Kruskal-Wallis H-test):")
    print(f"  {'Horizon':<10}  {'H-statistic':>12}  {'P-value':>12}  {'Significant?':>14}")
    print(f"  {'-'*55}")

    for horizon_label, horizon_col in HORIZON_MAP.items():
        if horizon_col not in df.columns:
            continue

        groups = []
        for regime_name in REGIMES:
            subset = df[df["Regime"] == regime_name]
            bps = compute_bps(subset, exit_col=horizon_col)
            if bps is not None and len(bps) >= 5:
                groups.append(bps.values)

        if len(groups) >= 2:
            h_stat, h_pval = scipy_stats.kruskal(*groups)
            sig = "YES ***" if h_pval < 0.01 else ("YES **" if h_pval < 0.05 else "no")
            print(f"  {horizon_label:<10}  {h_stat:>12.4f}  {h_pval:>12.2e}  {sig:>14}")
        else:
            print(f"  {horizon_label:<10}  insufficient data for test")

    print(f"\n{'='*110}")


if __name__ == "__main__":
    main()
