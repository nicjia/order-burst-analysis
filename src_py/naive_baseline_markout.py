#!/usr/bin/env python3
"""
naive_baseline_markout.py — Microstructure Baseline: Unconditional Burst PnL

Computes the gross PnL-per-trade (PPT) in basis points for ALL raw bursts
BEFORE any Optuna filtering is applied. This establishes the "naive momentum"
baseline that reviewers demanded.

Expected Result:
  The unconditional baseline should show NEGATIVE expected value (EV)
  at short horizons due to bid-ask bounce, proving that the complex
  filtering pipeline is necessary to extract genuine alpha.

Horizons computed:
  - 1m, 3m, 5m, 10m (intraday, from Mid_Xm columns)
  - tCLOSE (intraday close)
  - CLOP   (overnight: next-day open)
  - CLCL   (overnight: next-day close)

Usage:
    python3 src_py/naive_baseline_markout.py results/bursts_NVDA_baseline_unfiltered.csv
    python3 src_py/naive_baseline_markout.py results/bursts_NVDA_baseline_unfiltered.csv --filtered results/bursts_NVDA_baseline_filtered.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_markout_bps(df, exit_col, entry_col="StartPrice", direction_col="Direction"):
    """
    Compute gross PnL in basis points per trade.

    markout_bps = Direction × (exit_price - entry_price) / entry_price × 10000
    """
    if exit_col not in df.columns:
        return None

    entry = df[entry_col].astype(float)
    exit_px = df[exit_col].astype(float)
    direction = df[direction_col].astype(float)

    valid = (entry > 0) & exit_px.notna() & direction.notna() & (direction != 0)
    if valid.sum() < 10:
        return None

    bps = direction[valid] * (exit_px[valid] - entry[valid]) / entry[valid] * 10000.0
    return bps


def report_markout(bps, label):
    """Print statistical summary of a markout series."""
    if bps is None or len(bps) < 10:
        print(f"  {label:<20}  N/A (insufficient data)")
        return None

    n = len(bps)
    mean = np.mean(bps)
    std = np.std(bps, ddof=1)
    se = std / np.sqrt(n)
    t_stat = mean / se if se > 0 else 0.0
    p_val = 2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), df=n - 1))
    median = np.median(bps)
    pct_positive = 100.0 * (bps > 0).sum() / n
    sharpe_ann = (mean / std * np.sqrt(252)) if std > 0 else 0.0

    sig = ""
    if p_val < 0.01:
        sig = "***"
    elif p_val < 0.05:
        sig = "**"
    elif p_val < 0.10:
        sig = "*"

    print(f"  {label:<20}  N={n:>7,}  Mean={mean:>8.2f} bps  Median={median:>8.2f}  "
          f"Std={std:>8.2f}  t={t_stat:>6.2f}{sig:<4}  "
          f"Win%={pct_positive:>5.1f}%  Sharpe(ann)={sharpe_ann:>6.2f}")

    return {
        "horizon": label,
        "n": n,
        "mean_bps": mean,
        "median_bps": median,
        "std_bps": std,
        "t_stat": t_stat,
        "p_value": p_val,
        "win_pct": pct_positive,
        "sharpe_ann": sharpe_ann,
    }


def analyze_dataset(df, label):
    """Run full markout analysis on a dataset."""
    print(f"\n{'='*100}")
    print(f"  MARKOUT ANALYSIS: {label}")
    print(f"  Bursts: {len(df):,}  |  Days: {df['Date'].nunique()}  "
          f"|  Tickers: {df['Ticker'].nunique() if 'Ticker' in df.columns else 'N/A'}")
    print(f"{'='*100}")
    print(f"  {'Horizon':<20}  {'N':>7}  {'Mean':>11}  {'Median':>11}  "
          f"{'Std':>8}  {'t-stat':>10}  {'Win%':>7}  {'Sharpe':>13}")
    print(f"  {'-'*100}")

    results = []

    # Intraday horizons from C++ pipeline mid-price snapshots
    horizon_map = {
        "1m":      "Mid_1m",
        "3m":      "Mid_3m",
        "5m":      "Mid_5m",
        "10m":     "Mid_10m",
        "tCLOSE":  "CloseMid",
    }

    for label_h, col in horizon_map.items():
        bps = compute_markout_bps(df, exit_col=col)
        result = report_markout(bps, label_h)
        if result:
            results.append(result)

    # Overnight horizons (from compute_permanence.py if available)
    # These use CRSP prices, not raw mid snapshots.
    # For the CLOP/CLCL we need the permanence columns or raw CRSP lookups.
    # If Perm_CLOP/CLCL exist, reverse-engineer the BPS from them;
    # otherwise, try the CRSP columns if they exist.
    for label_h, perm_col in [("CLOP", "Perm_CLOP"), ("CLCL", "Perm_CLCL")]:
        if perm_col in df.columns:
            # Permanence = arcsinh(Volume * Direction * (exit - entry))
            # We want BPS = Direction * (exit - entry) / entry * 10000
            # Since we don't have the raw exit price here, use the permanence sign
            # as a proxy for direction correctness, and compute from available prices.
            valid = df[perm_col].notna()
            if valid.sum() >= 10:
                # Use sinh to invert arcsinh, then normalize to BPS
                perm_vals = df.loc[valid, perm_col].astype(float)
                volumes = df.loc[valid, "Volume"].astype(float).clip(lower=1)
                entry_px = df.loc[valid, "StartPrice"].astype(float).clip(lower=0.01)
                # arcsinh(V*D*(exit-entry)) → raw = V*D*(exit-entry)
                raw_perm = np.sinh(perm_vals)
                # bps = raw_perm / (V * entry) * 10000
                bps = raw_perm / (volumes * entry_px) * 10000.0
                # Clip extreme outliers for reporting stability
                bps = bps.clip(-500, 500)
                result = report_markout(bps.values, label_h)
                if result:
                    results.append(result)
        else:
            report_markout(None, label_h)

    print(f"{'='*100}")
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Naive momentum baseline: PnL per trade in BPS for unfiltered bursts")
    ap.add_argument("unfiltered_csv",
                    help="Path to unfiltered bursts CSV (all raw bursts)")
    ap.add_argument("--filtered", default=None,
                    help="Optional: path to filtered bursts CSV for comparison")
    ap.add_argument("--ticker", default=None,
                    help="Filter to a specific ticker")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    args = ap.parse_args()

    def load_and_filter(path):
        df = pd.read_csv(path)
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

        # Ensure Direction column exists and handle mixed bursts
        if "Direction" in df.columns:
            # For baseline: include ALL bursts (even mixed direction=0)
            # Assign mixed bursts the sign of their dominant side
            mixed_mask = df["Direction"] == 0
            if mixed_mask.any():
                buy_dominant = df.loc[mixed_mask, "BuyVolume"] >= df.loc[mixed_mask, "SellVolume"]
                df.loc[mixed_mask & buy_dominant, "Direction"] = 1
                df.loc[mixed_mask & ~buy_dominant, "Direction"] = -1
        return df

    # ── Load and analyze unfiltered ──
    print(f"Loading unfiltered bursts from {args.unfiltered_csv}...")
    df_raw = load_and_filter(args.unfiltered_csv)
    print(f"  Loaded {len(df_raw):,} bursts")

    if len(df_raw) < 10:
        print("ERROR: Too few bursts to analyze.")
        sys.exit(1)

    raw_results = analyze_dataset(df_raw, "ALL RAW BURSTS (Naive Baseline — No Filtering)")

    # ── Optionally analyze filtered ──
    if args.filtered:
        print(f"\nLoading filtered bursts from {args.filtered}...")
        df_filt = load_and_filter(args.filtered)
        print(f"  Loaded {len(df_filt):,} bursts")

        if len(df_filt) >= 10:
            filt_results = analyze_dataset(df_filt, "FILTERED BURSTS (Post-Optuna Pipeline)")

            # ── Comparison ──
            print(f"\n{'='*100}")
            print(f"  FILTERING UPLIFT: Raw Baseline vs Filtered Pipeline")
            print(f"{'='*100}")
            raw_dict = {r["horizon"]: r for r in raw_results}
            filt_dict = {r["horizon"]: r for r in filt_results}
            for h in raw_dict:
                if h in filt_dict:
                    raw_m = raw_dict[h]["mean_bps"]
                    filt_m = filt_dict[h]["mean_bps"]
                    uplift = filt_m - raw_m
                    print(f"  {h:<20}  Raw={raw_m:>8.2f} bps  →  Filtered={filt_m:>8.2f} bps  "
                          f"  Uplift={uplift:>+8.2f} bps")
            print(f"{'='*100}")


if __name__ == "__main__":
    main()
