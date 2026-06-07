#!/usr/bin/env python3
"""
transaction_cost_grid.py — Reviewer R5/B8: Transaction Cost Sensitivity Grid

Decouples signal generation from execution costs entirely.
Takes the gross PPT signal and applies fixed transaction cost levels
(0 to 5 bps) to prove the alpha survives liquidity-taking friction.

Usage:
    python3 src_py/transaction_cost_grid.py results/bursts_NVDA_baseline_filtered.csv --ticker NVDA
    python3 src_py/transaction_cost_grid.py results/bursts_NVDA_baseline_unfiltered.csv --ticker NVDA --vol-frac 0.001
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))
from silence_optimized_sweep import compute_trailing_adv, classify_and_filter


# Cost levels to test (round-trip basis points)
COST_LEVELS_BPS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

HORIZON_MAP = {
    "1m":      "Mid_1m",
    "3m":      "Mid_3m",
    "5m":      "Mid_5m",
    "10m":     "Mid_10m",
    "tCLOSE":  "CloseMid",
}


def compute_gross_bps(df, exit_col, entry_col="StartPrice", dir_col="Direction"):
    """Compute directional gross PnL in basis points per trade."""
    if exit_col not in df.columns:
        return None, None, None
    entry = df[entry_col].astype(float)
    exit_px = df[exit_col].astype(float)
    direction = df[dir_col].astype(float)
    valid = (entry > 0) & exit_px.notna() & direction.notna() & (direction != 0)
    if valid.sum() < 10:
        return None, None, None
    gross_bps = direction[valid] * (exit_px[valid] - entry[valid]) / entry[valid] * 10000.0
    return gross_bps.values, entry[valid].values, valid


def main():
    ap = argparse.ArgumentParser(
        description="Transaction cost sensitivity grid (Reviewer R5/B8)")
    ap.add_argument("data_csv", help="Path to bursts CSV")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--vol-frac", type=float, default=0.0)
    ap.add_argument("--dir-thresh", type=float, default=0.7)
    ap.add_argument("--vol-ratio", type=float, default=0.4)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--output-csv", default=None,
                    help="Optional CSV output path for the grid")
    args = ap.parse_args()

    # ── Load data ──
    df = pd.read_csv(args.data_csv)
    try:
        df["Date"] = df["Date"].astype(int)
    except (ValueError, TypeError):
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").astype(int)

    if args.ticker:
        if "Ticker" in df.columns:
            df = df[df["Ticker"] == args.ticker].copy()

    if args.start_date:
        start_int = int(pd.to_datetime(args.start_date).strftime("%Y%m%d"))
        df = df[df["Date"] >= start_int].copy()
    if args.end_date:
        end_int = int(pd.to_datetime(args.end_date).strftime("%Y%m%d"))
        df = df[df["Date"] <= end_int].copy()

    # ── Apply filters if vol_frac > 0 ──
    if args.vol_frac > 0 and args.ticker:
        adv_series = compute_trailing_adv(df, window=14, stock_folder=f"data/{args.ticker}")
        burst_adv = df["Date"].map(adv_series)
        min_vol = (args.vol_frac * burst_adv).reindex(df.index)
        df = classify_and_filter(
            df, min_vol=0, dir_thresh=args.dir_thresh,
            vol_ratio=args.vol_ratio, kappa=args.kappa,
            require_directional=False, min_vol_per_burst=min_vol
        ).reset_index(drop=True)

    # Handle mixed direction
    if "Direction" in df.columns:
        mixed = df["Direction"] == 0
        if mixed.any():
            buy_dom = df.loc[mixed, "BuyVolume"] >= df.loc[mixed, "SellVolume"]
            df.loc[mixed & buy_dom, "Direction"] = 1
            df.loc[mixed & ~buy_dom, "Direction"] = -1

    print(f"\n{'='*120}")
    print(f"  TRANSACTION COST SENSITIVITY GRID")
    print(f"  Data: {args.data_csv}")
    print(f"  Bursts: {len(df):,}  |  Days: {df['Date'].nunique()}")
    if args.ticker:
        print(f"  Ticker: {args.ticker}")
    print(f"{'='*120}")

    # ── Build the grid ──
    grid_rows = []

    print(f"\n  {'Horizon':<10}", end="")
    for cost in COST_LEVELS_BPS:
        print(f"  {'TC='+str(cost)+'bps':>14}", end="")
    print(f"  {'N':>8}  {'Gross t':>8}")
    print(f"  {'-'*120}")

    for horizon_label, horizon_col in HORIZON_MAP.items():
        gross_bps, entry_px, valid = compute_gross_bps(df, exit_col=horizon_col)
        if gross_bps is None:
            print(f"  {horizon_label:<10}  N/A")
            continue

        n = len(gross_bps)
        gross_mean = np.mean(gross_bps)
        gross_std = np.std(gross_bps, ddof=1)
        gross_se = gross_std / np.sqrt(n)
        gross_t = gross_mean / gross_se if gross_se > 0 else 0.0

        line = f"  {horizon_label:<10}"

        for cost_bps in COST_LEVELS_BPS:
            # Net BPS = Gross BPS - round_trip_cost_bps
            net_bps = gross_bps - cost_bps
            net_mean = np.mean(net_bps)
            net_std = np.std(net_bps, ddof=1)
            net_se = net_std / np.sqrt(n)
            net_t = net_mean / net_se if net_se > 0 else 0.0

            sig = ""
            p_val = 2.0 * (1.0 - scipy_stats.t.cdf(abs(net_t), df=max(n - 1, 1)))
            if p_val < 0.01:
                sig = "***"
            elif p_val < 0.05:
                sig = "**"
            elif p_val < 0.10:
                sig = "*"

            line += f"  {net_mean:>8.2f}{sig:<5}"

            grid_rows.append({
                "horizon": horizon_label,
                "cost_bps": cost_bps,
                "n": n,
                "gross_mean_bps": gross_mean,
                "net_mean_bps": net_mean,
                "net_std_bps": net_std,
                "net_t_stat": net_t,
                "net_p_value": p_val,
            })

        line += f"  {n:>8,}  {gross_t:>8.2f}"
        print(line)

    # ── Per-horizon summary ──
    print(f"\n  Breakeven Analysis:")
    print(f"  {'Horizon':<10}  {'Gross Mean BPS':>16}  {'Max TC for +EV':>16}  {'Status':>10}")
    print(f"  {'-'*60}")

    for horizon_label, horizon_col in HORIZON_MAP.items():
        gross_bps, _, _ = compute_gross_bps(df, exit_col=horizon_col)
        if gross_bps is None:
            continue
        gross_mean = np.mean(gross_bps)
        # Breakeven: the gross mean IS the max TC you can afford
        if gross_mean > 0:
            status = "VIABLE" if gross_mean > 1.0 else "marginal"
        else:
            status = "NEGATIVE"
        print(f"  {horizon_label:<10}  {gross_mean:>16.2f}  {gross_mean:>16.2f}  {status:>10}")

    print(f"\n{'='*120}")

    # ── Save CSV ──
    if args.output_csv and grid_rows:
        pd.DataFrame(grid_rows).to_csv(args.output_csv, index=False)
        print(f"  Grid saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
