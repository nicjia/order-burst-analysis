#!/usr/bin/env python3
"""
polygon_fetch_prices.py — Pull 2025-2026 daily open/close from Polygon.io
and merge with existing CRSP matrices to produce updated open_all.csv / close_all.csv.

Usage:
    python3 polygon_fetch_prices.py --api-key YOUR_KEY
    python3 polygon_fetch_prices.py --api-key YOUR_KEY --paid-tier   # skips rate limiting
"""

import argparse
import os
import time

import pandas as pd
import requests


def load_universe(path="universes/full_500.txt"):
    """Load ticker list, stripping comments and blanks."""
    tickers = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line.split()[0])  # take first token only
    return tickers


def fetch_polygon_daily(ticker, api_key, start_date, end_date):
    """Fetch daily OHLC bars from Polygon.io Aggregates endpoint."""
    url = (
        f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start_date}/{end_date}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "results" not in data or len(data["results"]) == 0:
        return None, None

    df = pd.DataFrame(data["results"])
    # Polygon 't' = Unix timestamp in milliseconds
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.strftime("%Y%m%d").astype(int)
    df.set_index("date", inplace=True)
    return df["o"], df["c"]  # open, close series


def main():
    ap = argparse.ArgumentParser(description="Fetch 2025-2026 prices from Polygon.io")
    ap.add_argument("--api-key", required=True, help="Polygon.io API key")
    ap.add_argument("--start-date", default="2025-01-01")
    ap.add_argument("--end-date", default="2026-12-31")
    ap.add_argument("--universe", default="universes/full_500.txt")
    ap.add_argument("--existing-open", default="open_all.csv",
                    help="Existing open price matrix (2016-2024) to merge with")
    ap.add_argument("--existing-close", default="close_all.csv",
                    help="Existing close price matrix (2016-2024) to merge with")
    ap.add_argument("--paid-tier", action="store_true",
                    help="Skip rate limiting (paid Polygon tier)")
    ap.add_argument("--output-dir", default=".", help="Output directory")
    args = ap.parse_args()

    tickers = load_universe(args.universe)
    print(f"Fetching {args.start_date} → {args.end_date} for {len(tickers)} tickers")
    if not args.paid_tier:
        print("  (Free tier: 5 calls/min — estimated ~100 min for 500 tickers)")
        print("  Use --paid-tier to skip rate limiting")

    open_dict = {}
    close_dict = {}
    success = 0
    failed = 0
    no_data = 0

    for i, ticker in enumerate(tickers, 1):
        try:
            o, c = fetch_polygon_daily(ticker, args.api_key, args.start_date, args.end_date)
            if o is not None:
                open_dict[ticker] = o
                close_dict[ticker] = c
                success += 1
                print(f"  [{i}/{len(tickers)}] ✓ {ticker}: {len(o)} days")
            else:
                no_data += 1
                print(f"  [{i}/{len(tickers)}] — {ticker}: no data")
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(tickers)}] ✗ {ticker}: {e}")

        # Rate limit: free tier = 5 calls/min → 12s between calls
        if not args.paid_tier:
            time.sleep(12)

    print(f"\nDone: {success} success, {no_data} no data, {failed} failed")

    # ── Build the 2025-2026 update matrices ──
    open_update = pd.DataFrame(open_dict)
    close_update = pd.DataFrame(close_dict)
    open_update.index.name = "date"
    close_update.index.name = "date"

    # Drop all-NaN rows (weekends/holidays)
    open_update = open_update.dropna(how="all")
    close_update = close_update.dropna(how="all")

    # Save the raw update
    open_update.to_csv(os.path.join(args.output_dir, "open_update_2025_2026.csv"))
    close_update.to_csv(os.path.join(args.output_dir, "close_update_2025_2026.csv"))
    print(f"Saved update files: {len(open_update)} trading days, {len(open_update.columns)} tickers")

    # ── Merge with existing 2016-2024 matrices ──
    if os.path.exists(args.existing_open) and os.path.exists(args.existing_close):
        print(f"\nMerging with existing matrices...")
        old_open = pd.read_csv(args.existing_open, index_col="date")
        old_close = pd.read_csv(args.existing_close, index_col="date")

        # Concatenate, keeping old data where there's overlap (shouldn't be any)
        merged_open = pd.concat([old_open, open_update])
        merged_close = pd.concat([old_close, close_update])

        # Remove duplicate dates (keep first = old data takes precedence)
        merged_open = merged_open[~merged_open.index.duplicated(keep="first")]
        merged_close = merged_close[~merged_close.index.duplicated(keep="first")]

        # Sort by date
        merged_open = merged_open.sort_index()
        merged_close = merged_close.sort_index()

        # Save merged
        merged_open.to_csv(os.path.join(args.output_dir, "open_all.csv"))
        merged_close.to_csv(os.path.join(args.output_dir, "close_all.csv"))
        print(f"Merged: {len(merged_open)} total days, "
              f"{len(merged_open.columns)} tickers, "
              f"date range {merged_open.index.min()} → {merged_open.index.max()}")
    else:
        print(f"\nNo existing matrices found at {args.existing_open} / {args.existing_close}")
        print("Saving update-only files. Manually merge if needed.")

    print("\n✓ Done. Upload open_all.csv and close_all.csv to the cluster:")
    print("  scp open_all.csv close_all.csv hoffman2.idre.ucla.edu:/u/scratch/n/nicjia/order-burst-analysis/")


if __name__ == "__main__":
    main()
