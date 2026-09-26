#!/usr/bin/env python3
"""
precompute_lob_volume.py — Merge C++ ADV side-outputs into unified true_adv_daily.csv

The C++ data_processor now emits *_adv.csv files alongside burst CSVs.
This script simply merges those per-ticker ADV files into the unified
results/true_adv_daily.csv that downstream scripts expect.

This replaces the old approach of re-parsing all LOBSTER message files in Python.

Usage:
    python3 src_py/precompute_lob_volume.py
    python3 src_py/precompute_lob_volume.py --tickers NVDA,TSLA,JPM,MS --results-dir results
"""

import argparse
import glob
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Merge C++ ADV side-outputs into unified true_adv_daily.csv")
    ap.add_argument("--tickers", default="NVDA,TSLA,JPM,MS,AAPL,LLY,SPY",
                    help="Comma-separated tickers to include")
    ap.add_argument("--results-dir", default="results",
                    help="Directory containing *_adv.csv files (default: results)")
    ap.add_argument("--output", default="results/true_adv_daily.csv",
                    help="Output path (default: results/true_adv_daily.csv)")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    all_frames = []

    for ticker in tickers:
        # Look for ADV files from the C++ side-output
        patterns = [
            os.path.join(args.results_dir, f"bursts_{ticker}_*_adv.csv"),
            os.path.join(args.results_dir, f"{ticker}_*_adv.csv"),
        ]

        found = False
        for pattern in patterns:
            adv_files = sorted(glob.glob(pattern))
            if adv_files:
                # Use the first (baseline) ADV file
                adv_path = adv_files[0]
                try:
                    df = pd.read_csv(adv_path)
                    if "Ticker" not in df.columns:
                        df["Ticker"] = ticker
                    all_frames.append(df)
                    print(f"  {ticker}: {len(df)} days from {adv_path}")
                    found = True
                    break
                except Exception as e:
                    print(f"  {ticker}: Error reading {adv_path}: {e}")

        if not found:
            # Fallback: try the old data/ folder approach with raw LOBSTER files
            stock_folder = os.path.join("data", ticker)
            msg_files = sorted(glob.glob(os.path.join(stock_folder, "*_message_*.csv")))
            if msg_files:
                print(f"  {ticker}: No C++ ADV file found; falling back to LOBSTER parsing ({len(msg_files)} files)...")
                from concurrent.futures import ProcessPoolExecutor

                def process_file(fpath):
                    try:
                        msg_df = pd.read_csv(fpath, header=None, usecols=[0, 1, 3],
                                             names=['Time', 'Type', 'Size'], engine='c')
                        rth = (msg_df['Time'] >= 34200.0) & (msg_df['Time'] <= 57600.0)
                        traded = msg_df[rth & msg_df['Type'].isin([4, 5])]['Size'].sum()
                        fname = os.path.basename(fpath)
                        date_str = fname.split('_')[1]
                        return date_str, traded
                    except Exception as e:
                        return None, None

                results = []
                with ProcessPoolExecutor(max_workers=4) as executor:
                    for date_str, vol in executor.map(process_file, msg_files):
                        if date_str is not None:
                            results.append({"Ticker": ticker, "Date": date_str, "TradedVolume": vol})
                if results:
                    all_frames.append(pd.DataFrame(results))
            else:
                print(f"  {ticker}: No ADV file and no LOBSTER data found; skipping")

    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
        # Deduplicate (prefer C++ side-output if duplicates exist)
        merged = merged.drop_duplicates(subset=["Ticker", "Date"], keep="first")
        merged.to_csv(args.output, index=False)
        print(f"\nSaved {len(merged)} daily volume records to {args.output}")
    else:
        print("WARNING: No daily volume data found for any ticker.")


if __name__ == "__main__":
    main()
