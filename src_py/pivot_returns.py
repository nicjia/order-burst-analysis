#!/usr/bin/env python3
"""
pivot_returns.py — Build ticker×date pivot tables from daily CRSP files.

Modes:
  Single year:   python src_py/pivot_returns.py yearly/2016
  All years:     python src_py/pivot_returns.py yearly/

When given a parent folder containing year subfolders (2016/, 2017/, …),
processes every year and produces merged all-years files so burst data
spanning multiple years (e.g. AMZN 2022-01-01 to 2024-01-01) can be
looked up from a single matrix.

Outputs (in output_dir, default '.'):
  Per-year:   pvCLCL_2016.csv, OPCL_2016.csv, open_2016.csv, close_2016.csv, …
  Merged:     pvCLCL_all.csv,  OPCL_all.csv,  open_all.csv,  close_all.csv

Lookup:
    df = pd.read_csv('open_all.csv', index_col='date')
    val = df.loc[20220103, 'AMZN']
"""

import pandas as pd
import glob
import os
import sys


def process_year_folder(data_folder, output_dir):
    """Process a single year folder. Returns the combined long-form DataFrame."""
    year = os.path.basename(os.path.normpath(data_folder))

    gz_files = sorted(glob.glob(os.path.join(data_folder, '*.csv.gz')))
    if not gz_files:
        print(f"  {year}: no *.csv.gz files — skipping")
        return None

    rows = []
    for f in gz_files:
        df = pd.read_csv(f, compression='gzip',
                         usecols=['ticker', 'date', 'pvCLCL', 'OPCL', 'open', 'close'])
        rows.append(df)

    combined = pd.concat(rows, ignore_index=True)
    print(f"  {year}: {len(gz_files)} files, {len(combined)} rows, "
          f"{combined['ticker'].nunique()} tickers, {combined['date'].nunique()} dates")

    # Write per-year pivots
    for col, prefix in [('pvCLCL', 'pvCLCL'), ('OPCL', 'OPCL'),
                        ('open', 'open'), ('close', 'close')]:
        piv = combined.pivot(index='date', columns='ticker', values=col)
        piv.index.name = 'date'
        piv.to_csv(os.path.join(output_dir, f'{prefix}_{year}.csv'))

    return combined


def main():
    if len(sys.argv) < 2:
        print("Usage: python pivot_returns.py <data_folder> [output_dir]")
        print()
        print("  data_folder: a single year folder (yearly/2016) or")
        print("               a parent folder with year subfolders (yearly/)")
        print("  output_dir:  where to write CSVs (default: current dir)")
        sys.exit(1)

    data_folder = sys.argv[1]
    output_dir  = sys.argv[2] if len(sys.argv) > 2 else '.'
    os.makedirs(output_dir, exist_ok=True)

    # Detect mode: does data_folder contain year subfolders?
    subdirs = sorted([d for d in os.listdir(data_folder)
                      if os.path.isdir(os.path.join(data_folder, d)) and d.isdigit()])

    if subdirs:
        # ── Multi-year mode ──────────────────────────────────
        print(f"Found {len(subdirs)} year folders: {subdirs[0]} … {subdirs[-1]}")
        all_dfs = []
        for sub in subdirs:
            year_path = os.path.join(data_folder, sub)
            df = process_year_folder(year_path, output_dir)
            if df is not None:
                all_dfs.append(df)

        if not all_dfs:
            print("Error: No data found in any year folder.")
            sys.exit(1)

        # Merge all years
        merged = pd.concat(all_dfs, ignore_index=True)
        print(f"\nMerged: {len(merged)} total rows, "
              f"{merged['ticker'].nunique()} tickers, "
              f"{merged['date'].nunique()} dates")

        for col, prefix in [('pvCLCL', 'pvCLCL'), ('OPCL', 'OPCL'),
                            ('open', 'open'), ('close', 'close')]:
            piv = merged.pivot(index='date', columns='ticker', values=col)
            piv.index.name = 'date'
            out_path = os.path.join(output_dir, f'{prefix}_all.csv')
            piv.to_csv(out_path)
            print(f"Wrote {out_path}  ({piv.shape[0]} dates × {piv.shape[1]} tickers)")

    else:
        # ── Single-year mode ─────────────────────────────────
        print(f"Single-year mode: {data_folder}")
        df = process_year_folder(data_folder, output_dir)
        if df is None:
            sys.exit(1)
        year = os.path.basename(os.path.normpath(data_folder))
        print(f"\nDone — per-year files written for {year}")


if __name__ == '__main__':
    main()
