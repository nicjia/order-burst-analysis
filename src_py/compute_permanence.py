#!/usr/bin/env python3
"""
compute_permanence.py — Phase I Permanence Calculation

Computes permanence ratios for every burst at multiple horizons:

  φ(b; x) = Direction × (x − m_tb) / PeakImpact(b)

Horizons:
  tCLOSE:  x = CloseMid            (last intraday mid from C++ pipeline)
  CLOP:    x = o_{i,t+1}           (next-day open  from CRSP)
  CLCL:    x = c_{i,t+1}           (next-day close from CRSP)
  1m/3m/5m/10m:  x = Mid at burst end + offset (from C++ pipeline)

Short-Horizon Decay Filter (Eq 3.2):
  D_b  = (1/4) Σ Direction × (Mid_τ − m_tb)   for τ ∈ {1m, 3m, 5m, 10m}
  Keep burst iff  D_b ≥ κ × PeakImpact(b)       (default κ = 0.10)

Usage:
    python compute_permanence.py <bursts_csv> <open_prices_csv> <close_prices_csv> [--kappa K]

Example:
    python src_py/compute_permanence.py bursts_tsla.csv open_all.csv close_all.csv --kappa 0.10
"""

import pandas as pd
import numpy as np
import argparse
import os


def permanence(direction, x, m_tb, peak_impact):
    """φ(b; x) = direction × (x − m_tb) / PeakImpact"""
    if peak_impact == 0 or pd.isna(x):
        return np.nan
    return direction * (x - m_tb) / peak_impact


def main():
    ap = argparse.ArgumentParser(
        description="Compute permanence ratios and apply short-horizon decay filter.")
    ap.add_argument('bursts_csv',
        help='CSV from data_processor (Ticker, Date, StartPrice, PeakPrice, CloseMid, Mid_1m …)')
    ap.add_argument('open_prices_csv',
        help='Pivot matrix of raw open prices (rows=date, cols=ticker)')
    ap.add_argument('close_prices_csv',
        help='Pivot matrix of raw close prices (rows=date, cols=ticker)')
    ap.add_argument('--kappa', type=float, default=0.10,
        help='Decay-filter threshold κ  (default 0.10). Set to 0 to disable.')
    args = ap.parse_args()

    bursts_file = args.bursts_csv
    open_file   = args.open_prices_csv
    close_file  = args.close_prices_csv
    kappa       = args.kappa

    # ── Load data ────────────────────────────────────────────
    bursts = pd.read_csv(bursts_file)
    print(f"Loaded {len(bursts)} bursts from {bursts_file}")

    open_px  = pd.read_csv(open_file, index_col='date')
    close_px = pd.read_csv(close_file, index_col='date')
    print(f"Open prices:  {open_px.shape[0]} dates × {open_px.shape[1]} tickers")
    print(f"Close prices: {close_px.shape[0]} dates × {close_px.shape[1]} tickers")

    # ── Filter zero-impact bursts ────────────────────────────
    bursts['PeakImpact'] = (bursts['PeakPrice'] - bursts['StartPrice']).abs()
    before = len(bursts)
    bursts = bursts[bursts['PeakImpact'] > 0].copy()
    print(f"Filtered {before - len(bursts)} zero-impact bursts → {len(bursts)} remaining")

    # ── RTH safety filter (belt-and-suspenders with C++) ─────
    #  9:30 AM = 34200 SPM,  4:00 PM = 57600 SPM
    RTH_START, RTH_END = 34200.0, 57600.0
    if 'StartTime' in bursts.columns:
        rth_mask = (bursts['StartTime'] >= RTH_START) & (bursts['StartTime'] <= RTH_END)
        n_outside = (~rth_mask).sum()
        if n_outside > 0:
            print(f"Dropped {n_outside} bursts outside RTH [{RTH_START:.0f}, {RTH_END:.0f}]")
            bursts = bursts[rth_mask].copy()
        else:
            print("RTH check: all bursts within regular trading hours ✓")

    # ── Convert burst dates to YYYYMMDD int to match CRSP ────
    # C++ outputs "2026-01-02", CRSP pivots use 20160104
    bursts['date_int'] = bursts['Date'].astype(str).str.replace('-', '').astype(int)

    # ── Build next-trading-day map from CRSP calendar ────────
    trading_days = sorted(close_px.index.tolist())
    next_day_map = {trading_days[i]: trading_days[i + 1]
                    for i in range(len(trading_days) - 1)}

    # ── tCLOSE: uses CloseMid from burst CSV (intraday) ─────
    bursts['Perm_tCLOSE'] = bursts.apply(
        lambda r: permanence(r['Direction'], r['CloseMid'], r['StartPrice'], r['PeakImpact']),
        axis=1)

    # ── Intraday forward returns (1m, 3m, 5m, 10m) ──────────
    for label, col in [('1m', 'Mid_1m'), ('3m', 'Mid_3m'),
                       ('5m', 'Mid_5m'), ('10m', 'Mid_10m')]:
        if col in bursts.columns:
            bursts[f'Perm_t{label}'] = bursts.apply(
                lambda r, c=col: permanence(r['Direction'], r[c], r['StartPrice'], r['PeakImpact']),
                axis=1)

    # ── D_b: Short-Horizon Decay Filter (Eq 3.2) ────────────
    #  D_b = (1/4) Σ_τ Direction × (Mid_τ − StartPrice)
    #  Keep burst iff  D_b ≥ κ × PeakImpact
    disp_cols = []
    for label, col in [('1m', 'Mid_1m'), ('3m', 'Mid_3m'),
                       ('5m', 'Mid_5m'), ('10m', 'Mid_10m')]:
        dcol = f'Disp_{label}'
        if col in bursts.columns:
            bursts[dcol] = bursts['Direction'] * (bursts[col] - bursts['StartPrice'])
            disp_cols.append(dcol)

    if disp_cols:
        bursts['D_b'] = bursts[disp_cols].mean(axis=1)
    else:
        bursts['D_b'] = np.nan
        print("WARNING: No Mid_1m/3m/5m/10m columns found – cannot compute D_b")

    # ── CLOP & CLCL: look up CRSP prices ────────────────────
    def lookup_crsp(row, price_df, use_next_day):
        """Look up a CRSP price for this burst's ticker on today or next trading day."""
        ticker = row['Ticker']
        today  = row['date_int']

        if ticker not in price_df.columns:
            return np.nan

        if use_next_day:
            target_day = next_day_map.get(today)
            if target_day is None:
                return np.nan   # last trading day in dataset
        else:
            target_day = today

        if target_day not in price_df.index:
            return np.nan

        return price_df.loc[target_day, ticker]

    # CLOP: x = open price of next trading day  (o_{i,t+1})
    bursts['x_CLOP'] = bursts.apply(lambda r: lookup_crsp(r, open_px, use_next_day=True), axis=1)
    bursts['Perm_CLOP'] = bursts.apply(
        lambda r: permanence(r['Direction'], r['x_CLOP'], r['StartPrice'], r['PeakImpact']),
        axis=1)

    # CLCL: x = close price of next trading day  (c_{i,t+1})
    bursts['x_CLCL'] = bursts.apply(lambda r: lookup_crsp(r, close_px, use_next_day=True), axis=1)
    bursts['Perm_CLCL'] = bursts.apply(
        lambda r: permanence(r['Direction'], r['x_CLCL'], r['StartPrice'], r['PeakImpact']),
        axis=1)

    # ── Clean up helper columns ───────────────────────────────
    drop_cols = ['date_int', 'x_CLOP', 'x_CLCL'] + [c for c in bursts.columns if c.startswith('Disp_')]
    bursts.drop(columns=[c for c in drop_cols if c in bursts.columns], inplace=True)

    # ── Duration column (seconds) ────────────────────────────
    if 'StartTime' in bursts.columns and 'EndTime' in bursts.columns:
        bursts['Duration'] = bursts['EndTime'] - bursts['StartTime']

    # ── Save UNFILTERED (all permanence columns, with D_b) ───
    base, ext = os.path.splitext(bursts_file)
    unfiltered_file = f"{base}_unfiltered{ext}"
    bursts.to_csv(unfiltered_file, index=False)
    print(f"\nWrote {len(bursts)} bursts (unfiltered) → {unfiltered_file}")

    # ── Apply D_b ≥ κ × PeakImpact filter ────────────────────
    before_filter = len(bursts)
    if kappa > 0 and 'D_b' in bursts.columns:
        mask = bursts['D_b'] >= kappa * bursts['PeakImpact']
        # Also drop rows where D_b couldn't be computed
        mask = mask & bursts['D_b'].notna()
        filtered = bursts[mask].copy()
        print(f"Decay filter (D_b ≥ {kappa}×PeakImpact): "
              f"{before_filter} → {len(filtered)} bursts  "
              f"({before_filter - len(filtered)} removed, "
              f"{100*len(filtered)/max(before_filter,1):.1f}% kept)")
    else:
        filtered = bursts.copy()
        print(f"Decay filter disabled (kappa={kappa})")

    filtered_file = f"{base}_filtered{ext}"
    filtered.to_csv(filtered_file, index=False)
    print(f"Wrote {len(filtered)} bursts (filtered)   → {filtered_file}")

    # ── Summary ──────────────────────────────────────────────
    perm_cols = [c for c in bursts.columns if c.startswith('Perm_')]
    print(f"\nPermanence columns: {perm_cols}")
    print()
    show_cols = ['Ticker', 'Date', 'Direction', 'PeakImpact', 'D_b'] + perm_cols
    show_cols = [c for c in show_cols if c in bursts.columns]
    print("Unfiltered sample (first 10):")
    print(bursts[show_cols].head(10).to_string(index=False))
    if len(filtered) > 0:
        print("\nFiltered sample (first 10):")
        print(filtered[show_cols].head(10).to_string(index=False))

    # Coverage stats
    print(f"\nCoverage (unfiltered):")
    for col in perm_cols + ['D_b']:
        if col in bursts.columns:
            valid = bursts[col].notna().sum()
            print(f"  {col}: {valid}/{len(bursts)} ({100*valid/len(bursts):.1f}%)")
    print(f"\nCoverage (filtered):")
    for col in perm_cols + ['D_b']:
        if col in filtered.columns:
            valid = filtered[col].notna().sum()
            print(f"  {col}: {valid}/{len(filtered)} ({100*valid/max(len(filtered),1):.1f}%)")


if __name__ == '__main__':
    main()
