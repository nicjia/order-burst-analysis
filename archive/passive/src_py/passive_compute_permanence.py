#!/usr/bin/env python3
"""
passive_compute_permanence.py — Passive Burst Target Calculation

Computes the Transformed Price Drift target for passive limit order bursts.
Unlike the aggressive trade pipeline which uses Volume-Scaled Impact (VSI),
the passive target is PURELY price-based because:

  1. Passive volume ($Q_b$) is resting intent, not consumed capital.
  2. If an institution's bid is so informational that HFTs front-run it,
     the bid may never fill — but the INFORMATION perfectly predicted
     the subsequent price surge.
  3. Scaling by fill probability would assign target=0 to our strongest signals.

Target Variable (Transformed Price Drift):
    Target(τ) = arcsinh(side_b × (P_τ - m_{end}) / m_{end} × 10000)

  - side_b: +1 for Bid-heavy bursts, -1 for Ask-heavy bursts
  - m_{end}: mid-price at burst end
  - P_τ: future price at horizon τ (Close, Next Open, Next Close)
  - The ×10000 converts fractional returns to basis points before arcsinh.

The passive volume Q_b is left in the feature matrix for the ML model.

Usage:
    python passive_compute_permanence.py <bursts_csv> <open_csv> <close_csv> [--kappa K] [--ticker T]
"""

import pandas as pd
import numpy as np
import argparse
import os


def main():
    ap = argparse.ArgumentParser(
        description="Compute Transformed Price Drift targets for passive bursts.")
    ap.add_argument('bursts_csv',
        help='CSV from passive_data_processor')
    ap.add_argument('open_prices_csv',
        help='Pivot matrix of raw open prices (rows=date, cols=ticker)')
    ap.add_argument('close_prices_csv',
        help='Pivot matrix of raw close prices (rows=date, cols=ticker)')
    ap.add_argument('--kappa', type=float, default=0.0,
        help='Decay-filter threshold κ (default 0.0 = disabled for passive)')
    ap.add_argument('--ticker', type=str, default=None,
        help='Force the Ticker column to this value')
    args = ap.parse_args()

    bursts = pd.read_csv(args.bursts_csv)
    print(f"Loaded {len(bursts)} passive bursts from {args.bursts_csv}")

    open_px  = pd.read_csv(args.open_prices_csv, index_col='date')
    close_px = pd.read_csv(args.close_prices_csv, index_col='date')
    print(f"Open prices:  {open_px.shape[0]} dates × {open_px.shape[1]} tickers")
    print(f"Close prices: {close_px.shape[0]} dates × {close_px.shape[1]} tickers")

    if args.ticker is not None:
        bursts['Ticker'] = args.ticker
        print(f"Force-Set Ticker column to '{args.ticker}'")

    # ── Ensure we have a BurstVolume column for features ─────
    if 'BurstVolume' not in bursts.columns:
        if 'Volume' in bursts.columns:
            bursts['BurstVolume'] = bursts['Volume']
        else:
            bursts['BurstVolume'] = 1.0

    # ── PeakImpact (still a valid feature) ───────────────────
    bursts['PeakImpact'] = (bursts['PeakPrice'] - bursts['StartPrice']).abs()

    # ── RTH safety filter ────────────────────────────────────
    RTH_START, RTH_END = 34200.0, 57600.0
    if 'StartTime' in bursts.columns:
        rth_mask = (bursts['StartTime'] >= RTH_START) & (bursts['StartTime'] <= RTH_END)
        n_outside = (~rth_mask).sum()
        if n_outside > 0:
            print(f"Dropped {n_outside} bursts outside RTH")
            bursts = bursts[rth_mask].copy()

    # ── Reference price: mid-price at burst END ──────────────
    # For passive bursts, the reference is the mid at burst close,
    # NOT at start, because we want to measure the drift AFTER
    # the passive intent was deposited.
    ref_price = bursts['EndPrice'].astype('float64')
    # Fallback to StartPrice if EndPrice is missing
    ref_price = ref_price.fillna(bursts['StartPrice'].astype('float64'))

    # ── tCLOSE: Transformed Price Drift to same-day close ────
    bursts['Perm_tCLOSE'] = np.arcsinh(
        bursts['Direction'] * (bursts['CloseMid'] - ref_price) / ref_price * 10000
    )

    # ── D_b for optional filtering ───────────────────────────
    # D_b here uses the pure price drift (no volume scaling)
    disp_cols = []
    for label, col in [('1m', 'Mid_1m'), ('3m', 'Mid_3m'),
                       ('5m', 'Mid_5m'), ('10m', 'Mid_10m')]:
        dcol = f'Disp_{label}'
        if col in bursts.columns:
            bursts[dcol] = bursts['Direction'] * (bursts[col] - ref_price) / ref_price * 10000
            disp_cols.append(dcol)

    if disp_cols:
        bursts['D_b'] = bursts[disp_cols].mean(axis=1)
    else:
        bursts['D_b'] = np.nan

    # ── Queue Exhaustion Rate (placeholder — requires tracking fills) ──
    # For now, we can approximate with CancelRatio if available
    if 'CancelRatio' in bursts.columns:
        bursts['QueueExhaustionRate'] = 1.0 - bursts['CancelRatio']
    else:
        bursts['QueueExhaustionRate'] = np.nan

    # ── CRSP lookup for overnight and next-day targets ───────
    bursts['date_int'] = bursts['Date'].astype(str).str.replace('-', '').astype(int)
    trading_days = np.array(sorted(pd.Index(close_px.index).astype(int).tolist()), dtype=np.int64)
    idx = np.searchsorted(trading_days, bursts['date_int'].to_numpy(dtype=np.int64), side='right')
    target_day = np.full(len(bursts), np.nan, dtype=float)
    valid_idx = idx < len(trading_days)
    target_day[valid_idx] = trading_days[idx[valid_idx]]
    bursts['target_day'] = target_day

    bursts['CRSP_OP'] = np.nan
    bursts['CRSP_CL'] = np.nan

    ticker_arr = bursts['Ticker'].astype(str).to_numpy()
    target_arr = bursts['target_day'].to_numpy()

    for tkr in pd.unique(ticker_arr):
        row_mask = ticker_arr == tkr
        if not np.any(row_mask):
            continue

        if tkr in open_px.columns:
            s_op = open_px[tkr].dropna()
            if not s_op.empty:
                op_days = pd.Index(s_op.index).astype(int).to_numpy(dtype=np.int64)
                op_vals = s_op.to_numpy(dtype=float)
                m = row_mask & ~np.isnan(target_arr)
                if np.any(m):
                    td = target_arr[m].astype(np.int64)
                    ii = np.searchsorted(op_days, td, side='left')
                    ok = ii < len(op_days)
                    if np.any(ok):
                        ridx = np.where(m)[0][ok]
                        bursts.loc[ridx, 'CRSP_OP'] = op_vals[ii[ok]]

        if tkr in close_px.columns:
            s_cl = close_px[tkr].dropna()
            if not s_cl.empty:
                cl_days = pd.Index(s_cl.index).astype(int).to_numpy(dtype=np.int64)
                cl_vals = s_cl.to_numpy(dtype=float)
                m = row_mask & ~np.isnan(target_arr)
                if np.any(m):
                    td = target_arr[m].astype(np.int64)
                    ii = np.searchsorted(cl_days, td, side='left')
                    ok = ii < len(cl_days)
                    if np.any(ok):
                        ridx = np.where(m)[0][ok]
                        bursts.loc[ridx, 'CRSP_CL'] = cl_vals[ii[ok]]

    # ── CLOP: Transformed Price Drift to next-day open ───────
    # The reference price must be the CloseMid, as the portfolio is executed
    # exactly at the MOC (4:00 PM). We want the return from Close to Open.
    bursts['Perm_CLOP'] = np.arcsinh(
        bursts['Direction'] * (bursts['CRSP_OP'] - bursts['CloseMid']) / bursts['CloseMid'] * 10000
    )

    # ── CLCL: Transformed Price Drift to next-day close ──────
    bursts['Perm_CLCL'] = np.arcsinh(
        bursts['Direction'] * (bursts['CRSP_CL'] - bursts['CloseMid']) / bursts['CloseMid'] * 10000
    )

    # ── Coverage diagnostics ─────────────────────────────────
    for col in ['Perm_tCLOSE', 'Perm_CLOP', 'Perm_CLCL']:
        valid = bursts[col].notna().sum()
        print(f"  {col}: {valid}/{len(bursts)} ({100*valid/max(len(bursts),1):.1f}%)")

    # ── Cleanup ──────────────────────────────────────────────
    drop_cols = ['date_int', 'target_day', 'CRSP_OP', 'CRSP_CL'] + \
                [c for c in bursts.columns if c.startswith('Disp_')]
    bursts.drop(columns=[c for c in drop_cols if c in bursts.columns], inplace=True)

    if 'StartTime' in bursts.columns and 'EndTime' in bursts.columns:
        bursts['Duration'] = bursts['EndTime'] - bursts['StartTime']

    # ── Apply D_b filter ─────────────────────────────────────
    base, ext = os.path.splitext(args.bursts_csv)
    kappa = args.kappa
    before = len(bursts)
    if kappa > 0 and 'D_b' in bursts.columns:
        mask = (bursts['D_b'] >= kappa) & bursts['D_b'].notna()
        filtered = bursts[mask].copy()
        print(f"Decay filter (D_b >= {kappa}): {before} → {len(filtered)} "
              f"({before - len(filtered)} removed)")
    else:
        filtered = bursts.copy()
        print(f"Decay filter disabled (kappa={kappa})")

    out_file = f"{base}_filtered{ext}"
    filtered.to_csv(out_file, index=False)
    print(f"Wrote {len(filtered)} passive bursts → {out_file}")

    # Summary
    perm_cols = [c for c in filtered.columns if c.startswith('Perm_')]
    show_cols = ['Ticker', 'Date', 'Direction', 'Volume', 'PeakImpact', 'D_b'] + perm_cols
    show_cols = [c for c in show_cols if c in filtered.columns]
    if len(filtered) > 0:
        print("\nSample (first 5):")
        print(filtered[show_cols].head(5).to_string(index=False))


if __name__ == '__main__':
    main()
