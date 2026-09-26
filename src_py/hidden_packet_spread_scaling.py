#!/usr/bin/env python3
"""Packet-level hidden-execution footprint and quoted-spread scaling."""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA
import execution_packets as EP


def _tick_sign(prices):
    out = np.zeros(len(prices), int); last = np.nan; sign = 0
    for i, price in enumerate(np.asarray(prices, float)):
        if np.isfinite(last):
            if price > last:
                sign = 1
            elif price < last:
                sign = -1
        out[i] = sign
        if np.isfinite(price):
            last = price
    return out


def summarize(packets, context, ticker, date):
    p = packets[(packets.n_hidden > 0) & (packets.time >= EP.RTH0)
                & (packets.time < EP.RTH1 - 180.0)].copy()
    if p.empty:
        return pd.DataFrame()
    p = p.sort_values(["time", "packet_id"], kind="stable").reset_index(drop=True)
    mixed = p.n_visible.to_numpy(int) > 0
    sign = p.sign.to_numpy(int)
    outside = (~mixed) & (sign != 0)
    unsigned = (~mixed) & (sign == 0)
    mid = (p.pre_bid.to_numpy(float) + p.pre_ask.to_numpy(float)) / 2.0
    price = p.vwap.to_numpy(float); tol = 0.5 / BA.SCALE
    midpoint = unsigned & (np.abs(p.min_price.to_numpy(float) - mid) <= tol) \
        & (np.abs(p.max_price.to_numpy(float) - mid) <= tol)
    away = unsigned & ~midpoint
    quote_sign = np.where(price > mid + tol, 1, np.where(price < mid - tol, -1, 0))
    midpoint_tick = np.zeros(len(p), int)
    midpoint_idx = np.flatnonzero(midpoint)
    midpoint_tick[midpoint_idx] = _tick_sign(price[midpoint_idx])
    conventional = sign.copy()
    conventional[away] = quote_sign[away]
    conventional[midpoint] = midpoint_tick[midpoint]
    known = mixed | outside
    bt, bm, _bb, _ba, _bbsz, _basz, _ofi, _trades = context
    reference = BA.mid_at(bt, bm, p.time.to_numpy(float) + 1.0)
    future = BA.mid_at(bt, bm, p.time.to_numpy(float) + 180.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = (future - reference) / reference * 1e4
        halfspread = (p.pre_ask.to_numpy(float) - p.pre_bid.to_numpy(float)) \
            / (2.0 * mid) * 1e4
    result = {"ticker": ticker, "date": int(date)}
    for name, selector, signs in (
        ("known", known, sign), ("outside", outside, sign),
        ("away_quote", away, quote_sign),
        ("all_conventional", conventional != 0, conventional),
    ):
        use = selector & (signs != 0) & np.isfinite(raw) & (np.abs(raw) <= 1000) \
            & np.isfinite(halfspread) & (halfspread > 0)
        result["n_" + name] = int(use.sum())
        result["mk3_" + name] = (
            float(np.mean(signs[use] * raw[use])) if use.any() else np.nan
        )
        result["halfspread_" + name] = (
            float(np.mean(halfspread[use])) if use.any() else np.nan
        )
    return pd.DataFrame([result])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()
    context, packets = EP.reconstruct_packets(args.msg)
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(args.msg))
    date = int("".join(match.groups())) if match else 0
    frame = summarize(packets, context, args.ticker, date)
    if not frame.empty:
        frame.to_csv(sys.stdout, index=False, header=args.header)


if __name__ == "__main__":
    main()
