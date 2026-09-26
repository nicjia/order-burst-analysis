#!/usr/bin/env python3
"""Economic-packet revalidation and sign bounds for hidden executions."""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA
import execution_packets as EP


HORIZONS = (180, 900, 1800)


def _tick_sign(prices):
    signs = np.zeros(len(prices), int)
    last_price = np.nan; last_sign = 0
    for i, price in enumerate(np.asarray(prices, float)):
        if np.isfinite(last_price):
            if price > last_price:
                last_sign = 1
            elif price < last_price:
                last_sign = -1
        signs[i] = last_sign
        if np.isfinite(price):
            last_price = price
    return signs


def _signed_mean(raw, signs, selector):
    use = selector & np.isfinite(raw) & (np.abs(raw) <= 1000) & (signs != 0)
    return int(use.sum()), float(np.mean(signs[use] * raw[use])) if use.any() else np.nan


def summarize(packets, context, ticker, date):
    hidden = packets[(packets.n_hidden > 0) & (packets.time >= EP.RTH0)
                     & (packets.time < EP.RTH1 - max(HORIZONS))].copy()
    if hidden.empty:
        return pd.DataFrame()
    hidden = hidden.sort_values(["time", "packet_id"], kind="stable").reset_index(drop=True)
    time = hidden.time.to_numpy(float)
    sign = hidden.sign.to_numpy(int)
    mixed = hidden.n_visible.to_numpy(int) > 0
    outside = (~mixed) & (sign != 0)
    unsigned = (~mixed) & (sign == 0)
    pre_bid = hidden.pre_bid.to_numpy(float); pre_ask = hidden.pre_ask.to_numpy(float)
    pre_mid = (pre_bid + pre_ask) / 2.0
    price = hidden.vwap.to_numpy(float)
    tol = 0.5 / BA.SCALE
    midpoint = unsigned & (np.abs(hidden.min_price.to_numpy(float) - pre_mid) <= tol) \
        & (np.abs(hidden.max_price.to_numpy(float) - pre_mid) <= tol)
    away = unsigned & ~midpoint
    quote_sign = np.where(price > pre_mid + tol, 1,
                          np.where(price < pre_mid - tol, -1, 0))
    midpoint_idx = np.flatnonzero(midpoint)
    midpoint_tick = np.zeros(len(hidden), int)
    midpoint_tick[midpoint_idx] = _tick_sign(price[midpoint_idx])
    conventional = sign.copy()
    conventional[away] = quote_sign[away]
    conventional[midpoint] = midpoint_tick[midpoint]
    known = mixed | outside
    hidden_volume = hidden.hidden_volume.to_numpy(float)
    result = {
        "ticker": ticker, "date": int(date),
        "n_hidden_packets": int(len(hidden)),
        "n_hidden_messages": int(hidden.n_hidden.sum()),
        "hidden_volume": float(hidden_volume.sum()),
        "n_mixed_packets": int(mixed.sum()),
        "n_outside_packets": int(outside.sum()),
        "n_unsigned_packets": int(unsigned.sum()),
        "n_unsigned_away_packets": int(away.sum()),
        "n_midpoint_packets": int(midpoint.sum()),
        "n_unresolved_convention_packets": int((unsigned & (conventional == 0)).sum()),
        "mixed_hidden_volume": float(hidden_volume[mixed].sum()),
        "outside_hidden_volume": float(hidden_volume[outside].sum()),
        "unsigned_hidden_volume": float(hidden_volume[unsigned].sum()),
        "midpoint_hidden_volume": float(hidden_volume[midpoint].sum()),
    }
    bt, bm, _bb, _ba, _bbsz, _basz, _ofi, _trades = context
    reference = BA.mid_at(bt, bm, time + 1.0)
    for horizon in HORIZONS:
        label = "%dm" % (horizon // 60)
        future = BA.mid_at(bt, bm, time + horizon)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = (future - reference) / reference * 1e4
        valid = np.isfinite(raw) & np.isfinite(reference) & (reference > 0) \
            & (np.abs(raw) <= 1000)
        for name, selector, signs in (
            ("mixed", mixed, sign), ("outside", outside, sign),
            ("known", known, sign), ("away_quote", away, quote_sign),
            ("mid_tick", midpoint, midpoint_tick),
            ("all_conventional", conventional != 0, conventional),
        ):
            n, mean = _signed_mean(raw, signs, selector)
            result["n_%s_%s" % (name, label)] = n
            result["mk_%s_%s" % (name, label)] = mean
        fixed = valid & known
        unknown = valid & unsigned
        n_bound = int(fixed.sum() + unknown.sum())
        fixed_sum = float(np.sum(sign[fixed] * raw[fixed]))
        ambiguity = float(np.sum(np.abs(raw[unknown])))
        result["n_bound_all_%s" % label] = n_bound
        result["bound_all_lo_%s" % label] = (
            (fixed_sum - ambiguity) / n_bound if n_bound else np.nan
        )
        result["bound_all_hi_%s" % label] = (
            (fixed_sum + ambiguity) / n_bound if n_bound else np.nan
        )
        away_valid = valid & away
        result["n_bound_away_%s" % label] = int(away_valid.sum())
        result["bound_away_lo_%s" % label] = (
            -float(np.mean(np.abs(raw[away_valid]))) if away_valid.any() else np.nan
        )
        result["bound_away_hi_%s" % label] = (
            float(np.mean(np.abs(raw[away_valid]))) if away_valid.any() else np.nan
        )
    return pd.DataFrame([result])


def extract(msg_path, ticker):
    context, packets = EP.reconstruct_packets(msg_path)
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(msg_path))
    date = int("".join(match.groups())) if match else 0
    return summarize(packets, context, ticker, date)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()
    frame = extract(args.msg, args.ticker)
    if not frame.empty:
        frame.to_csv(sys.stdout, index=False, header=args.header)


if __name__ == "__main__":
    main()
