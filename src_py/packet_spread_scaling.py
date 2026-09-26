#!/usr/bin/env python3
"""Packet-level recomputation of the spread-scaling law (VERIFIED_RESULTS.md 1.5).

The original result -- ``mk3 = 0.033 + 0.709 x half-spread``, cross-name correlation
+0.790, zero of forty names clearing a round trip -- was measured on execution
*messages* and on same-side message runs.  Both inputs are now suspect: one incoming
order emits several rows, so message-count weighting over-weights large orders, and
same-side run formation conditions on sign.  This script rebuilds the same regression
on economic packets from ``execution_packets.reconstruct_packets``.

The economic question is unchanged and is the reason the law matters.  If the signed
markout is a roughly constant *fraction* of the half-spread, then net of a round trip
every directional definition is dead at every spread by construction, which explains
68 failed definitions with one mechanism rather than 68 accidents.  If it is instead
roughly constant in *bps*, there is a spread threshold below which it is capturable.
The day's half-spread is therefore emitted beside every markout and never pre-binned.

Definitions, all on signed economic packets:
    all      every signed packet
    blk5     packets above 5x the day's median packet volume
    blk10    packets above 10x
    run3     runs of >=3 same-sign packets with sub-second gaps (sign-conditioned
             formation, retained only to reproduce the original construction)
    clust3   time clusters of >=3 packets with sub-second gaps, signed after the fact
             by net signed volume (price-free, sign-free formation)

Markouts are reported from the signal time and from a one-second buffer, since the
buffered figure is the defensible one and the gap between them is itself evidence.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA
import execution_packets as EP

HORIZONS = (180, 1800)
VARIANTS = ("all", "blk5", "blk10", "run3", "clust3")
MAX_H = max(HORIZONS)
GAP = 1.0
MIN_RUN = 3


def _trimmed_mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (np.abs(x) <= 1000)]
    return float(np.mean(x)) if len(x) else np.nan


def _runs(time, sign):
    """Same-sign consecutive packets with sub-second gaps.  Returns (end_time, sign)."""
    ends, signs = [], []
    i = 0
    n = len(time)
    while i < n:
        j = i
        while j + 1 < n and sign[j + 1] == sign[i] and (time[j + 1] - time[j]) < GAP:
            j += 1
        if j - i + 1 >= MIN_RUN:
            ends.append(time[j])
            signs.append(int(sign[i]))
        i = j + 1
    return np.asarray(ends, float), np.asarray(signs, int)


def _clusters(time, sign, volume):
    """Time-only clusters, signed after the fact by net signed volume."""
    ends, signs = [], []
    i = 0
    n = len(time)
    while i < n:
        j = i
        while j + 1 < n and (time[j + 1] - time[j]) < GAP:
            j += 1
        if j - i + 1 >= MIN_RUN:
            net = float(np.dot(sign[i:j + 1], volume[i:j + 1]))
            if net != 0.0:
                ends.append(time[j])
                signs.append(1 if net > 0 else -1)
        i = j + 1
    return np.asarray(ends, float), np.asarray(signs, int)


def _score(context, times, signs):
    """Half-spread at the signal and signed markouts from t and from t+1s."""
    bt, bm, bb, ba = context[0], context[1], context[2], context[3]
    out = {"n": int(len(times))}
    if len(times) < 3:
        out["hs"] = np.nan
        for horizon in HORIZONS:
            out["mk%d_t0" % (horizon // 60)] = np.nan
            out["mk%d_t1" % (horizon // 60)] = np.nan
        return out
    bid, ask = BA.bbo_at(bt, bb, ba, np.nextafter(times, -np.inf))
    mid_pre = 0.5 * (bid + ask)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["hs"] = _trimmed_mean(0.5 * (ask - bid) / mid_pre * 1e4)
    ref0 = BA.mid_at(bt, bm, times)
    ref1 = BA.mid_at(bt, bm, times + 1.0)
    for horizon in HORIZONS:
        future = BA.mid_at(bt, bm, times + horizon)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["mk%d_t0" % (horizon // 60)] = _trimmed_mean(
                signs * (future - ref0) / ref0 * 1e4)
            out["mk%d_t1" % (horizon // 60)] = _trimmed_mean(
                signs * (future - ref1) / ref1 * 1e4)
    return out


def summarize(packets, context, ticker, date):
    signed = EP.signed_packets(packets)
    if signed.empty:
        return pd.DataFrame()
    signed = signed[(signed.time >= EP.RTH0) & (signed.time < EP.RTH1 - MAX_H)]
    signed = signed.sort_values(["time", "packet_id"], kind="stable")
    if len(signed) < 50:
        return pd.DataFrame()
    time = signed.time.to_numpy(float)
    sign = signed.sign.to_numpy(int)
    volume = signed.volume.to_numpy(float)

    bt, bm, bb, ba = context[0], context[1], context[2], context[3]
    grid = np.arange(EP.RTH0, EP.RTH1 - MAX_H, 60.0)
    gbid, gask = BA.bbo_at(bt, bb, ba, grid)
    gmid = BA.mid_at(bt, bm, grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        halfsp_day = _trimmed_mean(0.5 * (gask - gbid) / gmid * 1e4)
    if not np.isfinite(halfsp_day) or halfsp_day <= 0:
        return pd.DataFrame()

    medvol = float(np.median(volume))
    selections = {
        "all": np.ones(len(signed), bool),
        "blk5": volume > 5.0 * medvol,
        "blk10": volume > 10.0 * medvol,
    }
    result = {
        "ticker": ticker, "date": int(date),
        "halfsp_day": halfsp_day,
        "med_packet_volume": medvol,
        "n_signed_packets": int(len(signed)),
        "n_messages_in_signed": int(signed.n_messages.sum()),
    }
    for name, mask in selections.items():
        for key, value in _score(context, time[mask], sign[mask]).items():
            result["%s_%s" % (name, key)] = value
    run_t, run_s = _runs(time, sign)
    for key, value in _score(context, run_t, run_s).items():
        result["run3_%s" % key] = value
    clu_t, clu_s = _clusters(time, sign, volume)
    for key, value in _score(context, clu_t, clu_s).items():
        result["clust3_%s" % key] = value
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
