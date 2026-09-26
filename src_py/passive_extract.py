#!/usr/bin/env python3
"""program-evidence-v1 module H, stage 1: non-round limit-order submissions for one ticker-day.

From a LOBSTER message file (regular hours), keeps type-1 adds with non-round sizes and records:
side (LOBSTER direction: +1 bid, -1 ask), size, position against the quote prevailing strictly
before the add (0 inside the spread, 1 at the touch, 2 behind, 3 no valid quote), whether the add
is the add half of an ITCH replace (same timestamp and side as a delete), and whether it shares a
timestamp with an execution (the posted remainder of a marketable order). Licensed-data
derivative: keep on the cluster.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0


def classify(px, side, bid_ticks, ask_ticks):
    valid = np.isfinite(bid_ticks) & np.isfinite(ask_ticks) & (ask_ticks > bid_ticks)
    ref_touch = np.where(side > 0, bid_ticks, ask_ticks)
    better = np.where(side > 0, px > ref_touch, px < ref_touch)
    at = px == ref_touch
    cls = np.where(better, 0, np.where(at, 1, 2))
    return np.where(valid, cls, 3).astype(np.int8)


def extract(msg_path):
    df = pd.read_csv(msg_path, header=None, usecols=[0, 1, 3, 4, 5], names=["t", "ty", "sz", "px", "dr"])
    context = BA.reconstruct(msg_path)
    bt, _bm, bb, ba = context[0], context[1], context[2], context[3]
    df = df[(df.t >= RTH0) & (df.t < RTH1)]
    t = df.t.to_numpy(float); ty = df.ty.to_numpy(int); sz = df.sz.to_numpy(np.int64)
    px = df.px.to_numpy(np.int64); dr = df.dr.to_numpy(int)
    add = ty == 1
    ta, sa, pa, da = t[add], sz[add], px[add], dr[add]
    bid, ask = BA.bbo_at(bt, bb, ba, np.nextafter(ta, -np.inf))
    bid_ticks = np.round(bid * BA.SCALE); ask_ticks = np.round(ask * BA.SCALE)
    cls = classify(pa, da, bid_ticks, ask_ticks)
    deletes = pd.MultiIndex.from_arrays([t[ty == 3], dr[ty == 3]])
    replace = pd.MultiIndex.from_arrays([ta, da]).isin(deletes)
    exec_times = np.unique(t[(ty == 4) | (ty == 5)])
    remainder = np.isin(ta, exec_times)
    nonround = sa % 100 != 0
    counts = {}
    for side in (1, -1):
        for c in range(4):
            m = (da == side) & (cls == c)
            counts["side%+d_class%d" % (side, c)] = dict(
                all=int(m.sum()), nonround=int((m & nonround).sum()),
                nonround_plain=int((m & nonround & ~replace & ~remainder).sum()))
    keep = nonround
    return dict(time=ta[keep], side=da[keep].astype(np.int8), size=sa[keep], pos=cls[keep],
                replace=replace[keep], remainder=remainder[keep]), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arrays, counts = extract(args.msg)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(out)
    print(json.dumps({"ticker": args.ticker, "nonround_adds": int(len(arrays["time"])), "counts": counts}))


if __name__ == "__main__":
    main()
