#!/usr/bin/env python3
"""program-evidence-v1 module H, stage 2: passive identical-size fingerprints (one ticker).

H1: identical-size same-side pairs among plain non-round adds (not replace halves, not posted
    remainders), keyed by position class, within day and across adjacent days at the same clock lag.
H2: pairs of an untruncated non-round aggressive packet and a plain non-round add of identical
    size, in either order, for the same economic side (buy aggressor with bid add) and, as a
    control, the opposite side; within day and across adjacent days.
Counts only; ratios are formed by the aggregator.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import fingerprint_stats as FS

EDGES = np.array([0, 0.5, 2, 10, 30.0])


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def plain_adds(a):
    return ~a["replace"].astype(bool) & ~a["remainder"].astype(bool)


def aggressive(p):
    size, masks = FS.size_class_masks(p, np.array([], np.int64))
    return masks["u_nonround"], size


def h1_within(a):
    out = np.zeros((2, len(EDGES) - 1), np.int64)                 # pairs, matches
    m0 = plain_adds(a)
    for side in (1, -1):
        m = m0 & (a["side"] == side)
        t = a["time"][m]; pos = a["pos"][m].astype(np.int64); size = a["size"][m].astype(np.int64)
        out[0] += FS.pair_counts(t, EDGES, key=pos)
        out[1] += FS.pair_counts(t, EDGES, key=pos * 10**7 + size)
    return out


def h1_cross(a, b):
    out = np.zeros((2, len(EDGES) - 1), np.int64)
    ma0, mb0 = plain_adds(a), plain_adds(b)
    for side in (1, -1):
        ma, mb = ma0 & (a["side"] == side), mb0 & (b["side"] == side)
        ta, tb = a["time"][ma], b["time"][mb]
        pa, pb = a["pos"][ma].astype(np.int64), b["pos"][mb].astype(np.int64)
        ka = pa * 10**7 + a["size"][ma].astype(np.int64); kb = pb * 10**7 + b["size"][mb].astype(np.int64)
        out[0] += FS.cross_counts(ta, tb, EDGES, pa, pb) + FS.cross_counts(tb, ta, EDGES, pb, pa)
        out[1] += FS.cross_counts(ta, tb, EDGES, ka, kb) + FS.cross_counts(tb, ta, EDGES, kb, ka)
    return out


def h2_counts(p, a):
    """[relation (same economic side, opposite), pairs/matches, lag bin]; lags in either order."""
    out = np.zeros((2, 2, len(EDGES) - 1), np.int64)
    u, size = aggressive(p)
    ma0 = plain_adds(a)
    for s in (1, -1):
        m = u & (p["sign"] == s)
        tp, sp = p["time"][m], size[m]
        for rel, add_side in ((0, s), (1, -s)):
            ma = ma0 & (a["side"] == add_side)
            ta, sa = a["time"][ma], a["size"][ma].astype(np.int64)
            out[rel, 0] += FS.cross_counts(tp, ta, EDGES) + FS.cross_counts(ta, tp, EDGES)
            out[rel, 1] += FS.cross_counts(tp, ta, EDGES, sp, sa) + FS.cross_counts(ta, tp, EDGES, sa, sp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adds", required=True, help="directory of <date>.npz from passive_extract.py")
    ap.add_argument("--packets", required=True, help="fingerprint-v1 packet directory for the ticker")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs = [tuple(x.split()) for x in Path(args.pairs).read_text().splitlines() if x.strip()]
    adds, packets = {}, {}
    for d in [x for p in pairs for x in p]:
        fa, fp = Path(args.adds) / (d + ".npz"), Path(args.packets) / (d + ".npz")
        if fa.is_file() and fp.is_file():
            a, p = load(fa), load(fp)
            if len(a.get("time", [])) and len(p.get("time", [])):
                adds[d], packets[d] = a, p
    D = sorted(adds); P = [q for q in pairs if q[0] in adds and q[1] in adds]
    res = dict(dates=np.array(D), pairs=np.array(["%s_%s" % q for q in P]),
               h1_within=np.zeros((len(D), 2, len(EDGES) - 1), np.int64),
               h1_cross=np.zeros((len(P), 2, len(EDGES) - 1), np.int64),
               h2_within=np.zeros((len(D), 2, 2, len(EDGES) - 1), np.int64),
               h2_cross=np.zeros((len(P), 2, 2, len(EDGES) - 1), np.int64),
               n_plain_adds=np.zeros(len(D), np.int64))
    for i, d in enumerate(D):
        res["h1_within"][i] = h1_within(adds[d])
        res["h2_within"][i] = h2_counts(packets[d], adds[d])
        res["n_plain_adds"][i] = int(plain_adds(adds[d]).sum())
    for i, (a, b) in enumerate(P):
        res["h1_cross"][i] = h1_cross(adds[a], adds[b])
        res["h2_cross"][i] = h2_counts(packets[a], adds[b]) + h2_counts(packets[b], adds[a])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, ticker=np.array(args.ticker), edges=EDGES, **res)
    tmp.rename(out)
    print(json.dumps({"ticker": args.ticker, "days": len(D), "pairs": len(P)}))


if __name__ == "__main__":
    main()
