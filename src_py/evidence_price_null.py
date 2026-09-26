#!/usr/bin/env python3
"""program-evidence-v1 module B4 (post-hoc, after the 2021 module B read): price-matched nulls.

Dollar-sized orders from unrelated traders share a share count only when prices are close, so a
depth-matched cross-day null (different price level) can leave a common-state excess on both sides.
Here within-day and cross-day pairs must also share a price bucket: floor(ln(mid) / ln(1 + w)) for
w = 10 and 50 bps. Identical-size matches for same-side and opposite-side pairs, u_nonround. Cross-day
price-matched pairs exist only where adjacent days' prices overlap, so the aggregator also uses the
same day's price-matched rate at lags >= 1 hour as a null (it absorbs programs lasting hours).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import fingerprint_stats as FS

EDGES = np.array([0, 0.5, 2, 10, 60, 600, 3600, 23400.0])
WIDTHS_BPS = (10, 50)


def price_bucket(mid, w_bps):
    m = np.asarray(mid, float)
    out = np.full(len(m), -1, np.int64)
    ok = np.isfinite(m) & (m > 0)
    out[ok] = np.floor(np.log(m[ok]) / np.log1p(w_bps / 1e4)).astype(np.int64) + 10**6
    return out


def keys(dbin, pbin, size=None):
    k = np.asarray(dbin, np.int64) * 10**7 + np.asarray(pbin, np.int64)
    return k if size is None else k * 10**7 + np.asarray(size, np.int64)


def within(t, sign, dbin, pbin, size, ok):
    out = np.zeros((2, 2, len(EDGES) - 1), np.int64)            # [relation, pairs/matches, lag]
    for s in (1, -1):
        m = ok & (sign == s); o = ok & (sign == -s)
        out[0, 0] += FS.pair_counts(t[m], EDGES, key=keys(dbin[m], pbin[m]))
        out[0, 1] += FS.pair_counts(t[m], EDGES, key=keys(dbin[m], pbin[m], size[m]))
        out[1, 0] += FS.cross_counts(t[m], t[o], EDGES, keys(dbin[m], pbin[m]), keys(dbin[o], pbin[o]))
        out[1, 1] += FS.cross_counts(t[m], t[o], EDGES, keys(dbin[m], pbin[m], size[m]), keys(dbin[o], pbin[o], size[o]))
    return out


def cross(A, B):
    out = np.zeros((2, 2, len(EDGES) - 1), np.int64)
    for s in (1, -1):
        for rel, sb in ((0, s), (1, -s)):
            ma = A["ok"] & (A["sign"] == s); mb = B["ok"] & (B["sign"] == sb)
            ka, kb = keys(A["dbin"][ma], A["pbin"][ma]), keys(B["dbin"][mb], B["pbin"][mb])
            ksa, ksb = keys(A["dbin"][ma], A["pbin"][ma], A["size"][ma]), keys(B["dbin"][mb], B["pbin"][mb], B["size"][mb])
            ta, tb = A["t"][ma], B["t"][mb]
            out[rel, 0] += FS.cross_counts(ta, tb, EDGES, ka, kb) + FS.cross_counts(tb, ta, EDGES, kb, ka)
            out[rel, 1] += FS.cross_counts(ta, tb, EDGES, ksa, ksb) + FS.cross_counts(tb, ta, EDGES, ksb, ksa)
    return out


def analyze(days, pairs):
    P = [p for p in pairs if p[0] in days and p[1] in days]
    D = [d for p in P for d in p]
    res = dict(dates=np.array(D), pairs=np.array(["%s_%s" % p for p in P]))
    for w in WIDTHS_BPS:
        res["within_%d" % w] = np.zeros((len(D), 2, 2, len(EDGES) - 1), np.int64)
        res["cross_%d" % w] = np.zeros((len(P), 2, 2, len(EDGES) - 1), np.int64)
    for pi, (a, b) in enumerate(P):
        pool = np.concatenate([days[x]["exec_depth"][days[x]["untruncated"].astype(bool) & (days[x]["sign"] != 0)] for x in (a, b)])
        pool = pool[np.isfinite(pool)]
        thr = np.quantile(pool, FS.DEPTH_QUANTILES) if len(pool) else np.array([np.inf])
        prep = {}
        for x in (a, b):
            day = days[x]
            size, masks = FS.size_class_masks(day, np.array([], np.int64))
            prep[x] = dict(t=day["time"].astype(float), sign=day["sign"].astype(int), size=size,
                           dbin=FS.depth_bins(day["exec_depth"], thr), mid=day["mid"].astype(float), u=masks["u_nonround"])
        for w in WIDTHS_BPS:
            for x in (a, b):
                q = prep[x]
                q["pbin"] = price_bucket(q["mid"], w); q["ok"] = q["u"] & (q["pbin"] >= 0)
                res["within_%d" % w][D.index(x)] = within(q["t"], q["sign"], q["dbin"], q["pbin"], q["size"], q["ok"])
            res["cross_%d" % w][pi] = cross(prep[a], prep[b])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs = [tuple(x.split()) for x in Path(args.pairs).read_text().splitlines() if x.strip()]
    days = FS.load_days(args.packets, [d for p in pairs for d in p])
    res = analyze(days, pairs)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, ticker=np.array(args.ticker), edges=EDGES, **res)
    tmp.rename(out)
    print(json.dumps({"ticker": args.ticker, "pairs": len(res["pairs"])}))


if __name__ == "__main__":
    main()
