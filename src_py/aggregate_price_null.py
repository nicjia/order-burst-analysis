#!/usr/bin/env python3
"""Aggregate post-hoc module B4: identical-size matches under price-matched nulls (one group).

For each name, observed / expected identical-size matches for same-side and opposite-side pairs
(non-round untruncated, depth quartile and price bucket matched) at 2-10 s, 10-60 s and 60-600 s,
against (a) the same day's price-matched rate at lags >= 3600 s and (b) the price-matched cross-day
rate by lag bin. Name medians with a 95% name bootstrap; names need >= 20 expected matches.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

BOOT = 1000
RANGES = ((2, 10), (10, 60), (60, 600))
MIN_EXPECTED = 20.0


def med(v, rng):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if not len(v):
        return None
    b = np.median(v[rng.integers(0, len(v), (BOOT, len(v)))], axis=1)
    return dict(median=float(np.median(v)), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], names=int(len(v)))


def name_ratios(z, width):
    edges = z["edges"]
    w = z["within_%d" % width].sum(0).astype(float)          # [rel, pairs/matches, lag]
    c = z["cross_%d" % width].sum(0).astype(float)
    longm = edges[:-1] >= 3600
    out = {}
    for rel in (0, 1):
        lp, lm = w[rel, 0][longm].sum(), w[rel, 1][longm].sum()
        long_rate = lm / lp if lp > 0 else np.nan
        cross_rate = np.where(c[rel, 0] > 0, c[rel, 1] / np.maximum(c[rel, 0], 1), np.nan)
        for lo, hi in RANGES:
            m = (edges[:-1] >= lo) & (edges[1:] <= hi)
            obs = w[rel, 1][m].sum()
            e_long = w[rel, 0][m].sum() * long_rate
            e_cross = np.nansum(w[rel, 0][m] * cross_rate[m])
            key = "%s_%g-%g" % (("same", "opposite")[rel], lo, hi)
            out[key + "_vs_longlag"] = obs / e_long if e_long >= MIN_EXPECTED else np.nan
            out[key + "_vs_crossday"] = obs / e_cross if e_cross >= MIN_EXPECTED else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260914)
    names = []
    for p in sorted(glob.glob(args.stats)):
        with np.load(p) as z:
            if len(z["pairs"]):
                names.append({k: z[k] for k in z.files})
    res = dict(names=len(names))
    for width in (10, 50):
        per = [name_ratios(z, width) for z in names]
        keys = sorted(per[0])
        res["width_%dbps" % width] = {k: med([r[k] for r in per], rng) for k in keys}
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    for width in (10, 50):
        print("width", width)
        for k, v in res["width_%dbps" % width].items():
            if v:
                print("  %-34s %.3f [%.3f, %.3f] n=%d" % (k, v["median"], v["ci95"][0], v["ci95"][1], v["names"]))


if __name__ == "__main__":
    main()
