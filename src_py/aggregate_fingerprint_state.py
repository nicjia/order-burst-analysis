#!/usr/bin/env python3
"""Aggregate the corrected E3 statistic (fingerprint_state.py): lag-held-fixed state similarity.

Per name, sum counts and absolute-difference sums over days. Within a lag range, use only log-lag bins
with at least MIN_BIN_CONTROL control pairs, and reweight control means to the matched pairs' bin
counts. The ratio is matched mean / reweighted control mean, so lag composition cannot drive it.
Names need MIN_NAME_MATCHED matched pairs across usable bins. Median over names, bootstrap over names.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

FEATURES = ("spread_bps", "log_exec_depth", "imbalance")
RANGES = {"0.5-2s": (0.5, 2.0), "2-10s": (2.0, 10.0), "10-30s": (10.0, 30.0),
          "30-120s": (30.0, 120.0), "120-600s": (120.0, 600.0)}
MIN_BIN_CONTROL = 20
MIN_NAME_MATCHED = 30
BOOT = 1000


def load(group):
    names = []
    for path in sorted(glob.glob(str(Path(group) / "state_v3" / "*.npz"))):
        with np.load(path) as z:
            if len(z["dates"]) == 0:
                continue
            names.append(dict(ticker=str(z["ticker"]), edges=z["lag_edges"], count=z["count"].sum(0), sums=z["sums"].sum(0)))
    if not names:
        raise FileNotFoundError("no state_v3 outputs in " + group)
    return names


def ratio_one(count, sums, edges, lo, hi, f):
    mid = np.sqrt(edges[:-1] * edges[1:])
    sel = (mid >= lo) & (mid < hi) & (count[1] >= MIN_BIN_CONTROL) & (count[0] > 0)
    w = count[0, sel].astype(float)
    if w.sum() < MIN_NAME_MATCHED:
        return np.nan, 0.0
    m = sums[0, sel, f] / count[0, sel]; c = sums[1, sel, f] / count[1, sel]
    den = (w * c).sum()
    return ((w * m).sum() / den if den > 0 else np.nan), w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260913)
    names = load(args.group)
    N = len(names)
    edges = names[0]["edges"]
    result = {"group": args.group, "names": N, "method": __doc__.strip().splitlines()[0], "ranges": {}}
    pooled_count = np.sum([x["count"] for x in names], axis=0); pooled_sums = np.sum([x["sums"] for x in names], axis=0)
    for label, (lo, hi) in RANGES.items():
        for f, feat in enumerate(FEATURES):
            per = np.array([ratio_one(x["count"], x["sums"], edges, lo, hi, f)[0] for x in names])
            med = lambda idx: np.nanmedian(per[idx]) if np.isfinite(per[idx]).sum() >= 5 else np.nan
            boots = [med(rng.integers(0, N, N)) for _ in range(BOOT)]
            boots = [b for b in boots if np.isfinite(b)]
            pooled, pairs = ratio_one(pooled_count, pooled_sums, edges, lo, hi, f)
            result["ranges"]["%s/%s" % (label, feat)] = dict(
                names_eligible=int(np.isfinite(per).sum()), name_median_ratio=float(med(np.arange(N))),
                name_median_ratio_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
                names_ratio_below_1=float(np.nanmean(per < 1)) if np.isfinite(per).any() else None,
                pooled_ratio=float(pooled), pooled_matched_pairs=float(pairs))
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    for k, v in result["ranges"].items():
        print(k, v["names_eligible"], round(v["name_median_ratio"], 3), v["name_median_ratio_ci95"], round(v["pooled_ratio"], 3))


if __name__ == "__main__":
    main()
