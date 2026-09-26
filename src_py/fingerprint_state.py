#!/usr/bin/env python3
"""fingerprint-v1 E3, corrected: do identical-size recurrences occur in more similar book states?

The v2 statistic compared a same-size recurrence (i, j) with the packet nearest in time to t_j of a
similar size. That control's lag to i differs from lag(i, j) by a random jitter, and book-state
differences grow with lag, so the control is biased toward larger differences at short lags
(identified on partial 2024 exploration output; the 2021 confirmation data were not read).

Here both pairs start at the same packet i and are compared within narrow lag bins:
  matched  j = next same-side untruncated non-round packet with the identical size;
  control  k = next same-side untruncated non-round packet with a similar, different size
               (|s_k - s_i| <= max(1, 0.2 s_i)).
For each of 40 log-spaced lag bins (0.5 s to 600 s) the counts and sums of absolute differences in
pre-trade spread, log executed-side depth and signed imbalance are stored. The aggregator reweights
controls to the matched lag distribution within a range, so lag is held fixed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import fingerprint_stats as FS

LAG_EDGES = np.geomspace(0.5, 600.0, 41)
MAX_SCAN = 400


def state_features(day):
    return np.column_stack([day["spread_bps"], np.log(np.maximum(day["exec_depth"], 1)), day["imbalance"]])


def pairs_for_day(day):
    size, masks = FS.size_class_masks(day, np.array([], np.int64))
    unr = masks["u_nonround"]
    t = day["time"]; sign = day["sign"]; feat = state_features(day)
    nb = len(LAG_EDGES) - 1
    count = np.zeros((2, nb), np.int64); sums = np.zeros((2, nb, 3))
    for s in (1, -1):
        idx = np.flatnonzero(unr & (sign == s))
        idx = idx[np.argsort(t[idx], kind="stable")]
        n = len(idx)
        if n < 2:
            continue
        ts, ss = t[idx], size[idx]
        band = np.maximum(1, np.floor(FS.CONTROL_BAND * ss))
        matched = np.full(n, -1); control = np.full(n, -1)
        for off in range(1, MAX_SCAN + 1):
            if off >= n:
                break
            a = np.arange(n - off); b = a + off
            same = (matched[a] < 0) & (ss[b] == ss[a])
            matched[a[same]] = b[same]
            near = (control[a] < 0) & (ss[b] != ss[a]) & (np.abs(ss[b] - ss[a]) <= band[a])
            control[a[near]] = b[near]
            if (matched >= 0).all() and (control >= 0).all():
                break
        for which, partner in ((0, matched), (1, control)):
            ok = partner >= 0
            i = np.flatnonzero(ok); j = partner[ok]
            lag = ts[j] - ts[i]
            keep = (lag >= LAG_EDGES[0]) & (lag < LAG_EDGES[-1])
            i, j, lag = i[keep], j[keep], lag[keep]
            diff = np.abs(feat[idx[i]] - feat[idx[j]])
            fin = np.isfinite(diff).all(axis=1)
            b = np.searchsorted(LAG_EDGES, lag[fin], side="right") - 1
            np.add.at(count[which], b, 1)
            for f in range(3):
                np.add.at(sums[which, :, f], b, diff[fin, f])
    return count, sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dates = [d for line in Path(args.pairs).read_text().splitlines() if line.strip() for d in line.split()]
    days = FS.load_days(args.packets, dates)
    D = [d for d in dates if d in days]
    count = np.zeros((len(D), 2, len(LAG_EDGES) - 1), np.int64); sums = np.zeros((len(D), 2, len(LAG_EDGES) - 1, 3))
    for k, d in enumerate(D):
        count[k], sums[k] = pairs_for_day(days[d])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, ticker=np.array(args.ticker), dates=np.array(D), lag_edges=LAG_EDGES, count=count, sums=sums)
    tmp.rename(out)
    print(json.dumps({"ticker": args.ticker, "days": len(D)}))


if __name__ == "__main__":
    main()
