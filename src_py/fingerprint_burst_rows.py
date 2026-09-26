#!/usr/bin/env python3
"""Stage 2b of fingerprint-v1: one row per burst, for a program-likeness score.

For a fixed burst definition, each burst with >= 3 packets gets:
  - fingerprint evidence: same-side untruncated non-round packet pairs inside the burst (lag < 300s)
    whose packets share an executed-side depth quartile, how many share an identical size, and the
    chance expectation from the depth-quartile-specific cross-day rate (primary null of
    fingerprint-v1). Depth-specific rates matter here: a single burst in a thin book has a much
    higher chance match rate than the name-day average;
  - features that do NOT use child sizes (so a score built from them cannot restate the evidence):
    timing regularity, intensity, duration, packet count, truncation (book-consumption) share,
    hidden share, pre-burst spread / depth / signed imbalance at the first packet, time of day,
    trailing 300s activity, and opposite-side interleaving.
Size descriptors are written with a `size_` prefix for description only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import fingerprint_stats as FS

MAX_LAG = 300.0


def pairs_in_group(t, size, lag=MAX_LAG):
    """(pairs, same-size pairs) among one burst side's packets with lags < lag (O(n log n))."""
    if len(t) < 2:
        return 0, 0
    edges = np.array([0.0, lag])
    return int(FS.pair_counts(t, edges).sum()), int(FS.pair_counts(t, edges, key=size).sum())


def burst_rows(day, date, ticker, rule, gap, dbin, q_bin, common_sizes):
    t = day["time"]; sign = day["sign"].astype(int)
    size = np.rint(day["volume"]).astype(np.int64)
    unr = FS.size_class_masks(day, np.array([], np.int64))[1]["u_nonround"]
    ids, _bsize = FS.burst_ids(t, sign, gap, rule)
    common_arr = np.fromiter(common_sizes, np.int64) if common_sizes else None
    act = np.searchsorted(t, t, side="left") - np.searchsorted(t, t - 300.0, side="left")
    rows = []
    order = np.argsort(ids, kind="stable")  # stream ids are not contiguous in time
    bounds = np.flatnonzero(np.r_[True, ids[order][1:] != ids[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if len(idx) < 3:
            continue
        idx = idx[np.argsort(t[idx], kind="stable")]
        s = sign[idx]
        if not (s != 0).any():
            continue
        side = int(np.sign(s.sum())) if s.sum() != 0 else int(s[s != 0][0])
        own = idx[s == side]
        if len(own) < 3:
            continue
        u = own[unr[own]]
        pairs = rep = 0; expected = 0.0
        for k in np.unique(dbin[u]) if len(u) else []:
            uk = u[dbin[u] == k]
            pk, rk = pairs_in_group(t[uk], size[uk])
            pairs += pk; rep += rk
            qk = q_bin.get((side, int(k)), np.nan)
            expected += pk * qk if np.isfinite(qk) else np.nan
        tt = t[own]; gaps = np.diff(tt); first = own[0]
        dur = float(tt[-1] - tt[0])
        depth0 = day["exec_depth"][first]
        rows.append(dict(
            ticker=ticker, date=date, rule=rule, gap_s=gap, side=side, start=float(tt[0]), end=float(tt[-1]),
            n_packets=len(own), n_opposite=int((s == -side).sum()), n_unsigned=int((s == 0).sum()),
            n_unr=len(u), dm_pairs=pairs, dm_repeats=rep, dm_expected=expected,
            duration=dur, intensity=len(own) / max(dur, 1e-3),
            iat_cv=float(gaps.std() / gaps.mean()) if len(gaps) > 1 and gaps.mean() > 0 else 0.0,
            iat_median=float(np.median(gaps)) if len(gaps) else 0.0,
            truncated_share=float((~day["untruncated"][own].astype(bool)).mean()),
            hidden_share=float(np.nanmean(day["hidden_share"][own])),
            spread_bps=float(day["spread_bps"][first]),
            log_exec_depth=float(np.log(max(depth0, 1))) if np.isfinite(depth0) else np.nan,
            imbalance=float(day["imbalance"][first]),
            tod=float((tt[0] - FS.RTH0) / 23400.0),
            trailing_activity=int(act[first]),
            size_roundlot_share=float((size[own] % 100 == 0).mean()),
            size_median=float(np.median(size[own])),
            size_cv=float(size[own].std() / size[own].mean()) if size[own].mean() > 0 else 0.0,
            size_common_share=float(np.isin(size[own], common_arr).mean()) if common_arr is not None else np.nan,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--rule", required=True, choices=FS.RULES)
    ap.add_argument("--gap", required=True, type=float)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    date_pairs = [tuple(x.split()) for x in Path(args.pairs).read_text().splitlines() if x.strip()]
    days = FS.load_days(args.packets, [d for p in date_pairs for d in p])
    counts = {d: FS.size_counts(days[d]) for d in days}
    rows = []
    for a, b in date_pairs:
        if a not in days or b not in days:
            continue
        pool = np.concatenate([days[x]["exec_depth"][days[x]["untruncated"].astype(bool) & (days[x]["sign"] != 0)]
                               for x in (a, b)])
        pool = pool[np.isfinite(pool)]
        thr = np.quantile(pool, FS.DEPTH_QUANTILES) if len(pool) else np.array([np.inf])
        bins = {x: FS.depth_bins(days[x]["exec_depth"], thr) for x in (a, b)}
        masks = {x: FS.size_class_masks(days[x], np.array([], np.int64))[1]["u_nonround"] for x in (a, b)}
        sizes = {x: np.rint(days[x]["volume"]).astype(np.int64) for x in (a, b)}
        edge = np.array([0.0, 1800.0])
        q_bin = {}
        for side in (1, -1):
            for k in range(len(np.unique(thr)) + 1):
                sel = {x: masks[x] & (days[x]["sign"] == side) & (bins[x] == k) for x in (a, b)}
                ta, tb = days[a]["time"][sel[a]], days[b]["time"][sel[b]]
                sa, sb = sizes[a][sel[a]], sizes[b][sel[b]]
                cp = FS.cross_counts(ta, tb, edge).sum() + FS.cross_counts(tb, ta, edge).sum()
                cm = FS.cross_counts(ta, tb, edge, sa, sb).sum() + FS.cross_counts(tb, ta, edge, sb, sa).sum()
                q_bin[(side, k)] = cm / cp if cp else np.nan
        common = FS.common_sizes(counts, {a, b})
        for d in (a, b):
            rows += burst_rows(days[d], d, args.ticker, args.rule, args.gap, bins[d], q_bin, common)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(len(rows))


if __name__ == "__main__":
    main()
