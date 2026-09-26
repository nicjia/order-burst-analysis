#!/usr/bin/env python3
"""Stage 2 of fingerprint-v1: same-origin size fingerprints and burst-definition validation.

LOBSTER gives no participant identity for the aggressor of an execution (verified 2026-09-13:
MPID attribution is on <0.5% of executed orders, type-5 rows carry no order id, and
same-timestamp aggressor-to-order links are ~0.2% of execution timestamps). This module uses a
weaker but testable signature of common origin instead.

If one algorithm slices a parent into identical child sizes, same-side packets with the same
*untruncated* size recur at short lags more often than chance. Untruncated means the packet did
not exhaust displayed touch depth or walk the book, so its size is the incoming order's size and
not a property of the book. Chance is the identical statistic computed ACROSS a pair of adjacent
trading days at the same clock lag (same name, side and size class). That null keeps intraday
seasonality of the size distribution but cannot contain a same-day pair from one program.

Everything is a pair count, so the aggregator can form excess = matches - pairs x null_rate at
any aggregation level. Nothing here is fitted; nothing uses future prices.
"""
import argparse
import json
from pathlib import Path

import numpy as np

RTH0 = 34200.0
SPAN = 1.0e5                     # > session length + max lag, separates composite key blocks
FINE_EDGES = np.array([0, .05, .1, .25, .5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 900, 1200, 1800.0])
BURST_EDGES = np.array([0, .1, .5, 2, 10, 30, 120, 300, 600, 1800.0])
GAPS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
RULES = ("run", "stream", "timing")
MINSIZES = (2, 3)
CLASSES = ("u_nonround", "u_round", "truncated", "u_rare", "u_visible")
BURST_CLASSES = ("u_nonround", "u_rare", "truncated", "u_visible")
RECUR_EDGES = np.array([0, 2, 10, 30, 120, 600.0])
FEATURES = ("spread_bps", "log_exec_depth", "imbalance")
HIST_STEP, HIST_MAX = 0.05, 120.0
RARE_MAX_SHARE = 0.002
DEPTH_QUANTILES = (0.25, 0.5, 0.75)  # depth-matched null: pairs must share an executed-side depth quartile
# Sensitivity only (FP_MATCH_ACTIVITY=1): matched pairs must also share a tercile of trailing-60s
# signed-packet activity (thresholds pooled over the date pair). Default output is unchanged.
MATCH_ACTIVITY = __import__("os").environ.get("FP_MATCH_ACTIVITY", "0") == "1"
ACTIVITY_QUANTILES = (1 / 3, 2 / 3)
CONTROL_BAND = 0.2               # size-similar control: |s' - s| <= max(1, 0.2 s), s' != s


def _sorted_composite(t, key):
    n = len(t)
    order = np.lexsort((np.arange(n), t, key))
    comp = key[order].astype(float) * SPAN + (t[order] - RTH0)
    return order, comp


def pair_counts(t, edges, key=None, group=None, excluded=None):
    """Pairs i<j (time order) with edges[k] <= t_j - t_i < edges[k+1].

    key: pairs must share it (e.g. size). group: pairs must share it; must be nondecreasing
    in time within each key (burst ids are). excluded: boolean, packets that may not form
    pairs (e.g. members of bursts below a minimum size).
    """
    t = np.asarray(t, float); n = len(t)
    out = np.zeros(len(edges) - 1, np.int64)
    if n < 2:
        return out
    if key is None:
        key = np.zeros(n, np.int64)
    else:  # dense ranks keep composite keys small enough for exact float comparisons
        key = np.unique(np.asarray(key), return_inverse=True)[1].astype(np.int64)
    order, comp = _sorted_composite(t, key)
    idx = np.arange(n)
    stop = None
    if group is not None:
        g = np.asarray(group, np.int64)[order].astype(float)
        gspan = float(g.max() + 2)
        gcomp = key[order].astype(float) * gspan + g
        if np.any(np.diff(gcomp) < 0):
            raise ValueError("group ids must be nondecreasing in time within key")
        stop = np.searchsorted(gcomp, gcomp, side="right")
    if excluded is not None:
        ex = np.asarray(excluded, bool)[order]
        stop = np.where(ex, idx + 1, stop if stop is not None else n)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        a = np.maximum(np.searchsorted(comp, comp + lo, side="left"), idx + 1)
        b = np.searchsorted(comp, comp + hi, side="left")
        if stop is not None:
            b = np.minimum(b, stop)
        out[k] = int(np.clip(b - a, 0, None).sum())
    return out


def cross_counts(tA, tB, edges, keyA=None, keyB=None):
    """Pairs (i in A, j in B) with edges[k] <= tB_j - tA_i < edges[k+1], sharing key."""
    tA = np.asarray(tA, float); tB = np.asarray(tB, float)
    out = np.zeros(len(edges) - 1, np.int64)
    if not len(tA) or not len(tB):
        return out
    if keyA is None or keyB is None:
        keyA = np.zeros(len(tA), np.int64); keyB = np.zeros(len(tB), np.int64)
    else:
        ranks = np.unique(np.r_[np.asarray(keyA), np.asarray(keyB)], return_inverse=True)[1]
        keyA = ranks[:len(tA)].astype(np.int64); keyB = ranks[len(tA):].astype(np.int64)
    _order, compB = _sorted_composite(tB, keyB)
    compA = keyA.astype(float) * SPAN + (tA - RTH0)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        out[k] = int((np.searchsorted(compB, compA + hi, side="left")
                      - np.searchsorted(compB, compA + lo, side="left")).sum())
    return out


def burst_ids(t, sign, gap, rule):
    """Burst id and burst size (packets) per packet, computed on the full tape.

    run: same-side runs; a sign change or an unsigned packet ends a run (project convention).
    stream: each side's own sequence; opposite-side and unsigned packets are ignored.
    timing: all packets, sign-blind.
    """
    t = np.asarray(t, float); sign = np.asarray(sign, int); n = len(t)
    ids = np.zeros(n, np.int64)
    if n == 0:
        return ids, ids.copy()
    if rule in ("run", "timing"):
        cut = np.diff(t) >= gap
        if rule == "run":
            cut |= (sign[1:] != sign[:-1]) | (sign[1:] == 0) | (sign[:-1] == 0)
        ids = np.r_[0, np.cumsum(cut)]
    elif rule == "stream":
        offset = 0
        for s in (1, -1):
            where = np.flatnonzero(sign == s)
            if len(where):
                local = np.r_[0, np.cumsum(np.diff(t[where]) >= gap)]
                ids[where] = local + offset
                offset += int(local[-1]) + 1
        zero = np.flatnonzero(sign == 0)
        ids[zero] = offset + np.arange(len(zero))
    else:
        raise ValueError("unknown rule " + rule)
    sizes = np.bincount(ids)[ids]
    return ids, sizes


def size_class_masks(day, rare_sizes):
    size = np.rint(day["volume"]).astype(np.int64)
    signed = day["sign"] != 0
    unt = day["untruncated"].astype(bool)
    roundlot = size % 100 == 0
    masks = {
        "u_nonround": unt & ~roundlot,
        "u_round": unt & roundlot,
        "truncated": signed & ~unt,
    }
    masks["u_rare"] = masks["u_nonround"] & np.isin(size, rare_sizes)
    # Sensitivity: an IOC filled only by hidden liquidity inside the spread can look untruncated while
    # its size is set by the hidden resting order. Fully visible packets exclude that case.
    hidden = day["hidden_share"] if "hidden_share" in day else np.zeros(len(size))
    masks["u_visible"] = masks["u_nonround"] & (np.asarray(hidden, float) == 0)
    return size, masks


def depth_bins(depths, thresholds):
    """Executed-side depth quartile index; thresholds are pooled over both days of a date pair.

    Untruncated sizes are censored by displayed depth, and local size distributions move with the
    book. Requiring both packets of a pair (within-day and cross-day alike) to share a depth
    quartile removes that source of spurious same-size matches without absorbing same-day repeats.
    """
    d = np.nan_to_num(np.asarray(depths, float), nan=-1.0)
    return np.searchsorted(np.unique(thresholds), d, side="right").astype(np.int64)


def composite(bins, size):
    return np.asarray(bins, np.int64) * 10**9 + np.asarray(size, np.int64)


def recurrence(day, size, mask, same_side=True):
    """Consecutive same-size recurrences and a lag-matched different-size control.

    For each untruncated packet i, j is the next packet with the same size (same side, or the
    opposite side as a non-program control). The state-similarity control is the same-subset
    packet nearest in time to t_j whose size differs from i's.
    """
    nb = len(RECUR_EDGES) - 1
    count = np.zeros((2, nb), np.int64); sums = np.zeros((2, nb, len(FEATURES)))
    hist = np.zeros(int(round(HIST_MAX / HIST_STEP)), np.int64)
    t = day["time"]; sign = day["sign"]
    raw = np.asarray(size, np.int64)
    size = np.unique(size, return_inverse=True)[1].astype(np.int64)  # dense ranks, same equality
    feat = np.column_stack([day["spread_bps"], np.log(np.maximum(day["exec_depth"], 1)),
                            day["imbalance"]])
    for s in (1, -1):
        src = np.flatnonzero(mask & (sign == s))
        dst = np.flatnonzero(mask & (sign == (s if same_side else -s)))
        if not len(src) or not len(dst):
            continue
        dst_order = np.lexsort((t[dst], size[dst])); d = dst[dst_order]
        comp_d = size[d].astype(float) * SPAN + (t[d] - RTH0)
        comp_s = size[src].astype(float) * SPAN + (t[src] - RTH0)
        pos = np.searchsorted(comp_d, comp_s, side="right")
        ok = pos < len(d)
        j = np.full(len(src), -1); j[ok] = d[pos[ok]]
        ok &= (j >= 0)
        ok[ok] &= size[j[ok]] == size[src[ok]]
        i = src[ok]; j = j[ok]
        lag = t[j] - t[i]
        keep = (lag >= 0) & (lag < RECUR_EDGES[-1]); i, j, lag = i[keep], j[keep], lag[keep]
        h = np.floor(lag / HIST_STEP).astype(int); h = h[h < len(hist)]
        np.add.at(hist, h, 1)
        # Control: the destination packet nearest in time to t_j whose size is similar to but not
        # equal to s_i (|s' - s| <= max(1, 0.2 s)), so both pairs face the same depth censoring.
        tsorted = dst[np.argsort(t[dst], kind="stable")]
        tj = np.searchsorted(t[tsorted], t[j], side="left")
        raw_i = raw[i]
        band = np.maximum(1, np.floor(CONTROL_BAND * raw_i))
        best = np.full(len(j), -1); bestgap = np.full(len(j), np.inf)
        for off in range(-25, 26):
            cand_pos = tj + off
            valid = (cand_pos >= 0) & (cand_pos < len(tsorted))
            cand = np.full(len(j), -1); cand[valid] = tsorted[cand_pos[valid]]
            valid &= (cand != j) & (cand != i)
            cr = np.where(valid, raw[np.maximum(cand, 0)], 0)
            valid &= (cr != raw_i) & (np.abs(cr - raw_i) <= band)
            gap = np.where(valid, np.abs(t[np.maximum(cand, 0)] - t[j]), np.inf)
            better = gap < bestgap
            best = np.where(better, cand, best); bestgap = np.where(better, gap, bestgap)
        b = np.searchsorted(RECUR_EDGES, lag, side="right") - 1
        for which, partner in ((0, j), (1, best)):
            good = partner >= 0
            diff = np.abs(feat[i[good]] - feat[partner[good]])
            fin = np.isfinite(diff).all(axis=1)
            bb = b[good][fin]
            np.add.at(count[which], bb, 1)
            for f in range(len(FEATURES)):
                np.add.at(sums[which, :, f], bb, diff[fin, f])
    return count, sums, hist


def size_counts(day):
    size = np.rint(day["volume"]).astype(np.int64)
    m = day["untruncated"].astype(bool) & (size % 100 != 0)
    u, c = np.unique(size[m], return_counts=True)
    return dict(zip(u.tolist(), c.tolist())), int(m.sum())


def common_sizes(counts_by_date, exclude, share=RARE_MAX_SHARE):
    """Sizes that are NOT rare among untruncated non-round packets on the name's other days.

    Returns None when no other day exists, so rarity is never defined from the evaluated days.
    """
    pooled = {}; total = 0
    for date, (counts, n) in counts_by_date.items():
        if date in exclude:
            continue
        total += n
        for s, c in counts.items():
            pooled[s] = pooled.get(s, 0) + c
    if total == 0:
        return None
    return {s for s, c in pooled.items() if c / total > share}


def load_days(packet_dir, dates):
    days = {}
    for date in dates:
        path = Path(packet_dir) / (date + ".npz")
        if not path.is_file():
            continue
        with np.load(path) as z:
            if "time" not in z.files or len(z["time"]) == 0:
                continue
            days[date] = {k: z[k] for k in z.files}
    return days


def analyze(days, date_pairs, ticker=""):
    D = [d for pair in date_pairs for d in pair if d in days]
    P = [pair for pair in date_pairs if pair[0] in days and pair[1] in days]
    nF = len(FINE_EDGES) - 1; nB = len(BURST_EDGES) - 1
    res = {
        "dates": np.array(D), "pairs": np.array(["%s_%s" % p for p in P]),
        "within_pairs": np.zeros((len(D), len(CLASSES), 2, nF), np.int64),
        "within_matches": np.zeros((len(D), len(CLASSES), 2, nF), np.int64),
        "within_pairs_dm": np.zeros((len(D), len(CLASSES), nF), np.int64),
        "within_matches_dm": np.zeros((len(D), len(CLASSES), nF), np.int64),
        "cross_pairs_dm": np.zeros((len(P), len(CLASSES), nF), np.int64),
        "cross_matches_dm": np.zeros((len(P), len(CLASSES), nF), np.int64),
        "cross_pairs": np.zeros((len(P), len(CLASSES), nF), np.int64),
        "cross_matches": np.zeros((len(P), len(CLASSES), nF), np.int64),
        "burst_pairs": np.zeros((len(D), len(RULES), len(GAPS), len(MINSIZES), len(BURST_CLASSES), nB), np.int64),
        "burst_matches": np.zeros((len(D), len(RULES), len(GAPS), len(MINSIZES), len(BURST_CLASSES), nB), np.int64),
        "burst_pairs_dm": np.zeros((len(D), len(RULES), len(GAPS), len(MINSIZES), len(BURST_CLASSES), nB), np.int64),
        "burst_matches_dm": np.zeros((len(D), len(RULES), len(GAPS), len(MINSIZES), len(BURST_CLASSES), nB), np.int64),
        "recur_count": np.zeros((len(D), 2, 2, len(RECUR_EDGES) - 1), np.int64),
        "recur_sum": np.zeros((len(D), 2, 2, len(RECUR_EDGES) - 1, len(FEATURES))),
        "recur_hist": np.zeros((len(D), 2, int(round(HIST_MAX / HIST_STEP))), np.int64),
        "n_class": np.zeros((len(D), len(CLASSES)), np.int64),
        "n_packets": np.zeros(len(D), np.int64), "n_signed": np.zeros(len(D), np.int64),
        "n_bursts3": np.zeros((len(D), len(RULES), len(GAPS)), np.int64),
        "packets_in_bursts3": np.zeros((len(D), len(RULES), len(GAPS)), np.int64),
        "rare_defined": np.zeros(len(D), np.bool_),
    }
    masks_by_date = {}; size_by_date = {}
    pair_of = {d: pair for pair in P for d in pair}
    counts_by_date = {d: size_counts(days[d]) for d in days}

    def signed_untruncated_depths(date):
        day = days[date]
        return day["exec_depth"][day["untruncated"].astype(bool) & (day["sign"] != 0)]
    thresholds = {}
    activity = {}
    for d in D:
        tt = days[d]["time"]; signed_t = tt[days[d]["sign"] != 0]
        activity[d] = (np.searchsorted(signed_t, tt, side="left")
                       - np.searchsorted(signed_t, tt - 60.0, side="left")).astype(float)
    act_thresholds = {}
    for date in D:
        pool = np.concatenate([signed_untruncated_depths(x) for x in pair_of.get(date, (date,))])
        pool = pool[np.isfinite(pool)]
        thresholds[date] = np.quantile(pool, DEPTH_QUANTILES) if len(pool) else np.array([np.inf])
        apool = np.concatenate([activity[x] for x in pair_of.get(date, (date,))])
        act_thresholds[date] = np.quantile(apool, ACTIVITY_QUANTILES) if len(apool) else np.array([np.inf])
    bins_by_date = {}
    for di, date in enumerate(D):
        day = days[date]
        common = common_sizes(counts_by_date, set(pair_of.get(date, (date,))))
        if common is None:
            rare = np.array([], np.int64)
        else:
            size_all = np.rint(day["volume"]).astype(np.int64)
            cand = np.unique(size_all[day["untruncated"].astype(bool) & (size_all % 100 != 0)])
            rare = np.array([s for s in cand if int(s) not in common], np.int64)
            res["rare_defined"][di] = True
        size, masks = size_class_masks(day, rare)
        masks_by_date[date] = masks; size_by_date[date] = size
        t = day["time"]; sign = day["sign"].astype(int)
        dbin = depth_bins(day["exec_depth"], thresholds[date])
        if MATCH_ACTIVITY:
            dbin = dbin * 10 + np.searchsorted(np.unique(act_thresholds[date]), activity[date], side="right")
        bins_by_date[date] = dbin
        dkey = composite(dbin, size)
        res["n_packets"][di] = len(t); res["n_signed"][di] = int((sign != 0).sum())
        for ci, c in enumerate(CLASSES):
            res["n_class"][di, ci] = int(masks[c].sum())
            for s in (1, -1):
                m = masks[c] & (sign == s)
                res["within_pairs"][di, ci, 0] += pair_counts(t[m], FINE_EDGES)
                res["within_matches"][di, ci, 0] += pair_counts(t[m], FINE_EDGES, key=size[m])
                o = masks[c] & (sign == -s)
                res["within_pairs"][di, ci, 1] += cross_counts(t[m], t[o], FINE_EDGES)
                res["within_matches"][di, ci, 1] += cross_counts(t[m], t[o], FINE_EDGES, size[m], size[o])
                res["within_pairs_dm"][di, ci] += pair_counts(t[m], FINE_EDGES, key=dbin[m])
                res["within_matches_dm"][di, ci] += pair_counts(t[m], FINE_EDGES, key=dkey[m])
        for ri, rule in enumerate(RULES):
            for gi, gap in enumerate(GAPS):
                ids, bsize = burst_ids(t, sign, gap, rule)
                big = bsize >= 3
                # count bursts with >=3 packets and >=1 signed packet
                res["n_bursts3"][di, ri, gi] = len(np.unique(ids[big & (sign != 0)]))
                res["packets_in_bursts3"][di, ri, gi] = int((big & (sign != 0)).sum())
                for mi, minsize in enumerate(MINSIZES):
                    excluded = bsize < minsize
                    for bi, c in enumerate(BURST_CLASSES):
                        for s in (1, -1):
                            m = masks[c] & (sign == s)
                            if m.sum() < 2:
                                continue
                            res["burst_pairs"][di, ri, gi, mi, bi] += pair_counts(
                                t[m], BURST_EDGES, group=ids[m], excluded=excluded[m])
                            res["burst_matches"][di, ri, gi, mi, bi] += pair_counts(
                                t[m], BURST_EDGES, key=size[m], group=ids[m], excluded=excluded[m])
                            res["burst_pairs_dm"][di, ri, gi, mi, bi] += pair_counts(
                                t[m], BURST_EDGES, key=dbin[m], group=ids[m], excluded=excluded[m])
                            res["burst_matches_dm"][di, ri, gi, mi, bi] += pair_counts(
                                t[m], BURST_EDGES, key=dkey[m], group=ids[m], excluded=excluded[m])
        for rel, same in ((0, True), (1, False)):
            cnt, sm, hist = recurrence(day, size, masks["u_nonround"], same_side=same)
            res["recur_count"][di, rel] = cnt; res["recur_sum"][di, rel] = sm
            res["recur_hist"][di, rel] = hist
    for pi, (a, b) in enumerate(P):
        for ci, c in enumerate(CLASSES):
            for s in (1, -1):
                ma = masks_by_date[a][c] & (days[a]["sign"] == s)
                mb = masks_by_date[b][c] & (days[b]["sign"] == s)
                ta, tb = days[a]["time"][ma], days[b]["time"][mb]
                sa, sb = size_by_date[a][ma], size_by_date[b][mb]
                res["cross_pairs"][pi, ci] += cross_counts(ta, tb, FINE_EDGES) + cross_counts(tb, ta, FINE_EDGES)
                res["cross_matches"][pi, ci] += (cross_counts(ta, tb, FINE_EDGES, sa, sb)
                                                 + cross_counts(tb, ta, FINE_EDGES, sb, sa))
                ba, bb = bins_by_date[a][ma], bins_by_date[b][mb]
                res["cross_pairs_dm"][pi, ci] += (cross_counts(ta, tb, FINE_EDGES, ba, bb)
                                                  + cross_counts(tb, ta, FINE_EDGES, bb, ba))
                res["cross_matches_dm"][pi, ci] += (cross_counts(ta, tb, FINE_EDGES, composite(ba, sa), composite(bb, sb))
                                                    + cross_counts(tb, ta, FINE_EDGES, composite(bb, sb), composite(ba, sa)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True, help="directory of <date>.npz for one ticker")
    ap.add_argument("--pairs", required=True, help="file with one 'date1 date2' pair per line")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    date_pairs = [tuple(line.split()) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    dates = [d for p in date_pairs for d in p]
    days = load_days(args.packets, dates)
    res = analyze(days, date_pairs, args.ticker)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, ticker=np.array(args.ticker), **res)
    tmp.rename(out)
    print(json.dumps({"ticker": args.ticker, "days": len(res["dates"]), "pairs": len(res["pairs"])}))


if __name__ == "__main__":
    main()
