#!/usr/bin/env python3
"""program-evidence-v1, modules A, B and I, from fingerprint-v1 cached packets (one ticker).

A  timing fingerprint: phase of same-side pair lags around whole seconds, within day and across
   adjacent days, for all, identical-size and opposite-side pairs; within-burst locked pairs for
   the 33 fingerprint-v1 definitions; absolute timestamp phase.
B  side structure: identical-size matches for same-side and opposite-side pairs out to 6.5 hours,
   depth-matched, with lag-specific cross-day nulls for adjacent and month-apart days; rare-size
   chains linked sign-blind, against chains built after permuting sizes within 30-minute windows.
I  dollar-sized children: near and far size differences against the sign of the mid change.

Everything is a count, so the aggregator forms ratios and bootstraps names. Nothing is fitted and
nothing uses information after the packets involved. See PROGRAM_EVIDENCE_DESIGN.md.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import fingerprint_stats as FS

RTH0, SPAN = FS.RTH0, FS.SPAN
# Module A
PHI_MS = np.array([-500, -250, -100, -50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 100, 250, 500.0])
K_MAX = 60
N_PHI = len(PHI_MS) - 1
PHASE_EDGES = np.unique(np.round((np.arange(1, K_MAX + 1)[:, None] + PHI_MS[None, :] / 1e3).ravel(), 9))
DELTA = 0.010
LOCK_EDGES = np.unique(np.round((np.arange(1, K_MAX + 1)[:, None]
                                 + np.array([-0.5, -DELTA, DELTA, 0.5])[None, :]).ravel(), 9))
SIZE_LAGS = np.array([0.5, K_MAX + 0.5])
# Module B
B_EDGES = np.array([0, .5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 14400, 23400.0])
B_CLASSES = ("u_nonround", "u_rare")
CHAIN_GAP, CHAIN_MIN, PERMUTE_WINDOW = 300.0, 3, 1800.0
CHAIN_N_EDGES = np.array([3, 4, 5, 10, 20, 10**9])
CHAIN_OS_EDGES = np.linspace(0, 1, 11)
CHAIN_DUR_EDGES = np.array([0, 10, 60, 300, 1800, 3600, 1e9])
# Module I
I_LAG = (0.5, 10.0)
NEAR = (1, 2, 3)
FAR = tuple(range(10, 31))
E_EDGES = np.array([0, 0.1, 0.5, np.inf])


def within_cum(comp, edges, stop=None):
    """For sorted composite times, ordered pairs i<j with comp_j - comp_i < e, for each e > 0."""
    n = len(comp)
    out = np.zeros(len(edges), np.int64)
    if n < 2:
        return out
    idx = np.arange(n)
    for k, e in enumerate(edges):
        b = np.searchsorted(comp, comp + e, side="left")
        if stop is not None:
            b = np.minimum(b, stop)
        out[k] = int(np.clip(b - idx - 1, 0, None).sum())
    return out


def cross_cum(compA, compB, edges):
    """Pairs (i in A, j in B, compB sorted) with 0 <= compB_j - compA_i < e, for each e > 0."""
    out = np.zeros(len(edges), np.int64)
    if not len(compA) or not len(compB):
        return out
    base = np.searchsorted(compB, compA, side="left")
    for k, e in enumerate(edges):
        out[k] = int((np.searchsorted(compB, compA + e, side="left") - base).sum())
    return out


def keyed(t, key=None):
    """Sorted composite (dense key rank * SPAN + time) and the sort order."""
    t = np.asarray(t, float)
    if key is None:
        order = np.argsort(t, kind="stable")
        return (t[order] - RTH0), order
    rank = np.unique(np.asarray(key), return_inverse=True)[1].astype(np.int64)
    order = np.lexsort((t, rank))
    return rank[order].astype(float) * SPAN + (t[order] - RTH0), order


def keyed_pair(tA, tB, keyA=None, keyB=None):
    tA = np.asarray(tA, float); tB = np.asarray(tB, float)
    if keyA is None:
        return tA - RTH0, np.sort(tB) - RTH0
    ranks = np.unique(np.r_[np.asarray(keyA), np.asarray(keyB)], return_inverse=True)[1].astype(np.int64)
    rA, rB = ranks[:len(tA)], ranks[len(tA):]
    compA = rA.astype(float) * SPAN + (tA - RTH0)
    compB = np.sort(rB.astype(float) * SPAN + (tB - RTH0))
    return compA, compB


def phase_within(t, key=None, edges=PHASE_EDGES):
    comp, _ = keyed(t, key)
    return np.diff(within_cum(comp, edges))


def phase_cross(tA, tB, keyA=None, keyB=None, edges=PHASE_EDGES):
    cA, cB = keyed_pair(tA, tB, keyA, keyB)
    return np.diff(cross_cum(cA, cB, edges))


def group_stop(comp_key_rank, group_sorted, excluded_sorted):
    """Index one past each packet's (key, group) block in composite order."""
    n = len(group_sorted)
    gspan = float(group_sorted.max() + 2) if n else 1.0
    gcomp = comp_key_rank * gspan + group_sorted
    if np.any(np.diff(gcomp) < 0):
        raise ValueError("group ids must be nondecreasing in time within key")
    stop = np.searchsorted(gcomp, gcomp, side="right")
    return np.where(excluded_sorted, np.arange(n) + 1, stop)


def within_group_counts(t, group, excluded, edges, key=None):
    t = np.asarray(t, float)
    if len(t) < 2:
        return np.zeros(len(edges) - 1, np.int64)
    if key is None:
        rank = np.zeros(len(t), np.int64)
    else:
        rank = np.unique(np.asarray(key), return_inverse=True)[1].astype(np.int64)
    order = np.lexsort((t, rank))
    comp = rank[order].astype(float) * SPAN + (t[order] - RTH0)
    stop = group_stop(rank[order].astype(float), np.asarray(group, np.int64)[order].astype(float),
                      np.asarray(excluded, bool)[order])
    return np.diff(within_cum(comp, edges, stop))


def chains(t, sign, size, gap=CHAIN_GAP):
    """Sign-blind chains of identical sizes: returns per chain (n, n_buy, n_sell, duration)."""
    if len(t) == 0:
        return np.zeros((0, 4))
    order = np.lexsort((t, size))
    ts, ss, gs = t[order], size[order], sign[order]
    cut = np.r_[True, (ss[1:] != ss[:-1]) | (np.diff(ts) > gap)]
    cid = np.cumsum(cut) - 1
    n = np.bincount(cid)
    nb = np.bincount(cid, weights=(gs > 0).astype(float))
    ns = np.bincount(cid, weights=(gs < 0).astype(float))
    first = np.flatnonzero(cut); last = np.r_[first[1:], len(ts)] - 1
    dur = ts[last] - ts[first]
    return np.column_stack([n, nb, ns, dur])


def chain_hist(ch):
    h = np.zeros((len(CHAIN_N_EDGES) - 1, len(CHAIN_OS_EDGES) - 1, len(CHAIN_DUR_EDGES) - 1), np.int64)
    if not len(ch):
        return h
    keep = ch[:, 0] >= CHAIN_MIN
    ch = ch[keep]
    if not len(ch):
        return h
    os_ = np.abs(ch[:, 1] - ch[:, 2]) / ch[:, 0]
    a = np.searchsorted(CHAIN_N_EDGES, ch[:, 0], side="right") - 1
    b = np.clip(np.searchsorted(CHAIN_OS_EDGES, os_, side="right") - 1, 0, len(CHAIN_OS_EDGES) - 2)
    c = np.searchsorted(CHAIN_DUR_EDGES, ch[:, 3], side="right") - 1
    np.add.at(h, (a, b, c), 1)
    return h


def permute_within_windows(t, size, seed):
    rng = np.random.default_rng(seed)
    out = size.copy()
    win = np.floor((t - RTH0) / PERMUTE_WINDOW).astype(np.int64)
    for w in np.unique(win):
        idx = np.flatnonzero(win == w)
        out[idx] = size[idx][rng.permutation(len(idx))]
    return out


def dollar_counts(t, size, mid, dbin):
    """[near/far, e-bin, dm (up, down, zero), ds (neg, pos)] for one side's u_nonround packets."""
    out = np.zeros((2, len(E_EDGES) - 1, 3, 2), np.int64)
    n = len(t)
    if n < 2:
        return out
    key = dbin.astype(np.int64) * 10**7 + size.astype(np.int64)
    uniq = np.unique(key)
    rank = np.searchsorted(uniq, key)
    order = np.lexsort((t, rank))
    comp = rank[order].astype(float) * SPAN + (t[order] - RTH0)
    for which, ds_set in ((0, NEAR), (1, FAR)):
        for mag in ds_set:
            for d in (mag, -mag):
                target = key + d
                pos = np.searchsorted(uniq, target)
                ok = pos < len(uniq)
                ok[ok] = uniq[pos[ok]] == target[ok]
                if not ok.any():
                    continue
                i_idx = np.flatnonzero(ok)
                base = pos[i_idx].astype(float) * SPAN + (t[i_idx] - RTH0)
                lo = np.searchsorted(comp, base + I_LAG[0], side="left")
                hi = np.searchsorted(comp, base + I_LAG[1], side="left")
                cnt = hi - lo
                tot = int(cnt.sum())
                if tot == 0:
                    continue
                ii = np.repeat(i_idx, cnt)
                start = np.repeat(lo, cnt)
                offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                jj = order[start + offs]
                m0, m1 = mid[ii], mid[jj]
                good = np.isfinite(m0) & np.isfinite(m1) & (m0 > 0)
                ii, jj, m0, m1 = ii[good], jj[good], m0[good], m1[good]
                dm = m1 - m0
                e = size[ii] * np.abs(dm) / m0
                eb = np.searchsorted(E_EDGES, e, side="right") - 1
                dmc = np.where(dm > 0, 0, np.where(dm < 0, 1, 2))
                dsc = 0 if d < 0 else 1
                np.add.at(out[which], (eb, dmc, np.full(len(eb), dsc)), 1)
    return out


def analyze(days, date_pairs, ticker="", modules="ABI"):
    D = [d for pair in date_pairs for d in pair if d in days]
    P = [pair for pair in date_pairs if pair[0] in days and pair[1] in days]
    # Distant pairs: first day of each complete pair with the first day of the next complete pair.
    Q = [(P[i][0], P[(i + 1) % len(P)][0]) for i in range(len(P))] if len(P) > 1 else []
    nK, nB = K_MAX, len(B_EDGES) - 1
    nR, nG = len(FS.RULES), len(FS.GAPS)
    z = lambda *s: np.zeros(s, np.int64)
    res = {
        "dates": np.array(D), "pairs": np.array(["%s_%s" % p for p in P]),
        "distant": np.array(["%s_%s" % q for q in Q]),
        # A: [day or pair, k, phi]
        "a_w_all": z(len(D), nK, N_PHI), "a_w_id": z(len(D), nK, N_PHI), "a_w_opp": z(len(D), nK, N_PHI),
        "a_w_sig": z(len(D), nK, N_PHI),
        "a_c_all": z(len(P), nK, N_PHI), "a_c_id": z(len(P), nK, N_PHI), "a_c_sig": z(len(P), nK, N_PHI),
        "a0_frac": z(len(D), 2, 1000),
        # within-burst locked counts [day, rule, gap, variant(all, id), k, (left, locked, right)]
        "a_b": z(len(D), nR, nG, 2, nK, 3),
        "a_b_dm": z(len(D), nR, nG, 2),          # (pairs, matches) over SIZE_LAGS, depth-matched, in bursts
        "a_w_dm": z(len(D), 2), "a_c_dm": z(len(P), 2),
        # B: [.., class, lag]; relation 0 same side, 1 opposite side
        "b_w_pairs_dm": z(len(D), 2, 2, nB), "b_w_match_dm": z(len(D), 2, 2, nB),
        "b_w_pairs": z(len(D), 2, 2, nB), "b_w_match": z(len(D), 2, 2, nB),
        "b_c_pairs_dm": z(len(P), 2, 2, nB), "b_c_match_dm": z(len(P), 2, 2, nB),
        "b_c_pairs": z(len(P), 2, 2, nB), "b_c_match": z(len(P), 2, 2, nB),
        "b_d_pairs_dm": z(len(Q), 2, nB), "b_d_match_dm": z(len(Q), 2, nB),   # u_nonround only
        "b_chain_real": z(len(D), len(CHAIN_N_EDGES) - 1, len(CHAIN_OS_EDGES) - 1, len(CHAIN_DUR_EDGES) - 1),
        "b_chain_perm": z(len(D), len(CHAIN_N_EDGES) - 1, len(CHAIN_OS_EDGES) - 1, len(CHAIN_DUR_EDGES) - 1),
        "b_rare_defined": np.zeros(len(D), np.bool_),
        # I: [day, side(buy, sell), near/far, e-bin, dm, ds]
        "i_counts": z(len(D), 2, 2, len(E_EDGES) - 1, 3, 2),
        "n_class": z(len(D), 3),
    }
    pair_of = {d: pair for pair in P for d in pair}
    counts_by_date = {d: FS.size_counts(days[d]) for d in days}

    def pooled_thresholds(dates):
        pool = np.concatenate([days[x]["exec_depth"][days[x]["untruncated"].astype(bool) & (days[x]["sign"] != 0)]
                               for x in dates])
        pool = pool[np.isfinite(pool)]
        return np.quantile(pool, FS.DEPTH_QUANTILES) if len(pool) else np.array([np.inf])

    prep = {}
    for di, date in enumerate(D):
        day = days[date]
        common = FS.common_sizes(counts_by_date, set(pair_of.get(date, (date,))))
        if common is None:
            rare = np.array([], np.int64)
        else:
            size_all = np.rint(day["volume"]).astype(np.int64)
            cand = np.unique(size_all[day["untruncated"].astype(bool) & (size_all % 100 != 0)])
            rare = np.array([s for s in cand if int(s) not in common], np.int64)
            res["b_rare_defined"][di] = True
        size, masks = FS.size_class_masks(day, rare)
        t = day["time"].astype(float); sign = day["sign"].astype(int)
        dbin = FS.depth_bins(day["exec_depth"], pooled_thresholds(pair_of.get(date, (date,))))
        prep[date] = dict(t=t, sign=sign, size=size, masks=masks, dbin=dbin)
        u = masks["u_nonround"]; sg = sign != 0
        res["n_class"][di] = [int(sg.sum()), int(u.sum()), int(masks["u_rare"].sum())]
        # ---- A: within-day phase counts
        for s in ((1, -1) if "A" in modules else ()):
            m = u & (sign == s); o = u & (sign == -s)
            res["a_w_all"][di] += phase_within(t[m]).reshape(nK, N_PHI)
            res["a_w_id"][di] += phase_within(t[m], size[m]).reshape(nK, N_PHI)
            res["a_w_opp"][di] += phase_cross(t[m], t[o]).reshape(nK, N_PHI)
            res["a_w_sig"][di] += phase_within(t[sg & (sign == s)]).reshape(nK, N_PHI)
            res["a_w_dm"][di, 0] += FS.pair_counts(t[m], SIZE_LAGS, key=dbin[m]).sum()
            res["a_w_dm"][di, 1] += FS.pair_counts(t[m], SIZE_LAGS, key=FS.composite(dbin[m], size[m])).sum()
        for ci, sel in enumerate((u, sg) if "A" in modules else ()):
            frac = np.floor(np.mod(t[sel], 1.0) * 1000).astype(np.int64)
            res["a0_frac"][di, ci] = np.bincount(np.clip(frac, 0, 999), minlength=1000)
        # ---- A3: within-burst locked and depth-matched counts
        for ri, rule in enumerate(FS.RULES if "A" in modules else ()):
            for gi, gap in enumerate(FS.GAPS):
                ids, bsize = FS.burst_ids(t, sign, gap, rule)
                excluded = bsize < 3
                for s in (1, -1):
                    m = u & (sign == s)
                    if m.sum() < 2:
                        continue
                    for vi, key in enumerate((None, size[m])):
                        c = within_group_counts(t[m], ids[m], excluded[m], LOCK_EDGES, key)
                        # LOCK_EDGES bins per k: [k-.5,k-d), [k-d,k+d), [k+d,k+.5); consecutive k share edges
                        res["a_b"][di, ri, gi, vi] += c.reshape(nK, 3)
                    res["a_b_dm"][di, ri, gi, 0] += FS.pair_counts(t[m], SIZE_LAGS, key=dbin[m], group=ids[m],
                                                                   excluded=excluded[m]).sum()
                    res["a_b_dm"][di, ri, gi, 1] += FS.pair_counts(t[m], SIZE_LAGS, key=FS.composite(dbin[m], size[m]),
                                                                   group=ids[m], excluded=excluded[m]).sum()
        # ---- B: within-day same/opposite identical-size matches
        for ci, cls in enumerate(B_CLASSES if "B" in modules else ()):
            mk = masks[cls]
            for s in (1, -1):
                m = mk & (sign == s); o = mk & (sign == -s)
                res["b_w_pairs_dm"][di, ci, 0] += FS.pair_counts(t[m], B_EDGES, key=dbin[m])
                res["b_w_match_dm"][di, ci, 0] += FS.pair_counts(t[m], B_EDGES, key=FS.composite(dbin[m], size[m]))
                res["b_w_pairs"][di, ci, 0] += FS.pair_counts(t[m], B_EDGES)
                res["b_w_match"][di, ci, 0] += FS.pair_counts(t[m], B_EDGES, key=size[m])
                res["b_w_pairs_dm"][di, ci, 1] += FS.cross_counts(t[m], t[o], B_EDGES, dbin[m], dbin[o])
                res["b_w_match_dm"][di, ci, 1] += FS.cross_counts(t[m], t[o], B_EDGES, FS.composite(dbin[m], size[m]),
                                                                  FS.composite(dbin[o], size[o]))
                res["b_w_pairs"][di, ci, 1] += FS.cross_counts(t[m], t[o], B_EDGES)
                res["b_w_match"][di, ci, 1] += FS.cross_counts(t[m], t[o], B_EDGES, size[m], size[o])
        # ---- B chains (rare sizes, sign-blind) and window-permuted comparison
        r = masks["u_rare"] & ("B" in modules)
        seed = int.from_bytes(hashlib.sha256(("program-evidence-v1|%s|%s" % (ticker, date)).encode()).digest()[:4], "little")
        res["b_chain_real"][di] = chain_hist(chains(t[r], sign[r], size[r]))
        res["b_chain_perm"][di] = chain_hist(chains(t[r], sign[r], permute_within_windows(t[r], size[r], seed)))
        # ---- I: dollar-sized children
        mid = day["mid"].astype(float)
        for si, s in enumerate((1, -1) if "I" in modules else ()):
            m = u & (sign == s)
            res["i_counts"][di, si] = dollar_counts(t[m], size[m], mid[m], dbin[m])

    def both_ways(ta, tb, edges, ka=None, kb=None):
        """Cross-day pairs in either temporal order (each unordered pair counted once)."""
        return FS.cross_counts(ta, tb, edges, ka, kb) + FS.cross_counts(tb, ta, edges, kb, ka)

    def side_relation_counts(a, b, cls, bins):
        """[(pairs_dm, match_dm, pairs, match)] for relation 0 (same side) and 1 (opposite side)."""
        pa, pb = prep[a], prep[b]
        out = np.zeros((2, 4, nB), np.int64)
        for s in (1, -1):
            for rel, sb_ in ((0, s), (1, -s)):
                ma = pa["masks"][cls] & (pa["sign"] == s)
                mb = pb["masks"][cls] & (pb["sign"] == sb_)
                ta, tb = pa["t"][ma], pb["t"][mb]
                sa, sz = pa["size"][ma], pb["size"][mb]
                ka, kb = bins[a][ma], bins[b][mb]
                out[rel, 0] += both_ways(ta, tb, B_EDGES, ka, kb)
                out[rel, 1] += both_ways(ta, tb, B_EDGES, FS.composite(ka, sa), FS.composite(kb, sz))
                out[rel, 2] += both_ways(ta, tb, B_EDGES)
                out[rel, 3] += both_ways(ta, tb, B_EDGES, sa, sz)
        return out

    for pi, (a, b) in enumerate(P):
        pa, pb = prep[a], prep[b]
        thr = pooled_thresholds((a, b))
        bins = {x: FS.depth_bins(days[x]["exec_depth"], thr) for x in (a, b)}
        ua, ub = pa["masks"]["u_nonround"], pb["masks"]["u_nonround"]
        for s in ((1, -1) if "A" in modules else ()):
            ma, mb = ua & (pa["sign"] == s), ub & (pb["sign"] == s)
            ga, gb = (pa["sign"] == s), (pb["sign"] == s)
            ta, tb = pa["t"], pb["t"]
            res["a_c_all"][pi] += (phase_cross(ta[ma], tb[mb]) + phase_cross(tb[mb], ta[ma])).reshape(nK, N_PHI)
            res["a_c_id"][pi] += (phase_cross(ta[ma], tb[mb], pa["size"][ma], pb["size"][mb])
                                  + phase_cross(tb[mb], ta[ma], pb["size"][mb], pa["size"][ma])).reshape(nK, N_PHI)
            res["a_c_sig"][pi] += (phase_cross(ta[ga], tb[gb]) + phase_cross(tb[gb], ta[ga])).reshape(nK, N_PHI)
            ka, kb = bins[a][ma], bins[b][mb]
            res["a_c_dm"][pi, 0] += (FS.cross_counts(ta[ma], tb[mb], SIZE_LAGS, ka, kb)
                                     + FS.cross_counts(tb[mb], ta[ma], SIZE_LAGS, kb, ka)).sum()
            res["a_c_dm"][pi, 1] += (FS.cross_counts(ta[ma], tb[mb], SIZE_LAGS, FS.composite(ka, pa["size"][ma]),
                                                     FS.composite(kb, pb["size"][mb]))
                                     + FS.cross_counts(tb[mb], ta[ma], SIZE_LAGS, FS.composite(kb, pb["size"][mb]),
                                                       FS.composite(ka, pa["size"][ma]))).sum()
        for ci, cls in enumerate(B_CLASSES if "B" in modules else ()):
            c = side_relation_counts(a, b, cls, bins)
            res["b_c_pairs_dm"][pi, ci] += c[:, 0]; res["b_c_match_dm"][pi, ci] += c[:, 1]
            res["b_c_pairs"][pi, ci] += c[:, 2]; res["b_c_match"][pi, ci] += c[:, 3]
    for qi, (a, b) in enumerate(Q if "B" in modules else ()):
        thr = pooled_thresholds((a, b))
        bins = {x: FS.depth_bins(days[x]["exec_depth"], thr) for x in (a, b)}
        c = side_relation_counts(a, b, "u_nonround", bins)
        res["b_d_pairs_dm"][qi] += c[:, 0]; res["b_d_match_dm"][qi] += c[:, 1]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modules", default="ABI", help="subset of A, B, I (default all; outputs for others stay zero)")
    args = ap.parse_args()
    date_pairs = [tuple(line.split()) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    days = FS.load_days(args.packets, [d for p in date_pairs for d in p])
    res = analyze(days, date_pairs, args.ticker, modules=args.modules)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, ticker=np.array(args.ticker), **res)
    tmp.rename(out)
    print(json.dumps({"ticker": args.ticker, "days": len(res["dates"]), "pairs": len(res["pairs"])}))


if __name__ == "__main__":
    main()
