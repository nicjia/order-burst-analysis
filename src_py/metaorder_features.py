#!/usr/bin/env python3
"""Metaorder-v1 per-burst quantities for any fingerprint-v1 burst rule (vectorized, one day).

burst_table:     burst ids, sides and size-free features (whole burst, first-three-packet prefix,
                 backward context) for bursts with >= 3 own-side packets.
within_evidence: same-side same-depth-quartile untruncated non-round pairs inside each burst
                 (lags < 300 s), identical-size repeats, and chance expectation.
link_evidence:   pairs between a burst's untruncated non-round packets and those of other bursts
                 starting within 1,800 s after it ends (same side and opposite side), identical-size
                 matches, and chance expectation from depth-quartile cross-day rates.
Nothing here uses prices after the burst except the explicit forward-window link evidence.
"""
import numpy as np

import fingerprint_stats as FS

RTH0, SPAN = FS.RTH0, FS.SPAN
WITHIN_LAG = 300.0
LINK_WINDOW = 1800.0
CTX_WINDOWS = (300.0, 1800.0)


def group_stats(key, values, n_groups):
    """Sum, count per group key (ints 0..n_groups-1) ignoring NaN values."""
    ok = np.isfinite(values)
    s = np.bincount(key[ok], weights=values[ok], minlength=n_groups)
    c = np.bincount(key[ok], minlength=n_groups).astype(float)
    return s, c


def burst_table(day, rule, gap, min_packets=3):
    """Returns member (burst index per packet, -1 if none) and a dict of per-burst arrays."""
    t = np.asarray(day["time"], float); sign = np.asarray(day["sign"], int); n = len(t)
    if n == 0:
        return np.full(0, -1), {}
    ids, _ = FS.burst_ids(t, sign, gap, rule)
    # own side: majority sign (ties: first nonzero); own packets are those with the own sign
    order = np.lexsort((t, ids))
    ids_s = ids[order]
    starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])
    uniq = ids_s[starts]
    gidx = np.searchsorted(uniq, ids)                                       # group index per packet
    ng = len(uniq)
    ssum = np.bincount(gidx, weights=sign.astype(float), minlength=ng)
    first_nz = np.full(ng, 0)
    nz = np.flatnonzero(sign[order] != 0)
    if len(nz):
        g_nz = gidx[order][nz]
        first_pos = np.full(ng, -1)
        rev = nz[::-1]; first_pos[gidx[order][rev]] = rev                  # earliest nonzero per group
        ok = first_pos >= 0
        first_nz[ok] = sign[order][first_pos[ok]]
    side_g = np.where(ssum != 0, np.sign(ssum), first_nz).astype(int)
    own = (sign == side_g[gidx]) & (sign != 0)
    n_own = np.bincount(gidx[own], minlength=ng)
    keep = (n_own >= min_packets) & (side_g != 0)
    kept = np.flatnonzero(keep)
    remap = np.full(ng, -1); remap[kept] = np.arange(len(kept))
    member = np.where(own, remap[gidx], -1)
    if not len(kept):
        return member, {}
    K = len(kept)
    ok = member >= 0
    mk = member[ok]; tt = t[ok]
    o = np.lexsort((tt, mk)); mk_s = mk[o]; t_s = tt[o]; idx_s = np.flatnonzero(ok)[o]
    first_pos = np.searchsorted(mk_s, np.arange(K), side="left"); last_pos = np.searchsorted(mk_s, np.arange(K), side="right") - 1
    first = idx_s[first_pos]; last = idx_s[last_pos]
    start, end = t[first], t[last]
    cnt = (last_pos - first_pos + 1).astype(float)
    dur = end - start
    # inter-arrival gaps within each burst's own packets
    same = mk_s[1:] == mk_s[:-1]
    gk = mk_s[1:][same]; gv = np.diff(t_s)[same]
    gsum, gcnt = group_stats(gk, gv, K)
    gsq, _ = group_stats(gk, gv ** 2, K)
    gmean = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), 0.0)
    gstd = np.sqrt(np.maximum(np.where(gcnt > 0, gsq / np.maximum(gcnt, 1) - gmean ** 2, 0.0), 0.0))
    iat_cv = np.where((gcnt > 1) & (gmean > 0), gstd / np.where(gmean > 0, gmean, 1), 0.0)
    oo = np.lexsort((gv, gk)); ks = gk[oo]; vs = gv[oo]
    lo_i = np.searchsorted(ks, np.arange(K), "left"); hi_i = np.searchsorted(ks, np.arange(K), "right")
    iat_med = np.zeros(K); has = hi_i > lo_i
    a = lo_i + (hi_i - lo_i - 1) // 2; b = lo_i + (hi_i - lo_i) // 2
    iat_med[has] = 0.5 * (vs[a[has]] + vs[b[has]])
    unt = np.asarray(day["untruncated"], bool)
    trunc = np.bincount(mk, weights=(~unt[ok]).astype(float), minlength=K) / cnt
    hid_v = np.asarray(day["hidden_share"], float)[ok]
    hs, hc = group_stats(mk, hid_v, K)
    hidden = np.where(hc > 0, hs / np.maximum(hc, 1), np.nan)
    vol = np.asarray(day["volume"], float)
    bvol = np.bincount(mk, weights=vol[ok], minlength=K)
    act = np.searchsorted(t, t, "left") - np.searchsorted(t, t - 300.0, "left")
    depth0 = np.asarray(day["exec_depth"], float)[first]
    tab = dict(first=first, last=last, side=side_g[kept], start=start, end=end, n_packets=cnt, volume=bvol, duration=dur,
               intensity=cnt / np.maximum(dur, 1e-3), iat_cv=iat_cv, iat_median=iat_med, truncated_share=trunc,
               hidden_share=hidden, spread_bps=np.asarray(day["spread_bps"], float)[first],
               log_exec_depth=np.where(np.isfinite(depth0), np.log(np.maximum(depth0, 1)), np.nan),
               imbalance=np.asarray(day["imbalance"], float)[first], tod=(start - RTH0) / 23400.0,
               trailing_activity=act[first].astype(float))
    # prefix features on the first three own packets
    p2 = idx_s[first_pos + 1]; p3 = idx_s[first_pos + 2]
    g1, g2 = t[p2] - t[first], t[p3] - t[p2]
    pm = (g1 + g2) / 2
    tab["t3"] = t[p3]
    tab["pre_duration"] = t[p3] - t[first]
    tab["pre_iat_mean"] = pm
    tab["pre_iat_cv"] = np.where(pm > 0, np.abs(g1 - g2) / 2 / np.where(pm > 0, pm, 1), 0.0)
    three = np.stack([first, p2, p3])
    tab["pre_truncated_share"] = (~unt[three]).mean(0)
    h3 = np.asarray(day["hidden_share"], float)[three]
    tab["pre_hidden_share"] = np.where(np.isfinite(h3).any(0), np.nanmean(np.where(np.isfinite(h3), h3, np.nan), 0), np.nan)
    # backward context from each burst's first packet: same/opposite-side in-burst volume shares of trailing signed volume
    signed = sign != 0
    cum_all = np.r_[0, np.cumsum(np.where(signed, vol, 0.0))]
    inb_side = np.where(member >= 0, tab["side"][np.maximum(member, 0)], 0)
    for s_label, s_val in (("buy", 1), ("sell", -1)):
        cum = np.r_[0, np.cumsum(np.where(inb_side == s_val, vol, 0.0))]
        tab["_cum_" + s_label] = cum
    for w in CTX_WINDOWS:
        lo = np.searchsorted(t, start - w, "left"); hi = np.searchsorted(t, start, "left")
        allv = cum_all[hi] - cum_all[lo]
        buy = tab["_cum_buy"][hi] - tab["_cum_buy"][lo]; sell = tab["_cum_sell"][hi] - tab["_cum_sell"][lo]
        same_v = np.where(tab["side"] > 0, buy, sell); opp_v = np.where(tab["side"] > 0, sell, buy)
        tab["ctx_same_share_%d" % w] = np.where(allv > 0, same_v / np.maximum(allv, 1e-9), 0.0)
        tab["ctx_opp_share_%d" % w] = np.where(allv > 0, opp_v / np.maximum(allv, 1e-9), 0.0)
    del tab["_cum_buy"], tab["_cum_sell"]
    # time since the previous same-side burst ended, and its size
    prev_gap = np.full(K, 23400.0); prev_n = np.zeros(K)
    for s_val in (1, -1):
        sel = np.flatnonzero(tab["side"] == s_val)
        if len(sel) < 2:
            continue
        srt = sel[np.argsort(start[sel], kind="stable")]
        e_sorted = np.maximum.accumulate(end[srt])
        prev_gap[srt[1:]] = np.clip(start[srt[1:]] - e_sorted[:-1], 0, 23400.0)
        prev_n[srt[1:]] = cnt[srt[:-1]]
    tab["log_gap_prev_same"] = np.log1p(prev_gap)
    tab["prev_same_log_packets"] = np.log1p(prev_n)
    return member, tab


def depth_rates(days_pair, thresholds):
    """Cross-day identical-size match rates over lags [0, 1800) by (relation, side of first packet, depth bin)."""
    a, b = days_pair
    out = {}
    edge = np.array([0.0, LINK_WINDOW])
    prep = {}
    for x in (a, b):
        size, masks = FS.size_class_masks(x, np.array([], np.int64))
        prep[id(x)] = (x["time"].astype(float), x["sign"].astype(int), size, masks["u_nonround"],
                       FS.depth_bins(x["exec_depth"], thresholds))
    ta, sa, za, ua, da = prep[id(a)]; tb, sb, zb, ub, db = prep[id(b)]
    n_bins = len(np.unique(thresholds)) + 1
    for rel in (0, 1):
        for s in (1, -1):
            s2 = s if rel == 0 else -s
            for k in range(n_bins):
                ma = ua & (sa == s) & (da == k); mb = ub & (sb == s2) & (db == k)
                cp = FS.cross_counts(ta[ma], tb[mb], edge).sum() + FS.cross_counts(tb[mb], ta[ma], edge).sum()
                cm = (FS.cross_counts(ta[ma], tb[mb], edge, za[ma], zb[mb]).sum()
                      + FS.cross_counts(tb[mb], ta[ma], edge, zb[mb], za[ma]).sum())
                out[(rel, s, k)] = cm / cp if cp else np.nan
    return out


def within_evidence(day, member, tab, dbin, rates):
    """Per burst: (pairs, repeats, expected) among own u_nonround packets in the same depth bin, lag < 300 s."""
    K = len(tab["side"]) if tab else 0
    pairs = np.zeros(K); reps = np.zeros(K); exp = np.zeros(K)
    if K == 0:
        return pairs, reps, exp
    t = day["time"].astype(float)
    size, masks = FS.size_class_masks(day, np.array([], np.int64))
    sel = np.flatnonzero(masks["u_nonround"] & (member >= 0))
    if len(sel) < 2:
        return pairs, reps, exp
    key = member[sel] * 10 + dbin[sel]
    o = np.lexsort((t[sel], key)); s_idx = sel[o]; k_s = key[o]; t_s = t[s_idx]; z_s = size[s_idx]
    d = 1
    while True:
        if d >= len(s_idx):
            break
        same = (k_s[d:] == k_s[:-d]) & (t_s[d:] - t_s[:-d] < WITHIN_LAG)
        if not same.any():
            if not (k_s[d:] == k_s[:-d]).any():
                break
            d += 1
            continue
        bk = member[s_idx[:-d][same]]; db = dbin[s_idx[:-d][same]]
        np.add.at(pairs, bk, 1.0)
        np.add.at(reps, bk, (z_s[:-d][same] == z_s[d:][same]).astype(float))
        side = tab["side"][bk]
        r = np.array([rates.get((0, int(s), int(q)), np.nan) for s, q in zip(side, db)])
        np.add.at(exp, bk, np.nan_to_num(r, nan=np.nan))
        d += 1
    return pairs, reps, exp


def link_evidence(day, member, tab, dbin, rates):
    """Per burst and relation: pairs, identical-size matches and expectation with packets of other bursts
    in (end, end + 1800 s]. Returns arrays [K, 2 relations, 3]."""
    K = len(tab["side"]) if tab else 0
    out = np.zeros((K, 2, 3))
    if K == 0:
        return out
    t = day["time"].astype(float); sign = day["sign"].astype(int)
    size, masks = FS.size_class_masks(day, np.array([], np.int64))
    u = masks["u_nonround"] & (member >= 0)
    src = np.flatnonzero(u)
    n_bins = int(dbin.max()) + 1 if len(dbin) else 1
    end = tab["end"]
    for rel in (0, 1):
        for s in (1, -1):
            s2 = s if rel == 0 else -s
            a = src[sign[src] == s]
            b = src[sign[src] == s2]
            if not len(a) or not len(b):
                continue
            # pair counts by depth bin: packets of b in window (end_k, end_k + W]
            tb = t[b]; kb = dbin[b]
            bk_a = member[a]; t_lo = end[bk_a]; t_hi = t_lo + LINK_WINDOW
            for q in range(n_bins):
                mq = kb == q
                if not mq.any():
                    continue
                tq = np.sort(tb[mq])
                aq = dbin[a] == q
                if not aq.any():
                    continue
                cnt = np.searchsorted(tq, t_hi[aq], "right") - np.searchsorted(tq, t_lo[aq], "right")
                np.add.at(out[:, rel, 0], bk_a[aq], cnt.astype(float))
                r = rates.get((rel, s, q), np.nan)
                np.add.at(out[:, rel, 2], bk_a[aq], cnt * (r if np.isfinite(r) else np.nan))
            # identical-size matches: composite key (bin, size), time window
            kb2 = dbin[b].astype(np.int64) * 10**7 + size[b]
            ka2 = dbin[a].astype(np.int64) * 10**7 + size[a]
            uniq = np.unique(kb2)
            rb = np.searchsorted(uniq, kb2)
            comp = np.sort(rb.astype(float) * SPAN + (tb - RTH0))
            pos = np.searchsorted(uniq, ka2)
            okk = pos < len(uniq)
            okk[okk] = uniq[pos[okk]] == ka2[okk]
            base = pos.astype(float) * SPAN
            m_cnt = np.zeros(len(a))
            m_cnt[okk] = (np.searchsorted(comp, base[okk] + (t_hi[okk] - RTH0), "right")
                          - np.searchsorted(comp, base[okk] + (t_lo[okk] - RTH0), "right"))
            np.add.at(out[:, rel, 1], bk_a, m_cnt)
    # packets of the same burst cannot fall in (end, end + W] for the same side; opposite-side packets of other
    # bursts are by construction in other bursts.
    return out


# ----------------------------------------------------------------------------- model features
WHOLE = ["log_packets", "log_duration", "log_intensity", "iat_cv", "log_iat_median", "truncated_share", "hidden_share",
         "log_spread", "log_exec_depth", "imbalance", "tod", "tod2", "log_activity"]
CONTEXT = ["ctx_same_share_300", "ctx_same_share_1800", "ctx_opp_share_300", "ctx_opp_share_1800",
           "log_gap_prev_same", "prev_same_log_packets"]
PREFIX = ["log_pre_duration", "log_pre_iat_mean", "pre_iat_cv", "pre_truncated_share", "pre_hidden_share",
          "log_spread", "log_exec_depth", "imbalance", "tod", "tod2", "log_activity"]
MODEL_FEATURES = {"program": WHOLE, "link": WHOLE + CONTEXT, "realtime": PREFIX + CONTEXT}


def derived(frame):
    """Adds transformed columns to a mapping of burst columns (DataFrame or dict of arrays)."""
    g = frame
    out = {}
    out["log_packets"] = np.log1p(np.asarray(g["n_packets"], float))
    out["log_duration"] = np.log1p(np.asarray(g["duration"], float))
    out["log_intensity"] = np.log(np.clip(np.asarray(g["intensity"], float), 1e-6, None))
    out["iat_cv"] = np.asarray(g["iat_cv"], float)
    out["log_iat_median"] = np.log1p(np.asarray(g["iat_median"], float))
    out["truncated_share"] = np.asarray(g["truncated_share"], float)
    out["hidden_share"] = np.nan_to_num(np.asarray(g["hidden_share"], float), nan=0.0)
    out["log_spread"] = np.log(np.clip(np.asarray(g["spread_bps"], float), 1e-3, None))
    out["log_exec_depth"] = np.asarray(g["log_exec_depth"], float)
    out["imbalance"] = np.asarray(g["imbalance"], float)
    out["tod"] = np.asarray(g["tod"], float); out["tod2"] = out["tod"] ** 2
    out["log_activity"] = np.log1p(np.asarray(g["trailing_activity"], float))
    for c in CONTEXT:
        out[c] = np.asarray(g[c], float)
    out["log_pre_duration"] = np.log1p(np.asarray(g["pre_duration"], float))
    out["log_pre_iat_mean"] = np.log1p(np.asarray(g["pre_iat_mean"], float))
    out["pre_iat_cv"] = np.asarray(g["pre_iat_cv"], float)
    out["pre_truncated_share"] = np.asarray(g["pre_truncated_share"], float)
    out["pre_hidden_share"] = np.nan_to_num(np.asarray(g["pre_hidden_share"], float), nan=0.0)
    return out


def score(model, frame):
    d = derived(frame)
    X = np.column_stack([d[f] for f in model["features"]])
    Z = np.column_stack([np.ones(len(X)), (X - np.asarray(model["mu"])) / np.asarray(model["sd"])])
    return Z @ np.asarray(model["beta"])
