#!/usr/bin/env python3
"""burst-defs-raw-v5 (cluster, per name-day): the three robustness checks, on a larger universe. Forecasting only.

Events (up to CAP sampled per type per stock-day; decision td = the moment the event is known):
  bursts     run0.01, run0.1, run0.5, run60 (td = t_e + gap + 0.1), early5 (run0.5 at its 5th order, +0.1 s),
             levelclear (+0.6), cancel (+0.6), hidden (+1.1)                                       -- as v4
  pair       run0.5 sequences of exactly two orders (+0.6)
  any        any marketable order, 0.1 s after it (nothing known about what follows)
  iso        isolated orders: a run0.5 of one order (+0.6, the same confirmation lag as run0.5)
  iso_large  isolated orders at or above the 80th percentile of the isolated orders seen earlier that day (+0.6)
  pseudo     a random time in [10:00, 15:30] with a sampled run0.1 burst's duration and side      -- as v4
Features (data strictly before td):
  v4 set (CTRL, BOOK, BURST) + walk_share; qofi_* normalised by the mean touch depth SO FAR (v4 used the whole day's
  mean; the v4 versions are kept as *_v4 to measure that leak); since_open is missing before 9:31.
  GENERIC (burst-blind): mid return, trade-flow imbalance, log trade count and quote OFI over the last 1, 5, 10, 30,
  60, 300 s; volume relative to today's rate so far over the last 10 / 60 / 300 s; realized variance over the last
  60 / 300 s; spread in ticks.
  DEPTH (visible book rebuilt from the messages, removals of pre-file orders ignored): depth imbalance within 1, 2, 3,
  5, 10 ticks of the best (far minus near side), log depth within 1 and 5 ticks on each side relative to the mean
  touch depth so far, at td and just before the event began.
  BHIST: run0.1 bursts decided in the last 60 / 300 / 1800 s, same and opposite side (log counts) and their signed
  volume share.
Targets (signed by the event side, bps of the decision mid): mid (r), far-side quote (f; the side the event did not
trade against), near-side quote (n), microprice (u) at +1, 5, 10, 30, 60, 300, 1800 s and the close; direction and
waiting time of the first change of the mid, the far quote and the near quote after td (within 300 s).
Bins: v4's 5-minute bins (for the panel replication on the larger universe). Burst list (_blist): every run0.1 burst of
5+ orders (decision time, side, size) for the cross-stock (market-wide vs idiosyncratic) features built locally.
Check: rebuilt level-1 depth vs the BBO path's sizes at every decision; the match shares go to stdout as JSON.
Writes OUT (events) and OUT with _bins suffix. Usage: burst_defs_raw5.py --msg FILE --ticker TK --out OUT.csv.gz
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import argparse, bisect, json, re, zlib
from pathlib import Path
import numpy as np
import pandas as pd
import p4_extract as X
import p4_packets as PK
import fingerprint_packets as FP
import fingerprint_stats as FS
import burst_defs_raw2 as B2
import burst_defs_raw3 as B3

CAP = 10
HS = (1, 5, 10, 30, 60, 300, 1800)
WIN = (1, 5, 10, 30, 60, 300)
KT = (1, 2, 3, 5, 10)
TICK = 100            # one cent in LOBSTER integer prices (dollars x 10000)
BIN = 300.0
US = 10 ** 11         # key stride for (price, side, microsecond) sort keys


class Depth:
    """Visible depth per (price, side) rebuilt from the messages: adds (1) +size; partial cancels (2), deletions (3)
    and visible executions (4) -size, except for orders never added in this file (their volume is unknown, so their
    removals are ignored rather than driving levels negative). at() returns the volume strictly before q."""

    def __init__(self, msg):
        t = msg["t"].to_numpy(float); ty = msg["ty"].to_numpy(int); oid = msg["oid"].to_numpy(np.int64)
        sz = msg["sz"].to_numpy(np.int64); px = msg["px"].to_numpy(np.int64); dr = msg["dr"].to_numpy(int)
        vis = np.isin(ty, [1, 2, 3, 4]) & np.isin(dr, [1, -1])
        added = np.unique(oid[vis & (ty == 1)])
        keep = vis & ((ty == 1) | np.isin(oid, added))
        chg = np.where(ty[keep] == 1, sz[keep], -sz[keep])
        key = px[keep] * 2 + (dr[keep] > 0)
        self.ukey, kidx = np.unique(key, return_inverse=True)
        comb = kidx.astype(np.int64) * US + np.rint(t[keep] * 1e6).astype(np.int64)
        o = np.argsort(comb, kind="stable")
        self.comb = comb[o]; self.cum = np.cumsum(chg[o])
        self.start = np.searchsorted(self.comb, np.arange(len(self.ukey), dtype=np.int64) * US, "left")
        self.base = np.where(self.start > 0, self.cum[np.maximum(self.start - 1, 0)], 0)

    def at(self, price_i, bid_side, q):
        price_i = np.asarray(price_i, np.int64); q = np.asarray(q, float)
        key = price_i * 2 + np.asarray(bid_side, np.int64)
        j = np.searchsorted(self.ukey, key)
        jj = np.minimum(j, len(self.ukey) - 1)
        found = (j < len(self.ukey)) & (self.ukey[jj] == key) & np.isfinite(q)
        qq = np.where(np.isfinite(q), q, 0.0)
        pos = np.searchsorted(self.comb, jj.astype(np.int64) * US + np.rint(qq * 1e6).astype(np.int64), "left") - 1
        ok = found & (pos >= self.start[jj])
        v = np.where(ok, self.cum[np.maximum(pos, 0)] - self.base[jj], 0)
        return np.maximum(v, 0).astype(float)

    def ladder(self, q, bid_i, ask_i, nlev=10):
        """Visible volume at best + k ticks, k = 0..nlev-1, on each side: two (n, nlev) arrays (bid, ask)."""
        k = np.arange(nlev)[None, :] * TICK
        qq = np.repeat(np.asarray(q, float)[:, None], nlev, 1)
        bp = np.asarray(bid_i, np.int64)[:, None] - k; ap = np.asarray(ask_i, np.int64)[:, None] + k
        vb = self.at(bp.ravel(), 1, qq.ravel()).reshape(bp.shape)
        va = self.at(ap.ravel(), 0, qq.ravel()).reshape(ap.shape)
        return vb, va


def next_change(v):
    """For each index i of a path v, the first index j > i with v[j] != v[i] (len(v) if none)."""
    n = len(v)
    if n == 0:
        return np.zeros(0, np.int64)
    blk = np.r_[0, np.cumsum(v[1:] != v[:-1])]
    starts = np.r_[np.flatnonzero(np.r_[True, v[1:] != v[:-1]]), n]
    return starts[blk + 1]


def run(msg_path, ticker, date, helper):
    if date in B2.EARLY:
        return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame()
    msg = X.read_messages(msg_path)
    context, _, _, _ = X.bbo_context(msg_path, msg, helper)
    mid = X.MidPath(context[0], context[1], context[2], context[3])
    bt, bm, bb, ba, qb, qa = context[0], context[1], context[2], context[3], context[4], context[5]
    cum_ofi = B2.ofi_cumulative(bt, bb, ba, qb, qa)
    dd = np.where(qb + qa > 0, (qb + qa) / 2, np.nan)
    depth_day = np.nanmean(dd) if len(qb) else np.nan                   # v4's (look-ahead) OFI scale, kept for *_v4
    rth_u = np.isfinite(dd) & (bt >= X.RTH0)
    dep_c = np.cumsum(np.where(rth_u, dd, 0.0)); dep_n = np.cumsum(rth_u)
    D = Depth(msg)
    packets = PK.fast_packets(msg, context)
    if not len(packets):
        return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame()
    P = packets.sort_values(["time", "packet_id"], kind="stable").reset_index(drop=True)
    day = FP.packet_arrays(packets)
    r = (day["time"] >= X.RTH0) & (day["time"] < X.RTH1)
    t, sign, vol = day["time"][r], day["sign"][r].astype(int), day["volume"][r]
    imb, unt, xdep = day["imbalance"][r], day["untruncated"][r], day["exec_depth"][r]
    hid = P.hidden_volume.to_numpy(float)[r] / np.maximum(P.volume.to_numpy(float)[r], 1e-9)
    walk = (P.min_price.to_numpy(float) != P.max_price.to_numpy(float))[r]
    if len(t) < 3:                                     # the minimum a run needs
        return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame()
    csv, cvol = np.cumsum(sign * vol), np.cumsum(vol)

    def cum_at(c, q):
        i = np.searchsorted(t, q, "left")
        return np.where(i > 0, c[np.maximum(i - 1, 0)], 0.0)

    def count_in(q0, q1):
        return np.searchsorted(t, q1, "left") - np.searchsorted(t, q0, "left")

    def depth_so_far(q):
        i = np.searchsorted(bt, q, "left") - 1
        n = np.where(i >= 0, dep_n[np.maximum(i, 0)], 0)
        return np.where(n > 0, dep_c[np.maximum(i, 0)] / np.maximum(n, 1), np.nan)

    def ofi(a, b, scale):
        ia = np.searchsorted(bt, a, "left") - 1; ib = np.searchsorted(bt, b, "left") - 1
        v = np.where(ib >= 0, cum_ofi[np.maximum(ib, 0)], 0.0) - np.where(ia >= 0, cum_ofi[np.maximum(ia, 0)], 0.0)
        return v / scale

    def book_at(q):
        i = np.searchsorted(bt, q, "left") - 1; ok = i >= 0; j = np.maximum(i, 0)
        return (np.where(ok, bb[j], np.nan), np.where(ok, ba[j], np.nan), np.where(ok, qb[j], np.nan), np.where(ok, qa[j], np.nan))

    mt, mty, mdr = msg["t"].to_numpy(float), msg["ty"].to_numpy(int), msg["dr"].to_numpy(int)
    mpx = msg["px"].to_numpy(float) / X.BA.SCALE; msz = msg["sz"].to_numpy(float)
    cm = np.isin(mty, [2, 3]) & (mt >= X.RTH0) & (mt < X.RTH1)
    cbid, cask, _, _ = book_at(mt[cm])
    at_touch = np.where(mdr[cm] > 0, np.isclose(mpx[cm], cbid), np.isclose(mpx[cm], cask))
    ct = mt[cm][at_touch]; cside = -mdr[cm][at_touch]; cvolc = msz[cm][at_touch]
    c_ask = np.cumsum(cside > 0); c_bid = np.cumsum(cside < 0)

    def canc(q0, q1, which):
        c = c_ask if which > 0 else c_bid
        i0 = np.searchsorted(ct, q0, "left"); i1 = np.searchsorted(ct, q1, "left")
        return np.where(i1 > 0, c[np.maximum(i1 - 1, 0)], 0) - np.where(i0 > 0, c[np.maximum(i0 - 1, 0)], 0)

    # ------------------------------------------------------------------------------------------ definitions (as v4)
    defs = {}
    for g in (0.01, 0.1, 0.5, 60.0):
        ids, _ = FS.burst_ids(t, sign, g, "run")
        b, m = B2.make_bursts(np.where(sign != 0, ids, -1), t, sign, vol)
        if b is not None:
            defs["run%g" % g] = (b, m, b["t_e"] + g + 0.1)
    if "run0.5" in defs:
        b0, m0, _ = defs["run0.5"]
        bid0, ask0, _, _ = book_at(b0["t_b"] - 0.001); bid1, ask1, _, _ = book_at(b0["t_e"] + 0.01)
        moved = np.where(b0["side"] > 0, ask1 > ask0, bid1 < bid0)
        bl, ml = B3.subset(b0, m0, np.flatnonzero(moved))
        defs["levelclear"] = (bl, ml, bl["t_e"] + 0.6)
    if len(ct) >= 5:                                   # the minimum a cancellation burst needs
        ids_c, _ = FS.burst_ids(ct, cside, 0.5, "stream")
        b, _m = B2.make_bursts(ids_c, ct, cside, cvolc, minsize=5)
        if b is not None:
            defs["cancel"] = (b, None, b["t_e"] + 0.6)
    hh = np.flatnonzero((hid >= 0.5) & (sign != 0))
    if len(hh) >= 2:                                   # the minimum a run of two needs (no whole-day condition)
        idsh = np.full(len(t), -1); ids_h, _ = FS.burst_ids(t[hh], sign[hh], 1.0, "run"); idsh[hh] = ids_h
        b, m = B2.make_bursts(idsh, t, sign, vol, minsize=2)
        if b is not None:
            defs["hidden"] = (b, m, b["t_e"] + 1.1)
    # pairs, and single orders (a run0.5 of one order = isolated)
    ids5, _ = FS.burst_ids(t, sign, 0.5, "run")
    ids5 = np.where(sign != 0, ids5, -1)
    b2, m2 = B2.make_bursts(ids5, t, sign, vol, minsize=2)
    if b2 is not None:
        bp, mp = B3.subset(b2, m2, np.flatnonzero(b2["n"] == 2))
        if bp is not None and len(bp["t_b"]):
            defs["pair"] = (bp, mp, bp["t_e"] + 0.6)
    ok5 = ids5 >= 0
    cnt = np.zeros(len(t), int)
    if ok5.any():
        _, inv, cn = np.unique(ids5[ok5], return_inverse=True, return_counts=True)
        cnt[ok5] = cn[inv]
    iso = np.flatnonzero(cnt == 1)

    m_open = mid.at(np.array([X.RTH0 + 60.0]))[0]; m_close = mid.at(np.array([X.RTH1]))[0]
    rng = np.random.default_rng(zlib.crc32(("v5|%s|%s" % (ticker, date)).encode()))
    # all run0.1 bursts, for the burst-history features
    if "run0.1" in defs:
        bh, _, bh_td = defs["run0.1"]
        o = np.argsort(bh_td); bh_t = bh_td[o]; bh_s = bh["side"][o]; bh_sv = np.cumsum(bh["side"][o] * bh["vol"][o])
        bh_cp = np.cumsum(bh_s > 0); bh_cn = np.cumsum(bh_s < 0)
    else:
        bh_t = np.zeros(0)
    if not len(bh_t):                                                       # no run0.1 bursts today: empty windows
        bh_t = np.array([np.inf]); bh_sv = bh_cp = bh_cn = np.zeros(1)

    def win_count(c, q0, q1):
        i0 = np.searchsorted(bh_t, q0, "left"); i1 = np.searchsorted(bh_t, q1, "left")
        return np.where(i1 > 0, c[np.maximum(i1 - 1, 0)], 0) - np.where(i0 > 0, c[np.maximum(i0 - 1, 0)], 0)

    # path indices for first-change targets
    nx_m, nx_b, nx_a = next_change(bm), next_change(bb), next_change(ba)
    lev_stats = []

    def features(dn, kind, side, tb_, te_, td, members):
        side = np.asarray(side, float); tb_ = np.asarray(tb_, float); te_ = np.asarray(te_, float); td = np.asarray(td, float)
        rows = []
        for j in range(len(td)):
            mem = members[j] if members is not None else np.array([], dtype=int)
            zm, tm = vol[mem], t[mem]; gaps = np.diff(tm)
            un = unt[mem]; zu = np.rint(zm[un]) if len(mem) else np.array([])
            if len(zu):
                vals, cnts = np.unique(zu, return_counts=True); k = np.argmax(cnts)
                ms, nr = cnts[k] / len(mem), float(vals[k] % 100 != 0)
            else:
                ms, nr = np.nan, np.nan
            rows.append(dict(n_used=len(mem), vol_used=zm.sum(), mode_share=ms, nonround=nr,
                             size_cv=zm.std() / zm.mean() if len(zm) and zm.mean() > 0 else np.nan,
                             size_to_depth=np.nanmean(zm / xdep[mem]) if len(mem) and np.isfinite(xdep[mem]).any() else np.nan,
                             iat_cv=gaps.std() / gaps.mean() if len(gaps) > 1 and gaps.mean() > 0 else np.nan,
                             iat_med=np.median(gaps) if len(gaps) else np.nan,
                             phase_R=float(np.abs(np.exp(2j * np.pi * (tm % 1.0)).mean())) if len(tm) else np.nan,
                             walk_share=float(walk[mem].mean()) if len(mem) else np.nan,
                             hid_share=float(hid[mem].mean()) if len(mem) else np.nan,
                             imb_first=imb[mem[0]] * side[j] * np.sign(sign[mem[0]]) if len(mem) else np.nan,
                             imb_last=imb[mem[-1]] * side[j] * np.sign(sign[mem[-1]]) if len(mem) else np.nan))
        f = pd.DataFrame(rows)
        f.insert(0, "defn", dn); f.insert(1, "kind", kind); f.insert(2, "date", date); f.insert(3, "ticker", ticker)
        f["side"], f["t_b"], f["t_e"], f["t_dec"] = side, tb_, te_, td
        f["dur"], f["tod"] = te_ - tb_, (tb_ - X.RTH0) / 23400.0
        c = {}
        m_b, m_d = mid.at(tb_), mid.at(td)
        bd, ad, qbd, qad = book_at(td); bb0, ab0, qbb, qab = book_at(tb_)
        scale = depth_so_far(td)
        with np.errstate(invalid="ignore", divide="ignore"):
            c["spread_b"] = (ab0 - bb0) / ((ab0 + bb0) / 2) * 1e4; c["spread_dec"] = (ad - bd) / ((ad + bd) / 2) * 1e4
            c["spread_change"] = c["spread_dec"] - c["spread_b"]
            c["spread_ticks"] = np.rint((ad - bd) * X.BA.SCALE / TICK)
            c["spread_ticks_b"] = np.rint((ab0 - bb0) * X.BA.SCALE / TICK)
            c["micro_gap"] = side * ((ad * qbd + bd * qad) / (qbd + qad) - m_d) / m_d * 1e4
            c["qimb_dec"] = side * (qbd - qad) / (qbd + qad)
            c["opp_consumed"] = f["vol_used"].to_numpy(float) / np.where(side > 0, qab, qbb)
            c["move_during"] = side * (m_d - m_b) / m_b * 1e4
            p60, p30 = mid.at(tb_ - 60), mid.at(np.maximum(tb_ - 1800, X.RTH0))
            c["pre60"] = side * (m_b - p60) / p60 * 1e4; c["pre30m"] = side * (m_b - p30) / p30 * 1e4
            c["since_open"] = np.where(td >= X.RTH0 + 60.0, side * (m_d - m_open) / m_open * 1e4, np.nan)
            c["qofi_pre60"] = side * ofi(tb_ - 60, tb_, scale); c["qofi_during"] = side * ofi(tb_, td, scale)
            c["qofi_pre60_v4"] = side * ofi(tb_ - 60, tb_, depth_day); c["qofi_during_v4"] = side * ofi(tb_, td, depth_day)
            c["tfi_pre60"] = side * (cum_at(csv, tb_) - cum_at(csv, tb_ - 60)) / np.maximum(cum_at(cvol, tb_) - cum_at(cvol, tb_ - 60), 1e-9)
            c["canc_opp"] = np.where(side > 0, canc(tb_, td, +1), canc(tb_, td, -1))
            c["canc_same"] = np.where(side > 0, canc(tb_, td, -1), canc(tb_, td, +1))
            # GENERIC multi-window features (burst-blind), ending at the decision
            rate = cum_at(cvol, td) / np.maximum(td - X.RTH0, 1.0)
            for W in WIN:
                mw = mid.at(td - W)
                c["g_ret%d" % W] = side * (m_d - mw) / mw * 1e4
                c["g_tfi%d" % W] = side * (cum_at(csv, td) - cum_at(csv, td - W)) / np.maximum(cum_at(cvol, td) - cum_at(cvol, td - W), 1e-9)
                c["g_nt%d" % W] = np.log1p(count_in(td - W, td))
                c["g_ofi%d" % W] = side * ofi(td - W, td, scale)
            for W in (10, 60, 300):
                c["g_volr%d" % W] = np.log((cum_at(cvol, td) - cum_at(cvol, td - W) + 1.0) / (rate * W + 1.0))
            for W in (60, 300):
                grid = td[:, None] - np.arange(W, -1, -1)[None, :]
                mg = mid.at(np.maximum(grid, X.RTH0).ravel()).reshape(grid.shape)
                c["g_lrv%d" % W] = np.log1p(np.nansum((np.diff(np.log(mg), axis=1) * 1e4) ** 2, axis=1))
            # DEPTH from the rebuilt book, at td and just before the event began
            for tag, q, bq, aq in (("dec", td, bd, ad), ("pre", tb_ - 0.001, bb0, ab0)):
                bi = np.rint(np.nan_to_num(bq) * X.BA.SCALE).astype(np.int64); ai = np.rint(np.nan_to_num(aq) * X.BA.SCALE).astype(np.int64)
                vb, va = D.ladder(q, bi, ai)
                good = np.isfinite(bq) & np.isfinite(aq)
                cb, ca = np.cumsum(vb, 1), np.cumsum(va, 1)
                far = np.where(side[:, None] > 0, cb, ca); near = np.where(side[:, None] > 0, ca, cb)
                sc = depth_so_far(q)
                for k in KT:
                    Fk, Nk = far[:, k - 1], near[:, k - 1]
                    c["d_imb%d_%s" % (k, tag)] = np.where(good & (Fk + Nk > 0), (Fk - Nk) / (Fk + Nk), np.nan)
                for k in (1, 5):
                    c["d_lfar%d_%s" % (k, tag)] = np.where(good, np.log((far[:, k - 1] + 1.0) / (sc + 1.0)), np.nan)
                    c["d_lnear%d_%s" % (k, tag)] = np.where(good, np.log((near[:, k - 1] + 1.0) / (sc + 1.0)), np.nan)
                if tag == "dec":
                    qbx = np.where(good, qbd, np.nan); qax = np.where(good, qad, np.nan)
                    lev_stats.append(np.c_[vb[:, 0], qbx, va[:, 0], qax])
            # BHIST: run0.1 bursts decided in the last W seconds
            for W in (60, 300, 1800):
                npos, nneg = win_count(bh_cp, td - W, td), win_count(bh_cn, td - W, td)
                same = np.where(side > 0, npos, nneg); opp = np.where(side > 0, nneg, npos)
                c["bh_same%d" % W] = np.log1p(same); c["bh_opp%d" % W] = np.log1p(opp)
                sv = win_count(bh_sv, td - W, td)
                c["bh_flow%d" % W] = side * sv / np.maximum(cum_at(cvol, td) - cum_at(cvol, td - W), 1.0)
            # TARGETS
            mu_d = (ad * qbd + bd * qad) / (qbd + qad)
            far_d = np.where(side > 0, bd, ad); near_d = np.where(side > 0, ad, bd)
            for h, q in [(str(h), np.minimum(td + h, X.RTH1)) for h in HS] + [("_close", np.full(len(td), X.RTH1))]:
                bh_, ah_, qbh, qah = book_at(q); mh = mid.at(q)
                c["r%s" % h] = side * (mh - m_d) / m_d * 1e4
                c["f%s" % h] = side * (np.where(side > 0, bh_, ah_) - far_d) / m_d * 1e4
                c["n%s" % h] = side * (np.where(side > 0, ah_, bh_) - near_d) / m_d * 1e4
                c["u%s" % h] = side * ((ah_ * qbh + bh_ * qah) / (qbh + qah) - mu_d) / mu_d * 1e4
            i0 = np.searchsorted(bt, td, "left") - 1; okp = i0 >= 0; i0c = np.maximum(i0, 0)
            for nm in ("m", "f", "n"):
                if nm == "m":
                    jn = nx_m[i0c]; cur = bm[i0c]; val = bm[np.minimum(jn, len(bm) - 1)]
                else:
                    use_bid = (side > 0) if nm == "f" else (side < 0)
                    jn = np.where(use_bid, nx_b[i0c], nx_a[i0c])
                    cur = np.where(use_bid, bb[i0c], ba[i0c])
                    val = np.where(use_bid, bb[np.minimum(jn, len(bb) - 1)], ba[np.minimum(jn, len(ba) - 1)])
                valid = okp & (jn < len(bt))
                wait = np.where(valid, bt[np.minimum(jn, len(bt) - 1)] - td, np.nan)
                hit = valid & (wait <= 300.0)
                c["first_%s" % nm] = np.where(hit, side * np.sign(val - cur), np.nan)
                c["wait_%s" % nm] = np.where(valid, np.minimum(wait, 300.0), np.nan)
        f = pd.concat([f, pd.DataFrame(c, index=f.index)], axis=1)
        return f

    def members_of(member, pick, t_end=None):
        if member is None:
            return None
        ok = member >= 0
        s = pd.Series(np.flatnonzero(ok)).groupby(member[ok])
        out = []
        for j, i in enumerate(pick):
            idx = s.get_group(i).to_numpy() if i in s.groups else np.array([], dtype=int)
            if t_end is not None:
                idx = idx[t[idx] <= t_end[j]]
            out.append(idx)
        return out

    frames = []
    for dn in ("run0.01", "run0.1", "run0.5", "levelclear", "cancel", "hidden", "run60", "pair"):
        if dn not in defs:
            continue
        b, member, tdec = defs[dn]
        ok = np.flatnonzero((tdec <= X.RTH1 - 30.0) & (b["t_b"] >= X.RTH0))
        if not len(ok):
            continue
        pick = np.sort(rng.choice(ok, size=min(CAP, len(ok)), replace=False))
        frames.append(features(dn, "real", b["side"][pick], b["t_b"][pick], b["t_e"][pick], tdec[pick], members_of(member, pick)))
        if dn == "run0.1":
            dur = b["t_e"][pick] - b["t_b"][pick]
            u = rng.uniform(X.RTH0 + 1800, X.RTH1 - 1800 - dur - 0.2)
            frames.append(features("run0.1", "pseudo", b["side"][pick], u, u + dur, u + dur + 0.2, None))
    if "run0.5" in defs:
        b0, m0, _ = defs["run0.5"]
        elig = np.flatnonzero(b0["n"] >= 5)
        if len(elig):
            kth, _vk = B3.kth_member(m0, t, vol, 5)
            tk = kth.reindex(elig).to_numpy(float)
            ok = np.flatnonzero(np.isfinite(tk) & (tk + 0.1 <= X.RTH1 - 30.0))
            if len(ok):
                sel = np.sort(rng.choice(ok, size=min(CAP, len(ok)), replace=False)); pick = elig[sel]
                frames.append(features("early5", "real", b0["side"][pick], b0["t_b"][pick], tk[sel], tk[sel] + 0.1,
                                       members_of(m0, pick, tk[sel])))
    # single orders
    signed = np.flatnonzero((sign != 0) & (t + 0.6 <= X.RTH1 - 30.0)); set_signed = set(signed.tolist())
    for dn, pool, lag in (("any", signed, 0.1), ("iso", np.intersect1d(iso, signed), 0.6)):
        if len(pool):
            pick = np.sort(rng.choice(pool, size=min(CAP, len(pool)), replace=False))
            frames.append(features(dn, "single", sign[pick], t[pick], t[pick], t[pick] + lag, [np.array([i]) for i in pick]))
    # large = at or above the 80th percentile of the isolated orders seen EARLIER today (at least 20 of them)
    big, seen = [], []
    for i in iso:
        if len(seen) >= 20 and vol[i] >= seen[int(0.8 * (len(seen) - 1))] and i in set_signed:
            big.append(i)
        bisect.insort(seen, vol[i])
    big = np.array(big, dtype=int)
    if len(big):
        pick = np.sort(rng.choice(big, size=min(CAP, len(big)), replace=False))
        frames.append(features("iso_large", "single", sign[pick], t[pick], t[pick], t[pick] + 0.6, [np.array([i]) for i in pick]))
    ev = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # rebuilt-book check: level-1 depth vs the BBO path's sizes at every decision
    chk = {}
    if lev_stats:
        L = np.vstack(lev_stats); gb = np.isfinite(L[:, 1]); ga = np.isfinite(L[:, 3])
        chk = dict(n=int(len(L)), bid_exact=float(np.mean(L[gb, 0] == L[gb, 1])) if gb.any() else None,
                   ask_exact=float(np.mean(L[ga, 2] == L[ga, 3])) if ga.any() else None,
                   bid_ratio_med=float(np.nanmedian(L[gb, 0] / np.maximum(L[gb, 1], 1))) if gb.any() else None,
                   ask_ratio_med=float(np.nanmedian(L[ga, 2] / np.maximum(L[ga, 3], 1))) if ga.any() else None)
    # 5-minute bins (as v4)
    edges = X.RTH0 + BIN * np.arange(79)
    bins = pd.DataFrame(dict(date=date, ticker=ticker, bin=np.arange(78), m_start=mid.at(edges[:-1]), m_end=mid.at(edges[1:]),
                             flow=cum_at(csv, edges[1:]) - cum_at(csv, edges[:-1]), volume=cum_at(cvol, edges[1:]) - cum_at(cvol, edges[:-1]),
                             qofi=ofi(edges[:-1], edges[1:], depth_so_far(edges[1:]))))
    bins["m_open"], bins["m_close"] = m_open, m_close
    lm = np.log(mid.at(X.RTH0 + np.arange(0, 23401, dtype=float)))
    bins["rv"] = np.nansum(((np.diff(lm) * 1e4) ** 2).reshape(78, 300), axis=1)
    bins["npk"] = np.bincount(np.clip(((t - X.RTH0) // BIN).astype(int), 0, 77), minlength=78)
    for dn in ("run0.01", "run0.1", "run0.5", "levelclear", "cancel", "hidden", "run60"):
        if dn not in defs:
            continue
        b, member, tdec = defs[dn]
        j = ((tdec - X.RTH0) // BIN).astype(int); live = (j >= 0) & (j < 78)
        bins["sv_" + dn] = np.bincount(j[live], weights=(b["side"] * b["vol"])[live], minlength=78)
        bins["nb_" + dn] = np.bincount(j[live], minlength=78)
    # burst list for the cross-stock features: every run0.1 burst of 5 or more orders (decision time, side, size)
    blist = pd.DataFrame()
    if "run0.1" in defs:
        b, _m, tdec = defs["run0.1"]
        k = (b["n"] >= 5) & (b["t_b"] >= X.RTH0)
        blist = pd.DataFrame(dict(t_dec=np.round(tdec[k], 3), side=b["side"][k].astype(int), n=b["n"][k].astype(int),
                                  vol=b["vol"][k]))
    return ev, bins, chk, blist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--helper", default=None)
    a = ap.parse_args()
    date = "".join(re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(a.msg).name).groups())
    ev, bins, chk, blist = run(a.msg, a.ticker, date, a.helper)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(a.out, index=False)
    bins.to_csv(a.out.replace(".csv.gz", "_bins.csv.gz"), index=False)
    blist.to_csv(a.out.replace(".csv.gz", "_blist.csv.gz"), index=False)
    print(json.dumps(dict(ticker=a.ticker, date=date, events=int(len(ev)), **chk)))


if __name__ == "__main__":
    main()
