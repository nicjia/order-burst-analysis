#!/usr/bin/env python3
"""burst-defs-raw-v3 (cluster, per name-day): ~20 more definitions from IDEAS_BACKLOG.md in one download.

Definitions (decision time in brackets; every feature is measured before it):
  run0.01 / run0.025 / run0.05 / run0.1 / run0.5   same-side runs                          [t_e + G + 0.1 s]
  adapt2 / adapt5     runs with G = c x the stock's median inter-packet time in 9:30-10:00, bursts after 10:00 only
  tol0.5              side streams (0.5 s) with at most one opposite-side packet inside   [t_e + 0.6 s]
  dom0.5              side streams (0.5 s) with >= 75% same-side volume inside            [t_e + 0.6 s]
  hawkesfix           the original Hawkes detector, decided when its cluster is CONFIRMED  [t_e + ln(lam/0.3) + 0.1 s]
  hawkesside          Hawkes intensity run separately on each side (directional by construction) [same rule]
  timer               side streams (5 s), >= 4 members, inter-arrival CV < 0.2            [t_e + 5.1 s]
  phase               side streams (5 s), >= 4 members, > 3 s long, sub-second phase R > 0.8 [t_e + 5.1 s]
  twap                clip chains (30 s), >= 4 members, inter-arrival CV < 0.25            [t_e + 30.1 s]
  sweep               packets that walked >= 2 price levels, same-side sweeps within 1 s grouped [t_e + 1.1 s]
  levelclear          run0.5 bursts during which the opposite best price moved against them [t_e + 0.6 s]
  absorbed            run0.5 bursts with no mid change over the burst                     [t_e + 0.6 s]
  cancel              touch-level cancellations on one side, >= 5 within 0.5 s gaps; side = -(cancelled side),
                      so pulled asks count as a buy signal                                 [t_e + 0.6 s]
  hidden              hidden-heavy signed packets (>= 50% hidden volume), same side within 1 s, >= 2 [t_e + 1.1 s]
  early3 / early5     run0.5 bursts acted on at their 3rd / 5th member (decision DURING the burst)
Features: v2 set plus microprice gap, share of the opposite touch consumed, spread change, touch cancellations on
the opposite and the same side during the burst. Targets: signed mid move from decision to +10 s, +60 s, +300 s,
+1800 s and the close; latency variants (start 1 s and 10 s later); continuation (another same-side event of the
same definition starts within 60 s; opposite-side likewise); for early3/5 whether the run grew past k and by how
much volume. Up to CAP events per definition per name-day, sampled at random (seeded).
Usage: burst_defs_raw3.py --msg FILE --ticker TK --out OUT.csv.gz [--helper p4_bbo]
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import argparse, re, zlib
from pathlib import Path
import numpy as np
import pandas as pd
import p4_extract as X
import p4_packets as PK
import fingerprint_packets as FP
import fingerprint_stats as FS
import burst_defs_raw2 as B2

CAP = 8
BETA, TRIG = 1.0, 0.3


def hawkes_with_lambda(t):
    """Cluster ids and the intensity just after each event (original rule: beta 1, trigger 0.3)."""
    n = len(t); ids = np.zeros(n, np.int64); lam_after = np.ones(n)
    if n == 0:
        return ids, lam_after
    lam, k = 1.0, 0
    for i in range(1, n):
        dec = lam * np.exp(-BETA * (t[i] - t[i - 1]))
        if dec < TRIG:
            k += 1; lam = 1.0
        else:
            lam = dec + 1.0
        ids[i] = k; lam_after[i] = lam
    return ids, lam_after


def confirm_wait(lam_end):
    return np.log(np.maximum(lam_end, TRIG) / TRIG) / BETA + 0.1


def member_stats(member, t, idx):
    """per-burst arrays over members (vectorised): inter-arrival CV and sub-second phase concentration R."""
    ok = member >= 0
    df = pd.DataFrame(dict(b=member[ok], t=t[ok]))
    df["gap"] = df.groupby("b").t.diff()
    g = df.groupby("b").gap.agg(["mean", "std", "count"])
    cv_all = (g["std"] / g["mean"]).where(g["count"] >= 2)
    ph = np.exp(2j * np.pi * (df.t.to_numpy() % 1.0))
    df["c"], df["s"] = ph.real, ph.imag
    pr = df.groupby("b")[["c", "s"]].mean()
    R_all = np.sqrt(pr.c ** 2 + pr.s ** 2)
    return cv_all.reindex(idx).to_numpy(float), R_all.reindex(idx).to_numpy(float)


def first_last_member(member):
    ok = np.flatnonzero(member >= 0)
    s_ = pd.Series(ok).groupby(member[ok])
    return s_.min(), s_.max()


def kth_member(member, t, vol, k):
    ok = member >= 0
    df = pd.DataFrame(dict(b=member[ok], t=t[ok], v=vol[ok]))
    df["r"] = df.groupby("b").cumcount()
    kth = df[df.r == k - 1].set_index("b").t
    vk = df[df.r < k].groupby("b").v.sum()
    return kth, vk


def subset(b, member, keep_idx):
    rm = np.full(len(b["t_b"]), -1); rm[keep_idx] = np.arange(len(keep_idx))
    nb = {k: v[keep_idx] for k, v in b.items()}
    nm = np.where(member >= 0, rm[np.maximum(member, 0)], -1)
    return nb, nm


def run(msg_path, ticker, date, helper):
    if date in B2.EARLY:
        return pd.DataFrame()
    msg = X.read_messages(msg_path)
    context, bid_i, ask_i, _ = X.bbo_context(msg_path, msg, helper)
    mid = X.MidPath(context[0], context[1], context[2], context[3])
    bt, bb, ba, qb, qa = context[0], context[2], context[3], context[4], context[5]
    cum_ofi = B2.ofi_cumulative(bt, bb, ba, qb, qa)
    depth = np.nanmean(np.where(qb + qa > 0, (qb + qa) / 2, np.nan)) if len(qb) else np.nan
    packets = PK.fast_packets(msg, context)
    if not len(packets):
        return pd.DataFrame()
    P = packets.sort_values(["time", "packet_id"], kind="stable").reset_index(drop=True)
    day = FP.packet_arrays(packets)
    r = (day["time"] >= X.RTH0) & (day["time"] < X.RTH1)
    t, sign, vol = day["time"][r], day["sign"][r].astype(int), day["volume"][r]
    imb, unt, xdep = day["imbalance"][r], day["untruncated"][r], day["exec_depth"][r]
    hid = (P.hidden_volume.to_numpy(float)[r] / np.maximum(P.volume.to_numpy(float)[r], 1e-9))
    multi = (P.max_price.to_numpy(float)[r] > P.min_price.to_numpy(float)[r])
    if len(t) < 30:
        return pd.DataFrame()
    csv, cvol = np.cumsum(sign * vol), np.cumsum(vol)
    cbuy, csell = np.cumsum(sign > 0), np.cumsum(sign < 0)
    cbv, csv_ = np.cumsum(np.where(sign > 0, vol, 0.0)), np.cumsum(np.where(sign < 0, vol, 0.0))

    def cum_at(c, q, side="left"):
        i = np.searchsorted(t, q, side)
        return np.where(i > 0, c[np.maximum(i - 1, 0)], 0.0)

    def wsum(c, a, b):
        return cum_at(c, b) - cum_at(c, a)

    def ofi(a, b):
        ia = np.searchsorted(bt, a, "left") - 1; ib = np.searchsorted(bt, b, "left") - 1
        v = np.where(ib >= 0, cum_ofi[np.maximum(ib, 0)], 0.0) - np.where(ia >= 0, cum_ofi[np.maximum(ia, 0)], 0.0)
        return v / depth if depth and np.isfinite(depth) else np.full(np.shape(a), np.nan)

    def book_at(q):
        i = np.searchsorted(bt, q, "left") - 1; ok = i >= 0; j = np.maximum(i, 0)
        return (np.where(ok, bb[j], np.nan), np.where(ok, ba[j], np.nan), np.where(ok, qb[j], np.nan), np.where(ok, qa[j], np.nan))

    # touch-level cancellations (full deletions and partial cancels at the best price on their side)
    mt, mty, mdr = msg["t"].to_numpy(float), msg["ty"].to_numpy(int), msg["dr"].to_numpy(int)
    mpx = msg["px"].to_numpy(float) / X.BA.SCALE; msz = msg["sz"].to_numpy(float)
    cm = np.isin(mty, [2, 3]) & (mt >= X.RTH0) & (mt < X.RTH1)
    cbid, cask, _, _ = book_at(mt[cm])
    at_touch = np.where(mdr[cm] > 0, np.isclose(mpx[cm], cbid), np.isclose(mpx[cm], cask))
    ct = mt[cm][at_touch]; cside = -mdr[cm][at_touch]            # pulled asks (dr -1) -> +1 (bullish)
    cvolc = msz[cm][at_touch]
    c_ask = np.cumsum(cside > 0); c_bid = np.cumsum(cside < 0)

    def canc(q0, q1, which):
        c = c_ask if which > 0 else c_bid
        i0 = np.searchsorted(ct, q0, "left"); i1 = np.searchsorted(ct, q1, "left")
        return np.where(i1 > 0, c[np.maximum(i1 - 1, 0)], 0) - np.where(i0 > 0, c[np.maximum(i0 - 1, 0)], 0)

    defs = {}   # name -> (bursts dict, member array or None, t_dec array)
    for g in (0.01, 0.025, 0.05, 0.1, 0.5):
        ids, _ = FS.burst_ids(t, sign, g, "run")
        b, m = B2.make_bursts(np.where(sign != 0, ids, -1), t, sign, vol)
        if b is not None:
            defs["run%g" % g] = (b, m, b["t_e"] + g + 0.1)
    early = t < X.RTH0 + 1800
    med_iat = np.median(np.diff(t[early])) if early.sum() > 20 else np.nan
    for c in (2, 5):
        if np.isfinite(med_iat):
            g = float(np.clip(c * med_iat, 0.01, 60.0))
            ids, _ = FS.burst_ids(t, sign, g, "run")
            b, m = B2.make_bursts(np.where(sign != 0, ids, -1), t, sign, vol)
            if b is not None:
                k = np.flatnonzero(b["t_b"] >= X.RTH0 + 1800)
                b, m = subset(b, m, k)
                defs["adapt%d" % c] = (b, m, b["t_e"] + g + 0.1)
    ids, _ = FS.burst_ids(t, sign, 0.5, "stream")
    sb, sm = B2.make_bursts(np.where(sign != 0, ids, -1), t, sign, vol)
    if sb is not None:
        opp_n = np.where(sb["side"] > 0, cum_at(csell, sb["t_e"], "right") - cum_at(csell, sb["t_b"]),
                         cum_at(cbuy, sb["t_e"], "right") - cum_at(cbuy, sb["t_b"]))
        same_v = np.where(sb["side"] > 0, cum_at(cbv, sb["t_e"], "right") - cum_at(cbv, sb["t_b"]),
                          cum_at(csv_, sb["t_e"], "right") - cum_at(csv_, sb["t_b"]))
        all_v = cum_at(cvol, sb["t_e"], "right") - cum_at(cvol, sb["t_b"])
        for name, keep in (("tol0.5", opp_n <= 1), ("dom0.5", same_v / np.maximum(all_v, 1e-9) >= 0.75)):
            b, m = subset(sb, sm, np.flatnonzero(keep))
            defs[name] = (b, m, b["t_e"] + 0.6)
    # Hawkes, confirmed
    sg = np.flatnonzero(sign != 0)
    if len(sg) > 20:
        cid, lam = hawkes_with_lambda(t[sg])
        nb_ = int(cid[-1]) + 1
        buy = sign[sg] > 0
        nbuy = np.bincount(cid, weights=buy.astype(float), minlength=nb_); ntot = np.bincount(cid, minlength=nb_).astype(float)
        vb = np.bincount(cid, weights=np.where(buy, vol[sg], 0.0), minlength=nb_); vs = np.bincount(cid, weights=np.where(buy, 0.0, vol[sg]), minlength=nb_)
        d = np.zeros(nb_)
        d[(nbuy / ntot >= 0.763) & (vb > 0) & (vs <= 0.28 * vb)] = 1
        d[(1 - nbuy / ntot >= 0.763) & (vs > 0) & (vb <= 0.28 * vs)] = -1
        keepc = (d != 0) & (vb + vs >= 0.00197 * vol.sum()) & (ntot >= 3)
        lam_end = np.zeros(nb_); last = np.searchsorted(cid, np.arange(nb_), "right") - 1; lam_end = lam[last]
        idsf = np.full(len(t), -1); idsf[sg] = np.where(keepc[cid], cid, -1)
        b, m = B2.make_bursts(idsf, t, sign, vol)
        if b is not None:
            # map each kept burst back to its cluster's lambda at the last event
            fm, _lm = first_last_member(m)
            clus = idsf[fm.reindex(np.arange(len(b["t_b"]))).fillna(0).astype(int).to_numpy()]
            defs["hawkesfix"] = (b, m, b["t_e"] + confirm_wait(lam_end[np.maximum(clus, 0)]))
        # side-specific Hawkes
        idsx = np.full(len(t), -1); lamx = np.zeros(len(t)); off = 0
        for s in (1, -1):
            w = np.flatnonzero(sign == s)
            if len(w) > 3:
                c2, l2 = hawkes_with_lambda(t[w])
                idsx[w] = c2 + off; lamx[w] = l2; off += int(c2[-1]) + 1
        b, m = B2.make_bursts(idsx, t, sign, vol)
        if b is not None:
            _fm, lm = first_last_member(m)
            lend = lamx[lm.reindex(np.arange(len(b["t_b"]))).fillna(0).astype(int).to_numpy()]
            defs["hawkesside"] = (b, m, b["t_e"] + confirm_wait(lend))
    # regularity-based
    ids, _ = FS.burst_ids(t, sign, 5.0, "stream")
    s5, m5 = B2.make_bursts(np.where(sign != 0, ids, -1), t, sign, vol)
    if s5 is not None:
        idx = np.arange(len(s5["t_b"]))
        cv, R = member_stats(m5, t, idx)
        dur = s5["t_e"] - s5["t_b"]
        for name, keep in (("timer", (s5["n"] >= 4) & (cv < 0.2)), ("phase", (s5["n"] >= 4) & (dur > 3) & (R > 0.8))):
            b, m = subset(s5, m5, np.flatnonzero(keep))
            defs[name] = (b, m, b["t_e"] + 5.1)
    cb, cmb = B2.make_bursts(B2.clip_ids(t, sign, vol, unt, 30.0), t, sign, vol)
    if cb is not None:
        cv, _ = member_stats(cmb, t, np.arange(len(cb["t_b"])))
        b, m = subset(cb, cmb, np.flatnonzero((cb["n"] >= 4) & (cv < 0.25)))
        defs["twap"] = (b, m, b["t_e"] + 30.1)
    # price-path based
    sw = np.flatnonzero(multi & (sign != 0))
    if len(sw):
        idsw = np.full(len(t), -1)
        ids_sw, _ = FS.burst_ids(t[sw], sign[sw], 1.0, "run"); idsw[sw] = ids_sw
        b, m = B2.make_bursts(idsw, t, sign, vol, minsize=1)
        if b is not None:
            defs["sweep"] = (b, m, b["t_e"] + 1.1)
    if "run0.5" in defs:
        b0, m0, _ = defs["run0.5"]
        bid0, ask0, _, _ = book_at(b0["t_b"] - 0.001); bid1, ask1, _, _ = book_at(b0["t_e"] + 0.01)
        moved_against = np.where(b0["side"] > 0, ask1 > ask0, bid1 < bid0)
        mid0, mid1 = (bid0 + ask0) / 2, (bid1 + ask1) / 2
        for name, keep in (("levelclear", moved_against), ("absorbed", np.isclose(mid0, mid1) & (b0["n"] >= 3))):
            b, m = subset(b0, m0, np.flatnonzero(keep))
            defs[name] = (b, m, b["t_e"] + 0.6)
    # cancellation bursts (message level)
    if len(ct) > 10:
        ids_c, _ = FS.burst_ids(ct, cside, 0.5, "stream")
        b, _m = B2.make_bursts(ids_c, ct, cside, cvolc, minsize=5)
        if b is not None:
            defs["cancel"] = (b, None, b["t_e"] + 0.6)
    # hidden-heavy runs
    hh = np.flatnonzero((hid >= 0.5) & (sign != 0))
    if len(hh) > 2:
        idsh = np.full(len(t), -1)
        ids_h, _ = FS.burst_ids(t[hh], sign[hh], 1.0, "run"); idsh[hh] = ids_h
        b, m = B2.make_bursts(idsh, t, sign, vol, minsize=2)
        if b is not None:
            defs["hidden"] = (b, m, b["t_e"] + 1.1)

    m_open = mid.at(np.array([X.RTH0 + 60.0]))[0]; m_close = mid.at(np.array([X.RTH1]))[0]
    rng = np.random.default_rng(zlib.crc32(("v3|%s|%s" % (ticker, date)).encode()))
    out = []

    def assemble(dn, b, member, tdec, pick, tb_used, te_used, extra=None):
        side = b["side"][pick]; td = tdec[pick]; tb_, te_ = tb_used, te_used
        rows = []
        for j, i in enumerate(pick):
            if member is not None:
                mem = np.flatnonzero(member == i)
                mem = mem[t[mem] <= te_[j]]
            else:
                mem = np.array([], dtype=int)
            zm, tm = vol[mem], t[mem]
            gaps = np.diff(tm)
            un = unt[mem]; zu = np.rint(zm[un]) if len(mem) else np.array([])
            if len(zu):
                vals, cnts = np.unique(zu, return_counts=True); k = np.argmax(cnts)
                ms, nr = cnts[k] / len(mem), float(vals[k] % 100 != 0)
            else:
                ms, nr = np.nan, np.nan
            rows.append(dict(n_used=len(mem) if member is not None else b["n"][i], vol_used=zm.sum() if member is not None else b["vol"][i],
                             mode_share=ms, nonround=nr,
                             size_cv=zm.std() / zm.mean() if len(zm) and zm.mean() > 0 else np.nan,
                             size_to_depth=np.nanmean(zm / xdep[mem]) if len(mem) and np.isfinite(xdep[mem]).any() else np.nan,
                             iat_cv=gaps.std() / gaps.mean() if len(gaps) > 1 and gaps.mean() > 0 else np.nan,
                             iat_med=np.median(gaps) if len(gaps) else np.nan,
                             phase_R=float(np.abs(np.exp(2j * np.pi * (tm % 1.0)).mean())) if len(tm) else np.nan,
                             imb_first=imb[mem[0]] * side[j] * np.sign(sign[mem[0]]) if len(mem) else np.nan,
                             imb_last=imb[mem[-1]] * side[j] * np.sign(sign[mem[-1]]) if len(mem) else np.nan))
        f = pd.DataFrame(rows)
        f.insert(0, "defn", dn); f.insert(1, "date", date); f.insert(2, "ticker", ticker)
        f["side"], f["t_b"], f["t_e"], f["t_dec"] = side, tb_, te_, td
        f["n_total"], f["vol_total"] = b["n"][pick], b["vol"][pick]
        f["dur"], f["tod"] = te_ - tb_, (tb_ - X.RTH0) / 23400.0
        m_b, m_d = mid.at(tb_), mid.at(td)
        bd, ad, qbd, qad = book_at(td); bb0, ab0, qbb, qab = book_at(tb_)
        with np.errstate(invalid="ignore", divide="ignore"):
            f["spread_b"] = (ab0 - bb0) / ((ab0 + bb0) / 2) * 1e4; f["spread_dec"] = (ad - bd) / ((ad + bd) / 2) * 1e4
            f["spread_change"] = f.spread_dec - f.spread_b
            micro = (ad * qbd + bd * qad) / (qbd + qad)
            f["micro_gap"] = side * (micro - m_d) / m_d * 1e4
            opp_depth = np.where(side > 0, qab, qbb)
            f["opp_consumed"] = f.vol_used / opp_depth
            f["move_during"] = side * (m_d - m_b) / m_b * 1e4
            p60, p30 = mid.at(tb_ - 60), mid.at(np.maximum(tb_ - 1800, X.RTH0))
            f["pre60"] = side * (m_b - p60) / p60 * 1e4; f["pre30m"] = side * (m_b - p30) / p30 * 1e4
            f["since_open"] = side * (m_d - m_open) / m_open * 1e4
            f["qofi_pre60"] = side * ofi(tb_ - 60, tb_); f["qofi_during"] = side * ofi(tb_, td)
            f["tfi_pre60"] = side * wsum(csv, tb_ - 60, tb_) / np.maximum(wsum(cvol, tb_ - 60, tb_), 1e-9)
            f["canc_opp"] = np.where(side > 0, canc(tb_, td, +1), canc(tb_, td, -1))
            f["canc_same"] = np.where(side > 0, canc(tb_, td, -1), canc(tb_, td, +1))
            for h in (10, 60, 300, 1800):
                f["r%d" % h] = side * (mid.at(np.minimum(td + h, X.RTH1)) - m_d) / m_d * 1e4
            f["r_close"] = side * (m_close - m_d) / m_d * 1e4
            for L in (1, 10):
                mL = mid.at(np.minimum(td + L, X.RTH1))
                f["r10_L%d" % L] = side * (mid.at(np.minimum(td + L + 10, X.RTH1)) - mL) / mL * 1e4
                f["r60_L%d" % L] = side * (mid.at(np.minimum(td + L + 60, X.RTH1)) - mL) / mL * 1e4
        # continuation: another event of the same definition starting within 60 s after the decision
        tb_all, sd_all = b["t_b"], b["side"]
        order = np.argsort(tb_all); tbs, sds = tb_all[order], sd_all[order]
        same, opp = np.zeros(len(pick), bool), np.zeros(len(pick), bool)
        for j in range(len(pick)):
            lo, hi = np.searchsorted(tbs, td[j], "right"), np.searchsorted(tbs, td[j] + 60, "right")
            s_ = sds[lo:hi]
            same[j] = (s_ == side[j]).any(); opp[j] = (s_ == -side[j]).any()
        f["cont60_same"], f["cont60_opp"] = same.astype(int), opp.astype(int)
        if extra is not None:
            for k, v in extra.items():
                f[k] = v
        return f

    for dn, (b, member, tdec) in defs.items():
        if b is None or not len(b["t_b"]):
            continue
        ok = np.flatnonzero(tdec <= X.RTH1 - 30.0)
        if not len(ok):
            continue
        pick = np.sort(rng.choice(ok, size=min(CAP, len(ok)), replace=False))
        out.append(assemble(dn, b, member, tdec, pick, b["t_b"][pick], b["t_e"][pick]))
    # decide DURING the burst: at the k-th member of run0.5 bursts
    if "run0.5" in defs:
        b0, m0, _ = defs["run0.5"]
        for k in (3, 5):
            elig = np.flatnonzero(b0["n"] >= k)
            if not len(elig):
                continue
            kth, vk_all = kth_member(m0, t, vol, k)
            tk = kth.reindex(elig).to_numpy(float)
            ok = np.flatnonzero(np.isfinite(tk) & (tk + 0.1 <= X.RTH1 - 30.0))
            if not len(ok):
                continue
            sel = np.sort(rng.choice(ok, size=min(CAP, len(ok)), replace=False))
            pick = elig[sel]
            tdec_k = np.full(len(b0["t_b"]), np.nan); tdec_k[pick] = tk[sel] + 0.1
            vol_k = vk_all.reindex(pick).to_numpy(float)
            extra = dict(grew=(b0["n"][pick] > k).astype(int), rem_vol=b0["vol"][pick] - vol_k, k=k)
            out.append(assemble("early%d" % k, b0, m0, tdec_k, pick, b0["t_b"][pick], tk[sel], extra))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--helper", default=None)
    a = ap.parse_args()
    date = "".join(re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(a.msg).name).groups())
    df = run(a.msg, a.ticker, date, a.helper)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(len(df))


if __name__ == "__main__":
    main()
