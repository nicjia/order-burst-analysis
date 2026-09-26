#!/usr/bin/env python3
"""burst-defs-raw-v4 (cluster, per name-day): forecasting products only (no trading).

(A) Term structure. For run0.01, run0.1, early5 (run0.5 acted on at its 5th child), levelclear, cancel, hidden and run60,
    plus a matched PSEUDO event for every sampled run0.1 burst (uniform random time in [10:00, 15:30], same duration
    and side): the v3 feature set, and the signed mid move from the decision to +1, 2, 5, 10, 30, 60, 120, 300, 600,
    1800, 3600 s and to the close.
(B) Volatility. Realized variance of 1-second mid log returns over the next 60, 300 and 1800 s (bps^2, summed), and
    over the 60, 300 and 1800 s BEFORE the burst began (rvpre*, the control a volatility forecast must beat).
(C) 5-minute panel over ALL bursts: per 5-minute bin, signed volume and count of bursts DECIDED in the bin for each
    definition, all-trade signed volume, total volume and packet count, quote OFI, 1-second realized variance, and the
    bin's mid at start and end.
Writes OUT (events) and OUT with _bins suffix. Usage: burst_defs_raw4.py --msg FILE --ticker TK --out OUT.csv.gz
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
import burst_defs_raw3 as B3

CAP = 10
HS = (1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)
BIN = 300.0


def run(msg_path, ticker, date, helper):
    if date in B2.EARLY:
        return pd.DataFrame(), pd.DataFrame()
    msg = X.read_messages(msg_path)
    context, _, _, _ = X.bbo_context(msg_path, msg, helper)
    mid = X.MidPath(context[0], context[1], context[2], context[3])
    bt, bb, ba, qb, qa = context[0], context[2], context[3], context[4], context[5]
    cum_ofi = B2.ofi_cumulative(bt, bb, ba, qb, qa)
    depth = np.nanmean(np.where(qb + qa > 0, (qb + qa) / 2, np.nan)) if len(qb) else np.nan
    packets = PK.fast_packets(msg, context)
    if not len(packets):
        return pd.DataFrame(), pd.DataFrame()
    P = packets.sort_values(["time", "packet_id"], kind="stable").reset_index(drop=True)
    day = FP.packet_arrays(packets)
    r = (day["time"] >= X.RTH0) & (day["time"] < X.RTH1)
    t, sign, vol = day["time"][r], day["sign"][r].astype(int), day["volume"][r]
    imb, unt, xdep = day["imbalance"][r], day["untruncated"][r], day["exec_depth"][r]
    hid = P.hidden_volume.to_numpy(float)[r] / np.maximum(P.volume.to_numpy(float)[r], 1e-9)
    if len(t) < 30:
        return pd.DataFrame(), pd.DataFrame()
    csv, cvol = np.cumsum(sign * vol), np.cumsum(vol)

    def cum_at(c, q):
        i = np.searchsorted(t, q, "left")
        return np.where(i > 0, c[np.maximum(i - 1, 0)], 0.0)

    def ofi(a, b):
        ia = np.searchsorted(bt, a, "left") - 1; ib = np.searchsorted(bt, b, "left") - 1
        v = np.where(ib >= 0, cum_ofi[np.maximum(ib, 0)], 0.0) - np.where(ia >= 0, cum_ofi[np.maximum(ia, 0)], 0.0)
        return v / depth if depth and np.isfinite(depth) else np.full(np.shape(a), np.nan)

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
    if len(ct) > 10:
        ids_c, _ = FS.burst_ids(ct, cside, 0.5, "stream")
        b, _m = B2.make_bursts(ids_c, ct, cside, cvolc, minsize=5)
        if b is not None:
            defs["cancel"] = (b, None, b["t_e"] + 0.6)
    hh = np.flatnonzero((hid >= 0.5) & (sign != 0))
    if len(hh) > 2:
        idsh = np.full(len(t), -1); ids_h, _ = FS.burst_ids(t[hh], sign[hh], 1.0, "run"); idsh[hh] = ids_h
        b, m = B2.make_bursts(idsh, t, sign, vol, minsize=2)
        if b is not None:
            defs["hidden"] = (b, m, b["t_e"] + 1.1)

    m_open = mid.at(np.array([X.RTH0 + 60.0]))[0]; m_close = mid.at(np.array([X.RTH1]))[0]
    rng = np.random.default_rng(zlib.crc32(("v4|%s|%s" % (ticker, date)).encode()))

    def features(dn, kind, side, tb_, te_, td, members):
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
                             imb_first=imb[mem[0]] * side[j] * np.sign(sign[mem[0]]) if len(mem) else np.nan,
                             imb_last=imb[mem[-1]] * side[j] * np.sign(sign[mem[-1]]) if len(mem) else np.nan))
        f = pd.DataFrame(rows)
        f.insert(0, "defn", dn); f.insert(1, "kind", kind); f.insert(2, "date", date); f.insert(3, "ticker", ticker)
        f["side"], f["t_b"], f["t_e"], f["t_dec"] = side, tb_, te_, td
        f["dur"], f["tod"] = te_ - tb_, (tb_ - X.RTH0) / 23400.0
        m_b, m_d = mid.at(tb_), mid.at(td)
        bd, ad, qbd, qad = book_at(td); bb0, ab0, qbb, qab = book_at(tb_)
        with np.errstate(invalid="ignore", divide="ignore"):
            f["spread_b"] = (ab0 - bb0) / ((ab0 + bb0) / 2) * 1e4; f["spread_dec"] = (ad - bd) / ((ad + bd) / 2) * 1e4
            f["spread_change"] = f.spread_dec - f.spread_b
            f["micro_gap"] = side * ((ad * qbd + bd * qad) / (qbd + qad) - m_d) / m_d * 1e4
            f["qimb_dec"] = side * (qbd - qad) / (qbd + qad)
            f["opp_consumed"] = f.vol_used / np.where(side > 0, qab, qbb)
            f["move_during"] = side * (m_d - m_b) / m_b * 1e4
            p60, p30 = mid.at(tb_ - 60), mid.at(np.maximum(tb_ - 1800, X.RTH0))
            f["pre60"] = side * (m_b - p60) / p60 * 1e4; f["pre30m"] = side * (m_b - p30) / p30 * 1e4
            f["since_open"] = side * (m_d - m_open) / m_open * 1e4
            f["qofi_pre60"] = side * ofi(tb_ - 60, tb_); f["qofi_during"] = side * ofi(tb_, td)
            f["tfi_pre60"] = side * (cum_at(csv, tb_) - cum_at(csv, tb_ - 60)) / np.maximum(cum_at(cvol, tb_) - cum_at(cvol, tb_ - 60), 1e-9)
            f["canc_opp"] = np.where(side > 0, canc(tb_, td, +1), canc(tb_, td, -1))
            f["canc_same"] = np.where(side > 0, canc(tb_, td, -1), canc(tb_, td, +1))
            for h in HS:
                f["r%d" % h] = side * (mid.at(np.minimum(td + h, X.RTH1)) - m_d) / m_d * 1e4
            f["r_close"] = side * (m_close - m_d) / m_d * 1e4
            for H in (60, 300, 1800):
                grid = td[:, None] + np.arange(0, H + 1)[None, :]
                mg = mid.at(np.minimum(grid, X.RTH1).ravel()).reshape(grid.shape)
                lr = np.diff(np.log(mg), axis=1) * 1e4
                f["rv%d" % H] = np.nansum(lr ** 2, axis=1)
                grid = tb_[:, None] - np.arange(H, -1, -1)[None, :]      # realized variance BEFORE the burst (control)
                mg = mid.at(np.maximum(grid, X.RTH0).ravel()).reshape(grid.shape)
                lr = np.diff(np.log(mg), axis=1) * 1e4
                f["rvpre%d" % H] = np.nansum(lr ** 2, axis=1)
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
    for dn in ("run0.01", "run0.1", "levelclear", "cancel", "hidden", "run60"):
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
    ev = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # (C) 5-minute bins over all bursts
    edges = X.RTH0 + BIN * np.arange(79)
    bins = pd.DataFrame(dict(date=date, ticker=ticker, bin=np.arange(78), m_start=mid.at(edges[:-1]), m_end=mid.at(edges[1:]),
                             flow=cum_at(csv, edges[1:]) - cum_at(csv, edges[:-1]), volume=cum_at(cvol, edges[1:]) - cum_at(cvol, edges[:-1]),
                             qofi=ofi(edges[:-1], edges[1:])))
    bins["m_open"], bins["m_close"] = m_open, m_close
    lm = np.log(mid.at(X.RTH0 + np.arange(0, 23401, dtype=float)))       # 1 s mid grid -> realized variance per bin
    bins["rv"] = np.nansum(((np.diff(lm) * 1e4) ** 2).reshape(78, 300), axis=1)
    bins["npk"] = np.bincount(np.clip(((t - X.RTH0) // BIN).astype(int), 0, 77), minlength=78)
    for dn, (b, member, tdec) in defs.items():
        j = ((tdec - X.RTH0) // BIN).astype(int); live = (j >= 0) & (j < 78)
        bins["sv_" + dn] = np.bincount(j[live], weights=(b["side"] * b["vol"])[live], minlength=78)
        bins["nb_" + dn] = np.bincount(j[live], minlength=78)
    return ev, bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--helper", default=None)
    a = ap.parse_args()
    date = "".join(re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(a.msg).name).groups())
    ev, bins = run(a.msg, a.ticker, date, a.helper)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(a.out, index=False)
    bins.to_csv(a.out.replace(".csv.gz", "_bins.csv.gz"), index=False)
    print(len(ev), len(bins))


if __name__ == "__main__":
    main()
