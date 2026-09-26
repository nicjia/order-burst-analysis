#!/usr/bin/env python3
"""
idea_zoo.py — sweep harness for alternative burst ideas on a single name.

Emits LONG format so heterogeneous ideas can share one file:

    ticker,date,idea,family,param,n,v3,v10,v30,net3

  family DIR   v3/v10/v30 are signed midpoint markouts (bps) at 3/10/30 min from the event;
               net3 = v3 - 2*(half-spread at the event), i.e. after a round-trip crossing.
  family VOL   v* are information coefficients: corr(predictor, |future return|).
  family LIQ   v* are corr(predictor, future half-spread).
  family FLOW  v* are corr(predictor, future signed volume).
  family _daily one row per statistic; v3 holds the value. Daily-scale ideas (campaign
               persistence, rate acceleration, mix drift, turnover, gap magnitude) are built
               from these in aggregation, since they need many days rather than many events.

Design rules inherited from earlier results: formation never conditions on price, and any
visible message uses its NATIVE Direction field rather than an inferred sign.
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
OUT = "{tk},{d},{idea},{fam},{p},{n},{a},{b},{c},{e}"


def wm(x, cap=1000.0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[np.abs(x) <= cap]
    return float(np.mean(x)) if len(x) else np.nan


def ic(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    k = np.isfinite(x) & np.isfinite(y)
    if k.sum() < 20 or np.std(x[k]) < 1e-12 or np.std(y[k]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[k], y[k])[0, 1])


def row(tk, d, idea, fam, p, n, a, b, c, e=np.nan):
    f = lambda v: ("%.5f" % v) if np.isfinite(v) else "nan"
    print(OUT.format(tk=tk, d=d, idea=idea, fam=fam, p=p, n=int(n),
                     a=f(a), b=f(b), c=f(c), e=f(e)))


def dir_score(tk, d, idea, p, bt, bm, bb, ba, ends, dirs):
    if len(ends) < 5:
        row(tk, d, idea, "DIR", p, len(ends), np.nan, np.nan, np.nan); return
    b0 = BA.mid_at(bt, bm, ends)
    lo, hi = BA.bbo_at(bt, bb, ba, ends)
    v = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for dt in (180.0, 600.0, 1800.0):
            v.append(wm(dirs * (BA.mid_at(bt, bm, ends + dt) - b0) / b0 * 1e4))
        hs = wm(0.5 * (hi - lo) / b0 * 1e4)
    row(tk, d, idea, "DIR", p, len(ends), v[0], v[1], v[2], v[0] - 2 * hs)


def clusters(t, minrun, gap):
    out = []; i, n = 0, len(t)
    while i < n:
        j = i
        while j + 1 < n and (t[j + 1] - t[j]) < gap:
            j += 1
        if j - i + 1 >= minrun:
            out.append((i, j))
        i = j + 1
    return out


def emit(t, sign, sl, cons=0.7):
    e, dd = [], []
    for a, b in sl:
        s = sign[a:b + 1]; nz = s[s != 0]
        if len(nz) < 2:
            continue
        net = nz.sum()
        if net == 0 or abs(net) / len(nz) < cons:
            continue
        e.append(t[b]); dd.append(int(np.sign(net)))
    return np.array(e, float), np.array(dd, int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    a = ap.parse_args()
    tk = a.ticker
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(a.msg))
    d = int(m.group(1) + m.group(2) + m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(a.msg)
        df = pd.read_csv(a.msg, header=None, usecols=[0, 1, 3, 4, 5],
                         names=["t", "ty", "sz", "px", "dr"])
        df = df[(df.t >= RTH0) & (df.t < RTH1)]
        if len(df) < 500 or len(bt) < 100:
            return
        T = df.t.to_numpy(float); TY = df.ty.to_numpy(int)
        SZ = df.sz.to_numpy(float); PX = df.px.to_numpy(float) / BA.SCALE
        DR = df.dr.to_numpy(int)

        vm = TY == 4
        tv, sv, pv, av = T[vm], SZ[vm], PX[vm], -DR[vm]      # av = native aggressor side
        hm = TY == 5
        th, sh, ph = T[hm], SZ[hm], PX[hm]
        qm = (TY == 1) | (TY == 2) | (TY == 3)
        tq = T[qm]
        medsz = float(np.median(sv)) if len(sv) else 1.0

        # ---------------- daily summary stats (feed the daily-scale ideas) -------------
        grid = np.arange(RTH0, RTH1, 60.0)
        gm = BA.mid_at(bt, bm, grid)
        gl, gh = BA.bbo_at(bt, bb, ba, grid)
        with np.errstate(invalid="ignore", divide="ignore"):
            hs_bps = wm(0.5 * (gh - gl) / gm * 1e4)
            rets = np.diff(gm) / gm[:-1]
        rv = float(np.nanstd(rets) * np.sqrt(len(rets))) * 1e4 if len(rets) > 10 else np.nan
        stats = {
            "vol_visible": float(sv.sum()), "vol_hidden": float(sh.sum()),
            "hidden_share": float(sh.sum() / max(sv.sum() + sh.sum(), 1)),
            "signed_vis": float((av * sv).sum()), "n_vis": len(tv), "n_hid": len(th),
            "n_quote": int(qm.sum()), "halfsp_bps": hs_bps, "rv_bps": rv,
            "open_mid": float(BA.mid_at(bt, bm, np.array([RTH0 + 60.0]))[0]),
            "close_mid": float(BA.mid_at(bt, bm, np.array([RTH1 - 60.0]))[0]),
            "medsz": medsz,
        }
        for k, v in stats.items():
            row(tk, d, "_daily", "_daily", k, 1, v, np.nan, np.nan)

        # ================= DIRECTIONAL =================
        # i13 metaorder: sustained one-sided visible flow over a window
        for W in (60.0, 300.0, 900.0):
            edges = np.arange(RTH0, RTH1 - 1800.0, W)
            if len(edges) < 10 or len(tv) < 50:
                continue
            idx = np.clip(((tv - RTH0) / W).astype(int), 0, len(edges) - 1)
            net = np.zeros(len(edges)); tot = np.zeros(len(edges))
            np.add.at(net, idx, av * sv); np.add.at(tot, idx, sv)
            with np.errstate(invalid="ignore", divide="ignore"):
                imb = net / np.maximum(tot, 1)
            sel = np.abs(imb) > 0.3
            dir_score(tk, d, "i13_metaorder", int(W), bt, bm, bb, ba,
                      edges[sel] + W, np.sign(imb[sel]).astype(int))

        # i14 iceberg: repeated hidden prints at the SAME price within W
        for W in (5.0, 30.0, 120.0):
            if len(th) < 20:
                continue
            e, dd = [], []
            i = 0
            while i < len(th):
                j = i
                while j + 1 < len(th) and th[j + 1] - th[i] <= W and abs(ph[j + 1] - ph[i]) < 1e-9:
                    j += 1
                if j - i + 1 >= 3:
                    lo, hi = BA.bbo_at(bt, bb, ba, np.array([th[i] - 1e-3]))
                    s = 1 if ph[i] >= hi[0] else (-1 if ph[i] <= lo[0] else 0)
                    if s:
                        e.append(th[j]); dd.append(s)
                i = j + 1
            dir_score(tk, d, "i14_iceberg", int(W), bt, bm, bb, ba,
                      np.array(e, float), np.array(dd, int))

        # i18 algo schedule: repeated identical trade sizes (slice fingerprint)
        for K in (3, 5, 10):
            if len(tv) < 50:
                continue
            e, dd = [], []
            i = 0
            while i < len(tv):
                j = i
                while j + 1 < len(tv) and sv[j + 1] == sv[i] and av[j + 1] == av[i] \
                        and tv[j + 1] - tv[j] < 60.0:
                    j += 1
                if j - i + 1 >= K:
                    e.append(tv[j]); dd.append(int(av[i]))
                i = j + 1
            dir_score(tk, d, "i18_algosched", K, bt, bm, bb, ba,
                      np.array(e, float), np.array(dd, int))

        # i24 volume clock: bursts defined in volume time, not calendar time
        if len(tv) > 100:
            cum = np.cumsum(sv); tot = cum[-1]
            for F in (0.002, 0.005, 0.01):
                step = tot * F
                if step <= 0:
                    continue
                bkt = (cum / step).astype(int)
                e, dd = [], []
                for b in np.unique(bkt):
                    k = bkt == b
                    if k.sum() < 3:
                        continue
                    net = (av[k] * sv[k]).sum()
                    if abs(net) / max(sv[k].sum(), 1) > 0.3:
                        e.append(tv[k][-1]); dd.append(int(np.sign(net)))
                dir_score(tk, d, "i24_volclock", F, bt, bm, bb, ba,
                          np.array(e, float), np.array(dd, int))

        # i25 directional-change events (intrinsic time)
        gmid = BA.mid_at(bt, bm, np.arange(RTH0, RTH1, 1.0))
        gt = np.arange(RTH0, RTH1, 1.0)
        ok = np.isfinite(gmid) & (gmid > 0)
        gmid, gt = gmid[ok], gt[ok]
        for DLT in (5.0, 10.0, 25.0):
            if len(gmid) < 100:
                continue
            e, dd = [], []
            ext = gmid[0]; mode = 0
            for i in range(1, len(gmid)):
                ch = (gmid[i] - ext) / ext * 1e4
                if mode >= 0 and ch <= -DLT:
                    e.append(gt[i]); dd.append(-1); ext = gmid[i]; mode = -1
                elif mode <= 0 and ch >= DLT:
                    e.append(gt[i]); dd.append(1); ext = gmid[i]; mode = 1
                elif (mode >= 0 and gmid[i] > ext) or (mode <= 0 and gmid[i] < ext):
                    ext = gmid[i]
            dir_score(tk, d, "i25_dirchange", DLT, bt, bm, bb, ba,
                      np.array(e, float), np.array(dd, int))

        # i26 quote-update bursts (maker side), signed by submission side
        for MR in (10, 25, 50):
            tq2 = T[qm]; dq2 = DR[qm]
            if len(tq2) < 100:
                continue
            dir_score(tk, d, "i26_quoteburst", MR, bt, bm, bb, ba,
                      *emit(tq2, dq2, clusters(tq2, MR, 0.5), cons=0.75))

        # ================= VOLATILITY =================
        for W in (60.0, 300.0, 900.0):
            edges = np.arange(RTH0, RTH1 - 1800.0, W)
            if len(edges) < 20:
                continue
            cnt = np.histogram(tv, bins=np.append(edges, edges[-1] + W))[0].astype(float)
            hcnt = np.histogram(th, bins=np.append(edges, edges[-1] + W))[0].astype(float)
            m0 = BA.mid_at(bt, bm, edges + W)
            fwd = BA.mid_at(bt, bm, edges + W + 600.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                absr = np.abs((fwd - m0) / m0) * 1e4
            row(tk, d, "i19_burstvol", "VOL", int(W), len(edges),
                ic(cnt, absr), ic(hcnt, absr), ic(cnt + hcnt, absr))
            # i23 vol-of-vol: dispersion of arrival rate predicts dispersion of |move|
            row(tk, d, "i23_volvol", "VOL", int(W), len(edges),
                ic(np.abs(np.diff(cnt, prepend=cnt[0])), absr), np.nan, np.nan)

        # ================= LIQUIDITY =================
        for W in (60.0, 300.0):
            edges = np.arange(RTH0, RTH1 - 1800.0, W)
            if len(edges) < 20:
                continue
            cnt = np.histogram(th, bins=np.append(edges, edges[-1] + W))[0].astype(float)
            e0 = edges + W
            l0, h0 = BA.bbo_at(bt, bb, ba, e0)
            l1, h1 = BA.bbo_at(bt, bb, ba, e0 + 300.0)
            mm = BA.mid_at(bt, bm, e0)
            with np.errstate(invalid="ignore", divide="ignore"):
                fut_hs = 0.5 * (h1 - l1) / mm * 1e4
                cur_hs = 0.5 * (h0 - l0) / mm * 1e4
            row(tk, d, "i07_spreadwiden", "LIQ", int(W), len(edges),
                ic(cnt, fut_hs), ic(cnt, fut_hs - cur_hs), np.nan)
            bs0, as0 = BA.bbo_at(bt, bbsz, basz, e0)
            bs1, as1 = BA.bbo_at(bt, bbsz, basz, e0 + 300.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                dep_chg = (bs1 + as1) / np.maximum(bs0 + as0, 1) - 1.0
            row(tk, d, "i08_queuedepl", "LIQ", int(W), len(edges),
                ic(cnt, dep_chg), np.nan, np.nan)

        # ================= FLOW ANTICIPATION =================
        for W in (60.0, 300.0, 900.0):
            edges = np.arange(RTH0, RTH1 - 1800.0, W)
            if len(edges) < 20 or len(tv) < 50:
                continue
            idx = np.clip(((tv - RTH0) / W).astype(int), 0, len(edges) - 1)
            net = np.zeros(len(edges))
            np.add.at(net, idx, av * sv)
            row(tk, d, "i13b_flowpersist", "FLOW", int(W), len(edges),
                ic(net[:-1], net[1:]), ic(net[:-2], net[2:]), np.nan)
    except Exception as e:
        print(f"{tk},{d},ERR,ERR,{e},0,nan,nan,nan,nan", file=sys.stderr)


if __name__ == "__main__":
    main()
