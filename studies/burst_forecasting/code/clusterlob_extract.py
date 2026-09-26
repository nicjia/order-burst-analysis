#!/usr/bin/env python3
"""ClusterLOB replication (Zhang, Cucuringu, Shestopaloff, Zohren, Quantitative Finance 2026), per name-day, cluster.

The archives hold only LOBSTER message files, so the visible book is rebuilt from messages (adds 1, partial
cancels 2, deletions 3, visible executions 4; hidden executions 5 do not change the visible book). Unknown orders
from before the file's start are clamped at zero. For every regular-hours add / cancel / visible execution the six
ClusterLOB features are computed from the book *before* the event:
  vol_level  visible volume at the event's price on its side
  t_mid      time since the mid last changed
  t_first    time since the price level last became non-empty (the level's age)
  t_prev     time since the previous event at this price and side
  sbs        same-side volume from the event price to the best price on that side (inclusive), within 50 ticks
  obs        opposite-side volume from the best opposite price to the mirror price 2*mid - price, within 50 ticks
Event sign for order-flow imbalance: bid add +, ask add -, bid cancel -, ask cancel +, execution of a resting ask
(aggressive buy) +, of a resting bid -.
Modes:
  sample  write a random sample of events (features, type, sign, size, time)            -> OUT (csv.gz)
  apply   standardise with the training means / sds, assign each event to the nearest of the K training centres,
          aggregate per 30-minute bucket and cluster: size-based and count-based OFI and total size; plus bucket
          mids at start / end and the closing mid                                        -> OUT (csv.gz)
Usage: clusterlob_extract.py --msg FILE --ticker TK --out OUT --mode sample|apply [--centers JSON] [--helper p4_bbo]
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import argparse, json, re, zlib
from pathlib import Path
import numpy as np
import pandas as pd
import p4_extract as X

TICK = 100            # LOBSTER prices are dollars x 10000; one cent = 100
MAXT = 50             # ticks scanned for sbs / obs
SAMPLE = 3000
BUCKET = 1800.0
FEATS = ["vol_level", "t_mid", "t_first", "t_prev", "sbs", "obs"]


def features(msg, mid, context):
    t = msg["t"].to_numpy(float); ty = msg["ty"].to_numpy(int); oid = msg["oid"].to_numpy(np.int64)
    sz = msg["sz"].to_numpy(np.int64); px = msg["px"].to_numpy(np.int64); dr = msg["dr"].to_numpy(int)
    # best bid / ask prevailing strictly before each message, from the C++ book path, in LOBSTER integer prices
    bt = context[0]
    k = np.searchsorted(bt, t, "left") - 1; okk = k >= 0; kk = np.maximum(k, 0)
    bbest = np.where(okk, np.rint(context[2][kk] * X.BA.SCALE), np.nan)
    abest = np.where(okk, np.rint(context[3][kk] * X.BA.SCALE), np.nan)
    sel = np.isin(ty, [1, 2, 3, 4]) & np.isin(dr, [1, -1])
    rth = sel & (t >= X.RTH0) & (t < X.RTH1)
    book = {1: {}, -1: {}}; first = {1: {}, -1: {}}; last = {1: {}, -1: {}}
    orders = {}
    F = np.full((int(rth.sum()), 6), np.nan); rows = np.flatnonzero(rth); r_i = 0
    for i in np.flatnonzero(sel):
        e_ty, e_sd, e_px, e_sz, e_t = ty[i], dr[i], px[i], sz[i], t[i]
        if rth[i]:
            lv = book[e_sd]
            vol_level = lv.get(e_px, 0)
            t_first = e_t - first[e_sd][e_px] if e_px in first[e_sd] else 0.0
            t_prev = e_t - last[e_sd][e_px] if e_px in last[e_sd] else np.nan
            bs, bo = (bbest[i], abest[i]) if e_sd == 1 else (abest[i], bbest[i])
            sbs = 0; obs = 0
            if np.isfinite(bs):
                lo, hi = (e_px, int(bs)) if e_sd == 1 else (int(bs), e_px)
                if 0 <= (hi - lo) // TICK <= MAXT:
                    for q in range(lo, hi + 1, TICK):
                        sbs += lv.get(q, 0)
            if np.isfinite(bs) and np.isfinite(bo):
                m = (bs + bo) / 2.0; mirror = 2 * m - e_px
                ol = book[-e_sd]
                lo, hi = (bo, mirror) if e_sd == 1 else (mirror, bo)
                if 0 <= (hi - lo) / TICK <= MAXT:
                    q = int(np.ceil(lo / TICK) * TICK)
                    while q <= hi:
                        obs += ol.get(q, 0); q += TICK
            F[r_i] = (vol_level, np.nan, t_first, t_prev, sbs, obs); r_i += 1
        if e_ty == 1:
            nv = book[e_sd].get(e_px, 0) + e_sz
            if nv == e_sz:
                first[e_sd][e_px] = e_t
            book[e_sd][e_px] = nv
            orders[oid[i]] = [e_sz]
        else:
            o = orders.get(oid[i]); dec = e_sz
            if o is not None:
                dec = min(e_sz, o[0]); o[0] -= dec
                if o[0] <= 0 or e_ty == 3:
                    orders.pop(oid[i], None)
            v = book[e_sd].get(e_px, 0) - dec
            if v <= 0:
                book[e_sd].pop(e_px, None); first[e_sd].pop(e_px, None)
            else:
                book[e_sd][e_px] = v
        last[e_sd][e_px] = e_t
    te = t[rows]
    chg = np.r_[0, np.flatnonzero(np.diff(mid.m) != 0) + 1]
    chg_t = mid.t[chg]
    j = np.searchsorted(chg_t, te, "right") - 1
    F[:, 1] = np.where(j >= 0, te - chg_t[np.maximum(j, 0)], np.nan)
    f = pd.DataFrame(F, columns=FEATS)
    e_ty, e_sd = ty[rows], dr[rows]
    f["t"], f["ty"], f["size"] = te, e_ty, sz[rows]
    f["sign"] = np.where(e_ty == 1, e_sd, -e_sd)        # adds keep their side; cancels and executions flip
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--mode", required=True, choices=["sample", "apply"])
    ap.add_argument("--centers", default=None); ap.add_argument("--helper", default=None)
    a = ap.parse_args()
    date = "".join(re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(a.msg).name).groups())
    msg = X.read_messages(a.msg)
    context, _, _, _ = X.bbo_context(a.msg, msg, a.helper)
    mid = X.MidPath(context[0], context[1], context[2], context[3])
    f = features(msg, mid, context)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "sample":
        rng = np.random.default_rng(zlib.crc32(("clob|%s|%s" % (a.ticker, date)).encode()))
        s = f.iloc[np.sort(rng.choice(len(f), size=min(SAMPLE, len(f)), replace=False))] if len(f) else f
        s.insert(0, "date", date); s.insert(1, "ticker", a.ticker)
        s.to_csv(a.out, index=False)
        print(len(f), len(s))
        return
    c = json.loads(Path(a.centers).read_text())
    Z = f[FEATS].to_numpy(float)
    Z = np.log1p(np.clip(np.nan_to_num(Z, nan=0.0), 0, None))        # heavy-tailed features on a log scale
    Z = (Z - np.array(c["mu"])) / np.array(c["sd"])
    C = np.array(c["centers"])
    lab = np.argmin(((Z[:, None, :] - C[None, :, :]) ** 2).sum(2), axis=1)
    edges = X.RTH0 + BUCKET * np.arange(14)
    j = np.clip(((f.t.to_numpy() - X.RTH0) // BUCKET).astype(int), 0, 12)
    rows = []
    sgn, size = f.sign.to_numpy(), f["size"].to_numpy(float)
    for b in range(13):
        mb = j == b
        r = dict(date=date, ticker=a.ticker, bucket=b, m_start=float(mid.at(np.array([edges[b]]))[0]),
                 m_end=float(mid.at(np.array([edges[b + 1]]))[0]), m_close=float(mid.at(np.array([X.RTH1]))[0]),
                 m_open=float(mid.at(np.array([X.RTH0 + 60]))[0]), total_size=float(size[mb].sum()))
        for k in range(len(C)):
            mk = mb & (lab == k)
            r["ofi_size_%d" % k] = float((sgn[mk] * size[mk]).sum()); r["ofi_count_%d" % k] = float(sgn[mk].sum())
            r["size_%d" % k] = float(size[mk].sum()); r["n_%d" % k] = int(mk.sum())
        r["ofi_size_all"] = float((sgn[mb] * size[mb]).sum()); r["ofi_count_all"] = float(sgn[mb].sum())
        rows.append(r)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(len(f))


if __name__ == "__main__":
    main()
