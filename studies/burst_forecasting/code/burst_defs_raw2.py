#!/usr/bin/env python3
"""burst-defs-raw-v2 (cluster, per name-day): short-gap and regularity-based burst definitions, decided in REAL
TIME (the moment a burst is known to have ended), with sizing/timing-regularity and order-book features, plus
30-minute bucket aggregates over ALL bursts.

Why v2: v1 decided 10 minutes after the burst started (the P4 rule), which throws away anything that plays out in
seconds to minutes -- order-book effects in particular.

Definitions (economic packets, native signs, regular hours):
  runG     same-side runs, G in {0.1, 0.25, 0.5, 1, 2, 5, 60} s; an opposite-side or unsigned packet ends a run
  streamG  each side's own sequence (the other side's trades in between are ignored), G in {0.5, 1, 2} s
  clipG    same side AND the same untruncated non-round child size, consecutive within G in {5, 30} s
  hawkes   the original C++ detector (intensity beta 1, trigger 0.3, direction 0.763 / 0.28, volume floor)
All need >= 3 packets. Decision time: runs/streams/clips t_e + G + 0.1 s (the end is known only once the gap has
passed); hawkes t_e + 1.3 s (the intensity has fallen below the trigger).

Features at the decision time: size, children, duration, time of day; SIZE REGULARITY (modal-clip share, non-round
clip, child-size CV, mean child size / touch depth); TIMING REGULARITY (inter-arrival CV, median inter-arrival,
sub-second phase concentration R = |mean exp(2 pi i frac(t))|); ORDER BOOK (queue imbalance at the first and last
packet, quote order-flow imbalance over the 60 s before and during the burst, trade-flow imbalance over the 60 s
before, spread at start and at decision); PRICE PATH (move during the burst to the decision, 60-s and 30-min
pre-moves, move since the open). Outcomes (never features): signed mid move from the decision to +10 s, +60 s,
+300 s, +1800 s and to the close, bps. Up to CAP bursts per definition per name-day, sampled at random (seeded).

Bucket file: per 30-minute bucket, mids at the bucket's start and end, all-trade flow and quote OFI in the bucket,
and per definition the signed volume and count of bursts DECIDED inside the bucket (all bursts, not sampled).
Usage: burst_defs_raw2.py --msg FILE --ticker TK --out OUT.csv.gz [--helper p4_bbo]   (writes OUT and OUT_buckets)
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
import p4_q0_legacy as LG

CAP = 10
BUCKET = 1800.0
try:
    import p4_aggregate as AG
    EARLY = set(AG.EARLY_CLOSE)
except Exception:
    EARLY = set()


def make_bursts(ids, t, sign, vol, minsize=3, side_override=None):
    """Bursts from per-packet ids (-1 = excluded). Returns dict of per-burst arrays and per-packet member index."""
    ok = ids >= 0
    if not ok.any():
        return None, None
    u, inv, cnt = np.unique(ids[ok], return_inverse=True, return_counts=True)
    keep = cnt >= minsize
    pk = np.flatnonzero(ok)
    member = np.full(len(ids), -1)
    remap = np.full(len(u), -1); remap[keep] = np.arange(keep.sum())
    member[pk] = remap[inv]
    m = member >= 0
    if not m.any():
        return None, None
    nb = int(keep.sum())
    tb = np.full(nb, np.inf); te = np.full(nb, -np.inf)
    np.minimum.at(tb, member[m], t[m]); np.maximum.at(te, member[m], t[m])
    v = np.bincount(member[m], weights=vol[m], minlength=nb)
    n = np.bincount(member[m], minlength=nb).astype(float)
    if side_override is None:
        sv = np.bincount(member[m], weights=sign[m] * vol[m], minlength=nb)
        side = np.sign(sv)
    else:
        side = side_override
    b = dict(t_b=tb, t_e=te, side=side.astype(float), n=n, vol=v)
    good = b["side"] != 0
    if not good.all():
        idx = np.flatnonzero(good); rm = np.full(nb, -1); rm[idx] = np.arange(len(idx))
        member = np.where(member >= 0, rm[np.maximum(member, 0)], -1)
        b = {k: a[idx] for k, a in b.items()}
    return b, member


def clip_ids(t, sign, size, unt, gap):
    elig = (sign != 0) & unt & (np.rint(size) % 100 != 0)
    ids = np.full(len(t), -1)
    idx = np.flatnonzero(elig)
    if len(idx) < 3:
        return ids
    order = idx[np.lexsort((t[idx], np.rint(size[idx]), sign[idx]))]
    s, z, tt = sign[order], np.rint(size[order]), t[order]
    cut = np.r_[True, (s[1:] != s[:-1]) | (z[1:] != z[:-1]) | (np.diff(tt) > gap)]
    ids[order] = np.cumsum(cut) - 1
    return ids


def hawkes_bursts(t, sign, vol, day_vol):
    sg = np.flatnonzero(sign != 0)
    if len(sg) < 10:
        return None, None
    cid = LG.hawkes_ids(t[sg])
    ids = np.full(len(t), -1); ids[sg] = cid
    nb = int(cid[-1]) + 1
    buy = sign[sg] > 0
    nbuy = np.bincount(cid, weights=buy.astype(float), minlength=nb); ntot = np.bincount(cid, minlength=nb).astype(float)
    vb = np.bincount(cid, weights=np.where(buy, vol[sg], 0.0), minlength=nb); vs = np.bincount(cid, weights=np.where(buy, 0.0, vol[sg]), minlength=nb)
    d = np.zeros(nb)
    d[(nbuy / ntot >= LG.DIR_THRESH) & (vb > 0) & (vs <= LG.VOL_RATIO * vb)] = 1
    d[(1 - nbuy / ntot >= LG.DIR_THRESH) & (vs > 0) & (vb <= LG.VOL_RATIO * vs)] = -1
    keep = (d != 0) & (vb + vs >= LG.VOL_FRAC * day_vol) & (ntot >= 3)
    ids = np.where((ids >= 0) & keep[np.maximum(ids, 0)], ids, -1)
    b, member = make_bursts(ids, t, sign, vol, 3)
    if b is None:
        return None, None
    # side = the cluster's direction (majority), recomputed from signed volume inside make_bursts
    return b, member


def ofi_cumulative(bt, bb, ba, qb, qa):
    e = np.zeros(len(bt))
    if len(bt) > 1:
        b0, b1, a0, a1 = bb[:-1], bb[1:], ba[:-1], ba[1:]
        e[1:] = (np.where(b1 >= b0, qb[1:], 0) - np.where(b1 <= b0, qb[:-1], 0)
                 - np.where(a1 <= a0, qa[1:], 0) + np.where(a1 >= a0, qa[:-1], 0))
    return np.cumsum(np.nan_to_num(e))


def run(msg_path, ticker, date, helper):
    if date in EARLY:
        return pd.DataFrame(), pd.DataFrame()
    msg = X.read_messages(msg_path)
    context, _, _, _ = X.bbo_context(msg_path, msg, helper)
    mid = X.MidPath(context[0], context[1], context[2], context[3])
    bt, bb, ba, qb, qa = context[0], context[2], context[3], context[4], context[5]
    cum_ofi = ofi_cumulative(bt, bb, ba, qb, qa)
    depth = np.nanmean(np.where(qb + qa > 0, (qb + qa) / 2, np.nan)) if len(qb) else np.nan
    packets = PK.fast_packets(msg, context)
    if not len(packets):
        return pd.DataFrame(), pd.DataFrame()
    day = FP.packet_arrays(packets)
    r = (day["time"] >= X.RTH0) & (day["time"] < X.RTH1)
    t, sign, vol = day["time"][r], day["sign"][r].astype(int), day["volume"][r]
    imb, unt, xdep = day["imbalance"][r], day["untruncated"][r], day["exec_depth"][r]
    if len(t) < 20:
        return pd.DataFrame(), pd.DataFrame()
    csv, cvol = np.cumsum(sign * vol), np.cumsum(vol)

    def wsum(c, a, b):
        ia = np.searchsorted(t, a, "left"); ib = np.searchsorted(t, b, "left")
        return np.where(ib > 0, c[np.maximum(ib - 1, 0)], 0.0) - np.where(ia > 0, c[np.maximum(ia - 1, 0)], 0.0)

    def ofi(a, b):
        ia = np.searchsorted(bt, a, "left") - 1; ib = np.searchsorted(bt, b, "left") - 1
        v = np.where(ib >= 0, cum_ofi[np.maximum(ib, 0)], 0.0) - np.where(ia >= 0, cum_ofi[np.maximum(ia, 0)], 0.0)
        return v / depth if depth and np.isfinite(depth) else np.full(np.shape(a), np.nan)

    defs, waits = {}, {}
    for g in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 60.0):
        ids, _ = FS.burst_ids(t, sign, g, "run")
        defs["run%g" % g] = make_bursts(np.where(sign != 0, ids, -1), t, sign, vol); waits["run%g" % g] = g + 0.1
    for g in (0.5, 1.0, 2.0):
        ids, _ = FS.burst_ids(t, sign, g, "stream")
        defs["stream%g" % g] = make_bursts(np.where(sign != 0, ids, -1), t, sign, vol); waits["stream%g" % g] = g + 0.1
    for g in (5.0, 30.0):
        defs["clip%g" % g] = make_bursts(clip_ids(t, sign, vol, unt, g), t, sign, vol); waits["clip%g" % g] = g + 0.1
    defs["hawkes"] = hawkes_bursts(t, sign, vol, vol.sum()); waits["hawkes"] = 1.3

    m_open = mid.at(np.array([X.RTH0 + 60.0]))[0]; m_close = mid.at(np.array([X.RTH1]))[0]
    rng = np.random.default_rng(zlib.crc32(("%s|%s" % (ticker, date)).encode()))
    edges = X.RTH0 + BUCKET * np.arange(14)
    bk = pd.DataFrame(dict(bucket=np.arange(13), m_start=mid.at(edges[:-1]), m_end=mid.at(edges[1:]),
                           flow=[wsum(csv, edges[j], edges[j + 1]) for j in range(13)],
                           volume=[wsum(cvol, edges[j], edges[j + 1]) for j in range(13)],
                           qofi=ofi(edges[:-1], edges[1:])))
    bk["m_open"], bk["m_close"] = m_open, m_close
    out = []
    for dn, (b, member) in defs.items():
        if b is None or not len(b["t_b"]):
            bk["sv_" + dn] = 0.0; bk["nb_" + dn] = 0
            continue
        tdec = b["t_e"] + waits[dn]
        j = np.clip(((tdec - X.RTH0) // BUCKET).astype(int), 0, 12)
        live = tdec < X.RTH1
        bk["sv_" + dn] = np.bincount(j[live], weights=(b["side"] * b["vol"])[live], minlength=13)
        bk["nb_" + dn] = np.bincount(j[live], minlength=13)
        ok = np.flatnonzero(tdec <= X.RTH1 - 30.0)
        if not len(ok):
            continue
        pick = np.sort(rng.choice(ok, size=min(CAP, len(ok)), replace=False))
        rows = []
        for i in pick:
            mem = np.flatnonzero(member == i)
            tm, zm, sm = t[mem], vol[mem], sign[mem]
            side = b["side"][i]; tb, te, td = b["t_b"][i], b["t_e"][i], tdec[i]
            gaps = np.diff(tm)
            un = unt[mem]
            zu = np.rint(zm[un])
            if len(zu):
                vals, cnts = np.unique(zu, return_counts=True); k = np.argmax(cnts)
                mode_share, nonround = cnts[k] / len(mem), float(vals[k] % 100 != 0)
            else:
                mode_share, nonround = 0.0, 0.0
            ph = np.exp(2j * np.pi * (tm % 1.0))
            rows.append(dict(
                defn=dn, date=date, ticker=ticker, side=side, t_b=tb, t_e=te, t_dec=td, n=len(mem), vol=zm.sum(),
                dur=te - tb, tod=(tb - X.RTH0) / 23400.0,
                mode_share=mode_share, nonround=nonround, size_cv=zm.std() / zm.mean() if zm.mean() > 0 else np.nan,
                size_to_depth=np.nanmean(zm / xdep[mem]) if np.isfinite(xdep[mem]).any() else np.nan,
                iat_cv=gaps.std() / gaps.mean() if len(gaps) > 1 and gaps.mean() > 0 else np.nan,
                iat_med=np.median(gaps) if len(gaps) else np.nan, phase_R=float(np.abs(ph.mean())),
                imb_first=imb[mem[0]] * side * sm[0] if sm[0] != 0 else np.nan,
                imb_last=imb[mem[-1]] * side * sm[-1] if sm[-1] != 0 else np.nan,
                qofi_pre60=side * ofi(np.array([tb - 60]), np.array([tb]))[0],
                qofi_during=side * ofi(np.array([tb]), np.array([td]))[0],
                tfi_pre60=side * wsum(csv, tb - 60, tb) / max(wsum(cvol, tb - 60, tb), 1e-9),
                ))
        f = pd.DataFrame(rows)
        tb_, td_ = f.t_b.to_numpy(), f.t_dec.to_numpy()
        m_b, m_d = mid.at(tb_), mid.at(td_)
        s = f.side.to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            f["spread_b"] = mid.spread_bps(tb_); f["spread_dec"] = mid.spread_bps(td_)
            f["move_during"] = s * (m_d - m_b) / m_b * 1e4
            p60, p30 = mid.at(tb_ - 60), mid.at(np.maximum(tb_ - 1800, X.RTH0))
            f["pre60"] = s * (m_b - p60) / p60 * 1e4; f["pre30m"] = s * (m_b - p30) / p30 * 1e4
            f["since_open"] = s * (m_d - m_open) / m_open * 1e4
            for h in (10, 60, 300, 1800):
                f["r%d" % h] = s * (mid.at(np.minimum(td_ + h, X.RTH1)) - m_d) / m_d * 1e4
            f["r_close"] = s * (m_close - m_d) / m_d * 1e4
        out.append(f)
    bursts = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    bk.insert(0, "date", date); bk.insert(1, "ticker", ticker)
    return bursts, bk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--helper", default=None)
    a = ap.parse_args()
    date = "".join(re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(a.msg).name).groups())
    bursts, bk = run(a.msg, a.ticker, date, a.helper)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bursts.to_csv(a.out, index=False)
    bk.to_csv(a.out.replace(".csv.gz", "_buckets.csv.gz"), index=False)
    print(len(bursts), len(bk))


if __name__ == "__main__":
    main()
