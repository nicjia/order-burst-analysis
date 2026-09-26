#!/usr/bin/env python3
"""burst-pseudo-v1 (cluster, per name-day): the placebo for the real-time short-horizon result.

For run bursts with a 0.5 s and a 1 s gap, sample up to CAP real bursts, and for each make a PSEUDO event at a
uniformly random time in [10:00, 15:30] with the same duration and the same side. Real and pseudo events get
exactly the same features and outcomes, computed by the same function from times alone:
  PATH   side * mid move over the event window [t_b, t_dec), side * 60-s pre-move, 30-min pre-move, move since
         the open; spread at decision
  BOOK   queue imbalance at t_e and at t_b (from the touch sizes), quote OFI over the 60 s before t_b and over
         [t_b, t_dec), trade-flow imbalance over the 60 s before t_b
  OUT    side * mid move from t_dec to +10 s, +60 s, +300 s, +1800 s and to the close (bps)
If the real-time burst forecast is burst information, a model trained on real bursts should forecast real
bursts much better than pseudo events, and the recent-move signal should be stronger at real bursts.
Usage: burst_pseudo_raw.py --msg FILE --ticker TK --out OUT.csv.gz [--helper p4_bbo]
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

CAP = 15


def run(msg_path, ticker, date, helper):
    if date in B2.EARLY:
        return pd.DataFrame()
    msg = X.read_messages(msg_path)
    context, _, _, _ = X.bbo_context(msg_path, msg, helper)
    mid = X.MidPath(context[0], context[1], context[2], context[3])
    bt, bb, ba, qb, qa = context[0], context[2], context[3], context[4], context[5]
    cum_ofi = B2.ofi_cumulative(bt, bb, ba, qb, qa)
    depth = np.nanmean(np.where(qb + qa > 0, (qb + qa) / 2, np.nan)) if len(qb) else np.nan
    packets = PK.fast_packets(msg, context)
    if not len(packets):
        return pd.DataFrame()
    day = FP.packet_arrays(packets)
    r = (day["time"] >= X.RTH0) & (day["time"] < X.RTH1)
    t, sign, vol = day["time"][r], day["sign"][r].astype(int), day["volume"][r]
    if len(t) < 20:
        return pd.DataFrame()
    csv, cvol = np.cumsum(sign * vol), np.cumsum(vol)

    def wsum(c, a, b):
        ia = np.searchsorted(t, a, "left"); ib = np.searchsorted(t, b, "left")
        return np.where(ib > 0, c[np.maximum(ib - 1, 0)], 0.0) - np.where(ia > 0, c[np.maximum(ia - 1, 0)], 0.0)

    def ofi(a, b):
        ia = np.searchsorted(bt, a, "left") - 1; ib = np.searchsorted(bt, b, "left") - 1
        v = np.where(ib >= 0, cum_ofi[np.maximum(ib, 0)], 0.0) - np.where(ia >= 0, cum_ofi[np.maximum(ia, 0)], 0.0)
        return v / depth if depth and np.isfinite(depth) else np.full(np.shape(a), np.nan)

    def qimb(q):
        i = np.searchsorted(bt, q, "left") - 1
        ok = i >= 0
        b_, a_ = np.where(ok, qb[np.maximum(i, 0)], np.nan), np.where(ok, qa[np.maximum(i, 0)], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (b_ - a_) / (b_ + a_)

    m_open = mid.at(np.array([X.RTH0 + 60.0]))[0]; m_close = mid.at(np.array([X.RTH1]))[0]

    def features(kind, dn, tb, te, tdec, side):
        m_b, m_d = mid.at(tb), mid.at(tdec)
        with np.errstate(invalid="ignore", divide="ignore"):
            p60, p30 = mid.at(tb - 60), mid.at(np.maximum(tb - 1800, X.RTH0))
            f = pd.DataFrame(dict(kind=kind, defn=dn, date=date, ticker=ticker, side=side, t_b=tb, t_e=te, t_dec=tdec,
                                  dur=te - tb, tod=(tb - X.RTH0) / 23400.0,
                                  move_during=side * (m_d - m_b) / m_b * 1e4, pre60=side * (m_b - p60) / p60 * 1e4,
                                  pre30m=side * (m_b - p30) / p30 * 1e4, since_open=side * (m_d - m_open) / m_open * 1e4,
                                  spread_dec=mid.spread_bps(tdec),
                                  qimb_e=side * qimb(te), qimb_b=side * qimb(tb),
                                  qofi_pre60=side * ofi(tb - 60, tb), qofi_during=side * ofi(tb, tdec),
                                  tfi_pre60=side * wsum(csv, tb - 60, tb) / np.maximum(wsum(cvol, tb - 60, tb), 1e-9)))
            for h in (10, 60, 300, 1800):
                f["r%d" % h] = side * (mid.at(np.minimum(tdec + h, X.RTH1)) - m_d) / m_d * 1e4
            f["r_close"] = side * (m_close - m_d) / m_d * 1e4
        return f

    rng = np.random.default_rng(zlib.crc32(("pseudo|%s|%s" % (ticker, date)).encode()))
    out = []
    for g in (0.5, 1.0):
        ids, _ = FS.burst_ids(t, sign, g, "run")
        b, _member = B2.make_bursts(np.where(sign != 0, ids, -1), t, sign, vol)
        if b is None:
            continue
        wait = g + 0.1
        tdec = b["t_e"] + wait
        ok = np.flatnonzero((b["t_b"] >= X.RTH0 + 1800) & (tdec <= X.RTH1 - 1800))
        if not len(ok):
            continue
        pick = np.sort(rng.choice(ok, size=min(CAP, len(ok)), replace=False))
        tb, te, sd = b["t_b"][pick], b["t_e"][pick], b["side"][pick]
        dn = "run%g" % g
        out.append(features("real", dn, tb, te, te + wait, sd))
        u = rng.uniform(X.RTH0 + 1800, X.RTH1 - 1800 - (te - tb) - wait)
        out.append(features("pseudo", dn, u, u + (te - tb), u + (te - tb) + wait, sd))
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
