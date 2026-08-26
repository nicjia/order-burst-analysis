#!/usr/bin/env python3
"""
hidden_perprint.py — the decisive check: does the footprint exist WITHOUT any burst at all?

Array 14314701 found that the measured footprint degrades as price-conditioning is removed
from the burst construction: +1.48 with same-sign runs and contemporaneous-mid signing, +0.60
with time-only clustering and pre-print-mid signing, +0.03 with time-only clustering and
outside-pre-quote signing. But that last arm changed TWO things at once (formation and
signing) and its clusters are sparse, and it contradicts the per-print outside-quote results
already in the paper (+2.09 in tab:signrobust, +1.32 post-print in the event-time table).

Both cannot be right, so we remove clustering entirely. Every hidden print is its own
observation. Nothing here forms a burst, so nothing here can inherit formation circularity;
the only choice left is how the print is signed.

  now : sign by the CONTEMPORANEOUS midpoint          (the paper's rule; can be stale)
  pre : sign by the midpoint 1ms BEFORE the print     (cannot have been moved by the print)
  far : sign ONLY prints outside the PRE-print quote  (side unambiguous under any mid error)

Each is measured from the print (t) and from t+1s, the latter excluding movement coincident
with the touch changing. If `far` from t is ~+1.3, clustering is what destroyed the signal in
arm C and the flow-level result stands. If `far` is ~0, the footprint does not survive clean
signing and the measurement is an artifact.

Output: ticker,date,n_all,n_now,n_pre,n_far,
        mk_now_t,mk_now_1s,mk_pre_t,mk_pre_1s,mk_far_t,mk_far_1s,mk_far_15,mk_far_30
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
NCOL = 14
NA = "{t},{d}" + ",nan" * (NCOL - 2)


def wm(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[np.abs(x) <= 1000]
    return float(np.mean(x)) if len(x) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    a = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(a.msg))
    date = int(m.group(1) + m.group(2) + m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(a.msg)
        df = pd.read_csv(a.msg, header=None, usecols=[0, 1, 3, 4], names=["t", "ty", "sz", "px"])
        h = df[df.ty == 5]
        if len(h) < 10 or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE
        k = (ht >= RTH0) & (ht < RTH1 - 1800.0)
        ht, hpx = ht[k], hpx[k]
        if len(ht) < 10:
            print(NA.format(t=a.ticker, d=date)); return

        m_now = BA.mid_at(bt, bm, ht)
        m_pre = BA.mid_at(bt, bm, ht - 1e-3)
        pb, pa = BA.bbo_at(bt, bb, ba, ht - 1e-3)
        b1 = BA.mid_at(bt, bm, ht + 1.0)
        f3 = BA.mid_at(bt, bm, ht + 180.0)
        f15 = BA.mid_at(bt, bm, ht + 900.0)
        f30 = BA.mid_at(bt, bm, ht + 1800.0)

        ok = np.isfinite(m_now) & (m_now > 0) & np.isfinite(m_pre) & (m_pre > 0)
        for v in (ht, hpx, m_now, m_pre, pb, pa, b1, f3, f15, f30):
            pass
        ht, hpx, m_now, m_pre, pb, pa, b1, f3, f15, f30 = (
            v[ok] for v in (ht, hpx, m_now, m_pre, pb, pa, b1, f3, f15, f30))
        if len(ht) < 10:
            print(NA.format(t=a.ticker, d=date)); return
        n_all = len(ht)

        s_now = np.where(hpx > m_now, 1, np.where(hpx < m_now, -1, 0))
        s_pre = np.where(hpx > m_pre, 1, np.where(hpx < m_pre, -1, 0))
        s_far = np.zeros(n_all, int)
        s_far[np.isfinite(pa) & (hpx > pa)] = 1
        s_far[np.isfinite(pb) & (hpx < pb)] = -1

        def mk(sign, base, fwd):
            sel = sign != 0
            if sel.sum() < 3:
                return np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                r = sign[sel] * (fwd[sel] - base[sel]) / base[sel] * 1e4
            return wm(r)

        vals = [n_all, int((s_now != 0).sum()), int((s_pre != 0).sum()), int((s_far != 0).sum()),
                mk(s_now, m_now, f3), mk(s_now, b1, f3),
                mk(s_pre, m_pre, f3), mk(s_pre, b1, f3),
                mk(s_far, m_now, f3), mk(s_far, b1, f3),
                mk(s_far, m_now, f15), mk(s_far, m_now, f30)]
        out = [a.ticker, str(date)]
        for v in vals:
            out.append(str(int(v)) if isinstance(v, (int, np.integer)) else "%.5f" % v)
        print(",".join(out))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
