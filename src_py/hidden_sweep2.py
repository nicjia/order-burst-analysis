#!/usr/bin/env python3
"""
hidden_sweep2.py — non-circular version of the sequential-sweep test.

The first version conditioned on the touch being swept within 100ms-1s and then measured the
3-minute markout FROM the print. That is circular in the same way an earlier refill test was:
the footprint is ~60% impounded within one second, so conditioning on a quote move inside the
first second conditions on most of the outcome. The resulting +3.39 vs +0.19 split is
mechanically guaranteed and says nothing about the sweep channel's economic role.

Here the conditioning window and the measurement window are disjoint. We classify the print
on what happens in [t, t+100ms] only, then measure the forward move starting from t+1s and
from t+5s -- strictly after the displacement has occurred. If swept prints still carry a
larger forward move measured from AFTER the sweep, the sweep marks genuinely more informative
flow; if the gap collapses, the original split was displacement mechanics.

We report the from-t markout alongside so the contamination is visible rather than asserted.

Output: ticker,date,n_agg,fsw,
        mkT_sw,mkT_un,      3-min markout from t          (circular, reference only)
        mk1_sw,mk1_un,      t+1s -> t+180s                (clean)
        mk5_sw,mk5_un,      t+5s -> t+185s                (clean, larger buffer)
        dep_sw,dep_un       mean depth change at +100ms (%), by group
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
        if len(h) < 5 or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE
        mid = BA.mid_at(bt, bm, ht)
        ok = np.isfinite(mid) & (mid > 0) & (ht >= RTH0 + 180.0) & (ht < RTH1 - 1800.0)
        ht, hpx, mid = ht[ok], hpx[ok], mid[ok]
        q = np.where(hpx > mid, 1, np.where(hpx < mid, -1, 0))
        agg = q != 0
        if agg.sum() < 10:
            print(NA.format(t=a.ticker, d=date)); return
        ta, qa, m0 = ht[agg], q[agg], mid[agg]

        # --- conditioning window: [t, t+100ms] only ---
        pb0, pa0 = BA.bbo_at(bt, bb, ba, ta - 1e-3)
        bs0, as0 = BA.bbo_at(bt, bbsz, basz, ta - 1e-3)
        pb1, pa1 = BA.bbo_at(bt, bb, ba, ta + 0.1)
        bs1, as1 = BA.bbo_at(bt, bbsz, basz, ta + 0.1)
        p_pre = np.where(qa > 0, pa0, pb0); p_post = np.where(qa > 0, pa1, pb1)
        s_pre = np.where(qa > 0, as0, bs0); s_post = np.where(qa > 0, as1, bs1)
        good = np.isfinite(p_pre) & np.isfinite(p_post) & np.isfinite(s_pre) \
            & np.isfinite(s_post) & (s_pre > 0)
        moved = np.where(qa > 0, p_post > p_pre, p_post < p_pre)
        shrank = (p_post == p_pre) & (s_post < 0.5 * s_pre)
        sw = (moved | shrank) & good
        un = (~(moved | shrank)) & good
        if sw.sum() < 3 or un.sum() < 3:
            print(NA.format(t=a.ticker, d=date)); return

        # --- measurement windows, disjoint from the conditioning window ---
        b1 = BA.mid_at(bt, bm, ta + 1.0)
        b5 = BA.mid_at(bt, bm, ta + 5.0)
        f0 = BA.mid_at(bt, bm, ta + 180.0)
        f1 = BA.mid_at(bt, bm, ta + 180.0)
        f5 = BA.mid_at(bt, bm, ta + 185.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mkT = qa * (f0 - m0) / m0 * 1e4
            mk1 = qa * (f1 - b1) / b1 * 1e4
            mk5 = qa * (f5 - b5) / b5 * 1e4
            dep = np.where(good, (s_post - s_pre) / np.maximum(s_pre, 1.0) * 100.0, np.nan)

        vals = [int(agg.sum()), float(sw.sum()) / max(good.sum(), 1),
                wm(mkT[sw]), wm(mkT[un]),
                wm(mk1[sw]), wm(mk1[un]),
                wm(mk5[sw]), wm(mk5[un]),
                wm(dep[sw]), wm(dep[un]),
                int(sw.sum()), int(un.sum())]
        out = [a.ticker, str(date)]
        for v in vals:
            out.append(str(int(v)) if isinstance(v, (int, np.integer)) else "%.5f" % v)
        print(",".join(out))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
