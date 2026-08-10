#!/usr/bin/env python3
"""
hidden_depletion.py — a NON-CIRCULAR test of the mechanical quote-displacement account.

The earlier refill test conditioned on whether the quote came back, which is a statement
about the same price path as the outcome; it was circular and is discarded. This test
conditions only on quantities known STRICTLY BEFORE the print:

    depletion ratio  d = (trade size) / (displayed depth at the touch, pre-trade)

The mechanical account says the footprint is queue depletion that never refills, so impact
must scale with the fraction of the visible queue consumed, and a print taking a small
slice of a deep queue cannot displace the mid at all. The informational account says impact
reflects what the trade reveals, which need not scale with d -- an informed trader taking
5% of the queue still causes belief revision.

The discriminating cell is the LOW-d subset: if prints consuming <10% of displayed depth
still carry a permanent footprint, mechanical displacement cannot be the explanation.

We also record pre-trade depth and spread so the markout can be conditioned on book state
without touching the realized path.

Output: ticker,date,n_agg,
        mk_all,                            markout t->t+180s, all aggressive
        mk_d1..mk_d5,                      markout by pre-trade depletion-ratio quintile
        n_d1..n_d5,
        mk_lo,n_lo,                        d < 0.10  (small slice of a deep queue)
        mk_hi,n_hi,                        d >= 1.0  (consumed the whole visible level)
        med_d,med_depth                    median depletion ratio, median touch depth
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
NCOL = 20
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
        if len(h) < 3 or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE
        hsz = h.sz.to_numpy(float)
        k = (ht >= RTH0) & (ht < RTH1)
        ht, hpx, hsz = ht[k], hpx[k], hsz[k]
        if len(ht) < 3:
            print(NA.format(t=a.ticker, d=date)); return

        m0 = BA.mid_at(bt, bm, ht)
        m180 = BA.mid_at(bt, bm, ht + 180.0)
        # book state strictly BEFORE the print
        pb, pa = BA.bbo_at(bt, bb, ba, ht - 1e-3)
        pbs, pas = BA.bbo_at(bt, bbsz, basz, ht - 1e-3)

        ok = (np.isfinite(m0) & (m0 > 0) & np.isfinite(m180) & (m180 > 0)
              & np.isfinite(pb) & np.isfinite(pa) & (pa > pb)
              & np.isfinite(pbs) & np.isfinite(pas) & (pbs > 0) & (pas > 0))
        if ok.sum() < 3:
            print(NA.format(t=a.ticker, d=date)); return
        hpx, hsz, m0, m180 = hpx[ok], hsz[ok], m0[ok], m180[ok]
        pb, pa, pbs, pas = pb[ok], pa[ok], pbs[ok], pas[ok]

        q = np.where(hpx > m0, 1.0, np.where(hpx < m0, -1.0, 0.0))
        agg = q != 0
        if agg.sum() < 5:
            print(NA.format(t=a.ticker, d=date)); return
        A = lambda x: x[agg]
        q, hsz, m0, m180 = A(q), A(hsz), A(m0), A(m180)
        pbs, pas = A(pbs), A(pas)

        mk = q * (m180 - m0) / m0 * 1e4
        depth = np.where(q > 0, pas, pbs)          # depth on the side being consumed
        d = hsz / np.maximum(depth, 1.0)           # pre-trade depletion ratio

        vals = [wm(mk)]
        good = np.isfinite(d) & np.isfinite(mk)
        if good.sum() >= 5:
            try:
                qt = pd.qcut(d[good], 5, labels=False, duplicates="drop")
            except Exception:
                qt = np.zeros(int(good.sum()), int)
            mkg = mk[good]
            for i in range(5):
                sel = qt == i
                vals.append(wm(mkg[sel]) if sel.sum() else np.nan)
            ns = [int((qt == i).sum()) for i in range(5)]
        else:
            vals += [np.nan] * 5; ns = [0] * 5

        lo = good & (d < 0.10); hi = good & (d >= 1.0)
        print("{},{},{},".format(a.ticker, date, int(agg.sum()))
              + ",".join("%.5f" % v for v in vals) + ","
              + ",".join(str(x) for x in ns)
              + ",{:.5f},{},{:.5f},{},{:.5f},{:.1f}".format(
                  wm(mk[lo]), int(lo.sum()), wm(mk[hi]), int(hi.sum()),
                  float(np.nanmedian(d[good])) if good.sum() else np.nan,
                  float(np.nanmedian(depth[good])) if good.sum() else np.nan))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
