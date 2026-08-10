#!/usr/bin/env python3
"""
hidden_preprint.py — does the burst move the price, or follow a price already moving?

The reverse-causality objection: a detector conditioned on aggressive same-side executions
may select moments when the mid is ALREADY travelling in that direction, so the forward
markout measures continuation of a pre-existing move rather than impact of the flow. The
re-estimated Hasbrouck VAR (which conditions on 30 lags of returns and finds no lasting
response) is consistent with exactly this.

The direct test is the event-time profile on BOTH sides of the print. For each aggressive
hidden execution we measure the signed midpoint move over the windows ENDING at the print
(-180s, -30s, -5s, -1s -> t) as well as the forward markout. If the footprint is impact,
pre-print drift should be small relative to the forward move; if it is flow chasing, price
should already have run in the trade's direction before it arrives.

We report the whole profile on all aggressive prints and on outside-the-quote prints alone,
where the aggressor sign cannot be wrong.

Output: ticker,date,n_agg,n_far,
        pre180,pre30,pre5,pre1,           signed mid move INTO the print (bps)
        post180,                          forward markout from t
        f_pre30,f_pre5,f_post180          same, outside-quote prints only
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
        if len(h) < 3 or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE
        # keep prints with a full 180s of history inside RTH so pre-windows are well defined
        k = (ht >= RTH0 + 180.0) & (ht < RTH1)
        ht, hpx = ht[k], hpx[k]
        if len(ht) < 5:
            print(NA.format(t=a.ticker, d=date)); return

        m0 = BA.mid_at(bt, bm, ht)
        pre = {w: BA.mid_at(bt, bm, ht - w) for w in (180.0, 30.0, 5.0, 1.0)}
        fwd = BA.mid_at(bt, bm, ht + 180.0)
        pb, pa = BA.bbo_at(bt, bb, ba, ht - 1e-3)

        ok = np.isfinite(m0) & (m0 > 0) & np.isfinite(fwd) & (fwd > 0) & np.isfinite(pb) & np.isfinite(pa)
        for v in pre.values():
            ok &= np.isfinite(v) & (v > 0)
        if ok.sum() < 5:
            print(NA.format(t=a.ticker, d=date)); return
        hpx, m0, fwd = hpx[ok], m0[ok], fwd[ok]
        pre = {w: v[ok] for w, v in pre.items()}
        pb, pa = pb[ok], pa[ok]

        q = np.where(hpx > m0, 1.0, np.where(hpx < m0, -1.0, 0.0))
        agg = q != 0
        if agg.sum() < 5:
            print(NA.format(t=a.ticker, d=date)); return
        A = lambda x: x[agg]
        q, hpx, m0, fwd = A(q), A(hpx), A(m0), A(fwd)
        pre = {w: A(v) for w, v in pre.items()}
        pb, pa = A(pb), A(pa)

        # signed move INTO the print: from the mid w seconds earlier up to the print mid
        into = {w: q * (m0 - pre[w]) / pre[w] * 1e4 for w in pre}
        post = q * (fwd - m0) / m0 * 1e4
        far = (hpx > pa) | (hpx < pb)

        vals = [wm(into[180.0]), wm(into[30.0]), wm(into[5.0]), wm(into[1.0]), wm(post),
                wm(into[30.0][far]), wm(into[5.0][far]), wm(post[far])]
        print("{},{},{},{},".format(a.ticker, date, int(agg.sum()), int(far.sum()))
              + ",".join("%.5f" % v for v in vals))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
