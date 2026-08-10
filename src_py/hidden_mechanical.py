#!/usr/bin/env python3
"""
hidden_mechanical.py — does the permanent footprint survive the mechanical
quote-displacement account? Two discriminating tests on aggressive-hidden prints, plus
the hidden/visible-OFI correlation.

(A) BASELINE DISPLACEMENT. A queue-depletion account says the print consumes the inside
    level, the mid relocates within milliseconds, and the "permanent" footprint is that
    one-off relocation never refilled. An informational account says price accrues over
    minutes. So we measure the markout from displaced baselines: from m(t+1s), m(t+5s),
    m(t+30s) to m(t+180s), not only from m(t). If the footprint is quote relocation it
    is impounded almost entirely by t+1s and the residual accrual is ~0; if it is
    information it keeps accruing after the book has re-settled.

(B) REFILL CONDITIONING. Split prints by whether the consumed side replenishes: for a buy,
    whether the best ask returns to (or inside) its pre-print level within N seconds.
    Mechanical displacement should vanish in the refilled subset by construction --
    the quote came back. Informational impact should not.

Both are reported on all aggressive prints and on the outside-the-quote subset alone,
where the aggressor sign cannot be wrong.

(C) corr(signed hidden volume, visible CKS OFI) on a 10s clock -- to document the
    suppression that makes the hidden coefficient RISE under the OFI control.

Output: ticker,date,n_agg,n_far,
        acc1,acc5,acc30,acc180,          (markout from m(t) to t+{1,5,30,180}s, all agg)
        res5,res30,                      (markout from m(t+5s)/m(t+30s) to t+180s, all agg)
        far180,far_res5,far_res30,       (same, outside-quote prints only)
        n_rf,n_nrf,rf180,nrf180,         (refilled vs not, markout t->t+180)
        corr_ofi                         (10s-bin corr of hidden flow with visible OFI)
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
REFILL_N = 10.0          # seconds allowed for the consumed level to replenish
DELTA = 10.0             # bin width for the OFI correlation
NCOL = 18
NA = "{t},{d}" + ",nan" * (NCOL - 2)


def wm(x):
    """winsorized mean, matching the convention used by the other hidden extractors"""
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
        keep = (ht >= RTH0) & (ht < RTH1)
        ht, hpx, hsz = ht[keep], hpx[keep], hsz[keep]
        if len(ht) < 3:
            print(NA.format(t=a.ticker, d=date)); return

        m0 = BA.mid_at(bt, bm, ht)
        mids = {k: BA.mid_at(bt, bm, ht + k) for k in (1.0, 5.0, 30.0, 180.0)}
        pbid, pask = BA.bbo_at(bt, bb, ba, ht - 1e-3)   # quote just BEFORE the print
        rbid, rask = BA.bbo_at(bt, bb, ba, ht + REFILL_N)

        ok = np.isfinite(m0) & (m0 > 0) & np.isfinite(pbid) & np.isfinite(pask) & (pask > pbid)
        for v in mids.values():
            ok &= np.isfinite(v) & (v > 0)
        if ok.sum() < 3:
            print(NA.format(t=a.ticker, d=date)); return
        ht, hpx, m0 = ht[ok], hpx[ok], m0[ok]
        M = {k: v[ok] for k, v in mids.items()}
        pbid, pask, rbid, rask = pbid[ok], pask[ok], rbid[ok], rask[ok]

        q = np.where(hpx > m0, 1.0, np.where(hpx < m0, -1.0, 0.0))
        agg = q != 0
        if agg.sum() < 3:
            print(NA.format(t=a.ticker, d=date)); return
        A = lambda x: x[agg]
        hpxA, m0A, qA = A(hpx), A(m0), A(q)
        MA = {k: A(v) for k, v in M.items()}
        pbidA, paskA, rbidA, raskA = A(pbid), A(pask), A(rbid), A(rask)

        mk = lambda base, fwd: qA * (fwd - base) / base * 1e4
        acc1, acc5, acc30, acc180 = (mk(m0A, MA[k]) for k in (1.0, 5.0, 30.0, 180.0))
        res5 = mk(MA[5.0], MA[180.0])       # accrual AFTER the book has re-settled 5s
        res30 = mk(MA[30.0], MA[180.0])

        # outside-the-quote subset: aggressor sign unambiguous
        far = (hpxA > paskA) | (hpxA < pbidA)
        F = lambda x: x[far]
        far180 = mk(m0A, MA[180.0])[far]
        far_res5, far_res30 = res5[far], res30[far]

        # refill: did the consumed side come back within REFILL_N seconds?
        # a buy consumes the ask -> refilled if the ask is back at/inside its pre-print level
        okr = np.isfinite(raskA) & np.isfinite(rbidA)
        refilled = np.where(qA > 0, raskA <= paskA, rbidA >= pbidA) & okr
        nrf = okr & ~refilled
        rf180 = acc180[refilled]; nrf180 = acc180[nrf]

        # (C) hidden signed volume vs visible OFI on a 10s clock
        edges = np.arange(RTH0, RTH1 + DELTA, DELTA); nb = len(edges) - 1
        hflow = np.zeros(nb)
        bi = np.clip(((ht[agg] - RTH0) / DELTA).astype(int), 0, nb - 1)
        np.add.at(hflow, bi, qA * A(hsz[ok]))
        oflow = np.zeros(nb)
        for sec, v in ofi.items():
            if RTH0 <= sec < RTH1:
                oflow[int((sec - RTH0) / DELTA)] += v
        c = np.nan
        if np.std(hflow) > 1e-9 and np.std(oflow) > 1e-9:
            c = float(np.corrcoef(hflow, oflow)[0, 1])

        vals = [wm(acc1), wm(acc5), wm(acc30), wm(acc180), wm(res5), wm(res30),
                wm(far180), wm(far_res5), wm(far_res30)]
        print("{},{},{},{},".format(a.ticker, date, int(agg.sum()), int(far.sum()))
              + ",".join("%.5f" % v for v in vals)
              + ",{},{},{:.5f},{:.5f},{:.5f}".format(
                  int(refilled.sum()), int(nrf.sum()), wm(rf180), wm(nrf180), c))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
