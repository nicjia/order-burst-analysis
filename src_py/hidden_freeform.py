#!/usr/bin/env python3
"""
hidden_freeform.py — is the footprint an artifact of price-conditioned burst FORMATION?

The published construction signs each hidden print by its price against the CONTEMPORANEOUS
midpoint, and then defines a burst as a run of >=3 consecutive SAME-SIGN prints within 1s
gaps. Sign therefore enters formation: if the midpoint is drifting, prints land mechanically
on one side of it and a "burst" is manufactured by the midpoint's own motion rather than by
any clustering of trader intent. The markout is then measured against that same midpoint
series. That is a genuine circularity and it has not been isolated -- the existing signing
robustness checks vary how prints are SIGNED, not how bursts are FORMED.

We break the loop by forming clusters on TIME ALONE and signing them with information that
cannot have been moved by the prints in question.

  ARM A (baseline)     sign by contemporaneous mid; runs of same-sign prints.  As published.
  ARM B (time-formed)  cluster by inter-arrival gap only, no sign condition; then sign each
                       print against the mid 1ms BEFORE it, and take the cluster direction as
                       the sign of the net. Formation is price-free; signing uses a midpoint
                       the print cannot have displaced.
  ARM C (strict)       cluster by inter-arrival gap only; sign ONLY prints that execute
                       outside the PRE-print quote, where the aggressor's side is unambiguous
                       whatever the midpoint does. Cluster direction is the sign of the net.
                       Formation is price-free and signing is quote-free of the print itself.

All three measure the same outcome: the directional midpoint markout from the CLUSTER-END mid
forward 3/15/30 minutes. If the footprint survives B and C, price-conditioned formation is not
what produces it. If it collapses, it is.

Output: ticker,date,nA,mkA3,mkA15,mkA30,nB,mkB3,mkB15,mkB30,nC,mkC3,mkC15,mkC30
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
NCOL = 14
NA = "{t},{d}" + ",nan" * (NCOL - 2)
GAP, MINRUN = 1.0, 3


def wm(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[np.abs(x) <= 1000]
    return float(np.mean(x)) if len(x) else np.nan


def runs_same_sign(ht, sign, minrun=MINRUN, gap=GAP):
    """ARM A: published rule -- runs of consecutive identical nonzero signs."""
    nz = sign != 0
    t2, s2 = ht[nz], sign[nz]
    ends, dirs = [], []
    i, n = 0, len(t2)
    while i < n:
        j = i
        while j + 1 < n and s2[j + 1] == s2[i] and (t2[j + 1] - t2[j]) < gap:
            j += 1
        if j - i + 1 >= minrun:
            ends.append(t2[j]); dirs.append(int(s2[i]))
        i = j + 1
    return np.array(ends, float), np.array(dirs, int)


def clusters_by_time(ht, minrun=MINRUN, gap=GAP):
    """ARMS B/C: group consecutive prints on inter-arrival gap alone. Returns index slices."""
    out = []
    i, n = 0, len(ht)
    while i < n:
        j = i
        while j + 1 < n and (ht[j + 1] - ht[j]) < gap:
            j += 1
        if j - i + 1 >= minrun:
            out.append((i, j))
        i = j + 1
    return out


def direct_clusters(ht, sign, slices, minsigned=2):
    """Cluster direction = sign of the net of its signed prints."""
    ends, dirs = [], []
    for a, b in slices:
        seg = sign[a:b + 1]
        nz = int((seg != 0).sum())
        if nz < minsigned:
            continue
        net = int(np.sign(seg.sum()))
        if net == 0:
            continue
        ends.append(ht[b]); dirs.append(net)
    return np.array(ends, float), np.array(dirs, int)


def markout(bt, bm, ends, dirs, dt):
    if len(ends) == 0:
        return np.nan
    b0 = BA.mid_at(bt, bm, ends)
    e = BA.mid_at(bt, bm, ends + dt)
    return wm(dirs * (e - b0) / b0 * 1e4)


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

        mid_now = BA.mid_at(bt, bm, ht)                      # contemporaneous
        mid_pre = BA.mid_at(bt, bm, ht - 1e-3)               # 1ms before the print
        pb, pa = BA.bbo_at(bt, bb, ba, ht - 1e-3)            # quote before the print
        ok = np.isfinite(mid_now) & (mid_now > 0) & np.isfinite(mid_pre) & (mid_pre > 0)
        ht, hpx, mid_now, mid_pre, pb, pa = (v[ok] for v in (ht, hpx, mid_now, mid_pre, pb, pa))
        if len(ht) < 10:
            print(NA.format(t=a.ticker, d=date)); return

        # ---- ARM A: published construction ----
        sA = np.where(hpx > mid_now, 1, np.where(hpx < mid_now, -1, 0))
        eA, dA = runs_same_sign(ht, sA)

        slices = clusters_by_time(ht)

        # ---- ARM B: time-formed, signed on the pre-print midpoint ----
        sB = np.where(hpx > mid_pre, 1, np.where(hpx < mid_pre, -1, 0))
        eB, dB = direct_clusters(ht, sB, slices, minsigned=2)

        # ---- ARM C: time-formed, signed only outside the pre-print quote ----
        sC = np.zeros(len(ht), int)
        far_up = np.isfinite(pa) & (hpx > pa)
        far_dn = np.isfinite(pb) & (hpx < pb)
        sC[far_up] = 1; sC[far_dn] = -1
        eC, dC = direct_clusters(ht, sC, slices, minsigned=2)

        vals = []
        for e, d in ((eA, dA), (eB, dB), (eC, dC)):
            vals.append(len(e))
            for dt in (180.0, 900.0, 1800.0):
                vals.append(markout(bt, bm, e, d, dt))
        out = [a.ticker, str(date)]
        for v in vals:
            out.append(str(int(v)) if isinstance(v, (int, np.integer)) else "%.5f" % v)
        print(",".join(out))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
