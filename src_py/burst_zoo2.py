#!/usr/bin/env python3
"""
burst_zoo2.py — iteration 2: does the block-print signal scale with the spread?

Iteration 1 (10 definitions, 10 dev names) produced exactly one standout: single large
visible executions ("blocks"), natively signed, +4.10 bps at three minutes and flat to ten.
Every definition was negative net of a round-trip spread cost, but d10's shortfall implies a
~5.8 bps half-spread on a mid-cap dev set, and the same signal on a 1.5 bps half-spread name
would clear the cost.

Whether that is a real opportunity or an arithmetic mirage depends on ONE thing. If the block
markout is a roughly constant fraction of the spread -- as the hidden footprint was, at about
a third of the half-spread -- then net = (1/3)h - 2h < 0 at every spread and the signal is
dead everywhere by construction. If instead it is roughly constant in BPS, there is a spread
threshold below which it is capturable.

So we emit the day's half-spread alongside the markouts and let the aggregation stratify,
rather than pre-binning. We also sweep the size threshold (a block at 3x median is a different
object from one at 20x) and extend the horizon to 30 minutes, since d10 was still flat at 10.

Variants, all natively signed, none price-conditioned in formation:
    k3,k5,k10,k20   single visible prints exceeding k x the day's median trade size
    aggr            blocks executing at or through the pre-print touch
    repeat          blocks followed by another same-direction block within 30s

Output: ticker,date,halfsp,medsz, then for each variant: n,mk3,mk10,mk30
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
VAR = ["k3", "k5", "k10", "k20", "aggr", "repeat"]
NCOL = 4 + 4 * len(VAR)
NA = "{t},{d}" + ",nan" * (NCOL - 2)


def wm(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[np.abs(x) <= 1000]
    return float(np.mean(x)) if len(x) else np.nan


def score(bt, bm, ends, dirs):
    if len(ends) < 3:
        return (0, np.nan, np.nan, np.nan)
    b0 = BA.mid_at(bt, bm, ends)
    out = [len(ends)]
    for dt in (180.0, 600.0, 1800.0):
        p = BA.mid_at(bt, bm, ends + dt)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(wm(dirs * (p - b0) / b0 * 1e4))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    a = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(a.msg))
    date = int(m.group(1) + m.group(2) + m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(a.msg)
        df = pd.read_csv(a.msg, header=None, usecols=[0, 1, 3, 4, 5],
                         names=["t", "ty", "sz", "px", "dr"])
        df = df[(df.t >= RTH0) & (df.t < RTH1 - 1800.0)]
        v = df[df.ty == 4]
        if len(v) < 50 or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        tv = v.t.to_numpy(float); sv = v.sz.to_numpy(float)
        pv = v.px.to_numpy(float) / BA.SCALE
        agg = -v.dr.to_numpy(int)          # aggressor is opposite the resting order's side

        # day-level context: mean half-spread in bps, sampled on a 1-minute grid
        grid = np.arange(RTH0, RTH1 - 1800.0, 60.0)
        gb, gaq = BA.bbo_at(bt, bb, ba, grid)
        gm = BA.mid_at(bt, bm, grid)
        with np.errstate(invalid="ignore", divide="ignore"):
            hs = 0.5 * (gaq - gb) / gm * 1e4
        halfsp = wm(hs)
        medsz = float(np.median(sv))
        if not np.isfinite(halfsp) or halfsp <= 0:
            print(NA.format(t=a.ticker, d=date)); return

        res = {}
        for k, lab in ((3.0, "k3"), (5.0, "k5"), (10.0, "k10"), (20.0, "k20")):
            sel = sv > k * medsz
            if sel.sum() >= 3:
                res[lab] = score(bt, bm, tv[sel], agg[sel])

        blk = sv > 5.0 * medsz
        if blk.sum() >= 3:
            tb, ab, pb_ = tv[blk], agg[blk], pv[blk]
            lo, hi = BA.bbo_at(bt, bb, ba, tb - 1e-3)
            # aggressive: a buy block lifting at/through the ask, a sell hitting at/below bid
            at_touch = np.where(ab > 0, pb_ >= hi, pb_ <= lo)
            at_touch &= np.isfinite(lo) & np.isfinite(hi)
            if at_touch.sum() >= 3:
                res["aggr"] = score(bt, bm, tb[at_touch], ab[at_touch])
            # repeat: another same-direction block within 30s
            rep = np.zeros(len(tb), bool)
            for i in range(len(tb) - 1):
                j = i + 1
                while j < len(tb) and tb[j] - tb[i] <= 30.0:
                    if ab[j] == ab[i]:
                        rep[i] = True; break
                    j += 1
            if rep.sum() >= 3:
                res["repeat"] = score(bt, bm, tb[rep], ab[rep])

        out = [a.ticker, str(date), "%.5f" % halfsp, "%.1f" % medsz]
        for k in VAR:
            n, m3, m10, m30 = res.get(k, (0, np.nan, np.nan, np.nan))
            out += [str(int(n)), "%.5f" % m3, "%.5f" % m10, "%.5f" % m30]
        print(",".join(out))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
