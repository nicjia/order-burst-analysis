#!/usr/bin/env python3
"""
burst_zoo.py — exploration harness for alternative burst definitions.

Two design rules, both forced on us by earlier results:

  (1) FORMATION MUST NOT SEE PRICE. Array 14314701 showed that defining a burst as a run of
      same-side prints, where "side" comes from price against the contemporaneous midpoint,
      manufactures much of the footprint: a drifting mid puts prints on one side of itself.
      Every definition here forms clusters on arrival timing, size, or message type only.

  (2) PREFER NATIVE SIGNS. LOBSTER discloses Direction for type-1/2/3/4 messages; only type-5
      (hidden) prints are unsigned, which is where the whole signing problem came from. Where
      a definition uses visible messages we take the native sign and infer nothing.

Outcome for every definition: signed midpoint markout from CLUSTER END at 3 and 10 minutes,
plus a tradable figure `net3` = mk3 - 2*(half-spread at cluster end), i.e. what remains after
paying to cross in and out. A definition is only interesting if net3 > 0.

Definitions (10):
  d1 hidden_time    type-5, time clusters, signed by outside-pre-quote prints only
  d2 hidden_rate    type-5 clusters whose local arrival rate exceeds 3x the day's median
  d3 vis_time       type-4, time clusters, NATIVE direction majority
  d4 vis_big        type-4 time clusters with volume > 2x the day's median cluster volume
  d5 cancel         type-2/3 one-sided cancel clusters; sign = opposite the cancelled side
  d6 submit         type-1 one-sided submission clusters; sign = submitted side
  d7 mixed          clusters containing BOTH type-4 and type-5 within the window; native sign
  d8 accel          type-4 clusters whose inter-arrival gaps are shrinking
  d9 oddlot         type-4 odd-lot (non-multiple-of-100) clusters; native sign
  d10 block         single type-4 prints > 5x the day's median trade size; native sign

Output: ticker,date, then for each definition: n,mk3,mk10,net3
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
DEFS = ["d1_hidden_time", "d2_hidden_rate", "d3_vis_time", "d4_vis_big", "d5_cancel",
        "d6_submit", "d7_mixed", "d8_accel", "d9_oddlot", "d10_block"]
NCOL = 2 + 4 * len(DEFS)
NA = "{t},{d}" + ",nan" * (NCOL - 2)


def wm(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[np.abs(x) <= 1000]
    return float(np.mean(x)) if len(x) else np.nan


def time_clusters(t, minrun=3, gap=1.0):
    """Consecutive events with inter-arrival < gap; returns (start_idx, end_idx) pairs."""
    out = []
    i, n = 0, len(t)
    while i < n:
        j = i
        while j + 1 < n and (t[j + 1] - t[j]) < gap:
            j += 1
        if j - i + 1 >= minrun:
            out.append((i, j))
        i = j + 1
    return out


def emit(t, sign, slices, minsigned=2, consistency=0.0):
    """Cluster end times and net direction, requiring optional directional consistency."""
    ends, dirs = [], []
    for a, b in slices:
        seg = sign[a:b + 1]
        nz = seg[seg != 0]
        if len(nz) < minsigned:
            continue
        net = nz.sum()
        if net == 0:
            continue
        if consistency > 0 and abs(net) / len(nz) < consistency:
            continue
        ends.append(t[b]); dirs.append(int(np.sign(net)))
    return np.array(ends, float), np.array(dirs, int)


def score(bt, bm, bb, ba, ends, dirs):
    """Returns (n, mk3, mk10, net3) where net3 is mk3 less a round-trip spread cost."""
    if len(ends) < 3:
        return (0, np.nan, np.nan, np.nan)
    b0 = BA.mid_at(bt, bm, ends)
    p3 = BA.mid_at(bt, bm, ends + 180.0)
    p10 = BA.mid_at(bt, bm, ends + 600.0)
    lo, hi = BA.bbo_at(bt, bb, ba, ends)
    with np.errstate(invalid="ignore", divide="ignore"):
        mk3 = dirs * (p3 - b0) / b0 * 1e4
        mk10 = dirs * (p10 - b0) / b0 * 1e4
        halfsp = 0.5 * (hi - lo) / b0 * 1e4
        net3 = mk3 - 2.0 * halfsp
    return (len(ends), wm(mk3), wm(mk10), wm(net3))


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
        df = df[(df.t >= RTH0) & (df.t < RTH1 - 900.0)]
        if len(df) < 200 or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        T = df.t.to_numpy(float); TY = df.ty.to_numpy(int)
        SZ = df.sz.to_numpy(float); PX = df.px.to_numpy(float) / BA.SCALE
        DR = df.dr.to_numpy(int)
        res = {}

        # ---------- hidden (type 5): unsigned in the feed, so sign outside the pre-quote ----
        hm = TY == 5
        th, ph = T[hm], PX[hm]
        if len(th) >= 10:
            pb, pa = BA.bbo_at(bt, bb, ba, th - 1e-3)
            sh = np.zeros(len(th), int)
            sh[np.isfinite(pa) & (ph > pa)] = 1
            sh[np.isfinite(pb) & (ph < pb)] = -1
            sl = time_clusters(th)
            res["d1_hidden_time"] = score(bt, bm, bb, ba, *emit(th, sh, sl))
            # d2: keep only clusters whose local rate beats 3x the day's median gap rate
            if sl:
                med_gap = np.median(np.diff(th)) if len(th) > 2 else 1.0
                fast = [(x, y) for (x, y) in sl
                        if (th[y] - th[x]) / max(y - x, 1) < med_gap / 3.0]
                res["d2_hidden_rate"] = score(bt, bm, bb, ba, *emit(th, sh, fast))

        # ---------- visible (type 4): NATIVE direction, no inference ----------
        vm = TY == 4
        tv, dv, sv, pv = T[vm], DR[vm], SZ[vm], PX[vm]
        # LOBSTER Direction on an execution is the RESTING order's side; the aggressor is
        # the opposite, so a hit on a resting sell order (-1) is an aggressive BUY.
        agg = -dv
        if len(tv) >= 20:
            slv = time_clusters(tv, minrun=5)
            res["d3_vis_time"] = score(bt, bm, bb, ba, *emit(tv, agg, slv, consistency=0.7))
            if slv:
                vols = np.array([sv[x:y + 1].sum() for (x, y) in slv])
                thr = 2.0 * np.median(vols) if len(vols) else np.inf
                big = [s for s, v in zip(slv, vols) if v > thr]
                res["d4_vis_big"] = score(bt, bm, bb, ba, *emit(tv, agg, big, consistency=0.7))
                # d8: accelerating clusters (second half faster than first half)
                acc = []
                for (x, y) in slv:
                    if y - x < 5:
                        continue
                    g = np.diff(tv[x:y + 1]); h = len(g) // 2
                    if h and g[h:].mean() < g[:h].mean():
                        acc.append((x, y))
                res["d8_accel"] = score(bt, bm, bb, ba, *emit(tv, agg, acc, consistency=0.7))
            # d9: odd lots
            odd = (sv % 100) != 0
            if odd.sum() >= 20:
                to, ao = tv[odd], agg[odd]
                res["d9_oddlot"] = score(bt, bm, bb, ba,
                                         *emit(to, ao, time_clusters(to, minrun=5), consistency=0.7))
            # d10: single block prints
            med = np.median(sv)
            blk = sv > 5.0 * med
            if blk.sum() >= 5:
                res["d10_block"] = score(bt, bm, bb, ba, tv[blk], agg[blk])

        # ---------- cancels (2/3) and submissions (1): native side ----------
        cm = (TY == 2) | (TY == 3)
        tc, dc = T[cm], DR[cm]
        if len(tc) >= 20:
            # cancelling resting asks (-1) removes supply -> bullish, so sign = -DR
            res["d5_cancel"] = score(bt, bm, bb, ba,
                                     *emit(tc, -dc, time_clusters(tc, minrun=5), consistency=0.8))
        sm = TY == 1
        ts, ds = T[sm], DR[sm]
        if len(ts) >= 20:
            res["d6_submit"] = score(bt, bm, bb, ba,
                                     *emit(ts, ds, time_clusters(ts, minrun=5), consistency=0.8))

        # ---------- d7: hidden and visible executions co-occurring ----------
        em = (TY == 4) | (TY == 5)
        te, tye, dre = T[em], TY[em], DR[em]
        if len(te) >= 20:
            sle = time_clusters(te, minrun=4)
            mixed = [(x, y) for (x, y) in sle
                     if (tye[x:y + 1] == 4).any() and (tye[x:y + 1] == 5).any()]
            se = np.where(tye == 4, -dre, 0)     # sign only from the visible leg
            res["d7_mixed"] = score(bt, bm, bb, ba, *emit(te, se, mixed, consistency=0.7))

        out = [a.ticker, str(date)]
        for k in DEFS:
            n, m3, m10, n3 = res.get(k, (0, np.nan, np.nan, np.nan))
            out += [str(int(n)), "%.5f" % m3, "%.5f" % m10, "%.5f" % n3]
        print(",".join(out))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
