#!/usr/bin/env python3
"""
hidden_tickplacebo.py — two referee items on one pass over the message stream.

ITEM 1 (threatens the central table). The at-midpoint leg of the bifurcation is signed by
the tick rule, which labels a print a buy when the price just ticked up. Short-horizon mean
reversion of that preceding tick then manufactures a negative forward markout for ANY set of
moments signed this way -- no liquidity-provision economics required. We test it two ways:

  (a) MATCHED PLACEBO. Take the at-midpoint print times and shift them ALL by one random
      per-day offset. Inter-arrival structure (hence the burst clustering) is preserved
      exactly, but the moments are decoupled from actual hidden midpoint executions. Sign
      the resulting midpoint series by the same tick rule, cluster identically, and measure
      the same markouts. If the placebo also earns ~-0.4 bps, the at-mid row is an artifact
      of tick-rule conditioning rather than evidence of uninformed liquidity provision.

  (b) TICK-CONDITIONING ON THE AGGRESSIVE LEG. Re-sign the aggressive (away-from-mid)
      prints by the PURE tick rule with no quote-rule seed, and compare to their quote-rule
      markout. This isolates what tick conditioning alone does to a population we know is
      informed.

ITEM 4 (the markout/VAR reconciliation, currently verbal). Decompose the forward markout
into the part explained by pre-print drift and the part orthogonal to it, on the SAME panel
and in the SAME units, so the arithmetic closes. Per name-day we regress the signed forward
markout on the signed pre-print drift over the preceding 30s,

    m_h = a_h + b_h * d ,      hence      mean(m_h) = a_h + b_h * mean(d)

exactly within each name-day. a_h is the component orthogonal to the prior path -- the
quantity the VAR estimates -- and b_h * mean(d) is continuation of a move already underway.
Run at 3, 10 and 30 minutes so the profile of a_h can be compared with the VAR's sign
change. Restricted to prints executing OUTSIDE the prevailing quote, where the aggressor's
side cannot be misassigned, with the all-aggressive version reported alongside.

Output: ticker,date,
        n_agg,n_mid,n_far,
        mk3_agg_qd,mk3_agg_tick,                      item 1b
        n_mb,mk3_mid,mk15_mid,mk30_mid,               at-mid baseline (burst level)
        n_pb,mk3_plc,mk15_plc,mk30_plc,               item 1a placebo (burst level)
        dpre,m3,m10,m30,a3,a10,a30,e3,e10,e30         item 4, outside-quote prints
        dpreA,m3A,m10A,m30A,a3A,a10A,a30A,e3A,e10A,e30A   item 4, all aggressive prints
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
NCOL = 35
NA = "{t},{d}" + ",nan" * (NCOL - 2)


def wm(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[np.abs(x) <= 1000]
    return float(np.mean(x)) if len(x) else np.nan


def tick_sign(px, seed):
    """Tick rule; a non-zero seed entry keeps its firm sign. Matches hidden_emo_clnv.py."""
    s = np.zeros(len(px), int); last = np.nan; lastsign = 0
    for k in range(len(px)):
        if seed[k] != 0:
            s[k] = seed[k]
        elif np.isfinite(last):
            if px[k] > last: s[k] = 1
            elif px[k] < last: s[k] = -1
            else: s[k] = lastsign
        if np.isfinite(px[k]) and (not np.isfinite(last) or px[k] != last):
            last = px[k]
        if s[k] != 0: lastsign = s[k]
    return s


def bursts_from(ht, sign, minrun=3, gap=1.0):
    """Identical clustering to the published tables."""
    nz = sign != 0
    ht2, sign2 = ht[nz], sign[nz]
    ends, dirs = [], []
    n = len(sign2); i = 0
    while i < n:
        j = i
        while j + 1 < n and sign2[j + 1] == sign2[i] and (ht2[j + 1] - ht2[j]) < gap:
            j += 1
        if j - i + 1 >= minrun:
            ends.append(ht2[j]); dirs.append(int(sign2[i]))
        i = j + 1
    return np.array(ends, float), np.array(dirs, int)


def burst_markout(bt, bm, ends, dirs, dt):
    if len(ends) == 0:
        return np.nan
    b0 = BA.mid_at(bt, bm, ends)
    e = BA.mid_at(bt, bm, ends + dt)
    return wm(dirs * (e - b0) / b0 * 1e4)


def decompose(d, m):
    """OLS m = a + b*d on finite, outlier-trimmed pairs. Returns (a, b*mean(d), mean(m), mean(d))."""
    k = np.isfinite(d) & np.isfinite(m) & (np.abs(d) <= 1000) & (np.abs(m) <= 1000)
    d, m = d[k], m[k]
    if len(d) < 10 or np.std(d) < 1e-9:
        return (np.nan,) * 4
    db, mb = d.mean(), m.mean()
    b = float(((d - db) * (m - mb)).sum() / ((d - db) ** 2).sum())
    a = float(mb - b * db)
    return a, float(b * db), float(mb), float(db)


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
        k = (ht >= RTH0 + 180.0) & (ht < RTH1)
        ht, hpx = ht[k], hpx[k]
        if len(ht) < 5:
            print(NA.format(t=a.ticker, d=date)); return

        mid = BA.mid_at(bt, bm, ht)
        # Quote strictly BEFORE the print. Taking the BBO at ht risks reading a quote the
        # print itself has already moved, which makes "outside the quote" endogenous and
        # selects a different, far smaller population. Matches hidden_preprint.py.
        pb, pa = BA.bbo_at(bt, bb, ba, ht - 1e-3)
        ok = np.isfinite(mid) & (mid > 0) & np.isfinite(pb) & np.isfinite(pa) & (pa > pb)
        ht, hpx, mid, pb, pa = ht[ok], hpx[ok], mid[ok], pb[ok], pa[ok]
        if len(ht) < 5:
            print(NA.format(t=a.ticker, d=date)); return

        qd = np.where(hpx > mid, 1, np.where(hpx < mid, -1, 0))
        agg = qd != 0
        amid = ~agg
        n_agg, n_mid = int(agg.sum()), int(amid.sum())

        # ---- item 1b: aggressive prints, quote sign vs PURE tick sign (burst level) ----
        mk3_qd = mk3_tk = np.nan
        if n_agg >= 5:
            e1, d1 = bursts_from(ht[agg], qd[agg])
            mk3_qd = burst_markout(bt, bm, e1, d1, 180.0)
            ts = tick_sign(hpx[agg], np.zeros(n_agg, int))
            e2, d2 = bursts_from(ht[agg], ts)
            mk3_tk = burst_markout(bt, bm, e2, d2, 180.0)

        # ---- at-midpoint baseline, tick-signed (reproduces the published row) ----
        n_mb = 0; mk3_m = mk15_m = mk30_m = np.nan
        if n_mid >= 5:
            ms = tick_sign(hpx[amid], np.zeros(n_mid, int))
            em, dm = bursts_from(ht[amid], ms)
            n_mb = len(em)
            mk3_m = burst_markout(bt, bm, em, dm, 180.0)
            mk15_m = burst_markout(bt, bm, em, dm, 900.0)
            mk30_m = burst_markout(bt, bm, em, dm, 1800.0)

        # ---- item 1a: matched placebo. Same inter-arrival structure, decoupled moments. ----
        n_pb = 0; mk3_p = mk15_p = mk30_p = np.nan
        if n_mid >= 5:
            rng = np.random.default_rng(abs(hash((a.ticker, date))) % (2 ** 32))
            tm = ht[amid]
            span = RTH1 - 1800.0 - (RTH0 + 180.0)
            best = None
            for _ in range(6):                     # a few draws; keep the first that fits RTH
                sh = rng.uniform(600.0, 3600.0) * rng.choice([-1.0, 1.0])
                cand = tm + sh
                if cand[0] >= RTH0 + 180.0 and cand[-1] < RTH1 - 1800.0:
                    best = cand; break
            if best is None:                        # fall back: wrap into the session interior
                sh = rng.uniform(600.0, 3600.0)
                best = RTH0 + 180.0 + np.mod(tm - RTH0 - 180.0 + sh, max(span, 1.0))
                best = np.sort(best)
            pmid = BA.mid_at(bt, bm, best)
            good = np.isfinite(pmid) & (pmid > 0)
            if good.sum() >= 5:
                bp, pmid2 = best[good], pmid[good]
                ps = tick_sign(pmid2, np.zeros(len(pmid2), int))
                ep, dp = bursts_from(bp, ps)
                n_pb = len(ep)
                mk3_p = burst_markout(bt, bm, ep, dp, 180.0)
                mk15_p = burst_markout(bt, bm, ep, dp, 900.0)
                mk30_p = burst_markout(bt, bm, ep, dp, 1800.0)

        # ---- item 4: pre-drift decomposition ----
        # Reported twice: on outside-quote prints, where the aggressor's side cannot be
        # misassigned and the pre-drift is therefore clean, and on all aggressive prints,
        # whose pre-drift is contaminated by contemporaneous-mid sign assignment and is
        # carried only for comparison.
        def decomp_block(selector):
            n = int(selector.sum())
            if n < 10:
                return (np.nan,) * 10
            tf, qf, mf = ht[selector], qd[selector], mid[selector]
            p30 = BA.mid_at(bt, bm, tf - 30.0)
            f3 = BA.mid_at(bt, bm, tf + 180.0)
            f10 = BA.mid_at(bt, bm, tf + 600.0)
            f30 = BA.mid_at(bt, bm, tf + 1800.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                d = qf * (mf - p30) / p30 * 1e4
                r3 = qf * (f3 - mf) / mf * 1e4
                r10 = qf * (f10 - mf) / mf * 1e4
                r30 = qf * (f30 - mf) / mf * 1e4
            A3, E3, M3, D = decompose(d, r3)
            A10, E10, M10, _ = decompose(d, r10)
            A30, E30, M30, _ = decompose(d, r30)
            return D, M3, M10, M30, A3, A10, A30, E3, E10, E30

        far = agg & ((hpx > pa) | (hpx < pb))
        n_far = int(far.sum())
        F = decomp_block(far)
        G = decomp_block(agg)

        vals = [n_agg, n_mid, n_far, mk3_qd, mk3_tk, n_mb, mk3_m, mk15_m, mk30_m,
                n_pb, mk3_p, mk15_p, mk30_p] + list(F) + list(G)
        out = [a.ticker, str(date)]
        for v in vals:
            out.append(str(int(v)) if isinstance(v, (int, np.integer)) else "%.5f" % v)
        print(",".join(out))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
