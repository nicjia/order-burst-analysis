#!/usr/bin/env python3
"""
hidden_sweep3.py — four referee analyses in one pass.

(A) SWEEP DECOMPOSITION: CONSUMPTION vs WITHDRAWAL.  The previous sweep flag fired when the
    touch price moved away OR its size more than halved. The second clause captures
    cancellations as well as executions, and those are opposite mechanisms: the aggressor's
    marketable remainder EATING the queue is mechanical displacement, whereas makers PULLING
    quotes 50ms after a hidden print is other participants revising beliefs -- Glosten-Milgrom
    operating, i.e. price discovery. ITCH separates them message by message. For a buy print
    the touch is the ask, and resting orders there are sell limit orders (direction = -1), so
    inside (t, t+100ms] at the pre-print ask price we classify:
        consumption : a type-4 (visible execution) message
        withdrawal  : a type-2/3 (cancel/delete) message
    and report markouts for consumption-only, withdrawal-only, both, and unswept.

(B) HUANG-STOLL BY SWEEP GROUP.  The paper's "51% of the effective half-spread" is measured
    from t on all aggressive prints. If most of the level is sweep-coincident, the
    adverse-selection share must be reported as a range. We re-run the decomposition
    separately on swept and unswept prints, and additionally with the price-impact leg
    measured from t+1s so displacement is excluded from the numerator.

(C) VAR-FREQUENCY BRIDGE.  The pre-drift decomposition conditioned on a single 30-second
    summary and found no continuation. But the VAR conditions on thirty ONE-MINUTE lags -- a
    richer set at a coarser frequency, and flow chasing returns at the 1-30 minute scale would
    produce exactly the observed pattern. We therefore regress the forward markout on the
    thirty preceding one-minute signed returns, the VAR's own conditioning set, and report the
    intercept (the component orthogonal to that path) against the raw mean.

(D) d x SWEPT CROSS-TAB.  The pre-trade depletion table and the sweep table currently split a
    picture that belongs in one place.

Output: ticker,date,n_agg,n_sw,
        f_cons,f_wdraw,f_both,                       composition of swept prints
        mk_cons,mk_wdraw,mk_both,mk_unsw,            3-min markout from t
        mk1_cons,mk1_wdraw,mk1_both,mk1_unsw,        from t+1s
        qs_sw,eff_sw,pi_sw,pi1_sw,                   Huang-Stoll, swept
        qs_un,eff_un,pi_un,pi1_un,                   Huang-Stoll, unswept
        vb_n,vb_mean,vb_a,vb_r2,                     VAR-frequency bridge
        dlo_un,dlo_sw,dhi_un,dhi_sw                  depletion x sweep
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
NCOL = 31
NA = "{t},{d}" + ",nan" * (NCOL - 2)
WIN = 0.1
NLAG = 30
TICK = 1e-9


def wm(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[np.abs(x) <= 1000]
    return float(np.mean(x)) if len(x) else np.nan


def any_event_at(ev_t, ev_px, t0, t1, target):
    """True where some event in (t0, t1] sits at price `target`. Arrays are time-sorted."""
    lo = np.searchsorted(ev_t, t0, side="right")
    hi = np.searchsorted(ev_t, t1, side="right")
    out = np.zeros(len(t0), bool)
    for k in range(len(t0)):
        a, b = lo[k], hi[k]
        if b > a:
            out[k] = np.any(np.abs(ev_px[a:b] - target[k]) < TICK)
    return out


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
        mt = df.t.to_numpy(float); mty = df.ty.to_numpy(int)
        mpx = df.px.to_numpy(float) / BA.SCALE; mdr = df.dr.to_numpy(int)

        h = df[df.ty == 5]
        if len(h) < 10 or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE
        hsz = h.sz.to_numpy(float)
        mid = BA.mid_at(bt, bm, ht)
        ok = np.isfinite(mid) & (mid > 0) & (ht >= RTH0 + 1860.0) & (ht < RTH1 - 1800.0)
        ht, hpx, hsz, mid = ht[ok], hpx[ok], hsz[ok], mid[ok]
        q = np.where(hpx > mid, 1, np.where(hpx < mid, -1, 0))
        agg = q != 0
        if agg.sum() < 20:
            print(NA.format(t=a.ticker, d=date)); return
        ta, qa, m0, sza = ht[agg], q[agg], mid[agg], hsz[agg]
        n_agg = len(ta)

        pb0, pa0 = BA.bbo_at(bt, bb, ba, ta - 1e-3)
        bs0, as0 = BA.bbo_at(bt, bbsz, basz, ta - 1e-3)
        pb1, pa1 = BA.bbo_at(bt, bb, ba, ta + WIN)
        bs1, as1 = BA.bbo_at(bt, bbsz, basz, ta + WIN)
        touch_px = np.where(qa > 0, pa0, pb0)
        touch_sz = np.where(qa > 0, as0, bs0)
        post_px = np.where(qa > 0, pa1, pb1)
        post_sz = np.where(qa > 0, as1, bs1)
        good = np.isfinite(touch_px) & np.isfinite(post_px) & np.isfinite(touch_sz) \
            & np.isfinite(post_sz) & (touch_sz > 0) & np.isfinite(pb0) & np.isfinite(pa0) \
            & (pa0 > pb0)
        moved = np.where(qa > 0, post_px > touch_px, post_px < touch_px)
        shrank = (np.abs(post_px - touch_px) < TICK) & (post_sz < 0.5 * touch_sz)
        sw = (moved | shrank) & good
        un = (~(moved | shrank)) & good
        if sw.sum() < 3 or un.sum() < 3:
            print(NA.format(t=a.ticker, d=date)); return

        # ---- (A) what happened at the touch: consumption vs withdrawal ----
        # resting orders at the ask are sell limit orders (dr=-1) and vice versa
        want_dr = np.where(qa > 0, -1, 1)
        cons = np.zeros(n_agg, bool); wdraw = np.zeros(n_agg, bool)
        for side in (-1, 1):
            s = want_dr == side
            if not s.any():
                continue
            ex = (mty == 4) & (mdr == side)
            cx = ((mty == 2) | (mty == 3)) & (mdr == side)
            for flag, sel in ((cons, ex), (wdraw, cx)):
                et, ex_px = mt[sel], mpx[sel]
                if len(et) == 0:
                    continue
                flag[s] = any_event_at(et, ex_px, ta[s], ta[s] + WIN, touch_px[s])

        f3 = BA.mid_at(bt, bm, ta + 180.0)
        b1 = BA.mid_at(bt, bm, ta + 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mkT = qa * (f3 - m0) / m0 * 1e4
            mk1 = qa * (f3 - b1) / b1 * 1e4

        c_only = sw & cons & ~wdraw
        w_only = sw & wdraw & ~cons
        both = sw & cons & wdraw
        nsw = max(int(sw.sum()), 1)

        # ---- (B) Huang-Stoll by sweep group ----
        with np.errstate(invalid="ignore", divide="ignore"):
            eff = qa * (hpx[agg] - m0) / m0 * 1e4
            qs = 0.5 * (pa0 - pb0) / m0 * 1e4
            pi = mkT
            pi1 = mk1

        # ---- (C) VAR-frequency bridge: thirty one-minute signed lags ----
        vb_n = 0; vb_mean = vb_a = vb_r2 = np.nan
        if n_agg >= 100:
            lags = np.empty((n_agg, NLAG))
            prev = BA.mid_at(bt, bm, ta)
            for L in range(1, NLAG + 1):
                cur = BA.mid_at(bt, bm, ta - 60.0 * L)
                with np.errstate(invalid="ignore", divide="ignore"):
                    lags[:, L - 1] = qa * (prev - cur) / cur * 1e4
                prev = cur
            y = mkT
            keep = np.isfinite(y) & (np.abs(y) <= 1000) & np.all(np.isfinite(lags), axis=1) \
                & np.all(np.abs(lags) <= 1000, axis=1)
            if keep.sum() >= NLAG + 40:
                X = np.column_stack([np.ones(keep.sum()), lags[keep]])
                yy = y[keep]
                try:
                    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
                    resid = yy - X @ beta
                    ss = ((yy - yy.mean()) ** 2).sum()
                    vb_n = int(keep.sum()); vb_mean = float(yy.mean())
                    vb_a = float(beta[0])
                    vb_r2 = float(1.0 - (resid ** 2).sum() / ss) if ss > 0 else np.nan
                except np.linalg.LinAlgError:
                    pass

        # ---- (D) depletion x sweep ----
        with np.errstate(invalid="ignore", divide="ignore"):
            d = sza / np.maximum(touch_sz, 1.0)
        dlo, dhi = d < 0.10, d >= 1.0

        vals = [n_agg, int(sw.sum()),
                c_only.sum() / nsw, w_only.sum() / nsw, both.sum() / nsw,
                wm(mkT[c_only]), wm(mkT[w_only]), wm(mkT[both]), wm(mkT[un]),
                wm(mk1[c_only]), wm(mk1[w_only]), wm(mk1[both]), wm(mk1[un]),
                wm(qs[sw]), wm(eff[sw]), wm(pi[sw]), wm(pi1[sw]),
                wm(qs[un]), wm(eff[un]), wm(pi[un]), wm(pi1[un]),
                vb_n, vb_mean, vb_a, vb_r2,
                wm(mkT[dlo & un]), wm(mkT[dlo & sw]), wm(mkT[dhi & un]), wm(mkT[dhi & sw])]
        out = [a.ticker, str(date)]
        for v in vals:
            out.append(str(int(v)) if isinstance(v, (int, np.integer)) else "%.5f" % v)
        print(",".join(out))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
