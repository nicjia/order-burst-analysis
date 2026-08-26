#!/usr/bin/env python3
"""
hidden_incr.py — does NON-DISPLAYED execution count add anything over displayed count?

The volatility result established on 472 names uses two regressors that I had been calling
"burst intensity" but which are simply per-bucket message counts: type-4 (visible executions)
and type-5 (hidden executions). Trade count forecasting volatility is a classic result --
Jones, Kaul and Lipson (1994), and the mixture-of-distributions literature -- so the
incremental-over-HAR finding, while robust, is close to known territory.

The one element that is NOT standard is entering hidden count separately, because most
datasets cannot identify non-displayed execution. This script asks whether that term earns
its place, by nesting the models:

    M0  lagged volatility only (intraday HAR: lag-1 plus a 6-bucket trailing mean)
    M1  M0 + visible count
    M2  M1 + hidden count          <- t on the hidden term is the decisive number
    M3  M0 + hidden count          (hidden alone, for the reverse ordering)
    M4  M0 + visible count + dollar volume + hidden count   (strictest: hidden must beat
                                                             BOTH count and volume)

If hidden count is incremental to visible count and volume, non-displayed execution intensity
carries volatility information that displayed activity does not. If it is not, the earlier
result is a well-replicated confirmation of a thirty-year-old finding.

Output: ticker,date,W,nb,r2_0,r2_v,r2_vh,r2_h,r2_full,t_h_given_v,t_v_given_h,t_h_given_vvol
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA
RTH0, RTH1 = 34200.0, 57600.0
NCOL = 12
NA = "{t},{d},{w},0" + ",nan" * (NCOL - 4)

def fit(y, cols):
    """OLS with intercept. Returns (r2, coefs, t-stats)."""
    X = np.column_stack([np.ones(len(y))] + list(cols))
    try:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, None, None
    res = y - X @ b
    ss = ((y - y.mean()) ** 2).sum()
    r2 = (1 - (res ** 2).sum() / ss) if ss > 0 else np.nan
    n, p = X.shape
    s2 = (res ** 2).sum() / max(n - p, 1)
    try:
        se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
        t = np.divide(b, se, out=np.full_like(b, np.nan), where=se > 0)
    except Exception:
        t = np.full(p, np.nan)
    return r2, b, t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    a = ap.parse_args(); tk = a.ticker
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(a.msg))
    d = int(m.group(1) + m.group(2) + m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(a.msg)
        df = pd.read_csv(a.msg, header=None, usecols=[0, 1, 3, 4], names=["t", "ty", "sz", "px"])
        df = df[(df.t >= RTH0) & (df.t < RTH1)]
        if len(df) < 500 or len(bt) < 100:
            for W in (60, 300): print(NA.format(t=tk, d=d, w=W))
            return
        T = df.t.to_numpy(float); TY = df.ty.to_numpy(int)
        SZ = df.sz.to_numpy(float); PX = df.px.to_numpy(float) / BA.SCALE
        tv, sv, pv = T[TY == 4], SZ[TY == 4], PX[TY == 4]
        th = T[TY == 5]
        for W in (60.0, 300.0):
            edges = np.arange(RTH0, RTH1 - W, W)
            if len(edges) < 40:
                print(NA.format(t=tk, d=d, w=int(W))); continue
            mg = BA.mid_at(bt, bm, edges)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.diff(mg) / mg[:-1] * 1e4
            rv = np.abs(r)
            bins = np.append(edges, edges[-1] + W)
            cnt = np.histogram(tv, bins=bins)[0].astype(float)[:len(rv)]
            hcn = np.histogram(th, bins=bins)[0].astype(float)[:len(rv)]
            dv = np.histogram(tv, bins=bins, weights=sv * pv)[0].astype(float)[:len(rv)]
            L = 6
            if len(rv) < L + 30:
                print(NA.format(t=tk, d=d, w=int(W))); continue
            y = rv[L:]; lag1 = rv[L - 1:-1]
            lagK = np.array([rv[i - L:i].mean() for i in range(L, len(rv))])
            xv, xh, xd = cnt[L:], hcn[L:], dv[L:]
            k = np.isfinite(y) & np.isfinite(lag1) & np.isfinite(lagK) \
                & np.isfinite(xv) & np.isfinite(xh) & np.isfinite(xd)
            if k.sum() < 40 or np.std(xh[k]) < 1e-12:
                print(NA.format(t=tk, d=d, w=int(W))); continue
            Y = y[k]; l1, lk, v, h, dvv = lag1[k], lagK[k], xv[k], xh[k], xd[k]
            r2_0, _, _ = fit(Y, [l1, lk])
            r2_v, _, _ = fit(Y, [l1, lk, v])
            r2_vh, _, t2 = fit(Y, [l1, lk, v, h])          # t2[4] = hidden | visible
            r2_h, _, _ = fit(Y, [l1, lk, h])
            r2_f, _, t4 = fit(Y, [l1, lk, v, dvv, h])      # t4[5] = hidden | visible + volume
            _, _, t3 = fit(Y, [l1, lk, h, v])              # t3[4] = visible | hidden
            g = lambda arr, i: (arr[i] if arr is not None and len(arr) > i else np.nan)
            vals = [k.sum(), r2_0, r2_v, r2_vh, r2_h, r2_f,
                    g(t2, 4), g(t3, 4), g(t4, 5)]
            f = lambda x: ("%.5f" % x) if np.isfinite(x) else "nan"
            print("%s,%d,%d,%d,%s" % (tk, d, int(W), int(vals[0]),
                                      ",".join(f(x) for x in vals[1:])))
    except Exception as e:
        print(f"{tk},{d},ERR,{e}", file=sys.stderr)
        for W in (60, 300): print(NA.format(t=tk, d=d, w=W))

if __name__ == "__main__":
    main()
