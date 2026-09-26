#!/usr/bin/env python3
"""
hidden_hasbrouck.py — Hasbrouck (1991)-style permanent/transitory decomposition for
aggressive hidden order flow, per name-day, via a fixed-clock VAR and a NUMERICAL
impulse response. The cumulative impulse response of price to an order-flow shock is
the model-implied markout curve; whether it plateaus (permanent) or reverts (transitory)
is the structural benchmark for the reduced-form markout.

Construction (per name-day):
  bin the regular session into Delta=10s bins.
  x_b = signed aggressive-hidden volume in bin b (sum of sign*size), then standardized.
        sign = quote rule vs the pre-trade mid (away-from-mid = aggressive).
  r_b = 1e4 * (log mid_end - log mid_start) over bin b, in bps.
Reduced-form VAR(p) on y=(x,r); structural shock via Cholesky (x ordered first). The IRF
is iterated forward numerically (no matrix inversion), and the cumulative price response
is read at the contemporaneous bin, 3 min, and 10 min.

Output row: ticker,date,n_events,imp0_bps,imp3_bps,imp10_bps,frac_10_3
  imp0  = price response in the shock bin (immediate)
  imp3  = cumulative price response at 3 min per 1-SD order-flow shock
  imp10 = cumulative price response at 10 min
  frac_10_3 = imp10/imp3  (~1 permanent past 3 min, <1 reverting, >1 still building)
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
DELTA = 10.0
P = 12                       # lags (2 min at 10s)
H3, H10 = 18, 60             # 3 min, 10 min in bins
MINEV = 100
NA = "{t},{d},0,nan,nan,nan,nan"


def var_irf(x, r, p=P, H=H10):
    n = len(x)
    if n < 6 * p + 60 or np.std(x) < 1e-9 or np.std(r) < 1e-12:
        return None
    Y = np.column_stack([x, r])                       # ordered (x, r)
    rows = n - p
    Z = np.ones((rows, 1 + 2 * p))
    for i in range(1, p + 1):
        Z[:, i] = x[p - i:n - i]
        Z[:, p + i] = r[p - i:n - i]
    try:
        ZTZi = np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        return None
    B = ZTZi @ (Z.T @ Y[p:])                           # (1+2p) x 2 coefficient matrix
    resid = Y[p:] - Z @ B
    S = np.cov(resid.T)
    if not np.all(np.isfinite(S)) or S[0, 0] <= 0:
        return None
    # lag matrices A_i (2x2), mapping (x,r)_{t-i} -> (x,r)_t
    A = []
    for i in range(1, p + 1):
        # B columns: [:,0] -> x eqn, [:,1] -> r eqn; rows: 0=const, 1..p = x-lags, p+1..2p = r-lags
        Ai = np.array([[B[i, 0], B[p + i, 0]],
                       [B[i, 1], B[p + i, 1]]])
        A.append(Ai)
    try:
        L = np.linalg.cholesky(S)                      # lower-tri, x-shock hits r contemporaneously
    except np.linalg.LinAlgError:
        return None
    # stationarity: reject if the VAR companion matrix has a root on/outside the unit circle
    comp = np.zeros((2 * p, 2 * p))
    for i in range(p):
        comp[0:2, 2 * i:2 * i + 2] = A[i]
    if p > 1:
        comp[2:2 * p, 0:2 * (p - 1)] = np.eye(2 * (p - 1))
    if np.max(np.abs(np.linalg.eigvals(comp))) >= 1.0:
        return None                                    # non-stationary: IRF would diverge
    # numerical IRF to a 1-SD structural x-shock
    R = [L[:, 0].copy()]                               # response at h=0
    for h in range(1, H + 1):
        acc = np.zeros(2)
        for i in range(1, min(h, p) + 1):
            acc += A[i - 1] @ R[h - i]
        R.append(acc)
    rpath = np.array([v[1] for v in R])                # price-response path
    cum = np.cumsum(rpath)
    return cum


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
        if len(h) < MINEV or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE; hsz = h.sz.to_numpy(float)
        mid_pre = BA.mid_at(bt, bm, ht)
        s = np.where(hpx > mid_pre, 1.0, np.where(hpx < mid_pre, -1.0, 0.0))
        agg = (s != 0) & (ht >= RTH0) & (ht <= RTH1)
        n_ev = int(agg.sum())
        if n_ev < MINEV:
            print(NA.format(t=a.ticker, d=date)); return
        # fixed-clock bins
        edges = np.arange(RTH0, RTH1 + DELTA, DELTA)
        nb = len(edges) - 1
        # signed aggressive volume per bin
        bidx = np.clip(((ht[agg] - RTH0) / DELTA).astype(int), 0, nb - 1)
        sv = (s[agg] * hsz[agg])
        x = np.bincount(bidx, weights=sv, minlength=nb)[:nb]
        # mid at bin edges -> bin returns
        me = BA.mid_at(bt, bm, edges)
        ok = np.isfinite(me) & (me > 0)
        if ok.sum() < nb * 0.5:
            print(NA.format(t=a.ticker, d=date)); return
        me = pd.Series(me).ffill().bfill().to_numpy()
        r = 1e4 * (np.log(me[1:]) - np.log(me[:nb]))
        r = np.clip(r, -500, 500)
        xs = (x - x.mean()) / (x.std() + 1e-12)         # standardize order flow
        cum = var_irf(xs, r)
        if cum is None:
            print(NA.format(t=a.ticker, d=date)); return
        imp0, imp3, imp10 = cum[0], cum[H3], cum[H10]
        frac = imp10 / imp3 if abs(imp3) > 1e-9 else np.nan
        print(f"{a.ticker},{date},{n_ev},{imp0:.5f},{imp3:.5f},{imp10:.5f},{frac:.5f}")
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
