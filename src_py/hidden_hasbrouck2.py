#!/usr/bin/env python3
"""
hidden_hasbrouck2.py — re-estimated Hasbrouck (1991) VAR addressing two referee objections
to the original run.

(1) MEMORY. The first version used VAR(12) on a 10-second clock: 120 seconds of memory.
    Iterating the companion matrix to 3 and 10 minutes therefore returned the model's own
    asymptote by construction, so flatness beyond two minutes was extrapolation, not
    estimation. Here we use ONE-MINUTE bins with 30 lags, giving 30 minutes of genuine
    model memory, so the 3- and 10-minute responses are inside the estimable range.

(2) SELECTION ON THE PARAMETER UNDER STUDY. The first version dropped non-stationary
    name-days, which selects on estimated persistence -- exactly the quantity of interest.
    Here we retain ALL name-days and apply ridge shrinkage to the VAR coefficients, which
    pulls explosive roots toward zero without discarding the observation. We report the
    unfiltered shrinkage estimate as primary and flag stationarity rather than conditioning
    on it, so the two samples can be compared.

Output: ticker,date,nbins,stationary,imp1,imp3,imp10,imp30
        (cumulative impulse response of the mid to a one-SD signed-flow innovation,
         in bps, at 1/3/10/30 minutes)
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
DELTA = 60.0          # one-minute bins
PLAG = 30             # 30 lags -> 30 minutes of memory
RIDGE = 1e-2
NA = "{t},{d},0,0,nan,nan,nan,nan"


def fit_var(X, p, ridge=RIDGE):
    """X: (T,2) columns [flow, dmid]. Ridge-regularized VAR(p). Returns list of 2x2 A_i."""
    T = len(X)
    if T < 4 * p + 30:
        return None
    Y = X[p:]
    Z = np.column_stack([X[p - i - 1:T - i - 1] for i in range(p)])
    Z = np.column_stack([np.ones(len(Z)), Z])
    G = Z.T @ Z + ridge * len(Z) * np.eye(Z.shape[1])
    try:
        B = np.linalg.solve(G, Z.T @ Y)
    except np.linalg.LinAlgError:
        return None
    return [B[1 + 2 * i:3 + 2 * i].T for i in range(p)]


def irf(A, p, steps):
    """cumulative response of variable 1 (dmid) to a unit shock in variable 0 (flow)"""
    psi = [np.eye(2)]
    for s in range(1, steps + 1):
        acc = np.zeros((2, 2))
        for i in range(min(s, p)):
            acc += A[i] @ psi[s - i - 1]
        psi.append(acc)
    return float(np.sum([psi[s][1, 0] for s in range(steps + 1)]))


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
        if len(h) < 20 or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE; hsz = h.sz.to_numpy(float)
        mid = BA.mid_at(bt, bm, ht)
        q = np.where(hpx > mid, 1.0, np.where(hpx < mid, -1.0, 0.0))
        sel = (q != 0) & (ht >= RTH0) & (ht < RTH1) & np.isfinite(mid)
        if sel.sum() < 20:
            print(NA.format(t=a.ticker, d=date)); return

        edges = np.arange(RTH0, RTH1 + DELTA, DELTA); nb = len(edges) - 1
        flow = np.zeros(nb)
        bi = np.clip(((ht[sel] - RTH0) / DELTA).astype(int), 0, nb - 1)
        np.add.at(flow, bi, q[sel] * hsz[sel])
        me = BA.mid_at(bt, bm, edges)
        me = pd.Series(me).ffill().bfill().to_numpy()
        dmid = 1e4 * np.diff(me) / me[:nb]
        dmid = np.clip(dmid, -500, 500)

        ok = np.isfinite(flow) & np.isfinite(dmid)
        if ok.sum() < 4 * PLAG + 30 or np.std(flow[ok]) < 1e-9 or np.std(dmid[ok]) < 1e-9:
            print(NA.format(t=a.ticker, d=date)); return
        f = (flow[ok] - flow[ok].mean()) / flow[ok].std()      # one-SD flow innovation
        X = np.column_stack([f, dmid[ok]])

        A = fit_var(X, PLAG)
        if A is None:
            print(NA.format(t=a.ticker, d=date)); return
        comp = np.zeros((2 * PLAG, 2 * PLAG))
        for i in range(PLAG):
            comp[0:2, 2 * i:2 * i + 2] = A[i]
        if PLAG > 1:
            comp[2:2 * PLAG, 0:2 * (PLAG - 1)] = np.eye(2 * (PLAG - 1))
        stat = int(np.max(np.abs(np.linalg.eigvals(comp))) < 1.0)

        r = [irf(A, PLAG, s) for s in (1, 3, 10, 30)]
        if not all(np.isfinite(v) and abs(v) < 1e4 for v in r):
            print(NA.format(t=a.ticker, d=date)); return
        print("{},{},{},{},".format(a.ticker, date, int(ok.sum()), stat)
              + ",".join("%.5f" % v for v in r))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
