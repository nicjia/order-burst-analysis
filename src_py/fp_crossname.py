#!/usr/bin/env python3
"""cross-name-v1 (idea 6, listed untested in LLM_README section 10): does burst flow in OTHER names predict a
name's next-minute return, beyond its own burst flow?

Per date: minute midpoint grids and trade bursts for every covered name. f[i,m] = signed burst volume started in
minute m, divided by the name's burst volume that day. r[i,m] = midpoint return over minute m, bps.
Regression pooled over (name, minute): r[i, m+1] = a f[i,m] + b P[i,m] + c r[i,m] + name-day fixed effect,
where P[i,m] is the mean f over the other names present that minute. Sufficient statistics (X'X, X'y) are
accumulated per date so nothing large moves; standard errors cluster by date.
Usage: fp_crossname.py OUTDIR CELLDIR [CELLDIR ...]
"""
import sys, os
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
import numpy as np

RTH0, NMIN = 34200.0, 390
K = 4  # own flow, peer flow, own lagged return, intercept-free (name-day demeaned)


def one_date(args):
    files = args
    mids, flows = [], []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if "grid_mid" not in z.files or "T_t_b" not in z.files:
            continue
        g = z["grid_mid"].astype(float)
        if len(g) != NMIN or not np.isfinite(g).sum() > 300:
            continue
        tb, side, vol = z["T_t_b"].astype(float), z["T_side"].astype(float), z["T_vol"].astype(float)
        tot = vol.sum()
        fl = np.zeros(NMIN)
        if len(tb) and tot > 0:
            idx = np.clip(((tb - RTH0) // 60).astype(int), 0, NMIN - 1)
            np.add.at(fl, idx, side * vol / tot)
        mids.append(g); flows.append(fl)
    if len(mids) < 30:
        return None
    M = np.vstack(mids); F = np.vstack(flows)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.diff(M, axis=1) / M[:, :-1] * 1e4           # returns for minutes 0..388
    n = len(M)
    S = np.nansum(F, axis=0)
    P = (S[None, :] - F) / max(n - 1, 1)                    # peer mean flow, excluding self
    y = R[:, 1:]                                            # r at m+1, m = 0..387
    x1 = F[:, 1:-1]                                         # own flow at m
    x2 = P[:, 1:-1]
    x3 = R[:, :-1]                                          # own return at m
    Rf = np.where(np.isfinite(R), R, 0.0); okR = np.isfinite(R).astype(float)
    peer_r = (Rf.sum(axis=0)[None, :] - Rf) / np.maximum(okR.sum(axis=0)[None, :] - okR, 1.0)
    x4 = peer_r[:, :-1]                                     # peers' contemporaneous return at m (staleness control)
    ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2) & np.isfinite(x3) & np.isfinite(x4)
    y = np.where(ok, y, np.nan); X = np.stack([x1, x2, x3, x4], axis=2)
    # name-day demeaning
    cnt = ok.sum(axis=1, keepdims=True).astype(float)
    cnt[cnt == 0] = np.nan
    y = y - np.nansum(np.where(ok, y, 0), axis=1, keepdims=True) / cnt
    for k in range(X.shape[2]):
        Xk = X[:, :, k]
        X[:, :, k] = Xk - np.nansum(np.where(ok, Xk, 0), axis=1, keepdims=True) / cnt
    m = ok.ravel()
    Y = y.ravel()[m]; XX = X.reshape(-1, X.shape[2])[m]
    good = np.isfinite(Y) & np.isfinite(XX).all(axis=1)
    Y, XX = Y[good], XX[good]
    if len(Y) < 1000:
        return None
    return XX.T @ XX, XX.T @ Y, len(Y)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    per_date = defaultdict(list)
    for cd in sys.argv[2:]:
        for pdir in Path(cd).iterdir():
            if pdir.is_dir() and pdir.name.isdigit():
                for f in pdir.glob("*.npz"):
                    per_date[f.stem].append(f)
    dates = sorted(per_date)
    A = np.zeros((4, 4)); b = np.zeros(4); N = 0
    per = []
    with Pool(int(os.environ.get("NSLOTS", "8"))) as pool:
        for res in pool.imap_unordered(one_date, [per_date[d] for d in dates], chunksize=1):
            if res is None:
                continue
            a_d, b_d, n_d = res
            A += a_d; b += b_d; N += n_d
            per.append((a_d, b_d))
    beta = np.linalg.solve(A, b)
    Ainv = np.linalg.inv(A)
    meat = np.zeros((4, 4))
    for a_d, b_d in per:
        s = b_d - a_d @ beta
        meat += np.outer(s, s)
    V = Ainv @ meat @ Ainv
    se = np.sqrt(np.diag(V))
    names = ["own_flow", "peer_flow", "own_lag_return", "peer_return"]
    res = {n: dict(b=float(beta[i]), t=float(beta[i] / se[i])) for i, n in enumerate(names)}
    res["obs"] = int(N); res["dates"] = int(len(per))
    (out / "crossname.json").write_text(__import__("json").dumps(res, indent=1))
    for n in names:
        print("%-16s %+10.4f  t %7.2f" % (n, res[n]["b"], res[n]["t"]))
    print("obs", N, "dates", len(per), flush=True)


if __name__ == "__main__":
    main()
