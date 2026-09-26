#!/usr/bin/env python3
"""ETF -> constituent lead (cluster; backlog T1.10 / B11.3 / G4). Does SPY / QQQ burst flow in minute m predict an
individual stock's mid return in minute m+1, beyond the stock's own burst flow, other stocks' burst flow, its own
lagged return and other stocks' contemporaneous return (the stale-price control)?

Per date: minute mid grids and run-60 burst flow of every TEST-cell stock (p4-revisit-v1 npz), and run-0.5 s burst
flow of SPY and QQQ (etf_bursts), each normalized by that security's burst volume that day. Pooled regression over
(stock, minute) with stock-day demeaning; sufficient statistics accumulated per date; standard errors clustered by
date. Reported separately for 2022-23 and 2024.
Usage: etf_lead.py OUTDIR NPZ_CELLDIR ETF_OUTDIR
"""
import sys, os, json
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

RTH0, NMIN = 34200.0, 390
NAMES = ["own_flow", "peer_flow", "spy_flow", "qqq_flow", "own_lag_ret", "peer_ret"]
ETF_DIR = None


def etf_minute_flow(date, etf):
    f = Path(ETF_DIR) / {"SPY": "84398", "QQQ": "86755"}[etf] / ("%s.csv.gz" % date)
    fl = np.zeros(NMIN)
    if not f.exists():
        return None
    try:
        e = pd.read_csv(f)
    except Exception:
        return None
    if not len(e):
        return fl
    tot = e.vol.sum()
    idx = np.clip(((e.t_b.to_numpy() - RTH0) // 60).astype(int), 0, NMIN - 1)
    np.add.at(fl, idx, e.side.to_numpy() * e.vol.to_numpy() / tot)
    return fl


def one_date(args):
    date, files = args
    spy, qqq = etf_minute_flow(date, "SPY"), etf_minute_flow(date, "QQQ")
    if spy is None or qqq is None:
        return None
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
        tb, sd, vol = z["T_t_b"].astype(float), z["T_side"].astype(float), z["T_vol"].astype(float)
        fl = np.zeros(NMIN)
        if len(tb) and vol.sum() > 0:
            np.add.at(fl, np.clip(((tb - RTH0) // 60).astype(int), 0, NMIN - 1), sd * vol / vol.sum())
        mids.append(g); flows.append(fl)
    if len(mids) < 30:
        return None
    M = np.vstack(mids); F = np.vstack(flows)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.diff(M, axis=1) / M[:, :-1] * 1e4
    n = len(M)
    P = (np.nansum(F, 0)[None, :] - F) / max(n - 1, 1)
    Rf = np.where(np.isfinite(R), R, 0.0); okR = np.isfinite(R).astype(float)
    PR = (Rf.sum(0)[None, :] - Rf) / np.maximum(okR.sum(0)[None, :] - okR, 1.0)
    y = R[:, 1:]
    Xs = [F[:, 1:-1], P[:, 1:-1], np.broadcast_to(spy[1:-1], y.shape), np.broadcast_to(qqq[1:-1], y.shape),
          R[:, :-1], PR[:, :-1]]
    X = np.stack(Xs, axis=2).astype(float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=2)
    cnt = ok.sum(1, keepdims=True).astype(float); cnt[cnt == 0] = np.nan
    y = np.where(ok, y, 0.0); y = y - y.sum(1, keepdims=True) / cnt
    for k in range(X.shape[2]):
        xk = np.where(ok, X[:, :, k], 0.0)
        X[:, :, k] = xk - xk.sum(1, keepdims=True) / cnt
    m = ok.ravel()
    Y = y.ravel()[m]; XX = X.reshape(-1, X.shape[2])[m]
    g = np.isfinite(Y) & np.isfinite(XX).all(1)
    Y, XX = Y[g], XX[g]
    if len(Y) < 1000:
        return None
    return date, XX.T @ XX, XX.T @ Y, len(Y)


def solve(parts):
    k = len(NAMES)
    A = sum(p[1] for p in parts); b = sum(p[2] for p in parts)
    beta = np.linalg.solve(A, b); Ai = np.linalg.inv(A)
    meat = sum(np.outer(p[2] - p[1] @ beta, p[2] - p[1] @ beta) for p in parts)
    se = np.sqrt(np.diag(Ai @ meat @ Ai))
    return {nm: dict(b=float(beta[i]), t=float(beta[i] / se[i])) for i, nm in enumerate(NAMES)}


def main():
    global ETF_DIR
    out, cd, ETF_DIR = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    per = defaultdict(list)
    for p in cd.iterdir():
        if p.is_dir() and p.name.isdigit():
            for f in p.glob("*.npz"):
                if f.stem[:4] in ("2022", "2023", "2024"):
                    per[f.stem].append(f)
    dates = sorted(per)
    with Pool(int(os.environ.get("NSLOTS", "8"))) as pool:
        parts = [r for r in pool.imap_unordered(one_date, [(d, per[d]) for d in dates], chunksize=1) if r is not None]
    res = {"2022-23": solve([p for p in parts if p[0][:4] in ("2022", "2023")]),
           "2024": solve([p for p in parts if p[0][:4] == "2024"])}
    res["dates"] = {k: sum(1 for p in parts if (p[0][:4] in ("2022", "2023")) == (k == "2022-23")) for k in ("2022-23", "2024")}
    (out / "etf_lead.json").write_text(json.dumps(res, indent=1))
    for per_ in ("2022-23", "2024"):
        print(per_, " ".join("%s %+.3f (t %.2f)" % (k, v["b"], v["t"]) for k, v in res[per_].items()))


if __name__ == "__main__":
    main()
