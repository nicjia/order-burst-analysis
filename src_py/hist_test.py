#!/usr/bin/env python3
"""
hist_test.py — 2017-2021 out-of-sample probe on the NYSE subset. Run AFTER the
hist_flow array completes (results/hist_flow/out/*.csv exist). Reuses the paper's
constructions on the historical panel:
  1. tick-constrained reversal (bottom-K by price, dollar-neutral, H=20, net 1bp)
     -> Sharpe/t on 2017-2021 (a genuine out-of-sample regime: 2018, 2020);
  2. burst-INTENSITY -> next-day realized vol (day-level Fama-MacBeth);
  3. multi-day campaign reversion.
This is a *pre-specified* replication of the 2022-2026 tests on new data, so it is
the discovery/confirmation split the referee asked for -- report as confirmation
only if it holds.
"""
import glob, math
import numpy as np, pandas as pd

COST_BPS, H, BURN = 1.0, 20, 252


def nw_t(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 30: return (np.nan, np.nan, T)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return (m, m / np.sqrt(v / T), T)


def zrows(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


def sh_t(s):
    s = s[s != 0].dropna()
    if len(s) < 60: return (np.nan, np.nan, 0)
    return (s.mean() / s.std() * math.sqrt(252), s.mean() / (s.std() / math.sqrt(len(s))), len(s))


def fm_slope(ypan, xpan, dates, ctrl=None):
    sl = []
    for d in dates:
        if d not in ypan.index or d not in xpan.index: continue
        df = pd.DataFrame({"y": ypan.loc[d], "x": xpan.loc[d]})
        if ctrl is not None and d in ctrl.index: df["c"] = ctrl.loc[d]
        df = df.dropna()
        if len(df) < 15: continue
        cs = ["x"] + (["c"] if "c" in df.columns else [])
        X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cs])
        b, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None); sl.append(b[1])
    return nw_t(sl)


def load():
    d = pd.concat([pd.read_csv(f) for f in glob.glob("results/hist_flow/out/*.csv")], ignore_index=True)
    for c in ["date", "netflow", "n_bursts"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    FL = d.pivot_table(index="date", columns="ticker", values="netflow")
    CNT = d.pivot_table(index="date", columns="ticker", values="n_bursts")
    close = pd.read_csv("close_all.csv", index_col="date"); close.index = close.index.astype(int)
    openp = pd.read_csv("open_all.csv", index_col="date"); openp.index = openp.index.astype(int)
    dates = sorted(x for x in FL.index if 20170101 <= x <= 20211231)
    cols = [c for c in FL.columns if c in close.columns]
    FL = FL.reindex(index=dates, columns=cols); CNT = CNT.reindex(index=dates, columns=cols)
    R = close.reindex(dates)[cols].pct_change(fill_method=None)
    cpx = close.reindex(dates)[cols]
    return dates, cols, FL, CNT, R, cpx, close, openp


def reversal(dates, cols, FL, R, cpx, K):
    Z = zrows(FL); sub = set(cpx.mean().nsmallest(K).index)
    mask = pd.DataFrame(False, index=dates, columns=cols)
    mask.loc[:, [c for c in cols if c in sub]] = True
    z = Z.where(mask); P = -np.sign(z); P = P.where(np.isfinite(P), 0.0)
    W = P.rolling(H, min_periods=1).mean().shift(1); W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan); W = W.div(g, axis=0).fillna(0.0)
    ret = (W * R).sum(axis=1); turn = (W - W.shift(1)).abs().sum(axis=1)
    return sh_t(ret - (COST_BPS / 1e4) * turn)


def campaigns(dates, cols, FL, R, k=3, post=5):
    idx = list(FL.index); cont, rev = [], []
    for name in cols:
        s = np.sign(FL[name].values); r = R[name].values; n = len(s); i = 0
        while i < n:
            if not np.isfinite(s[i]) or s[i] == 0: i += 1; continue
            j = i
            while j + 1 < n and s[j + 1] == s[i]: j += 1
            if j - i + 1 >= k:
                d = s[i]
                for tt in range(i, j + 1):
                    if tt + 1 < n and np.isfinite(r[tt + 1]): cont.append((idx[tt + 1], d * r[tt + 1]))
                for tt in range(j + 1, min(j + 1 + post, n)):
                    if np.isfinite(r[tt]): rev.append((idx[tt], d * r[tt]))
    def agg(p):
        return nw_t(pd.DataFrame(p, columns=["date", "x"]).groupby("date")["x"].mean().values)
    return agg(cont), agg(rev)


def main():
    dates, cols, FL, CNT, R, cpx, close, openp = load()
    print("2017-2021 NYSE panel: %d names, %d dates" % (len(cols), len(dates)))
    print("\n=== 1. TICK-CONSTRAINED REVERSAL (out-of-sample regime) ===")
    for K in (50, 100, 150):
        s, t, n = reversal(dates, cols, FL, R, cpx, K)
        print("  K=%3d: Sharpe %+.2f (t=%+.2f, n=%d)" % (K, s, t, n))
    print("\n=== 2. BURST INTENSITY -> NEXT-DAY VOL (day-level FM) ===")
    absret = R.abs(); nextabs = absret.shift(-1); CNTz = zrows(CNT)
    m, t, T = fm_slope(nextabs, CNTz, dates); print("  next|ret| ~ count_z: %+.2f bps t=%+.2f" % (m * 1e4, t))
    m, t, T = fm_slope(nextabs, CNTz, dates, ctrl=absret); print("    + control today|ret|: %+.2f t=%+.2f" % (m * 1e4, t))
    print("\n=== 3. MULTI-DAY CAMPAIGN REVERSION ===")
    (cm, ct, _), (rm, rt, _) = campaigns(dates, cols, FL, R)
    print("  continuation during: %+.2f bps t=%+.2f | reversion after: %+.2f bps t=%+.2f" % (cm * 1e4, ct, rm * 1e4, rt))


if __name__ == "__main__":
    main()
