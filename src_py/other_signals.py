#!/usr/bin/env python3
"""
other_signals.py — additional exploratory probes on EXISTING 2022-2026 data only
(lobster2 down; no new streaming). Day-level inference; reported as exploratory.

1. CROSS-ASSET MARKET TIMING: does aggregate burst-flow breadth predict the
   next-day market (SPY/QQQ) return? (panel -> index lead-lag)
2. AGGREGATE INTENSITY -> MARKET VOL: does total burst count predict next-day
   |market return| (index-level volatility timing)?
3. SHORT-TILT ASYMMETRY: are SELL-signed campaigns' reversions stronger than BUY
   (short sellers = best-documented informed traders)?
"""
import math
import numpy as np, pandas as pd
import m7_reversal_baseline as m7


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 30: return (np.nan, np.nan, T)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return (m, m / np.sqrt(v / T), T)


def ols_nw(y, x, L=10):
    """simple regression y ~ x with NW t on the slope; x standardized."""
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(d) < 60: return (np.nan, np.nan, len(d))
    xs = (d["x"] - d["x"].mean()) / d["x"].std()
    X = np.column_stack([np.ones(len(d)), xs.values])
    b, *_ = np.linalg.lstsq(X, d["y"].values, rcond=None)
    e = d["y"].values - X @ b
    XtXi = np.linalg.inv(X.T @ X); S = X * e[:, None]; G = S.T @ S
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); Gl = S[l:].T @ S[:-l]; G += w * (Gl + Gl.T)
    se = np.sqrt(np.diag(XtXi @ G @ XtXi))
    return (b[1], b[1] / se[1], len(d))


def main():
    dis, cols, FL, PR, SD, R, cpx, close = m7.load_panels()
    c2 = pd.read_csv("close_all.csv", index_col="date"); c2.index = c2.index.astype(int); c2 = c2.sort_index()

    def etf(tk):
        return c2[tk].pct_change().reindex(dis) if tk in c2.columns else pd.Series(np.nan, index=dis)
    spy, qqq = etf("SPY"), etf("QQQ")

    sign = np.sign(FL)
    breadth = sign.mean(axis=1)            # net long-short breadth of burst flow
    print("=" * 70 + "\n1. CROSS-ASSET MARKET TIMING (flow breadth -> next-day index)\n" + "=" * 70)
    for nm, mkt in [("SPY", spy), ("QQQ", qqq)]:
        b, t, n = ols_nw(mkt.shift(-1), breadth)
        print(f"  next-day {nm} ret ~ flow-breadth_z : {b*1e4:+.2f} bps/1sd  NW t={t:+.2f}  (n={n})")
        b, t, n = ols_nw(mkt.shift(-1), breadth, )
    # contemporaneous check (should be positive: buying breadth <-> up day)
    b, t, n = ols_nw(spy, breadth); print(f"  [contemp. SPY ret ~ breadth_z (sanity): {b*1e4:+.2f} bps t={t:+.2f}]")

    print("\n" + "=" * 70 + "\n2. AGGREGATE INTENSITY -> NEXT-DAY MARKET VOL\n" + "=" * 70)
    intensity = FL.abs().sum(axis=1)       # total gross flow as activity proxy
    b, t, n = ols_nw(spy.abs().shift(-1), intensity)
    print(f"  next-day |SPY| ~ total-|flow|_z         : {b*1e4:+.2f} bps/1sd  NW t={t:+.2f}")
    b, t, n = ols_nw(spy.abs().shift(-1), intensity - 0)  # placeholder same
    b, t, n = ols_nw(spy.abs().shift(-1), intensity, )
    # control for today's |SPY|
    d = pd.DataFrame({"y": spy.abs().shift(-1), "x": intensity, "c": spy.abs()}).dropna()
    if len(d) > 60:
        xs = (d["x"] - d["x"].mean()) / d["x"].std(); cc = (d["c"] - d["c"].mean()) / d["c"].std()
        X = np.column_stack([np.ones(len(d)), xs.values, cc.values])
        bb, *_ = np.linalg.lstsq(X, d["y"].values, rcond=None); e = d["y"].values - X @ bb
        XtXi = np.linalg.inv(X.T @ X); S = X * e[:, None]; G = S.T @ S
        for l in range(1, 11):
            w = 1 - l / 11; Gl = S[l:].T @ S[:-l]; G += w * (Gl + Gl.T)
        se = np.sqrt(np.diag(XtXi @ G @ XtXi))
        print(f"    + control today's |SPY|              : {bb[1]*1e4:+.2f} bps/1sd  NW t={bb[1]/se[1]:+.2f} (incremental)")

    print("\n" + "=" * 70 + "\n3. SHORT-TILT ASYMMETRY (buy vs sell campaign reversion)\n" + "=" * 70)
    idx = list(FL.index)
    for want, lab in [(+1, "BUY campaigns"), (-1, "SELL campaigns")]:
        rev = []
        for name in cols:
            s = np.sign(FL[name].values); r = R[name].values; n = len(s); i = 0
            while i < n:
                if not np.isfinite(s[i]) or s[i] == 0: i += 1; continue
                j = i
                while j + 1 < n and s[j + 1] == s[i]: j += 1
                if j - i + 1 >= 3 and s[i] == want:
                    for tt in range(j + 1, min(j + 6, n)):
                        if np.isfinite(r[tt]): rev.append((idx[tt], want * r[tt]))
                i = j + 1
        m, t, _ = nw(pd.DataFrame(rev, columns=["d", "x"]).groupby("d")["x"].mean().values)
        print(f"  {lab:15s} post-reversion: {m*1e4:+.2f} bps/day  NW t={t:+.2f}  (n_obs={len(rev):,})")


if __name__ == "__main__":
    main()
