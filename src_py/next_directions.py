#!/usr/bin/env python3
"""
next_directions.py — pre-specified exploratory probes on EXISTING data only
(no new data buys). Day-level (Fama-MacBeth + Newey-West) inference throughout;
these are exploratory and reported as such.

A. UNSIGNED TARGET: does burst INTENSITY (count) predict next-day volatility
   (|close-to-close| and |close-to-open gap|), beyond today's own volatility?
   Vol is far more predictable than signed returns; the paper's conclusion names
   this. Predictor = hidden-burst count n (results/hidden_xsec, 2023-2024).

B. MULTI-DAY CAMPAIGNS: define a campaign as a run of >=k consecutive same-signed
   daily net-flow days; test continuation DURING the run and reversion AFTER it
   ends (Bucci-style post-metaorder decay). Predictor = the paper's daily flow
   signal (m7, 2022-2026).
"""
import glob, math
import numpy as np, pandas as pd
import m7_reversal_baseline as m7


def nw_t(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 30: return (np.nan, np.nan, T)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return (m, m / np.sqrt(v / T), T)


def zrows(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


def fm_slope(ypan, xpan, dates, ctrl=None):
    slopes = []
    for d in dates:
        if d not in ypan.index or d not in xpan.index: continue
        df = pd.DataFrame({"y": ypan.loc[d], "x": xpan.loc[d]})
        if ctrl is not None and d in ctrl.index: df["c"] = ctrl.loc[d]
        df = df.dropna()
        if len(df) < 20: continue
        cs = ["x"] + (["c"] if "c" in df.columns else [])
        X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cs])
        b, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None)
        slopes.append(b[1])
    return nw_t(slopes)


def part_A():
    print("=" * 72 + "\nA. UNSIGNED VOL PREDICTION FROM BURST INTENSITY (day-level FM)\n" + "=" * 72)
    close = pd.read_csv("close_all.csv", index_col="date"); close.index = close.index.astype(int); close = close.sort_index()
    openp = pd.read_csv("open_all.csv", index_col="date"); openp.index = openp.index.astype(int); openp = openp.sort_index()
    rows = []
    for f in glob.glob("results/hidden_xsec/out/*.csv"):
        d = pd.read_csv(f)
        if "n" in d.columns: rows.append(d[["ticker", "date", "n"]])
    H = pd.concat(rows, ignore_index=True)
    for c in ["date", "n"]: H[c] = pd.to_numeric(H[c], errors="coerce")
    H = H[H["n"] > 0]
    CNT = H.pivot_table(index="date", columns="ticker", values="n")
    dates = sorted(CNT.index); cols = [c for c in CNT.columns if c in close.columns]
    CNT = CNT[cols]
    clc = close.reindex(dates)[cols].pct_change()
    absret = clc.abs()
    nextabs = absret.shift(-1)
    gap = ((openp.reindex(dates)[cols] - close.reindex(dates)[cols].shift(1)) / close.reindex(dates)[cols].shift(1)).abs()
    nextgap = gap.shift(-1)
    CNTz = zrows(CNT)
    m, t, T = fm_slope(nextabs, CNTz, dates)
    print(f"  next-day |ret|  ~ burst-count_z            : {m*1e4:+.2f} bps/1sd  NW t={t:+.2f}  (T={T} days)")
    m, t, T = fm_slope(nextabs, CNTz, dates, ctrl=absret)
    print(f"  next-day |ret|  ~ burst-count_z + today|ret|: {m*1e4:+.2f} bps/1sd  NW t={t:+.2f}  (incremental)")
    m, t, T = fm_slope(nextgap, CNTz, dates)
    print(f"  next-day |gap|  ~ burst-count_z            : {m*1e4:+.2f} bps/1sd  NW t={t:+.2f}")
    m, t, T = fm_slope(nextgap, CNTz, dates, ctrl=gap)
    print(f"  next-day |gap|  ~ burst-count_z + today|gap|: {m*1e4:+.2f} bps/1sd  NW t={t:+.2f}  (incremental)")


def part_B(k=3, post=5):
    print("\n" + "=" * 72 + f"\nB. MULTI-DAY CAMPAIGNS (run>={k} same-sign flow days; day-level)\n" + "=" * 72)
    dis, cols, FL, PR, SD, R, cpx, close = m7.load_panels()
    idx = list(FL.index)
    cont, rev = [], []
    Rv = R.reindex(index=dis, columns=FL.columns)
    for name in FL.columns:
        s = np.sign(FL[name].values); r = Rv[name].values; n = len(s); i = 0
        while i < n:
            if not np.isfinite(s[i]) or s[i] == 0: i += 1; continue
            j = i
            while j + 1 < n and s[j + 1] == s[i]: j += 1
            Lrun = j - i + 1; d = s[i]
            if Lrun >= k:
                for tt in range(i, j + 1):
                    if tt + 1 < n and np.isfinite(r[tt + 1]): cont.append((idx[tt + 1], d * r[tt + 1]))
                for tt in range(j + 1, min(j + 1 + post, n)):
                    if np.isfinite(r[tt]): rev.append((idx[tt], d * r[tt]))
            i = j + 1
    def agg(pairs, lab):
        df = pd.DataFrame(pairs, columns=["date", "x"])
        daily = df.groupby("date")["x"].mean()
        m, t, T = nw_t(daily.values)
        print(f"  {lab:32s}: {m*1e4:+.2f} bps/day  NW t={t:+.2f}  (n_obs={len(df):,}, days={T})")
    agg(cont, "continuation (during campaign)")
    agg(rev, f"reversion (<= {post}d after campaign)")


if __name__ == "__main__":
    part_A()
    part_B()
