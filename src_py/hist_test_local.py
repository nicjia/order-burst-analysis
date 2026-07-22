#!/usr/bin/env python3
"""
hist_test_local.py — analyze the historical NYSE probe extracted LOCALLY
(scratchpad/hist_rows/*.row) vs local price panels. Window set by env DLO/DHI
(default 2020). Runs on partial or complete data. Battery (day-level inference):
  1. tick-constrained reversal (bottom-K by price, H=20, net 1bp)
  2. burst intensity -> next-day |ret| (vol)
  3. campaign reversion, BUY vs SELL
  4. next-day close-to-close IC (flow -> next ret): reversal vs continuation sign
  5. overnight IC (flow -> next close-to-open gap): the paper's core question
"""
import glob, math, os
import numpy as np, pandas as pd

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
REPO = "/Users/nick/order-burst-analysis"
DLO = int(os.environ.get("DLO", "20200101")); DHI = int(os.environ.get("DHI", "20201231"))
COST_BPS, H = 1.0, 20


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 20: return (np.nan, np.nan, T)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return (m, m / np.sqrt(v / T), T)


def zrows(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


def sh(s):
    s = s[s != 0].dropna()
    if len(s) < 40: return (np.nan, np.nan, len(s))
    return (s.mean() / s.std() * math.sqrt(252), s.mean() / (s.std() / math.sqrt(len(s))), len(s))


def fm(ypan, xpan, dates, ctrl=None):
    sl = []
    for d in dates:
        if d not in ypan.index or d not in xpan.index: continue
        df = pd.DataFrame({"y": ypan.loc[d], "x": xpan.loc[d]})
        if ctrl is not None and d in ctrl.index: df["c"] = ctrl.loc[d]
        df = df.dropna()
        if len(df) < 8: continue
        cs = ["x"] + (["c"] if "c" in df.columns else [])
        X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cs])
        b, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None); sl.append(b[1])
    return nw(sl)


def load():
    rows = []
    for f in glob.glob(SP + "/hist_rows/*.row"):
        t = open(f).read().strip()
        if t and len(t.split(",")) == 6: rows.append(t.split(","))
    d = pd.DataFrame(rows, columns=["ticker", "date", "netflow", "n_bursts", "buy", "sell"])
    for c in ["date", "netflow", "n_bursts"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date"]); d = d[(d.date >= DLO) & (d.date <= DHI)]
    FL = d.pivot_table(index="date", columns="ticker", values="netflow")
    CNT = d.pivot_table(index="date", columns="ticker", values="n_bursts")
    close = pd.read_csv(REPO + "/close_all.csv", index_col="date"); close.index = close.index.astype(int)
    openp = pd.read_csv(REPO + "/open_all.csv", index_col="date"); openp.index = openp.index.astype(int)
    dates = sorted(x for x in FL.index if DLO <= x <= DHI)
    cols = [c for c in FL.columns if c in close.columns]
    FL = FL.reindex(index=dates, columns=cols); CNT = CNT.reindex(index=dates, columns=cols)
    R = close.reindex(dates)[cols].pct_change(fill_method=None)
    gap = (openp.reindex(dates)[cols] - close.reindex(dates)[cols].shift(1)) / close.reindex(dates)[cols].shift(1)
    return dates, cols, FL, CNT, R, gap, close, d


def main():
    dates, cols, FL, CNT, R, gap, close, d = load()
    print("window %d-%d | panel: %d names, %d dates, %d non-zero name-days" %
          (DLO, DHI, len(cols), len(dates), int((d.netflow != 0).sum())))
    Z = zrows(FL)

    print("\n1. TICK REVERSAL (bottom-K, H=20, net 1bp)")
    for K in (20, 30, 40):
        sub = set(close.reindex(dates)[cols].mean().nsmallest(K).index)
        m = pd.DataFrame(False, index=dates, columns=cols); m.loc[:, [c for c in cols if c in sub]] = True
        z = Z.where(m); P = -np.sign(z); P = P.where(np.isfinite(P), 0.0)
        W = P.rolling(H, min_periods=1).mean().shift(1); W = W.sub(W.mean(axis=1), axis=0)
        g = W.abs().sum(axis=1).replace(0, np.nan); W = W.div(g, axis=0).fillna(0.0)
        ret = (W * R).sum(axis=1) - (COST_BPS / 1e4) * (W - W.shift(1)).abs().sum(axis=1)
        s, t, n = sh(ret); print("   K=%2d: Sharpe %+.2f (t=%+.2f, n=%d)" % (K, s, t, n))

    print("\n2. INTENSITY -> NEXT-DAY |RET| (vol)")
    ar = R.abs(); m, t, _ = fm(ar.shift(-1), zrows(CNT), dates); print("   raw:  %+.1f bps t=%+.2f" % (m*1e4, t))
    m, t, _ = fm(ar.shift(-1), zrows(CNT), dates, ctrl=ar); print("   +ctrl today|ret|: %+.1f bps t=%+.2f" % (m*1e4, t))

    print("\n3. CAMPAIGN REVERSION (>=3 same-sign days, next 5d), BUY vs SELL")
    idx = list(FL.index)
    for want, lab in [(+1, "BUY "), (-1, "SELL")]:
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
        m, t, _ = nw(pd.DataFrame(rev, columns=["d", "x"]).groupby("d")["x"].mean().values) if rev else (np.nan, np.nan, 0)
        print("   %s post: %+.1f bps/day t=%+.2f (n=%d)" % (lab, m*1e4, t, len(rev)))

    print("\n4. NEXT-DAY RETURN IC (flow_z -> next close-to-close): sign?")
    m, t, _ = fm(R.shift(-1), Z, dates); print("   %+.1f bps/1sd t=%+.2f  (%s)" % (m*1e4, t, "reversal" if m < 0 else "continuation"))
    print("5. OVERNIGHT IC (flow_z -> next close-to-open gap): paper's core")
    m, t, _ = fm(gap.shift(-1), Z, dates); print("   %+.1f bps/1sd t=%+.2f  (%s)" % (m*1e4, t, "reversal" if m < 0 else "continuation"))


if __name__ == "__main__":
    main()
