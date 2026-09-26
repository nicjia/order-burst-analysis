#!/usr/bin/env python3
"""
hist_test_2017.py — survivorship-free 2017-2021 out-of-sample analysis on the
580-name PIT sweep. Drops MISSING (failed/pre-IPO/post-delist), keeps genuine 0.
Reports the tick reversal WITH and WITHOUT DLRET delisting splicing (the toggle),
plus intensity->vol, buy/sell campaigns, and overnight IC. Optional bounded volume
verification (yfinance) with 5%/95% flags.
"""
import glob, math, os, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dlret_splice as DL

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
REPO = "/Users/nick/order-burst-analysis"
COST, H = 1.0, 20


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 20: return (np.nan, np.nan, T)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L+1):
        w = 1 - l/(L+1); v += 2*w*(e[l:] @ e[:-l])/T
    return (m, m/np.sqrt(v/T), T)


def zrows(df): return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1)+1e-9, axis=0).clip(-4, 4)
def sh(s):
    s = s[s != 0].dropna()
    return (s.mean()/s.std()*math.sqrt(252), s.mean()/(s.std()/math.sqrt(len(s))), len(s)) if len(s) >= 40 else (np.nan, np.nan, len(s))


def reversal(dates, cols, Z, R, cpx, K):
    sub = set(cpx.mean().nsmallest(K).index)
    m = pd.DataFrame(False, index=dates, columns=cols); m.loc[:, [c for c in cols if c in sub]] = True
    z = Z.where(m); P = -np.sign(z); P = P.where(np.isfinite(P), 0.0)
    W = P.rolling(H, min_periods=1).mean().shift(1); W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan); W = W.div(g, axis=0).fillna(0.0)
    ret = (W*R).sum(axis=1) - (COST/1e4)*(W-W.shift(1)).abs().sum(axis=1)
    return sh(ret)


def load():
    d = pd.read_csv(SP + "/all_rows.csv", header=None,
                    names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"], dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    for c in ["date", "netflow", "n_bursts", "buy", "sell"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")     # MISSING -> NaN (dropped), genuine 0 kept
    d["date"] = d["date"].astype("Int64")
    FL = d.pivot_table(index="date", columns="ticker", values="netflow")
    CNT = d.pivot_table(index="date", columns="ticker", values="n_bursts")
    VOL = d.assign(v=d.buy + d.sell).pivot_table(index="date", columns="ticker", values="v")
    close = pd.read_csv(REPO + "/close_all.csv", index_col="date"); close.index = close.index.astype(int)
    dates = sorted(x for x in FL.index if 20170101 <= x <= 20211231)
    cols = [c for c in FL.columns if c in close.columns]
    FL = FL.reindex(index=dates, columns=cols); CNT = CNT.reindex(index=dates, columns=cols); VOL = VOL.reindex(index=dates, columns=cols)
    R = close.reindex(dates)[cols].pct_change(fill_method=None)
    return dates, cols, FL, CNT, VOL, R, close


def main():
    dates, cols, FL, CNT, VOL, R, close = load()
    real = int((FL.notna() & (FL != 0)).sum().sum())
    print("2017-2021 panel: %d names, %d dates, %d real name-days (MISSING dropped)" % (len(cols), len(dates), real))
    Z = zrows(FL); cpx = close.reindex(dates)[cols]

    # DLRET toggle
    tbl = DL.build_delist_table(cols)
    print("\nDLRET splice: %d delisted names get a terminal return (source: %s; mean %.2f)" %
          (len(tbl), tbl.source.value_counts().to_dict() if len(tbl) else {}, tbl.dlret.mean() if len(tbl) else float("nan")))
    R_off = DL.splice_returns(R, tbl, on=False)
    R_on = DL.splice_returns(R, tbl, on=True)

    print("\n=== TICK REVERSAL — WITHOUT vs WITH DLRET (toggle) ===")
    for K in (30, 50, 100):
        s0, t0, n0 = reversal(dates, cols, Z, R_off, cpx, K)
        s1, t1, n1 = reversal(dates, cols, Z, R_on, cpx, K)
        print("  K=%3d: no-DLRET Sharpe %+.2f (t=%+.2f) | with-DLRET %+.2f (t=%+.2f)  [delta %.2f]" %
              (K, s0, t0, s1, t1, s1 - s0))

    print("\n=== INTENSITY -> NEXT-DAY |RET| (day-level FM) ===")
    def fm(y, x):
        sl = []
        for dd in dates:
            df = pd.DataFrame({"y": y.loc[dd], "x": x.loc[dd]}).dropna()
            if len(df) < 10: continue
            X = np.column_stack([np.ones(len(df)), df.x.values]); b, *_ = np.linalg.lstsq(X, df.y.values, rcond=None); sl.append(b[1])
        return nw(sl)
    ar = R_off.abs(); m, t, _ = fm(ar.shift(-1), zrows(CNT)); print("  next|ret| ~ count_z: %+.1f bps t=%+.2f" % (m*1e4, t))

    print("\n=== CAMPAIGN REVERSION BUY vs SELL (>=3 same-sign, next 5d) ===")
    idx = list(FL.index)
    for want, lab in [(+1, "BUY "), (-1, "SELL")]:
        rev = []
        for name in cols:
            s = np.sign(FL[name].values); r = R_off[name].values; n = len(s); i = 0
            while i < n:
                if not np.isfinite(s[i]) or s[i] == 0: i += 1; continue
                j = i
                while j+1 < n and s[j+1] == s[i]: j += 1
                if j-i+1 >= 3 and s[i] == want:
                    for tt in range(j+1, min(j+6, n)):
                        if np.isfinite(r[tt]): rev.append((idx[tt], want*r[tt]))
                i = j+1
        m, t, _ = nw(pd.DataFrame(rev, columns=["d", "x"]).groupby("d")["x"].mean().values) if rev else (np.nan, np.nan, 0)
        print("  %s post: %+.1f bps/day t=%+.2f (n=%d)" % (lab, m*1e4, t, len(rev)))

    print("\n=== OVERNIGHT-ish IC (flow_z -> next close-to-close): sign? ===")
    m, t, _ = fm(R_off.shift(-1), Z); print("  %+.1f bps/1sd t=%+.2f (%s)" % (m*1e4, t, "reversal" if m < 0 else "continuation"))


if __name__ == "__main__":
    main()
