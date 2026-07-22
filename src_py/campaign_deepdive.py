#!/usr/bin/env python3
"""
campaign_deepdive.py — stress-test the sell-campaign reversion (the one variant that
cleared t>2 in the menu). Questions a referee will ask:
  (a) Is it just market beta?  -> build a DOLLAR-NEUTRAL version (long post-sell,
      short the cross-section / short post-buy) and compare to long-only.
  (b) Is it a 2020-COVID artifact? -> year-by-year Sharpe.
  (c) How does it depend on campaign length, holding period, campaign magnitude?
  (d) Long-short symmetric (fade sell AND fade buy) vs sell-only.
All day-level P&L, Newey-West t, 1bp/side cost.
"""
import math, os, sys
import numpy as np, pandas as pd

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
REPO = "/Users/nick/order-burst-analysis"
COST = 1.0


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 20: return (np.nan, np.nan, T)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return (m, m / np.sqrt(v / T), T)


def sharpe(ret):
    r = np.asarray(ret, float); r = r[np.isfinite(r)]
    if len(r) < 30: return (np.nan, np.nan, len(r))
    return (r.mean() / (r.std() + 1e-12) * math.sqrt(252), nw(r)[1], len(r))


def load():
    d = pd.read_csv(SP + "/all_rows.csv", header=None,
                    names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"],
                    dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    for c in ["date", "netflow", "n_bursts", "buy", "sell"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = d["date"].astype("Int64")
    FL = d.pivot_table(index="date", columns="ticker", values="netflow")
    TOT = (d.assign(v=d.buy + d.sell).pivot_table(index="date", columns="ticker", values="v"))
    close = pd.read_csv(REPO + "/close_all.csv", index_col="date"); close.index = close.index.astype(int)
    dates = sorted(x for x in FL.index if 20170101 <= x <= 20211231)
    cols = [c for c in FL.columns if c in close.columns]
    FL = FL.reindex(index=dates, columns=cols); TOT = TOT.reindex(index=dates, columns=cols)
    R = close.reindex(dates)[cols].pct_change(fill_method=None)
    return dates, cols, FL, TOT, R


def campaign_flag(FL, minrun=3, hold=5, want=-1, mag=None):
    """Return a 0/1 DataFrame: 1 if the name is within `hold` days AFTER a completed
       run of >=minrun consecutive `want`-sign days. If mag given, also require the
       run's mean |imbalance-ish netflow rank| to exceed that quantile (strong campaigns)."""
    dates = list(FL.index)
    F = pd.DataFrame(0.0, index=FL.index, columns=FL.columns)
    signs = np.sign(FL.values)
    arr = FL.values
    for jc, name in enumerate(FL.columns):
        s = signs[:, jc]; n = len(s); i = 0
        while i < n:
            if not np.isfinite(s[i]) or s[i] == 0: i += 1; continue
            j = i
            while j + 1 < n and s[j + 1] == s[i]: j += 1
            L = j - i + 1
            if L >= minrun and s[i] == want:
                ok = True
                if mag is not None:
                    strength = np.nanmean(np.abs(arr[i:j + 1, jc]))
                    ok = strength >= mag[jc]
                if ok:
                    for tt in range(j + 1, min(j + 1 + hold, n)):
                        F.iat[tt, jc] = 1.0
            i = j + 1
    return F


def neutral_book(F_long, R, F_short=None):
    """Dollar-neutral: +1 to flagged-long names, -1 to short set (or the rest of the
       universe if F_short None), normalized to gross 1 each day, trade next day."""
    W = F_long.copy()
    if F_short is not None:
        W = F_long - F_short
    else:
        # short the equal-weighted rest of the tradeable cross-section
        avail = R.notna().astype(float)
        rest = (avail - F_long).clip(lower=0)
        nlong = F_long.sum(axis=1).replace(0, np.nan)
        nrest = rest.sum(axis=1).replace(0, np.nan)
        W = F_long.div(nlong, axis=0).fillna(0) - rest.div(nrest, axis=0).fillna(0)
    W = W.shift(1)
    W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    ret = (W * R).sum(axis=1) - (COST / 1e4) * turn
    return ret


def longonly_book(F_long, R):
    W = F_long.shift(1)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    return (W * R).sum(axis=1) - (COST / 1e4) * turn


def yr(ret, dates):
    s = pd.Series(np.asarray(ret, float), index=dates)
    out = {}
    for y in (2017, 2018, 2019, 2020, 2021):
        sub = s[(s.index >= y * 10000) & (s.index < (y + 1) * 10000)].values
        sub = sub[np.isfinite(sub)]
        out[y] = (sub.mean() / (sub.std() + 1e-12) * math.sqrt(252)) if len(sub) > 30 else np.nan
    return out


def main():
    dates, cols, FL, TOT, R = load()
    print("panel %d names x %d dates\n" % (len(cols), len(dates)))

    print("=== A) SELL-campaign: long-only vs market-NEUTRAL (minrun3, hold5) ===")
    Fsell = campaign_flag(FL, 3, 5, want=-1)
    Fbuy = campaign_flag(FL, 3, 5, want=+1)
    lo = longonly_book(Fsell, R)
    nb = neutral_book(Fsell, R)                 # long post-sell, short the rest
    ls = neutral_book(Fsell, R, F_short=Fbuy)   # long post-sell, short post-buy
    for tag, ret in [("long-only post-sell (has market beta)", lo),
                     ("NEUTRAL long post-sell / short rest", nb),
                     ("LONG-SHORT post-sell vs post-buy", ls)]:
        s, t, n = sharpe(ret)
        print("  %-42s Sharpe %+5.2f  t=%+5.2f" % (tag, s, t))

    print("\n=== B) YEAR-BY-YEAR (is it a 2020 artifact?) ===")
    for tag, ret in [("long-only", lo), ("neutral(vs rest)", nb), ("long-short", ls)]:
        y = yr(ret, dates)
        print("  %-18s " % tag + " ".join("%d:%+.2f" % (k, v) for k, v in y.items()))

    print("\n=== C) CAMPAIGN-LENGTH x HOLDING (neutral long post-sell/short rest) ===")
    print("        hold=1   hold=3   hold=5   hold=10")
    for mr in (2, 3, 4, 5):
        cells = []
        for h in (1, 3, 5, 10):
            s, t, n = sharpe(neutral_book(campaign_flag(FL, mr, h, want=-1), R))
            cells.append("%+.2f(%+.1f)" % (s, t))
        print("  minrun%d  " % mr + "  ".join(cells))

    print("\n=== D) STRONG vs weak sell-campaigns (magnitude split, minrun3 hold5, neutral) ===")
    q = np.nanquantile(np.abs(FL.values), 0.5, axis=0)  # per-name median |netflow|
    Fstrong = campaign_flag(FL, 3, 5, want=-1, mag=q)
    s, t, n = sharpe(neutral_book(Fstrong, R))
    print("  strong sell-campaigns only (|flow| > own median): Sharpe %+.2f t=%+.2f" % (s, t))

    print("\n=== E) BUY-side symmetric check (fade buy-campaigns, neutral) ===")
    s, t, n = sharpe(neutral_book(Fbuy, R))   # long post-buy/short rest = momentum test
    print("  long post-BUY / short rest (expect ~0 or negative): Sharpe %+.2f t=%+.2f" % (s, t))


if __name__ == "__main__":
    main()
