#!/usr/bin/env python3
"""
explore_flow_price.py — two genuinely different directional angles:

(1) FLOW-vs-PRICE conditioning (accumulation / distribution). The idea: net flow is
    most informative when it OPPOSES the contemporaneous price move.
      distribution : price UP  + net SELLING  -> fade up   (short)   [smart sellers]
      absorption   : price DOWN + net BUYING  -> fade down (long)    [smart buyers]
    Trade only "disagreement" names (flow opposes move) vs "agreement" names.

(2) Does flow add to the STANDARD short-term price reversal (ST_Rev)? Build ST_Rev
    (fade trailing 5d return), the flow reversal, and their combination; then a
    SPANNING regression of flow-reversal returns on ST_Rev returns -> is the alpha
    real and orthogonal? If orthogonal, the combo diversifies and Sharpe rises.

All day-level, Newey-West t, 1bp/side cost.
"""
import math, os
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


def zrows(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


def load():
    d = pd.read_csv(SP + "/all_rows.csv", header=None,
                    names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"],
                    dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    for c in ["date", "netflow", "n_bursts", "buy", "sell"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = d["date"].astype("Int64")
    FL = d.pivot_table(index="date", columns="ticker", values="netflow")
    close = pd.read_csv(REPO + "/close_all.csv", index_col="date"); close.index = close.index.astype(int)
    dates = sorted(x for x in FL.index if 20170101 <= x <= 20211231)
    cols = [c for c in FL.columns if c in close.columns]
    FL = FL.reindex(index=dates, columns=cols)
    R = close.reindex(dates)[cols].pct_change(fill_method=None)
    return dates, cols, FL, R


def bt(W, R, gross=1.0):
    """position DataFrame W (already the desired sign & pre-shift), normalize+trade next day."""
    W = W.shift(1)
    W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0) * gross
    turn = (W - W.shift(1)).abs().sum(axis=1)
    return (W * R).sum(axis=1) - (COST / 1e4) * turn


def main():
    dates, cols, FL, R = load()
    print("panel %d names x %d dates\n" % (len(cols), len(dates)))

    fz = zrows(FL)                      # flow z (higher = buying)
    r1 = R                              # today's return
    r5 = R.rolling(5, min_periods=3).sum()   # trailing 5d return (for ST_Rev)

    print("=== 1) FLOW-vs-PRICE conditioning (H=5 averaging of the position) ===")
    same = np.sign(fz) == np.sign(r1)  # flow agrees with today's move
    opp = np.sign(fz) == -np.sign(r1)  # flow opposes today's move (accum/distribution)
    # base reversal (fade flow), then split by agreement
    P = (-fz).rolling(5, min_periods=1).mean()
    for tag, mask in [("ALL names (base reversal)", None),
                      ("only flow-OPPOSES-price (disagreement)", opp),
                      ("only flow-AGREES-price (agreement)", same)]:
        W = P if mask is None else P.where(mask.reindex_like(P).fillna(False))
        s, t, n = sharpe(bt(W, R))
        print("  fade-flow, %-40s Sharpe %+5.2f t=%+5.2f" % (tag, s, t))
    # pure distribution/absorption directional (fade the price move where flow opposes)
    Wdist = (-np.sign(r1)).where(opp).rolling(5, min_periods=1).mean()
    s, t, n = sharpe(bt(Wdist, R))
    print("  fade PRICE where flow opposes (accum/distr)   Sharpe %+5.2f t=%+5.2f" % (s, t))

    print("\n=== 2) FLOW vs PRICE short-term reversal — standalone & combined ===")
    Wf = (-fz).rolling(5, min_periods=1).mean()               # flow reversal
    Wp = (-zrows(r5))                                          # ST_Rev (fade 5d return)
    retf = bt(Wf, R); retp = bt(Wp, R)
    # combined = average of the two z-signals (re-rank)
    Wc = (-(zrows(FL) + zrows(r5)) / 2).rolling(5, min_periods=1).mean()
    retc = bt(Wc, R)
    for tag, ret in [("flow reversal", retf), ("price ST_Rev (fade 5d)", retp),
                     ("COMBINED (avg z)", retc)]:
        s, t, n = sharpe(ret)
        print("  %-28s Sharpe %+5.2f t=%+5.2f" % (tag, s, t))

    print("\n=== 3) SPANNING: does flow reversal have alpha orthogonal to ST_Rev? ===")
    a = pd.Series(np.asarray(retf, float), index=dates)
    b = pd.Series(np.asarray(retp, float), index=dates)
    df = pd.concat([a, b], axis=1).dropna(); df.columns = ["flow", "price"]
    X = np.column_stack([np.ones(len(df)), df["price"].values])
    coef, *_ = np.linalg.lstsq(X, df["flow"].values, rcond=None)
    resid = df["flow"].values - X @ coef
    alpha_daily = coef[0]; beta = coef[1]
    # NW t on the alpha via residual + intercept series
    m, tstat, _ = nw(df["flow"].values - beta * df["price"].values)
    corr = df["flow"].corr(df["price"])
    print("  corr(flow-rev, price-rev) = %+.2f" % corr)
    print("  flow-rev = alpha + beta*price-rev :  beta=%+.2f  alpha=%+.2f bps/day (t=%+.2f)"
          % (beta, alpha_daily * 1e4, tstat))
    print("  => alpha t>2 means flow reversal is NOT spanned by price reversal (adds info).")

    print("\n=== 4) EQUAL-RISK combo of the two orthogonal legs (risk-parity blend) ===")
    za = (a - a.mean()) / (a.std() + 1e-12); zb = (b - b.mean()) / (b.std() + 1e-12)
    blend = (za + zb) / 2
    s, t, n = sharpe(blend.values)
    print("  risk-parity(flow-rev, price-rev)  Sharpe %+5.2f t=%+5.2f" % (s, t))


if __name__ == "__main__":
    main()
