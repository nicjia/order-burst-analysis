#!/usr/bin/env python3
"""
flow_filtered_reversal.py — the one non-redundant idea from the search:
use order flow to SELECT which short-term price reversals to trade.

Economics: a 1-day price move can be (i) liquidity/overreaction -> reverts, or
(ii) informed -> continues. Order-flow bursts help tell them apart:
  * price move OPPOSED by net flow  (up on net selling / down on net buying)
      -> the move was NOT flow-supported; flow points toward reversal -> FADE it.
  * price move CONFIRMED by net flow (up on net buying / down on net selling)
      -> flow-supported / possibly informed -> DON'T fade (may even continue).

So: standard short-term reversal, but keep only names where flow points in the
reversal direction ("flow-confirmed reversals"), vs the anti-set. If the confirmed
set has materially higher Sharpe AND the confirmed-minus-anti spread is significant,
order flow adds genuine, monetizable selection value on top of price reversal.

All day-level, Newey-West t, 1bp/side cost. Also: year-by-year, holding sweep,
spanning vs plain reversal, and a blended book.
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


def bt(W, R, H=1):
    if H > 1:
        W = W.rolling(H, min_periods=1).mean()
    W = W.shift(1)
    W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    return (W * R).sum(axis=1) - (COST / 1e4) * turn


def yrs(ret, dates, tag):
    s = pd.Series(np.asarray(ret, float), index=dates)
    out = []
    for y in (2017, 2018, 2019, 2020, 2021):
        sub = s[(s.index >= y * 10000) & (s.index < (y + 1) * 10000)].values
        sub = sub[np.isfinite(sub)]
        out.append("%d:%+.2f" % (y, sub.mean() / (sub.std() + 1e-12) * math.sqrt(252)) if len(sub) > 30 else "%d:na" % y)
    print("  %-26s " % tag + " ".join(out))


def main():
    dates, cols, FL, R = load()
    print("panel %d names x %d dates\n" % (len(cols), len(dates)))

    fz = zrows(FL)
    rev = -np.sign(R)                          # base short-term (1d) reversal direction
    flow_dir = np.sign(FL)                     # +1 net buying, -1 net selling
    move_dir = np.sign(R)                      # today's price move

    confirm = (flow_dir == -move_dir)          # flow points toward the reversal (opposes the move)
    contra = (flow_dir == move_dir)            # flow confirms the move (informed?)

    print("=== A) short-term (1d) reversal: ALL vs flow-confirmed vs flow-contradicted (H=5) ===")
    base = bt(rev, R, H=5)
    conf = bt(rev.where(confirm), R, H=5)
    anti = bt(rev.where(contra), R, H=5)
    for tag, ret in [("ALL reversals", base), ("flow-CONFIRMED reversals", conf),
                     ("flow-CONTRADICTED reversals", anti)]:
        s, t, n = sharpe(ret); print("  %-30s Sharpe %+5.2f t=%+5.2f" % (tag, s, t))
    # confirmed-minus-contradicted spread (the pure selection value)
    W = (rev.where(confirm).fillna(0) - rev.where(contra).fillna(0))
    s, t, n = sharpe(bt(W, R, H=5))
    print("  %-30s Sharpe %+5.2f t=%+5.2f  <-- selection value" % ("CONFIRMED - CONTRADICTED", s, t))

    print("\n=== B) magnitude of flow matters? keep only |flow_z|>q as the confirmation ===")
    for q in (0.0, 0.5, 1.0, 1.5):
        strong = confirm & (fz.abs() > q)
        s, t, n = sharpe(bt(rev.where(strong), R, H=5))
        print("  confirmed & |flow_z|>%.1f : Sharpe %+5.2f t=%+5.2f" % (q, s, t))

    print("\n=== C) holding-period sweep for flow-confirmed reversal ===")
    for H in (1, 2, 3, 5, 10):
        s, t, n = sharpe(bt(rev.where(confirm), R, H=H)); print("  H=%2d  Sharpe %+5.2f t=%+5.2f" % (H, s, t))

    print("\n=== D) use 5-day price move instead of 1-day (ST_Rev horizon) ===")
    r5 = R.rolling(5, min_periods=3).sum()
    rev5 = -np.sign(r5); conf5 = (np.sign(FL) == np.sign(r5))  # flow opposes the 5d move
    # careful: 'confirm reversal' = flow opposes the move; here reversal dir = -sign(r5), flow opposes move = sign(FL)==-sign(r5)
    conf5 = (np.sign(FL) == -np.sign(r5))
    for tag, mask in [("ALL 5d-rev", None), ("flow-confirmed 5d-rev", conf5)]:
        W = rev5 if mask is None else rev5.where(mask)
        s, t, n = sharpe(bt(W, R, H=5)); print("  %-24s Sharpe %+5.2f t=%+5.2f" % (tag, s, t))

    print("\n=== E) YEAR-BY-YEAR robustness (flow-confirmed 1d reversal, H=5) ===")
    yrs(conf, dates, "flow-confirmed reversal")
    yrs(base, dates, "all reversals (baseline)")

    print("\n=== F) SPANNING: does flow-confirmed reversal beat plain reversal? ===")
    a = pd.Series(np.asarray(conf, float), index=dates)
    b = pd.Series(np.asarray(base, float), index=dates)
    df = pd.concat([a, b], axis=1).dropna(); df.columns = ["conf", "base"]
    beta = np.polyfit(df["base"], df["conf"], 1)[0]
    m, t, _ = nw((df["conf"] - beta * df["base"]).values)
    print("  corr=%.2f  beta=%.2f  alpha=%+.2f bps/day t=%+.2f" %
          (df["conf"].corr(df["base"]), beta, m * 1e4, t))
    print("  (alpha t>2 => flow selection adds return beyond just doing plain reversal)")


if __name__ == "__main__":
    main()
