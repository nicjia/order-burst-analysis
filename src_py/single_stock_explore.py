#!/usr/bin/env python3
"""
single_stock_explore.py — the one axis not yet tested: PER-STOCK TIME-SERIES
predictability. Every earlier test was cross-sectional and dollar-neutral, which
z-scores each day across names and therefore throws away a stock's own history.
Here the signal is a stock's net burst flow relative to ITS OWN trailing
distribution, and the strategy trades that single name.

Three stages, escalating only if the previous one works:
  (1) DEEP DIVE on a handful of liquid names: own-flow TS-z -> next close-to-close,
      overnight, intraday. Continuation and reversal.
  (2) SCALE to all names: distribution of per-stock t-statistics against the null.
      If flow carries no per-stock information, ~5% of names exceed |t|>2 and the
      t-distribution is centered at zero.
  (3) SELECTION TEST (the one that matters): split the sample in half. Do the names
      that worked in the first half keep working in the second? A tradeable
      per-stock strategy requires that in-sample skill predicts out-of-sample skill.
      If it does not, stage-2 winners are just the right tail of noise.
"""
import math, os, sys
import numpy as np, pandas as pd

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = 60          # trailing window for the own-history z-score
COST = 1.0        # bps per side


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 30: return (np.nan, np.nan)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return m, m / np.sqrt(v / T)


def sh(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 60: return (np.nan, np.nan, len(r))
    return (r.mean() / (r.std() + 1e-12) * math.sqrt(252), nw(r)[1], len(r))


def ii(df):
    df = df.copy(); df.index = [int(pd.Timestamp(x).strftime("%Y%m%d")) for x in df.index]; return df


def load():
    d = pd.read_csv(SP + "/all_rows.csv", header=None,
                    names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"],
                    dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    for c in ["date", "netflow", "buy", "sell"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = d["date"].astype("Int64")
    FL = d.pivot_table(index="date", columns="ticker", values="netflow")
    TOT = d.assign(v=d.buy + d.sell).pivot_table(index="date", columns="ticker", values="v")
    O = ii(pd.read_parquet(SP + "/opens.parquet")); C = ii(pd.read_parquet(SP + "/closes.parquet"))
    dates = sorted(set(FL.index) & set(O.index))
    cols = [c for c in FL.columns if c in O.columns]
    FL = FL.reindex(dates, columns=cols); TOT = TOT.reindex(dates, columns=cols)
    O = O.reindex(dates, columns=cols); C = C.reindex(dates, columns=cols)
    return dates, cols, FL, TOT, O, C


def tsz(s, win=WIN):
    """own-history z-score, strictly trailing (no look-ahead)."""
    m = s.rolling(win, min_periods=20).mean()
    sd = s.rolling(win, min_periods=20).std()
    return ((s - m) / (sd + 1e-9)).clip(-4, 4)


def one_stock(tk, FL, O, C, verbose=True):
    """returns dict of Sharpe/t for each leg, continuation convention."""
    f = FL[tk]; o = O[tk]; c = C[tk]
    zz = tsz(f)
    ON = o.shift(-1) / c - 1.0
    ID = c.shift(-1) / o.shift(-1) - 1.0
    CC = c.shift(-1) / c - 1.0
    out = {}
    for lab, RET in [("overnight", ON), ("intraday", ID), ("close-close", CC)]:
        pos = np.sign(zz)                     # +1 follow flow, formed at close t
        pnl = pos * RET - (COST / 1e4) * (pos - pos.shift(1)).abs()
        s, t, n = sh(pnl.values)
        out[lab] = (s, t, n)
        if verbose:
            print("    %-12s follow-flow Sharpe %+5.2f  t=%+5.2f  (n=%d)" % (lab, s, t, n))
    return out


def main():
    dates, cols, FL, TOT, O, C = load()
    print("panel: %d names x %d dates\n" % (len(cols), len(dates)))

    print("=" * 78)
    print("STAGE 1 — single-stock deep dive (own-flow time-series z, %dd window)" % WIN)
    print("=" * 78)
    liq = TOT.median().sort_values(ascending=False)
    picks = [t for t in ["F", "BAC", "AAPL", "T", "INTC"] if t in cols][:5]
    for tk in picks:
        print("  %s  (median daily burst volume %,.0f)".replace(",", "") % (tk, liq.get(tk, np.nan)))
        one_stock(tk, FL, O, C)
        print()

    print("=" * 78)
    print("STAGE 2 — scale to all names: distribution of per-stock t-statistics")
    print("=" * 78)
    rows = []
    for tk in cols:
        if FL[tk].notna().sum() < 300: continue
        r = one_stock(tk, FL, O, C, verbose=False)
        rows.append({"ticker": tk, **{f"{k}_s": v[0] for k, v in r.items()},
                     **{f"{k}_t": v[1] for k, v in r.items()}})
    df = pd.DataFrame(rows).dropna()
    print("  %d names with sufficient history\n" % len(df))
    print("  %-13s %8s %8s %9s %9s %9s" % ("leg", "mean t", "med t", "%|t|>2", "%t>+2", "%t<-2"))
    for leg in ["overnight", "intraday", "close-close"]:
        t = df[f"{leg}_t"]
        print("  %-13s %+8.2f %+8.2f %8.1f%% %8.1f%% %8.1f%%" %
              (leg, t.mean(), t.median(), 100 * (t.abs() > 2).mean(),
               100 * (t > 2).mean(), 100 * (t < -2).mean()))
    print("\n  (under the null: mean t=0, ~5%% with |t|>2 split evenly between tails)")
    df.to_csv(SP + "/per_stock_ts.csv", index=False)

    print("\n" + "=" * 78)
    print("STAGE 3 — SELECTION TEST: does first-half skill survive into the second half?")
    print("=" * 78)
    mid = dates[len(dates) // 2]
    print("  split at %d (%d days each side)\n" % (mid, len(dates) // 2))
    d1 = [d for d in dates if d < mid]; d2 = [d for d in dates if d >= mid]
    res = []
    for tk in df.ticker:
        f = FL[tk]; o = O[tk]; c = C[tk]; zz = tsz(f)
        ON = o.shift(-1) / c - 1.0
        pos = np.sign(zz)
        pnl = (pos * ON - (COST / 1e4) * (pos - pos.shift(1)).abs())
        s1, t1, _ = sh(pnl.reindex(d1).values)
        s2, t2, _ = sh(pnl.reindex(d2).values)
        res.append((tk, s1, s2))
    r = pd.DataFrame(res, columns=["ticker", "sh1", "sh2"]).dropna()
    print("  overnight leg, first half vs second half (n=%d names)" % len(r))
    print("    rank correlation of Sharpe across halves: %+.3f" % r.sh1.corr(r.sh2, method="spearman"))
    top = r.nlargest(50, "sh1")
    bot = r.nsmallest(50, "sh1")
    print("    top-50 by 1st-half Sharpe -> 2nd-half mean Sharpe %+.2f" % top.sh2.mean())
    print("    bot-50 by 1st-half Sharpe -> 2nd-half mean Sharpe %+.2f" % bot.sh2.mean())
    print("    all names               -> 2nd-half mean Sharpe %+.2f" % r.sh2.mean())
    print("\n  => a positive rank correlation and a top-minus-bottom gap mean per-stock")
    print("     selection is real and tradeable. Near zero means stage-2 winners are noise.")


if __name__ == "__main__":
    main()
