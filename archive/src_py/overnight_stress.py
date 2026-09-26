#!/usr/bin/env python3
"""
overnight_stress.py — stress-test the overnight-continuation / intraday-reversal split
(the tug-of-war). Rule out the obvious artifacts before believing it's tradeable:
  (1) EARNINGS GAPS: exclude/winsorize big overnight moves (|ON|>5%) -> does the
      overnight continuation survive on the ordinary days?
  (2) LIQUIDITY: restrict to the most active names (top-K by burst turnover) where
      yfinance opens are reliable and MOO/MOC trading is realistic.
  (3) ROBUST sign: use sign(flow) not z; decile long-short.
  (4) COST curve: net Sharpe of each leg vs per-side cost 0..3 bps.
  (5) COMBINED strategy: overnight-momentum + intraday-reversal (flip at the open),
      the full tug-of-war harvest, with realistic 2-round-trip daily cost.
"""
import math, os
import numpy as np, pandas as pd

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 20: return (np.nan, np.nan, T)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return (m, m / np.sqrt(v / T), T)


def sh(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 30: return (np.nan, np.nan, len(r))
    return (r.mean() / (r.std() + 1e-12) * math.sqrt(252), nw(r)[1], len(r))


def zrows(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


def to_intidx(df):
    df = df.copy(); df.index = [int(pd.Timestamp(x).strftime("%Y%m%d")) for x in df.index]; return df


def load():
    d = pd.read_csv(SP + "/all_rows.csv", header=None,
                    names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"],
                    dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    for c in ["date", "netflow", "n_bursts", "buy", "sell"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = d["date"].astype("Int64")
    FL = d.pivot_table(index="date", columns="ticker", values="netflow")
    TURN = d.assign(v=d.buy + d.sell).pivot_table(index="date", columns="ticker", values="v")
    O = to_intidx(pd.read_parquet(SP + "/opens.parquet"))
    C = to_intidx(pd.read_parquet(SP + "/closes.parquet"))
    dates = sorted(x for x in FL.index if 20170101 <= x <= 20211231 and x in O.index)
    cols = [c for c in FL.columns if c in O.columns]
    FL = FL.reindex(dates, columns=cols); TURN = TURN.reindex(dates, columns=cols)
    O = O.reindex(dates, columns=cols); C = C.reindex(dates, columns=cols)
    ON = O.shift(-1) / C - 1.0
    ID = C.shift(-1) / O.shift(-1) - 1.0
    return dates, cols, FL, TURN, ON, ID


def book(sig, RET, cost=0.0):
    W = sig.sub(sig.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0)
    return (W * RET).sum(axis=1) - cost / 1e4


def main():
    dates, cols, FL, TURN, ON, ID = load()
    print("panel %d names x %d dates\n" % (len(cols), len(dates)))
    fz = zrows(FL)
    # position sign conventions:
    #   overnight CONTINUATION  = follow flow  = +fz
    #   intraday  REVERSAL      = fade flow    = -fz
    print("=== A) EARNINGS-GAP robustness: cap |overnight| at X%, overnight CONTINUATION ===")
    for cap in (None, 0.10, 0.05, 0.03):
        ONc = ON if cap is None else ON.clip(-cap, cap)
        s, t, n = sh(book(fz, ONc, 0.0))   # follow flow overnight
        lab = "raw" if cap is None else "cap|ON|<=%d%%" % int(cap * 100)
        print("  overnight-momentum %-12s Sharpe %+5.2f t=%+.2f" % (lab, s, t))
    # also drop the day entirely if |ON|>5% for that name (set weight 0)
    mask = ON.abs() <= 0.05
    s, t, n = sh(book(fz.where(mask), ON.where(mask), 0.0))
    print("  overnight-momentum drop-gap>5%%  Sharpe %+5.2f t=%+.2f" % (s, t))

    print("\n=== B) LIQUIDITY: top-K names by median burst turnover ===")
    med = TURN.median().sort_values(ascending=False)
    for K in (50, 100, 200, len(cols)):
        keep = list(med.head(K).index)
        onm = sh(book(fz[keep], ON[keep], 0.0))
        idm = sh(book(-fz[keep], ID[keep], 0.0))
        print("  top-%-4d  overnight-mom Sharpe %+5.2f (t=%+.2f) | intraday-rev Sharpe %+5.2f (t=%+.2f)"
              % (K, onm[0], onm[1], idm[0], idm[1]))

    print("\n=== C) ROBUST sign & decile long-short (overnight momentum) ===")
    s, t, n = sh(book(np.sign(FL), ON, 0.0)); print("  sign(flow) overnight  Sharpe %+5.2f t=%+.2f" % (s, t))
    rank = FL.rank(axis=1, pct=True)
    dec = pd.DataFrame(0.0, index=FL.index, columns=FL.columns)
    dec = dec.mask(rank >= 0.9, 1.0).mask(rank <= 0.1, -1.0)
    s, t, n = sh(book(dec, ON, 0.0)); print("  decile L/S overnight  Sharpe %+5.2f t=%+.2f" % (s, t))

    print("\n=== D) COST CURVE (net Sharpe vs per-side bps; each leg round-trips daily) ===")
    print("     per-side:   0.0    0.5    1.0    1.5    2.0")
    for tag, sig, RET in [("overnight-momentum", fz, ON), ("intraday-reversal", -fz, ID)]:
        cells = []
        for c in (0.0, 0.5, 1.0, 1.5, 2.0):
            s, t, n = sh(book(sig, RET, 2 * c))   # round trip = 2*per-side
            cells.append("%+.2f" % s)
        print("  %-20s " % tag + "  ".join(cells))

    print("\n=== E) COMBINED tug-of-war harvest (overnight-mom + intraday-rev), net ===")
    # each day: +fz overnight then -fz intraday. gross/day = ON-leg + ID-leg. cost = 4 side-trades/day.
    for c in (0.0, 0.5, 1.0):
        gross = book(fz, ON, 0.0) + book(-fz, ID, 0.0)
        net = gross - 4 * c / 1e4     # 4 trades/day (close in, open out+in, close out) ~ 2 round trips
        s, t, n = sh(net)
        print("  per-side %.1fbp:  combined Sharpe %+5.2f  t=%+.2f  (mean %+.2f bps/day)"
              % (c, s, t, np.nanmean(net) * 1e4))


if __name__ == "__main__":
    main()
