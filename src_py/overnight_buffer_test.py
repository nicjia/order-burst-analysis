#!/usr/bin/env python3
"""
overnight_buffer_test.py — does the overnight leg survive a 5-minute auction buffer?
(Referee concern 5.) The overnight Sharpe reported in the reversal sample uses Yahoo
official open/close, which embed the opening and closing auction prints and can be stale.
Here we re-measure the identical book on LOBSTER continuous-session midpoints with a
5-minute buffer on each side: sell at the 15:55 mid, buy back at the next day's 9:35 mid.
Genuine overnight-impounded information survives the buffer; an auction-print or
stale-quote artifact does not.

Both legs are computed on the SAME name-days so the comparison is apples-to-apples.
"""
import numpy as np, pandas as pd

SP = "measurements/data"


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    T = len(x)
    if T < 20: return np.nan, np.nan
    mu = x.mean(); e = x - mu; s = (e * e).sum() / T
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (e[l:] * e[:-l]).sum() / T
    return mu, mu / np.sqrt(max(s, 1e-20) / T)


def sh(r):
    m, t = nw(r)
    if not np.isfinite(m): return np.nan, np.nan, 0
    return m / np.nanstd(r) * np.sqrt(252), t, int(np.isfinite(r).sum())


def z(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def book(sig, RET):
    w = sig.where(RET.notna())
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return (w * RET).sum(axis=1, min_count=1)


def main():
    bm = pd.read_csv(SP + "/buf_mids_rows.csv")
    for c in ["mid_open", "mid_0935", "mid_1555", "mid_close"]:
        bm[c] = pd.to_numeric(bm[c], errors="coerce")
    bm["date"] = pd.to_numeric(bm["date"], errors="coerce")
    bm = bm.dropna(subset=["date"])
    bm = bm[bm[["mid_0935", "mid_1555", "mid_close", "mid_open"]].notna().all(axis=1)]
    P = lambda c: bm.pivot_table(index="date", columns="ticker", values=c)
    M0935, M1555, MOPEN, MCLOSE = P("mid_0935"), P("mid_1555"), P("mid_open"), P("mid_close")

    # Sample B signal: HIDDEN net flow, 2023-2024 -- the panel that produced the +2.44
    h = pd.read_csv("results/research/hidden_xsec_daily.csv")
    h["date"] = h["date"].astype(int)
    FL = h.assign(nf=h.buy - h.sell).pivot_table(index="date", columns="ticker", values="nf")

    ii = lambda x: x.set_axis(pd.Index([int(str(v)[:10].replace("-", "")) for v in x.index]))
    O = ii(pd.read_parquet(SP + "/opens26.parquet"))
    C = ii(pd.read_parquet(SP + "/closes26.parquet"))

    dates = sorted(set(FL.index) & set(M1555.index) & set(O.index))
    cols = sorted(set(FL.columns) & set(M1555.columns) & set(O.columns))
    R = lambda X: X.reindex(dates, columns=cols)
    FL, M0935, M1555, MOPEN, MCLOSE, O, C = (R(x) for x in (FL, M0935, M1555, MOPEN, MCLOSE, O, C))

    # candidate overnight definitions
    ON_yahoo = O.shift(-1) / C - 1.0                 # official close -> official open
    ON_mid = MOPEN.shift(-1) / MCLOSE - 1.0          # 15:59:59 mid -> 9:31 mid (no buffer)
    ON_buf = M0935.shift(-1) / M1555 - 1.0           # 15:55 mid -> next 9:35 mid (BUFFERED)

    # restrict every definition to the SAME name-days
    common = ON_yahoo.notna() & ON_mid.notna() & ON_buf.notna()
    ON_yahoo, ON_mid, ON_buf = (X.where(common) for X in (ON_yahoo, ON_mid, ON_buf))
    # the buffer discards 15:55-16:00 and 9:30-9:35; measure what is thereby dropped
    ID_buf = M1555.shift(-1) / M0935.shift(-1) - 1.0

    print("overlap panel: %d names x %d dates (2023-2024), %d name-days"
          % (len(cols), len(dates), int(common.sum().sum())))
    sig = z(FL)
    print("\n=== OVERNIGHT LEG UNDER THREE PRICE CONVENTIONS (identical name-days) ===")
    for tag, RET in [("Yahoo official close->open", ON_yahoo),
                     ("LOBSTER mid 16:00->09:31 (no buffer)", ON_mid),
                     ("LOBSTER mid 15:55->09:35 (BUFFERED)", ON_buf)]:
        s, t, n = sh(book(sig, RET))
        print("  %-38s Sharpe %+5.2f  t=%+5.2f  (n=%d)" % (tag, s, t, n))
    s, t, n = sh(book(sig, ID_buf))
    print("  %-38s Sharpe %+5.2f  t=%+5.2f  (n=%d)" % ("intraday 09:35->15:55 (complement)", s, t, n))

    print("\n=== WHERE THE RETURN LIVES: mean |leg| and mean signed leg (bps) ===")
    for tag, RET in [("close->open (Yahoo)", ON_yahoo), ("16:00->09:31 mid", ON_mid),
                     ("15:55->09:35 mid", ON_buf)]:
        print("  %-22s mean %+7.2f bps   mean|.| %7.2f bps"
              % (tag, 1e4 * RET.stack().mean(), 1e4 * RET.stack().abs().mean()))

    print("\n=== YEAR BY YEAR (buffered leg) ===")
    for yr in (2023, 2024):
        m = [dt for dt in dates if str(dt)[:4] == str(yr)]
        s, t, n = sh(book(sig, ON_buf).reindex(m))
        print("  %d  Sharpe %+5.2f  t=%+5.2f  (n=%d)" % (yr, s, t, n))


if __name__ == "__main__":
    main()
