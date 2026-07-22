#!/usr/bin/env python3
"""
tugofwar_2023.py — replication test of the overnight/intraday decomposition on an
INDEPENDENT sample and an INDEPENDENT signal: the 474-name hidden-execution daily
panel (2023-2024, results/research/hidden_xsec_daily.csv), whose signed flow is
buy-sell of type-5 (hidden) prints rather than aggressive burst flow.

If the tug-of-war (overnight continuation / intraday reversal) is a general property
of informed order flow, it should appear here too. If it does not, the 2017-2021
aggressive-burst result is period- or signal-specific -- which is the honest
resolution of the tension with the paper's 2022-2026 close-to-open null.
"""
import math, os
import numpy as np, pandas as pd

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def ii(df):
    df = df.copy(); df.index = [int(pd.Timestamp(x).strftime("%Y%m%d")) for x in df.index]; return df


def book(sig, RET, cost=0.0):
    W = sig.sub(sig.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0)
    return (W * RET).sum(axis=1) - cost / 1e4


def main():
    d = pd.read_csv(os.path.join(REPO, "results/research/hidden_xsec_daily.csv"))
    d["date"] = d["date"].astype(int)
    FL = d.assign(nf=d.buy - d.sell).pivot_table(index="date", columns="ticker", values="nf")
    COI = d.pivot_table(index="date", columns="ticker", values="COI")
    O = ii(pd.read_parquet(SP + "/opens24.parquet")); C = ii(pd.read_parquet(SP + "/closes24.parquet"))
    dates = sorted(x for x in FL.index if x in O.index)
    cols = [c for c in FL.columns if c in O.columns]
    FL = FL.reindex(dates, columns=cols); COI = COI.reindex(dates, columns=cols)
    O = O.reindex(dates, columns=cols); C = C.reindex(dates, columns=cols)
    ON = O.shift(-1) / C - 1.0
    ID = C.shift(-1) / O.shift(-1) - 1.0
    CC = C.shift(-1) / C - 1.0
    print("hidden-execution panel: %d names x %d dates (2023-2024)\n" % (len(cols), len(dates)))

    for lab, sig in [("hidden net flow (buy-sell)", zrows(FL)), ("COI (conditional order imb.)", zrows(COI))]:
        print("=== %s ===" % lab)
        for tag, RET in [("overnight (close->open)", ON), ("intraday (open->close)", ID),
                         ("full (close->close)", CC)]:
            s, t, n = sh(book(sig, RET))
            print("  follow-flow %-24s Sharpe %+5.2f  t=%+5.2f" % (tag, s, t))
        comb = book(sig, ON) + book(-sig, ID)
        s, t, n = sh(comb)
        print("  COMBINED harvest (o/n mom + intraday rev)  Sharpe %+5.2f t=%+5.2f  (%+.2f bps/day)"
              % (s, t, np.nanmean(comb) * 1e4))
        # year split
        for y in (2023, 2024):
            sub = [dd for dd in dates if y * 10000 <= dd < (y + 1) * 10000]
            s1, t1, _ = sh(book(sig, ON).reindex(sub))
            s2, t2, _ = sh(book(-sig, ID).reindex(sub))
            print("    %d: overnight-mom %+5.2f (t=%+.1f) | intraday-rev %+5.2f (t=%+.1f)" % (y, s1, t1, s2, t2))
        print()

    print("=== cost curve, hidden net flow (per-side bps; each leg round-trips daily) ===")
    sig = zrows(FL)
    print("     per-side:   0.0    0.5    1.0    1.5")
    for tag, s_, RET in [("overnight-momentum", sig, ON), ("intraday-reversal", -sig, ID)]:
        cells = [("%+.2f" % sh(book(s_, RET, 2 * c))[0]) for c in (0.0, 0.5, 1.0, 1.5)]
        print("  %-20s " % tag + "  ".join(cells))


if __name__ == "__main__":
    main()
