#!/usr/bin/env python3
"""
hidden_audit_and_lead.py — two jobs.

(1) MARKOUT PANEL AUDIT. The published hidden cross-section reports mk3 = +1.62 bps
    with a date-clustered t of 3.38, and mk30 with t = 15.23. The t-jump is not a
    formula inconsistency: ~22 name-days out of 221,261 carry impossible values
    (mk3 as low as -114,627 bps = -1146%), which inflate the mk3/mk15 variance and
    deflate their t. Cleaned, the effect is LARGER and far more consistent. Report
    the cleaned panel with Newey-West on the daily series (handles serial
    correlation, which an equal-weight daily-mean t ignores).

(2) THE TRADEABLE LEAD. Hidden-execution flow over 2023-2024 gives an overnight
    continuation (+2.44, t=3.36) that aggressive flow over the SAME window does not.
    Run the full battery that killed/kept previous candidates: permutation null,
    lag decay, gap exclusion, liquidity, cost curve, year split, and a size-neutral
    variant. This decides whether it is a signal or a Yahoo-open artifact.
"""
import math, os
import numpy as np, pandas as pd

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 20: return (np.nan, np.nan)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return m, m / np.sqrt(v / T)


def sh(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 30: return (np.nan, np.nan)
    return (r.mean() / (r.std() + 1e-12) * math.sqrt(252), nw(r)[1])


def z(df): return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)
def ii(df):
    df = df.copy(); df.index = [int(pd.Timestamp(x).strftime("%Y%m%d")) for x in df.index]; return df


def book(sig, RET, cost=0.0):
    W = sig.sub(sig.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0)
    return (W * RET).sum(axis=1) - cost / 1e4


def bucket_neutral(sig, size, q=5):
    out = sig.copy() * np.nan
    rk = size.rank(axis=1, pct=True)
    for i in range(q):
        m = (rk > i / q) & (rk <= (i + 1) / q)
        sub = sig.where(m)
        out = out.fillna(sub.sub(sub.mean(axis=1), axis=0))
    return out


def part1(d):
    print("=" * 78)
    print("(1) HIDDEN MARKOUT PANEL — outlier contamination audit")
    print("=" * 78)
    print("%-6s %-20s %8s %9s %9s" % ("col", "treatment", "mean", "t_daily", "t_NW"))
    for col in ["mk3", "mk15", "mk30"]:
        for lab, sub in [("raw (as published)", d), ("drop |x|>1000bp", d[d[col].abs() <= 1000])]:
            dm = sub.groupby("date")[col].mean()
            t_d = dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm)))
            _, t_nw = nw(dm.values)
            print("%-6s %-20s %+8.3f %+9.2f %+9.2f" % (col, lab, sub[col].mean(), t_d, t_nw))
        print()
    cl = d[(d.mk3.abs() <= 1000) & (d.mk15.abs() <= 1000) & (d.mk30.abs() <= 1000)]
    print("cleaned term structure (all three horizons finite, %d of %d rows kept):" % (len(cl), len(d)))
    for col in ["mk3", "mk15", "mk30"]:
        dm = cl.groupby("date")[col].mean(); _, t_nw = nw(dm.values)
        print("   %-5s mean %+.3f bps   NW t=%+.2f" % (col, cl[col].mean(), t_nw))
    print("\n=> the published mk3 t=3.38 is deflated by 22 corrupt rows (0.01%% of panel);")
    print("   cleaned, the footprint is larger and much more consistent. The panel needs")
    print("   an explicit outlier rule before any of these numbers are quoted.\n")


def part2(d):
    print("=" * 78)
    print("(2) TRADEABLE LEAD — hidden-flow overnight continuation, 2023-2024")
    print("=" * 78)
    FL = d.assign(nf=d.buy - d.sell).pivot_table(index="date", columns="ticker", values="nf")
    TOT = d.assign(v=d.buy + d.sell).pivot_table(index="date", columns="ticker", values="v")
    O = ii(pd.read_parquet(SP + "/opens24.parquet")); C = ii(pd.read_parquet(SP + "/closes24.parquet"))
    dates = sorted(set(FL.index) & set(O.index)); cols = [c for c in FL.columns if c in O.columns]
    FL = FL.reindex(dates, columns=cols); TOT = TOT.reindex(dates, columns=cols)
    O = O.reindex(dates, columns=cols); C = C.reindex(dates, columns=cols)
    ON = O.shift(-1) / C - 1.0
    sig = z(FL)
    base = sh(book(sig, ON))
    print("  baseline overnight (follow-flow)        Sharpe %+5.2f  t=%+5.2f" % base)

    print("\n  -- permutation null (1000 draws, cross-name shuffle within day) --")
    rng = np.random.default_rng(0); V = sig.values; ps = []
    for k in range(1000):
        Vs = V.copy()
        for i in range(Vs.shape[0]):
            row = Vs[i]; idx = np.where(np.isfinite(row))[0]
            if len(idx) > 1: row[idx] = row[rng.permutation(idx)]
        ps.append(sh(book(pd.DataFrame(Vs, index=sig.index, columns=sig.columns), ON))[0])
    ps = np.array(ps); ps = ps[np.isfinite(ps)]
    print("     real %+.2f | null %+.2f (sd %.2f) | p=%.4f | z=%.1f"
          % (base[0], ps.mean(), ps.std(), (ps >= base[0]).mean(), (base[0] - ps.mean()) / (ps.std() + 1e-12)))

    print("\n  -- lag decay (fresh information?) --")
    for lag in (0, 1, 2, 3):
        s, t = sh(book(sig.shift(lag), ON)); print("     lag %d  Sharpe %+5.2f t=%+5.2f" % (lag, s, t))

    print("\n  -- gap exclusion (earnings?) --")
    for cap in (0.10, 0.05, 0.03):
        s, t = sh(book(sig, ON.clip(-cap, cap)))
        print("     cap |overnight|<=%2d%%  Sharpe %+5.2f t=%+5.2f" % (int(cap * 100), s, t))
    m = ON.abs() <= 0.05
    s, t = sh(book(sig.where(m), ON.where(m))); print("     drop gaps>5%%          Sharpe %+5.2f t=%+5.2f" % (s, t))

    print("\n  -- liquidity / size --")
    med = TOT.median().sort_values(ascending=False)
    for K in (100, 200, len(cols)):
        keep = list(med.head(K).index)
        s, t = sh(book(z(FL[keep]), ON[keep])); print("     top-%-3d Sharpe %+5.2f t=%+5.2f" % (K, s, t))
    typ = TOT.rolling(20, min_periods=5).mean()
    s, t = sh(book(bucket_neutral(sig, typ), ON)); print("     size-quintile neutral  Sharpe %+5.2f t=%+5.2f" % (s, t))

    print("\n  -- year split --")
    r = pd.Series(np.asarray(book(sig, ON), float), index=dates)
    for y in (2023, 2024):
        v = r[(r.index >= y * 10000) & (r.index < (y + 1) * 10000)].values; v = v[np.isfinite(v)]
        print("     %d  Sharpe %+5.2f (n=%d)" % (y, v.mean() / (v.std() + 1e-12) * math.sqrt(252), len(v)))

    print("\n  -- cost curve (round trip = 2x per-side) --")
    cells = [("%+.2f" % sh(book(sig, ON, 2 * c))[0]) for c in (0.0, 0.5, 1.0, 1.5, 2.0)]
    print("     per-side 0/0.5/1/1.5/2 bp: " + "  ".join(cells))


def main():
    d = pd.read_csv(os.path.join(REPO, "results/research/hidden_xsec_daily.csv"))
    d["date"] = d["date"].astype(int)
    part1(d)
    part2(d)


if __name__ == "__main__":
    main()
