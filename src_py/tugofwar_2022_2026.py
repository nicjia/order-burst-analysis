#!/usr/bin/env python3
"""
tugofwar_2022_2026.py — the referee's decisive test: run the Section 12
overnight/intraday decomposition on the paper's PRIMARY 2022-2026 aggressive burst
panel (results/research/coi_panel_ungated_2026.csv, 482 names), the same sample where
Section 10 reports a reversal-signed / null overnight relation.

Two outcomes, both publishable:
  * overnight leg ~ +2.4  -> Section 10's null was a same-day-normalization artifact
  * overnight leg flat/neg -> the effect is period-specific and the "replicates across
                              period and reconstruction" claim must be retired.

Also reports the full normalization ladder so the COI sign question can be settled:
Section 10 reports a significantly NEGATIVE count-weighted COI IC (-2.68), whereas
Sample A/B showed positive-but-insignificant COI-like rows. Here we compute, on the
same panel Section 10 uses, both the volume-weighted and count-weighted COI overnight
legs so the sign is measured rather than asserted.
"""
import math, os, sys
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


def z(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


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


def main():
    d = pd.read_csv(SP + "/coi_panel_ungated_2026.csv")
    d["Date"] = d["Date"].astype(int)
    piv = lambda c: d.pivot_table(index="Date", columns="Ticker", values=c)
    BV, SV = piv("buy_vol"), piv("sell_vol")
    NB, NS = piv("n_buy"), piv("n_sell")
    FL = BV - SV                       # net signed burst volume (the Section 12 signal)
    TOT = BV + SV
    CNTIMB = NB - NS                   # count imbalance
    O = ii(pd.read_parquet(SP + "/opens26.parquet")); C = ii(pd.read_parquet(SP + "/closes26.parquet"))
    dates = sorted(set(FL.index) & set(O.index))
    cols = [c for c in FL.columns if c in O.columns]
    FL, TOT, CNTIMB, NB, NS = (X.reindex(dates, columns=cols) for X in (FL, TOT, CNTIMB, NB, NS))
    O = O.reindex(dates, columns=cols); C = C.reindex(dates, columns=cols)
    ON = O.shift(-1) / C - 1.0
    ID = C.shift(-1) / O.shift(-1) - 1.0
    CC = C.shift(-1) / C - 1.0
    print("PRIMARY 2022-2026 aggressive panel: %d names x %d dates\n" % (len(cols), len(dates)))

    print("=== A) THE DECISIVE TEST: three legs, follow-flow convention (raw signed volume z) ===")
    sig = z(FL)
    for tag, RET in [("overnight (close->open)", ON), ("intraday  (open->close)", ID),
                     ("close-to-close", CC)]:
        s, t, n = sh(book(sig, RET))
        print("  %-26s Sharpe %+5.2f  t=%+5.2f  (n=%d)" % (tag, s, t, n))

    print("\n=== B) NORMALIZATION LADDER (overnight leg) ===")
    typ = TOT.rolling(20, min_periods=5).mean()
    variants = [
        ("raw signed volume z", z(FL)),
        ("raw z, volume-quintile neutral", bucket_neutral(z(FL), typ)),
        ("net flow / own 20d volume", z(FL / typ.replace(0, np.nan))),
        ("net flow / same-day volume (COI, vol-wtd)", z(FL / TOT.replace(0, np.nan))),
        ("count imbalance / same-day count (COI, cnt-wtd)", z(CNTIMB / (NB + NS).replace(0, np.nan))),
        ("sign of net flow only", np.sign(FL)),
    ]
    for nm, s_ in variants:
        a, b, n = sh(book(s_, ON))
        print("  %-48s Sharpe %+5.2f  t=%+5.2f" % (nm, a, b))

    print("\n=== C) YEAR-BY-YEAR (overnight leg, raw signed volume z) ===")
    r = book(sig, ON); srs = pd.Series(np.asarray(r, float), index=dates)
    for y in range(2022, 2027):
        v = srs[(srs.index >= y * 10000) & (srs.index < (y + 1) * 10000)].values
        v = v[np.isfinite(v)]
        if len(v) > 30:
            print("  %d: Sharpe %+5.2f  (%d days)" % (y, v.mean() / (v.std() + 1e-12) * math.sqrt(252), len(v)))
        else:
            print("  %d: n=%d (partial year, skipped)" % (y, len(v)))

    print("\n=== D) ROBUSTNESS (overnight leg) ===")
    for cap in (0.10, 0.05):
        s, t, n = sh(book(sig, ON.clip(-cap, cap)))
        print("  cap |overnight| <= %2d%%          Sharpe %+5.2f t=%+5.2f" % (int(cap * 100), s, t))
    med = TOT.median().sort_values(ascending=False)
    for K in (100, 200):
        keep = list(med.head(K).index)
        s, t, n = sh(book(z(FL[keep]), ON[keep]))
        print("  top-%-3d by burst turnover      Sharpe %+5.2f t=%+5.2f" % (K, s, t))
    for lag in (0, 1, 2):
        s, t, n = sh(book(z(FL).shift(lag), ON))
        print("  flow lag %d                     Sharpe %+5.2f t=%+5.2f" % (lag, s, t))

    print("\n=== E) COST CURVE (overnight leg; round trip = 2x per-side) ===")
    print("     per-side:   0.0    0.5    1.0    1.5    2.0")
    cells = [("%+.2f" % sh(book(sig, ON, 2 * c))[0]) for c in (0.0, 0.5, 1.0, 1.5, 2.0)]
    print("  overnight-momentum  " + "  ".join(cells))

    print("\n=== F) PERMUTATION NULL (1000 draws, cross-name shuffle within day) ===")
    rng = np.random.default_rng(0); real = sh(book(sig, ON))[0]; ps = []
    V = sig.values
    for k in range(1000):
        Vs = V.copy()
        for i in range(Vs.shape[0]):
            row = Vs[i]; m = np.isfinite(row); idx = np.where(m)[0]
            if len(idx) > 1: row[idx] = row[rng.permutation(idx)]
        ps.append(sh(book(pd.DataFrame(Vs, index=sig.index, columns=sig.columns), ON))[0])
    ps = np.array(ps); ps = ps[np.isfinite(ps)]
    print("  real %+.2f | null mean %+.2f sd %.2f | p(null>=real) = %.4f | z = %.1f"
          % (real, ps.mean(), ps.std(), (ps >= real).mean(), (real - ps.mean()) / (ps.std() + 1e-12)))


if __name__ == "__main__":
    main()
