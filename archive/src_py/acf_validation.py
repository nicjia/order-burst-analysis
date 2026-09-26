#!/usr/bin/env python3
"""
acf_validation.py — three validation tests on the signed-flow autocorrelation that
underlies the alpha=1.86 metaorder claim.

IMPORTANT HONESTY NOTE built into this script: the Lillo-Mike-Farmer relation
gamma = alpha - 1 is derived for the TRADE-level order-sign series (tick-by-tick).
What we have is DAILY aggregated net flow. The daily autocorrelation is a related but
DISTINCT object; the alpha we recover is a daily-frequency analog, not the literature's
trade-level tail exponent. These tests check whether that daily object is (1) real and
not a coding artifact, (2) regime-dependent, (3) stable over time -- which is what a
physical signature should be. They do NOT prove the daily alpha equals the trade-level
alpha. We compute the autocorrelation of the SCALE-FREE imbalance = netflow/(buy+sell),
which is split-robust and the cleanest daily analog of order sign.
"""
import math, os
import numpy as np, pandas as pd

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAGS = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40])


def load_1721():
    d = pd.read_csv(SP + "/all_rows.csv", header=None,
                    names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"],
                    dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    for c in ["date", "netflow", "buy", "sell"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = d["date"].astype("Int64")
    nf = d.pivot_table(index="date", columns="ticker", values="netflow")
    tot = d.assign(v=d.buy + d.sell).pivot_table(index="date", columns="ticker", values="v")
    return (nf / tot.replace(0, np.nan))


def load_2226():
    d = pd.read_csv(SP + "/coi_panel_ungated_2026.csv")
    d["Date"] = d["Date"].astype(int)
    nf = (d.pivot_table(index="Date", columns="Ticker", values="buy_vol")
          - d.pivot_table(index="Date", columns="Ticker", values="sell_vol"))
    tot = (d.pivot_table(index="Date", columns="Ticker", values="buy_vol")
           + d.pivot_table(index="Date", columns="Ticker", values="sell_vol"))
    return (nf / tot.replace(0, np.nan))


def acf_curve(IMB, names=None, lags=LAGS, minobs=200):
    cols = names if names is not None else IMB.columns
    out = []
    for L in lags:
        v = [IMB[t].dropna().autocorr(int(L)) for t in cols
             if t in IMB.columns and IMB[t].notna().sum() > minobs]
        out.append(np.nanmean(v))
    return np.array(out)


def fit_alpha(C, lags=LAGS):
    ok = np.isfinite(C) & (C > 1e-4)
    if ok.sum() < 4: return (np.nan, np.nan, np.nan)
    slope, _ = np.polyfit(np.log(lags[ok]), np.log(C[ok]), 1)
    r2 = np.corrcoef(np.log(lags[ok]), np.log(C[ok]))[0, 1] ** 2
    gamma = -slope
    return (gamma, gamma + 1.0, r2)


def main():
    imb1 = load_1721(); imb2 = load_2226()

    print("=" * 78)
    print("TEST 1 — SHUFFLE PLACEBO (destroy the timeline, keep the distribution)")
    print("=" * 78)
    C = acf_curve(imb1)
    g, a, r2 = fit_alpha(C)
    print("  REAL 2017-2021:  C(1)=%.3f  gamma=%.3f  alpha=%.2f  R^2=%.3f" % (C[0], g, a, r2))
    rng = np.random.default_rng(0)
    sh = imb1.copy()
    for c in sh.columns:
        col = sh[c].values; m = np.isfinite(col); idx = np.where(m)[0]
        col[idx] = col[rng.permutation(idx)]         # permute within name: kills serial structure
    Cs = acf_curve(sh)
    gs, as_, r2s = fit_alpha(Cs)
    print("  SHUFFLED:        C(1)=%.3f  gamma=%.3f  alpha=%s  R^2=%.3f"
          % (Cs[0], gs, ("%.2f" % as_ if np.isfinite(as_) else "n/a"), r2s))
    print("  shuffled C(tau): " + " ".join("%+.3f" % x for x in Cs))
    print("  => real C(1)=%.3f collapses to ~%.3f; power law %s after shuffle."
          % (C[0], Cs[0], "SURVIVES (BUG!)" if r2s > 0.8 and Cs[0] > 0.05 else "destroyed (correct)"))

    print("\n" + "=" * 78)
    print("TEST 2 — REGIME SPLIT (large-tick / low-price vs small-tick / high-price)")
    print("=" * 78)
    close = pd.read_parquet(SP + "/closes26.parquet")
    px = {c: close[c].dropna().mean() for c in close.columns if close[c].notna().sum() > 100}
    for nm in ["NVDA", "JPM", "TSLA", "AAPL"]:
        if nm in imb2.columns:
            C = acf_curve(imb2, [nm], minobs=200)
            g, a, r2 = fit_alpha(C)
            print("  %-5s (avg px $%7.2f, full 2022-2026):  alpha=%.2f  R^2=%.3f  [single-name, noisy]"
                  % (nm, px.get(nm, float("nan")), a, r2))
    ser = pd.Series(px)
    lo = list(ser[ser <= ser.quantile(0.33)].index)   # low price = large-tick
    hi = list(ser[ser >= ser.quantile(0.67)].index)   # high price = small-tick
    for lab, grp in [("low-price / LARGE-tick tercile", lo), ("high-price / small-tick tercile", hi)]:
        C = acf_curve(imb2, grp)
        g, a, r2 = fit_alpha(C)
        print("  %-34s (n=%d):  alpha=%.2f  R^2=%.3f" % (lab, len(grp), a, r2))

    print("\n" + "=" * 78)
    print("TEST 3 — YEAR-BY-YEAR STABILITY (a physical law should not drift with regime)")
    print("=" * 78)
    print("  %-6s %8s %8s %8s   %s" % ("year", "alpha", "gamma", "R^2", "market"))
    def year_alpha(IMB, y):
        sub = IMB[(IMB.index >= y * 10000) & (IMB.index < (y + 1) * 10000)]
        if len(sub) < 120: return None
        C = acf_curve(sub, minobs=100)
        return fit_alpha(C)
    tags = {2017: "bull", 2018: "correction", 2019: "bull", 2020: "COVID crash+rally",
            2021: "bull", 2022: "bear", 2023: "recovery", 2024: "bull", 2025: "bull"}
    for panel in (imb1, imb2):
        for y in sorted(set(panel.index // 10000)):
            r = year_alpha(panel, y)
            if r and np.isfinite(r[1]):
                print("  %-6d %8.2f %8.3f %8.3f   %s" % (y, r[1], r[0], r[2], tags.get(y, "")))


if __name__ == "__main__":
    main()
