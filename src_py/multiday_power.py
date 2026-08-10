#!/usr/bin/env python3
"""
multiday_power.py — re-run the multi-day reversion test with an efficient design.

The published table sorts names cross-sectionally on the day's signed aggressive-hidden flow
and measures the signed return from that day's close out to k trading days, one independent
observation per formation date per horizon. At k=20 that leaves ~25 non-overlapping cohorts
per name-year, which is why the point estimate reaches -8.4 bps with t = -1.1 and the paper
has to record the horizon question as undetermined. The data are not the binding constraint;
the design is.

We therefore rebuild it as a calendar-time portfolio with overlapping holds (Jegadeesh--
Titman). On each day the book holds k cohorts formed on each of the previous k days, each at
1/k weight, so every trading day contributes an observation to a diversified portfolio rather
than every k-th day contributing one to a sparse sort. The k-day cumulative return is k times
the mean daily portfolio return, and inference runs on the daily series with Newey--West
errors -- which is both more efficient and better specified than overlapping raw returns,
whose mechanical autocorrelation the sparse design has to absorb into its standard error.

Weights use the paper's own canonical estimator (cross-sectional demean, unit gross exposure)
so the numbers are comparable to the published table rather than to a re-derived one.

Signal:  COI = (buy - sell) / (buy + sell) of quote-rule-signed aggressive hidden volume,
         from results/research/hidden_xsec_daily.csv
Returns: close-to-close from close_all.csv
"""
import sys
import numpy as np, pandas as pd

SIG = "results/research/hidden_xsec_daily.csv"
PX = "close_all.csv"
HORIZONS = [1, 2, 3, 5, 10, 20]


def nw_t(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan
    e = x - x.mean()
    v = (e * e).sum() / n
    for l in range(1, L + 1):
        v += 2.0 * (1 - l / (L + 1.0)) * (e[l:] * e[:-l]).sum() / n
    se = np.sqrt(v / n)
    return float(x.mean() / se) if se > 0 else np.nan


def zs(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


def weights(sig):
    """Paper's canonical book: cross-sectionally demeaned, unit gross exposure."""
    W = sig.sub(sig.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    return W.div(g, axis=0).fillna(0.0)


def main():
    d = pd.read_csv(SIG, usecols=["ticker", "date", "buy", "sell"])
    d = d[(d.buy + d.sell) > 0]
    d["coi"] = (d.buy - d.sell) / (d.buy + d.sell)
    sig = d.pivot_table(index="date", columns="ticker", values="coi", aggfunc="mean")
    sig.index = pd.to_datetime(sig.index.astype(int).astype(str), format="%Y%m%d")

    px = pd.read_csv(PX)
    px["date"] = pd.to_datetime(px["date"].astype(int).astype(str), format="%Y%m%d")
    px = px.set_index("date").sort_index()
    px = px[[c for c in sig.columns if c in px.columns]].apply(pd.to_numeric, errors="coerce")
    ret = px.pct_change(fill_method=None)

    lo, hi = sig.index.min(), sig.index.max()
    ret = ret.loc[(ret.index >= lo) & (ret.index <= hi + pd.Timedelta(days=45))]
    sig = sig.reindex(columns=ret.columns)
    W = weights(zs(sig)).reindex(ret.index).fillna(0.0)
    RET = ret.fillna(0.0)

    print("panel: %d dates, %d names (signal %s to %s)"
          % (len(sig), sig.shape[1], lo.date(), hi.date()))
    print("\n%-8s %14s %10s %10s %8s" % ("horizon", "cum ret (bps)", "NW t", "daily bps", "n days"))
    print("-" * 56)

    dates = list(RET.index)
    pos = {dt: i for i, dt in enumerate(dates)}
    Wv, Rv = W.to_numpy(), RET.to_numpy()

    out = {}
    for k in HORIZONS:
        daily = np.full(len(dates), np.nan)
        for i in range(k, len(dates)):
            acc, m = 0.0, 0
            for j in range(1, k + 1):
                w = Wv[i - j]
                if np.abs(w).sum() > 0:
                    acc += float(w @ Rv[i]); m += 1
            if m > 0:
                daily[i] = acc / k
        s = daily[np.isfinite(daily)]
        mu_bps = s.mean() * 1e4
        out[k] = (mu_bps * k, nw_t(s), mu_bps, len(s))
        print("%-8s %14.2f %10.2f %10.3f %8d"
              % ("%d day%s" % (k, "" if k == 1 else "s"), *out[k]))

    print("\nPublished sparse-sort comparison (same signal, non-overlapping cohorts):")
    print("%-8s %14s %10s %8s" % ("horizon", "cum ret (bps)", "NW t", "n obs"))
    print("-" * 46)
    for k in HORIZONS:
        fwd = (px.shift(-k) / px - 1.0).reindex(RET.index)
        r = (W * fwd.fillna(0.0)).sum(axis=1)
        r = r[np.isfinite(r) & (r != 0)]
        r = r.iloc[::k]                       # non-overlapping formation dates only
        print("%-8s %14.2f %10.2f %8d"
              % ("%d day%s" % (k, "" if k == 1 else "s"),
                 r.mean() * 1e4, nw_t(r.to_numpy()), len(r)))


if __name__ == "__main__":
    main()
