#!/usr/bin/env python3
"""
hidden_xsec_agg.py — aggregate the full-universe hidden-execution cross-section (M12).

Reads results/hidden_xsec/out/*.csv (one file per ticker; daily rows with 3/15/30-min
markouts, signed hidden buy/sell volume, and midpoint counts) and reports:
  * coverage (names, ticker-days) and the midpoint fraction (M12a);
  * multi-horizon day-clustered markout (each ticker-day = one obs), per-name and pooled;
  * daily hidden-COI -> next-day CLOP information coefficient (per-name + date-clustered pooled).
"""
import glob, math, numpy as np, pandas as pd
from scipy import stats

OUT = "results/hidden_xsec/out"


def main():
    files = glob.glob(f"{OUT}/*.csv")
    D = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if len(d): D.append(d)
    A = pd.concat(D, ignore_index=True)
    for c in ["n", "mk3", "mk15", "mk30", "buy", "sell", "n_mid", "n_sig"]:
        A[c] = pd.to_numeric(A[c], errors="coerce")
    A = A[A["n"].fillna(0) > 0].copy()
    print(f"=== M12 hidden-execution cross-section ===")
    print(f"  names with data: {A['ticker'].nunique()} | ticker-days: {len(A)} "
          f"| date range: {int(A['date'].min())}-{int(A['date'].max())}")

    # M12a — midpoint fraction
    tot_mid = A["n_mid"].sum(); tot_sig = A["n_sig"].sum()
    print(f"\n  M12a midpoint fraction of hidden prints: "
          f"{100*tot_mid/max(tot_mid+tot_sig,1):.1f}% at mid (dropped; unsignable by quote rule)")

    # markout by horizon: per-name day-clustered t (each ticker-day one obs)
    print("\n  Multi-horizon day-clustered markout (per-name t, then cross-name summary):")
    for h in ["mk3", "mk15", "mk30"]:
        tstats, means = [], []
        for tk, g in A.groupby("ticker"):
            x = g[h].dropna()
            if len(x) >= 30 and x.std() > 0:
                tstats.append(x.mean()/(x.std()/math.sqrt(len(x)))); means.append(x.mean())
        tstats = np.array(tstats); means = np.array(means)
        # pooled date-clustered: mean across names per date -> daily series -> t
        piv = A.pivot_table(index="date", columns="ticker", values=h)
        daily = piv.mean(axis=1).dropna()
        pooled_t = daily.mean()/(daily.std()/math.sqrt(len(daily))) if daily.std() > 0 else np.nan
        lab = {"mk3": "3-min", "mk15": "15-min", "mk30": "30-min"}[h]
        print(f"    {lab:6s}: cross-name mean markout={means.mean():+.3f} bps | "
              f"%names day-clustered t>2: {100*(tstats>2).mean():.0f}% | "
              f"pooled date-clustered t={pooled_t:+.2f} ({len(daily)} days)")

    # overnight: daily hidden COI -> next-day CLOP
    try:
        close = pd.read_csv("close_all.csv", index_col="date"); close.index = close.index.astype(int)
        opn = pd.read_csv("open_all.csv", index_col="date"); opn.index = opn.index.astype(int)
        tdays = np.array(sorted(close.index))
        A["COI"] = (A["buy"] - A["sell"]) / (A["buy"] + A["sell"]).replace(0, np.nan)
        # next-day CLOP per (ticker,date)
        def clop_row(r):
            tk = r["ticker"]; dt = int(r["date"])
            if tk not in close.columns or tk not in opn.columns: return np.nan
            j = np.searchsorted(tdays, dt, side="right")
            if j >= len(tdays): return np.nan
            nd = int(tdays[j]); c = close[tk].get(dt, np.nan); o = opn[tk].get(nd, np.nan)
            return (o - c)/c if (np.isfinite(c) and np.isfinite(o) and c > 0) else np.nan
        A["CLOP"] = A.apply(clop_row, axis=1)
        M = A.dropna(subset=["COI", "CLOP"])
        ics = []
        for tk, g in M.groupby("ticker"):
            if len(g) >= 30 and g["COI"].std() > 0:
                ics.append(stats.spearmanr(g["COI"], g["CLOP"]).correlation)
        ics = np.array([i for i in ics if np.isfinite(i)])
        # date-clustered pooled IC: cross-sectional IC per date -> t over days
        day_ic = []
        for dt, g in M.groupby("date"):
            if len(g) >= 10 and g["COI"].std() > 0 and g["CLOP"].std() > 0:
                r = stats.spearmanr(g["COI"], g["CLOP"]).correlation
                if np.isfinite(r): day_ic.append(r)
        day_ic = np.array(day_ic)
        pooled_t = day_ic.mean()/(day_ic.std()/math.sqrt(len(day_ic))) if len(day_ic) > 1 else np.nan
        print("\n  Overnight (daily hidden COI -> next-day CLOP):")
        print(f"    per-name IC: mean={ics.mean():+.4f}, median={np.median(ics):+.4f}, "
              f"%positive={100*(ics>0).mean():.0f}% ({len(ics)} names)")
        print(f"    date-clustered cross-sectional IC: mean={day_ic.mean():+.4f}, "
              f"t={pooled_t:+.2f} ({len(day_ic)} days)")
    except Exception as e:
        print(f"\n  overnight IC skipped: {e}")

    A.to_csv("results/research/hidden_xsec_daily.csv", index=False)
    print("\nsaved: results/research/hidden_xsec_daily.csv")


if __name__ == "__main__":
    main()
