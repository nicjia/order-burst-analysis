#!/usr/bin/env python3
"""
st_rev_regression.py — referee fix 4c: add the Short-Term Reversal factor (ST_Rev,
Ken French) to the FF5+MOM alpha regression of the tick-constrained burst-flow
reversal. Because the strategy IS a short-horizon reversal, ST_Rev is the single
most relevant control; this reports whether the alpha survives it.

Reuses the canonical m7 harness for the strategy returns, downloads the ST_Rev
daily factor from Ken French (cluster has internet), and runs an OLS with
Newey-West (HAC) standard errors on FF5+MOM and FF5+MOM+ST_Rev, for both the
full-sample and the walk-forward OOS series.
"""
import io, os, math, urllib.request, zipfile
import numpy as np, pandas as pd
import m7_reversal_baseline as m7

FF5 = "data_factors/F-F_Research_Data_5_Factors_2x3_daily.csv"
MOM = "data_factors/F-F_Momentum_Factor_daily.csv"
STREV_CSV = "data_factors/F-F_ST_Reversal_Factor_daily.csv"
STREV_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
             "F-F_ST_Reversal_Factor_daily_CSV.zip")


def parse_french(path, names):
    rows = []
    for line in open(path):
        p = line.strip().split(",")
        if len(p) < 2: continue
        d = p[0].strip()
        if len(d) == 8 and d.isdigit():
            try: vals = [float(x) for x in p[1:1 + len(names)]]
            except ValueError: continue
            if len(vals) == len(names): rows.append([int(d)] + vals)
    return pd.DataFrame(rows, columns=["Date"] + names).set_index("Date") / 100.0


def get_strev():
    if not os.path.exists(STREV_CSV):
        raw = urllib.request.urlopen(STREV_URL, timeout=40).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        nm = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        open(STREV_CSV, "wb").write(z.read(nm))
        print(f"[strev] downloaded {STREV_CSV}")
    return parse_french(STREV_CSV, ["ST_Rev"])


def nw_ols(y, X, L=10):
    """OLS with Newey-West HAC SEs. X includes the intercept column. Returns
    (beta, tstat, r2)."""
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    e = y - X @ beta
    n, k = X.shape
    S = (X * e[:, None])
    G = S.T @ S
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        Gl = S[l:].T @ S[:-l]
        G += w * (Gl + Gl.T)
    cov = XtXi @ G @ XtXi
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (e @ e) / (((y - y.mean()) ** 2).sum())
    return beta, beta / se, r2


def run(tag, series, factors, names):
    df = pd.DataFrame({"r": series}).join(factors, how="inner").dropna()
    # excess strategy return over RF is unnecessary for a dollar-neutral book; use raw
    y = df["r"].values
    X = np.column_stack([np.ones(len(df))] + [df[n].values for n in names])
    beta, t, r2 = nw_ols(y, X, L=10)
    print(f"\n  [{tag}]  n={len(df)}  R2={r2:.3f}")
    print(f"    alpha = {beta[0]*1e4:+.2f} bps/day  (t={t[0]:+.2f})")
    for nm, b, tt in zip(names, beta[1:], t[1:]):
        print(f"      {nm:<8s} beta={b:+.3f}  t={tt:+.2f}")
    return beta[0] * 1e4, t[0]


def main():
    dis, cols, FL, PR, SD, R, cpx, close = m7.load_panels()
    Z = m7.zscore(FL)
    oos_days = [d for d in dis if d >= dis[m7.BURN]]
    full = m7.strat_returns(Z, R, m7.full_sample_universe(cpx, dis, cols, m7.BOTTOM_K))
    full = full[full != 0].dropna()
    oos = m7.strat_returns(Z, R, m7.walkforward_universe(cpx, dis, cols, m7.BOTTOM_K)).loc[oos_days]
    oos = oos[oos != 0].dropna()

    ff = parse_french(FF5, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]).join(
        parse_french(MOM, ["Mom"]), how="inner")
    ff_sr = ff.join(get_strev(), how="inner")
    base_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    sr_names = base_names + ["ST_Rev"]

    print("=" * 72 + "\n4c  ST_Rev FACTOR CONTROL ON THE TICK-CONSTRAINED REVERSAL\n" + "=" * 72)
    print("\n--- FF5+MOM (baseline; should reproduce Table 15) ---")
    run("full  FF5+MOM", full, ff, base_names)
    run("OOS   FF5+MOM", oos, ff, base_names)
    print("\n--- FF5+MOM+ST_Rev (does alpha survive short-term reversal?) ---")
    run("full  +ST_Rev", full, ff_sr, sr_names)
    run("OOS   +ST_Rev", oos, ff_sr, sr_names)


if __name__ == "__main__":
    main()
