#!/usr/bin/env python3
"""
footprint_determinants.py — cross-sectional determinants of the permanent hidden-
execution footprint. Referee: "no cross-sectional determinants of the footprint
(institutional ownership, analyst coverage, PIN)." This tests what predicts a name's
3-min permanent markout, which turns the measurement from a single pooled number into
an economic result: the footprint should be larger where adverse selection is worse
(smaller, less-covered, less-liquid, more-concealed names).

Per-name footprint = mean mk3 (outlier-censored, 2023-24).
Characteristics: price, dollar volume, realized vol, hidden-fraction (n_mid share),
burst intensity, and -- via yfinance -- institutional ownership % and analyst count.
"""
import math, os, time
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "measurements", "data")
OUT = os.path.join(REPO, "measurements", "out")
CAP = 1000.0


def main():
    d = pd.read_csv(os.path.join(REPO, "results/research/hidden_xsec_daily.csv"))
    d["date"] = d["date"].astype(int)
    d = d[(d.mk3.abs() <= CAP) & (d.mk15.abs() <= CAP) & (d.mk30.abs() <= CAP)].copy()
    C = pd.read_parquet(os.path.join(DATA, "closes24.parquet"))

    # per-name aggregates
    g = d.groupby("ticker")
    feat = pd.DataFrame({
        "footprint": g.mk3.mean(),
        "footprint15": g.mk15.mean(),
        "intensity": g.n.mean(),
        "hidden_frac": g.apply(lambda x: x.n_mid.sum() / (x.n_mid.sum() + x.n_sig.sum() + 1e-9)),
        "dvol": g.apply(lambda x: (x.buy + x.sell).mean()),
        "absCOI": g.COI.apply(lambda s: s.abs().mean()),
        "ndays": g.size(),
    })
    px = {c: C[c].dropna().mean() for c in C.columns if C[c].notna().sum() > 100}
    rv = {c: C[c].pct_change().std() * math.sqrt(252) for c in C.columns if C[c].notna().sum() > 100}
    feat["price"] = pd.Series(px); feat["realvol"] = pd.Series(rv)
    feat["dollarvol"] = feat["dvol"] * feat["price"]
    feat = feat.dropna(subset=["price", "footprint"])
    feat = feat[feat.ndays >= 50]

    # yfinance fundamentals (institutional ownership, analyst coverage) -- best effort
    inst, ana, mcap = {}, {}, {}
    try:
        import yfinance as yf
        names = list(feat.index)
        for i in range(0, len(names), 50):
            for t in names[i:i + 50]:
                try:
                    info = yf.Ticker(t).get_info()
                    inst[t] = info.get("heldPercentInstitutions", np.nan)
                    ana[t] = info.get("numberOfAnalystOpinions", np.nan)
                    mcap[t] = info.get("marketCap", np.nan)
                except Exception:
                    pass
            print("  yfinance fundamentals %d/%d" % (min(i + 50, len(names)), len(names)))
            time.sleep(0.5)
    except Exception as e:
        print("  yfinance unavailable:", e)
    feat["inst_own"] = pd.Series(inst); feat["n_analyst"] = pd.Series(ana); feat["mktcap"] = pd.Series(mcap)
    feat.to_csv(os.path.join(OUT, "footprint_determinants.csv"))
    print("\nsaved %d names to measurements/out/footprint_determinants.csv\n" % len(feat))

    # univariate rank correlations with the footprint
    print("=== univariate Spearman corr with the 3-min permanent footprint ===")
    for c in ["price", "dollarvol", "realvol", "hidden_frac", "intensity", "absCOI",
              "inst_own", "n_analyst", "mktcap"]:
        if c in feat and feat[c].notna().sum() > 30:
            rho = feat["footprint"].corr(feat[c], method="spearman")
            n = feat[[c, "footprint"]].dropna().shape[0]
            print("  %-12s rho=%+.3f  (n=%d)" % (c, rho, n))

    # multivariate OLS on standardized regressors (log where sensible)
    print("\n=== OLS: footprint ~ standardized characteristics (log price/vol/mktcap) ===")
    X = pd.DataFrame(index=feat.index)
    X["log_price"] = np.log(feat.price)
    X["log_dvol"] = np.log(feat.dollarvol + 1)
    X["realvol"] = feat.realvol
    X["hidden_frac"] = feat.hidden_frac
    if feat.inst_own.notna().sum() > 100: X["inst_own"] = feat.inst_own
    if feat.n_analyst.notna().sum() > 100: X["log_analyst"] = np.log(feat.n_analyst.clip(lower=1))
    X = X.dropna()
    y = feat.loc[X.index, "footprint"]
    Xz = (X - X.mean()) / (X.std() + 1e-9)
    M = np.column_stack([np.ones(len(Xz)), Xz.values])
    beta, *_ = np.linalg.lstsq(M, y.values, rcond=None)
    resid = y.values - M @ beta
    n, k = M.shape
    se = np.sqrt(np.diag(np.linalg.inv(M.T @ M) * (resid @ resid) / (n - k)))
    r2 = 1 - (resid @ resid) / ((y.values - y.mean()) ** 2).sum()
    print("  n=%d names, R^2=%.3f" % (n, r2))
    print("  %-14s %8s %8s" % ("regressor", "coef", "t"))
    for nm, b, s in zip(["const"] + list(Xz.columns), beta, se):
        print("  %-14s %+8.2f %+8.2f" % (nm, b, b / s))
    print("\n  (coef = bps of footprint per 1sd; a negative log_price/inst_own coef means")
    print("   the permanent footprint is LARGER in cheaper, less-institutional names --")
    print("   the adverse-selection prediction.)")


if __name__ == "__main__":
    main()
