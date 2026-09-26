#!/usr/bin/env python3
"""daily-labels-v1 (DAILY_LABELS_DESIGN.md): burst flow vs daily retail and institutional (>= $50k) imbalances."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import p4_external_tests as X
import fp_multiday_h1 as H
import fp_multiday_h2 as H2


def sparse_fe_regression(df, ycol, xcols, fe="date", cluster="permno"):
    """Like p4_external_tests.fe_regression but winsorizes each regressor over its NON-ZERO values, so that
    sparse flow variables (zero on most name-days) are not collapsed to a constant."""
    import math
    d = df[[ycol, fe, cluster] + xcols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 50:
        return None
    lo, hi = d[ycol].quantile([0.01, 0.99]); d[ycol] = d[ycol].clip(lo, hi)
    keep = []
    for c in xcols:
        nz = d[c][d[c] != 0]
        if len(nz) < 100 or nz.nunique() < 10:
            continue
        lo, hi = nz.quantile([0.01, 0.99])
        d[c] = d[c].clip(min(lo, 0.0), max(hi, 0.0))
        if d[c].std() > 0:
            keep.append(c)
    dm = d.groupby(fe)[[ycol] + keep].transform(lambda s: s - s.mean())
    b, V = X.cluster_ols(dm[ycol].to_numpy(float), dm[keep].to_numpy(float), d[cluster].to_numpy())
    out = {c: dict(b=float(b[i]), t=float(b[i] / math.sqrt(V[i, i])) if V[i, i] > 0 else None) for i, c in enumerate(keep)}
    if "q_info" in keep and "q_other" in keep:
        i, j = keep.index("q_info"), keep.index("q_other")
        var = V[i, i] + V[j, j] - 2 * V[i, j]
        out["info_minus_other"] = dict(b=float(b[i] - b[j]), t=float((b[i] - b[j]) / math.sqrt(var)) if var > 0 else None)
    out["n"] = int(len(d)); out["names"] = int(d[cluster].nunique()); out["dropped"] = [c for c in xcols if c not in keep]
    return out

ROOT = Path(__file__).resolve().parents[1]


def labels(years):
    fr = []
    for y in years:
        p = X.DATA / ("iid_%d.csv.gz" % y)
        if p.exists():
            fr.append(pd.read_csv(p))
    iid = pd.concat(fr, ignore_index=True)
    iid["date"] = pd.to_datetime(iid.date).dt.strftime("%Y%m%d")
    rt = iid.buyvol_retail + iid.sellvol_retail
    it = iid.buyvol_inst50k + iid.sellvol_inst50k
    iid["RI"] = np.where(rt > 0, (iid.buyvol_retail - iid.sellvol_retail) / rt, np.nan)
    iid["II"] = np.where(it > 0, (iid.buyvol_inst50k - iid.sellvol_inst50k) / it, np.nan)
    return iid[["date", "sym_root", "RI", "II"]]


def main():
    cell = sys.argv[1]
    cal = H.calendar()
    nd = pd.read_csv(ROOT / "results" / "p4_revisit_v1" / "agg" / cell / ("nameday_%s.csv.gz" % cell), dtype={"date": str},
                     usecols=["permno", "date", "family", "S_info_1550_k50", "S_large_1550", "adv20", "turn20",
                              "cap_lag1", "dlyret", "ret_lag1"])
    nd = nd[nd.family == "T"].drop(columns="family")
    nd["x_info"] = nd.S_info_1550_k50.fillna(0) / nd.adv20
    nd["x_other"] = (nd.S_large_1550.fillna(0) - nd.S_info_1550_k50.fillna(0)) / nd.adv20
    lk = H2.daily_links(cell, cal, 3)
    g = nd.merge(lk[["permno", "date", "L", "M", "U"]], on=["permno", "date"], how="inner")
    for c, k in (("x_L", "L"), ("x_M", "M"), ("x_U", "U")):
        g[c] = g[k] / g.adv20
    years = sorted(g.date.str[:4].astype(int).unique())
    lab = labels(years)
    crsp = X.crsp_daily(years)[["permno", "date", "ticker"]]
    lab = lab.merge(crsp, left_on=["date", "sym_root"], right_on=["date", "ticker"])
    g = g.merge(lab[["permno", "date", "RI", "II"]], on=["permno", "date"], how="inner")
    g["log_cap"] = np.log(g.cap_lag1)
    xs = ["x_L", "x_M", "x_U", "x_info", "x_other", "dlyret", "ret_lag1", "log_cap", "turn20"]
    res = dict(cell=cell, n=int(len(g)), names=int(g.permno.nunique()),
               nonzero_share={c: float((g[c] != 0).mean()) for c in ("x_L", "x_M", "x_U", "x_info")})
    print("nonzero share:", {k: round(v, 4) for k, v in res["nonzero_share"].items()})
    for y in ("II", "RI"):
        d = g.rename(columns={"x_L": "q_info", "x_M": "q_other"})
        r = sparse_fe_regression(d, y, ["q_info", "q_other"] + xs[2:], fe="date", cluster="permno")
        res[y] = r
        if r:
            print("%s %-3s n %d  L %+.4f (t %5.2f)  M %+.4f (t %5.2f)  L-M %+.4f (t %5.2f) | info %+.4f (t %5.2f)  other %+.4f (t %5.2f)  ret %+.3f"
                  % (cell, y, r["n"], r["q_info"]["b"], r["q_info"]["t"], r["q_other"]["b"], r["q_other"]["t"],
                     r["info_minus_other"]["b"], r["info_minus_other"]["t"], r["x_info"]["b"], r["x_info"]["t"],
                     r["x_other"]["b"], r["x_other"]["t"], r["dlyret"]["b"]))
    (ROOT / "results" / "daily_labels_v1").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "daily_labels_v1" / ("%s.json" % cell)).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
