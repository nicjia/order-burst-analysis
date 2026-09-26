#!/usr/bin/env python3
"""fingerprint-multiday-v1 H2 (design frozen 49a39405cc61): is backward-linked fingerprint flow institutional?

L_{i,t}: signed volume on day t of fingerprint bursts (primary sizes) whose (side, modal size) also appears on the
previous trading day t-1 for the same name. U_{i,t}: the rest of the fingerprint signed volume. Only days whose
previous trading day is present enter (both sums). Daily values / adv20, summed by calendar quarter (>= 20 days),
then the p4 Q2 regressions (13F dIO, mutual-fund dholdings) with the same-quarter return as a control.
Usage: fp_multiday_h2.py CELL
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import p4_external_tests as X
import fp_multiday_h1 as H

ROOT = Path(__file__).resolve().parents[1]


def daily_links(cell, cal, m=3):
    """Amendment A1: program-linked (L), mirror (M) and other (U) fingerprint flow per name-day."""
    import fp_multiday_purity as P
    w = P.keyed(cell, cal)
    parts = []
    for side, o, sgn in (("B", "S", 1.0), ("S", "B", -1.0)):
        today = (w["nb_" + side] >= m) & (w["nb_" + o] == 0)
        same = today & (w["nb_%s_p" % side] >= m) & (w["nb_%s_p" % o] == 0)
        mirror = today & (w["nb_%s_p" % o] >= m) & (w["nb_%s_p" % side] == 0)
        v = w["vol_" + side] * sgn
        parts.append(pd.DataFrame(dict(permno=w.permno, day=w.day, L=np.where(same, v, 0.0), M=np.where(mirror, v, 0.0),
                                       U=np.where(same | mirror, 0.0, v), gL=np.where(same, w["vol_" + side], 0.0),
                                       gU=np.where(same, 0.0, w["vol_" + side]))))
    d = pd.concat(parts).groupby(["permno", "day"]).sum().reset_index()
    nd = pd.read_csv(H.T / ("%s_nd.csv.gz" % cell), dtype={"date": str})
    nd["day"] = nd.date.map(cal)
    present = nd[["permno", "day"]].drop_duplicates()
    pp = present.copy(); pp["day"] = pp.day + 1; pp["has_prev"] = 1
    days = nd[["permno", "date", "day"]].merge(pp, on=["permno", "day"], how="inner").drop(columns="has_prev")
    return days.merge(d, on=["permno", "day"], how="left").fillna({"L": 0.0, "M": 0.0, "U": 0.0, "gL": 0.0, "gU": 0.0})


def main():
    cell = sys.argv[1]; m = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    cal = H.calendar()
    lk = daily_links(cell, cal, m)
    agg = ROOT / "results" / "p4_revisit_v1" / "agg" / cell / ("nameday_%s.csv.gz" % cell)
    nd = pd.read_csv(agg, dtype={"date": str}, usecols=["permno", "date", "family", "adv20", "turn20"])
    nd = nd[nd.family == "T"].drop(columns="family")
    g = lk.merge(nd, on=["permno", "date"], how="inner")
    g["x_info"] = g.L / g.adv20; g["x_other"] = g.U / g.adv20; g["x_mirror"] = g.M / g.adv20
    g["dt"] = pd.to_datetime(g.date); g["quarter"] = g.dt.dt.to_period("Q")
    g = g.replace([np.inf, -np.inf], np.nan)
    years = sorted(g.date.str[:4].astype(int).unique())
    c = X.crsp_daily(sorted(set(years) | {min(years) - 1}))
    state = X.quarter_end_state(c)
    q = X.quarterly_panel(g, state)
    qm = g.groupby(["permno", "quarter"]).x_mirror.sum().rename("q_mirror")
    q = q.join(qm, on=["permno", "quarter"])
    q = q.join(state.set_index(["permno", "quarter"])[["ret"]].rename(columns={"ret": "ret_q"}), on=["permno", "quarter"])
    cusip_hist = pd.read_csv(X.DATA / "cusip_hist.csv.gz", dtype={"cusip": str})
    orig = X.fe_regression
    res = dict(cell=cell, link_share_of_fp_volume=float(g.gL.sum() / (g.gL.sum() + g.gU.sum())),
               corr_L_retq=float(q[["q_info", "ret_q"]].dropna().corr(method="spearman").iloc[0, 1]),
               corr_U_retq=float(q[["q_other", "ret_q"]].dropna().corr(method="spearman").iloc[0, 1]))
    for label, extra in (("ret_ctrl", ["ret_q", "q_mirror"]), ("base", ["q_mirror"])):
        X.fe_regression = (lambda df, y, xs, _e=extra, **kw: orig(df, y, xs + _e, **kw))
        res[label] = dict(b_13F=X.test_13f(q, cusip_hist), c_MF=X.test_mf(q, years))
    X.fe_regression = orig
    out = ROOT / "results" / "fp_multiday_v1" / ("H2_%s_m%d.json" % (cell, m))
    out.write_text(json.dumps(res, indent=1, default=float))
    print(cell, "linked share of fingerprint volume %.3f | spearman(qL, ret_q) %+.3f  (qU, ret_q) %+.3f"
          % (res["link_share_of_fp_volume"], res["corr_L_retq"], res["corr_U_retq"]))
    for label in ("base", "ret_ctrl"):
        for t in ("b_13F", "c_MF"):
            r = res[label][t]
            if r:
                print("  %-8s %-6s L %+.4f (t %5.2f)  M %+.4f (t %5.2f)  U %+.4f (t %5.2f)  L-U %+.4f (t %5.2f)  n %d"
                      % (label, t, r["q_info"]["b"], r["q_info"]["t"], r["q_mirror"]["b"], r["q_mirror"]["t"], r["q_other"]["b"], r["q_other"]["t"],
                         r["info_minus_other"]["b"], r["info_minus_other"]["t"], r["n"]))


if __name__ == "__main__":
    main()
