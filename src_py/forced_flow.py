#!/usr/bin/env python3
"""forced-flow-v1: does burst impact hold less on days when flow is mechanical?

Day types from the calendar alone: quarterly expiry (third Friday of Mar/Jun/Sep/Dec, when index futures and
options settle and S&P rebalances take effect), other monthly expiry, quarter-end and other month-end.
Outcome: the name-day mean post-decision displacement to the close of large bursts (P4 d_close, bps), and of
the informative class. Name fixed effects, standard errors clustered by date. Exploration DEV, confirmation TEST.
Usage: forced_flow.py CELL
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import p4_external_tests as X

ROOT = Path(__file__).resolve().parents[1]


def day_types(dates):
    d = pd.DataFrame({"date": sorted(set(dates))})
    dt = pd.to_datetime(d.date)
    d["y"], d["m"], d["dow"], d["dom"] = dt.dt.year, dt.dt.month, dt.dt.dayofweek, dt.dt.day
    fri = d[d.dow == 4].copy()
    fri["nth"] = fri.groupby(["y", "m"]).cumcount() + 1
    third = set(fri[fri.nth == 3].date)
    d["expiry_q"] = d.date.isin(third) & d.m.isin([3, 6, 9, 12])
    d["expiry_m"] = d.date.isin(third) & ~d.m.isin([3, 6, 9, 12])
    last_m = d.groupby(["y", "m"]).date.max()
    d["month_end"] = d.date.isin(set(last_m))
    d["quarter_end"] = d.month_end & d.m.isin([3, 6, 9, 12])
    d["month_end"] = d.month_end & ~d.quarter_end
    return d[["date", "expiry_q", "expiry_m", "quarter_end", "month_end"]]


def main():
    cell = sys.argv[1]
    nd = pd.read_csv(ROOT / "results" / "p4_revisit_v1" / "agg" / cell / ("nameday_%s.csv.gz" % cell), dtype={"date": str},
                     usecols=["permno", "date", "family", "sum_dclose_large", "n_dclose_large",
                              "sum_dclose_info_k50", "n_dclose_info_k50", "sum_dclose_non_k50", "n_dclose_non_k50",
                              "S_large_1550", "S_info_1550_k50", "adv20", "sigma20"])
    nd = nd[nd.family == "T"].drop(columns="family")
    nd["d_large"] = nd.sum_dclose_large / nd.n_dclose_large
    nd["d_info"] = nd.sum_dclose_info_k50 / nd.n_dclose_info_k50
    nd["d_non"] = nd.sum_dclose_non_k50 / nd.n_dclose_non_k50
    nd["abs_flow"] = (nd.S_large_1550 / nd.adv20).abs()
    nd = nd.merge(day_types(nd.date), on="date")
    types = ["expiry_q", "expiry_m", "quarter_end", "month_end"]
    for t in types:
        nd[t] = nd[t].astype(float)
    res = dict(cell=cell, name_days=int(len(nd)),
               counts={t: int(nd[t].sum()) for t in types},
               mean_by_type={t: {k: float(nd.loc[nd[t] == 1, k].mean()) for k in ("d_large", "d_info", "d_non", "abs_flow")} for t in types},
               mean_normal={k: float(nd.loc[nd[types].sum(1) == 0, k].mean()) for k in ("d_large", "d_info", "d_non", "abs_flow")})
    for y in ("d_large", "d_info", "d_non", "abs_flow"):
        r = X.fe_regression(nd.assign(q_info=nd.expiry_q, q_other=nd.quarter_end), y,
                            ["q_info", "q_other", "expiry_m", "month_end"], fe="permno", cluster="date")
        res[y] = {k: r[k] for k in ("q_info", "q_other", "expiry_m", "month_end", "n")} if r else None
    (ROOT / "results" / "forced_flow_v1").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "forced_flow_v1" / ("%s.json" % cell)).write_text(json.dumps(res, indent=1, default=float))
    print(cell, "name-days", res["name_days"], "| quarterly expiry", res["counts"]["expiry_q"], "quarter-end", res["counts"]["quarter_end"])
    print("  normal day means: d_large %+.2f  d_info %+.2f  d_non %+.2f  |flow| %.4f"
          % tuple(res["mean_normal"][k] for k in ("d_large", "d_info", "d_non", "abs_flow")))
    for y in ("d_large", "d_info", "d_non", "abs_flow"):
        r = res[y]
        print("  %-8s expiry_q %+.3f (t %5.2f)  quarter_end %+.3f (t %5.2f)  expiry_m %+.3f (t %5.2f)"
              % (y, r["q_info"]["b"], r["q_info"]["t"], r["q_other"]["b"], r["q_other"]["t"], r["expiry_m"]["b"], r["expiry_m"]["t"]))


if __name__ == "__main__":
    main()
