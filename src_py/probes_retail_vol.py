#!/usr/bin/env python3
"""Idea 5: are odd-lot bursts retail? Idea 10: does multi-day program intensity forecast volatility?

Idea 5 — name-day signed burst volume in odd-lot modal sizes (< 100) and in round/larger sizes, both / adv20,
against the BJZZ retail imbalance RI, day fixed effects, same-day return controlled. If lit odd-lot bursts are
retail, the odd-lot coefficient should exceed the other one.
Idea 10 — next-day |return| on today's total burst count (the Jones-Kaul-Lipson baseline, VERIFIED 1.16) and on
today's program-linked burst volume, with lagged |return|, sigma20, day FE, PERMNO clusters.
Usage: probes_retail_vol.py CELL
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import p4_external_tests as X
import fp_multiday_h1 as H
import fp_multiday_h2 as H2
import daily_labels as D

ROOT = Path(__file__).resolve().parents[1]
cell = sys.argv[1]
cal = H.calendar()
fp = pd.read_csv(H.T / ("%s_fp.csv.gz" % cell), dtype={"date": str})
fp["sv"] = fp.side * fp.vol
odd = fp[fp["size"] < 100].groupby(["permno", "date"]).sv.sum().rename("sv_odd")
big = fp[fp["size"] >= 100].groupby(["permno", "date"]).sv.sum().rename("sv_big")
nd = pd.read_csv(ROOT / "results" / "p4_revisit_v1" / "agg" / cell / ("nameday_%s.csv.gz" % cell), dtype={"date": str},
                 usecols=["permno", "date", "family", "adv20", "turn20", "cap_lag1", "dlyret", "ret_lag1",
                          "sigma20", "n_bursts"])
nd = nd[nd.family == "T"].drop(columns="family").join(odd, on=["permno", "date"]).join(big, on=["permno", "date"])
nd[["sv_odd", "sv_big"]] = nd[["sv_odd", "sv_big"]].fillna(0.0)
nd["x_odd"] = nd.sv_odd / nd.adv20; nd["x_big"] = nd.sv_big / nd.adv20
nd["log_cap"] = np.log(nd.cap_lag1)
years = sorted(nd.date.str[:4].astype(int).unique())
lab = D.labels(years)
crsp = X.crsp_daily(years)[["permno", "date", "ticker"]]
lab = lab.merge(crsp, left_on=["date", "sym_root"], right_on=["date", "ticker"])
g = nd.merge(lab[["permno", "date", "RI", "II"]], on=["permno", "date"], how="inner")
res = dict(cell=cell)
print("--- idea 5: odd-lot bursts vs retail imbalance")
for y in ("RI", "II"):
    r = D.sparse_fe_regression(g.rename(columns={"x_odd": "q_info", "x_big": "q_other"}), y,
                               ["q_info", "q_other", "dlyret", "ret_lag1", "log_cap", "turn20"], fe="date", cluster="permno")
    res["idea5_" + y] = r
    print("  %-3s n %7d | odd-lot %+7.4f (t %5.2f)  >=100 %+7.4f (t %5.2f)  odd-minus-big %+7.4f (t %5.2f)"
          % (y, r["n"], r["q_info"]["b"], r["q_info"]["t"], r["q_other"]["b"], r["q_other"]["t"],
             r["info_minus_other"]["b"], r["info_minus_other"]["t"]))
print("--- idea 10: program intensity and next-day volatility")
lk = H2.daily_links(cell, cal, 3)
v = nd.merge(lk[["permno", "date", "gL", "gU"]], on=["permno", "date"], how="inner").sort_values(["permno", "date"])
v["absret_next"] = v.groupby("permno").dlyret.shift(-1).abs() * 1e4
v["absret"] = v.dlyret.abs() * 1e4
v["prog"] = v.gL / v.adv20; v["nonprog"] = v.gU / v.adv20
v["log_n_bursts"] = np.log1p(v.n_bursts)
r = D.sparse_fe_regression(v.rename(columns={"prog": "q_info", "nonprog": "q_other"}), "absret_next",
                           ["q_info", "q_other", "log_n_bursts", "absret", "sigma20"], fe="date", cluster="permno")
res["idea10"] = r
print("  n %7d | program vol %+8.2f (t %5.2f)  other fp vol %+8.2f (t %5.2f)  log burst count %+7.2f (t %5.2f)  |ret| %+.3f (t %5.2f)"
      % (r["n"], r["q_info"]["b"], r["q_info"]["t"], r["q_other"]["b"], r["q_other"]["t"],
         r["log_n_bursts"]["b"], r["log_n_bursts"]["t"], r["absret"]["b"], r["absret"]["t"]))
(ROOT / "results" / "burst_probes_v1").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "burst_probes_v1" / ("retail_vol_%s.json" % cell)).write_text(json.dumps(res, indent=1, default=float))
