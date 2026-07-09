#!/usr/bin/env python3
"""Aggregate the term-structure job (hidden_term): burst vs TOD-placebo markout at
each horizon, with date-clustered inference and placebo-netted survival."""
import glob
import numpy as np, pandas as pd

d = pd.concat([pd.read_csv(f) for f in glob.glob("results/hidden_term/out/*.csv")], ignore_index=True)
for c in d.columns:
    if c not in ("ticker", "date"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[d["n"].fillna(0) > 0].copy()
print("name-days with bursts:", len(d), " names:", d.ticker.nunique(), " dates:", d.date.nunique())

H = [("mk3", "pmk3", "3 min"), ("mk15", "pmk15", "15 min"), ("mk30", "pmk30", "30 min"),
     ("mk60", "pmk60", "1 hour"), ("mk120", "pmk120", "2 hour"), ("mkclose", "pmkclose", "to close")]
print("%-9s %8s %8s %8s %11s %7s" % ("horizon", "burst", "placebo", "net", "net t(day)", "%surv"))
for mk, pm, lab in H:
    dd = d.dropna(subset=[mk, pm])
    b = dd.groupby("date")[mk].mean().mean()
    p = dd.groupby("date")[pm].mean().mean()
    netday = dd.assign(net=dd[mk] - dd[pm]).groupby("date")["net"].mean().dropna()
    nmean = netday.mean(); nt = nmean / (netday.std() / np.sqrt(len(netday)))
    surv = 100 * nmean / b if b != 0 else float("nan")
    print("%-9s %+8.2f %+8.2f %+8.2f %+11.2f %7.0f" % (lab, b, p, nmean, nt, surv))
