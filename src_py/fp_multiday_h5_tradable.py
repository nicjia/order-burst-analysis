#!/usr/bin/env python3
"""fingerprint-multiday-v1 H5, tradable timing: an episode's end is only known once a day passes with no
continuation, so entry is the close of d1+1. Returns are measured from d1+2 onward (abnormal, signed by side).
Usage: fp_multiday_h5_tradable.py CELL"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import fp_multiday_h1 as H
import earnings_flow as E

ROOT = Path(__file__).resolve().parents[1]
cell = sys.argv[1]
cal = H.calendar(); inv = {v: k for k, v in cal.items()}
d = pd.read_csv(ROOT / "results" / "fp_multiday_v1" / ("episodes_%s.csv.gz" % cell))
years = sorted({int(inv[x][:4]) for x in d.d1})
crsp, _ = E.returns([min(years) - 1] + years + [max(years) + 1])
crsp["day"] = crsp.date.map(cal)
ar = crsp.set_index(["permno", "day"]).ar.sort_index()


def cum(permno, a, b):
    try:
        v = ar.loc[(permno, slice(a, b))].to_numpy(float)
    except KeyError:
        return np.nan
    return np.nan if not len(v) else (np.prod(1 + v[np.isfinite(v)]) - 1) * 1e4


rows = []
for e in d.itertuples(index=False):
    rows.append(dict(permno=e.permno, L=e.L, phi=e.phi, dec=e.dec,
                     r5=e.side * cum(e.permno, e.d1 + 2, e.d1 + 6), r20=e.side * cum(e.permno, e.d1 + 2, e.d1 + 21)))
t = pd.DataFrame(rows).dropna(subset=["r5"])
out = {}
for key, g in t.groupby(t.L.clip(upper=4)):
    for h in ("r5", "r20"):
        v = g[h].dropna()
        out["L%d_%s" % (key, h)] = dict(n=int(len(v)), mean=float(v.mean()),
                                        t=float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 5 else None)
top = t[t.dec >= 9]
for h in ("r5", "r20"):
    v = top[h].dropna()
    out["topdecile_%s" % h] = dict(n=int(len(v)), mean=float(v.mean()), t=float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))))
(ROOT / "results" / "fp_multiday_v1" / ("H5_tradable_%s.json" % cell)).write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print("%-16s n %6d  mean %+8.2f bps  t %6.2f" % (k, v["n"], v["mean"], v["t"] if v["t"] else float("nan")))
