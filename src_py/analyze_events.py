#!/usr/bin/env python3
"""program-evidence-v1 module D4: program flow around S&P 500 and Nasdaq-100 changes (one shot).

E is the first trading day of membership for additions and the first trading day after membership
ends for deletions. For each event, PI (and NPI, PIR, NPIR) are z-scored against the name's own days
E-40..E-11; the statistic is the mean z over E-5..E-1. Gate: additions minus deletions > 0 for PI,
Welch t > 2. Reported beside it: NPI, the within-type ratios, and the PI - NPI contrast.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
WR = ROOT / "data" / "wrds"
BASE, PRE = (-40, -11), (-5, -1)
VARS = ("PI", "NPI", "PIR", "NPIR", "OI", "program_share")


def load_flows(paths):
    f = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    f["date"] = f.date.astype(str)
    f = f.drop_duplicates(["permno", "date"])
    tot = f.buy_vol + f.sell_vol
    f = f[tot > 0].copy(); tot = f.buy_vol + f.sell_vol
    pb, ps = f.program_buy_vol, f.program_sell_vol
    ob, os_ = f.buy_vol - pb, f.sell_vol - ps
    f["PI"] = (pb - ps) / tot; f["NPI"] = (ob - os_) / tot; f["OI"] = (f.buy_vol - f.sell_vol) / tot
    f["PIR"] = np.where(pb + ps > 0, (pb - ps) / (pb + ps), np.nan)
    f["NPIR"] = np.where(ob + os_ > 0, (ob - os_) / (ob + os_), np.nan)
    f["program_share"] = (pb + ps) / tot
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ev = pd.read_csv(WR / "index_events_2021_2024.csv")
    cal = sorted(pd.read_csv(WR / "crsp_dsi.csv.gz").date.astype(str).str.replace("-", ""))
    f = load_flows(args.flows)
    by = {k: g.set_index("date") for k, g in f.groupby("permno")}
    rows, profile = [], []
    for r in ev.itertuples():
        d = r.date.replace("-", "")
        if r.kind == "add":
            pos = next(i for i, x in enumerate(cal) if x >= d)
        else:
            pos = next(i for i, x in enumerate(cal) if x > d)
        g = by.get(r.permno)
        if g is None:
            rows.append(dict(index=r.index, kind=r.kind, permno=r.permno, date=r.date, usable=False, reason="no flow data"))
            continue
        rel = {cal[pos + k]: k for k in range(-45, 11) if 0 <= pos + k < len(cal)}
        w = g[g.index.isin(rel)].copy(); w["k"] = [rel[x] for x in w.index]
        base = w[(w.k >= BASE[0]) & (w.k <= BASE[1])]; pre = w[(w.k >= PRE[0]) & (w.k <= PRE[1])]
        row = dict(index=r.index, kind=r.kind, permno=r.permno, date=r.date, base_days=len(base), pre_days=len(pre))
        if len(base) < 20 or len(pre) < 3:
            row.update(usable=False, reason="coverage"); rows.append(row); continue
        row["usable"] = True
        for v in VARS:
            sd = base[v].std()
            row["z_" + v] = float((pre[v].mean() - base[v].mean()) / sd) if sd > 0 else np.nan
            for k, x in w.groupby("k")[v].mean().items():
                profile.append(dict(kind=r.kind, k=int(k), var=v, z=float((x - base[v].mean()) / sd) if sd > 0 else np.nan))
        rows.append(row)
    t = pd.DataFrame(rows)
    u = t[t.usable == True]  # noqa: E712
    res = dict(events=int(len(t)), usable=int(len(u)), by_kind=u.groupby(["index", "kind"]).size().to_dict())
    res["by_kind"] = {"%s_%s" % k: int(v) for k, v in res["by_kind"].items()}
    for v in VARS + ("PI_minus_NPI", "PIR_minus_NPIR"):
        if v == "PI_minus_NPI":
            a = (u.z_PI - u.z_NPI)[u.kind == "add"]; b = (u.z_PI - u.z_NPI)[u.kind == "delete"]
        elif v == "PIR_minus_NPIR":
            a = (u.z_PIR - u.z_NPIR)[u.kind == "add"]; b = (u.z_PIR - u.z_NPIR)[u.kind == "delete"]
        else:
            a = u["z_" + v][u.kind == "add"]; b = u["z_" + v][u.kind == "delete"]
        a, b = a.dropna(), b.dropna()
        tt = stats.ttest_ind(a, b, equal_var=False) if len(a) > 2 and len(b) > 2 else None
        res[v] = dict(mean_add=float(a.mean()), mean_delete=float(b.mean()), difference=float(a.mean() - b.mean()),
                      welch_t=float(tt.statistic) if tt else None, n_add=int(len(a)), n_delete=int(len(b)),
                      t_add_vs_zero=float(stats.ttest_1samp(a, 0).statistic) if len(a) > 2 else None,
                      t_delete_vs_zero=float(stats.ttest_1samp(b, 0).statistic) if len(b) > 2 else None)
    res["D4_gate"] = bool(res["PI"]["difference"] > 0 and (res["PI"]["welch_t"] or 0) > 2)
    prof = pd.DataFrame(profile)
    if len(prof):
        res["profile"] = prof.groupby(["kind", "var", "k"]).z.mean().round(4).reset_index().to_dict(orient="records")
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    t.to_csv(Path(args.out).with_suffix(".events.csv"), index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "profile"}, indent=1, default=float))


if __name__ == "__main__":
    main()
