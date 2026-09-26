#!/usr/bin/env python3
"""fingerprint-multiday-v1 H4/H5 (design 5c419eebb3ae): impact shape and decay of tape-detected program episodes.

Episode: maximal run of calendar-consecutive trading days on which one (name, side, modal size) key carries >= 3
one-sided bursts (amendment A1 label). Impact: signed cumulative CRSP return from the close before the episode to
the close of its last day. Decay: signed abnormal return 1, 5 and 20 days after. Usage: fp_multiday_episodes.py CELL
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import fp_multiday_h1 as H
import fp_multiday_purity as P
import earnings_flow as E

ROOT = Path(__file__).resolve().parents[1]
MIN_BURSTS = 3


def episodes(cell, cal):
    w = P.keyed(cell, cal)
    out = []
    for side, o, sgn in (("B", "S", 1), ("S", "B", -1)):
        q = w[(w["nb_" + side] >= MIN_BURSTS) & (w["nb_" + o] == 0)][["permno", "size", "day", "vol_" + side]]
        q = q.rename(columns={"vol_" + side: "vol"}).sort_values(["permno", "size", "day"])
        if not len(q):
            continue
        newrun = (q.permno != q.permno.shift()) | (q["size"] != q["size"].shift()) | (q.day != q.day.shift() + 1)
        q["ep"] = newrun.cumsum()
        g = q.groupby("ep").agg(permno=("permno", "first"), size=("size", "first"), d0=("day", "min"),
                                d1=("day", "max"), L=("day", "size"), Q=("vol", "sum")).reset_index(drop=True)
        g["side"] = sgn
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    cell = sys.argv[1]
    cal = H.calendar(); inv = {v: k for k, v in cal.items()}
    ep = episodes(cell, cal)
    nd = pd.read_csv(ROOT / "results" / "p4_revisit_v1" / "agg" / cell / ("nameday_%s.csv.gz" % cell),
                     dtype={"date": str}, usecols=["permno", "date", "family", "adv20", "sigma20"])
    nd = nd[nd.family == "T"].drop(columns="family")
    nd["day"] = nd.date.map(cal)
    ep = ep.merge(nd.rename(columns={"day": "d0"})[["permno", "d0", "adv20", "sigma20"]], on=["permno", "d0"], how="left")
    years = sorted({int(inv[d][:4]) for d in ep.d0})
    crsp, _ = E.returns([min(years) - 1] + years + [max(years) + 1])
    crsp["day"] = crsp.date.map(cal)
    r = crsp.set_index(["permno", "day"])[["dlyret", "ar"]].sort_index()

    def cum(col, permno, a, b):
        try:
            v = r.loc[(permno, slice(a, b)), col].to_numpy(float)
        except KeyError:
            return np.nan
        return np.nan if not len(v) else (np.prod(1 + v[np.isfinite(v)]) - 1)

    rows = []
    for e in ep.itertuples(index=False):
        if not np.isfinite(e.adv20) or e.adv20 <= 0:
            continue
        imp = cum("dlyret", e.permno, e.d0, e.d1)
        rows.append(dict(permno=e.permno, size=e.size, side=e.side, L=e.L, Q=e.Q, d0=e.d0, d1=e.d1,
                         phi=e.Q / (e.L * e.adv20), sigma20=e.sigma20,
                         impact=e.side * imp * 1e4 if np.isfinite(imp) else np.nan,
                         post1=e.side * cum("ar", e.permno, e.d1 + 1, e.d1 + 1) * 1e4,
                         post5=e.side * cum("ar", e.permno, e.d1 + 1, e.d1 + 5) * 1e4,
                         post20=e.side * cum("ar", e.permno, e.d1 + 1, e.d1 + 20) * 1e4))
    d = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["impact", "phi"])
    d = d[(d.phi > 0) & (d.phi < 1)]
    d["dec"] = pd.qcut(d.phi.rank(method="first"), 10, labels=False) + 1
    tab = d.groupby("dec").agg(n=("impact", "size"), phi=("phi", "mean"), impact=("impact", "mean"),
                               imp_sd=("impact", "std"), L=("L", "mean"), post5=("post5", "mean"),
                               post20=("post20", "mean")).reset_index()
    tab["t"] = tab.impact / (tab.imp_sd / np.sqrt(tab.n))
    pos = tab[tab.impact > 0]
    delta = np.polyfit(np.log(pos.phi), np.log(pos.impact), 1)[0] if len(pos) >= 4 else np.nan
    rng = np.random.default_rng(20260923); names = d.permno.unique(); boot = []
    for _ in range(500):
        s = d[d.permno.isin(rng.choice(names, len(names), replace=True))]
        if len(s) < 1000:
            continue
        s = s.assign(dec=pd.qcut(s.phi.rank(method="first"), 10, labels=False) + 1)
        tb = s.groupby("dec").agg(phi=("phi", "mean"), impact=("impact", "mean"))
        tb = tb[tb.impact > 0]
        if len(tb) >= 4:
            boot.append(np.polyfit(np.log(tb.phi), np.log(tb.impact), 1)[0])
    res = dict(cell=cell, episodes=int(len(d)), names=int(d.permno.nunique()),
               L_mean=float(d.L.mean()), L_share_ge2=float((d.L >= 2).mean()), L_max=int(d.L.max()),
               delta=float(delta), delta_ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))] if boot else None,
               by_decile=tab.round(4).to_dict("records"),
               by_length={str(k): dict(n=int(v.impact.size), impact=float(v.impact.mean()), post5=float(v.post5.mean()),
                                       post20=float(v.post20.mean())) for k, v in d.groupby(d.L.clip(upper=5))})
    (ROOT / "results" / "fp_multiday_v1" / ("H4_%s.json" % cell)).write_text(json.dumps(res, indent=1, default=float))
    d.to_csv(ROOT / "results" / "fp_multiday_v1" / ("episodes_%s.csv.gz" % cell), index=False)
    print("%s episodes %d names %d | mean L %.2f (max %d) | delta %.3f CI %s"
          % (cell, res["episodes"], res["names"], res["L_mean"], res["L_max"], res["delta"], res["delta_ci95"]))
    print(tab[["dec", "n", "phi", "L", "impact", "t", "post5", "post20"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
