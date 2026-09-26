#!/usr/bin/env python3
"""earnings-flow-v1 (EARNINGS_FLOW_DESIGN.md, frozen bb92aa0d85c6): pre-announcement burst flow vs CAR[0,+1].

Usage: earnings_flow.py --sample exploration|confirmation|replication [--family T|S] [--window 5]
"""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import p4_external_tests as X

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "results" / "p4_revisit_v1" / "agg"
OUT = ROOT / "results" / "earnings_flow_v1"
SAMPLES = {"exploration": ([("ERA2", 2012, 2016), ("DEV", 2017, 2019)], {0}),
           "confirmation": ([("VAL", 2020, 2021), ("TEST", 2022, 2025)], {1, 2}),
           "replication": ([("ERA2", 2012, 2016)], {1, 2})}


def group(p):
    return int(hashlib.sha256(("p4-revisit-v1|%d" % int(p)).encode()).hexdigest()[:8], 16) % 3


def returns(years):
    fr = []
    for y in sorted(set(years)):
        p = ROOT / "data" / "p4" / ("crsp_%d.csv.gz" % y)
        if p.exists():
            fr.append(pd.read_csv(p, usecols=["permno", "date", "dlyret", "dlycap"], dtype={"date": str}))
    c = pd.concat(fr, ignore_index=True).drop_duplicates(["permno", "date"])
    for k in ("dlyret", "dlycap"):
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c = c.sort_values(["permno", "date"])
    c["cap_lag"] = c.groupby("permno").dlycap.shift(1)
    w = c.dropna(subset=["dlyret", "cap_lag"])
    mkt = (w.dlyret * w.cap_lag).groupby(w.date).sum() / w.cap_lag.groupby(w.date).sum()
    c["ar"] = c.dlyret - c.date.map(mkt)
    return c, sorted(mkt.index)


def build(sample, family, window):
    cells, groups = SAMPLES[sample]
    nds = []
    for cell, y0, y1 in cells:
        d = pd.read_csv(AGG / cell / ("nameday_%s.csv.gz" % cell), dtype={"date": str},
                        usecols=["permno", "date", "family", "S_info_1550_k50", "S_large_1550", "S_all_1550",
                                 "adv20", "turn20", "cap_lag1"])
        d = d[(d.family == family) & d.date.str[:4].astype(int).between(y0, y1)]
        d = d[d.permno.map(group).isin(groups)]
        nds.append(d)
    nd = pd.concat(nds, ignore_index=True)
    for k in ("S_info_1550_k50", "S_large_1550", "S_all_1550"):
        nd[k] = nd[k].fillna(0.0)
    nd["x_info"] = nd.S_info_1550_k50 / nd.adv20
    nd["x_other"] = (nd.S_large_1550 - nd.S_info_1550_k50) / nd.adv20
    nd["x_all"] = nd.S_all_1550 / nd.adv20
    years = sorted(nd.date.str[:4].astype(int).unique())
    crsp, cal = returns([min(years) - 1] + years + [max(years) + 1])
    pos = {d: i for i, d in enumerate(cal)}
    ev = pd.read_csv(ROOT / "data" / "wrds" / "comp_rdq_2012_2025.csv.gz", dtype={"rdq": str})
    ev["rdq"] = ev.rdq.str.replace("-", "")
    ev = ev[ev.permno.isin(set(nd.permno))].drop_duplicates(["permno", "rdq"])
    ev["i0"] = np.searchsorted(cal, ev.rdq.to_numpy())
    ev = ev[(ev.i0 >= window + 1) & (ev.i0 < len(cal) - 22)]
    ev["d0"] = [cal[i] for i in ev.i0]
    ev = ev[ev.d0.str[:4].astype(int).isin(years)].drop_duplicates(["permno", "d0"])
    ndi = nd.set_index(["permno", "date"])
    ar = crsp.set_index(["permno", "date"]).ar
    capm = crsp.set_index(["permno", "date"]).dlycap
    rows = []
    for r in ev.itertuples(index=False):
        pre = [(r.permno, cal[r.i0 - k]) for k in range(1, window + 1)]
        have = [k for k in pre if k in ndi.index]
        if len(have) < math.ceil(0.6 * window):
            continue
        f = ndi.loc[have]
        scale = window / len(have)
        def car(a, b):
            v = [ar.get((r.permno, cal[r.i0 + k]), np.nan) for k in range(a, b + 1)]
            return np.nansum(v) * 1e4 if np.isfinite(v).sum() >= max(1, (b - a + 1) // 2) else np.nan
        rows.append(dict(permno=r.permno, d0=r.d0, x_info_pre=f.x_info.sum() * scale,
                         x_other_pre=f.x_other.sum() * scale, x_all_pre=f.x_all.sum() * scale,
                         turn=f.turn20.mean(), log_cap=np.log(capm.get((r.permno, cal[r.i0 - 1]), np.nan)),
                         car_pre=car(-window, -1), car01=car(0, 1), car_drift=car(2, 21)))
    return pd.DataFrame(rows)


def reg(df, y, xs):
    return X.fe_regression(df.assign(q_info=df.x_info_pre, q_other=df.x_other_pre), y,
                           ["q_info", "q_other"] + xs, fe="d0", cluster="permno")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, choices=list(SAMPLES))
    ap.add_argument("--family", default="T"); ap.add_argument("--window", type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ev = build(a.sample, a.family, a.window)
    ctrl = ["car_pre", "log_cap", "turn"]
    res = dict(sample=a.sample, family=a.family, window=a.window, events=int(len(ev)), names=int(ev.permno.nunique()),
               primary=reg(ev, "car01", ctrl), no_pre_return_control=reg(ev, "car01", ["log_cap", "turn"]),
               drift=reg(ev, "car_drift", ctrl))
    allf = X.fe_regression(ev.assign(q_info=ev.x_all_pre, q_other=0.0 * ev.x_all_pre + np.random.default_rng(0).normal(0, 1e-9, len(ev))),
                           "car01", ["q_info", "q_other"] + ctrl, fe="d0", cluster="permno")
    res["all_bursts"] = {"b": allf["q_info"]["b"], "t": allf["q_info"]["t"]} if allf else None
    tag = "%s_%s_w%d" % (a.sample, a.family, a.window)
    ev.to_csv(OUT / ("events_%s.csv.gz" % tag), index=False)
    (OUT / ("result_%s.json" % tag)).write_text(json.dumps(res, indent=1, default=float))
    p = res["primary"]
    print(tag, "events", res["events"], "names", res["names"])
    print("  primary  b_info %+.2f (t %.2f)  b_other %+.2f (t %.2f)  info-other t %.2f  car_pre t %.2f"
          % (p["q_info"]["b"], p["q_info"]["t"], p["q_other"]["b"], p["q_other"]["t"], p["info_minus_other"]["t"], p["car_pre"]["t"]))


if __name__ == "__main__":
    main()
