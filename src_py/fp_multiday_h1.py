#!/usr/bin/env python3
"""fingerprint-multiday-v1 H1 (FINGERPRINT_MULTIDAY_DESIGN.md, frozen 49a39405cc61): cross-day same-side excess.

Usage: fp_multiday_h1.py CELL [--sizes primary|secondary]
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "fp_multiday_v1" / "tables"
import os
LAGS = [int(x) for x in os.environ.get("FP_LAGS", "1,2,3,5,10,20,30,40,50,60").split(",")]
FAR = [int(x) for x in os.environ.get("FP_FAR", "30,40,50,60").split(",")]
MIN_PAIRS = 100


def calendar():
    dates = set()
    for p in sorted((ROOT / "data" / "p4").glob("crsp_*.csv.gz")):
        dates |= set(pd.read_csv(p, usecols=["date"], dtype={"date": str}).date.unique())
    return {d: i for i, d in enumerate(sorted(dates))}


def name_stats(g, days):
    """g: rows (day, side, size, nb) for one name; days: sorted array of present calendar indices."""
    tot = g.groupby(["day", "side"]).nb.sum().unstack(fill_value=0.0).reindex(columns=[-1, 1], fill_value=0.0)
    tot = tot.reindex(days, fill_value=0.0)
    present = pd.Series(1, index=days)
    out = {}
    key = g.set_index(["day", "side", "size"]).nb
    for k in LAGS:
        a = g[["day", "side", "size", "nb"]].copy(); a["day"] = a.day + k
        same = a.merge(g, on=["day", "side", "size"], suffixes=("_t", "_tk"))
        opp_a = a.copy(); opp_a["side"] = -opp_a.side
        opp = opp_a.merge(g, on=["day", "side", "size"], suffixes=("_t", "_tk"))
        P_same = float((same.nb_t * same.nb_tk).sum()); P_opp = float((opp.nb_t * opp.nb_tk).sum())
        t0 = tot.reindex(days); tk = tot.reindex(days + k)
        valid = present.reindex(days + k).notna().to_numpy()
        B0, S0 = t0[1].to_numpy()[valid], t0[-1].to_numpy()[valid]
        Bk, Sk = tk[1].to_numpy()[valid], tk[-1].to_numpy()[valid]
        N_same = float((B0 * Bk + S0 * Sk).sum()); N_opp = float((B0 * Sk + S0 * Bk).sum())
        out[k] = dict(pairs=int(valid.sum()), P_same=P_same, N_same=N_same, P_opp=P_opp, N_opp=N_opp,
                      r_same=P_same / N_same if N_same > 0 else np.nan, r_opp=P_opp / N_opp if N_opp > 0 else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cell"); ap.add_argument("--sizes", default="primary")
    a = ap.parse_args()
    cal = calendar()
    fp = pd.read_csv(T / ("%s_fp.csv.gz" % a.cell), dtype={"date": str})
    nd = pd.read_csv(T / ("%s_nd.csv.gz" % a.cell), dtype={"date": str})
    if a.sizes == "primary":
        fp = fp[fp["size"] % 100 != 0]
    else:
        fp = fp[(fp["size"] >= 10) & (fp["size"] % 10 != 0)]
    fp["day"] = fp.date.map(cal); nd["day"] = nd.date.map(cal)
    rows, prof = [], []
    for permno, g in fp.groupby("permno"):
        days = np.array(sorted(nd[nd.permno == permno].day.dropna().astype(int).unique()))
        if len(days) < 80:
            continue
        s = name_stats(g, days)
        if min(s[k]["pairs"] for k in [1] + FAR) < MIN_PAIRS:
            continue
        E = {k: s[k]["r_same"] - s[k]["r_opp"] for k in LAGS}
        if not all(np.isfinite(E[k]) for k in [1] + FAR):
            continue
        D = E[1] - np.mean([E[k] for k in FAR])
        rows.append(dict(permno=permno, D=D, **{"E%d" % k: E[k] for k in LAGS},
                         **{"rs%d" % k: s[k]["r_same"] for k in LAGS}, **{"ro%d" % k: s[k]["r_opp"] for k in LAGS}))
    r = pd.DataFrame(rows)
    rng = np.random.default_rng(20260921)
    boot = [r.D.to_numpy()[rng.integers(0, len(r), len(r))].mean() for _ in range(2000)]
    res = dict(cell=a.cell, sizes=a.sizes, names=int(len(r)), D_mean=float(r.D.mean()),
               D_t=float(r.D.mean() / (r.D.std(ddof=1) / np.sqrt(len(r)))), D_ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
               share_names_D_pos=float((r.D > 0).mean()),
               E_profile={k: dict(mean=float(r["E%d" % k].mean()), t=float(r["E%d" % k].mean() / (r["E%d" % k].std(ddof=1) / np.sqrt(len(r))))) for k in LAGS},
               r_same_profile={k: float(r["rs%d" % k].mean()) for k in LAGS}, r_opp_profile={k: float(r["ro%d" % k].mean()) for k in LAGS})
    out = ROOT / "results" / "fp_multiday_v1" / ("H1_%s_%s.json" % (a.cell, a.sizes))
    out.write_text(json.dumps(res, indent=1)); r.to_csv(out.with_suffix(".names.csv"), index=False)
    print("%s %s names %d  D %+.5f (t %.2f) CI [%+.5f, %+.5f]  share D>0 %.2f" % (a.cell, a.sizes, res["names"], res["D_mean"], res["D_t"], *res["D_ci95"], res["share_names_D_pos"]))
    print("  lag:      " + " ".join("%8d" % k for k in LAGS))
    print("  r_same:   " + " ".join("%8.5f" % res["r_same_profile"][k] for k in LAGS))
    print("  r_opp:    " + " ".join("%8.5f" % res["r_opp_profile"][k] for k in LAGS))
    print("  E t-stat: " + " ".join("%8.2f" % res["E_profile"][k]["t"] for k in LAGS))


if __name__ == "__main__":
    main()
