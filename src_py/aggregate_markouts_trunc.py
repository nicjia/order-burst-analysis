#!/usr/bin/env python3
"""Aggregate post-hoc F2: program-minus-bottom markouts within burst truncation-share strata."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

import aggregate_stage2 as S2

BOOT = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260914)
    names = []
    for p in sorted(glob.glob(args.stats)):
        with np.load(p) as z:
            if len(z["dates"]):
                names.append(dict(dates=z["dates"], m=z["markouts"], hz=z["horizons"], edges=z["trunc_edges"]))
    hz = list(names[0]["hz"]); nT = len(names[0]["edges"]) - 1
    res = dict(names=len(names), truncation_edges=names[0]["edges"].tolist())

    def contrast(m, hi, tbins):
        # m: [3, group, tbin, decile, horizon]; mean over (tbin, decile) cells with both groups present
        s, c = m[0, :, :, :, hi], m[2, :, :, :, hi]
        vals = []
        for tb in tbins:
            ok = (c[2, tb] > 0) & (c[0, tb] > 0)
            if ok.any():
                vals.append(np.mean(s[2, tb][ok] / c[2, tb][ok] - s[0, tb][ok] / c[0, tb][ok]))
        return np.mean(vals) if vals else np.nan

    for label, tbins in [("all_strata", list(range(nT)))] + [("stratum_%d" % k, [k]) for k in range(nT)]:
        out = {}
        for hi, h in enumerate(hz):
            per_date, per_name = {}, []
            for n in names:
                v = np.array([contrast(n["m"][i], hi, tbins) for i in range(len(n["dates"]))])
                for d, x in zip(n["dates"], v):
                    if np.isfinite(x):
                        per_date.setdefault(str(d), []).append(x)
                v = v[np.isfinite(v)]
                if len(v):
                    per_name.append(v.mean())
            daily = [np.mean(per_date[d]) for d in sorted(per_date)]
            mean, t = S2.nw_t(daily, 2)
            pn = np.array(per_name)
            boots = [pn[rng.integers(0, len(pn), len(pn))].mean() for _ in range(BOOT)] if len(pn) else []
            out["%ds" % h] = dict(mean_bps=mean, nw_t=t, names=int(len(pn)), name_boot_ci95=S2.ci(boots))
        res[label] = out
    # share of program and bottom packets in each truncation stratum (60 s counts)
    cnt = np.sum([n["m"][:, 2, :, :, :, 2].sum((0, 3)) for n in names], axis=0)      # [group, tbin]
    res["packet_share_by_stratum"] = {g: (cnt[gi] / cnt[gi].sum()).round(4).tolist() for gi, g in enumerate(("bottom", "middle", "program"))}
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: v.get("60s") if isinstance(v, dict) and "60s" in v else v for k, v in res.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
