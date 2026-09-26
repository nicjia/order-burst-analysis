#!/usr/bin/env python3
"""Aggregate metaorder-v1 M5 (queue-aware passive orders) for one group.

Q1: per-fill 60 s provider markout, program-trigger fills minus bottom-trigger fills; the difference is
formed per name-day, averaged per date (equal weight per name-day), Newey-West over dates (2 lags), with a
name bootstrap. Q2 (descriptive): expected P&L per posting (fill rate x mean markout, unfilled = 0) for
program, middle, bottom and placebo postings, with and without a 0.20 cent/share maker rebate.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate_stage2 as S2

ROOT = Path(__file__).resolve().parents[1]
BOOT = 1000
HZ = (1, 10, 60, 300)
LABELS = ("program", "middle", "bottom", "placebo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260914)
    files = sorted(glob.glob(str(ROOT / "results" / "metaorder_v1" / args.group / "passive" / "*" / "*.csv.gz")))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d.posted]
    d["rebate_bps"] = 0.002 / (d.price / 10000.0) * 1e4
    out = dict(group=args.group, name_days=int(d.groupby(["ticker", "date"]).ngroups), names=int(d.ticker.nunique()),
               postings=int(len(d)))
    summ = {}
    for lab in LABELS:
        g = d[d.label == lab]
        f = g[g.filled]
        row = dict(postings=int(len(g)), fill_rate=float(g.filled.mean()) if len(g) else None,
                   median_queue_ahead=float(g.ahead0.median()) if len(g) else None,
                   exit_reasons={k: int(v) for k, v in g.exit_reason.value_counts().items()})
        for h in HZ:
            m = f["markout_%ds_bps" % h]
            row["markout_%ds_per_fill" % h] = float(m.mean()) if len(m) else None
            pnl = np.where(g.filled, g["markout_%ds_bps" % h].fillna(0), 0.0)
            row["pnl_%ds_per_posting" % h] = float(np.mean(pnl)) if len(g) else None
            pnl_r = np.where(g.filled, g["markout_%ds_bps" % h].fillna(0) + g.rebate_bps, 0.0)
            row["pnl_%ds_per_posting_with_rebate" % h] = float(np.mean(pnl_r)) if len(g) else None
        summ[lab] = row
    out["by_label"] = summ

    def contrast(a, b, h, per_posting=False):
        col = "markout_%ds_bps" % h
        rows = []
        for (tk, dt), g in d.groupby(["ticker", "date"]):
            ga, gb = g[g.label == a], g[g.label == b]
            if per_posting:
                va = np.where(ga.filled, ga[col].fillna(0), 0).mean() if len(ga) else np.nan
                vb = np.where(gb.filled, gb[col].fillna(0), 0).mean() if len(gb) else np.nan
            else:
                fa, fb = ga[ga.filled][col].dropna(), gb[gb.filled][col].dropna()
                va = fa.mean() if len(fa) >= 3 else np.nan
                vb = fb.mean() if len(fb) >= 3 else np.nan
            rows.append(dict(ticker=tk, date=dt, diff=va - vb))
        c = pd.DataFrame(rows).dropna()
        daily = c.groupby("date")["diff"].mean().sort_index()
        mean, t = S2.nw_t(daily.to_numpy(), 2)
        per_name = c.groupby("ticker")["diff"].mean().to_numpy()
        boots = [per_name[rng.integers(0, len(per_name), len(per_name))].mean() for _ in range(BOOT)]
        return dict(mean_bps=mean, nw_t=t, dates=int(len(daily)), name_days=int(len(c)), name_mean_bps=float(per_name.mean()),
                    name_boot_ci95=S2.ci(boots))
    out["Q1_per_fill_program_minus_bottom"] = {"%ds" % h: contrast("program", "bottom", h) for h in HZ}
    out["per_fill_program_minus_placebo"] = {"%ds" % h: contrast("program", "placebo", h) for h in HZ}
    out["Q2_per_posting_program_minus_placebo"] = {"%ds" % h: contrast("program", "placebo", h, per_posting=True) for h in HZ}
    out["per_posting_program_minus_bottom"] = {"%ds" % h: contrast("program", "bottom", h, per_posting=True) for h in HZ}
    q = out["Q1_per_fill_program_minus_bottom"]["60s"]
    out["Q1_exploration_abs_t_gt_3"] = bool(q["nw_t"] is not None and np.isfinite(q["nw_t"]) and abs(q["nw_t"]) > 3)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "by_label"}, indent=1, default=float)[:3000])
    for lab, row in summ.items():
        print(lab, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "exit_reasons"})


if __name__ == "__main__":
    main()
