#!/usr/bin/env python3
"""program-evidence-v1 modules D and E: daily program and other signed flow for one name.

For each cached day (fingerprint-v1 packet format): run/60 bursts are scored with the stage-3b model.
Program bursts have score >= the 2024 training 80th percentile; bottom bursts <= the 20th. Volumes
are signed packet volumes (hidden-only packets inside the quote carry sign 0 and are excluded).
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import evidence_campaigns as EC
import program_bursts as PB


def day_row(day, model):
    t = day["time"].astype(float); sign = day["sign"].astype(int); vol = day["volume"].astype(float)
    member, table = PB.run_bursts(day)
    grp = np.full(len(t), 3)                                   # 0 bottom, 1 middle, 2 program, 3 no burst / unscored
    n_prog = 0
    if table:
        s = PB.score(table, model)
        g = np.where(~np.isfinite(s), 3, np.where(s >= model["program_threshold_q80"], 2,
                                                   np.where(s <= model["program_threshold_q20"], 0, 1)))
        inb = member >= 0
        grp[inb] = g[member[inb]]
        n_prog = int((g == 2).sum())
    row = dict(n_packets=len(t), n_program_bursts=n_prog, n_bursts=len(table["start"]) if table else 0)
    for gi, lab in enumerate(("bottom", "middle", "program", "other")):
        for side, sl in ((1, "buy"), (-1, "sell")):
            m = (grp == gi) & (sign == side)
            row["%s_%s_vol" % (lab, sl)] = float(vol[m].sum())
    row["buy_vol"] = float(vol[sign > 0].sum()); row["sell_vol"] = float(vol[sign < 0].sum())
    unt = day["untruncated"].astype(bool)
    row["untruncated_buy_vol"] = float(vol[unt & (sign > 0)].sum()); row["untruncated_sell_vol"] = float(vol[unt & (sign < 0)].sum())
    mid = day["mid"].astype(float)
    fin = np.flatnonzero(np.isfinite(mid))
    row["mid_first"] = float(mid[fin[0]]) if len(fin) else np.nan
    row["mid_last"] = float(mid[fin[-1]]) if len(fin) else np.nan
    row["spread_bps_mean"] = float(np.nanmean(day["spread_bps"])) if len(t) else np.nan
    row["sigma_day_bps"] = EC.daily_vol_bps(t, mid) if len(t) else np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True, help="directory of <date>.npz for one name")
    ap.add_argument("--job", required=True, help="job file with 'date ticker' lines")
    ap.add_argument("--permno", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = PB.load_model(args.model)
    tickers = dict(line.split() for line in Path(args.job).read_text().splitlines() if line.strip())
    rows = []
    for path in sorted(glob.glob(str(Path(args.packets) / "*.npz"))):
        date = Path(path).stem
        with np.load(path) as z:
            if "time" not in z.files or len(z["time"]) == 0:
                continue
            day = {k: z[k] for k in z.files}
        r = day_row(day, model)
        r.update(permno=int(args.permno), date=date, ticker=tickers.get(date, ""))
        rows.append(r)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(len(rows))


if __name__ == "__main__":
    main()
